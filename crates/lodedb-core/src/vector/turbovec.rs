//! Native TurboVec adapter for chunk-vector search.

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Instant;

use serde_json::Value;
use turbovec::IdMapIndex;

use crate::error::{CoreError, CoreErrorCode};
use crate::storage::tvim_delta::{read_delta_segment, TVIM_DELTA_DIR_SUFFIX};
use crate::vector::index::{
    CoreVectorChunk, VectorBackendMetadata, VectorIndexWriteMetrics, VectorSearchHit,
};
use crate::vector::stable_id::stable_uint64_ids_for_chunk_ids;

/// Smallest query batch worth the GPU's per-call host/device overhead; smaller
/// batches stay on the CPU kernel (single queries always do).
const GPU_MIN_BATCH: usize = 4;
/// Smallest corpus worth a GPU-resident scan; below this the CPU kernel wins.
const GPU_MIN_CORPUS: usize = 256;

/// Lazily built GPU-resident scan state for one index instance.
///
/// `unavailable` latches when no CUDA driver loads (permanent for the host, never
/// retried). `build_failed` latches a device/build error for the current rows and
/// is cleared on mutation so a healthy host rebuilds; both keep the hot query path
/// from re-attempting a failing GPU build every batch.
#[derive(Default)]
struct GpuState {
    session: Option<lodedb_gpu::GpuScanSession>,
    unavailable: bool,
    build_failed: bool,
}

/// Returns whether the GPU-resident scan is enabled by policy.
///
/// Shares the Python serving layer's knob: `LODEDB_GPU_DIRECT_TURBOVEC=off`
/// disables the GPU scan everywhere. Any other value (or unset) leaves it enabled,
/// still subject to runtime driver availability and the batch/corpus gates.
fn gpu_scan_enabled() -> bool {
    match std::env::var("LODEDB_GPU_DIRECT_TURBOVEC") {
        Ok(value) => !value.trim().eq_ignore_ascii_case("off"),
        Err(_) => true,
    }
}

/// Native TurboVec serving index plus stable-id lookup metadata.
pub struct TurboVecNativeIndex {
    index: IdMapIndex,
    chunk_ids_by_stable_id: BTreeMap<u64, String>,
    document_ids_by_stable_id: BTreeMap<u64, String>,
    stable_id_by_chunk_id: BTreeMap<String, u64>,
    dim: usize,
    bit_width: usize,
    generation: u64,
    build_seconds: f64,
    gpu: RefCell<GpuState>,
}

impl TurboVecNativeIndex {
    /// Builds an IdMapIndex from core chunk embeddings and stable chunk ids.
    pub fn build(
        chunks: &[CoreVectorChunk],
        native_dim: usize,
        bit_width: usize,
        generation: u64,
    ) -> Result<Self, CoreError> {
        validate_config(native_dim, bit_width)?;
        let started = Instant::now();
        let mut embeddings = Vec::with_capacity(chunks.len() * native_dim);
        for chunk in chunks {
            if chunk.embedding.len() != native_dim {
                return invalid("chunk embeddings do not match native_dim");
            }
            embeddings.extend_from_slice(&chunk.embedding);
        }
        let chunk_ids = chunks
            .iter()
            .map(|chunk| chunk.chunk_id.clone())
            .collect::<Vec<_>>();
        let stable_ids = stable_uint64_ids_for_chunk_ids(&chunk_ids);
        let mut index = IdMapIndex::new(native_dim, bit_width).map_err(core_error)?;
        if !embeddings.is_empty() {
            index
                .add_with_ids(&embeddings, &stable_ids)
                .map_err(core_error)?;
            index.prepare();
        }
        Ok(Self::from_parts(
            index,
            chunks,
            &stable_ids,
            native_dim,
            bit_width,
            generation,
            started.elapsed().as_secs_f64(),
        ))
    }

    /// Loads a `.tvim` index and attaches chunk/document metadata.
    pub fn load(
        path: impl AsRef<Path>,
        chunks: &[CoreVectorChunk],
        generation: u64,
    ) -> Result<Self, CoreError> {
        Self::load_with_manifest(path, None, chunks, generation)
    }

    /// Loads a `.tvim` index, replays committed `.tvd` deltas, and attaches metadata.
    pub fn load_with_manifest(
        path: impl AsRef<Path>,
        manifest: Option<&Value>,
        chunks: &[CoreVectorChunk],
        generation: u64,
    ) -> Result<Self, CoreError> {
        let started = Instant::now();
        let index = load_id_map_with_manifest(path.as_ref(), manifest)?;
        let native_dim = index.dim();
        let bit_width = index.bit_width();
        validate_config(native_dim, bit_width)?;
        let chunk_ids = chunks
            .iter()
            .map(|chunk| chunk.chunk_id.clone())
            .collect::<Vec<_>>();
        let stable_ids = stable_uint64_ids_for_chunk_ids(&chunk_ids);
        index.prepare();
        Ok(Self::from_parts(
            index,
            chunks,
            &stable_ids,
            native_dim,
            bit_width,
            generation,
            started.elapsed().as_secs_f64(),
        ))
    }

    /// Searches one query vector.
    ///
    /// Passes the borrowed query slice straight to the kernel instead of routing
    /// through `search_batch(&[query.to_vec()])`, which copied the query into an
    /// owned `Vec` and then again into the batch's flat query buffer. A single
    /// query already is the flat buffer, so neither copy is needed.
    pub fn search(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        allowlist_chunk_ids: &[String],
    ) -> Result<Vec<VectorSearchHit>, CoreError> {
        if top_k == 0 {
            return invalid("top_k must be positive");
        }
        if query_embedding.len() != self.dim {
            return invalid("query dimension does not match TurboVec index");
        }
        let allowlist = self.stable_ids_for_chunks(allowlist_chunk_ids);
        if !allowlist_chunk_ids.is_empty() && allowlist.is_empty() {
            return Ok(Vec::new());
        }
        let mut effective_top_k = top_k.min(self.index.len());
        if !allowlist.is_empty() {
            effective_top_k = effective_top_k.min(allowlist.len());
        }
        if effective_top_k == 0 {
            return Ok(Vec::new());
        }

        // Eligible unfiltered single queries also serve from the GPU-resident scan
        // (a batch of one); the cached device buffers keep the per-call overhead
        // small. Filtered/ineligible queries and any GPU error fall to the CPU
        // kernel below. No batch-size gate here: a single query is the batch.
        if allowlist.is_empty()
            && self.index.len() >= GPU_MIN_CORPUS
            && gpu_scan_enabled()
            && lodedb_gpu::cuda_runtime_available()
        {
            if let Some((scores, stable_ids)) = self.gpu_search(query_embedding, 1, effective_top_k)
            {
                return Ok(self
                    .assemble_rows(&scores, &stable_ids, 1, effective_top_k)?
                    .into_iter()
                    .next()
                    .unwrap_or_default());
            }
        }

        let (scores, stable_ids) = if allowlist.is_empty() {
            self.index.search(query_embedding, effective_top_k)
        } else {
            self.index
                .search_with_allowlist(query_embedding, effective_top_k, Some(&allowlist))
        };
        Ok(self
            .assemble_rows(&scores, &stable_ids, 1, effective_top_k)?
            .into_iter()
            .next()
            .unwrap_or_default())
    }

    /// Searches one query restricted to an allowlist of stable ids.
    ///
    /// The ANN candidate path already holds TurboVec stable ids (its postings are
    /// stable ids, resolved once at build), so it skips the chunk-id resolution
    /// `search` pays. Absent ids are filtered out (the underlying kernel panics on
    /// an unknown id), preserving the same fail-soft contract as the chunk-id path
    /// even though the cache-invalidation invariant already keeps candidates live.
    pub fn search_with_stable_allowlist(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        allowlist: &[u64],
    ) -> Result<Vec<VectorSearchHit>, CoreError> {
        if top_k == 0 {
            return invalid("top_k must be positive");
        }
        if query_embedding.len() != self.dim {
            return invalid("query dimension does not match TurboVec index");
        }
        let present: Vec<u64> = allowlist
            .iter()
            .copied()
            .filter(|id| self.index.contains(*id))
            .collect();
        if present.is_empty() {
            return Ok(Vec::new());
        }
        let effective_top_k = top_k.min(self.index.len()).min(present.len());
        if effective_top_k == 0 {
            return Ok(Vec::new());
        }
        let (scores, stable_ids) =
            self.index
                .search_with_allowlist(query_embedding, effective_top_k, Some(&present));
        Ok(self
            .assemble_rows(&scores, &stable_ids, 1, effective_top_k)?
            .into_iter()
            .next()
            .unwrap_or_default())
    }

    /// Searches a query batch with one shared chunk allowlist.
    pub fn search_batch(
        &self,
        query_embeddings: &[Vec<f32>],
        top_k: usize,
        allowlist_chunk_ids: &[String],
    ) -> Result<Vec<Vec<VectorSearchHit>>, CoreError> {
        if top_k == 0 {
            return invalid("top_k must be positive");
        }
        if query_embeddings.is_empty() {
            return Ok(Vec::new());
        }
        let mut queries = Vec::with_capacity(query_embeddings.len() * self.dim);
        for query in query_embeddings {
            if query.len() != self.dim {
                return invalid("query dimension does not match TurboVec index");
            }
            queries.extend_from_slice(query);
        }
        let nq = query_embeddings.len();
        let (scores, stable_ids, k) =
            self.scan_batch_flat(&queries, nq, top_k, allowlist_chunk_ids)?;
        if k == 0 {
            return Ok(vec![Vec::new(); nq]);
        }
        self.assemble_rows(&scores, &stable_ids, nq, k)
    }

    /// Flat-input, arrays-output batch search for the near-zero-copy boundary.
    ///
    /// `queries` is the flat `[nq * dim]` query buffer (no per-query `Vec`), and the
    /// result is flat `[nq * k]` `(scores, document_ids, k)` — no per-hit struct or
    /// JSON. `k` is 0 when nothing matches. The PyO3 layer turns scores into a
    /// numpy array and document ids into a string list directly, so a batch query
    /// crosses the boundary without the per-hit serialize/parse `search_batch` pays.
    pub fn search_batch_arrays(
        &self,
        queries: &[f32],
        nq: usize,
        top_k: usize,
        allowlist_chunk_ids: &[String],
    ) -> Result<(Vec<f32>, Vec<String>, usize), CoreError> {
        if top_k == 0 {
            return invalid("top_k must be positive");
        }
        if nq == 0 {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        if queries.len() != nq * self.dim {
            return invalid("query batch length does not match nq*dim");
        }
        let (scores, stable_ids, k) = self.scan_batch_flat(queries, nq, top_k, allowlist_chunk_ids)?;
        if k == 0 {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        let mut document_ids = Vec::with_capacity(stable_ids.len());
        for stable_id in &stable_ids {
            let document_id = self
                .document_ids_by_stable_id
                .get(stable_id)
                .ok_or_else(|| invalid_err("TurboVec returned an unknown stable id"))?;
            document_ids.push(document_id.clone());
        }
        Ok((scores, document_ids, k))
    }

    /// Shared batch scan: resolves the allowlist, runs the GPU-resident scan when
    /// eligible (else the CPU kernel), and returns flat `[nq * k]` `(scores,
    /// stable_ids, k)`. `k` is 0 when the (possibly filtered) result set is empty.
    fn scan_batch_flat(
        &self,
        queries: &[f32],
        nq: usize,
        top_k: usize,
        allowlist_chunk_ids: &[String],
    ) -> Result<(Vec<f32>, Vec<u64>, usize), CoreError> {
        let allowlist = self.stable_ids_for_chunks(allowlist_chunk_ids);
        if !allowlist_chunk_ids.is_empty() && allowlist.is_empty() {
            return Ok((Vec::new(), Vec::new(), 0));
        }
        let mut effective_top_k = top_k.min(self.index.len());
        if !allowlist.is_empty() {
            effective_top_k = effective_top_k.min(allowlist.len());
        }
        if effective_top_k == 0 {
            return Ok((Vec::new(), Vec::new(), 0));
        }

        // Eligible unfiltered batches on a CUDA host serve from the GPU-resident
        // scan; filtered/small batches and any GPU error fall to the CPU kernel.
        if allowlist.is_empty()
            && nq >= GPU_MIN_BATCH
            && self.index.len() >= GPU_MIN_CORPUS
            && gpu_scan_enabled()
            && lodedb_gpu::cuda_runtime_available()
        {
            if let Some((scores, stable_ids)) = self.gpu_search(queries, nq, effective_top_k) {
                return Ok((scores, stable_ids, effective_top_k));
            }
        }

        let (scores, stable_ids) = if allowlist.is_empty() {
            self.index.search(queries, effective_top_k)
        } else {
            self.index
                .search_with_allowlist(queries, effective_top_k, Some(&allowlist))
        };
        Ok((scores, stable_ids, effective_top_k))
    }

    /// Attempts the GPU-resident scan, building the session lazily.
    ///
    /// Returns `None` (caller uses the CPU kernel) when no GPU is available, when a
    /// build or device error occurs, or while a prior failure is latched. A build
    /// or search device error latches `build_failed` so the hot path does not
    /// re-attempt a failing GPU every batch; a missing driver latches `unavailable`
    /// permanently. Both are cleared (except `unavailable`) on the next mutation.
    fn gpu_search(&self, queries: &[f32], nq: usize, k: usize) -> Option<(Vec<f32>, Vec<u64>)> {
        let mut state = self.gpu.borrow_mut();
        if state.unavailable || state.build_failed {
            return None;
        }
        if state.session.is_none() {
            match self.build_gpu_session() {
                Ok(session) => state.session = Some(session),
                Err(lodedb_gpu::GpuScanError::Unavailable(_)) => {
                    state.unavailable = true;
                    return None;
                }
                Err(_) => {
                    state.build_failed = true;
                    return None;
                }
            }
        }
        let session = state.session.as_ref()?;
        match session.search(queries, nq, k) {
            Ok(result) => Some(result),
            Err(_) => {
                state.session = None;
                state.build_failed = true;
                None
            }
        }
    }

    /// Reconstructs the rotated-calibrated rows and uploads them to a GPU session.
    fn build_gpu_session(&self) -> Result<lodedb_gpu::GpuScanSession, lodedb_gpu::GpuScanError> {
        let rotation = self
            .index
            .rotation_matrix()
            .ok_or_else(|| lodedb_gpu::GpuScanError::Invalid("rotation unavailable".into()))?;
        let (stable_ids, rows) = self.index.reconstruct_all();
        lodedb_gpu::GpuScanSession::build(&rows, &stable_ids, &rotation, self.dim)
    }

    /// Drops any resident GPU session so the next eligible batch rebuilds it.
    ///
    /// Clears a latched build failure (a healthy host may now succeed) but leaves
    /// `unavailable` set, since a host with no driver stays without one.
    fn invalidate_gpu_session(&self) {
        let mut state = self.gpu.borrow_mut();
        state.session = None;
        state.build_failed = false;
    }

    /// Patches the resident GPU session in place for a set of in-place row updates.
    ///
    /// Called only when an upsert overwrote existing rows without appending, so each
    /// changed id keeps its slot. Reconstructs just those rows (`O(changed)`) and
    /// patches them into the resident buffer, avoiding a full rebuild. Any obstacle
    /// (no live session, missing rotation, a reconstruct or device error) drops the
    /// session so the next eligible query rebuilds it. The immutable index reads run
    /// before the `gpu` borrow so the two never overlap.
    fn patch_gpu_session(&self, updated_ids: &[u64]) {
        if updated_ids.is_empty() {
            return;
        }
        // Nothing resident yet means the next query builds fresh; do not pay for a
        // reconstruct we would only throw away.
        if self.gpu.borrow().session.is_none() {
            return;
        }
        let Some(rotation) = self.index.rotation_matrix() else {
            self.invalidate_gpu_session();
            return;
        };
        let rows = match self.index.reconstruct_rows(updated_ids) {
            Ok(rows) => rows,
            Err(_) => {
                self.invalidate_gpu_session();
                return;
            }
        };
        let mut state = self.gpu.borrow_mut();
        let Some(session) = state.session.as_mut() else {
            return;
        };
        if session.patch(updated_ids, &rows, &rotation).is_err() {
            state.session = None;
            state.build_failed = false;
        }
    }

    /// Returns the chunk id for a live stable id, or `None` when it is absent.
    /// Used to map ANN cluster postings (stable ids) back to chunk-id strings for
    /// the `.tvann` sidecar at persist time.
    pub fn chunk_id_for_stable_id(&self, stable_id: u64) -> Option<&str> {
        self.chunk_ids_by_stable_id
            .get(&stable_id)
            .map(String::as_str)
    }

    /// Returns active stable ids for chunk ids, filtering absent chunks.
    pub fn stable_ids_for_chunks(&self, chunk_ids: &[String]) -> Vec<u64> {
        let mut seen = BTreeSet::new();
        let mut stable_ids = Vec::new();
        for chunk_id in chunk_ids {
            if let Some(stable_id) = self.stable_id_by_chunk_id.get(chunk_id) {
                if seen.insert(*stable_id) {
                    stable_ids.push(*stable_id);
                }
            }
        }
        stable_ids
    }

    /// Upserts chunks into the live TurboVec index without rebuilding existing rows.
    pub fn upsert_chunks(&mut self, chunks: &[CoreVectorChunk]) -> Result<(), CoreError> {
        if chunks.is_empty() {
            return Ok(());
        }
        let mut embeddings = Vec::with_capacity(chunks.len() * self.dim);
        for chunk in chunks {
            if chunk.embedding.len() != self.dim {
                return invalid("chunk embeddings do not match native_dim");
            }
            embeddings.extend_from_slice(&chunk.embedding);
        }
        let chunk_ids = chunks
            .iter()
            .map(|chunk| chunk.chunk_id.clone())
            .collect::<Vec<_>>();
        let stable_ids = stable_uint64_ids_for_chunk_ids(&chunk_ids);
        let (_replaced, appended) = self
            .index
            .upsert_with_ids_2d(&embeddings, self.dim, &stable_ids)
            .map_err(core_error)?;
        for (stable_id, chunk) in stable_ids.iter().zip(chunks) {
            self.chunk_ids_by_stable_id
                .insert(*stable_id, chunk.chunk_id.clone());
            self.document_ids_by_stable_id
                .insert(*stable_id, chunk.document_id.clone());
            self.stable_id_by_chunk_id
                .insert(chunk.chunk_id.clone(), *stable_id);
        }
        if appended == 0 {
            // Pure in-place update: existing rows keep their slots, so patch only
            // the changed rows on the GPU instead of rebuilding the whole session.
            self.patch_gpu_session(&stable_ids);
        } else {
            // New rows extend the corpus (and grow the resident buffer), so drop the
            // session and let the next eligible query rebuild it.
            self.invalidate_gpu_session();
        }
        Ok(())
    }

    /// Removes chunks from the live TurboVec index if they are present.
    pub fn remove_chunks(&mut self, chunk_ids: &[String]) -> usize {
        let stable_ids = self.stable_ids_for_chunks(chunk_ids);
        let removed = self.index.remove_many(&stable_ids);
        for stable_id in stable_ids {
            if let Some(chunk_id) = self.chunk_ids_by_stable_id.remove(&stable_id) {
                self.stable_id_by_chunk_id.remove(&chunk_id);
            }
            self.document_ids_by_stable_id.remove(&stable_id);
        }
        self.invalidate_gpu_session();
        removed
    }

    /// Persists the `.tvim` payload and returns metrics-only write details.
    pub fn write(&self, path: impl AsRef<Path>) -> Result<VectorIndexWriteMetrics, CoreError> {
        let started = Instant::now();
        self.index.write(path.as_ref()).map_err(core_error)?;
        Ok(VectorIndexWriteMetrics {
            compact_backend: "turbovec_idmap".to_string(),
            snapshot_bytes: path.as_ref().metadata().map_err(core_error)?.len(),
            persist_ms: started.elapsed().as_secs_f64() * 1000.0,
            raw_payload_text_present: false,
        })
    }

    /// Returns safe backend metadata.
    pub fn backend_metadata(&self) -> VectorBackendMetadata {
        VectorBackendMetadata {
            compact_backend: "turbovec_idmap".to_string(),
            native_backend: "turbovec".to_string(),
            native_used: true,
            dim: self.dim,
            bit_width: self.bit_width,
            generation: self.generation,
            vector_count: self.index.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.len() == 0
    }

    pub fn build_seconds(&self) -> f64 {
        self.build_seconds
    }

    pub fn calibration_fingerprint(&self) -> u64 {
        self.index.calibration_fingerprint()
    }

    /// Returns the quantized codes and scales for `stable_ids`, in order, for
    /// writing a tvim delta. The ids must currently be present in the live index.
    pub fn export_encoded(&self, stable_ids: &[u64]) -> Result<(Vec<u8>, Vec<f32>), CoreError> {
        self.index.export_encoded(stable_ids).map_err(core_error)
    }

    /// Reconstructs every live row in rotated space, paired with its chunk id and
    /// stable id (all three aligned by position).
    ///
    /// These are the exact rows the scan scores against, so an ANN layer that
    /// clusters over them (and rotates the query by [`rotation_matrix`]) keeps
    /// candidate selection in the scan's coordinate space regardless of when a
    /// row was added, avoiding any raw-vs-rotated skew. The stable ids become the
    /// cluster postings directly, so the query path never re-resolves chunk ids.
    ///
    /// [`rotation_matrix`]: Self::rotation_matrix
    pub fn reconstruct_all_chunks(&self) -> (Vec<String>, Vec<u64>, Vec<f32>) {
        let (stable_ids, rows) = self.index.reconstruct_all();
        let mut chunk_ids = Vec::with_capacity(stable_ids.len());
        let mut kept_ids = Vec::with_capacity(stable_ids.len());
        let mut kept_rows = Vec::with_capacity(rows.len());
        for (position, stable_id) in stable_ids.iter().enumerate() {
            if let Some(chunk_id) = self.chunk_ids_by_stable_id.get(stable_id) {
                chunk_ids.push(chunk_id.clone());
                kept_ids.push(*stable_id);
                kept_rows.extend_from_slice(&rows[position * self.dim..(position + 1) * self.dim]);
            }
        }
        (chunk_ids, kept_ids, kept_rows)
    }

    /// The TurboVec rotation matrix (row-major `dim * dim`), or `None` before
    /// calibration fits one. A query must be rotated by this to share the space
    /// of [`reconstruct_all_chunks`](Self::reconstruct_all_chunks).
    pub fn rotation_matrix(&self) -> Option<Vec<f32>> {
        self.index.rotation_matrix()
    }

    fn from_parts(
        index: IdMapIndex,
        chunks: &[CoreVectorChunk],
        stable_ids: &[u64],
        dim: usize,
        bit_width: usize,
        generation: u64,
        build_seconds: f64,
    ) -> Self {
        let mut chunk_ids_by_stable_id = BTreeMap::new();
        let mut document_ids_by_stable_id = BTreeMap::new();
        let mut stable_id_by_chunk_id = BTreeMap::new();
        for (stable_id, chunk) in stable_ids.iter().zip(chunks) {
            chunk_ids_by_stable_id.insert(*stable_id, chunk.chunk_id.clone());
            document_ids_by_stable_id.insert(*stable_id, chunk.document_id.clone());
            stable_id_by_chunk_id.insert(chunk.chunk_id.clone(), *stable_id);
        }
        Self {
            index,
            chunk_ids_by_stable_id,
            document_ids_by_stable_id,
            stable_id_by_chunk_id,
            dim,
            bit_width,
            generation,
            build_seconds,
            gpu: RefCell::new(GpuState::default()),
        }
    }

    fn assemble_rows(
        &self,
        scores: &[f32],
        stable_ids: &[u64],
        query_count: usize,
        row_width: usize,
    ) -> Result<Vec<Vec<VectorSearchHit>>, CoreError> {
        if scores.len() != stable_ids.len() || scores.len() != query_count * row_width {
            return invalid("TurboVec returned malformed result buffers");
        }
        let mut rows = Vec::with_capacity(query_count);
        for query_index in 0..query_count {
            let start = query_index * row_width;
            let end = start + row_width;
            let mut row = Vec::with_capacity(row_width);
            for (&score, &stable_id) in scores[start..end].iter().zip(&stable_ids[start..end]) {
                let chunk_id = self
                    .chunk_ids_by_stable_id
                    .get(&stable_id)
                    .ok_or_else(|| invalid_err("TurboVec returned an unknown stable id"))?;
                let document_id = self
                    .document_ids_by_stable_id
                    .get(&stable_id)
                    .ok_or_else(|| invalid_err("TurboVec returned an unknown stable id"))?;
                row.push(VectorSearchHit {
                    chunk_id: chunk_id.clone(),
                    document_id: document_id.clone(),
                    stable_id,
                    score,
                });
            }
            rows.push(row);
        }
        Ok(rows)
    }
}

fn replay_deltas(
    base_path: &Path,
    index: &mut IdMapIndex,
    manifest: Option<&Value>,
) -> Result<(), CoreError> {
    let Some(manifest) = manifest else {
        return Ok(());
    };
    if let Some(base_fingerprint) = manifest
        .get("base")
        .and_then(Value::as_object)
        .and_then(|base| base.get("calibration_fingerprint"))
        .and_then(Value::as_u64)
    {
        if base_fingerprint != 0 && base_fingerprint != index.calibration_fingerprint() {
            return Err(CoreError::new(
                CoreErrorCode::CorruptStore,
                "TurboVec delta replay rejected: base calibration fingerprint mismatch",
            ));
        }
    }
    let delta_dir = base_path.with_file_name(format!(
        "{}{}",
        base_path.file_name().unwrap_or_default().to_string_lossy(),
        TVIM_DELTA_DIR_SUFFIX
    ));
    let mut previous_seq = None;
    for delta in manifest
        .get("deltas")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let sequence = delta.get("seq").and_then(Value::as_i64).unwrap_or(-1);
        if previous_seq.is_some_and(|previous| sequence <= previous) {
            return Err(CoreError::new(
                CoreErrorCode::CorruptStore,
                "TurboVec delta manifest has out-of-order segments",
            ));
        }
        previous_seq = Some(sequence);
        let file_name = delta.get("file_name").and_then(Value::as_str).unwrap_or("");
        let segment = read_delta_segment(&delta_dir.join(file_name)).map_err(core_error)?;
        if segment
            .header
            .get("calibration_fingerprint")
            .and_then(Value::as_u64)
            != Some(index.calibration_fingerprint())
        {
            return Err(CoreError::new(
                CoreErrorCode::CorruptStore,
                "TurboVec delta replay rejected: segment calibration fingerprint mismatch",
            ));
        }
        if !segment.removed_stable_ids.is_empty() {
            let removed = index.remove_many(&segment.removed_stable_ids);
            if removed != segment.removed_stable_ids.len() {
                return Err(CoreError::new(
                    CoreErrorCode::CorruptStore,
                    "TurboVec delta replay rejected: removed-id count mismatch",
                ));
            }
        }
        if !segment.upsert_stable_ids.is_empty() {
            index
                .add_encoded(
                    &segment.upsert_stable_ids,
                    &segment.upsert_codes,
                    &segment.upsert_scales,
                )
                .map_err(core_error)?;
        }
        if let Some(rows_after) = segment.header.get("rows_after").and_then(Value::as_u64) {
            if index.len() != rows_after as usize {
                return Err(CoreError::new(
                    CoreErrorCode::CorruptStore,
                    "TurboVec delta replay rejected: row count mismatch",
                ));
            }
        }
    }
    Ok(())
}

pub(crate) fn load_id_map_with_manifest(
    path: &Path,
    manifest: Option<&Value>,
) -> Result<IdMapIndex, CoreError> {
    let mut index = IdMapIndex::load(path).map_err(core_error)?;
    replay_deltas(path, &mut index, manifest)?;
    Ok(index)
}

fn validate_config(native_dim: usize, bit_width: usize) -> Result<(), CoreError> {
    if native_dim == 0 {
        return invalid("native_dim must be positive");
    }
    if !matches!(bit_width, 2 | 4) {
        return invalid("TurboVec bit_width must be 2 or 4");
    }
    Ok(())
}

fn invalid<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(invalid_err(message))
}

fn invalid_err(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::InvalidArgument, message)
}

fn core_error(error: impl std::fmt::Display) -> CoreError {
    CoreError::new(CoreErrorCode::Internal, error.to_string())
}
