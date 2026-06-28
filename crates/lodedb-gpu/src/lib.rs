//! Optional CUDA GPU-resident exact scan for the LodeDB native core.
//!
//! Mirrors the Python `engine/gpu_turbovec.py` path natively: a dequantized copy
//! of the index rows stays resident on the GPU and eligible batched queries are
//! scored with an exact GEMM plus a device top-k. The contract matches the
//! vendored reconstruction parity tests
//! (`third_party/turbovec/turbovec/tests/reconstruction.rs`): rows are exported in
//! ROTATED CALIBRATED space, and
//!
//! ```text
//! score(q, row) = <q @ rotation^T, reconstructed_row>
//! ```
//!
//! reproduces the CPU kernel's calibrated score. Queries are rotated on the host
//! with the deterministic rotation (cheap at batch scale), then a single cuBLAS
//! GEMM produces the batch x corpus score matrix and a custom top-k kernel selects
//! the per-query top-k on device, copying back only the top-k slots and scores.
//! The final ordering is fixed on the host (descending score, ascending stable id
//! on ties), matching the Python GPU path.
//!
//! Resident rows are fp32: at the corpus sizes the native core serves (<= ~1M x
//! 128) the matrix is a few hundred MB at most, and fp32 is both simpler and more
//! faithful than the Python path's fp16 rows. Mutation invalidates the session so
//! the next eligible batch rebuilds; O(changed) patching is a later refinement.
//!
//! This is the single crate in the workspace permitted to use `unsafe`: CUDA
//! kernel launches and the cuBLAS GEMM are inherently unsafe FFI. The core stays
//! `unsafe_code = "forbid"` and consumes only the safe API below. Built with
//! cudarc's `dynamic-loading`, the crate links nothing at build time and engages a
//! GPU only where a driver loads at runtime; every entry point fails closed to an
//! error the caller turns into a CPU fallback.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::compile_ptx;

/// Threads per block for the top-k kernel. Must stay a power of two for the
/// shared-memory reduction.
const TOPK_BLOCK: u32 = 256;

/// Exact per-query top-k over a dense `batch x corpus` score matrix.
///
/// One block scores one query: each pass finds the block-wide argmax of its score
/// row with a shared-memory reduction (ties resolve to the lower slot), records it
/// in descending order, then masks it to the sentinel for the next pass. `k` is
/// bounded by the corpus size by the caller, so every pass finds a real row.
const TOPK_KERNEL: &str = r#"
#define NEG_SENTINEL (-3.4028234e38f)
extern "C" __global__ void topk_argmax(
    float* scores,
    const int n,
    const int k,
    unsigned int* out_idx,
    float* out_val)
{
    const int q = blockIdx.x;
    const int t = threadIdx.x;
    const int nt = blockDim.x;
    float* row = scores + (size_t)q * (size_t)n;
    extern __shared__ char smem[];
    float* sval = (float*)smem;
    int* sidx = (int*)(sval + nt);
    for (int pass = 0; pass < k; ++pass) {
        float best = NEG_SENTINEL;
        int bi = -1;
        for (int i = t; i < n; i += nt) {
            float v = row[i];
            if (v > best) { best = v; bi = i; }
        }
        sval[t] = best;
        sidx[t] = bi;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (t < s) {
                float ov = sval[t + s];
                int oi = sidx[t + s];
                float cv = sval[t];
                int ci = sidx[t];
                if (ov > cv || (ov == cv && oi >= 0 && (ci < 0 || oi < ci))) {
                    sval[t] = ov;
                    sidx[t] = oi;
                }
            }
            __syncthreads();
        }
        if (t == 0) {
            int idx = sidx[0];
            out_idx[(size_t)q * (size_t)k + pass] = (unsigned int)(idx < 0 ? 0 : idx);
            out_val[(size_t)q * (size_t)k + pass] = sval[0];
            if (idx >= 0) row[idx] = NEG_SENTINEL;
        }
        __syncthreads();
    }
}
"#;

/// Why the GPU scan could not run; every variant is a signal to fall back to the
/// CPU kernel, never an error the caller should surface to the user.
#[derive(Debug, Clone)]
pub enum GpuScanError {
    /// No CUDA driver is loadable (CPU-only host) — do not retry this process.
    Unavailable(String),
    /// The driver loaded but a CUDA call failed (no device, OOM, kernel error).
    Device(String),
    /// The caller passed inconsistent shapes; a programming error, surfaced loudly.
    Invalid(String),
}

impl std::fmt::Display for GpuScanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuScanError::Unavailable(reason) => write!(f, "gpu unavailable: {reason}"),
            GpuScanError::Device(reason) => write!(f, "gpu device error: {reason}"),
            GpuScanError::Invalid(reason) => write!(f, "gpu invalid input: {reason}"),
        }
    }
}

impl std::error::Error for GpuScanError {}

/// Whether opt-in GPU diagnostics are enabled (`LODEDB_GPU_DEBUG` set).
///
/// Mirrors the Python serving layer's one-time backend log: a built session and
/// the first engaged scan each emit one stderr line, so a deployment can confirm
/// the GPU path is actually serving rather than silently falling back.
fn debug_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("LODEDB_GPU_DEBUG").is_some())
}

/// Returns whether a CUDA driver library can be opened at all.
///
/// Cached after the first probe. cudarc panics (rather than returning an error) if
/// the driver library is entirely absent, so this gate must run before any other
/// CUDA touch; it only attempts a `dlopen` and is documented panic-free.
pub fn cuda_runtime_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        // SAFETY: `is_culib_present` performs no CUDA API calls. It only tries to
        // `dlopen` the candidate driver libraries and reports whether any opened,
        // and is documented to never panic.
        unsafe { cudarc::driver::sys::is_culib_present() }
    })
}

/// Per-shape device scratch reused across queries of the same `(nq, k)`.
///
/// Allocating the score matrix and top-k buffers, and zeroing them, dominated the
/// per-call cost at the small batch sizes the serving layer drives; caching them
/// keyed by shape turns every steady-state query into copies + kernels with no
/// device allocation. The query buffer is overwritten in place each call.
struct GpuScratch {
    q_dev: CudaSlice<f32>,      // nq * dim, rotated queries
    scores_dev: CudaSlice<f32>, // nq * n, GEMM output (beta=0 overwrites, no zeroing needed)
    idx_dev: CudaSlice<u32>,    // nq * k, top-k slots
    val_dev: CudaSlice<f32>,    // nq * k, top-k scores
}

/// A generation's reconstructed rows resident on the GPU, plus the host-side
/// rotation and stable-id mapping needed to score and label a query batch.
pub struct GpuScanSession {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    // Keeps the loaded module alive for as long as `func` references it.
    _module: Arc<CudaModule>,
    func: CudaFunction,
    // Resident rows pre-multiplied by the rotation (rows @ rotation), so queries
    // are scored raw (no per-query rotation). See `build_inner`.
    rows_dev: CudaSlice<f32>,
    stable_ids: Vec<u64>,
    dim: usize,
    n: usize,
    // Device scratch reused across calls, keyed by (nq, k). Interior mutability so
    // the hot query path stays `&self`; the engine is thread-confined.
    scratch: RefCell<HashMap<(usize, usize), GpuScratch>>,
}

impl GpuScanSession {
    /// Builds a resident session from reconstructed rotated-calibrated rows.
    ///
    /// `rows` is the flat `[n * dim]` row buffer in slot order (as returned by
    /// `IdMapIndex::reconstruct_all`), `stable_ids` the `[n]` slot-to-id map, and
    /// `rotation` the flat `[dim * dim]` row-major rotation matrix. Returns
    /// [`GpuScanError::Unavailable`] when no driver loads (caller uses the CPU
    /// kernel and should not retry) and [`GpuScanError::Device`] on any CUDA call
    /// failure.
    pub fn build(
        rows: &[f32],
        stable_ids: &[u64],
        rotation: &[f32],
        dim: usize,
    ) -> Result<Self, GpuScanError> {
        if dim == 0 {
            return Err(GpuScanError::Invalid("dim must be positive".into()));
        }
        let n = stable_ids.len();
        if rows.len() != n * dim {
            return Err(GpuScanError::Invalid(format!(
                "rows length {} does not match n*dim {}",
                rows.len(),
                n * dim
            )));
        }
        if rotation.len() != dim * dim {
            return Err(GpuScanError::Invalid(format!(
                "rotation length {} does not match dim*dim {}",
                rotation.len(),
                dim * dim
            )));
        }
        if n == 0 {
            return Err(GpuScanError::Invalid("cannot build a session over 0 rows".into()));
        }
        if !cuda_runtime_available() {
            return Err(GpuScanError::Unavailable("no CUDA driver library".into()));
        }
        // cudarc loads libcublas/libnvrtc lazily on first use and PANICS (rather
        // than returning an error) when a library is absent or on a path its
        // `dlopen` does not search. A GPU host with an unusual CUDA layout must
        // fall back to the CPU kernel, never crash the process, so the CUDA build
        // runs under `catch_unwind` and a panic becomes an `Unavailable` fallback.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Self::build_inner(rows, stable_ids, rotation, dim, n)
        })) {
            Ok(result) => result,
            Err(_) => Err(GpuScanError::Unavailable(
                "CUDA library load panicked (missing libcublas/libnvrtc); using CPU kernel".into(),
            )),
        }
    }

    /// The CUDA-touching half of [`Self::build`], isolated so a cudarc panic from a
    /// missing/unfindable library is contained by the caller's `catch_unwind`.
    fn build_inner(
        rows: &[f32],
        stable_ids: &[u64],
        rotation: &[f32],
        dim: usize,
        n: usize,
    ) -> Result<Self, GpuScanError> {
        let ctx = CudaContext::new(0).map_err(|err| GpuScanError::Unavailable(err.to_string()))?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        // Bake the rotation into the resident rows ONCE at build, so per-query the
        // queries go in raw. Since score(q, row) = <q @ rotation^T, row> reassociates
        // to <q, row @ rotation>, storing `rows @ rotation` makes the per-batch host
        // rotation (the dominant cost: a naive O(nq * dim^2) loop) disappear, and the
        // scores GEMM then takes raw queries unchanged. Done on the GPU as a single
        // GEMM, so the build stays fast.
        let recon_dev = stream
            .clone_htod(rows)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let rotation_dev = stream
            .clone_htod(rotation)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let mut rows_dev = stream
            .alloc_zeros::<f32>(n * dim)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        // Row-major rowrot[r][d] = sum_d' rows[r][d'] * rotation[d'][d] = (rows @ rotation).
        // Computed column-major as rowrot^T (dim x n): a = rotation (OP_N), b = rows (OP_N).
        let bake = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: dim as i32,
            n: n as i32,
            k: dim as i32,
            alpha: 1.0f32,
            lda: dim as i32,
            ldb: dim as i32,
            beta: 0.0f32,
            ldc: dim as i32,
        };
        // SAFETY: recon_dev is n*dim, rotation_dev is dim*dim, rows_dev is n*dim; the
        // GEMM is synchronized below before the transient inputs are dropped.
        unsafe { blas.gemm(bake, &rotation_dev, &recon_dev, &mut rows_dev) }
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        stream
            .synchronize()
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        drop(recon_dev);
        drop(rotation_dev);

        let ptx = compile_ptx(TOPK_KERNEL)
            .map_err(|err| GpuScanError::Device(format!("nvrtc compile failed: {err}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let func = module
            .load_function("topk_argmax")
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        if debug_enabled() {
            eprintln!("lodedb_gpu: resident GPU session built (rows={n}, dim={dim})");
        }
        Ok(Self {
            stream,
            blas,
            _module: module,
            func,
            rows_dev,
            stable_ids: stable_ids.to_vec(),
            dim,
            n,
            scratch: RefCell::new(HashMap::new()),
        })
    }

    /// Number of resident rows.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the session holds no rows (never true for a built session).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Scores a query batch and returns flat `[nq * k]` `(scores, stable_ids)`.
    ///
    /// `queries` is the flat `[nq * dim]` unrotated query buffer. The result rows
    /// are ordered descending by score, ascending by stable id on exact ties, so
    /// the layout matches `IdMapIndex::search` and the caller's row assembly is
    /// reused unchanged. `k` must be `<= len()`.
    pub fn search(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
    ) -> Result<(Vec<f32>, Vec<u64>), GpuScanError> {
        // Defense in depth: a built session has already loaded its CUDA libraries,
        // so a panic here is not expected, but the scan must never crash the host
        // process; a panic falls back to the CPU kernel like any device error.
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.search_inner(queries, nq, k)
        })) {
            Ok(result) => result,
            Err(_) => Err(GpuScanError::Device("CUDA scan panicked; using CPU kernel".into())),
        }
    }

    fn search_inner(
        &self,
        queries: &[f32],
        nq: usize,
        k: usize,
    ) -> Result<(Vec<f32>, Vec<u64>), GpuScanError> {
        if nq == 0 || k == 0 {
            return Err(GpuScanError::Invalid("nq and k must be positive".into()));
        }
        if k > self.n {
            return Err(GpuScanError::Invalid(format!(
                "k {} exceeds resident rows {}",
                k, self.n
            )));
        }
        if queries.len() != nq * self.dim {
            return Err(GpuScanError::Invalid(format!(
                "queries length {} does not match nq*dim {}",
                queries.len(),
                nq * self.dim
            )));
        }

        // Queries are scored raw: the rotation is baked into the resident rows at
        // build (rows @ rotation), so there is no per-query host rotation.

        // Reuse device scratch for this (nq, k) shape; allocate only on first sight.
        // The serving layer drives a small set of shapes, so steady-state queries
        // do no device allocation or zeroing — only copies and kernels.
        let mut cache = self.scratch.borrow_mut();
        if !cache.contains_key(&(nq, k)) {
            let scratch = GpuScratch {
                q_dev: self
                    .stream
                    .alloc_zeros::<f32>(nq * self.dim)
                    .map_err(|err| GpuScanError::Device(err.to_string()))?,
                scores_dev: self
                    .stream
                    .alloc_zeros::<f32>(nq * self.n)
                    .map_err(|err| GpuScanError::Device(err.to_string()))?,
                idx_dev: self
                    .stream
                    .alloc_zeros::<u32>(nq * k)
                    .map_err(|err| GpuScanError::Device(err.to_string()))?,
                val_dev: self
                    .stream
                    .alloc_zeros::<f32>(nq * k)
                    .map_err(|err| GpuScanError::Device(err.to_string()))?,
            };
            cache.insert((nq, k), scratch);
        }
        let GpuScratch {
            q_dev,
            scores_dev,
            idx_dev,
            val_dev,
        } = cache.get_mut(&(nq, k)).expect("scratch was just inserted");

        // Upload the raw queries into the cached query buffer (in place).
        self.stream
            .memcpy_htod(queries, q_dev)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        // Row-major scores[query*n + row] = q_rot(nq x dim) @ rows(n x dim)^T,
        // computed column-major as scores^T (n x nq): a = rows (OP_T), b = q_rot.
        // beta=0 overwrites every element of scores_dev, so it needs no zeroing.
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: self.n as i32,
            n: nq as i32,
            k: self.dim as i32,
            alpha: 1.0f32,
            lda: self.dim as i32,
            ldb: self.dim as i32,
            beta: 0.0f32,
            ldc: self.n as i32,
        };
        // SAFETY: shapes/leading dims are validated above; all buffers live on
        // `self.stream` and are sized nq*n / nq*dim for this (nq, k) shape.
        unsafe { self.blas.gemm(cfg, &self.rows_dev, &*q_dev, &mut *scores_dev) }
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        let cfg_launch = LaunchConfig {
            grid_dim: (nq as u32, 1, 1),
            block_dim: (TOPK_BLOCK, 1, 1),
            shared_mem_bytes: TOPK_BLOCK
                * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()) as u32,
        };
        let n_i32 = self.n as i32;
        let k_i32 = k as i32;
        let mut builder = self.stream.launch_builder(&self.func);
        builder
            .arg(&mut *scores_dev)
            .arg(&n_i32)
            .arg(&k_i32)
            .arg(&mut *idx_dev)
            .arg(&mut *val_dev);
        // SAFETY: the kernel reads/writes only the bound buffers within bounds
        // (`scores_dev` is nq*n, `idx_dev`/`val_dev` are nq*k, grid is nq blocks);
        // shared memory matches `block_dim`'s two per-thread scratch arrays.
        unsafe { builder.launch(cfg_launch) }
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        if debug_enabled() {
            static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            LOGGED.get_or_init(|| {
                eprintln!("lodedb_gpu: GPU scan engaged (nq={nq}, k={k}, rows={})", self.n);
            });
        }

        // clone_dtoh issues a synchronous copy, which waits on the GEMM and kernel
        // queued above, so no separate stream synchronize is needed.
        let host_idx = self
            .stream
            .clone_dtoh(&*idx_dev)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;
        let host_val = self
            .stream
            .clone_dtoh(&*val_dev)
            .map_err(|err| GpuScanError::Device(err.to_string()))?;

        Ok(self.finalize(&host_idx, &host_val, nq, k))
    }

    /// Maps device slots to stable ids and fixes the deterministic ordering
    /// (descending score, ascending stable id on exact ties) per query.
    fn finalize(
        &self,
        host_idx: &[u32],
        host_val: &[f32],
        nq: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<u64>) {
        let mut scores = vec![0.0f32; nq * k];
        let mut ids = vec![0u64; nq * k];
        let mut row: Vec<(f32, u64)> = Vec::with_capacity(k);
        for q in 0..nq {
            row.clear();
            for j in 0..k {
                let slot = host_idx[q * k + j] as usize;
                let stable_id = self.stable_ids.get(slot).copied().unwrap_or(0);
                row.push((host_val[q * k + j], stable_id));
            }
            row.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (j, (score, id)) in row.iter().enumerate() {
                scores[q * k + j] = *score;
                ids[q * k + j] = *id;
            }
        }
        (scores, ids)
    }
}

/// Rotates a query batch on the host: `q_rot[b][out] = sum_in q[b][in] *
/// rotation[out*dim + in]` (row-major rotation), i.e. `q @ rotation^T`.
///
/// Only the parity test uses this now: the production path bakes the rotation into
/// the resident rows at build (see `build_inner`) so queries are scored raw.
#[cfg(test)]
fn rotate_queries(queries: &[f32], nq: usize, dim: usize, rotation: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; nq * dim];
    for b in 0..nq {
        let query = &queries[b * dim..(b + 1) * dim];
        let target = &mut out[b * dim..(b + 1) * dim];
        for (out_d, slot) in target.iter_mut().enumerate() {
            let rotation_row = &rotation[out_d * dim..(out_d + 1) * dim];
            let mut acc = 0.0f32;
            for (q_value, r_value) in query.iter().zip(rotation_row) {
                acc += q_value * r_value;
            }
            *slot = acc;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rotate_queries_matches_manual_inner_product() {
        // 2x2 rotation, two queries; verify q @ rotation^T against hand math.
        let rotation = vec![1.0, 2.0, 3.0, 4.0]; // [[1,2],[3,4]] row-major
        let queries = vec![1.0, 0.0, 0.0, 1.0]; // e0, e1
        let rotated = rotate_queries(&queries, 2, 2, &rotation);
        // q=e0: out0 = 1*1+0*2 = 1; out1 = 1*3+0*4 = 3
        // q=e1: out0 = 0*1+1*2 = 2; out1 = 0*3+1*4 = 4
        assert_eq!(rotated, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn cuda_probe_is_stable() {
        // Must not panic on a CPU-only host; just exercises the cached probe.
        let _ = cuda_runtime_available();
    }

    /// Deterministic pseudo-random f32 in [-1, 1) via an LCG, so the parity test
    /// needs no `rand` dependency and is reproducible on any host.
    fn pseudo_fill(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let bits = (state >> 33) as u32;
                (bits as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn gpu_topk_matches_cpu_reference() {
        // Runs only where a CUDA driver is present; a no-op elsewhere (CI/macOS).
        if !cuda_runtime_available() {
            eprintln!("skipping gpu_topk_matches_cpu_reference: no CUDA runtime");
            return;
        }
        let (n, dim, nq, k) = (300usize, 64usize, 8usize, 10usize);
        let rows = pseudo_fill(n * dim, 0x1234_5678);
        let rotation = pseudo_fill(dim * dim, 0x9abc_def0);
        let queries = pseudo_fill(nq * dim, 0x55aa_55aa);
        let stable_ids: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(2_654_435_761) | 1).collect();

        let session = GpuScanSession::build(&rows, &stable_ids, &rotation, dim)
            .expect("session build on a CUDA host");
        let (scores, ids) = session.search(&queries, nq, k).expect("gpu search");
        assert_eq!(scores.len(), nq * k);
        assert_eq!(ids.len(), nq * k);

        let id_to_slot: std::collections::HashMap<u64, usize> = stable_ids
            .iter()
            .enumerate()
            .map(|(slot, &id)| (id, slot))
            .collect();

        for q in 0..nq {
            let rotated = rotate_queries(&queries[q * dim..(q + 1) * dim], 1, dim, &rotation);
            // CPU reference: the exact maximum score over all rows for this query.
            let mut best = f32::NEG_INFINITY;
            for slot in 0..n {
                let row = &rows[slot * dim..(slot + 1) * dim];
                let score: f32 = rotated.iter().zip(row).map(|(a, b)| a * b).sum();
                if score > best {
                    best = score;
                }
            }
            // The top-1 score must equal the true maximum (id may be ambiguous only
            // under an exact tie, which random data does not produce).
            let tol_top = 1e-2 * best.abs().max(1.0);
            assert!(
                (scores[q * k] - best).abs() <= tol_top,
                "query {q} top-1 score {} != cpu max {best}",
                scores[q * k]
            );
            // Each returned row is in descending order and carries the correct score.
            let mut prev = f32::INFINITY;
            for j in 0..k {
                let score = scores[q * k + j];
                assert!(score <= prev + 1e-3, "query {q} rank {j} not descending");
                prev = score;
                let slot = id_to_slot[&ids[q * k + j]];
                let row = &rows[slot * dim..(slot + 1) * dim];
                let cpu_score: f32 = rotated.iter().zip(row).map(|(a, b)| a * b).sum();
                let tol = 1e-2 * cpu_score.abs().max(1.0);
                assert!(
                    (score - cpu_score).abs() <= tol,
                    "query {q} rank {j} id {} score {score} != cpu {cpu_score}",
                    ids[q * k + j]
                );
            }
        }
    }
}
