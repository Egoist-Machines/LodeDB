pub mod commit_manifest;
pub mod legacy;
pub mod lexical_store;
pub mod lsn;
pub mod multivec_store;
pub mod state_journal;
pub mod text_store;
pub mod tvim_delta;
mod util;
pub mod wal;

use crate::error::CoreError;
use crate::storage::commit_manifest::{
    base_json_path, base_tvim_path, base_tvlex_path, base_tvmv_path, base_tvtext_path,
    build_commit_body, commit_manifest_path, generation_dir, list_base_epochs,
    read_commit_manifest, write_commit_manifest, CommitBodyInput, CommitManifest,
};
use crate::storage::lexical_store::TokenLists;
use crate::storage::multivec_store::MultiVecMap;
use crate::storage::state_journal::StateJournalDeltaInput;
use crate::storage::tvim_delta::{TvimDeltaAppendInput, TvimDeltaArray};
use crate::storage::util::{corrupt, get_str, invalid, value_object, CoreResult};
use crate::storage::wal::WalRecord;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreLayout {
    Generation,
    LegacyTopLevelJson,
}

#[derive(Debug, Clone)]
pub struct LoadedStore {
    pub layout: StoreLayout,
    pub index_key: String,
    pub generation: u64,
    pub base_epoch: u64,
    pub state: Value,
    pub tvim_path: Option<PathBuf>,
    pub tvim_manifest: Option<Value>,
    pub raw_text: BTreeMap<String, String>,
    pub lexical_tokens: BTreeMap<String, TokenLists>,
    pub multivec: MultiVecMap,
    pub wal_records: Vec<WalRecord>,
}

impl LoadedStore {
    pub fn document_count(&self) -> usize {
        self.state
            .get("document_hashes")
            .and_then(Value::as_object)
            .map_or(0, |documents| documents.len())
    }

    pub fn chunk_count(&self) -> usize {
        self.state
            .get("chunks")
            .and_then(Value::as_array)
            .map_or(0, |chunks| chunks.len())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    pub read_only: bool,
    pub read_wal: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            read_only: true,
            read_wal: false,
        }
    }
}

pub fn load_store(
    persistence_dir: impl AsRef<Path>,
    index_key: &str,
    options: LoadOptions,
) -> Result<LoadedStore, CoreError> {
    let persistence_dir = persistence_dir.as_ref();
    if let Some(manifest) =
        commit_manifest::read_commit_manifest(&commit_manifest_path(persistence_dir, index_key))?
    {
        return load_generation_store(persistence_dir, manifest, options);
    }
    let legacy_path = persistence_dir.join(format!("{index_key}.json"));
    if legacy_path.is_file() {
        let legacy = legacy::load_top_level_json(persistence_dir, index_key)?;
        return Ok(LoadedStore {
            layout: StoreLayout::LegacyTopLevelJson,
            index_key: index_key.to_string(),
            generation: 0,
            base_epoch: 0,
            state: legacy.payload,
            tvim_path: legacy_tvim_path(persistence_dir, index_key),
            tvim_manifest: None,
            raw_text: legacy.raw_text,
            lexical_tokens: BTreeMap::new(),
            multivec: MultiVecMap::new(),
            wal_records: Vec::new(),
        });
    }
    Err(invalid(format!(
        "no persisted store found for index key {index_key}"
    )))
}

pub fn load_generation_store(
    persistence_dir: &Path,
    manifest: CommitManifest,
    options: LoadOptions,
) -> CoreResult<LoadedStore> {
    let body = value_object(&manifest.body, "commit manifest body")?;
    let index_key = get_str(body, "index_key").to_string();
    if index_key.is_empty() {
        return Err(corrupt("commit manifest body is missing index_key"));
    }
    let base_epoch = manifest.base_epoch();
    let state_base_path = base_json_path(persistence_dir, &index_key, base_epoch);
    let json_manifest = manifest.store_manifest("json");
    let mut state = state_journal::read_base_payload(&state_base_path, json_manifest)?;
    if let Some(json_manifest) = json_manifest {
        state_journal::replay_onto_payload(&mut state, &state_base_path, json_manifest)?;
    }
    let tvim_base_path = base_tvim_path(persistence_dir, &index_key, base_epoch);
    let tvim_manifest = manifest.store_manifest("tvim");
    tvim_delta::validate(&tvim_base_path, tvim_manifest)?;
    let tvim_path = if tvim_manifest.is_some() {
        Some(tvim_base_path)
    } else {
        None
    };

    let raw_text = match manifest.store_manifest("tvtext") {
        Some(tvtext_manifest) => {
            // A journaled tvtext manifest carries a `base` object (text_store
            // writes and reads it). The unreleased pre-journal shape is
            // `{present, sha256}`, pinning a single `text-g<gen>.tvtext` file the
            // native reader cannot load: it would silently yield empty text and
            // then overwrite the real text on the next durable write. Refuse so
            // the caller falls back to the Python reader, which understands the
            // legacy shape and migrates it into the journal.
            if tvtext_manifest.get("base").is_none() {
                return Err(invalid(
                    "unsupported pre-journal raw-text layout (no journaled base); \
                     the Python reader must migrate this store",
                ));
            }
            text_store::load(
                &base_tvtext_path(persistence_dir, &index_key, base_epoch),
                Some(tvtext_manifest),
            )?
        }
        None => BTreeMap::new(),
    };
    let lexical_tokens = match manifest.store_manifest("tvlex") {
        Some(tvlex_manifest) => lexical_store::load(
            &base_tvlex_path(persistence_dir, &index_key, base_epoch),
            Some(tvlex_manifest),
        )?,
        None => BTreeMap::new(),
    };
    let multivec = match manifest.store_manifest("tvmv") {
        Some(tvmv_manifest) => multivec_store::load(
            &base_tvmv_path(persistence_dir, &index_key, base_epoch),
            Some(tvmv_manifest),
        )?,
        None => MultiVecMap::new(),
    };
    let wal_records = if options.read_wal && !options.read_only {
        wal::read_records(&wal::wal_path(persistence_dir, &index_key))?
    } else {
        Vec::new()
    };
    Ok(LoadedStore {
        layout: StoreLayout::Generation,
        index_key,
        generation: manifest.generation(),
        base_epoch,
        state,
        tvim_path,
        tvim_manifest: tvim_manifest.cloned(),
        raw_text,
        lexical_tokens,
        multivec,
        wal_records,
    })
}

fn legacy_tvim_path(persistence_dir: &Path, index_key: &str) -> Option<PathBuf> {
    let path = persistence_dir.join(format!("{index_key}.tvim"));
    path.is_file().then_some(path)
}

pub fn fixture_root(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

#[derive(Debug, Clone)]
pub struct GenerationWriteOptions {
    pub fsync: bool,
    pub retained_epochs: usize,
}

impl Default for GenerationWriteOptions {
    fn default() -> Self {
        Self {
            fsync: false,
            retained_epochs: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TvimBaseWrite<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub calibration_fingerprint: u64,
}

#[derive(Debug, Clone)]
pub struct GenerationCommitInput<'a> {
    pub index_key: &'a str,
    pub generation: u64,
    pub base_epoch: u64,
    pub state: &'a Value,
    pub tvim: Option<TvimBaseWrite<'a>>,
    pub raw_text: Option<&'a BTreeMap<String, String>>,
    pub lexical_tokens: Option<&'a BTreeMap<String, TokenLists>>,
    pub multivec: Option<&'a MultiVecMap>,
    /// Whether the retained document-text base is zstd-compressed. Threaded from
    /// the engine's effective (persisted-or-seeded) flag down to
    /// [`text_store::record_base`].
    pub compress_text: bool,
}

pub fn write_generation_commit(
    persistence_dir: impl AsRef<Path>,
    input: GenerationCommitInput<'_>,
    options: GenerationWriteOptions,
) -> CoreResult<Value> {
    let persistence_dir = persistence_dir.as_ref();
    fs::create_dir_all(generation_dir(persistence_dir, input.index_key)).map_err(|error| {
        corrupt(format!(
            "generation directory could not be created for {}: {error}",
            input.index_key
        ))
    })?;

    let state_path = base_json_path(persistence_dir, input.index_key, input.base_epoch);
    state_journal::write_base_json(&state_path, input.state, options.fsync)?;
    let document_count = input
        .state
        .get("document_hashes")
        .and_then(Value::as_object)
        .map_or(0, |documents| documents.len());
    let chunk_count = input
        .state
        .get("chunks")
        .and_then(Value::as_array)
        .map_or(0, |chunks| chunks.len());
    let json_manifest =
        state_journal::record_base(&state_path, document_count, chunk_count, options.fsync)?;

    let tvim_manifest = match input.tvim {
        Some(tvim) => Some(tvim_delta::persist_base_bytes(
            &base_tvim_path(persistence_dir, input.index_key, input.base_epoch),
            tvim.bytes,
            tvim.rows,
            tvim.calibration_fingerprint,
            options.fsync,
        )?),
        None => None,
    };
    let text_manifest = match input.raw_text {
        Some(raw_text) if !raw_text.is_empty() => Some(text_store::record_base(
            &base_tvtext_path(persistence_dir, input.index_key, input.base_epoch),
            raw_text,
            options.fsync,
            input.compress_text,
        )?),
        _ => None,
    };
    let lexical_manifest = match input.lexical_tokens {
        Some(tokens) if !tokens.is_empty() => Some(lexical_store::record_base(
            &base_tvlex_path(persistence_dir, input.index_key, input.base_epoch),
            tokens,
            options.fsync,
        )?),
        _ => None,
    };
    let multivec_manifest = match input.multivec {
        Some(multivec) if !multivec.is_empty() => Some(multivec_store::record_base(
            &base_tvmv_path(persistence_dir, input.index_key, input.base_epoch),
            multivec,
            options.fsync,
        )?),
        _ => None,
    };
    let body = build_commit_body(CommitBodyInput {
        index_key: input.index_key,
        generation: input.generation,
        base_epoch: input.base_epoch,
        document_count,
        chunk_count,
        json_manifest: Some(json_manifest),
        tvim_manifest,
        tvtext_manifest: text_manifest,
        tvlex_manifest: lexical_manifest,
        tvmv_manifest: multivec_manifest,
    });
    write_commit_manifest(
        &commit_manifest_path(persistence_dir, input.index_key),
        &body,
        options.fsync,
    )?;
    gc_after_base_rewrite(
        persistence_dir,
        input.index_key,
        input.base_epoch,
        options.retained_epochs,
    )?;
    Ok(body)
}

/// Encoded tvim row changes for a generation delta. Owned so callers can build it
/// from the live index without lifetime juggling across the export boundary.
#[derive(Debug, Clone, Default)]
pub struct TvimDeltaWrite {
    pub upsert_stable_ids: Vec<u64>,
    pub upsert_codes: Vec<u8>,
    pub upsert_scales: Vec<f32>,
    pub removed_stable_ids: Vec<u64>,
    pub rows_after: usize,
    pub calibration_fingerprint: u64,
}

impl TvimDeltaWrite {
    fn is_empty(&self) -> bool {
        self.upsert_stable_ids.is_empty() && self.removed_stable_ids.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct GenerationDeltaInput<'a> {
    pub index_key: &'a str,
    pub generation: u64,
    pub base_epoch: u64,
    pub state_header: &'a Value,
    pub upserted_documents: Vec<Value>,
    pub deleted_document_ids: Vec<String>,
    pub document_count_after: usize,
    pub chunk_count_after: usize,
    /// `Some` when the index has vectors; an empty write reuses the base manifest.
    pub tvim: Option<TvimDeltaWrite>,
    /// `Some` only when `store_text` is on; carries the changed documents' text.
    pub raw_text_upserts: Option<BTreeMap<String, String>>,
    /// Live documents whose retained raw text was cleared since the base. A delta
    /// cannot express the clear by omission (absence means "unchanged", so the
    /// base's old text would resurrect on reload); these are written to the text
    /// delta as explicit deletes.
    pub raw_text_clears: Vec<String>,
    /// `Some` only when `index_text` is on; carries the changed documents' tokens.
    pub lexical_upserts: Option<BTreeMap<String, TokenLists>>,
    /// Live documents whose lexical tokens were cleared since the base; written to
    /// the lexical delta as explicit deletes, like `raw_text_clears`.
    pub lexical_clears: Vec<String>,
    /// `Some` only for late-interaction stores; the changed documents' matrices.
    pub multivec_upserts: Option<MultiVecMap>,
    pub document_deletes: Vec<String>,
    /// Whether a new document-text delta segment is zstd-compressed. Threaded
    /// from the engine's effective (persisted-or-seeded) flag down to
    /// [`text_store::append_delta`].
    pub compress_text: bool,
}

/// Appends an O(changed) generation delta onto the live base and re-seals the
/// commit manifest. Unlike [`write_generation_commit`], this never rewrites the
/// full bases: it appends per-store delta segments for what changed and carries
/// the unchanged stores' manifests forward from the previous commit, so a
/// single-row write stays O(changed). The base epoch is unchanged; only the
/// generation advances. Requires a prior commit at `base_epoch` (the engine falls
/// back to a full base on a cold build or compaction).
pub fn write_generation_delta(
    persistence_dir: impl AsRef<Path>,
    input: GenerationDeltaInput<'_>,
    fsync: bool,
) -> CoreResult<Value> {
    let persistence_dir = persistence_dir.as_ref();
    let previous =
        read_commit_manifest(&commit_manifest_path(persistence_dir, input.index_key))?
            .ok_or_else(|| corrupt("generation delta requires an existing commit manifest"))?;

    let state_base_path = base_json_path(persistence_dir, input.index_key, input.base_epoch);
    let json_manifest = state_journal::append_delta(
        &state_base_path,
        StateJournalDeltaInput {
            upserted_documents: input.upserted_documents,
            deleted_document_ids: input.deleted_document_ids,
            state_header: input.state_header.clone(),
            document_count_after: input.document_count_after,
            chunk_count_after: input.chunk_count_after,
            generation: input.generation,
            fsync,
        },
    )?;

    // tvim: append a delta when vectors changed, else carry the base manifest.
    let tvim_manifest = match input.tvim {
        Some(tvim) if !tvim.is_empty() => {
            let base_tvim_path = base_tvim_path(persistence_dir, input.index_key, input.base_epoch);
            let upsert_id_bytes = u64_slice_to_le_bytes(&tvim.upsert_stable_ids);
            let removed_id_bytes = u64_slice_to_le_bytes(&tvim.removed_stable_ids);
            let scale_bytes = f32_slice_to_le_bytes(&tvim.upsert_scales);
            // Match the host writer's array layout exactly so the Python reader
            // replays these deltas: ids and scales are 1-D, codes are 2-D
            // (rows x bytes_per_vector). The native reader keys off byte length,
            // but the Python reader reshapes by these stored shapes.
            let code_width = if tvim.upsert_stable_ids.is_empty() {
                0
            } else {
                tvim.upsert_codes.len() / tvim.upsert_stable_ids.len()
            };
            let arrays = [
                TvimDeltaArray {
                    name: "upsert_stable_ids",
                    dtype: "uint64",
                    shape: vec![tvim.upsert_stable_ids.len()],
                    bytes: &upsert_id_bytes,
                },
                TvimDeltaArray {
                    name: "upsert_codes",
                    dtype: "uint8",
                    shape: vec![tvim.upsert_stable_ids.len(), code_width],
                    bytes: &tvim.upsert_codes,
                },
                TvimDeltaArray {
                    name: "upsert_scales",
                    dtype: "float32",
                    shape: vec![tvim.upsert_scales.len()],
                    bytes: &scale_bytes,
                },
                TvimDeltaArray {
                    name: "removed_stable_ids",
                    dtype: "uint64",
                    shape: vec![tvim.removed_stable_ids.len()],
                    bytes: &removed_id_bytes,
                },
            ];
            Some(tvim_delta::append_delta_arrays(
                &base_tvim_path,
                TvimDeltaAppendInput {
                    generation: input.generation,
                    calibration_fingerprint: tvim.calibration_fingerprint,
                    rows_after: tvim.rows_after,
                    arrays: &arrays,
                    upsert_rows: tvim.upsert_stable_ids.len(),
                    removed_rows: tvim.removed_stable_ids.len(),
                    fsync,
                },
            )?)
        }
        _ => previous.store_manifest("tvim").cloned(),
    };

    let text_manifest = match input.raw_text_upserts {
        Some(upserts)
            if !upserts.is_empty()
                || !input.document_deletes.is_empty()
                || !input.raw_text_clears.is_empty() =>
        {
            let mut deleted = input.document_deletes.clone();
            deleted.extend(input.raw_text_clears.iter().cloned());
            Some(text_store::append_delta(
                &base_tvtext_path(persistence_dir, input.index_key, input.base_epoch),
                &upserts,
                &deleted,
                input.document_count_after,
                fsync,
                input.compress_text,
            )?)
        }
        _ => previous.store_manifest("tvtext").cloned(),
    };
    let lexical_manifest = match input.lexical_upserts {
        Some(upserts)
            if !upserts.is_empty()
                || !input.document_deletes.is_empty()
                || !input.lexical_clears.is_empty() =>
        {
            let mut deleted = input.document_deletes.clone();
            deleted.extend(input.lexical_clears.iter().cloned());
            Some(lexical_store::append_delta(
                &base_tvlex_path(persistence_dir, input.index_key, input.base_epoch),
                &upserts,
                &deleted,
                input.document_count_after,
                fsync,
            )?)
        }
        _ => previous.store_manifest("tvlex").cloned(),
    };
    // Only append a multi-vector delta for a store that actually has a tvmv base:
    // a patch-matrix upsert, or a delete against an existing multi-vector base.
    // A non-late-interaction store never wrote a tvmv base, so its deletes must not
    // try to append one (there is no manifest to read).
    let multivec_manifest = match input.multivec_upserts {
        Some(upserts)
            if !upserts.is_empty()
                || (!input.document_deletes.is_empty()
                    && previous.store_manifest("tvmv").is_some()) =>
        {
            Some(multivec_store::append_delta(
                &base_tvmv_path(persistence_dir, input.index_key, input.base_epoch),
                &upserts,
                &input.document_deletes,
                input.document_count_after,
                fsync,
            )?)
        }
        _ => previous.store_manifest("tvmv").cloned(),
    };

    let body = build_commit_body(CommitBodyInput {
        index_key: input.index_key,
        generation: input.generation,
        base_epoch: input.base_epoch,
        document_count: input.document_count_after,
        chunk_count: input.chunk_count_after,
        json_manifest: Some(json_manifest),
        tvim_manifest,
        tvtext_manifest: text_manifest,
        tvlex_manifest: lexical_manifest,
        tvmv_manifest: multivec_manifest,
    });
    write_commit_manifest(
        &commit_manifest_path(persistence_dir, input.index_key),
        &body,
        fsync,
    )?;
    Ok(body)
}

/// Returns the state-journal delta backlog at `base_epoch` as
/// `(segment_count, journaled_document_count)`, used to decide when a fresh base
/// rewrite (compaction) is cheaper than another delta. Every generation commit
/// appends exactly one state-journal segment.
pub fn generation_delta_backlog(
    persistence_dir: &Path,
    index_key: &str,
    base_epoch: u64,
) -> CoreResult<(usize, usize)> {
    let base_path = base_json_path(persistence_dir, index_key, base_epoch);
    let Some(manifest) = state_journal::read_manifest_optional(&base_path)? else {
        return Ok((0, 0));
    };
    let deltas = manifest
        .as_object()
        .and_then(|object| object.get("deltas"))
        .and_then(Value::as_array);
    let Some(deltas) = deltas else {
        return Ok((0, 0));
    };
    let document_sum = deltas
        .iter()
        .map(|delta| {
            let upserted = delta
                .get("upserted_documents")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            let deleted = delta
                .get("deleted_documents")
                .and_then(Value::as_u64)
                .unwrap_or(0);
            (upserted + deleted) as usize
        })
        .sum();
    Ok((deltas.len(), document_sum))
}

fn u64_slice_to_le_bytes(values: &[u64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn f32_slice_to_le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

pub fn gc_after_base_rewrite(
    persistence_dir: &Path,
    index_key: &str,
    live_epoch: u64,
    retained_epochs: usize,
) -> CoreResult<()> {
    let mut epochs = list_base_epochs(persistence_dir, index_key)?
        .into_iter()
        .collect::<Vec<_>>();
    epochs.sort_by(|left, right| right.cmp(left));
    let keep = epochs
        .iter()
        .take(retained_epochs)
        .copied()
        .chain(std::iter::once(live_epoch))
        .collect::<std::collections::BTreeSet<_>>();
    for epoch in epochs {
        if keep.contains(&epoch) {
            continue;
        }
        for path in [
            base_json_path(persistence_dir, index_key, epoch),
            base_tvim_path(persistence_dir, index_key, epoch),
            base_tvtext_path(persistence_dir, index_key, epoch),
            base_tvlex_path(persistence_dir, index_key, epoch),
        ] {
            let _ = fs::remove_file(&path);
            for suffix in [
                state_journal::STATE_JOURNAL_DIR_SUFFIX,
                tvim_delta::TVIM_DELTA_DIR_SUFFIX,
                text_store::DOCUMENT_TEXT_DELTA_DIR_SUFFIX,
                lexical_store::LEXICAL_INDEX_DELTA_DIR_SUFFIX,
            ] {
                if let Some(name) = path.file_name() {
                    let _ = fs::remove_dir_all(path.with_file_name(format!(
                        "{}{}",
                        name.to_string_lossy(),
                        suffix
                    )));
                }
            }
        }
    }
    Ok(())
}
