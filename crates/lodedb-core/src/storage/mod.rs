pub mod commit_manifest;
pub mod legacy;
pub mod lexical_store;
pub mod state_journal;
pub mod text_store;
pub mod tvim_delta;
mod util;
pub mod wal;

use crate::error::CoreError;
use crate::storage::commit_manifest::{
    base_json_path, base_tvim_path, base_tvlex_path, base_tvtext_path, build_commit_body,
    commit_manifest_path, generation_dir, list_base_epochs, write_commit_manifest, CommitBodyInput,
    CommitManifest,
};
use crate::storage::lexical_store::TokenLists;
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
        Some(tvtext_manifest) => text_store::load(
            &base_tvtext_path(persistence_dir, &index_key, base_epoch),
            Some(tvtext_manifest),
        )?,
        None => BTreeMap::new(),
    };
    let lexical_tokens = match manifest.store_manifest("tvlex") {
        Some(tvlex_manifest) => lexical_store::load(
            &base_tvlex_path(persistence_dir, &index_key, base_epoch),
            Some(tvlex_manifest),
        )?,
        None => BTreeMap::new(),
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
    let body = build_commit_body(CommitBodyInput {
        index_key: input.index_key,
        generation: input.generation,
        base_epoch: input.base_epoch,
        document_count,
        chunk_count,
        json_manifest: Some(json_manifest),
        tvim_present: tvim_manifest.is_some(),
        tvim_manifest,
        tvtext_manifest: text_manifest,
        tvlex_manifest: lexical_manifest,
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
