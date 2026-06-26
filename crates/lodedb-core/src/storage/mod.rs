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
    base_json_path, base_tvim_path, base_tvlex_path, base_tvtext_path, commit_manifest_path,
    CommitManifest,
};
use crate::storage::lexical_store::TokenLists;
use crate::storage::util::{corrupt, get_str, invalid, value_object, CoreResult};
use crate::storage::wal::WalRecord;
use serde_json::Value;
use std::collections::BTreeMap;
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
    tvim_delta::validate(&tvim_base_path, manifest.store_manifest("tvim"))?;

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
        raw_text,
        lexical_tokens,
        wal_records,
    })
}

pub fn fixture_root(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}
