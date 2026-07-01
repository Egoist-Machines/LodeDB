use crate::storage::util::{
    body_sha256, corrupt, get_i64, get_str, py_canonical_json, read_json_object, write_text_atomic,
    CoreResult,
};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub const COMMIT_MANIFEST_SUFFIX: &str = ".commit.json";
pub const COMMIT_MANIFEST_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub struct CommitManifest {
    pub body: Value,
}

impl CommitManifest {
    pub fn index_key(&self) -> &str {
        self.body
            .as_object()
            .and_then(|object| object.get("index_key"))
            .and_then(Value::as_str)
            .unwrap_or("")
    }

    pub fn generation(&self) -> u64 {
        self.body
            .as_object()
            .and_then(|object| object.get("generation"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    }

    pub fn base_epoch(&self) -> u64 {
        self.body
            .as_object()
            .and_then(|object| object.get("base_epoch"))
            .and_then(Value::as_u64)
            .unwrap_or(0)
    }

    pub fn store_manifest(&self, key: &str) -> Option<&Value> {
        self.body
            .as_object()?
            .get(key)
            .filter(|value| !value.is_null())
    }
}

pub fn commit_manifest_path(persistence_dir: &Path, index_key: &str) -> PathBuf {
    persistence_dir.join(format!("{index_key}{COMMIT_MANIFEST_SUFFIX}"))
}

pub fn generation_dir(persistence_dir: &Path, index_key: &str) -> PathBuf {
    persistence_dir.join(format!("{index_key}.gen"))
}

pub fn base_json_path(persistence_dir: &Path, index_key: &str, epoch: u64) -> PathBuf {
    generation_dir(persistence_dir, index_key).join(format!("g{epoch}.json"))
}

pub fn base_tvim_path(persistence_dir: &Path, index_key: &str, epoch: u64) -> PathBuf {
    generation_dir(persistence_dir, index_key).join(format!("g{epoch}.tvim"))
}

pub fn base_tvtext_path(persistence_dir: &Path, index_key: &str, epoch: u64) -> PathBuf {
    generation_dir(persistence_dir, index_key).join(format!("g{epoch}.tvtext"))
}

pub fn base_tvlex_path(persistence_dir: &Path, index_key: &str, epoch: u64) -> PathBuf {
    generation_dir(persistence_dir, index_key).join(format!("g{epoch}.tvlex"))
}

pub fn base_tvmv_path(persistence_dir: &Path, index_key: &str, epoch: u64) -> PathBuf {
    generation_dir(persistence_dir, index_key).join(format!("g{epoch}.tvmv"))
}

pub fn read_commit_manifest(path: &Path) -> CoreResult<Option<CommitManifest>> {
    if !path.is_file() {
        return Ok(None);
    }
    let document = read_json_object(path, "commit manifest")?;
    if get_i64(&document, "schema_version", -1) != COMMIT_MANIFEST_SCHEMA_VERSION {
        return Err(corrupt("unsupported commit manifest schema version"));
    }
    let body = document
        .get("body")
        .ok_or_else(|| corrupt("commit manifest is missing its body or checksum"))?;
    if !body.is_object() || get_str(&document, "body_sha256").is_empty() {
        return Err(corrupt("commit manifest is missing its body or checksum"));
    }
    if body_sha256(body)? != get_str(&document, "body_sha256") {
        return Err(corrupt("commit manifest failed body checksum"));
    }
    Ok(Some(CommitManifest { body: body.clone() }))
}

#[derive(Debug, Clone)]
pub struct CommitBodyInput<'a> {
    pub index_key: &'a str,
    pub generation: u64,
    pub base_epoch: u64,
    pub document_count: usize,
    pub chunk_count: usize,
    pub json_manifest: Option<Value>,
    pub tvim_manifest: Option<Value>,
    pub tvtext_manifest: Option<Value>,
    pub tvlex_manifest: Option<Value>,
    pub tvmv_manifest: Option<Value>,
}

pub fn build_commit_body(input: CommitBodyInput<'_>) -> Value {
    serde_json::json!({
        "index_key": input.index_key,
        "generation": input.generation,
        "base_epoch": input.base_epoch,
        "document_count": input.document_count,
        "chunk_count": input.chunk_count,
        "json": input.json_manifest,
        "tvim": input.tvim_manifest,
        "tvtext": input.tvtext_manifest,
        "tvlex": input.tvlex_manifest,
        "tvmv": input.tvmv_manifest,
    })
}

pub fn write_commit_manifest(path: &Path, body: &Value, fsync: bool) -> CoreResult<usize> {
    if !body.is_object() {
        return Err(corrupt("commit manifest body must be a JSON object"));
    }
    let body_json = py_canonical_json(body)?;
    let body_sha = body_sha256(body)?;
    let document_json = format!(
        "{{\"body\":{body_json},\"body_sha256\":\"{body_sha}\",\"schema_version\":{COMMIT_MANIFEST_SCHEMA_VERSION}}}"
    );
    write_text_atomic(path, &document_json, fsync)
}

pub fn list_base_epochs(persistence_dir: &Path, index_key: &str) -> CoreResult<BTreeSet<u64>> {
    let mut epochs = BTreeSet::new();
    let dir = generation_dir(persistence_dir, index_key);
    if !dir.is_dir() {
        return Ok(epochs);
    }
    for entry in fs::read_dir(&dir)
        .map_err(|error| corrupt(format!("generation directory could not be read: {error}")))?
    {
        let entry = entry
            .map_err(|error| corrupt(format!("generation directory entry is corrupt: {error}")))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if let Some(epoch) = name
            .strip_prefix('g')
            .and_then(|rest| rest.strip_suffix(".json"))
            .and_then(|digits| digits.parse::<u64>().ok())
        {
            epochs.insert(epoch);
        }
    }
    Ok(epochs)
}
