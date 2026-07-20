use crate::storage::util::{
    body_sha256, corrupt, get_i64, get_str, py_canonical_json, read_json_object, write_text_atomic,
    CoreResult,
};
use serde_json::{Map, Value};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

pub const COMMIT_MANIFEST_SUFFIX: &str = ".commit.json";
pub const COMMIT_MANIFEST_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub struct CommitManifest {
    pub body: Value,
    /// The verified body checksum from disk. Changes on every root-manifest swap, so
    /// a lock-free reader can use it as a cheap seqlock token without re-hashing the
    /// body (see `CoreEngine::committed_root_tokens`).
    pub body_sha256: String,
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

    /// Highest LSN durably folded into this generation: the max of the writer's
    /// own generation-as-LSN and any appended-record LSNs it checkpointed. A reader
    /// uses it as the base watermark for read-your-writes. Absent on manifests
    /// written before this field (an ignored optional body key, not a schema bump),
    /// where the generation is the best available watermark.
    pub fn applied_lsn(&self) -> u64 {
        self.body
            .as_object()
            .and_then(|object| object.get("applied_lsn"))
            .and_then(Value::as_u64)
            .unwrap_or_else(|| self.generation())
    }

    /// The index's native vector dimension, when this manifest records it. Fixed at
    /// index creation, so [`load_store_metadata`](crate::storage::load_store_metadata)
    /// reads it here to skip loading the state base. Absent on manifests written
    /// before this field (an ignored optional body key, not a schema bump); the
    /// caller then falls back to the state base, which always carries it.
    pub fn native_dim(&self) -> Option<u64> {
        self.body
            .as_object()
            .and_then(|object| object.get("native_dim"))
            .and_then(Value::as_u64)
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

pub fn base_tvann_path(persistence_dir: &Path, index_key: &str, epoch: u64) -> PathBuf {
    generation_dir(persistence_dir, index_key).join(format!("g{epoch}.tvann"))
}

pub fn read_commit_manifest(path: &Path) -> CoreResult<Option<CommitManifest>> {
    if !path.is_file() {
        return Ok(None);
    }
    let document = read_json_object(path, "commit manifest")?;
    validate_commit_manifest_document(&document).map(Some)
}

/// Parses and validates pointer-document bytes exactly as
/// [`read_commit_manifest`] validates the on-disk file (schema version and
/// body checksum), without touching the filesystem. For consumers that hold a
/// pointer document from somewhere other than disk (the transfer plane's
/// object-store pointer mirror), so validation has one implementation.
pub fn parse_commit_manifest(bytes: &[u8]) -> CoreResult<CommitManifest> {
    let document: Value = serde_json::from_slice(bytes)
        .map_err(|error| corrupt(format!("commit manifest is not valid JSON: {error}")))?;
    let document = document
        .as_object()
        .ok_or_else(|| corrupt("commit manifest is not a JSON object"))?;
    validate_commit_manifest_document(document)
}

fn validate_commit_manifest_document(document: &Map<String, Value>) -> CoreResult<CommitManifest> {
    if get_i64(document, "schema_version", -1) != COMMIT_MANIFEST_SCHEMA_VERSION {
        return Err(corrupt("unsupported commit manifest schema version"));
    }
    let body = document
        .get("body")
        .ok_or_else(|| corrupt("commit manifest is missing its body or checksum"))?;
    if !body.is_object() || get_str(document, "body_sha256").is_empty() {
        return Err(corrupt("commit manifest is missing its body or checksum"));
    }
    let stored_sha = get_str(document, "body_sha256").to_string();
    if body_sha256(body)? != stored_sha {
        return Err(corrupt("commit manifest failed body checksum"));
    }
    Ok(CommitManifest {
        body: body.clone(),
        body_sha256: stored_sha,
    })
}

#[derive(Debug, Clone)]
pub struct CommitBodyInput<'a> {
    pub index_key: &'a str,
    pub generation: u64,
    /// Highest LSN durably folded into this generation (see
    /// [`CommitManifest::applied_lsn`]); always `>= generation`.
    pub applied_lsn: u64,
    pub base_epoch: u64,
    /// The index's native vector dimension, recorded so a metadata read need not load
    /// the state base (see [`CommitManifest::native_dim`]). `None` omits the key.
    pub native_dim: Option<u64>,
    pub document_count: usize,
    pub chunk_count: usize,
    pub json_manifest: Option<Value>,
    pub tvim_manifest: Option<Value>,
    pub tvtext_manifest: Option<Value>,
    pub tvlex_manifest: Option<Value>,
    pub tvmv_manifest: Option<Value>,
    pub tvann_manifest: Option<Value>,
    /// Original-precision vector sidecar manifest. Unlike `tvann`, this key is
    /// omitted when disabled so existing rescore-off bodies remain byte-identical.
    pub tvvf_manifest: Option<Value>,
}

pub fn build_commit_body(input: CommitBodyInput<'_>) -> Value {
    let mut body = serde_json::json!({
        "index_key": input.index_key,
        "generation": input.generation,
        "applied_lsn": input.applied_lsn.max(input.generation),
        "base_epoch": input.base_epoch,
        "document_count": input.document_count,
        "chunk_count": input.chunk_count,
        "json": input.json_manifest,
        "tvim": input.tvim_manifest,
        "tvtext": input.tvtext_manifest,
        "tvlex": input.tvlex_manifest,
        "tvmv": input.tvmv_manifest,
        "tvann": input.tvann_manifest,
    });
    // Optional key: only recorded when the dimension is known, so a manifest carried
    // forward from a pre-`native_dim` commit stays byte-for-byte as it was.
    if let Some(native_dim) = input.native_dim {
        body["native_dim"] = serde_json::json!(native_dim);
    }
    if let Some(tvvf_manifest) = input.tvvf_manifest {
        body["tvvf"] = tvvf_manifest;
    }
    body
}

pub fn write_commit_manifest(path: &Path, body: &Value, fsync: bool) -> CoreResult<usize> {
    let document_json = render_commit_manifest(body)?;
    write_text_atomic(path, &document_json, fsync)
}

/// Renders the exact pointer-document text a `<key>.commit.json` carrying
/// `body` holds on disk (the canonical body JSON, its checksum, and the
/// schema envelope) without touching the filesystem. [`write_commit_manifest`]
/// persists exactly this text, so the two can never drift.
pub fn render_commit_manifest(body: &Value) -> CoreResult<String> {
    if !body.is_object() {
        return Err(corrupt("commit manifest body must be a JSON object"));
    }
    let body_json = py_canonical_json(body)?;
    let body_sha = body_sha256(body)?;
    Ok(format!(
        "{{\"body\":{body_json},\"body_sha256\":\"{body_sha}\",\"schema_version\":{COMMIT_MANIFEST_SCHEMA_VERSION}}}"
    ))
}

/// The engine-canonical `body_sha256` for `body`, the digest a
/// `<key>.commit.json` pointer carrying exactly this body records. Exported
/// for the transfer plane, whose identity and lineage decisions key on it;
/// the alternative is a scratch-file round trip through
/// [`write_commit_manifest`]/[`read_commit_manifest`] per digest.
pub fn commit_body_sha256(body: &Value) -> CoreResult<String> {
    if !body.is_object() {
        return Err(corrupt("commit manifest body must be a JSON object"));
    }
    body_sha256(body)
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
