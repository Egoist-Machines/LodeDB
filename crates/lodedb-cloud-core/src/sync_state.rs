//! The sync sidecar: `<dir>/<key>.orecloud`, the recorded base.
//!
//! Three-pointer sync needs a durable record of the last state local and
//! remote agreed on. That record lives *next to* the index it describes — a
//! small JSON sidecar beside `<key>.commit.json` — because it is a claim about
//! this particular local copy's history, not about the remote (two clones of
//! one remote each carry their own base).
//!
//! Naming and placement are deliberate. The file deliberately does NOT end in
//! `.json`: the engine's load path globs `*.json` in the persistence directory
//! and treats anything that is not a commit manifest or a known metadata file
//! as a *legacy index snapshot* (verified against lodedb rev `f242d3a`,
//! `engine/core.py _load_persisted_indexes`), so a `<key>.orecloud.json`
//! sidecar would break reopening the database. `<key>.orecloud` is invisible
//! to that glob, and the engine's epoch GC removes files only under
//! `<key>.gen/`, so a sidecar survives any number of commits. Both properties
//! are pinned by tests (the reopen-safety one from Python, where the real
//! engine runs).
//!
//! The sidecar is written atomically (temp file + rename, the same discipline
//! as [`LocalArtifactStore`](crate::LocalArtifactStore)'s pointer writes) and
//! carries a self-checksum over its payload. A torn, tampered, or
//! half-migrated sidecar therefore reads as *absent-but-corrupt* — which the
//! classifier maps to [`Unknown`](crate::sync_plan::SyncClassification::Unknown),
//! requiring an explicit force — rather than as a trusted base.

use crate::digest::sha256_hex;
use crate::error::{ArtifactStoreError, Result};
use crate::paths::resolve_within;
use crate::sync_plan::SnapRef;
use serde_json::{json, Value};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The sidecar file suffix (`<key>.orecloud`). Must never end in `.json` —
/// see the module docs on the engine's legacy-snapshot glob.
pub const SYNC_STATE_SUFFIX: &str = ".orecloud";

/// The sidecar schema this crate writes and accepts.
const SYNC_STATE_SCHEMA_VERSION: u64 = 1;

/// The recorded base: what local and remote last agreed on, and where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncState {
    pub index_key: String,
    /// The remote target string the base was established against.
    pub remote: String,
    /// Identity of the last-synced generation.
    pub base: SnapRef,
    /// Unix seconds of the last sidecar write (informational).
    pub updated_unix: u64,
}

/// The result of reading a sidecar: the state if one was trustworthy, plus a
/// flag distinguishing "no sidecar" from "a sidecar was present but corrupt".
///
/// Both cases classify the same way (no trusted base -> `Unknown` when the
/// ends differ), but a frontend should *warn* on corruption — it usually means
/// a torn write or manual tampering, not a fresh directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarRead {
    pub state: Option<SyncState>,
    pub corrupt: bool,
}

/// The sidecar path for `key` under `dir`: `<dir>/<key>.orecloud`.
pub fn sync_state_path(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}{SYNC_STATE_SUFFIX}"))
}

/// Reads the sidecar for `key` under `dir`.
///
/// An absent file is `{state: None, corrupt: false}`. Any parse failure,
/// schema mismatch, checksum mismatch, or an `index_key` that disagrees with
/// `key` is `{state: None, corrupt: true}` — the base is untrusted, never
/// partially honored. Only genuine I/O failures (permissions etc.) are errors.
pub fn read_sync_state(dir: &Path, key: &str) -> Result<SidecarRead> {
    let path = resolve_within(dir, &sync_state_path(dir, key))?;
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(SidecarRead {
                state: None,
                corrupt: false,
            })
        }
        Err(error) => return Err(ArtifactStoreError::Io(error)),
    };
    Ok(match parse_sidecar(&bytes, key) {
        Some(state) => SidecarRead {
            state: Some(state),
            corrupt: false,
        },
        None => SidecarRead {
            state: None,
            corrupt: true,
        },
    })
}

/// Writes the sidecar for `state.index_key` under `dir`, atomically.
///
/// Temp file + rename in the same directory, so a crash mid-write leaves
/// either the previous sidecar or a stray `.tmp` — never a torn document (and
/// a torn document would fail its self-checksum anyway).
pub fn write_sync_state(dir: &Path, state: &SyncState) -> Result<()> {
    let path = resolve_within(dir, &sync_state_path(dir, &state.index_key))?;
    let document = serde_json::to_vec_pretty(&document_json(state)).map_err(|error| {
        ArtifactStoreError::Integrity(format!("failed to serialize sync sidecar: {error}"))
    })?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    // A uniquely-named temp file (not a fixed `.tmp` sibling), so two
    // concurrent syncs of one key cannot collide mid-write; the rename keeps
    // the publish atomic either way, last writer wins.
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut scratch = tempfile::NamedTempFile::new_in(dir)?;
    scratch.write_all(&document)?;
    scratch
        .persist(&path)
        .map_err(|error| ArtifactStoreError::Io(error.error))?;
    Ok(())
}

/// The checksummed payload — everything except the `sha256` field itself.
///
/// Built with one fixed construction order shared by the write path and the
/// read-side verification, so the serialized bytes (and therefore the digest)
/// are identical regardless of how the document was produced.
fn payload_json(state: &SyncState) -> Value {
    json!({
        "schema_version": SYNC_STATE_SCHEMA_VERSION,
        "index_key": state.index_key,
        "remote": state.remote,
        "base": {
            "snapshot_id": state.base.snapshot_id,
            "logical_id": state.base.logical_id,
            "generation": state.base.generation,
            "text_id": state.base.text_id,
            "lexical_id": state.base.lexical_id,
        },
        "updated_unix": state.updated_unix,
    })
}

/// The full on-disk document: the payload plus its self-checksum.
fn document_json(state: &SyncState) -> Value {
    let payload = payload_json(state);
    let digest = sha256_hex(&serde_json::to_vec(&payload).unwrap_or_default());
    let mut document = payload;
    if let Some(object) = document.as_object_mut() {
        object.insert("sha256".to_string(), Value::String(digest));
    }
    document
}

/// Parses and validates sidecar bytes; `None` means "do not trust this file".
fn parse_sidecar(bytes: &[u8], key: &str) -> Option<SyncState> {
    let document: Value = serde_json::from_slice(bytes).ok()?;
    let object = document.as_object()?;
    if object.get("schema_version")?.as_u64()? != SYNC_STATE_SCHEMA_VERSION {
        return None;
    }
    let base = object.get("base")?.as_object()?;
    let state = SyncState {
        index_key: object.get("index_key")?.as_str()?.to_string(),
        remote: object.get("remote")?.as_str()?.to_string(),
        base: SnapRef {
            snapshot_id: nonempty(base.get("snapshot_id")?.as_str()?)?,
            logical_id: nonempty(base.get("logical_id")?.as_str()?)?,
            generation: base.get("generation")?.as_u64()?,
            text_id: optional_id(base.get("text_id")?)?,
            lexical_id: optional_id(base.get("lexical_id")?)?,
        },
        updated_unix: object.get("updated_unix")?.as_u64()?,
    };
    if state.index_key != key {
        return None;
    }
    // Recompute the digest from the parsed fields via the same construction
    // the writer used; a mismatch means torn/tampered bytes.
    let recorded = object.get("sha256")?.as_str()?;
    let expected = sha256_hex(&serde_json::to_vec(&payload_json(&state)).ok()?);
    if recorded != expected {
        return None;
    }
    Some(state)
}

/// Rejects an empty identity string (a base with no id is no base).
fn nonempty(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

/// Parses a nullable store-identity field: JSON `null` is an absent store
/// (`Some(None)`), a non-empty string is its identity, anything else is
/// corrupt (`None`, propagated by `?`).
#[allow(clippy::option_option)]
fn optional_id(value: &serde_json::Value) -> Option<Option<String>> {
    match value {
        serde_json::Value::Null => Some(None),
        serde_json::Value::String(id) => Some(Some(nonempty(id)?)),
        _ => None,
    }
}
