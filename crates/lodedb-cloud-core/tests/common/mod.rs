//! Shared fixtures for the artifact-store tests.
//!
//! Committed generations are built with `lodedb-core`'s own commit-writer, so the
//! fixtures match the exact on-disk format the engine produces.

#![allow(dead_code)]

use lodedb_core::storage::commit_manifest::{
    build_commit_body, commit_manifest_path, write_commit_manifest, CommitBodyInput,
};
use lodedb_cloud_core::{ArtifactRef, GenerationInventory};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

/// Lowercase-hex SHA-256, matching what the engine records per artifact.
pub fn sha_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Writes a store's base file (and any delta segments) under `<dir>/<key>.gen/`,
/// returning the `{base, deltas}` sub-manifest that names them — the exact shape
/// the engine records and the inventory reads.
pub fn store_sub(
    dir: &Path,
    key: &str,
    base_name: &str,
    base_bytes: &[u8],
    delta_dir_suffix: &str,
    deltas: &[(&str, &[u8])],
) -> Value {
    let gen_dir = dir.join(format!("{key}.gen"));
    fs::create_dir_all(&gen_dir).unwrap();
    fs::write(gen_dir.join(base_name), base_bytes).unwrap();
    let mut delta_entries = Vec::new();
    if !deltas.is_empty() {
        let delta_dir = gen_dir.join(format!("{base_name}{delta_dir_suffix}"));
        fs::create_dir_all(&delta_dir).unwrap();
        for (seq, (name, bytes)) in deltas.iter().enumerate() {
            fs::write(delta_dir.join(name), bytes).unwrap();
            delta_entries.push(json!({
                "file_name": name,
                "sha256": sha_hex(bytes),
                "file_bytes": bytes.len(),
                "seq": seq,
            }));
        }
    }
    json!({
        "base": {
            "file_name": base_name,
            "sha256": sha_hex(base_bytes),
            "file_bytes": base_bytes.len(),
        },
        "deltas": delta_entries,
    })
}

/// Builds a committed root body (via the engine's own `build_commit_body`) and
/// writes the `<key>.commit.json` pointer.
#[allow(clippy::too_many_arguments)]
pub fn write_commit(
    dir: &Path,
    key: &str,
    generation: u64,
    base_epoch: u64,
    json_manifest: Option<Value>,
    tvim_manifest: Option<Value>,
    tvtext_manifest: Option<Value>,
    tvlex_manifest: Option<Value>,
    tvmv_manifest: Option<Value>,
) -> Value {
    let body = build_commit_body(CommitBodyInput {
        index_key: key,
        generation,
        applied_lsn: generation,
        base_epoch,
        native_dim: None,
        document_count: 0,
        chunk_count: 0,
        json_manifest,
        tvim_manifest,
        tvtext_manifest,
        tvlex_manifest,
        tvmv_manifest,
        tvann_manifest: None,
        tvvf_manifest: None,
    });
    write_commit_manifest(&commit_manifest_path(dir, key), &body, false).unwrap();
    body
}

/// Convenience: commit a generation whose only store is `json`.
pub fn write_json_commit(
    dir: &Path,
    key: &str,
    generation: u64,
    base_epoch: u64,
    json_manifest: Value,
) -> Value {
    write_commit(
        dir,
        key,
        generation,
        base_epoch,
        Some(json_manifest),
        None,
        None,
        None,
        None,
    )
}

/// Builds a commit body without touching the filesystem — for CAS tests that only
/// need a well-formed pointer payload.
pub fn commit_body(key: &str, generation: u64, base_epoch: u64, json_manifest: Value) -> Value {
    build_commit_body(CommitBodyInput {
        index_key: key,
        generation,
        applied_lsn: generation,
        base_epoch,
        native_dim: None,
        document_count: 0,
        chunk_count: 0,
        json_manifest: Some(json_manifest),
        tvim_manifest: None,
        tvtext_manifest: None,
        tvlex_manifest: None,
        tvmv_manifest: None,
        tvann_manifest: None,
        tvvf_manifest: None,
    })
}

/// A minimal but valid engine `state` object for a ready, empty index.
///
/// Enough for `write_generation_commit` to produce a generation that
/// `load_store` reads back. `key` is used for both `index_key` and
/// `client_id_hash`, matching what the engine records.
pub fn engine_state(key: &str) -> Value {
    json!({
        "cache_reuse_count": 0,
        "chunks": [],
        "client_id_hash": key,
        "columnar_generation": 1,
        "created_at": "2026-06-26T00:00:00+00:00",
        "delete_count": 0,
        "deleted_chunk_count": 0,
        "document_chunk_ids": {},
        "document_hashes": {},
        "document_metadata": {},
        "embedded_chunk_count": 0,
        "fallback_count": 0,
        "fallback_reasons": {},
        "index_id": "default",
        "index_key": key,
        "metadata": {},
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "name": "lodedb-local",
        "native_dim": 384,
        "provider": "local_open",
        "query_count": 0,
        "route_profile": "minilm-turbovec",
        "schema_version": 1,
        "status": "ready",
        "storage_profile": "turbovec_direct",
        "task": "direct-turbovec",
        "turbovec_bit_width": 4,
        "updated_at": "2026-06-26T00:00:00+00:00"
    })
}

/// Commits one engine-written generation into `dir` through the engine's own
/// `write_generation_commit`, returning the committed body.
///
/// `salt` lands in the state's metadata so two commits with different salts
/// have different content (different snapshot/logical ids); `raw_text` makes
/// the generation payload-bearing (a non-null `tvtext` sub-manifest), which is
/// what distinguishes a full push from a redacted one.
pub fn commit_engine_generation(
    dir: &Path,
    key: &str,
    generation: u64,
    base_epoch: u64,
    salt: &str,
    raw_text: Option<&[(&str, &str)]>,
) -> Value {
    commit_engine_generation_with_lexical(dir, key, generation, base_epoch, salt, raw_text, false)
}

/// [`commit_engine_generation`], optionally also writing a lexical store
/// (`tvlex`) — for tests that need both payload-bearing stores.
pub fn commit_engine_generation_with_lexical(
    dir: &Path,
    key: &str,
    generation: u64,
    base_epoch: u64,
    salt: &str,
    raw_text: Option<&[(&str, &str)]>,
    lexical: bool,
) -> Value {
    use lodedb_core::storage::lexical_store::TokenLists;
    use lodedb_core::storage::{
        write_generation_commit, GenerationCommitInput, GenerationWriteOptions,
    };
    let mut state = engine_state(key);
    state["metadata"] = json!({ "salt": salt });
    let text: std::collections::BTreeMap<String, String> = raw_text
        .unwrap_or_default()
        .iter()
        .map(|(id, body)| (id.to_string(), body.to_string()))
        .collect();
    let tokens: std::collections::BTreeMap<String, TokenLists> = if lexical {
        raw_text
            .unwrap_or_default()
            .iter()
            .map(|(id, body)| {
                let token_lists = vec![body
                    .split_whitespace()
                    .map(str::to_string)
                    .collect::<Vec<_>>()];
                (id.to_string(), token_lists)
            })
            .collect()
    } else {
        Default::default()
    };
    write_generation_commit(
        dir,
        GenerationCommitInput {
            index_key: key,
            generation,
            applied_lsn: 0,
            base_epoch,
            state: &state,
            tvim: None,
            raw_text: raw_text.map(|_| &text),
            lexical_tokens: lexical.then_some(&tokens),
            multivec: None,
            ann: None,
            tvvf_manifest: None,
            compress_text: false,
        },
        GenerationWriteOptions::default(),
    )
    .unwrap()
}

/// A hand-built [`ArtifactRef`] for the pure `diff_inventories` tests.
pub fn artifact(name: &str, sha256: &str, is_base: bool) -> ArtifactRef {
    ArtifactRef {
        name: name.into(),
        sha256: sha256.into(),
        size_bytes: 8,
        kind: "json".into(),
        epoch: 0,
        is_base,
    }
}

/// A minimal [`GenerationInventory`] wrapping the given artifacts.
pub fn inventory(artifacts: Vec<ArtifactRef>) -> GenerationInventory {
    GenerationInventory {
        index_key: "idx".into(),
        generation: 1,
        base_epoch: 0,
        document_count: 0,
        chunk_count: 0,
        root_body: json!({}),
        artifacts,
    }
}
