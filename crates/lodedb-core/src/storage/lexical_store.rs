use crate::storage::util::{
    body_sha256, checksummed_body_payload, corrupt, get_i64, get_str, read_checksummed_body,
    read_json, sha256_file_hex, sidecar_base_block, validate_sidecar_manifest, value_object,
    write_pretty_json_atomic, write_py_json, CoreResult,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const LEXICAL_INDEX_DELTA_DIR_SUFFIX: &str = ".tvlex-delta";
pub const LEXICAL_INDEX_MANIFEST_NAME: &str = "manifest.json";
pub const LEXICAL_INDEX_SCHEMA_VERSION: i64 = 1;

pub type TokenLists = Vec<Vec<String>>;

pub fn manifest_path(base_path: &Path) -> PathBuf {
    base_path
        .with_file_name(format!(
            "{}{}",
            base_path.file_name().unwrap().to_string_lossy(),
            LEXICAL_INDEX_DELTA_DIR_SUFFIX
        ))
        .join(LEXICAL_INDEX_MANIFEST_NAME)
}

pub fn load(
    base_path: &Path,
    manifest: Option<&Value>,
) -> CoreResult<BTreeMap<String, TokenLists>> {
    if !base_path.is_file() {
        return Ok(BTreeMap::new());
    }
    if let Some(manifest) = manifest {
        validate_sidecar_manifest(
            base_path,
            manifest,
            LEXICAL_INDEX_SCHEMA_VERSION,
            "lexical index",
        )?;
    }
    let mut documents = read_base(base_path)?;
    if let Some(manifest) = manifest {
        let manifest = value_object(manifest, "lexical index manifest")?;
        let delta_dir = base_path.with_file_name(format!(
            "{}{}",
            base_path.file_name().unwrap().to_string_lossy(),
            LEXICAL_INDEX_DELTA_DIR_SUFFIX
        ));
        let mut previous_seq = -1_i64;
        for entry in manifest
            .get("deltas")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let entry = value_object(entry, "lexical index delta manifest entry")?;
            let sequence = get_i64(entry, "seq", -1);
            if sequence <= previous_seq {
                return Err(corrupt("lexical index manifest has out-of-order segments"));
            }
            previous_seq = sequence;
            let file_name = get_str(entry, "file_name");
            let path = delta_dir.join(file_name);
            if file_name.is_empty() || !path.is_file() {
                return Err(corrupt(format!(
                    "lexical index segment is missing: {file_name}"
                )));
            }
            if sha256_file_hex(&path)? != get_str(entry, "sha256") {
                return Err(corrupt(format!(
                    "lexical index segment failed checksum: {file_name}"
                )));
            }
            let body = read_segment_body(&path)?;
            for deleted in body
                .get("deleted")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                documents.remove(deleted);
            }
            if let Some(upserted) = body.get("upserted").and_then(Value::as_object) {
                for (document_id, tokens) in upserted {
                    documents.insert(document_id.clone(), normalize_token_lists(tokens)?);
                }
            }
        }
    }
    Ok(documents)
}

fn read_base(path: &Path) -> CoreResult<BTreeMap<String, TokenLists>> {
    let body = read_checksummed_body(path, LEXICAL_INDEX_SCHEMA_VERSION, "lexical index")?;
    let body_object = value_object(&body, "lexical index base body")?;
    let documents = body_object
        .get("documents")
        .and_then(Value::as_object)
        .ok_or_else(|| corrupt("lexical index base documents must be an object"))?;
    documents
        .iter()
        .map(|(key, value)| Ok((key.clone(), normalize_token_lists(value)?)))
        .collect()
}

pub fn record_base(
    base_path: &Path,
    documents: &BTreeMap<String, TokenLists>,
    fsync: bool,
) -> CoreResult<Value> {
    let body = serde_json::json!({
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "documents": documents,
    });
    let payload = checksummed_body_payload(LEXICAL_INDEX_SCHEMA_VERSION, body)?;
    write_py_json(base_path, &payload, fsync)?;
    let manifest_path = manifest_path(base_path);
    let previous = if manifest_path.is_file() {
        Some(read_json(&manifest_path, "lexical index manifest")?)
    } else {
        None
    };
    let next_seq = previous
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|object| object.get("next_seq"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
        + 1;
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            corrupt(format!(
                "lexical index delta directory could not be created: {error}"
            ))
        })?;
    }
    let manifest = serde_json::json!({
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "base": sidecar_base_block(
            base_path,
            "lexical index",
            [("document_count", Value::from(documents.len()))],
        )?,
        "deltas": [],
        "next_seq": next_seq,
    });
    write_pretty_json_atomic(&manifest_path, &manifest, fsync)?;
    Ok(manifest)
}

pub fn append_delta(
    base_path: &Path,
    upserted: &BTreeMap<String, TokenLists>,
    deleted: &[String],
    document_count_after: usize,
    fsync: bool,
) -> CoreResult<Value> {
    let manifest_path = manifest_path(base_path);
    let mut manifest = read_json(&manifest_path, "lexical index manifest")?;
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| corrupt("lexical index manifest must be a JSON object"))?;
    let sequence = manifest_object
        .get("next_seq")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let body = serde_json::json!({
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "upserted": upserted,
        "deleted": deleted,
    });
    let segment = serde_json::json!({
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "seq": sequence,
        "document_count_after": document_count_after,
        "body_sha256": body_sha256(&body)?,
        "body": body,
    });
    let segment_name = format!("lexical-{sequence:08}.lxd");
    let delta_dir = base_path.with_file_name(format!(
        "{}{}",
        base_path.file_name().unwrap().to_string_lossy(),
        LEXICAL_INDEX_DELTA_DIR_SUFFIX
    ));
    let segment_path = delta_dir.join(&segment_name);
    write_py_json(&segment_path, &segment, fsync)?;
    let deltas = manifest_object
        .entry("deltas")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| corrupt("lexical index manifest deltas must be a list"))?;
    deltas.push(serde_json::json!({
        "file_name": segment_name,
        "sha256": sha256_file_hex(&segment_path)?,
        "file_bytes": segment_path.metadata().map_err(|error| corrupt(format!("lexical index segment metadata failed: {error}")))?.len(),
        "seq": sequence,
        "upserted": upserted.len(),
        "deleted": deleted.len(),
    }));
    manifest_object.insert("next_seq".to_string(), Value::from(sequence + 1));
    write_pretty_json_atomic(&manifest_path, &manifest, fsync)?;
    Ok(manifest)
}

fn read_segment_body(path: &Path) -> CoreResult<Value> {
    let segment = read_json(path, "lexical index segment")?;
    let segment = value_object(&segment, "lexical index segment")?;
    if get_i64(segment, "schema_version", -1) != LEXICAL_INDEX_SCHEMA_VERSION {
        return Err(corrupt("unsupported lexical index segment schema version"));
    }
    let body = segment
        .get("body")
        .ok_or_else(|| corrupt("lexical index segment body is missing"))?;
    if !body.is_object() {
        return Err(corrupt("lexical index segment body is missing"));
    }
    if body_sha256(body)? != get_str(segment, "body_sha256") {
        return Err(corrupt("lexical index segment failed checksum"));
    }
    Ok(body.clone())
}

fn normalize_token_lists(value: &Value) -> CoreResult<TokenLists> {
    let chunks = value
        .as_array()
        .ok_or_else(|| corrupt("lexical index token lists must be a list of chunks"))?;
    chunks
        .iter()
        .map(|chunk| {
            let tokens = chunk
                .as_array()
                .ok_or_else(|| corrupt("lexical index chunk must be a list of tokens"))?;
            Ok(tokens
                .iter()
                .map(|token| token.as_str().unwrap_or("").to_string())
                .collect())
        })
        .collect()
}
