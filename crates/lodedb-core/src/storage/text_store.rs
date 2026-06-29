use crate::storage::util::{
    body_sha256, corrupt, get_i64, get_str, read_json, read_maybe_zstd_json, sha256_file_hex,
    value_object, verify_file_sha256, write_pretty_json_atomic, write_py_json_zstd, CoreResult,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const DOCUMENT_TEXT_DELTA_DIR_SUFFIX: &str = ".tvtext-delta";
pub const DOCUMENT_TEXT_MANIFEST_NAME: &str = "manifest.json";
pub const DOCUMENT_TEXT_SCHEMA_VERSION: i64 = 2;

pub fn manifest_path(base_path: &Path) -> PathBuf {
    base_path
        .with_file_name(format!(
            "{}{}",
            base_path.file_name().unwrap().to_string_lossy(),
            DOCUMENT_TEXT_DELTA_DIR_SUFFIX
        ))
        .join(DOCUMENT_TEXT_MANIFEST_NAME)
}

pub fn load(base_path: &Path, manifest: Option<&Value>) -> CoreResult<BTreeMap<String, String>> {
    if !base_path.is_file() {
        return Ok(BTreeMap::new());
    }
    if let Some(manifest) = manifest {
        let manifest = value_object(manifest, "document text manifest")?;
        if get_i64(manifest, "schema_version", -1) != DOCUMENT_TEXT_SCHEMA_VERSION {
            return Err(corrupt("unsupported document text manifest schema version"));
        }
        if let Some(base) = manifest.get("base").and_then(Value::as_object) {
            verify_file_sha256(base_path, get_str(base, "sha256"), "document text base")?;
        }
    }
    let mut documents = read_wrapped_document_map(base_path, DOCUMENT_TEXT_SCHEMA_VERSION)?;
    if let Some(manifest) = manifest {
        let manifest = value_object(manifest, "document text manifest")?;
        let delta_dir = base_path.with_file_name(format!(
            "{}{}",
            base_path.file_name().unwrap().to_string_lossy(),
            DOCUMENT_TEXT_DELTA_DIR_SUFFIX
        ));
        let mut previous_seq = -1_i64;
        for entry in manifest
            .get("deltas")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let entry = value_object(entry, "document text delta manifest entry")?;
            let sequence = get_i64(entry, "seq", -1);
            if sequence <= previous_seq {
                return Err(corrupt("document text manifest has out-of-order segments"));
            }
            previous_seq = sequence;
            let file_name = get_str(entry, "file_name");
            let path = delta_dir.join(file_name);
            if file_name.is_empty() || !path.is_file() {
                return Err(corrupt(format!(
                    "document text segment is missing: {file_name}"
                )));
            }
            if sha256_file_hex(&path)? != get_str(entry, "sha256") {
                return Err(corrupt(format!(
                    "document text segment failed checksum: {file_name}"
                )));
            }
            let body = read_segment_body(&path, DOCUMENT_TEXT_SCHEMA_VERSION, "document text")?;
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
                for (document_id, text) in upserted {
                    documents.insert(document_id.clone(), text.as_str().unwrap_or("").to_string());
                }
            }
        }
    }
    Ok(documents)
}

pub fn read_legacy_text_sidecar(base_path: &Path) -> CoreResult<BTreeMap<String, String>> {
    if !base_path.is_file() {
        return Ok(BTreeMap::new());
    }
    read_wrapped_document_map(base_path, 1)
}

pub fn record_base(
    base_path: &Path,
    documents: &BTreeMap<String, String>,
    fsync: bool,
) -> CoreResult<Value> {
    let body = serde_json::json!({
        "schema_version": DOCUMENT_TEXT_SCHEMA_VERSION,
        "documents": documents,
    });
    let payload = serde_json::json!({
        "schema_version": DOCUMENT_TEXT_SCHEMA_VERSION,
        "body_sha256": body_sha256(&body)?,
        "body": body,
    });
    write_py_json_zstd(base_path, &payload, fsync)?;
    let manifest_path = manifest_path(base_path);
    let previous = if manifest_path.is_file() {
        Some(read_json(&manifest_path, "document text manifest")?)
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
                "document text delta directory could not be created: {error}"
            ))
        })?;
    }
    let manifest = serde_json::json!({
        "schema_version": DOCUMENT_TEXT_SCHEMA_VERSION,
        "base": {
            "file_name": base_path.file_name().unwrap_or_default().to_string_lossy(),
            "sha256": sha256_file_hex(base_path)?,
            "file_bytes": base_path.metadata().map_err(|error| corrupt(format!("document text base metadata failed: {error}")))?.len(),
            "document_count": documents.len(),
        },
        "deltas": [],
        "next_seq": next_seq,
    });
    write_pretty_json_atomic(&manifest_path, &manifest, fsync)?;
    Ok(manifest)
}

pub fn append_delta(
    base_path: &Path,
    upserted: &BTreeMap<String, String>,
    deleted: &[String],
    document_count_after: usize,
    fsync: bool,
) -> CoreResult<Value> {
    let manifest_path = manifest_path(base_path);
    let mut manifest = read_json(&manifest_path, "document text manifest")?;
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| corrupt("document text manifest must be a JSON object"))?;
    let sequence = manifest_object
        .get("next_seq")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let body = serde_json::json!({
        "schema_version": DOCUMENT_TEXT_SCHEMA_VERSION,
        "upserted": upserted,
        "deleted": deleted,
    });
    let segment = serde_json::json!({
        "schema_version": DOCUMENT_TEXT_SCHEMA_VERSION,
        "seq": sequence,
        "document_count_after": document_count_after,
        "body_sha256": body_sha256(&body)?,
        "body": body,
    });
    let segment_name = format!("text-{sequence:08}.txd");
    let delta_dir = base_path.with_file_name(format!(
        "{}{}",
        base_path.file_name().unwrap().to_string_lossy(),
        DOCUMENT_TEXT_DELTA_DIR_SUFFIX
    ));
    let segment_path = delta_dir.join(&segment_name);
    write_py_json_zstd(&segment_path, &segment, fsync)?;
    let deltas = manifest_object
        .entry("deltas")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| corrupt("document text manifest deltas must be a list"))?;
    deltas.push(serde_json::json!({
        "file_name": segment_name,
        "sha256": sha256_file_hex(&segment_path)?,
        "file_bytes": segment_path.metadata().map_err(|error| corrupt(format!("document text segment metadata failed: {error}")))?.len(),
        "seq": sequence,
        "upserted": upserted.len(),
        "deleted": deleted.len(),
    }));
    manifest_object.insert("next_seq".to_string(), Value::from(sequence + 1));
    write_pretty_json_atomic(&manifest_path, &manifest, fsync)?;
    Ok(manifest)
}

fn read_wrapped_document_map(
    path: &Path,
    schema_version: i64,
) -> CoreResult<BTreeMap<String, String>> {
    let payload = read_maybe_zstd_json(path, "document text base")?;
    let payload = value_object(&payload, "document text base")?;
    if get_i64(payload, "schema_version", -1) != schema_version {
        return Err(corrupt("unsupported document text base schema version"));
    }
    let body = payload
        .get("body")
        .ok_or_else(|| corrupt("document text base body is missing"))?;
    let body_object = value_object(body, "document text base body")?;
    if body_sha256(body)? != get_str(payload, "body_sha256") {
        return Err(corrupt("document text base failed checksum"));
    }
    let documents = body_object
        .get("documents")
        .and_then(Value::as_object)
        .ok_or_else(|| corrupt("document text base documents must be an object"))?;
    Ok(documents
        .iter()
        .map(|(key, value)| (key.clone(), value.as_str().unwrap_or("").to_string()))
        .collect())
}

fn read_segment_body(path: &Path, schema_version: i64, context: &str) -> CoreResult<Value> {
    let segment = read_maybe_zstd_json(path, &format!("{context} segment"))?;
    let segment = value_object(&segment, &format!("{context} segment"))?;
    if get_i64(segment, "schema_version", -1) != schema_version {
        return Err(corrupt(format!(
            "unsupported {context} segment schema version"
        )));
    }
    let body = segment
        .get("body")
        .ok_or_else(|| corrupt(format!("{context} segment body is missing")))?;
    if !body.is_object() {
        return Err(corrupt(format!("{context} segment body is missing")));
    }
    if body_sha256(body)? != get_str(segment, "body_sha256") {
        return Err(corrupt(format!("{context} segment failed checksum")));
    }
    Ok(body.clone())
}
