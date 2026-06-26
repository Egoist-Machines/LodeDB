use crate::storage::util::{
    corrupt, get_i64, get_str, read_json, sha256_bytes_hex, value_object, verify_file_sha256,
    CoreResult,
};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const STATE_JOURNAL_DIR_SUFFIX: &str = ".json-delta";
pub const STATE_JOURNAL_MANIFEST_NAME: &str = "manifest.json";
pub const STATE_JOURNAL_MAGIC: &[u8; 8] = b"EEJSD001";
pub const STATE_JOURNAL_SCHEMA_VERSION: i64 = 1;

pub fn manifest_path(base_path: &Path) -> PathBuf {
    base_path
        .with_file_name(format!(
            "{}{}",
            base_path.file_name().unwrap().to_string_lossy(),
            STATE_JOURNAL_DIR_SUFFIX
        ))
        .join(STATE_JOURNAL_MANIFEST_NAME)
}

pub fn read_manifest_optional(base_path: &Path) -> CoreResult<Option<Value>> {
    let path = manifest_path(base_path);
    if !path.is_file() {
        return Ok(None);
    }
    let manifest = read_json(&path, "state journal manifest")?;
    let object = value_object(&manifest, "state journal manifest")?;
    if get_i64(object, "schema_version", -1) != STATE_JOURNAL_SCHEMA_VERSION {
        return Err(corrupt("unsupported state journal manifest schema version"));
    }
    Ok(Some(manifest))
}

pub fn read_base_payload(base_path: &Path, manifest: Option<&Value>) -> CoreResult<Value> {
    if let Some(manifest) = manifest {
        let manifest = value_object(manifest, "state journal manifest")?;
        if let Some(base) = manifest.get("base").and_then(Value::as_object) {
            verify_file_sha256(
                base_path,
                get_str(base, "sha256"),
                "state journal base snapshot",
            )?;
        }
    }
    read_json(base_path, "state journal base snapshot")
}

pub fn replay_onto_payload(
    payload: &mut Value,
    base_path: &Path,
    manifest: &Value,
) -> CoreResult<()> {
    let manifest_object = value_object(manifest, "state journal manifest")?;
    let payload_object = payload
        .as_object_mut()
        .ok_or_else(|| corrupt("state journal base snapshot must be a JSON object"))?;
    let mut chunks_by_id = BTreeMap::<String, Value>::new();
    if let Some(Value::Array(chunks)) = payload_object.get("chunks") {
        for row in chunks {
            if let Some(chunk_id) = row.get("chunk_id").and_then(Value::as_str) {
                chunks_by_id.insert(chunk_id.to_string(), row.clone());
            }
        }
    }
    let mut document_hashes = object_string_values(payload_object.get("document_hashes"))?;
    let mut document_chunk_ids = object_string_arrays(payload_object.get("document_chunk_ids"))?;
    let mut document_metadata = object_objects(payload_object.get("document_metadata"))?;
    let mut previous_seq = -1_i64;
    let mut last_state_header: Option<Map<String, Value>> = None;
    let delta_dir = base_path.with_file_name(format!(
        "{}{}",
        base_path.file_name().unwrap().to_string_lossy(),
        STATE_JOURNAL_DIR_SUFFIX
    ));
    for entry in manifest_object
        .get("deltas")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let entry = value_object(entry, "state journal delta manifest entry")?;
        let sequence = get_i64(entry, "seq", -1);
        if sequence <= previous_seq {
            return Err(corrupt("state journal manifest has out-of-order segments"));
        }
        previous_seq = sequence;
        let file_name = get_str(entry, "file_name");
        let segment_path = delta_dir.join(file_name);
        if file_name.is_empty() || !segment_path.is_file() {
            return Err(corrupt(format!(
                "state journal segment is missing: {file_name}"
            )));
        }
        verify_file_sha256(
            &segment_path,
            get_str(entry, "sha256"),
            "state journal segment",
        )?;
        let segment = read_journal_segment(&segment_path)?;
        let header = value_object(
            segment.get("header").unwrap(),
            "state journal segment header",
        )?;
        let body = value_object(segment.get("body").unwrap(), "state journal segment body")?;
        for document_id in body
            .get("deleted_document_ids")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if let Some(chunk_ids) = document_chunk_ids.remove(document_id) {
                for chunk_id in chunk_ids {
                    chunks_by_id.remove(&chunk_id);
                }
            }
            document_hashes.remove(document_id);
            document_metadata.remove(document_id);
        }
        for document in body
            .get("upserted_documents")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let document = value_object(document, "state journal upserted document")?;
            let document_id = get_str(document, "document_id").to_string();
            if let Some(chunk_ids) = document_chunk_ids.get(&document_id) {
                for chunk_id in chunk_ids {
                    chunks_by_id.remove(chunk_id);
                }
            }
            for row in document
                .get("chunks")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if let Some(chunk_id) = row.get("chunk_id").and_then(Value::as_str) {
                    chunks_by_id.insert(chunk_id.to_string(), row.clone());
                }
            }
            document_hashes.insert(
                document_id.clone(),
                get_str(document, "document_hash").to_string(),
            );
            document_chunk_ids.insert(
                document_id.clone(),
                document
                    .get("chunk_ids")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect(),
            );
            document_metadata.insert(
                document_id,
                document
                    .get("metadata")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default(),
            );
        }
        let expected_documents = get_i64(header, "document_count_after", -1);
        if expected_documents >= 0 && document_hashes.len() != expected_documents as usize {
            return Err(corrupt(
                "state journal replay rejected: document count mismatch",
            ));
        }
        let expected_chunks = get_i64(header, "chunk_count_after", -1);
        if expected_chunks >= 0 && chunks_by_id.len() != expected_chunks as usize {
            return Err(corrupt(
                "state journal replay rejected: chunk count mismatch",
            ));
        }
        last_state_header = body.get("state_header").and_then(Value::as_object).cloned();
    }
    if let Some(header) = last_state_header {
        for (key, value) in header {
            if !matches!(
                key.as_str(),
                "chunks"
                    | "document_hashes"
                    | "document_chunk_ids"
                    | "document_metadata"
                    | "query_latency_ms"
            ) {
                payload_object.insert(key, value);
            }
        }
    }
    payload_object.insert(
        "chunks".to_string(),
        Value::Array(chunks_by_id.into_values().collect()),
    );
    payload_object.insert(
        "document_hashes".to_string(),
        Value::Object(
            document_hashes
                .into_iter()
                .map(|(key, value)| (key, Value::String(value)))
                .collect(),
        ),
    );
    payload_object.insert(
        "document_chunk_ids".to_string(),
        Value::Object(
            document_chunk_ids
                .into_iter()
                .map(|(key, chunk_ids)| {
                    (
                        key,
                        Value::Array(chunk_ids.into_iter().map(Value::String).collect()),
                    )
                })
                .collect(),
        ),
    );
    payload_object.insert(
        "document_metadata".to_string(),
        Value::Object(
            document_metadata
                .into_iter()
                .map(|(key, metadata)| (key, Value::Object(metadata)))
                .collect(),
        ),
    );
    Ok(())
}

pub fn read_journal_segment(path: &Path) -> CoreResult<Map<String, Value>> {
    let data = fs::read(path)
        .map_err(|error| corrupt(format!("state journal segment could not be read: {error}")))?;
    let prefix = STATE_JOURNAL_MAGIC.len() + 8;
    if data.len() < prefix || &data[..STATE_JOURNAL_MAGIC.len()] != STATE_JOURNAL_MAGIC {
        return Err(corrupt(format!(
            "not a state journal segment: {}",
            path.display()
        )));
    }
    let mut length = [0_u8; 8];
    length.copy_from_slice(&data[STATE_JOURNAL_MAGIC.len()..prefix]);
    let header_len = u64::from_le_bytes(length) as usize;
    let header_stop = prefix + header_len;
    if data.len() < header_stop {
        return Err(corrupt("state journal segment header is truncated"));
    }
    let header: Value = serde_json::from_slice(&data[prefix..header_stop])
        .map_err(|error| corrupt(format!("state journal segment header is corrupt: {error}")))?;
    let header_object = value_object(&header, "state journal segment header")?;
    if get_i64(header_object, "schema_version", -1) != STATE_JOURNAL_SCHEMA_VERSION {
        return Err(corrupt("unsupported state journal segment schema version"));
    }
    let body_bytes = header_object
        .get("body_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| corrupt("state journal segment body length is missing"))?
        as usize;
    let body_stop = header_stop + body_bytes;
    if data.len() < body_stop {
        return Err(corrupt("state journal segment body is truncated"));
    }
    let body_blob = &data[header_stop..body_stop];
    if sha256_bytes_hex(body_blob) != get_str(header_object, "body_sha256") {
        return Err(corrupt("state journal segment body failed checksum"));
    }
    let body: Value = serde_json::from_slice(body_blob)
        .map_err(|error| corrupt(format!("state journal segment body is corrupt: {error}")))?;
    let body_object = value_object(&body, "state journal segment body")?;
    for key in ["upserted_documents", "deleted_document_ids", "state_header"] {
        if !body_object.contains_key(key) {
            return Err(corrupt(format!(
                "state journal segment is missing field: {key}"
            )));
        }
    }
    let mut segment = Map::new();
    segment.insert("header".to_string(), header);
    segment.insert("body".to_string(), body);
    Ok(segment)
}

fn object_string_values(value: Option<&Value>) -> CoreResult<BTreeMap<String, String>> {
    match value {
        Some(Value::Object(object)) => Ok(object
            .iter()
            .map(|(key, value)| (key.clone(), value.as_str().unwrap_or("").to_string()))
            .collect()),
        Some(_) => Err(corrupt("state document hashes must be an object")),
        None => Ok(BTreeMap::new()),
    }
}

fn object_string_arrays(value: Option<&Value>) -> CoreResult<BTreeMap<String, Vec<String>>> {
    match value {
        Some(Value::Object(object)) => Ok(object
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    value
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(Value::as_str)
                        .map(ToString::to_string)
                        .collect(),
                )
            })
            .collect()),
        Some(_) => Err(corrupt("state document chunk ids must be an object")),
        None => Ok(BTreeMap::new()),
    }
}

fn object_objects(value: Option<&Value>) -> CoreResult<BTreeMap<String, Map<String, Value>>> {
    match value {
        Some(Value::Object(object)) => Ok(object
            .iter()
            .map(|(key, value)| (key.clone(), value.as_object().cloned().unwrap_or_default()))
            .collect()),
        Some(_) => Err(corrupt("state document metadata must be an object")),
        None => Ok(BTreeMap::new()),
    }
}
