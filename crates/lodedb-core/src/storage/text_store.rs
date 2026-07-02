use crate::storage::util::{
    body_sha256, checksummed_body_payload, corrupt, get_i64, get_str, read_json, read_json_object,
    read_maybe_zstd_json, sha256_file_hex, sidecar_base_block, validate_sidecar_manifest,
    value_object, write_pretty_json_atomic, write_py_json, write_py_json_zstd, CoreResult,
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
        validate_sidecar_manifest(
            base_path,
            manifest,
            DOCUMENT_TEXT_SCHEMA_VERSION,
            "document text",
        )?;
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
    compress: bool,
) -> CoreResult<Value> {
    let body = serde_json::json!({
        "schema_version": DOCUMENT_TEXT_SCHEMA_VERSION,
        "documents": documents,
    });
    let payload = checksummed_body_payload(DOCUMENT_TEXT_SCHEMA_VERSION, body)?;
    write_base_payload(base_path, &payload, fsync, compress)?;
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
        "compress": compress,
        "base": sidecar_base_block(
            base_path,
            "document text",
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
    upserted: &BTreeMap<String, String>,
    deleted: &[String],
    document_count_after: usize,
    fsync: bool,
    compress: bool,
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
    write_base_payload(&segment_path, &segment, fsync, compress)?;
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

/// Returns the persisted ``compress`` flag from the text-store manifest, or
/// ``None`` when there is no manifest or it has no ``compress`` field (a store
/// written before the flag existed). The engine seeds its effective value from
/// the open options and lets this persisted value win on reopen, so a store
/// keeps the compression it was created with.
pub fn persisted_compress(base_path: &Path) -> Option<bool> {
    let manifest_path = manifest_path(base_path);
    if !manifest_path.is_file() {
        return None;
    }
    read_json_object(&manifest_path, "document text manifest")
        .ok()?
        .get("compress")
        .and_then(Value::as_bool)
}

/// Writes a text base/segment payload, zstd-compressed when ``compress`` is set
/// (the default) or as plain canonical JSON otherwise. The read path detects the
/// zstd frame magic, so either form loads back the same way.
fn write_base_payload(
    path: &Path,
    payload: &Value,
    fsync: bool,
    compress: bool,
) -> CoreResult<usize> {
    if compress {
        write_py_json_zstd(path, payload, fsync)
    } else {
        write_py_json(path, payload, fsync)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    /// The first bytes of a zstd frame (little-endian magic ``0xFD2FB528``).
    const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

    fn unique_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "lodedb_text_{name}_{}_{}_{nanos}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn documents() -> BTreeMap<String, String> {
        BTreeMap::from([
            ("doc-1".to_string(), "first body".to_string()),
            ("doc-2".to_string(), "second body".to_string()),
        ])
    }

    #[test]
    fn record_base_uncompressed_round_trips_and_persists_flag() {
        let base_path = unique_dir("uncompressed").join("g00000001.tvtext");
        let documents = documents();

        let manifest = record_base(&base_path, &documents, false, false).expect("record base");

        // The on-disk base is plain canonical JSON (no zstd frame magic).
        let raw = fs::read(&base_path).expect("read base");
        assert!(
            !raw.starts_with(&ZSTD_MAGIC),
            "uncompressed base must not start with the zstd magic"
        );

        // The persisted flag is readable and reflects the uncompressed write.
        assert_eq!(persisted_compress(&base_path), Some(false));
        assert_eq!(
            manifest.get("compress").and_then(Value::as_bool),
            Some(false)
        );

        // The uncompressed base loads back to the same documents.
        let loaded = load(&base_path, Some(&manifest)).expect("load base");
        assert_eq!(loaded, documents);
    }

    #[test]
    fn record_base_compressed_writes_zstd_and_persists_flag() {
        let base_path = unique_dir("compressed").join("g00000001.tvtext");
        let documents = documents();

        let manifest = record_base(&base_path, &documents, false, true).expect("record base");

        let raw = fs::read(&base_path).expect("read base");
        assert!(
            raw.starts_with(&ZSTD_MAGIC),
            "compressed base must start with the zstd magic"
        );
        assert_eq!(persisted_compress(&base_path), Some(true));
        let loaded = load(&base_path, Some(&manifest)).expect("load base");
        assert_eq!(loaded, documents);
    }

    #[test]
    fn persisted_compress_is_none_without_a_manifest() {
        let base_path = unique_dir("no_manifest").join("g00000001.tvtext");
        assert_eq!(persisted_compress(&base_path), None);
    }
}
