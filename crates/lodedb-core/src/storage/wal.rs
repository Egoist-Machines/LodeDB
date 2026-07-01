use crate::storage::util::{corrupt, fsync_dir, CoreResult};
use crate::storage::LoadedStore;
use crate::text::chunk::{chunk_id_for_hash, chunk_text};
use crate::text::hash::{normalized_chunk_hash, sha256_text};
use serde_json::Value;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const WAL_SUFFIX: &str = ".wal";
pub const WAL_MAGIC: &[u8; 8] = b"EELWAL01";
pub const WAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct WalRecord {
    pub op: String,
    pub payload: Value,
    /// Log sequence number: the writer's generation counter at the mutation
    /// that produced this record. `None` for pre-LSN WALs read back from older
    /// stores, which carry no watermark and are always replayed in full.
    pub lsn: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalAppend {
    pub record_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalStats {
    pub op_count: usize,
    pub byte_count: usize,
}

pub fn wal_path(persistence_dir: &Path, index_key: &str) -> PathBuf {
    persistence_dir.join(format!("{index_key}{WAL_SUFFIX}"))
}

pub fn append_record(
    path: &Path,
    lsn: u64,
    op: &str,
    payload: &Value,
    fsync: bool,
) -> CoreResult<WalAppend> {
    let body = encode_body(op, payload, lsn)?;
    let mut frame = Vec::with_capacity(4 + body.len() + 4);
    frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
    frame.extend_from_slice(&body);
    frame.extend_from_slice(&crc32(&frame).to_be_bytes());
    let first_write = !path.exists();
    if first_write {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| corrupt(format!("WAL directory could not be created: {error}")))?;
        }
    }
    let mut handle = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| corrupt(format!("WAL could not be opened for append: {error}")))?;
    if first_write {
        handle
            .write_all(WAL_MAGIC)
            .and_then(|_| handle.write_all(&WAL_SCHEMA_VERSION.to_be_bytes()))
            .map_err(|error| corrupt(format!("WAL header could not be written: {error}")))?;
    }
    handle
        .write_all(&frame)
        .and_then(|_| handle.flush())
        .map_err(|error| corrupt(format!("WAL record could not be written: {error}")))?;
    if fsync {
        handle
            .sync_all()
            .map_err(|error| corrupt(format!("WAL could not be fsynced: {error}")))?;
    }
    drop(handle);
    if first_write && fsync {
        fsync_dir(path.parent().unwrap_or_else(|| Path::new(".")))?;
    }
    // Intentionally do NOT scan the WAL here. `scan_stats` reads and re-encodes
    // every prior record, which made each append O(WAL size) and the whole write
    // stream O(n^2) (a single multi-thousand-row seed record was re-parsed on
    // every subsequent add). Callers that need op_count/byte_count call
    // `scan_stats` explicitly, off the per-append hot path.
    Ok(WalAppend {
        record_bytes: frame.len(),
    })
}

pub fn truncate(path: &Path, fsync: bool) -> CoreResult<()> {
    if path.exists() {
        fs::remove_file(path)
            .map_err(|error| corrupt(format!("WAL could not be truncated: {error}")))?;
        if fsync {
            fsync_dir(path.parent().unwrap_or_else(|| Path::new(".")))?;
        }
    }
    Ok(())
}

pub fn scan_stats(path: &Path) -> CoreResult<WalStats> {
    let records = read_records(path)?;
    let mut byte_count = 0;
    for record in &records {
        let body = encode_body(&record.op, &record.payload, record.lsn.unwrap_or(0))?;
        byte_count += 4 + body.len() + 4;
    }
    Ok(WalStats {
        op_count: records.len(),
        byte_count,
    })
}

pub fn should_checkpoint(stats: WalStats, checkpoint_ops: usize, checkpoint_bytes: usize) -> bool {
    stats.op_count > 0 && (stats.op_count >= checkpoint_ops || stats.byte_count >= checkpoint_bytes)
}

pub fn read_records(path: &Path) -> CoreResult<Vec<WalRecord>> {
    Ok(read_records_with_valid_len(path)?.records)
}

/// The outcome of scanning a WAL: the replayable records plus the byte length of
/// the valid frame prefix and the file's total length. `total_len > valid_len`
/// means a crash left a torn trailing frame (which readers drop); `valid_len`
/// is where a repair should truncate so the next append lands after complete
/// records.
pub struct WalScan {
    pub records: Vec<WalRecord>,
    pub valid_len: u64,
    pub total_len: u64,
}

pub fn read_records_with_valid_len(path: &Path) -> CoreResult<WalScan> {
    if !path.is_file() {
        return Ok(WalScan {
            records: Vec::new(),
            valid_len: 0,
            total_len: 0,
        });
    }
    let data =
        std::fs::read(path).map_err(|error| corrupt(format!("WAL could not be read: {error}")))?;
    if data.len() < 12 {
        return Ok(WalScan {
            records: Vec::new(),
            valid_len: 0,
            total_len: data.len() as u64,
        });
    }
    if &data[..WAL_MAGIC.len()] != WAL_MAGIC {
        return Err(corrupt("not a LodeDB WAL file (bad magic)"));
    }
    let mut version = [0_u8; 4];
    version.copy_from_slice(&data[8..12]);
    let version = u32::from_be_bytes(version);
    if version != WAL_SCHEMA_VERSION {
        return Err(corrupt(format!(
            "unsupported WAL schema version: {version}"
        )));
    }
    let mut records = Vec::new();
    let mut offset = 12_usize;
    let total = data.len();
    while offset < total {
        if offset + 4 > total {
            break;
        }
        let mut len = [0_u8; 4];
        len.copy_from_slice(&data[offset..offset + 4]);
        let body_len = u32::from_be_bytes(len) as usize;
        let frame_end = offset + 4 + body_len;
        let crc_end = frame_end + 4;
        if crc_end > total {
            break;
        }
        let frame = &data[offset..frame_end];
        let mut recorded_crc = [0_u8; 4];
        recorded_crc.copy_from_slice(&data[frame_end..crc_end]);
        let recorded_crc = u32::from_be_bytes(recorded_crc);
        if crc32(frame) != recorded_crc {
            if crc_end == total {
                break;
            }
            return Err(corrupt("WAL record failed CRC32 (interior corruption)"));
        }
        records.push(decode_body(&frame[4..])?);
        offset = crc_end;
    }
    Ok(WalScan {
        records,
        valid_len: offset as u64,
        total_len: total as u64,
    })
}

/// Truncates the WAL to `valid_len`, dropping a torn trailing frame left by a
/// crash mid-append so the next append lands after the last complete record.
pub fn truncate_to(path: &Path, valid_len: u64) -> CoreResult<()> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|error| corrupt(format!("WAL could not be opened to repair: {error}")))?;
    file.set_len(valid_len)
        .map_err(|error| corrupt(format!("WAL torn tail could not be repaired: {error}")))?;
    Ok(())
}

pub fn replay_records_onto_store(
    store: &mut LoadedStore,
    records: &[WalRecord],
    chunk_character_limit: usize,
) -> CoreResult<usize> {
    for record in records {
        apply_record(store, record, chunk_character_limit)?;
    }
    Ok(records.len())
}

pub fn checkpoint_store(
    persistence_dir: &Path,
    store: &LoadedStore,
    next_generation: u64,
    fsync: bool,
) -> CoreResult<()> {
    // Keep the compression the store was created with: read the persisted flag
    // from the loaded base's text-store manifest, defaulting to compressed when
    // the store has no `.tvtext` manifest (no text written, or a store from
    // before the flag existed).
    let compress_text = crate::storage::text_store::persisted_compress(
        &crate::storage::commit_manifest::base_tvtext_path(
            persistence_dir,
            &store.index_key,
            store.base_epoch,
        ),
    )
    .unwrap_or(true);
    let input = crate::storage::GenerationCommitInput {
        index_key: &store.index_key,
        generation: next_generation,
        base_epoch: next_generation,
        state: &store.state,
        tvim: None,
        raw_text: Some(&store.raw_text),
        lexical_tokens: Some(&store.lexical_tokens),
        multivec: Some(&store.multivec),
        compress_text,
    };
    crate::storage::write_generation_commit(
        persistence_dir,
        input,
        crate::storage::GenerationWriteOptions {
            fsync,
            retained_epochs: 4,
        },
    )?;
    truncate(&wal_path(persistence_dir, &store.index_key), fsync)
}

fn decode_body(body: &[u8]) -> CoreResult<WalRecord> {
    let Some(newline) = body.iter().position(|byte| *byte == b'\n') else {
        return Err(corrupt("WAL record body is missing its op header"));
    };
    let op = std::str::from_utf8(&body[..newline])
        .map_err(|error| corrupt(format!("WAL record op is not UTF-8: {error}")))?
        .to_string();
    let mut payload: Value = serde_json::from_slice(&body[newline + 1..])
        .map_err(|error| corrupt(format!("WAL record payload is not valid JSON: {error}")))?;
    if !payload.is_object() {
        return Err(corrupt("WAL record payload must be a JSON object"));
    }
    // Lift the framing sequence number out of the body so replay sees only the
    // op payload; a record from a pre-LSN WAL simply has no `lsn` key.
    let lsn = payload
        .as_object_mut()
        .and_then(|object| object.remove("lsn"))
        .and_then(|value| value.as_u64());
    Ok(WalRecord { op, payload, lsn })
}

fn encode_body(op: &str, payload: &Value, lsn: u64) -> CoreResult<Vec<u8>> {
    if op.is_empty() || op.contains('\n') {
        return Err(corrupt("WAL record op must be non-empty and newline-free"));
    }
    let Some(object) = payload.as_object() else {
        return Err(corrupt("WAL record payload must be a JSON object"));
    };
    // Stamp the log sequence number into the JSON body rather than the binary
    // frame, so the frame layout (and the committed cross-version WAL fixtures)
    // stays byte-compatible. `decode_body` lifts it back out on read.
    let mut object = object.clone();
    object.insert("lsn".to_string(), Value::from(lsn));
    let mut body = Vec::new();
    body.extend_from_slice(op.as_bytes());
    body.push(b'\n');
    let payload = serde_json::to_vec(&Value::Object(object))
        .map_err(|error| corrupt(format!("WAL payload could not be encoded: {error}")))?;
    body.extend_from_slice(&payload);
    Ok(body)
}

fn apply_record(
    store: &mut LoadedStore,
    record: &WalRecord,
    chunk_character_limit: usize,
) -> CoreResult<()> {
    match record.op.as_str() {
        "upsert_documents" => {
            for document in record
                .payload
                .get("documents")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                upsert_document(store, document, chunk_character_limit)?;
            }
        }
        "delete_documents" => {
            for document_id in record
                .payload
                .get("document_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                delete_document(store, document_id)?;
            }
        }
        "update_document_payload" => {
            let document_id = record
                .payload
                .get("document_id")
                .and_then(Value::as_str)
                .ok_or_else(|| corrupt("WAL update payload missing document_id"))?;
            if let Some(metadata) = record.payload.get("metadata") {
                set_document_metadata(store, document_id, metadata)?;
            }
            if record
                .payload
                .get("clear_text")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                store.raw_text.remove(document_id);
            } else if let Some(text) = record.payload.get("text").and_then(Value::as_str) {
                store
                    .raw_text
                    .insert(document_id.to_string(), text.to_string());
            }
        }
        "upsert_vectors" => {
            for vector in record
                .payload
                .get("vectors")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let document_id = vector
                    .get("document_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| corrupt("WAL vector payload missing document_id"))?;
                replace_document_chunks(store, document_id, Vec::new())?;
                set_document_metadata(
                    store,
                    document_id,
                    vector.get("metadata").unwrap_or(&Value::Null),
                )?;
                if let Some(text) = vector.get("text").and_then(Value::as_str) {
                    store
                        .raw_text
                        .insert(document_id.to_string(), text.to_string());
                }
                set_document_hash(store, document_id, &sha256_text(document_id))?;
            }
        }
        "apply_embedded_documents" => {
            for document in record
                .payload
                .get("documents")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let document_id = document
                    .get("document_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| corrupt("WAL embedded payload missing document_id"))?;
                let chunk_ids = document
                    .get("chunk_ids")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(ToString::to_string)
                    .collect::<Vec<_>>();
                set_document_hash(
                    store,
                    document_id,
                    document
                        .get("content_hash")
                        .and_then(Value::as_str)
                        .unwrap_or(""),
                )?;
                set_document_metadata(
                    store,
                    document_id,
                    document.get("metadata").unwrap_or(&Value::Null),
                )?;
                set_document_chunk_ids(store, document_id, chunk_ids)?;
            }
            let removed = record
                .payload
                .get("removed_chunk_ids")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToString::to_string)
                .collect::<std::collections::BTreeSet<_>>();
            remove_chunks(store, &removed)?;
            for chunk in record
                .payload
                .get("added_chunks")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                add_chunk_row(store, chunk.clone())?;
            }
        }
        other => {
            return Err(corrupt(format!(
                "unknown WAL record op during replay: {other}"
            )))
        }
    }
    Ok(())
}

fn upsert_document(
    store: &mut LoadedStore,
    document: &Value,
    chunk_character_limit: usize,
) -> CoreResult<()> {
    let document_id = document
        .get("document_id")
        .and_then(Value::as_str)
        .ok_or_else(|| corrupt("WAL document payload missing document_id"))?;
    let text = document
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| corrupt("WAL document payload missing text"))?;
    let content_hash = sha256_text(text);
    let mut chunk_rows = Vec::new();
    for (position, chunk) in chunk_text(text, chunk_character_limit)?.iter().enumerate() {
        let chunk_hash = normalized_chunk_hash(chunk);
        chunk_rows.push(serde_json::json!({
            "chunk_id": chunk_id_for_hash(document_id, &chunk_hash, position),
            "content_hash": chunk_hash,
            "document_id": document_id,
        }));
    }
    replace_document_chunks(store, document_id, chunk_rows)?;
    set_document_hash(store, document_id, &content_hash)?;
    set_document_metadata(
        store,
        document_id,
        document.get("metadata").unwrap_or(&Value::Null),
    )?;
    store
        .raw_text
        .insert(document_id.to_string(), text.to_string());
    Ok(())
}

fn delete_document(store: &mut LoadedStore, document_id: &str) -> CoreResult<()> {
    let chunk_ids = store
        .state
        .get("document_chunk_ids")
        .and_then(Value::as_object)
        .and_then(|object| object.get(document_id))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToString::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    remove_chunks(store, &chunk_ids)?;
    for key in ["document_hashes", "document_chunk_ids", "document_metadata"] {
        if let Some(object) = store.state.get_mut(key).and_then(Value::as_object_mut) {
            object.remove(document_id);
        }
    }
    store.raw_text.remove(document_id);
    store.lexical_tokens.remove(document_id);
    Ok(())
}

fn replace_document_chunks(
    store: &mut LoadedStore,
    document_id: &str,
    chunk_rows: Vec<Value>,
) -> CoreResult<()> {
    let old = store
        .state
        .get("document_chunk_ids")
        .and_then(Value::as_object)
        .and_then(|object| object.get(document_id))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToString::to_string)
        .collect::<std::collections::BTreeSet<_>>();
    remove_chunks(store, &old)?;
    let chunk_ids = chunk_rows
        .iter()
        .filter_map(|row| row.get("chunk_id").and_then(Value::as_str))
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    for row in chunk_rows {
        add_chunk_row(store, row)?;
    }
    set_document_chunk_ids(store, document_id, chunk_ids)
}

fn remove_chunks(
    store: &mut LoadedStore,
    chunk_ids: &std::collections::BTreeSet<String>,
) -> CoreResult<()> {
    let chunks = store
        .state
        .get_mut("chunks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| corrupt("loaded state chunks must be an array"))?;
    chunks.retain(|row| match row.get("chunk_id").and_then(Value::as_str) {
        Some(chunk_id) => !chunk_ids.contains(chunk_id),
        None => true,
    });
    Ok(())
}

fn add_chunk_row(store: &mut LoadedStore, row: Value) -> CoreResult<()> {
    let chunk_id = row
        .get("chunk_id")
        .and_then(Value::as_str)
        .ok_or_else(|| corrupt("chunk row missing chunk_id"))?
        .to_string();
    let chunks = store
        .state
        .get_mut("chunks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| corrupt("loaded state chunks must be an array"))?;
    chunks.retain(|existing| {
        existing
            .get("chunk_id")
            .and_then(Value::as_str)
            .map_or(true, |existing_id| existing_id != chunk_id)
    });
    chunks.push(row);
    chunks.sort_by(|left, right| {
        left.get("chunk_id")
            .and_then(Value::as_str)
            .cmp(&right.get("chunk_id").and_then(Value::as_str))
    });
    Ok(())
}

fn set_document_hash(
    store: &mut LoadedStore,
    document_id: &str,
    content_hash: &str,
) -> CoreResult<()> {
    let object = store
        .state
        .get_mut("document_hashes")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| corrupt("loaded state document_hashes must be an object"))?;
    object.insert(
        document_id.to_string(),
        Value::String(content_hash.to_string()),
    );
    Ok(())
}

fn set_document_chunk_ids(
    store: &mut LoadedStore,
    document_id: &str,
    chunk_ids: Vec<String>,
) -> CoreResult<()> {
    let object = store
        .state
        .get_mut("document_chunk_ids")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| corrupt("loaded state document_chunk_ids must be an object"))?;
    object.insert(
        document_id.to_string(),
        Value::Array(chunk_ids.into_iter().map(Value::String).collect()),
    );
    Ok(())
}

fn set_document_metadata(
    store: &mut LoadedStore,
    document_id: &str,
    metadata: &Value,
) -> CoreResult<()> {
    let object = store
        .state
        .get_mut("document_metadata")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| corrupt("loaded state document_metadata must be an object"))?;
    let metadata = metadata
        .as_object()
        .map(|object| {
            object
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        Value::String(value.as_str().unwrap_or("").to_string()),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    object.insert(document_id.to_string(), Value::Object(metadata));
    Ok(())
}

pub(crate) fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            let mask = 0_u32.wrapping_sub(crc & 1);
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::{LoadOptions, StoreLayout};
    use serde_json::json;
    use std::collections::BTreeMap;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "lodedb-core-wal-{name}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&path).expect("create temp dir");
        path
    }

    fn empty_state(index_key: &str) -> Value {
        json!({
            "schema_version": 1,
            "client_id_hash": index_key,
            "index_id": "default",
            "index_key": index_key,
            "name": "lodedb-local",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "provider": "local_open",
            "task": "direct-turbovec",
            "route_profile": "minilm-turbovec",
            "storage_profile": "turbovec_direct",
            "native_dim": 384,
            "turbovec_bit_width": 4,
            "status": "ready",
            "metadata": {},
            "chunks": [],
            "document_hashes": {},
            "document_chunk_ids": {},
            "document_metadata": {},
            "embedded_chunk_count": 0,
            "delete_count": 0,
            "deleted_chunk_count": 0,
            "cache_reuse_count": 0,
            "fallback_count": 0,
            "fallback_reasons": {},
            "query_count": 0,
            "columnar_generation": 1,
            "created_at": "2026-06-26T00:00:00+00:00",
            "updated_at": "2026-06-26T00:00:00+00:00"
        })
    }

    #[test]
    fn crc32_matches_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }

    #[test]
    fn appends_and_reads_framed_records() {
        let dir = temp_dir("append-read");
        let path = dir.join("default.wal");
        append_record(&path, 1, "upsert_documents", &json!({"documents": []}), false)
            .expect("append first");
        let append = append_record(
            &path,
            2,
            "delete_documents",
            &json!({"document_ids": ["alpha"]}),
            false,
        )
        .expect("append second");

        let records = read_records(&path).expect("read records");
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].op, "upsert_documents");
        assert_eq!(records[1].payload["document_ids"], json!(["alpha"]));
        // The sequence number round-trips and is lifted back out of the payload.
        assert_eq!(records[0].lsn, Some(1));
        assert_eq!(records[1].lsn, Some(2));
        assert!(records[1].payload.get("lsn").is_none());
        // op_count/byte_count come from an explicit scan, not the append hot path.
        let stats = scan_stats(&path).expect("scan stats");
        assert_eq!(stats.op_count, 2);
        assert!(stats.byte_count >= append.record_bytes);
    }

    #[test]
    fn drops_torn_trailing_frame() {
        let dir = temp_dir("torn-tail");
        let path = dir.join("default.wal");
        append_record(&path, 1, "upsert_documents", &json!({"documents": []}), false)
            .expect("append first");
        append_record(
            &path,
            2,
            "delete_documents",
            &json!({"document_ids": ["alpha"]}),
            false,
        )
        .expect("append second");
        let mut handle = OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open wal");
        handle.write_all(&[0, 0, 1]).expect("write torn tail");

        let records = read_records(&path).expect("read records");
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn corrupt_interior_frame_fails_closed() {
        let dir = temp_dir("interior-corrupt");
        let path = dir.join("default.wal");
        append_record(&path, 1, "upsert_documents", &json!({"documents": []}), false)
            .expect("append first");
        append_record(
            &path,
            2,
            "delete_documents",
            &json!({"document_ids": ["alpha"]}),
            false,
        )
        .expect("append second");
        let mut raw = std::fs::read(&path).expect("read wal bytes");
        raw[16] ^= 0xFF;
        std::fs::write(&path, raw).expect("write corrupt wal");

        let error = read_records(&path).expect_err("interior corruption should fail");
        assert!(error.to_string().contains("interior corruption"));
    }

    #[test]
    fn mixed_text_wal_replay_replaces_embedded_chunk_rows() {
        let index_key = "mixed-text-key";
        let chunk_id = chunk_id_for_hash("alpha", &normalized_chunk_hash("Alpha text."), 0);
        let records = vec![
            WalRecord {
                op: "upsert_documents".to_string(),
                payload: json!({
                    "documents": [{
                        "document_id": "alpha",
                        "text": "Alpha text.",
                        "metadata": {"kind": "text"}
                    }]
                }),
                lsn: Some(1),
            },
            WalRecord {
                op: "apply_embedded_documents".to_string(),
                payload: json!({
                    "documents": [{
                        "document_id": "alpha",
                        "content_hash": sha256_text("Alpha text."),
                        "metadata": {"kind": "text"},
                        "chunk_ids": [chunk_id.clone()],
                        "tokens": [["alpha", "text"]]
                    }],
                    "added_chunks": [{
                        "chunk_id": chunk_id.clone(),
                        "document_id": "alpha",
                        "content_hash": normalized_chunk_hash("Alpha text."),
                        "embedding": [1.0, 0.0, 0.0]
                    }],
                    "removed_chunk_ids": []
                }),
                lsn: Some(2),
            },
            WalRecord {
                op: "apply_embedded_documents".to_string(),
                payload: json!({
                    "documents": [{
                        "document_id": "alpha",
                        "content_hash": sha256_text("Alpha text."),
                        "metadata": {"kind": "text"},
                        "chunk_ids": [chunk_id.clone()],
                        "tokens": [["alpha", "text"]]
                    }],
                    "added_chunks": [{
                        "chunk_id": chunk_id.clone(),
                        "document_id": "alpha",
                        "content_hash": normalized_chunk_hash("Alpha text."),
                        "embedding": [1.0, 0.0, 0.0]
                    }],
                    "removed_chunk_ids": []
                }),
                lsn: Some(3),
            },
        ];
        let mut store = LoadedStore {
            layout: StoreLayout::Generation,
            index_key: index_key.to_string(),
            generation: 1,
            base_epoch: 1,
            state: empty_state(index_key),
            tvim_path: None,
            tvim_manifest: None,
            raw_text: BTreeMap::new(),
            lexical_tokens: BTreeMap::new(),
            multivec: Default::default(),
            wal_records: Vec::new(),
        };

        replay_records_onto_store(&mut store, &records, 8192).expect("replay mixed text wal");

        let chunks = store.state["chunks"].as_array().expect("chunks array");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0]["chunk_id"], json!(chunk_id));
        assert_eq!(chunks[0]["embedding"], json!([1.0, 0.0, 0.0]));
        assert_eq!(
            store.raw_text.get("alpha").map(String::as_str),
            Some("Alpha text.")
        );
    }

    #[test]
    fn checkpoint_writes_generation_and_truncates_wal() {
        let dir = temp_dir("checkpoint");
        let index_key = "checkpoint-key";
        let wal = wal_path(&dir, index_key);
        append_record(
            &wal,
            1,
            "upsert_documents",
            &json!({
                "documents": [{
                    "document_id": "alpha",
                    "text": "Alpha text.",
                    "metadata": {}
                }]
            }),
            false,
        )
        .expect("append wal");
        let mut state = empty_state(index_key);
        state["document_hashes"]["alpha"] = json!(sha256_text("Alpha text."));
        let mut raw_text = BTreeMap::new();
        raw_text.insert("alpha".to_string(), "Alpha text.".to_string());
        let store = LoadedStore {
            layout: StoreLayout::Generation,
            index_key: index_key.to_string(),
            generation: 1,
            base_epoch: 1,
            state,
            tvim_path: None,
            tvim_manifest: None,
            raw_text,
            lexical_tokens: BTreeMap::new(),
            multivec: Default::default(),
            wal_records: Vec::new(),
        };

        checkpoint_store(&dir, &store, 2, false).expect("checkpoint");

        assert!(!wal.exists());
        let loaded = crate::storage::load_store(&dir, index_key, LoadOptions::default())
            .expect("load checkpoint");
        assert_eq!(loaded.generation, 2);
        assert_eq!(loaded.document_count(), 1);
        assert_eq!(
            loaded.raw_text.get("alpha").map(String::as_str),
            Some("Alpha text.")
        );
    }
}
