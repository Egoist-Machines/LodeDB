//! Durable per-document multi-vector (late-interaction) patch-matrix store.
//!
//! Late-interaction retrieval keeps a *matrix* of patch vectors per document (the
//! pooled vector is the indexed coarse row; the full matrix is rescored with
//! MaxSim). This store persists those matrices natively instead of riding the raw
//! per-row text sidecar as base64, so the native engine owns late-interaction
//! durability end to end.
//!
//! Layout mirrors the other native stores: a binary base segment plus an
//! append-only delta journal under ``<base>.tvmv-delta/`` with a JSON manifest.
//! Each segment is ``MAGIC | header_len (u64 LE) | header JSON | blobs`` where the
//! header lists every document's ``{id, dtype, patch_count, nbytes, sha256}`` and
//! the blobs are the matrices concatenated in header order. Blobs use the same
//! encoding as the Python writer (``little-endian f4``/``f2`` for float32/float16,
//! or per-vector symmetric int8: ``f4`` scales followed by ``i1`` codes), so a
//! store round-trips byte-identically across the two engines.

use crate::storage::util::{
    corrupt, f16_bits_to_f32, get_i64, get_str, read_json, sha256_bytes_hex, sha256_file_hex, sidecar_base_block,
    validate_sidecar_manifest, value_object, write_bytes_atomic, write_pretty_json_atomic,
    CoreResult,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const MULTIVEC_DELTA_DIR_SUFFIX: &str = ".tvmv-delta";
pub const MULTIVEC_MANIFEST_NAME: &str = "manifest.json";
pub const MULTIVEC_SCHEMA_VERSION: i64 = 1;
pub const MULTIVEC_BASE_MAGIC: &[u8; 8] = b"EEMVB001";
pub const MULTIVEC_DELTA_MAGIC: &[u8; 8] = b"EEMVD001";

/// One document's stored patch matrix: its storage dtype, patch count, and the
/// encoded matrix bytes (the exact buffer the Python `_encode_matrix` produces).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiVecRecord {
    pub dtype: String,
    pub patch_count: usize,
    pub bytes: Vec<u8>,
}

pub type MultiVecMap = BTreeMap<String, MultiVecRecord>;

impl MultiVecRecord {
    /// Decodes the stored patch matrix to row-major f32 (`patch_count * dim`),
    /// the inverse of the Python `_decode_matrix` for each storage dtype:
    /// little-endian f4/f2 for float32/float16, or per-vector symmetric int8
    /// (`n` f32 scales followed by `n * dim` i8 codes, value = code * scale / 127).
    pub fn decode(&self, dim: usize) -> Vec<f32> {
        match self.dtype.as_str() {
            "float16" => self
                .bytes
                .chunks_exact(2)
                .map(|chunk| f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect(),
            "int8" => {
                if dim == 0 {
                    return Vec::new();
                }
                let rows = self.bytes.len() / (4 + dim);
                let (scale_bytes, code_bytes) = self.bytes.split_at(rows * 4);
                let mut out = Vec::with_capacity(rows * dim);
                for row in 0..rows {
                    let scale = f32::from_le_bytes([
                        scale_bytes[row * 4],
                        scale_bytes[row * 4 + 1],
                        scale_bytes[row * 4 + 2],
                        scale_bytes[row * 4 + 3],
                    ]);
                    let factor = scale / 127.0;
                    for col in 0..dim {
                        out.push(code_bytes[row * dim + col] as i8 as f32 * factor);
                    }
                }
                out
            }
            // float32 and any unknown dtype fall back to raw little-endian f32.
            _ => self
                .bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect(),
        }
    }
}

fn delta_dir(base_path: &Path) -> PathBuf {
    base_path.with_file_name(format!(
        "{}{}",
        base_path.file_name().unwrap_or_default().to_string_lossy(),
        MULTIVEC_DELTA_DIR_SUFFIX
    ))
}

pub fn manifest_path(base_path: &Path) -> PathBuf {
    delta_dir(base_path).join(MULTIVEC_MANIFEST_NAME)
}

/// Loads the base segment then replays the journaled deltas in sequence order.
pub fn load(base_path: &Path, manifest: Option<&Value>) -> CoreResult<MultiVecMap> {
    if !base_path.is_file() {
        return Ok(MultiVecMap::new());
    }
    if let Some(manifest) = manifest {
        validate_sidecar_manifest(base_path, manifest, MULTIVEC_SCHEMA_VERSION, "multi-vector")?;
    }
    let mut documents = read_segment(base_path, MULTIVEC_BASE_MAGIC, "multi-vector base")?.1;
    let Some(manifest) = manifest else {
        return Ok(documents);
    };
    let manifest = value_object(manifest, "multi-vector manifest")?;
    let dir = delta_dir(base_path);
    let mut previous_seq = -1_i64;
    for entry in manifest
        .get("deltas")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let entry = value_object(entry, "multi-vector delta manifest entry")?;
        let sequence = get_i64(entry, "seq", -1);
        if sequence <= previous_seq {
            return Err(corrupt("multi-vector manifest has out-of-order segments"));
        }
        previous_seq = sequence;
        let file_name = get_str(entry, "file_name");
        let path = dir.join(file_name);
        if file_name.is_empty() || !path.is_file() {
            return Err(corrupt(format!(
                "multi-vector segment is missing: {file_name}"
            )));
        }
        if sha256_file_hex(&path)? != get_str(entry, "sha256") {
            return Err(corrupt(format!(
                "multi-vector segment failed checksum: {file_name}"
            )));
        }
        let (header, upserted) = read_segment(&path, MULTIVEC_DELTA_MAGIC, "multi-vector segment")?;
        let header = value_object(&header, "multi-vector segment header")?;
        for deleted in header
            .get("deleted")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            documents.remove(deleted);
        }
        for (id, record) in upserted {
            documents.insert(id, record);
        }
    }
    Ok(documents)
}

/// Frames a base or delta segment: `MAGIC | header_len | header JSON | blobs`.
fn encode_segment(magic: &[u8; 8], header: &Value, ordered: &[(&String, &MultiVecRecord)]) -> Vec<u8> {
    let header_blob = serde_json::to_vec(header).unwrap_or_default();
    let mut out = Vec::with_capacity(16 + header_blob.len());
    out.extend_from_slice(magic);
    out.extend_from_slice(&(header_blob.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_blob);
    for (_, record) in ordered {
        out.extend_from_slice(&record.bytes);
    }
    out
}

/// Builds a header `documents` array describing each blob in write order.
fn document_specs(ordered: &[(&String, &MultiVecRecord)]) -> Vec<Value> {
    ordered
        .iter()
        .map(|(id, record)| {
            serde_json::json!({
                "id": id,
                "dtype": record.dtype,
                "patch_count": record.patch_count,
                "nbytes": record.bytes.len(),
                "sha256": sha256_bytes_hex(&record.bytes),
            })
        })
        .collect()
}

/// Reads a segment file, validating the magic, header, and each blob's checksum.
/// Returns the parsed header and the documents it carries (in header order).
fn read_segment(path: &Path, magic: &[u8; 8], context: &str) -> CoreResult<(Value, MultiVecMap)> {
    let data = fs::read(path)
        .map_err(|error| corrupt(format!("{context} could not be read: {error}")))?;
    let prefix = magic.len() + 8;
    if data.len() < prefix || &data[..magic.len()] != magic {
        return Err(corrupt(format!("not a {context}: {}", path.display())));
    }
    let mut length = [0_u8; 8];
    length.copy_from_slice(&data[magic.len()..prefix]);
    let header_stop = prefix + u64::from_le_bytes(length) as usize;
    if data.len() < header_stop {
        return Err(corrupt(format!("{context} header is truncated")));
    }
    let header: Value = serde_json::from_slice(&data[prefix..header_stop])
        .map_err(|error| corrupt(format!("{context} header is corrupt: {error}")))?;
    let header_object = value_object(&header, context)?;
    if get_i64(header_object, "schema_version", -1) != MULTIVEC_SCHEMA_VERSION {
        return Err(corrupt(format!("unsupported {context} schema version")));
    }
    let mut documents = MultiVecMap::new();
    let mut offset = header_stop;
    for spec in header_object
        .get("documents")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let spec = value_object(spec, "multi-vector document spec")?;
        let id = get_str(spec, "id");
        if id.is_empty() {
            return Err(corrupt(format!("{context} document is missing its id")));
        }
        let nbytes = spec
            .get("nbytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| corrupt(format!("{context} document byte count is missing")))?
            as usize;
        let stop = offset + nbytes;
        if data.len() < stop {
            return Err(corrupt(format!("{context} document blob is truncated")));
        }
        let bytes = data[offset..stop].to_vec();
        if sha256_bytes_hex(&bytes) != get_str(spec, "sha256") {
            return Err(corrupt(format!("{context} document {id} failed checksum")));
        }
        documents.insert(
            id.to_string(),
            MultiVecRecord {
                dtype: get_str(spec, "dtype").to_string(),
                patch_count: get_i64(spec, "patch_count", 0).max(0) as usize,
                bytes,
            },
        );
        offset = stop;
    }
    Ok((header, documents))
}

/// Writes a fresh base segment + manifest, replacing any prior journal.
pub fn record_base(base_path: &Path, documents: &MultiVecMap, fsync: bool) -> CoreResult<Value> {
    let ordered: Vec<(&String, &MultiVecRecord)> = documents.iter().collect();
    let header = serde_json::json!({
        "schema_version": MULTIVEC_SCHEMA_VERSION,
        "documents": document_specs(&ordered),
    });
    write_bytes_atomic(base_path, &encode_segment(MULTIVEC_BASE_MAGIC, &header, &ordered), fsync)?;
    let path = manifest_path(base_path);
    let previous = if path.is_file() {
        Some(read_json(&path, "multi-vector manifest")?)
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
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            corrupt(format!(
                "multi-vector delta directory could not be created: {error}"
            ))
        })?;
    }
    let manifest = serde_json::json!({
        "schema_version": MULTIVEC_SCHEMA_VERSION,
        "base": sidecar_base_block(
            base_path,
            "multi-vector",
            [("document_count", Value::from(documents.len()))],
        )?,
        "deltas": [],
        "next_seq": next_seq,
    });
    write_pretty_json_atomic(&path, &manifest, fsync)?;
    Ok(manifest)
}

/// Appends an upserted/deleted delta segment onto the existing base + manifest.
pub fn append_delta(
    base_path: &Path,
    upserted: &MultiVecMap,
    deleted: &[String],
    document_count_after: usize,
    fsync: bool,
) -> CoreResult<Value> {
    let path = manifest_path(base_path);
    let mut manifest = read_json(&path, "multi-vector manifest")?;
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| corrupt("multi-vector manifest must be a JSON object"))?;
    let sequence = manifest_object
        .get("next_seq")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let ordered: Vec<(&String, &MultiVecRecord)> = upserted.iter().collect();
    let header = serde_json::json!({
        "schema_version": MULTIVEC_SCHEMA_VERSION,
        "seq": sequence,
        "document_count_after": document_count_after,
        "deleted": deleted,
        "documents": document_specs(&ordered),
    });
    let segment_name = format!("mv-{sequence:08}.mvd");
    let segment_path = delta_dir(base_path).join(&segment_name);
    write_bytes_atomic(
        &segment_path,
        &encode_segment(MULTIVEC_DELTA_MAGIC, &header, &ordered),
        fsync,
    )?;
    let deltas = manifest_object
        .entry("deltas")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| corrupt("multi-vector manifest deltas must be a list"))?;
    deltas.push(serde_json::json!({
        "file_name": segment_name,
        "sha256": sha256_file_hex(&segment_path)?,
        "file_bytes": segment_path.metadata().map_err(|error| corrupt(format!("multi-vector segment metadata failed: {error}")))?.len(),
        "seq": sequence,
        "upserted": upserted.len(),
        "deleted": deleted.len(),
    }));
    manifest_object.insert("next_seq".to_string(), Value::from(sequence + 1));
    write_pretty_json_atomic(&path, &manifest, fsync)?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_dir(name: &str) -> PathBuf {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "lodedb_multivec_{name}_{}_{}_{nanos}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed),
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn record(dtype: &str, patch_count: usize, bytes: &[u8]) -> MultiVecRecord {
        MultiVecRecord {
            dtype: dtype.to_string(),
            patch_count,
            bytes: bytes.to_vec(),
        }
    }

    #[test]
    fn base_then_deltas_round_trip() {
        let base = unique_dir("round_trip").join("g0.tvmv");
        let mut documents = MultiVecMap::new();
        documents.insert("a".to_string(), record("float32", 2, &[1, 2, 3, 4]));
        documents.insert("b".to_string(), record("int8", 1, &[9, 9]));
        let manifest = record_base(&base, &documents, false).unwrap();
        assert_eq!(load(&base, Some(&manifest)).unwrap(), documents);

        // Upsert "a" (changed), add "c", delete "b".
        let mut upserted = MultiVecMap::new();
        upserted.insert("a".to_string(), record("float16", 3, &[7, 7, 7, 7, 7, 7]));
        upserted.insert("c".to_string(), record("float32", 1, &[5, 6]));
        let manifest = append_delta(&base, &upserted, &["b".to_string()], 2, false).unwrap();

        let mut expected = MultiVecMap::new();
        expected.insert("a".to_string(), record("float16", 3, &[7, 7, 7, 7, 7, 7]));
        expected.insert("c".to_string(), record("float32", 1, &[5, 6]));
        assert_eq!(load(&base, Some(&manifest)).unwrap(), expected);
    }

    #[test]
    fn decode_matches_storage_encoding() {
        // float32: raw little-endian f32.
        let mut f32_bytes = Vec::new();
        for value in [1.0_f32, -2.5, 3.0, 0.0] {
            f32_bytes.extend_from_slice(&value.to_le_bytes());
        }
        assert_eq!(record("float32", 2, &f32_bytes).decode(2), vec![1.0, -2.5, 3.0, 0.0]);

        // int8: 1 patch, dim 2, scale 2.0, codes [127, -64].
        let mut int_bytes = 2.0_f32.to_le_bytes().to_vec();
        int_bytes.push(127_i8 as u8);
        int_bytes.push((-64_i8) as u8);
        let decoded = record("int8", 1, &int_bytes).decode(2);
        assert!((decoded[0] - 2.0).abs() < 1e-6);
        assert!((decoded[1] - (-64.0 * 2.0 / 127.0)).abs() < 1e-6);

        // float16: 1.0 == 0x3c00, -2.0 == 0xc000.
        let mut f16_bytes = 0x3c00_u16.to_le_bytes().to_vec();
        f16_bytes.extend_from_slice(&0xc000_u16.to_le_bytes());
        assert_eq!(record("float16", 1, &f16_bytes).decode(2), vec![1.0, -2.0]);
    }

    #[test]
    fn missing_base_is_empty() {
        let base = unique_dir("absent").join("absent.tvmv");
        assert!(load(&base, None).unwrap().is_empty());
    }

    #[test]
    fn corrupt_blob_is_rejected() {
        let base = unique_dir("corrupt").join("g0.tvmv");
        let mut documents = MultiVecMap::new();
        documents.insert("a".to_string(), record("float32", 1, &[1, 2, 3, 4]));
        let manifest = record_base(&base, &documents, false).unwrap();
        // Flip a payload byte (the blob trails the header); the file-level manifest
        // checksum must catch it.
        let mut raw = std::fs::read(&base).unwrap();
        let last = raw.len() - 1;
        raw[last] ^= 0xff;
        std::fs::write(&base, &raw).unwrap();
        assert!(load(&base, Some(&manifest)).is_err());
    }
}
