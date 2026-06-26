use crate::storage::util::{
    corrupt, get_i64, get_str, sha256_bytes_hex, sha256_file_hex, value_object, verify_file_sha256,
    CoreResult,
};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const TVIM_DELTA_DIR_SUFFIX: &str = ".tvim-delta";
pub const TVIM_DELTA_MANIFEST_NAME: &str = "manifest.json";
pub const TVIM_DELTA_MAGIC: &[u8; 8] = b"EETVD001";
pub const TVIM_DELTA_SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TvimDeltaSummary {
    pub segment_count: usize,
    pub upsert_rows: u64,
    pub removed_rows: u64,
}

pub fn manifest_path(base_path: &Path) -> PathBuf {
    base_path
        .with_file_name(format!(
            "{}{}",
            base_path.file_name().unwrap().to_string_lossy(),
            TVIM_DELTA_DIR_SUFFIX
        ))
        .join(TVIM_DELTA_MANIFEST_NAME)
}

pub fn validate(base_path: &Path, manifest: Option<&Value>) -> CoreResult<TvimDeltaSummary> {
    let Some(manifest) = manifest else {
        return Ok(TvimDeltaSummary {
            segment_count: 0,
            upsert_rows: 0,
            removed_rows: 0,
        });
    };
    let manifest = value_object(manifest, "TurboVec delta manifest")?;
    if get_i64(manifest, "schema_version", -1) != TVIM_DELTA_SCHEMA_VERSION {
        return Err(corrupt(
            "unsupported TurboVec delta manifest schema version",
        ));
    }
    if let Some(base) = manifest.get("base").and_then(Value::as_object) {
        verify_file_sha256(base_path, get_str(base, "sha256"), "TurboVec base snapshot")?;
    }
    let delta_dir = base_path.with_file_name(format!(
        "{}{}",
        base_path.file_name().unwrap().to_string_lossy(),
        TVIM_DELTA_DIR_SUFFIX
    ));
    let mut previous_seq = -1_i64;
    let mut summary = TvimDeltaSummary {
        segment_count: 0,
        upsert_rows: 0,
        removed_rows: 0,
    };
    for entry in manifest
        .get("deltas")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let entry = value_object(entry, "TurboVec delta manifest entry")?;
        let sequence = get_i64(entry, "seq", -1);
        if sequence <= previous_seq {
            return Err(corrupt("TurboVec delta manifest has out-of-order segments"));
        }
        previous_seq = sequence;
        let file_name = get_str(entry, "file_name");
        let segment_path = delta_dir.join(file_name);
        if file_name.is_empty() || !segment_path.is_file() {
            return Err(corrupt(format!(
                "TurboVec delta segment is missing: {file_name}"
            )));
        }
        if sha256_file_hex(&segment_path)? != get_str(entry, "sha256") {
            return Err(corrupt(format!(
                "TurboVec delta segment failed checksum: {file_name}"
            )));
        }
        let header = read_delta_segment_header(&segment_path)?;
        summary.segment_count += 1;
        summary.upsert_rows += entry
            .get("upsert_rows")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        summary.removed_rows += entry
            .get("removed_rows")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if let Some(arrays) = header.get("arrays").and_then(Value::as_array) {
            for spec in arrays {
                let spec = value_object(spec, "TurboVec delta array spec")?;
                if get_str(spec, "name").is_empty() {
                    return Err(corrupt("TurboVec delta array is missing its name"));
                }
            }
        }
    }
    Ok(summary)
}

pub fn read_delta_segment_header(path: &Path) -> CoreResult<Value> {
    let data = std::fs::read(path)
        .map_err(|error| corrupt(format!("TurboVec delta segment could not be read: {error}")))?;
    let prefix = TVIM_DELTA_MAGIC.len() + 8;
    if data.len() < prefix || &data[..TVIM_DELTA_MAGIC.len()] != TVIM_DELTA_MAGIC {
        return Err(corrupt(format!(
            "not a TurboVec delta segment: {}",
            path.display()
        )));
    }
    let mut length = [0_u8; 8];
    length.copy_from_slice(&data[TVIM_DELTA_MAGIC.len()..prefix]);
    let header_len = u64::from_le_bytes(length) as usize;
    let header_stop = prefix + header_len;
    if data.len() < header_stop {
        return Err(corrupt("TurboVec delta segment header is truncated"));
    }
    let header: Value = serde_json::from_slice(&data[prefix..header_stop])
        .map_err(|error| corrupt(format!("TurboVec delta segment header is corrupt: {error}")))?;
    let header_object = value_object(&header, "TurboVec delta segment header")?;
    if get_i64(header_object, "schema_version", -1) != TVIM_DELTA_SCHEMA_VERSION {
        return Err(corrupt("unsupported TurboVec delta segment schema version"));
    }
    let mut offset = header_stop;
    for spec in header_object
        .get("arrays")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let spec = value_object(spec, "TurboVec delta array spec")?;
        let nbytes = spec
            .get("nbytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| corrupt("TurboVec delta array byte count is missing"))?
            as usize;
        let stop = offset + nbytes;
        if data.len() < stop {
            return Err(corrupt("TurboVec delta array is truncated"));
        }
        if sha256_bytes_hex(&data[offset..stop]) != get_str(spec, "sha256") {
            return Err(corrupt(format!(
                "TurboVec delta array {} failed checksum",
                get_str(spec, "name")
            )));
        }
        offset = stop;
    }
    Ok(header)
}
