use crate::storage::util::{corrupt, CoreResult};
use serde_json::Value;
use std::path::{Path, PathBuf};

pub const WAL_SUFFIX: &str = ".wal";
pub const WAL_MAGIC: &[u8; 8] = b"EELWAL01";
pub const WAL_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct WalRecord {
    pub op: String,
    pub payload: Value,
}

pub fn wal_path(persistence_dir: &Path, index_key: &str) -> PathBuf {
    persistence_dir.join(format!("{index_key}{WAL_SUFFIX}"))
}

pub fn read_records(path: &Path) -> CoreResult<Vec<WalRecord>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let data =
        std::fs::read(path).map_err(|error| corrupt(format!("WAL could not be read: {error}")))?;
    if data.len() < 12 {
        return Ok(Vec::new());
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
    Ok(records)
}

fn decode_body(body: &[u8]) -> CoreResult<WalRecord> {
    let Some(newline) = body.iter().position(|byte| *byte == b'\n') else {
        return Err(corrupt("WAL record body is missing its op header"));
    };
    let op = std::str::from_utf8(&body[..newline])
        .map_err(|error| corrupt(format!("WAL record op is not UTF-8: {error}")))?
        .to_string();
    let payload: Value = serde_json::from_slice(&body[newline + 1..])
        .map_err(|error| corrupt(format!("WAL record payload is not valid JSON: {error}")))?;
    if !payload.is_object() {
        return Err(corrupt("WAL record payload must be a JSON object"));
    }
    Ok(WalRecord { op, payload })
}

fn crc32(bytes: &[u8]) -> u32 {
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
    use super::crc32;

    #[test]
    fn crc32_matches_known_vector() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
    }
}
