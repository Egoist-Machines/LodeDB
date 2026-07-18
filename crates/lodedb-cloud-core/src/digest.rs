//! Shared artifact checksum helper.
//!
//! Every artifact is addressed by the lowercase-hex SHA-256 of its bytes — the
//! exact digest the engine records per artifact (`sha256_file_hex`). Both the
//! write path (`LocalArtifactStore::write_bytes_if_absent`) and the verify path
//! (`verify_generation`) re-hash bytes and compare against the manifest's recorded
//! value, so the hashing lives in one place to guarantee they agree byte-for-byte.

use sha2::{Digest, Sha256};
use std::io::Read;

/// Returns the lowercase-hex SHA-256 of `data`.
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    hex(Sha256::digest(data).as_slice())
}

/// Finalizes an incremental hasher into the same lowercase-hex form.
pub(crate) fn sha256_hex_finish(hasher: Sha256) -> String {
    hex(hasher.finalize().as_slice())
}

/// Streams `reader` to EOF and returns its lowercase-hex SHA-256 plus the byte
/// count — the bounded-memory replacement for hashing a fully buffered
/// artifact (peak memory is one copy buffer, not the artifact).
pub(crate) fn sha256_hex_reader(reader: &mut dyn Read) -> std::io::Result<(String, u64)> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let mut total = 0u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            return Ok((sha256_hex_finish(hasher), total));
        }
        hasher.update(&buffer[..read]);
        total += read as u64;
    }
}

/// The fixed copy-buffer size every streaming path uses (hashing, staging
/// copies, uploads): peak transfer memory is a small multiple of this, never
/// a function of artifact size.
pub(crate) const COPY_BUFFER_BYTES: usize = 1024 * 1024;

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}
