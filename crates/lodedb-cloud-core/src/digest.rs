//! Shared artifact checksum helper.
//!
//! Every artifact is addressed by the lowercase-hex SHA-256 of its bytes — the
//! exact digest the engine records per artifact (`sha256_file_hex`). Both the
//! write path (`LocalArtifactStore::write_bytes_if_absent`) and the verify path
//! (`verify_generation`) re-hash bytes and compare against the manifest's recorded
//! value, so the hashing lives in one place to guarantee they agree byte-for-byte.

use sha2::{Digest, Sha256};

/// Returns the lowercase-hex SHA-256 of `data`.
pub(crate) fn sha256_hex(data: &[u8]) -> String {
    Sha256::digest(data)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
