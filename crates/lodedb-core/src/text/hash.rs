//! Hashing helpers that mirror the Python oracle.

use sha2::{Digest, Sha256};
use std::fmt::Write;

/// Returns a stable SHA-256 hex digest for a UTF-8 string.
pub fn sha256_text(value: &str) -> String {
    sha256_hex(value.as_bytes())
}

/// Hashes normalized chunk text so harmless whitespace changes reuse embeddings.
pub fn normalized_chunk_hash(text: &str) -> String {
    sha256_text(&text.split_whitespace().collect::<Vec<_>>().join(" "))
}

pub(crate) fn sha256_digest(bytes: &[u8]) -> [u8; 32] {
    let digest = Sha256::digest(bytes);
    let mut out = [0_u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = sha256_digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{normalized_chunk_hash, sha256_text};

    #[test]
    fn hashes_utf8_text() {
        assert_eq!(
            sha256_text("alpha document"),
            "49ec476cea2b49f4ea1308c54b61fc858efc50379892cc10ce328fb2720b17a7"
        );
    }

    #[test]
    fn normalizes_whitespace_like_python_split_join() {
        assert_eq!(
            normalized_chunk_hash("alpha\n\t beta   gamma"),
            sha256_text("alpha beta gamma")
        );
    }
}
