//! Content-addressed blob naming for the managed remote layout.
//!
//! A managed remote can store artifacts content-addressed —
//! `blobs/sha256/aa/<sha256>` under a per-tenant prefix — instead of under
//! their engine path names. Content addressing is what absorbs the
//! fork-collision case (two branches committing different artifacts under the
//! same engine name, e.g. `idx.gen/g7.json`, coexist as two blobs), and the
//! two-level `aa/` fan-out keeps listings and prefix operations tractable.
//!
//! These helpers are pure and tested; they pin the naming contract the
//! transfer plane builds on. The digest is always the lowercase-hex
//! SHA-256 of the blob's bytes — the same digest the engine records per
//! artifact and [`ArtifactStore::write_bytes_if_absent`] verifies.
//!
//! [`ArtifactStore::write_bytes_if_absent`]: crate::ArtifactStore::write_bytes_if_absent

use crate::error::{ArtifactStoreError, Result};

/// The fixed prefix every blob name lives under.
const BLOB_PREFIX: &str = "blobs/sha256";

/// Returns the store-relative blob name for a content digest:
/// `blobs/sha256/aa/<sha256>`, where `aa` is the digest's first two hex chars.
///
/// Rejects anything that is not exactly 64 lowercase hex characters — blob
/// names participate in authorization paths, so a malformed digest must fail
/// closed rather than mint a name outside the contract.
pub fn blob_name(sha256: &str) -> Result<String> {
    validate_sha256(sha256)?;
    Ok(format!("{BLOB_PREFIX}/{}/{sha256}", &sha256[..2]))
}

/// Parses a blob name back to its content digest — the exact inverse of
/// [`blob_name`]. Rejects any name that deviates from the contract (wrong
/// prefix, fan-out directory disagreeing with the digest, malformed digest).
pub fn parse_blob_name(name: &str) -> Result<String> {
    let malformed = || {
        ArtifactStoreError::Integrity(format!(
            "blob name {name:?} does not match {BLOB_PREFIX}/aa/<sha256>"
        ))
    };
    let rest = name.strip_prefix(BLOB_PREFIX).ok_or_else(malformed)?;
    let rest = rest.strip_prefix('/').ok_or_else(malformed)?;
    let (fan_out, sha256) = rest.split_once('/').ok_or_else(malformed)?;
    validate_sha256(sha256).map_err(|_| malformed())?;
    if fan_out != &sha256[..2] {
        return Err(malformed());
    }
    Ok(sha256.to_string())
}

/// Requires exactly 64 lowercase hex characters. Also used by the generation
/// inventory to validate every manifest digest at the trust boundary: digests
/// become staging file names and object keys, so a non-hex "digest" must fail
/// closed before it can name a path.
pub(crate) fn validate_sha256(sha256: &str) -> Result<()> {
    let valid = sha256.len() == 64
        && sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    if valid {
        Ok(())
    } else {
        Err(ArtifactStoreError::Integrity(format!(
            "{sha256:?} is not a lowercase-hex sha256 digest"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::{blob_name, parse_blob_name};
    use crate::error::ArtifactStoreError;

    const SHA: &str = "aa30b1cc05c10ac8a1f4e6d46f7d4b1a9c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f";

    #[test]
    fn names_a_blob_under_its_two_char_fan_out() {
        assert_eq!(blob_name(SHA).unwrap(), format!("blobs/sha256/aa/{SHA}"));
    }

    #[test]
    fn parse_inverts_blob_name() {
        assert_eq!(parse_blob_name(&blob_name(SHA).unwrap()).unwrap(), SHA);
    }

    #[test]
    fn rejects_malformed_digests() {
        for bad in [
            "",
            "abc",
            &SHA[..63],                 // too short
            &format!("{SHA}0"),         // too long
            &SHA.to_uppercase(),        // uppercase
            &format!("g{}", &SHA[1..]), // non-hex
        ] {
            assert!(matches!(
                blob_name(bad).unwrap_err(),
                ArtifactStoreError::Integrity(_)
            ));
        }
    }

    #[test]
    fn rejects_names_off_contract() {
        for bad in [
            format!("blobs/sha256/{SHA}"),             // no fan-out level
            format!("blobs/sha256/bb/{SHA}"),          // fan-out disagrees
            format!("blobs/md5/aa/{SHA}"),             // wrong algorithm segment
            format!("sha256/aa/{SHA}"),                // wrong prefix
            format!("blobs/sha256/aa/{}", &SHA[..63]), // malformed digest
        ] {
            assert!(matches!(
                parse_blob_name(&bad).unwrap_err(),
                ArtifactStoreError::Integrity(_)
            ));
        }
    }
}
