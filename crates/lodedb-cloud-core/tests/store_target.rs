//! Tests for `artifact_store_from_target`: mapping a target string (local path
//! or object-store URL) onto the right `ArtifactStore` backend.

use lodedb_cloud_core::{artifact_store_from_target, ArtifactStoreError};

#[test]
fn a_plain_path_resolves_to_a_local_store() {
    // Resolution is lazy (no filesystem access), so any path shape works here;
    // reads/writes are what touch disk.
    let store = artifact_store_from_target("/tmp/some-backup-dir").unwrap();
    // An empty directory has no committed pointer.
    assert!(matches!(
        store.read_bytes("idx.gen/absent"),
        Err(ArtifactStoreError::NotFound(_) | ArtifactStoreError::Io(_))
    ));
}

/// `unwrap_err` would need the store to be `Debug`, so unpack by hand.
fn expect_backend_error(target: &str) {
    match artifact_store_from_target(target) {
        Err(ArtifactStoreError::Backend(_)) => {}
        Err(other) => panic!("expected a Backend error, got {other:?}"),
        Ok(_) => panic!("expected {target:?} to be rejected"),
    }
}

#[test]
fn an_s3_url_without_a_bucket_is_rejected() {
    expect_backend_error("s3://");
}

#[test]
fn an_unknown_scheme_is_rejected() {
    expect_backend_error("ftp://bucket/prefix");
}
