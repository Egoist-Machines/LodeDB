//! Tests for the string-target client operations (`client_ops`): the facade the
//! CLI/binding calls. The typed primitives are covered by their own suites;
//! these tests pin the composition — target resolution, policy pass-through,
//! and pull's built-in open-verification.

mod common;

use common::*;
use lodedb_core::storage::{
    write_generation_commit, GenerationCommitInput, GenerationWriteOptions,
};
use lodedb_cloud_core::client_ops::{keys, pull, push, status, verify};
use lodedb_cloud_core::{ArtifactStoreError, TransferPolicy};

/// One engine-written committed generation in `dir` under `key`.
fn commit_engine_generation(dir: &std::path::Path, key: &str) {
    let state = engine_state(key);
    write_generation_commit(
        dir,
        GenerationCommitInput {
            index_key: key,
            generation: 1,
            applied_lsn: 0,
            base_epoch: 1,
            state: &state,
            tvim: None,
            raw_text: None,
            lexical_tokens: None,
            multivec: None,
            ann: None,
            tvvf_manifest: None,
            compress_text: true,
        },
        GenerationWriteOptions::default(),
    )
    .unwrap();
}

#[test]
fn push_status_pull_round_trip_by_target_strings() {
    const KEY: &str = "1111dec251fa5e544784ac1af95b0ae6530cad714a2d34f8c4615740ecbf8205";
    let src = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();
    let restored = tempfile::tempdir().unwrap();
    commit_engine_generation(src.path(), KEY);
    let src_s = src.path().to_str().unwrap();
    let remote_s = remote.path().to_str().unwrap();
    let restored_s = restored.path().to_str().unwrap();

    assert_eq!(keys(src_s).unwrap(), vec![KEY.to_string()]);

    let before = status(src_s, remote_s, KEY, TransferPolicy::redacted()).unwrap();
    assert!(!before.in_sync);

    let pushed = push(src_s, remote_s, KEY, TransferPolicy::redacted()).unwrap();
    assert!(pushed.pointer_published);

    let after = status(src_s, remote_s, KEY, TransferPolicy::redacted()).unwrap();
    assert!(after.in_sync);

    verify(remote_s, KEY).unwrap();

    // Pull restores AND proves the copy opens — one operation.
    let outcome = pull(remote_s, restored_s, KEY).unwrap();
    assert!(outcome.transfer.pointer_published);
    assert_eq!(outcome.open.index_key, KEY);
}

#[test]
fn pull_into_an_unopenable_destination_fails_after_transfer() {
    // A remote holding a pointer whose base artifact is missing: the transfer
    // of the pointer alone "succeeds" at the store level, but pull must fail
    // closed because the restored copy cannot actually open. This pins that the
    // open-verification is genuinely part of pull, not a separate step.
    const KEY: &str = "idx";
    let remote = tempfile::tempdir().unwrap();
    let restored = tempfile::tempdir().unwrap();
    // Hand-build a committed pointer referencing a base, but delete the base
    // file so the artifact copy fails checksum/absence checks during pull.
    let json = store_sub(remote.path(), KEY, "g0.json", b"state", ".json-delta", &[]);
    write_json_commit(remote.path(), KEY, 1, 0, json);
    std::fs::remove_file(remote.path().join("idx.gen/g0.json")).unwrap();

    let err = pull(
        remote.path().to_str().unwrap(),
        restored.path().to_str().unwrap(),
        KEY,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ArtifactStoreError::NotFound(_) | ArtifactStoreError::Io(_)
    ));
    // The destination pointer was never published, so the failed pull left no
    // half-restored generation behind.
    let dest = lodedb_cloud_core::LocalArtifactStore::new(restored.path(), false);
    use lodedb_cloud_core::ArtifactStore;
    assert!(dest.read_pointer(KEY).unwrap().is_none());
}

#[test]
fn a_bad_target_scheme_is_rejected_up_front() {
    let err = verify("ftp://nope/x", "idx").unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Backend(_)));
}
