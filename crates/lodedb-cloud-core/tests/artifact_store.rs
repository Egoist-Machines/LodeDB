//! Tests for `LocalArtifactStore`: immutable byte I/O and root-pointer CAS.

mod common;

use common::*;
use lodedb_cloud_core::{ArtifactStore, ArtifactStoreError, LocalArtifactStore};
use serde_json::Value;

#[test]
fn write_and_read_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    let data = b"generation-artifact-bytes";
    store
        .write_bytes_if_absent("idx.gen/g0.json", data, &sha_hex(data))
        .unwrap();
    assert_eq!(store.read_bytes("idx.gen/g0.json").unwrap(), data.to_vec());
}

#[test]
fn checksum_mismatch_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    let err = store
        .write_bytes_if_absent("idx.gen/g0.json", b"data", &sha_hex(b"other"))
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn rewrite_identical_bytes_is_noop() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    let data = b"immutable";
    let sha = sha_hex(data);
    store
        .write_bytes_if_absent("idx.gen/g0.json", data, &sha)
        .unwrap();
    store
        .write_bytes_if_absent("idx.gen/g0.json", data, &sha)
        .unwrap();
    assert_eq!(store.read_bytes("idx.gen/g0.json").unwrap(), data.to_vec());
}

#[test]
fn rewrite_different_bytes_conflicts() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    store
        .write_bytes_if_absent("idx.gen/g0.json", b"first", &sha_hex(b"first"))
        .unwrap();
    let err = store
        .write_bytes_if_absent("idx.gen/g0.json", b"second", &sha_hex(b"second"))
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
    // The original bytes are preserved — never overwritten in place.
    assert_eq!(
        store.read_bytes("idx.gen/g0.json").unwrap(),
        b"first".to_vec()
    );
}

#[test]
fn read_missing_artifact_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    let err = store.read_bytes("idx.gen/absent.json").unwrap_err();
    assert!(matches!(err, ArtifactStoreError::NotFound(_)));
}

#[test]
fn read_pointer_returns_committed_body_or_none() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    assert!(store.read_pointer("idx").unwrap().is_none());

    let json = store_sub(dir.path(), "idx", "g0.json", b"state", ".json-delta", &[]);
    let body = write_json_commit(dir.path(), "idx", 1, 0, json);
    assert_eq!(store.read_pointer("idx").unwrap(), Some(body));
}

#[test]
fn compare_and_swap_enforces_body_precondition() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    let json = store_sub(dir.path(), "idx", "g0.json", b"state", ".json-delta", &[]);
    let current = write_json_commit(dir.path(), "idx", 1, 0, json.clone());
    let next = commit_body("idx", 2, 0, json.clone());

    // Expecting-absent-but-present -> conflict.
    let err = store
        .compare_and_swap_pointer("idx", None, &next)
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::PointerConflict { .. }));

    // Expecting a body that is not the committed one -> conflict (even though a
    // number-only check on its generation could have been made to pass).
    let wrong = commit_body("idx", 9, 0, json);
    let err = store
        .compare_and_swap_pointer("idx", Some(&wrong), &next)
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::PointerConflict { .. }));

    // Expecting the exact committed body -> swap succeeds and advances the pointer.
    store
        .compare_and_swap_pointer("idx", Some(&current), &next)
        .unwrap();
    assert_eq!(
        store
            .read_pointer("idx")
            .unwrap()
            .unwrap()
            .get("generation")
            .and_then(Value::as_u64),
        Some(2)
    );
}

#[test]
fn compare_and_swap_rejects_same_generation_different_body() {
    // The ABA guard: a body sharing the committed generation *number* but carrying
    // different content must NOT satisfy the precondition — a number is not a
    // version token. A generation-only check would wrongly let this swap through.
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    let json = store_sub(dir.path(), "idx", "g0.json", b"state", ".json-delta", &[]);
    write_json_commit(dir.path(), "idx", 1, 0, json);

    // A divergent lineage's body that also calls itself generation 1.
    let other_gen1 = commit_body(
        "idx",
        1,
        0,
        serde_json::json!({
            "base": { "file_name": "g0.json", "sha256": "deadbeef", "file_bytes": 5 },
            "deltas": [],
        }),
    );
    let next = commit_body(
        "idx",
        2,
        0,
        serde_json::json!({
            "base": { "file_name": "g0.json", "sha256": "cafe", "file_bytes": 4 },
            "deltas": [],
        }),
    );
    let err = store
        .compare_and_swap_pointer("idx", Some(&other_gen1), &next)
        .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::PointerConflict { .. }));
}

#[test]
fn read_pointer_on_nonexistent_root_is_none() {
    // A store bound to a not-yet-created directory reads as empty rather than
    // erroring — the precondition for exporting into a fresh backup target.
    let parent = tempfile::tempdir().unwrap();
    let root = parent.path().join("not-created-yet");
    let store = LocalArtifactStore::new(&root, false);
    assert!(store.read_pointer("idx").unwrap().is_none());
}
