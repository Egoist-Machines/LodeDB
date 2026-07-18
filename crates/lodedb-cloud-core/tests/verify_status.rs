//! Tests for `verify_generation` (checksum re-hash), `compare_generations`
//! (push status from two inventories), and `status_for_push` (store-level,
//! policy-aware status).

mod common;

use common::*;
use lodedb_cloud_core::{
    compare_generations, export_generation, status_for_push, verify_generation, ArtifactStoreError,
    GenerationInventory, LocalArtifactStore, TransferPolicy,
};
use serde_json::json;
use std::fs;

#[test]
fn verify_accepts_an_intact_generation() {
    let dir = tempfile::tempdir().unwrap();
    let json = store_sub(
        dir.path(),
        "idx",
        "g0.json",
        b"base-state",
        ".json-delta",
        &[("delta-00000000.jsd", b"delta-a")],
    );
    write_json_commit(dir.path(), "idx", 1, 0, json);

    let store = LocalArtifactStore::new(dir.path(), false);
    let report = verify_generation(&store, "idx").unwrap();
    assert_eq!(report.generation, 1);
    assert_eq!(report.artifacts_verified, 2);
    assert_eq!(
        report.bytes_verified,
        (b"base-state".len() + b"delta-a".len()) as u64
    );
}

#[test]
fn verify_detects_a_tampered_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let json = store_sub(
        dir.path(),
        "idx",
        "g0.json",
        b"original",
        ".json-delta",
        &[],
    );
    write_json_commit(dir.path(), "idx", 1, 0, json);
    // Corrupt the base file on disk after it was committed: its bytes no longer
    // match the checksum the manifest records.
    fs::write(dir.path().join("idx.gen/g0.json"), b"tampered").unwrap();

    let store = LocalArtifactStore::new(dir.path(), false);
    let err = verify_generation(&store, "idx").unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn verify_missing_generation_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let store = LocalArtifactStore::new(dir.path(), false);
    let err = verify_generation(&store, "idx").unwrap_err();
    assert!(matches!(err, ArtifactStoreError::NotFound(_)));
}

#[test]
fn status_reports_a_full_upload_against_an_empty_remote() {
    let local = inventory(vec![
        artifact("idx.gen/g0.json", "sha-json", true),
        artifact("idx.gen/g0.json.json-delta/d0", "sha-d0", false),
    ]);
    let report = compare_generations("idx", Some(&local), None);
    assert_eq!(report.artifacts_to_upload, 2);
    assert!(report.ships_base);
    assert!(!report.in_sync);
    assert_eq!(report.local_generation, Some(1));
    assert_eq!(report.remote_generation, None);
}

#[test]
fn status_reports_in_sync_when_remote_holds_everything() {
    let local = inventory(vec![artifact("idx.gen/g0.json", "sha-json", true)]);
    let report = compare_generations("idx", Some(&local), Some(&local));
    assert_eq!(report.artifacts_to_upload, 0);
    assert_eq!(report.bytes_to_upload, 0);
    assert!(report.in_sync);
}

#[test]
fn status_with_no_local_generation_is_in_sync() {
    let remote = inventory(vec![artifact("idx.gen/g0.json", "sha-json", true)]);
    let report = compare_generations("idx", None, Some(&remote));
    assert!(report.in_sync);
    assert_eq!(report.local_generation, None);
    assert_eq!(report.remote_generation, Some(1));
    assert_eq!(report.artifacts_to_upload, 0);
}

#[test]
fn status_for_push_reflects_the_redacted_push_it_describes() {
    // Source commits a json store AND a payload-bearing text store.
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let json_sub = store_sub(src.path(), "idx", "g0.json", b"state", ".json-delta", &[]);
    let text_sub = store_sub(
        src.path(),
        "idx",
        "g0.tvtext",
        b"secret text",
        ".tvtext-delta",
        &[],
    );
    write_commit(
        src.path(),
        "idx",
        1,
        0,
        Some(json_sub),
        None,
        Some(text_sub),
        None,
        None,
    );
    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);

    // Redacted status counts only the json base — the text artifact would not
    // ship, so it must not be reported as pending upload.
    let redacted = status_for_push(&source, &dest, "idx", TransferPolicy::redacted()).unwrap();
    assert_eq!(redacted.artifacts_to_upload, 1);
    assert!(!redacted.in_sync);
    // A full-policy status against the same empty remote counts both.
    let full = status_for_push(&source, &dest, "idx", TransferPolicy::full()).unwrap();
    assert_eq!(full.artifacts_to_upload, 2);

    // After the redacted push actually runs, redacted status is in sync (the
    // remote holds exactly the redacted body) while full status is not (a full
    // push would still ship the text artifact and republish the pointer).
    export_generation(&source, &dest, "idx", TransferPolicy::redacted()).unwrap();
    let redacted = status_for_push(&source, &dest, "idx", TransferPolicy::redacted()).unwrap();
    assert!(redacted.in_sync);
    assert_eq!(redacted.artifacts_to_upload, 0);
    let full = status_for_push(&source, &dest, "idx", TransferPolicy::full()).unwrap();
    assert!(!full.in_sync);
    assert_eq!(full.artifacts_to_upload, 1);
}

#[test]
fn status_for_push_with_nothing_committed_anywhere_is_in_sync() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    let report = status_for_push(&source, &dest, "idx", TransferPolicy::redacted()).unwrap();
    assert!(report.in_sync);
    assert_eq!(report.local_generation, None);
    assert_eq!(report.remote_generation, None);
}

#[test]
fn status_is_not_in_sync_when_bodies_differ_despite_matching_artifacts() {
    // A redacted local against a previously-full remote: the shared json artifact
    // matches (nothing to upload), but the bodies differ (remote still references
    // text). A push would republish the pointer, so this is NOT in sync.
    let shared = vec![artifact("idx.gen/g0.json", "sha-json", true)];
    let local = GenerationInventory {
        index_key: "idx".into(),
        generation: 1,
        base_epoch: 0,
        document_count: 0,
        chunk_count: 0,
        root_body: json!({ "generation": 1, "json": {"present": true}, "tvtext": null }),
        artifacts: shared.clone(),
    };
    let remote = GenerationInventory {
        root_body: json!({ "generation": 1, "json": {"present": true}, "tvtext": {"present": true} }),
        ..local.clone()
    };
    let report = compare_generations("idx", Some(&local), Some(&remote));
    assert_eq!(report.artifacts_to_upload, 0);
    assert!(!report.in_sync);
}
