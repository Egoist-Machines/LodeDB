//! Tests for `snapshot_identity`: the ids must be the engine's own canonical
//! digests, and redaction must collapse to one logical id without collapsing
//! the snapshot id.

mod common;

use common::commit_engine_generation;
use lodedb_cloud_core::{logical_id, snapshot_id, TransferPolicy};
use serde_json::Value;

/// The id IS the `body_sha256` the engine writes in `<key>.commit.json`, not
/// a lookalike digest. Pinned against a real engine-committed store by reading
/// the pointer document raw.
#[test]
fn snapshot_id_equals_the_engine_recorded_body_sha256() {
    let dir = tempfile::tempdir().unwrap();
    let body = commit_engine_generation(dir.path(), "idx", 1, 1, "v1", None);

    let pointer_raw = std::fs::read(dir.path().join("idx.commit.json")).unwrap();
    let document: Value = serde_json::from_slice(&pointer_raw).unwrap();
    let recorded = document["body_sha256"].as_str().unwrap();

    assert_eq!(snapshot_id(&body).unwrap(), recorded);
}

#[test]
fn redaction_shares_the_logical_id_but_not_the_snapshot_id() {
    let dir = tempfile::tempdir().unwrap();
    // A payload-bearing commit: redacting it genuinely changes the body.
    let full = commit_engine_generation(
        dir.path(),
        "idx",
        1,
        1,
        "v1",
        Some(&[("doc-1", "the raw text")]),
    );
    let redacted = TransferPolicy::redacted().redact_body(&full);

    assert_eq!(
        logical_id(&full).unwrap(),
        logical_id(&redacted).unwrap(),
        "one engine commit must map to one logical id regardless of redaction"
    );
    assert_ne!(
        snapshot_id(&full).unwrap(),
        snapshot_id(&redacted).unwrap(),
        "different bytes must have different snapshot ids"
    );
    // A fully redacted body is its own logical form, the property the sync
    // classifier's Republish detection rests on.
    assert_eq!(
        snapshot_id(&redacted).unwrap(),
        logical_id(&redacted).unwrap()
    );
}

#[test]
fn different_content_has_different_ids() {
    let dir_a = tempfile::tempdir().unwrap();
    let dir_b = tempfile::tempdir().unwrap();
    let a = commit_engine_generation(dir_a.path(), "idx", 1, 1, "v1", None);
    let b = commit_engine_generation(dir_b.path(), "idx", 1, 1, "v2", None);

    assert_ne!(snapshot_id(&a).unwrap(), snapshot_id(&b).unwrap());
    assert_ne!(logical_id(&a).unwrap(), logical_id(&b).unwrap());
}
