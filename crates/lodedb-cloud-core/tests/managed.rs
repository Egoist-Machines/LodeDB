//! Tests for the managed (`orecloud://`) transfer helpers: plan/classify,
//! sidecar trust against a caller-supplied remote identity, pull
//! requirements, and staging-directory materialisation.

mod common;

use common::*;
use lodedb_cloud_core::{
    managed_materialize, managed_plan, managed_pull_requirements, managed_record_base, snapshot_id,
    ArtifactStore, ArtifactStoreError, TransferPolicy,
};
use std::fs;
use std::path::Path;

const KEY: &str = "idx";
const REMOTE: &str = "orecloud://acme/support/default#host=https://example.test";

fn dir_str(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// Copies every artifact a committed generation pins into a staging directory
/// under its content digest — what the Python edge does with downloaded blobs.
fn stage_generation(source: &Path, body: &serde_json::Value, staging: &Path) {
    let inventory = lodedb_cloud_core::inventory_from_body(KEY, Some(body))
        .unwrap()
        .unwrap();
    fs::create_dir_all(staging).unwrap();
    for artifact in inventory.artifacts {
        let bytes = fs::read(source.join(&artifact.name)).unwrap();
        fs::write(staging.join(&artifact.sha256), bytes).unwrap();
    }
}

#[test]
fn plan_for_a_fresh_local_generation_is_local_ahead_with_full_inventory() {
    let local = tempfile::tempdir().unwrap();
    let body = commit_engine_generation(local.path(), KEY, 1, 1, "a", None);

    let plan = managed_plan(
        dir_str(local.path()),
        KEY,
        REMOTE,
        None,
        TransferPolicy::redacted(),
    )
    .unwrap();

    assert_eq!(plan.report.classification.as_deref(), Some("local_ahead"));
    let local_part = plan.local.expect("local generation is committed");
    // The redacted policy nulls tvtext/tvlex, so the plan's identity is the
    // *redacted* body's — and for a text-free commit that equals the raw one.
    assert_eq!(local_part.side.snapshot_id, snapshot_id(&body).unwrap());
    assert!(!local_part.side.has_text);
    assert!(!local_part.artifacts.is_empty());
    // The pointer document round-trips to the same identity.
    let document: serde_json::Value = serde_json::from_str(&local_part.pointer_document).unwrap();
    assert_eq!(
        document.get("body_sha256").and_then(|v| v.as_str()),
        Some(local_part.side.snapshot_id.as_str())
    );
    assert_eq!(document.get("body").unwrap(), &local_part.body);
    assert!(plan.remote.is_none());
    assert!(plan.base.is_none());
}

#[test]
fn plan_trusts_the_sidecar_only_for_the_exact_remote_identity() {
    let local = tempfile::tempdir().unwrap();
    let body = commit_engine_generation(local.path(), KEY, 1, 1, "a", None);
    managed_record_base(dir_str(local.path()), KEY, REMOTE, &body).unwrap();

    // Same identity: base trusted, and with an equal remote head the pair is
    // in sync with a current base.
    let plan = managed_plan(
        dir_str(local.path()),
        KEY,
        REMOTE,
        Some(body.clone()),
        TransferPolicy::redacted(),
    )
    .unwrap();
    assert_eq!(plan.report.classification.as_deref(), Some("in_sync"));
    assert!(plan.base.is_some());
    assert!(plan.base_is_current);

    // A different remote identity (another org, another host) must not
    // inherit that base: a remote holding different content classifies as
    // unknown (force required), never as a fast-forward.
    let other = tempfile::tempdir().unwrap();
    let other_body = commit_engine_generation(other.path(), KEY, 2, 1, "b", None);
    let plan = managed_plan(
        dir_str(local.path()),
        KEY,
        "orecloud://other/testing/default#host=https://example.test",
        Some(other_body),
        TransferPolicy::redacted(),
    )
    .unwrap();
    assert_eq!(plan.report.classification.as_deref(), Some("unknown"));
    assert!(plan.base.is_none());
}

#[test]
fn plan_classifies_a_remote_advance_as_remote_ahead() {
    // One lineage, two checkouts: the "remote" is the second checkout's
    // commit on top of the shared base.
    let local = tempfile::tempdir().unwrap();
    let base_body = commit_engine_generation(local.path(), KEY, 1, 1, "a", None);
    managed_record_base(dir_str(local.path()), KEY, REMOTE, &base_body).unwrap();

    let ahead = tempfile::tempdir().unwrap();
    let ahead_body = commit_engine_generation(ahead.path(), KEY, 2, 1, "b", None);

    let plan = managed_plan(
        dir_str(local.path()),
        KEY,
        REMOTE,
        Some(ahead_body),
        TransferPolicy::redacted(),
    )
    .unwrap();
    assert_eq!(plan.report.classification.as_deref(), Some("remote_ahead"));
    assert!(!plan.base_is_current);
}

#[test]
fn pull_requirements_shrink_to_nothing_after_materialise() {
    let source = tempfile::tempdir().unwrap();
    let body = commit_engine_generation(source.path(), KEY, 1, 1, "a", Some(&[("d1", "text")]));

    let fresh = tempfile::tempdir().unwrap();
    let needed = managed_pull_requirements(dir_str(fresh.path()), KEY, &body).unwrap();
    assert!(!needed.is_empty());

    let staging = tempfile::tempdir().unwrap();
    stage_generation(source.path(), &body, staging.path());
    let outcome = managed_materialize(
        dir_str(fresh.path()),
        KEY,
        REMOTE,
        body.clone(),
        dir_str(staging.path()),
        false,
        None,
    )
    .unwrap();
    assert!(outcome.transfer.pointer_published);
    assert_eq!(outcome.transfer.generation, 1);
    assert_eq!(outcome.open.index_key, KEY);

    // Everything staged is now local: nothing left to download, and the
    // sidecar records the base so a re-plan against the same head is in sync.
    let needed_after = managed_pull_requirements(dir_str(fresh.path()), KEY, &body).unwrap();
    assert!(needed_after.is_empty());
    let plan = managed_plan(
        dir_str(fresh.path()),
        KEY,
        REMOTE,
        Some(body),
        TransferPolicy::full(),
    )
    .unwrap();
    assert_eq!(plan.report.classification.as_deref(), Some("in_sync"));
    assert!(plan.base_is_current);
}

#[test]
fn materialise_with_a_missing_staged_blob_fails_before_any_pointer_moves() {
    let source = tempfile::tempdir().unwrap();
    let body = commit_engine_generation(source.path(), KEY, 1, 1, "a", None);

    let fresh = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap(); // deliberately empty
    let err = managed_materialize(
        dir_str(fresh.path()),
        KEY,
        REMOTE,
        body,
        dir_str(staging.path()),
        false,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::NotFound(_)));
    assert!(!fresh.path().join(format!("{KEY}.commit.json")).exists());
}

#[test]
fn materialise_with_a_corrupt_staged_blob_fails_closed() {
    let source = tempfile::tempdir().unwrap();
    let body = commit_engine_generation(source.path(), KEY, 1, 1, "a", None);

    let fresh = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    stage_generation(source.path(), &body, staging.path());
    // Corrupt one staged blob: the restore re-hashes on write and must refuse.
    let victim = fs::read_dir(staging.path())
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    fs::write(victim.path(), b"corrupted-download").unwrap();

    let err = managed_materialize(
        dir_str(fresh.path()),
        KEY,
        REMOTE,
        body,
        dir_str(staging.path()),
        false,
        None,
    )
    .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
    assert!(!fresh.path().join(format!("{KEY}.commit.json")).exists());
}

#[test]
fn materialise_refuses_when_the_local_store_moved_after_classification() {
    // The sync caller pins the local snapshot it classified: a commit landing
    // between classification and materialization must refuse (re-run the
    // sync) instead of being silently overwritten by the pull.
    let source = tempfile::tempdir().unwrap();
    let body = commit_engine_generation(source.path(), KEY, 2, 2, "remote-v2", None);

    let local = tempfile::tempdir().unwrap();
    let classified = commit_engine_generation(local.path(), KEY, 1, 1, "local-v1", None);
    let classified_id = lodedb_cloud_core::snapshot_id(&classified).unwrap();
    // The local store commits again after "classification".
    let newer = commit_engine_generation(local.path(), KEY, 3, 3, "local-v3", None);

    let staging = tempfile::tempdir().unwrap();
    stage_generation(source.path(), &body, staging.path());
    let err = managed_materialize(
        dir_str(local.path()),
        KEY,
        REMOTE,
        body.clone(),
        dir_str(staging.path()),
        false,
        Some(&classified_id),
    )
    .unwrap_err();
    assert!(
        matches!(err, ArtifactStoreError::SyncConflict { .. }),
        "expected the stale-classification refusal, got: {err}"
    );
    // The newer local commit survives untouched.
    let current = lodedb_cloud_core::LocalArtifactStore::new(local.path(), false)
        .read_pointer(KEY)
        .unwrap()
        .unwrap();
    assert_eq!(current, newer);

    // Pinning to "classified as absent" ("") refuses over any commit too.
    let err = managed_materialize(
        dir_str(local.path()),
        KEY,
        REMOTE,
        body,
        dir_str(staging.path()),
        false,
        Some(""),
    )
    .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::SyncConflict { .. }));
}
