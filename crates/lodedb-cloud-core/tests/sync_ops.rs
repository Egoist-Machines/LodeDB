//! End-to-end tests for the `sync` verb: two working directories sharing one
//! remote, exercising every classification the way real use produces them,
//! plus `contains` on both store backends.

mod common;

use common::commit_engine_generation;
use object_store::memory::InMemory;
use lodedb_cloud_core::client_ops::{status, sync, SyncForce};
use lodedb_cloud_core::{ArtifactStore, ArtifactStoreError, ObjectArtifactStore, TransferPolicy};
use std::path::Path;
use std::sync::Arc;

const KEY: &str = "idx";

fn run_sync(dir: &Path, remote: &Path, force: SyncForce) -> lodedb_cloud_core::SyncOutcome {
    sync(
        dir.to_str().unwrap(),
        remote.to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        force,
    )
    .unwrap()
}

#[test]
fn the_two_clone_lifecycle() {
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();

    // Fresh push: no base, remote absent -> local_ahead -> push.
    commit_engine_generation(dir1.path(), KEY, 1, 1, "v1", None);
    let outcome = run_sync(dir1.path(), remote.path(), SyncForce::None);
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("local_ahead", "push")
    );

    // Nothing changed: in sync, no transfer.
    let outcome = run_sync(dir1.path(), remote.path(), SyncForce::None);
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("in_sync", "none")
    );
    assert_eq!(outcome.transfer, None);

    // A new local commit fast-forwards the remote.
    commit_engine_generation(dir1.path(), KEY, 2, 2, "v2", None);
    let report = status(
        dir1.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
    )
    .unwrap();
    assert_eq!(report.classification.as_deref(), Some("local_ahead"));
    assert!(report.sidecar_present);
    assert_eq!(report.base_generation, Some(1));
    let outcome = run_sync(dir1.path(), remote.path(), SyncForce::None);
    assert_eq!(outcome.action, "push");

    // A second clone pulls: local absent -> remote_ahead -> pull + verify-open.
    let outcome = run_sync(dir2.path(), remote.path(), SyncForce::None);
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("remote_ahead", "pull")
    );
    assert!(
        outcome.open.is_some(),
        "a pull proves the restored copy opens"
    );

    // The second clone commits and pushes; the first is now behind and pulls.
    commit_engine_generation(dir2.path(), KEY, 3, 3, "v3", None);
    let outcome = run_sync(dir2.path(), remote.path(), SyncForce::None);
    assert_eq!(outcome.action, "push");
    let outcome = run_sync(dir1.path(), remote.path(), SyncForce::None);
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("remote_ahead", "pull")
    );

    // Both commit independently: diverged, sync refuses without force. (The
    // two lineages use different base epochs here: colliding artifact *names*
    // across divergent lineages is the fork-collision case the Phase-2
    // content-addressed remote layout absorbs; a dumb remote refuses it at the
    // immutable-artifact check instead.)
    commit_engine_generation(dir1.path(), KEY, 4, 4, "v4-dir1", None);
    commit_engine_generation(dir2.path(), KEY, 4, 5, "v4-dir2", None);
    run_sync(dir2.path(), remote.path(), SyncForce::None); // dir2 pushes first
    let err = sync(
        dir1.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        SyncForce::None,
    )
    .unwrap_err();
    match &err {
        ArtifactStoreError::SyncConflict {
            classification,
            hint,
        } => {
            assert_eq!(classification, "diverged");
            assert!(hint.contains("--force-push") && hint.contains("--force-pull"));
        }
        other => panic!("expected SyncConflict, got {other:?}"),
    }

    // Force-push keeps dir1's copy; dir2 is then a plain fast-forward pull.
    let outcome = run_sync(dir1.path(), remote.path(), SyncForce::Push);
    assert_eq!((outcome.action.as_str(), outcome.forced), ("push", true));
    let outcome = run_sync(dir2.path(), remote.path(), SyncForce::None);
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("remote_ahead", "pull")
    );
}

#[test]
fn a_full_push_after_a_redacted_one_is_a_republish_not_a_divergence() {
    let dir = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();
    commit_engine_generation(dir.path(), KEY, 1, 1, "v1", Some(&[("doc-1", "raw text")]));

    // Redacted push first (the default posture).
    let outcome = run_sync(dir.path(), remote.path(), SyncForce::None);
    assert_eq!(outcome.action, "push");
    let remote_store = lodedb_cloud_core::LocalArtifactStore::new(remote.path(), false);
    let published = remote_store.read_pointer(KEY).unwrap().unwrap();
    assert!(
        published["tvtext"].is_null(),
        "redacted push must not ship text"
    );

    // The same commit pushed with text is an upgrade of the same lineage.
    let outcome = sync(
        dir.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::full(),
        SyncForce::None,
    )
    .unwrap();
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("republish", "push")
    );
    let published = remote_store.read_pointer(KEY).unwrap().unwrap();
    assert!(
        !published["tvtext"].is_null(),
        "republish must upgrade the remote to full"
    );

    // And it converges: the full policy now matches the full remote.
    let outcome = sync(
        dir.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::full(),
        SyncForce::None,
    )
    .unwrap();
    assert_eq!(outcome.action, "none");
}

#[test]
fn an_untrusted_base_requires_force() {
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();

    // Two ends that never synced (no sidecar) holding different content.
    // (Different base epochs; see the fork-collision note above.)
    commit_engine_generation(dir1.path(), KEY, 1, 1, "v1", None);
    run_sync(dir1.path(), remote.path(), SyncForce::None);
    commit_engine_generation(dir2.path(), KEY, 1, 2, "different", None);

    let err = sync(
        dir2.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        SyncForce::None,
    )
    .unwrap_err();
    match &err {
        ArtifactStoreError::SyncConflict { classification, .. } => {
            assert_eq!(classification, "unknown");
        }
        other => panic!("expected SyncConflict, got {other:?}"),
    }

    // Force-pull resolves toward the remote, opens it, and records a base.
    let outcome = run_sync(dir2.path(), remote.path(), SyncForce::Pull);
    assert_eq!((outcome.action.as_str(), outcome.forced), ("pull", true));
    assert!(outcome.open.is_some());
    let outcome = run_sync(dir2.path(), remote.path(), SyncForce::None);
    assert_eq!(outcome.action, "none");
}

/// The sidecar base is a claim about one specific pair of ends: pointing the
/// same directory at a *different* remote must not inherit the old remote's
/// history (that would classify unrelated content as a fast-forward and pull
/// it over the local copy without force).
#[test]
fn a_base_recorded_against_another_remote_is_not_trusted() {
    let dir = tempfile::tempdir().unwrap();
    let other_dir = tempfile::tempdir().unwrap();
    let remote_a = tempfile::tempdir().unwrap();
    let remote_b = tempfile::tempdir().unwrap();

    // dir syncs with remote A; remote B holds an unrelated lineage.
    commit_engine_generation(dir.path(), KEY, 1, 1, "v1", None);
    run_sync(dir.path(), remote_a.path(), SyncForce::None);
    commit_engine_generation(other_dir.path(), KEY, 2, 2, "unrelated", None);
    run_sync(other_dir.path(), remote_b.path(), SyncForce::None);

    // Against remote B, dir's local copy still equals remote A's base, but
    // that base must not be trusted here: no fast-forward pull of B's content.
    let err = sync(
        dir.path().to_str().unwrap(),
        remote_b.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        SyncForce::None,
    )
    .unwrap_err();
    match &err {
        ArtifactStoreError::SyncConflict { classification, .. } => {
            assert_eq!(classification, "unknown");
        }
        other => panic!("expected SyncConflict, got {other:?}"),
    }

    // Status agrees: the sidecar exists but is not trusted for this remote.
    let report = status(
        dir.path().to_str().unwrap(),
        remote_b.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
    )
    .unwrap();
    assert!(!report.sidecar_present);
    assert!(!report.sidecar_corrupt);
    assert_eq!(report.classification.as_deref(), Some("unknown"));

    // An explicit force re-homes the directory onto remote B.
    let outcome = run_sync(dir.path(), remote_b.path(), SyncForce::Pull);
    assert_eq!(outcome.action, "pull");
    let outcome = run_sync(dir.path(), remote_b.path(), SyncForce::None);
    assert_eq!(outcome.action, "none");
}

/// A remote that carries text but not lexical is *partially* redacted: a sync
/// whose policy adds the missing store must republish, not report in-sync
/// (regression: `Republish` originally fired only for fully-redacted remotes).
#[test]
fn a_partial_redaction_upgrade_republishes() {
    let dir = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();
    common::commit_engine_generation_with_lexical(
        dir.path(),
        KEY,
        1,
        1,
        "v1",
        Some(&[("doc-1", "raw text")]),
        true,
    );

    // First sync ships text but not lexical.
    let text_only = TransferPolicy {
        include_text: true,
        include_lexical: false,
    };
    let outcome = sync(
        dir.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        text_only,
        SyncForce::None,
    )
    .unwrap();
    assert_eq!(outcome.action, "push");
    let remote_store = lodedb_cloud_core::LocalArtifactStore::new(remote.path(), false);
    let published = remote_store.read_pointer(KEY).unwrap().unwrap();
    assert!(!published["tvtext"].is_null());
    assert!(published["tvlex"].is_null());

    // Full policy on the same commit: the missing lexical store must ship.
    let outcome = sync(
        dir.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::full(),
        SyncForce::None,
    )
    .unwrap();
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("republish", "push")
    );
    let published = remote_store.read_pointer(KEY).unwrap().unwrap();
    assert!(!published["tvtext"].is_null());
    assert!(!published["tvlex"].is_null());

    // And the reverse (policy narrower than the remote) stays a no-op.
    let outcome = sync(
        dir.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        text_only,
        SyncForce::None,
    )
    .unwrap();
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("in_sync", "none")
    );
}

/// The Phase-0 fork-collision limit, pinned: two lineages diverging from one
/// base reuse the same artifact file names with different bytes, and force
/// cannot resolve that against a name-addressed remote. Sync fails closed
/// with a recovery hint instead of overwriting an immutable artifact. (The
/// managed content-addressed layout of a later milestone absorbs this shape.)
#[test]
fn a_same_name_fork_fails_closed_with_a_recovery_hint_even_when_forced() {
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();

    // Both clones share base gen 1, then each commits gen 2 at epoch 2, the
    // natural fork shape: same artifact names, different bytes.
    commit_engine_generation(dir1.path(), KEY, 1, 1, "v1", None);
    run_sync(dir1.path(), remote.path(), SyncForce::None);
    run_sync(dir2.path(), remote.path(), SyncForce::None);
    commit_engine_generation(dir1.path(), KEY, 2, 2, "v2-dir1", None);
    let dir2_body = commit_engine_generation(dir2.path(), KEY, 2, 2, "v2-dir2", None);
    run_sync(dir2.path(), remote.path(), SyncForce::None); // dir2 publishes first

    // Unforced sync refuses as diverged; forced push hits the immutability
    // invariant and fails closed with guidance, leaving the remote intact.
    let err = sync(
        dir1.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        SyncForce::None,
    )
    .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::SyncConflict { .. }));

    let err = sync(
        dir1.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        SyncForce::Push,
    )
    .unwrap_err();
    match &err {
        ArtifactStoreError::Integrity(message) => {
            assert!(message.contains("divergent lineages"), "got: {message}");
            assert!(message.contains("fresh"), "got: {message}");
        }
        other => panic!("expected Integrity with recovery hint, got {other:?}"),
    }
    // The failed force moved no pointer: the remote still holds dir2's fork
    // (as its redacted push published it).
    let remote_store = lodedb_cloud_core::LocalArtifactStore::new(remote.path(), false);
    let published = remote_store.read_pointer(KEY).unwrap().unwrap();
    assert_eq!(
        published,
        TransferPolicy::redacted().redact_body(&dir2_body)
    );
}

/// An unpublished payload upgrade is not discarded by a fast-forward: if the
/// base was recorded redacted, the local copy now syncs with text, and the
/// remote has meanwhile advanced, both sides have moved and force is required.
#[test]
fn an_unpublished_payload_upgrade_blocks_a_fast_forward_pull() {
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();

    // dir1 publishes commit A redacted (the base records the redacted view).
    commit_engine_generation(dir1.path(), KEY, 1, 1, "v1", Some(&[("doc-1", "raw text")]));
    run_sync(dir1.path(), remote.path(), SyncForce::None);

    // A second clone advances the remote to commit B.
    run_sync(dir2.path(), remote.path(), SyncForce::None);
    commit_engine_generation(dir2.path(), KEY, 2, 2, "v2", None);
    run_sync(dir2.path(), remote.path(), SyncForce::None);

    // dir1 now asks to sync WITH text: its policy view carries payload the
    // base never published. Pulling B would silently drop that text.
    let err = sync(
        dir1.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::full(),
        SyncForce::None,
    )
    .unwrap_err();
    match &err {
        ArtifactStoreError::SyncConflict { classification, .. } => {
            assert_eq!(classification, "diverged");
        }
        other => panic!("expected SyncConflict, got {other:?}"),
    }

    // The redacted view, by contrast, adds nothing over the base: plain pull.
    let outcome = run_sync(dir1.path(), remote.path(), SyncForce::None);
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("remote_ahead", "pull")
    );
}

/// The remote is trusted by canonical identity, not raw spelling: two
/// spellings of one directory share a sidecar base, so a re-spelled target
/// does not spuriously demand force.
#[test]
fn equivalent_remote_spellings_share_the_base() {
    let dir = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();
    commit_engine_generation(dir.path(), KEY, 1, 1, "v1", None);
    run_sync(dir.path(), remote.path(), SyncForce::None);
    // Advance locally so a mistrusted base would classify unknown (both ends
    // present and different), not merely in_sync.
    commit_engine_generation(dir.path(), KEY, 2, 2, "v2", None);

    // Same directory, different spelling: a redundant `.` segment.
    let respelled = remote
        .path()
        .parent()
        .unwrap()
        .join(".")
        .join(remote.path().file_name().unwrap());
    let outcome = sync(
        dir.path().to_str().unwrap(),
        respelled.to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        SyncForce::None,
    )
    .unwrap();
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("local_ahead", "push")
    );
}

/// An in-sync no-op repairs a stale base (e.g. a prior transfer whose sidecar
/// write failed), so lineage recovers without a force.
#[test]
fn an_in_sync_no_op_repairs_a_stale_base() {
    let dir = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();
    commit_engine_generation(dir.path(), KEY, 1, 1, "v1", None);
    run_sync(dir.path(), remote.path(), SyncForce::None);
    let good = lodedb_cloud_core::read_sync_state(dir.path(), KEY)
        .unwrap()
        .state
        .unwrap();

    // Advance both ends together, then restore the OLD sidecar, the state a
    // crashed post-transfer sidecar write leaves behind.
    commit_engine_generation(dir.path(), KEY, 2, 2, "v2", None);
    run_sync(dir.path(), remote.path(), SyncForce::None);
    lodedb_cloud_core::write_sync_state(dir.path(), &good).unwrap();

    // In sync with a stale base: the no-op refreshes it...
    let outcome = run_sync(dir.path(), remote.path(), SyncForce::None);
    assert_eq!(outcome.action, "none");
    let refreshed = lodedb_cloud_core::read_sync_state(dir.path(), KEY)
        .unwrap()
        .state
        .unwrap();
    assert_eq!(refreshed.base.generation, 2);
}

/// An in-sync no-op against a remote with no trusted base records one, so
/// later runs classify as fast-forwards instead of unknown.
#[test]
fn an_in_sync_no_op_establishes_a_base() {
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();

    // dir1 publishes; dir2 obtains the identical content via pull (which
    // records a base), then loses its sidecar (e.g. a copy that dropped it).
    commit_engine_generation(dir1.path(), KEY, 1, 1, "v1", None);
    run_sync(dir1.path(), remote.path(), SyncForce::None);
    run_sync(dir2.path(), remote.path(), SyncForce::None);
    std::fs::remove_file(lodedb_cloud_core::sync_state::sync_state_path(dir2.path(), KEY)).unwrap();

    // In sync, no trusted base: the no-op records one...
    let outcome = run_sync(dir2.path(), remote.path(), SyncForce::None);
    assert_eq!(outcome.action, "none");

    // ...so a remote advance is now a plain fast-forward, not unknown.
    commit_engine_generation(dir1.path(), KEY, 2, 2, "v2", None);
    run_sync(dir1.path(), remote.path(), SyncForce::None);
    let outcome = run_sync(dir2.path(), remote.path(), SyncForce::None);
    assert_eq!(
        (outcome.classification.as_str(), outcome.action.as_str()),
        ("remote_ahead", "pull")
    );
}

#[test]
fn a_corrupt_sidecar_is_surfaced_and_treated_as_no_base() {
    let dir = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();
    commit_engine_generation(dir.path(), KEY, 1, 1, "v1", None);
    run_sync(dir.path(), remote.path(), SyncForce::None);

    // Corrupt the sidecar; the two ends are still identical, so sync is a
    // harmless no-op, but the corruption is reported, by status too.
    let sidecar = lodedb_cloud_core::sync_state::sync_state_path(dir.path(), KEY);
    std::fs::write(&sidecar, b"{ not json").unwrap();
    let report = status(
        dir.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
    )
    .unwrap();
    assert!(!report.sidecar_present);
    assert!(report.sidecar_corrupt);
    let outcome = run_sync(dir.path(), remote.path(), SyncForce::None);
    assert_eq!(outcome.action, "none");
    assert!(outcome.sidecar_corrupt);

    // The in-sync no-op re-established a trusted base over the corrupt file.
    let report = status(
        dir.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
    )
    .unwrap();
    assert!(report.sidecar_present);
    assert!(!report.sidecar_corrupt);

    // When the corruption actually causes a refusal (ends differ, base
    // untrusted), the error itself carries the diagnosis; the CLI's warning
    // path never runs on an error.
    commit_engine_generation(dir.path(), KEY, 2, 2, "v2", None);
    run_sync(dir.path(), remote.path(), SyncForce::None); // remote at v2
    commit_engine_generation(dir.path(), KEY, 3, 3, "v3", None); // local ahead
    std::fs::write(&sidecar, b"{ not json").unwrap();
    let err = sync(
        dir.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        SyncForce::None,
    )
    .unwrap_err();
    match &err {
        ArtifactStoreError::SyncConflict {
            classification,
            hint,
        } => {
            assert_eq!(classification, "unknown");
            assert!(hint.contains("corrupt"), "got hint: {hint}");
        }
        other => panic!("expected SyncConflict, got {other:?}"),
    }
}

#[test]
fn sync_rejects_a_url_local_end() {
    let err = sync(
        "s3://bucket/prefix",
        "/tmp/whatever",
        KEY,
        TransferPolicy::redacted(),
        SyncForce::None,
    )
    .unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Backend(_)));
}

#[test]
fn contains_on_both_backends() {
    // Local store.
    let dir = tempfile::tempdir().unwrap();
    commit_engine_generation(dir.path(), KEY, 1, 1, "v1", None);
    let local = lodedb_cloud_core::LocalArtifactStore::new(dir.path(), false);
    assert!(local.contains("idx.gen/g1.json").unwrap());
    assert!(!local.contains("idx.gen/no-such-artifact").unwrap());

    // Object store (HEAD-based override).
    let object = ObjectArtifactStore::new(Arc::new(InMemory::new()), "tenant").unwrap();
    let payload = b"payload";
    let digest = common::sha_hex(payload);
    object
        .write_bytes_if_absent("blob", payload, &digest)
        .unwrap();
    assert!(object.contains("blob").unwrap());
    assert!(!object.contains("missing").unwrap());
}

#[test]
fn sync_pull_refuses_a_pending_wal_and_force_pull_discards_it() {
    // dir2 is behind the remote AND holds acknowledged-but-uncheckpointed WAL
    // records: a plain sync must refuse (those records were acked against the
    // old lineage), and --force-pull (the explicit "keep the remote copy"
    // decision) discards them along with the local lineage.
    let dir1 = tempfile::tempdir().unwrap();
    let dir2 = tempfile::tempdir().unwrap();
    let remote = tempfile::tempdir().unwrap();

    commit_engine_generation(dir1.path(), KEY, 1, 1, "v1", None);
    run_sync(dir1.path(), remote.path(), SyncForce::None);
    run_sync(dir2.path(), remote.path(), SyncForce::None); // dir2 clones v1
    commit_engine_generation(dir1.path(), KEY, 2, 2, "v2", None);
    run_sync(dir1.path(), remote.path(), SyncForce::None); // remote moves ahead

    let wal = lodedb_core::storage::wal::wal_path(dir2.path(), KEY);
    lodedb_core::storage::wal::append_record(
        &wal,
        2,
        "add",
        serde_json::json!({"id": "acked-but-uncheckpointed"}),
        false,
    )
    .unwrap();

    let err = sync(
        dir2.path().to_str().unwrap(),
        remote.path().to_str().unwrap(),
        KEY,
        TransferPolicy::redacted(),
        SyncForce::None,
    )
    .unwrap_err();
    assert!(
        matches!(err, ArtifactStoreError::PendingWal { .. }),
        "expected the pending-WAL refusal, got: {err}"
    );
    assert_eq!(
        lodedb_core::storage::wal::scan_stats(&wal).unwrap().op_count,
        1,
        "a refusal must not touch the WAL"
    );

    let outcome = run_sync(dir2.path(), remote.path(), SyncForce::Pull);
    assert_eq!((outcome.action.as_str(), outcome.forced), ("pull", true));
    assert_eq!(
        lodedb_core::storage::wal::scan_stats(&wal).unwrap().op_count,
        0,
        "force-pull discards the pending records with the local lineage"
    );
}
