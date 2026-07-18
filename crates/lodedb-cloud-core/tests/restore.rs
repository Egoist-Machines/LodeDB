//! Tests for the pull-direction transfer (restore = `export_generation` with the
//! stores swapped) and confirming a restored local copy opens read-only through
//! the engine.

mod common;

use common::*;
use lodedb_core::storage::{
    write_generation_commit, GenerationCommitInput, GenerationWriteOptions,
};
use lodedb_cloud_core::{
    export_generation, verify_local_generation_opens, ArtifactStore, ArtifactStoreError,
    LocalArtifactStore, TransferPolicy,
};
use serde_json::json;
use std::fs;

#[test]
fn restore_copies_a_generation_that_opens_read_only() {
    const KEY: &str = "cc33dd44ee55ff6600112233445566778899aabbccddeeff0011223344556677";
    // The "remote" backup is just another artifact store holding a committed
    // generation the engine wrote.
    let remote = tempfile::tempdir().unwrap();
    write_generation_commit(
        remote.path(),
        GenerationCommitInput {
            index_key: KEY,
            generation: 1,
            applied_lsn: 0,
            base_epoch: 1,
            state: &engine_state(KEY),
            tvim: None,
            raw_text: None,
            lexical_tokens: None,
            multivec: None,
            ann: None,
            tvvf_manifest: None,
            compress_text: false,
        },
        GenerationWriteOptions::default(),
    )
    .unwrap();

    let local = tempfile::tempdir().unwrap();
    let source = LocalArtifactStore::new(remote.path(), false);
    let dest = LocalArtifactStore::new(local.path(), false);
    let result = export_generation(&source, &dest, KEY, TransferPolicy::full()).unwrap();
    assert!(result.pointer_published);
    assert_eq!(result.generation, 1);

    // The strongest acceptance check: the engine's own load path opens the
    // restored directory read-only.
    let report = verify_local_generation_opens(local.path(), KEY).unwrap();
    assert_eq!(report.index_key, KEY);
}

#[test]
fn pull_rebuilds_the_delta_journal_manifests() {
    // The engine's per-store journal manifest is working state the body never
    // pins, so the transfer doesn't ship it — but the O(changed) mutation
    // path requires it, so a restored copy must be WRITABLE, not just
    // readable. `pull` reconstructs each journal manifest verbatim from the
    // body's sub-manifest.
    const KEY: &str = "cc33dd44ee55ff6600112233445566778899aabbccddeeff0011223344556677";
    let remote = tempfile::tempdir().unwrap();
    commit_engine_generation(
        remote.path(),
        KEY,
        1,
        1,
        "journal-manifests",
        Some(&[("doc-1", "raw text one")]),
    );
    let remote_body = LocalArtifactStore::new(remote.path(), false)
        .read_pointer(KEY)
        .unwrap()
        .unwrap();

    let local = tempfile::tempdir().unwrap();
    lodedb_cloud_core::client_ops::pull(
        remote.path().to_str().unwrap(),
        local.path().to_str().unwrap(),
        KEY,
    )
    .unwrap();

    let gen_dir = local.path().join(format!("{KEY}.gen"));
    for (kind, suffix) in [("json", ".json-delta"), ("tvtext", ".tvtext-delta")] {
        let manifest_path = gen_dir.join(format!("g1.{kind}{suffix}")).join("manifest.json");
        let rebuilt: serde_json::Value =
            serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        assert_eq!(
            &rebuilt,
            remote_body.get(kind).unwrap(),
            "journal manifest for {kind} must equal the body sub-manifest"
        );
    }
}

#[test]
fn torn_restore_leaves_the_destination_pointer_unpublished() {
    // A corrupt source artifact must fail the restore before any pointer is
    // published, so the destination stays on its previous (here: absent) state.
    let remote = tempfile::tempdir().unwrap();
    let local = tempfile::tempdir().unwrap();
    let gen_dir = remote.path().join("idx.gen");
    fs::create_dir_all(&gen_dir).unwrap();
    fs::write(gen_dir.join("g0.json"), b"actual-bytes").unwrap();
    let corrupt_sub = json!({
        "base": {
            "file_name": "g0.json",
            "sha256": sha_hex(b"claimed-other-bytes"),
            "file_bytes": "actual-bytes".len(),
        },
        "deltas": [],
    });
    write_json_commit(remote.path(), "idx", 1, 0, corrupt_sub);

    let source = LocalArtifactStore::new(remote.path(), false);
    let dest = LocalArtifactStore::new(local.path(), false);
    let err = export_generation(&source, &dest, "idx", TransferPolicy::full()).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
    assert!(dest.read_pointer("idx").unwrap().is_none());
}

#[test]
fn restore_missing_source_generation_is_not_found() {
    let remote = tempfile::tempdir().unwrap();
    let local = tempfile::tempdir().unwrap();
    let source = LocalArtifactStore::new(remote.path(), false);
    let dest = LocalArtifactStore::new(local.path(), false);
    let err = export_generation(&source, &dest, "idx", TransferPolicy::full()).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::NotFound(_)));
}

#[test]
fn pull_refuses_while_an_engine_writer_holds_the_directory_lock() {
    // A restore is a writer of the database directory: it must contend on the
    // engine's own single-writer lock rather than interleave with a live
    // writer (whose in-memory state is based on the pointer the restore
    // replaces). The env var keeps the contention check immediate.
    const KEY: &str = "cc33dd44ee55ff6600112233445566778899aabbccddeeff0011223344556677";
    std::env::set_var("LODEDB_PERSIST_LOCK_TIMEOUT", "0");
    let remote = tempfile::tempdir().unwrap();
    commit_engine_generation(remote.path(), KEY, 1, 1, "locked", None);
    let local = tempfile::tempdir().unwrap();

    let held = lodedb_core::engine::acquire_dir_writer_lock(local.path()).unwrap();
    let err = lodedb_cloud_core::client_ops::pull(
        remote.path().to_str().unwrap(),
        local.path().to_str().unwrap(),
        KEY,
    )
    .unwrap_err();
    assert!(
        err.to_string().contains("lodedb lock"),
        "expected writer-lock contention, got: {err}"
    );
    drop(held);

    // With the writer gone the same pull succeeds.
    lodedb_cloud_core::client_ops::pull(
        remote.path().to_str().unwrap(),
        local.path().to_str().unwrap(),
        KEY,
    )
    .unwrap();
    std::env::remove_var("LODEDB_PERSIST_LOCK_TIMEOUT");
}

#[test]
fn pull_refuses_over_a_destination_wal_with_pending_records() {
    // The destination WAL's records were acknowledged against the OLD lineage;
    // replaying them onto a pulled snapshot (or silently truncating them)
    // corrupts or loses acked writes. Pull must refuse until the caller
    // checkpoints — force-pull (a sync flag) is the explicit discard.
    const KEY: &str = "cc33dd44ee55ff6600112233445566778899aabbccddeeff0011223344556677";
    let remote = tempfile::tempdir().unwrap();
    commit_engine_generation(remote.path(), KEY, 2, 1, "remote-side", None);

    let local = tempfile::tempdir().unwrap();
    commit_engine_generation(local.path(), KEY, 1, 1, "local-side", None);
    let wal = lodedb_core::storage::wal::wal_path(local.path(), KEY);
    lodedb_core::storage::wal::append_record(
        &wal,
        2,
        "add",
        serde_json::json!({"id": "pending-doc"}),
        false,
    )
    .unwrap();

    let before = LocalArtifactStore::new(local.path(), false)
        .read_pointer(KEY)
        .unwrap();
    let err = lodedb_cloud_core::client_ops::pull(
        remote.path().to_str().unwrap(),
        local.path().to_str().unwrap(),
        KEY,
    )
    .unwrap_err();
    assert!(
        matches!(err, ArtifactStoreError::PendingWal { .. }),
        "expected the pending-WAL refusal, got: {err}"
    );
    // Nothing moved: pointer unchanged, WAL records intact.
    let after = LocalArtifactStore::new(local.path(), false)
        .read_pointer(KEY)
        .unwrap();
    assert_eq!(before, after);
    assert_eq!(
        lodedb_core::storage::wal::scan_stats(&wal).unwrap().op_count,
        1
    );
}

#[test]
fn semantically_invalid_generation_never_replaces_the_destination_pointer() {
    // Byte checksums can be internally consistent while the artifact is
    // gibberish to the engine (a forged or corrupted-at-source remote): the
    // recorded digest MATCHES the broken bytes. The restore must verify the
    // candidate opens BEFORE the pointer swap, so the destination keeps its
    // previous, valid generation when the check fails.
    const KEY: &str = "cc33dd44ee55ff6600112233445566778899aabbccddeeff0011223344556677";
    let remote = tempfile::tempdir().unwrap();
    commit_engine_generation(remote.path(), KEY, 2, 2, "poisoned", None);
    // Corrupt the remote's state artifact and re-sign the manifest so every
    // byte checksum still validates.
    let base = remote.path().join(format!("{KEY}.gen/g2.json"));
    let garbage = b"not-a-state-journal";
    fs::write(&base, garbage).unwrap();
    let store = LocalArtifactStore::new(remote.path(), false);
    let mut body = store.read_pointer(KEY).unwrap().unwrap();
    body["json"]["base"]["sha256"] = serde_json::Value::String(sha_hex(garbage));
    body["json"]["base"]["file_bytes"] = serde_json::Value::from(garbage.len() as u64);
    lodedb_core::storage::commit_manifest::write_commit_manifest(
        &lodedb_core::storage::commit_manifest::commit_manifest_path(remote.path(), KEY),
        &body,
        false,
    )
    .unwrap();

    // The destination already holds a valid generation of its own.
    let local = tempfile::tempdir().unwrap();
    let valid = commit_engine_generation(local.path(), KEY, 1, 1, "still-good", None);

    let err = lodedb_cloud_core::client_ops::pull(
        remote.path().to_str().unwrap(),
        local.path().to_str().unwrap(),
        KEY,
    )
    .unwrap_err();
    assert!(
        !matches!(err, ArtifactStoreError::NotFound(_)),
        "the failure must come from the acceptance check, got: {err}"
    );
    // The destination still points at its previous generation and still opens.
    let current = LocalArtifactStore::new(local.path(), false)
        .read_pointer(KEY)
        .unwrap()
        .unwrap();
    assert_eq!(current, valid);
    verify_local_generation_opens(local.path(), KEY).unwrap();
}

#[test]
fn pull_confines_the_wal_probe_to_the_destination() {
    // The index key can arrive from CLI/remote input; a traversing spelling
    // must fail closed before the WAL check (or anything else) touches a
    // path outside the destination directory.
    let remote = tempfile::tempdir().unwrap();
    let local = tempfile::tempdir().unwrap();
    let err = lodedb_cloud_core::client_ops::pull(
        remote.path().to_str().unwrap(),
        local.path().to_str().unwrap(),
        "../escape",
    )
    .unwrap_err();
    assert!(
        matches!(err, ArtifactStoreError::Integrity(_)),
        "expected path containment to reject the key, got: {err}"
    );
}
