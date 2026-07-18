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
