//! Tests for `export_generation`: O(changed) copy + all-or-nothing pointer swap,
//! exercised against both hand-built and genuine engine-written generations.

mod common;

use common::*;
use lodedb_core::storage::{
    wal, write_generation_commit, GenerationCommitInput, GenerationWriteOptions,
};
use lodedb_cloud_core::{
    export_generation, inventory_committed_generation, verify_local_generation_opens,
    ArtifactStore, ArtifactStoreError, LocalArtifactStore, TransferPolicy,
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;

#[test]
fn export_copies_generation_and_publishes_pointer() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let json = store_sub(
        src.path(),
        "idx",
        "g0.json",
        b"base-state",
        ".json-delta",
        &[("delta-00000000.jsd", b"delta-a")],
    );
    write_json_commit(src.path(), "idx", 1, 0, json);

    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    let result = export_generation(&source, &dest, "idx", TransferPolicy::full()).unwrap();

    assert_eq!(result.artifacts_written, 2);
    assert_eq!(result.artifacts_skipped, 0);
    assert!(result.pointer_published);
    assert_eq!(
        dest.read_pointer("idx").unwrap(),
        source.read_pointer("idx").unwrap()
    );
    for name in [
        "idx.gen/g0.json",
        "idx.gen/g0.json.json-delta/delta-00000000.jsd",
    ] {
        assert_eq!(
            dest.read_bytes(name).unwrap(),
            source.read_bytes(name).unwrap()
        );
    }
}

#[test]
fn export_is_idempotent() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let json = store_sub(src.path(), "idx", "g0.json", b"state", ".json-delta", &[]);
    write_json_commit(src.path(), "idx", 1, 0, json);
    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);

    export_generation(&source, &dest, "idx", TransferPolicy::full()).unwrap();
    let again = export_generation(&source, &dest, "idx", TransferPolicy::full()).unwrap();
    assert_eq!(again.artifacts_written, 0);
    assert_eq!(again.bytes_written, 0);
    assert!(!again.pointer_published);
}

#[test]
fn export_missing_source_generation_is_not_found() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    let err = export_generation(&source, &dest, "idx", TransferPolicy::full()).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::NotFound(_)));
}

#[test]
fn export_rejects_corrupt_source_artifact_before_publishing() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    // Write a base file whose recorded checksum does not match its bytes.
    let gen_dir = src.path().join("idx.gen");
    fs::create_dir_all(&gen_dir).unwrap();
    fs::write(gen_dir.join("g0.json"), b"actual-bytes").unwrap();
    let corrupt_sub = json!({
        "base": {
            "file_name": "g0.json",
            "sha256": sha_hex(b"claimed-different-bytes"),
            "file_bytes": "actual-bytes".len(),
        },
        "deltas": [],
    });
    write_json_commit(src.path(), "idx", 1, 0, corrupt_sub);

    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    let err = export_generation(&source, &dest, "idx", TransferPolicy::full()).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
    // The destination pointer was never published.
    assert!(dest.read_pointer("idx").unwrap().is_none());
}

#[test]
fn exports_a_generation_written_by_the_engine() {
    const KEY: &str = "6f78dec251fa5e544784ac1af95b0ae6530cad714a2d34f8c4615740ecbf8205";
    let src = tempfile::tempdir().unwrap();
    let state = engine_state(KEY);
    write_generation_commit(
        src.path(),
        GenerationCommitInput {
            index_key: KEY,
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

    let inv = inventory_committed_generation(src.path(), KEY)
        .unwrap()
        .unwrap();
    assert!(inv.artifacts.iter().any(|a| a.kind == "json" && a.is_base));
    for reference in &inv.artifacts {
        let bytes = fs::read(src.path().join(&reference.name)).unwrap();
        assert_eq!(
            sha_hex(&bytes),
            reference.sha256,
            "checksum matches engine record"
        );
    }

    let dst = tempfile::tempdir().unwrap();
    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    let result = export_generation(&source, &dest, KEY, TransferPolicy::full()).unwrap();
    assert!(result.pointer_published);
    assert_eq!(
        dest.read_pointer(KEY).unwrap(),
        source.read_pointer(KEY).unwrap()
    );
}

#[test]
fn redacted_push_nulls_text_and_skips_its_artifacts() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let json = store_sub(src.path(), "idx", "g0.json", b"state", ".json-delta", &[]);
    let text = store_sub(
        src.path(),
        "idx",
        "g0.tvtext",
        b"secret document text",
        ".tvtext-delta",
        &[],
    );
    // Source commits both a redacted (json) and a payload-bearing (tvtext) store.
    write_commit(
        src.path(),
        "idx",
        1,
        0,
        Some(json),
        None,
        Some(text),
        None,
        None,
    );

    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    let result = export_generation(&source, &dest, "idx", TransferPolicy::redacted()).unwrap();

    // Only the json base shipped; the text base was neither counted nor uploaded.
    assert_eq!(result.artifacts_written, 1);
    assert!(matches!(
        dest.read_bytes("idx.gen/g0.tvtext").unwrap_err(),
        ArtifactStoreError::NotFound(_)
    ));
    assert!(dest.read_bytes("idx.gen/g0.json").is_ok());
    // The published body genuinely omits the text store.
    let body = dest.read_pointer("idx").unwrap().unwrap();
    assert_eq!(body.get("tvtext"), Some(&Value::Null));
    assert!(body.get("json").is_some_and(|json| !json.is_null()));
}

#[test]
fn full_push_ships_text_that_redacted_push_omits() {
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let json = store_sub(src.path(), "idx", "g0.json", b"state", ".json-delta", &[]);
    let text = store_sub(
        src.path(),
        "idx",
        "g0.tvtext",
        b"text",
        ".tvtext-delta",
        &[],
    );
    write_commit(
        src.path(),
        "idx",
        1,
        0,
        Some(json),
        None,
        Some(text),
        None,
        None,
    );

    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    export_generation(&source, &dest, "idx", TransferPolicy::full()).unwrap();

    assert!(dest.read_bytes("idx.gen/g0.tvtext").is_ok());
    assert_eq!(
        dest.read_pointer("idx").unwrap(),
        source.read_pointer("idx").unwrap()
    );
}

#[test]
fn redacted_push_of_engine_generation_opens_without_text() {
    const KEY: &str = "aa11bb22cc33dd44ee55ff6600112233445566778899aabbccddeeff00112233";
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    let mut raw_text = BTreeMap::new();
    raw_text.insert("doc-1".to_string(), "sensitive body text".to_string());
    write_generation_commit(
        src.path(),
        GenerationCommitInput {
            index_key: KEY,
            generation: 1,
            applied_lsn: 0,
            base_epoch: 1,
            state: &engine_state(KEY),
            tvim: None,
            raw_text: Some(&raw_text),
            lexical_tokens: None,
            multivec: None,
            ann: None,
            tvvf_manifest: None,
            compress_text: false,
        },
        GenerationWriteOptions::default(),
    )
    .unwrap();

    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    export_generation(&source, &dest, KEY, TransferPolicy::redacted()).unwrap();

    // The restored copy carries no text base at all, yet opens read-only through
    // the engine's own load path.
    assert!(!dst.path().join(format!("{KEY}.gen/g1.tvtext")).exists());
    let report = verify_local_generation_opens(dst.path(), KEY).unwrap();
    assert_eq!(report.index_key, KEY);
}

#[test]
fn export_excludes_uncommitted_wal_writes() {
    const KEY: &str = "bb22cc33dd44ee55ff6600112233445566778899aabbccddeeff001122334455";
    let src = tempfile::tempdir().unwrap();
    let dst = tempfile::tempdir().unwrap();
    write_generation_commit(
        src.path(),
        GenerationCommitInput {
            index_key: KEY,
            generation: 3,
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
    // An uncommitted WAL write sits alongside the committed generation.
    let wal_file = wal::wal_path(src.path(), KEY);
    wal::append_record(&wal_file, 4, "add", json!({"id": "pending"}), false).unwrap();
    assert!(wal::scan_stats(&wal_file).unwrap().op_count > 0);

    let source = LocalArtifactStore::new(src.path(), false);
    let dest = LocalArtifactStore::new(dst.path(), false);
    let result = export_generation(&source, &dest, KEY, TransferPolicy::full()).unwrap();

    // Only the committed generation shipped: no `.wal` reached the destination,
    // and the published generation is the committed one.
    assert_eq!(result.generation, 3);
    assert!(!wal::wal_path(dst.path(), KEY).exists());
    assert_eq!(
        dest.read_pointer(KEY)
            .unwrap()
            .unwrap()
            .get("generation")
            .and_then(Value::as_u64),
        Some(3)
    );
}
