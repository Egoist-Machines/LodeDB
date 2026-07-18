//! Tests for the read-only generation inventory and its O(changed) diff.

mod common;

use common::*;
use lodedb_cloud_core::{
    diff_inventories, inventory_committed_generation, inventory_from_body, list_index_keys,
    ArtifactStoreError,
};
use serde_json::json;
use std::fs;

#[test]
fn inventory_lists_base_and_delta_segments() {
    let dir = tempfile::tempdir().unwrap();
    let json = store_sub(
        dir.path(),
        "idx",
        "g0.json",
        b"base-state",
        ".json-delta",
        &[
            ("delta-00000000.jsd", b"delta-a"),
            ("delta-00000001.jsd", b"delta-b"),
        ],
    );
    write_json_commit(dir.path(), "idx", 1, 0, json);

    let inv = inventory_committed_generation(dir.path(), "idx")
        .unwrap()
        .unwrap();
    assert_eq!(inv.generation, 1);
    assert_eq!(inv.artifacts.len(), 3);

    let base = &inv.artifacts[0];
    assert_eq!(base.name, "idx.gen/g0.json");
    assert!(base.is_base);
    assert_eq!(base.kind, "json");
    assert_eq!(
        inv.artifacts[1].name,
        "idx.gen/g0.json.json-delta/delta-00000000.jsd"
    );
    assert!(!inv.artifacts[1].is_base);

    // Every referenced artifact exists on disk with the recorded checksum + size.
    for reference in &inv.artifacts {
        let bytes = fs::read(dir.path().join(&reference.name)).unwrap();
        assert_eq!(sha_hex(&bytes), reference.sha256);
        assert_eq!(bytes.len() as u64, reference.size_bytes);
    }
}

#[test]
fn inventory_covers_multivector_tvmv_store() {
    let dir = tempfile::tempdir().unwrap();
    let json = store_sub(dir.path(), "idx", "g0.json", b"state", ".json-delta", &[]);
    let tvmv = store_sub(
        dir.path(),
        "idx",
        "g0.tvmv",
        b"multivec-base",
        ".tvmv-delta",
        &[],
    );
    write_commit(
        dir.path(),
        "idx",
        1,
        0,
        Some(json),
        None,
        None,
        None,
        Some(tvmv),
    );

    let inv = inventory_committed_generation(dir.path(), "idx")
        .unwrap()
        .unwrap();
    assert!(
        inv.artifacts
            .iter()
            .any(|a| a.kind == "tvmv" && a.is_base && a.name == "idx.gen/g0.tvmv"),
        "multi-vector base must be inventoried so late-interaction indexes back up completely"
    );
}

#[test]
fn inventory_covers_the_ann_tvann_store() {
    // `tvann` (the persisted ANN cluster partition) is base-only and, to the
    // engine, a rebuildable cache — but a body referencing a tvann base the
    // transfer never shipped would fail byte-verification on pull, so the
    // inventory must cover it like any other store.
    let body = json!({
        "index_key": "idx",
        "generation": 1,
        "base_epoch": 1,
        "document_count": 0,
        "chunk_count": 0,
        "json": { "base": { "file_name": "g1.json", "sha256": "a", "file_bytes": 0 }, "deltas": [] },
        "tvim": null,
        "tvtext": null,
        "tvlex": null,
        "tvmv": null,
        "tvann": { "base": { "file_name": "g1.tvann", "sha256": "b", "file_bytes": 3 } },
    });
    let inv = inventory_from_body("idx", Some(&body)).unwrap().unwrap();
    assert!(inv.artifacts.iter().any(|artifact| artifact.kind == "tvann"
        && artifact.is_base
        && artifact.name == "idx.gen/g1.tvann"));
}

#[test]
fn inventory_covers_the_rescore_tvvf_store() {
    // `tvvf` (the rescore original-vector sidecar, engine 1.3.2+) is a journaled
    // {base, deltas} store: the engine refuses to open a rescore store without
    // its sidecar, so a push must ship the base AND any delta segments — a
    // pulled copy missing either would be unopenable, not merely degraded.
    let body = json!({
        "index_key": "idx",
        "generation": 2,
        "base_epoch": 1,
        "document_count": 0,
        "chunk_count": 0,
        "json": { "base": { "file_name": "g1.json", "sha256": "a", "file_bytes": 0 }, "deltas": [] },
        "tvim": null,
        "tvtext": null,
        "tvlex": null,
        "tvmv": null,
        "tvann": null,
        "tvvf": {
            "base": { "file_name": "g1.tvvf", "sha256": "b", "file_bytes": 8 },
            "deltas": [
                { "file_name": "delta-00000000.vfd", "sha256": "c", "file_bytes": 4 },
            ],
        },
    });
    let inv = inventory_from_body("idx", Some(&body)).unwrap().unwrap();
    assert!(inv.artifacts.iter().any(|artifact| artifact.kind == "tvvf"
        && artifact.is_base
        && artifact.name == "idx.gen/g1.tvvf"));
    assert!(inv.artifacts.iter().any(|artifact| artifact.kind == "tvvf"
        && !artifact.is_base
        && artifact.name == "idx.gen/g1.tvvf.tvvf-delta/delta-00000000.vfd"));
}

#[test]
fn inventory_rejects_an_unknown_store_sub_manifest() {
    // A future engine store this build does not know must refuse the transfer,
    // not silently drop its artifacts — an understated inventory ships a
    // generation whose referenced blobs were never uploaded.
    let body = json!({
        "index_key": "idx",
        "generation": 1,
        "base_epoch": 0,
        "json": { "base": { "file_name": "g0.json", "sha256": "a", "file_bytes": 0 }, "deltas": [] },
        "tvfuture": { "base": { "file_name": "g0.tvfuture", "sha256": "b", "file_bytes": 0 }, "deltas": [] },
    });
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
    assert!(err.to_string().contains("tvfuture"), "{err}");
}

#[test]
fn inventory_absent_generation_is_none() {
    let dir = tempfile::tempdir().unwrap();
    assert!(inventory_committed_generation(dir.path(), "idx")
        .unwrap()
        .is_none());
}

#[test]
fn list_index_keys_finds_committed_pointers() {
    let dir = tempfile::tempdir().unwrap();
    for key in ["beta", "alpha"] {
        let json = store_sub(dir.path(), key, "g0.json", b"state", ".json-delta", &[]);
        write_json_commit(dir.path(), key, 1, 0, json);
    }
    assert_eq!(
        list_index_keys(dir.path()).unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn diff_uploads_everything_when_remote_absent() {
    let local = inventory(vec![
        artifact("idx.gen/g0.json", "aaa", true),
        artifact("idx.gen/g0.json.json-delta/d0", "bbb", false),
    ]);
    let diff = diff_inventories(&local, None);
    assert_eq!(diff.to_upload.len(), 2);
    assert!(diff.ships_base);
}

#[test]
fn diff_uploads_nothing_when_identical() {
    let local = inventory(vec![artifact("idx.gen/g0.json", "aaa", true)]);
    let remote = local.clone();
    let diff = diff_inventories(&local, Some(&remote));
    assert!(diff.to_upload.is_empty());
    assert!(!diff.ships_base);
}

#[test]
fn diff_delta_only_onto_shared_base_does_not_ship_base() {
    let base = artifact("idx.gen/g0.json", "aaa", true);
    let remote = inventory(vec![
        base.clone(),
        artifact("idx.gen/g0.json.json-delta/d0", "bbb", false),
    ]);
    let local = inventory(vec![
        base,
        artifact("idx.gen/g0.json.json-delta/d0", "bbb", false),
        artifact("idx.gen/g0.json.json-delta/d1", "ccc", false),
    ]);
    let diff = diff_inventories(&local, Some(&remote));
    assert_eq!(diff.to_upload.len(), 1);
    assert_eq!(diff.to_upload[0].name, "idx.gen/g0.json.json-delta/d1");
    assert!(!diff.ships_base);
}

#[test]
fn diff_new_base_epoch_ships_base() {
    let remote = inventory(vec![artifact("idx.gen/g0.json", "aaa", true)]);
    let local = inventory(vec![artifact("idx.gen/g1.json", "zzz", true)]);
    let diff = diff_inventories(&local, Some(&remote));
    assert!(diff.ships_base);
    assert!(diff.to_upload.iter().any(|a| a.is_base));
}

#[test]
fn diff_ships_base_when_same_epoch_base_differs_by_checksum() {
    let remote = inventory(vec![artifact("idx.gen/g0.json", "aaa", true)]);
    let local = inventory(vec![artifact("idx.gen/g0.json", "DIFFERENT", true)]);
    let diff = diff_inventories(&local, Some(&remote));
    assert_eq!(diff.to_upload.len(), 1);
    assert!(diff.ships_base);
}

#[test]
fn inventory_rejects_a_traversing_delta_file_name() {
    // A tampered manifest whose delta file name climbs out of `<key>.gen` must be
    // rejected before any artifact path is built, so a restore cannot plant a file
    // under another index's key.
    let sub = json!({
        "base": { "file_name": "g0.json", "sha256": "x", "file_bytes": 0 },
        "deltas": [{ "file_name": "../../victim.commit.json", "sha256": "y", "file_bytes": 0, "seq": 0 }],
    });
    let body = commit_body("idx", 1, 0, sub);
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn inventory_rejects_a_traversing_base_file_name() {
    let sub = json!({
        "base": { "file_name": "../escape", "sha256": "x", "file_bytes": 0 },
        "deltas": [],
    });
    let body = commit_body("idx", 1, 0, sub);
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn inventory_rejects_a_baseless_store_manifest() {
    // A non-null store manifest must carry a journaled base; a base-less object
    // (e.g. from corruption) would otherwise silently drop the base from a backup.
    let body = commit_body("idx", 1, 0, json!({ "deltas": [] }));
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn inventory_rejects_a_non_object_delta_entry() {
    let sub = json!({
        "base": { "file_name": "g0.json", "sha256": "x", "file_bytes": 0 },
        "deltas": [null],
    });
    let body = commit_body("idx", 1, 0, sub);
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn inventory_includes_non_null_tvim_regardless_of_present_flag() {
    // The engine loads a non-null tvim manifest whatever `tvim_present` says, so the
    // inventory must too — otherwise a restore drops the tvim base it will open.
    let body = json!({
        "index_key": "idx",
        "generation": 1,
        "base_epoch": 1,
        "document_count": 0,
        "chunk_count": 0,
        "json": { "base": { "file_name": "g1.json", "sha256": "a", "file_bytes": 0 }, "deltas": [] },
        "tvim": { "base": { "file_name": "g1.tvim", "sha256": "b", "file_bytes": 0 }, "deltas": [] },
        "tvim_present": false,
        "tvtext": null,
        "tvlex": null,
        "tvmv": null,
    });
    let inv = inventory_from_body("idx", Some(&body)).unwrap().unwrap();
    assert!(inv.artifacts.iter().any(|artifact| artifact.kind == "tvim"
        && artifact.is_base
        && artifact.name == "idx.gen/g1.tvim"));
}

#[test]
fn inventory_rejects_a_base_file_name_that_disagrees_with_the_epoch() {
    // A valid basename that is not `g<base_epoch>.<kind>` is a tampered/inconsistent
    // pointer: the engine would open `g1.json` while the manifest names `g2.json`.
    let sub = json!({
        "base": { "file_name": "g2.json", "sha256": "x", "file_bytes": 0 },
        "deltas": [],
    });
    let body = commit_body("idx", 1, 1, sub);
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn inventory_rejects_a_missing_body_index_key() {
    // The engine rejects an empty/missing body index_key as corrupt, so inventory
    // must too rather than fall back to the requested key.
    let body = json!({
        "generation": 1,
        "base_epoch": 0,
        "json": { "base": { "file_name": "g0.json", "sha256": "x", "file_bytes": 0 }, "deltas": [] },
    });
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn inventory_rejects_a_non_object_store_manifest() {
    // A non-null, non-object store value is present-and-corrupt to the engine, not
    // absent, so it must fail closed instead of being silently skipped.
    let body = json!({
        "index_key": "idx",
        "generation": 1,
        "base_epoch": 0,
        "json": { "base": { "file_name": "g0.json", "sha256": "x", "file_bytes": 0 }, "deltas": [] },
        "tvtext": "bad",
    });
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}

#[test]
fn inventory_rejects_a_body_key_mismatch() {
    // The pointer file-name key ("idx") disagrees with the body's own index_key
    // ("other") — only possible via tampering, since the body checksum is valid.
    let sub = json!({
        "base": { "file_name": "g0.json", "sha256": "x", "file_bytes": 0 },
        "deltas": [],
    });
    let body = commit_body("other", 1, 0, sub);
    let err = inventory_from_body("idx", Some(&body)).unwrap_err();
    assert!(matches!(err, ArtifactStoreError::Integrity(_)));
}
