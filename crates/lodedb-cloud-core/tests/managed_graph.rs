//! Managed materialisation of LodeGraph generations.
//!
//! A production graph body is SIDECAR-ONLY: the topology sub-manifest plus
//! the scalar fields, no `json` state-journal base (OreCloud's graph writer
//! publishes exactly this shape). These tests pin that `managed_materialize`
//! restores it end to end: the content-addressed artifact lands, the
//! engine-facing `topology.sqlite3` copy is materialised BEFORE the pointer
//! swap, the acceptance check judges the topology through the graph engine's
//! own read-only verify (not the vector engine's state-journal open), and a
//! checksum-consistent blob that is not a healthy topology still fails
//! closed with the destination untouched.

mod common;

use common::*;
use lodedb_cloud_core::{
    managed_materialize, managed_plan, managed_pull_requirements, TransferPolicy,
};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

const KEY: &str = "idx";
const REMOTE: &str = "orecloud://acme/support/default#host=https://example.test";

fn dir_str(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// Real topology bytes authored through the graph engine itself: two
/// content-free episode anchors, exactly passport's posture. The engine is
/// dropped before the bytes are read, so SQLite's close-time checkpoint has
/// folded the WAL into the single-file artifact a transfer ships.
fn topology_bytes(scratch: &Path) -> Vec<u8> {
    let store_dir = scratch.join("author");
    {
        let mut graph = lodedb_graph::TemporalGraph::open(
            &store_dir,
            lodedb_graph::GraphConfig {
                vector_dim: 8,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        for (id, occurred_at) in [("m-1", 1_000i64), ("m-2", 2_000i64)] {
            graph
                .add_episode_with_id(Some(id), "passport", "", occurred_at, json!({}), &[])
                .unwrap();
        }
    }
    fs::read(store_dir.join("topology.sqlite3")).unwrap()
}

/// The production body shape (mirrors OreCloud's
/// `worker/graph_store.py::_graph_body`): the sidecar and the scalar
/// manifest fields, nothing else.
fn sidecar_graph_body(kind: &str, topology: &[u8], generation: u64) -> Value {
    json!({
        "index_key": KEY,
        "generation": generation,
        "base_epoch": generation,
        kind: {
            "base": {
                "file_name": "topology.sqlite3",
                "sha256": sha_hex(topology),
                "file_bytes": topology.len(),
            },
            "deltas": [],
        },
    })
}

fn stage_blob(staging: &Path, bytes: &[u8]) {
    fs::create_dir_all(staging).unwrap();
    fs::write(staging.join(sha_hex(bytes)), bytes).unwrap();
}

#[test]
fn materialise_restores_a_sidecar_only_graph_body() {
    let scratch = tempfile::tempdir().unwrap();
    let topology = topology_bytes(scratch.path());
    let body = sidecar_graph_body("gtopo", &topology, 1);

    let fresh = tempfile::tempdir().unwrap();
    let needed = managed_pull_requirements(dir_str(fresh.path()), KEY, &body).unwrap();
    assert_eq!(needed.len(), 1);

    let staging = tempfile::tempdir().unwrap();
    stage_blob(staging.path(), &topology);
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
    assert_eq!(outcome.open.document_count, 2); // the two episode anchors
    assert_eq!(outcome.open.chunk_count, 0);

    // The engine-facing copy landed under the body-recorded name with the
    // exact transferred bytes, beside the content-addressed artifact.
    let restored = fs::read(fresh.path().join("topology.sqlite3")).unwrap();
    assert_eq!(restored, topology);
    let artifact = fresh
        .path()
        .join(format!("{KEY}.gen"))
        .join(format!("{}.gtopo", sha_hex(&topology)));
    assert!(artifact.exists());

    // And the pulled copy opens through the graph engine itself.
    let graph = lodedb_graph::TemporalGraph::open(
        fresh.path(),
        lodedb_graph::GraphConfig {
            vector_dim: 8,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    drop(graph);

    // Nothing left to download, and a re-plan classifies in sync.
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
}

#[test]
fn a_text_bearing_topology_restores_the_same_way() {
    let scratch = tempfile::tempdir().unwrap();
    let topology = topology_bytes(scratch.path());
    let body = sidecar_graph_body("gtopotext", &topology, 1);

    let fresh = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    stage_blob(staging.path(), &topology);
    let outcome = managed_materialize(
        dir_str(fresh.path()),
        KEY,
        REMOTE,
        body,
        dir_str(staging.path()),
        false,
        None,
    )
    .unwrap();
    assert!(outcome.transfer.pointer_published);
    assert!(fresh.path().join("topology.sqlite3").exists());
}

#[test]
fn a_checksum_consistent_non_topology_blob_fails_before_any_pointer_moves() {
    // The blob's digest matches the body exactly, so every checksum gate
    // passes; only the graph verify-open can reject it.
    let garbage = b"not a sqlite database at all".to_vec();
    let body = sidecar_graph_body("gtopo", &garbage, 1);

    let fresh = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    stage_blob(staging.path(), &garbage);
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
    assert!(
        err.to_string().contains("verify-open"),
        "unexpected error: {err}"
    );
    // Nothing was published: no engine copy, no committed pointer.
    assert!(!fresh.path().join("topology.sqlite3").exists());
    assert!(!fresh.path().join(format!("{KEY}.commit.json")).exists());
}

#[test]
fn a_body_without_a_recorded_file_name_defaults_the_engine_copy() {
    let scratch = tempfile::tempdir().unwrap();
    let topology = topology_bytes(scratch.path());
    let mut body = sidecar_graph_body("gtopo", &topology, 1);
    body["gtopo"]["base"]
        .as_object_mut()
        .unwrap()
        .remove("file_name");

    let fresh = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    stage_blob(staging.path(), &topology);
    managed_materialize(
        dir_str(fresh.path()),
        KEY,
        REMOTE,
        body,
        dir_str(staging.path()),
        false,
        None,
    )
    .unwrap();
    assert!(fresh.path().join("topology.sqlite3").exists());
}

#[test]
fn repeat_materialise_is_idempotent() {
    let scratch = tempfile::tempdir().unwrap();
    let topology = topology_bytes(scratch.path());
    let body = sidecar_graph_body("gtopo", &topology, 1);

    let fresh = tempfile::tempdir().unwrap();
    let staging = tempfile::tempdir().unwrap();
    stage_blob(staging.path(), &topology);
    for expect_publish in [true, false] {
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
        assert_eq!(outcome.transfer.pointer_published, expect_publish);
        assert_eq!(
            fs::read(fresh.path().join("topology.sqlite3")).unwrap(),
            topology
        );
    }
}
