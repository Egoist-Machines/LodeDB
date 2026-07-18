//! Tests for the sync sidecar: round-trip, fail-closed on corruption, atomic
//! rewrite, and the two placement guarantees (never mistaken for an index,
//! never eaten by the engine's epoch GC).

mod common;

use common::commit_engine_generation;
use lodedb_cloud_core::generation_inventory::list_index_keys;
use lodedb_cloud_core::sync_state::sync_state_path;
use lodedb_cloud_core::{read_sync_state, write_sync_state, SnapRef, SyncState};
use std::fs;

fn state(key: &str, generation: u64) -> SyncState {
    SyncState {
        index_key: key.to_string(),
        remote: "s3://bucket/prefix".to_string(),
        base: SnapRef {
            snapshot_id: format!("snap-{generation}"),
            logical_id: format!("logical-{generation}"),
            generation,
            text_id: Some(format!("text-{generation}")),
            lexical_id: None,
        },
        updated_unix: 1_750_000_000,
    }
}

#[test]
fn round_trips_what_was_written() {
    let dir = tempfile::tempdir().unwrap();
    let written = state("idx", 7);
    write_sync_state(dir.path(), &written).unwrap();

    let read = read_sync_state(dir.path(), "idx").unwrap();
    assert!(!read.corrupt);
    assert_eq!(read.state, Some(written));
}

#[test]
fn an_absent_sidecar_reads_as_absent_not_corrupt() {
    let dir = tempfile::tempdir().unwrap();
    let read = read_sync_state(dir.path(), "idx").unwrap();
    assert_eq!(read.state, None);
    assert!(!read.corrupt);
}

#[test]
fn a_rewrite_replaces_the_previous_base() {
    let dir = tempfile::tempdir().unwrap();
    write_sync_state(dir.path(), &state("idx", 1)).unwrap();
    write_sync_state(dir.path(), &state("idx", 2)).unwrap();

    let read = read_sync_state(dir.path(), "idx").unwrap();
    assert_eq!(read.state.unwrap().base.generation, 2);
}

#[test]
fn corruption_reads_as_absent_with_a_warning() {
    let dir = tempfile::tempdir().unwrap();
    let path = sync_state_path(dir.path(), "idx");

    // Torn write: valid JSON prefix, truncated.
    write_sync_state(dir.path(), &state("idx", 1)).unwrap();
    let intact = fs::read(&path).unwrap();
    fs::write(&path, &intact[..intact.len() / 2]).unwrap();
    let read = read_sync_state(dir.path(), "idx").unwrap();
    assert_eq!(read.state, None);
    assert!(read.corrupt);

    // Tampering: parseable JSON whose fields disagree with the checksum.
    let tampered = String::from_utf8(intact.clone())
        .unwrap()
        .replace("\"generation\": 1", "\"generation\": 9");
    assert_ne!(
        tampered.as_bytes(),
        intact.as_slice(),
        "tamper must change bytes"
    );
    fs::write(&path, tampered).unwrap();
    let read = read_sync_state(dir.path(), "idx").unwrap();
    assert_eq!(read.state, None);
    assert!(read.corrupt);

    // A sidecar copied from another index: index_key mismatch.
    write_sync_state(dir.path(), &state("other", 1)).unwrap();
    fs::copy(sync_state_path(dir.path(), "other"), &path).unwrap();
    let read = read_sync_state(dir.path(), "idx").unwrap();
    assert_eq!(read.state, None);
    assert!(read.corrupt);
}

/// The sidecar sits beside `<key>.commit.json` but must never be discovered as
/// an index: the engine (and `keys`) scan for the `.commit.json` suffix only.
#[test]
fn the_sidecar_is_not_mistaken_for_an_index() {
    let dir = tempfile::tempdir().unwrap();
    commit_engine_generation(dir.path(), "idx", 1, 1, "v1", None);
    write_sync_state(dir.path(), &state("idx", 1)).unwrap();

    assert_eq!(
        list_index_keys(dir.path()).unwrap(),
        vec!["idx".to_string()]
    );
}

/// The engine's epoch GC (default: retain 4) deletes artifacts only under
/// `<key>.gen/`; the sidecar lives next to the pointer and must survive any
/// number of commits.
#[test]
fn the_sidecar_survives_engine_epoch_gc() {
    let dir = tempfile::tempdir().unwrap();
    commit_engine_generation(dir.path(), "idx", 1, 1, "v1", None);
    write_sync_state(dir.path(), &state("idx", 1)).unwrap();

    // Enough further commits to cycle the retained-epoch window several times.
    for epoch in 2..=8u64 {
        commit_engine_generation(dir.path(), "idx", epoch, epoch, &format!("v{epoch}"), None);
    }
    // GC actually ran: the first epoch's base artifact is gone.
    assert!(!dir.path().join("idx.gen/g1.json").exists());

    let read = read_sync_state(dir.path(), "idx").unwrap();
    assert!(!read.corrupt);
    assert_eq!(read.state, Some(state("idx", 1)));
}
