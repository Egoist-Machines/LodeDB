use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use lodedb_core::error::CoreErrorCode;
use lodedb_core::storage::commit_manifest::{
    base_json_path, base_tvim_path, base_tvlex_path, base_tvtext_path, commit_manifest_path,
    read_commit_manifest,
};
use lodedb_core::storage::{
    fixture_root, lexical_store, load_store, state_journal, text_store, tvim_delta, wal,
    LoadOptions, StoreLayout,
};
use serde_json::Value;

const INDEX_KEY: &str = "6f78dec251fa5e544784ac1af95b0ae6530cad714a2d34f8c4615740ecbf8205";

fn persisted_fixture(name: &str) -> PathBuf {
    fixture_root(&format!("tests/fixtures/persisted/{name}"))
}

#[test]
fn committed_generation_fixtures_load() {
    let generation = load_store(
        persisted_fixture("v0_4_generation"),
        INDEX_KEY,
        LoadOptions::default(),
    )
    .unwrap();
    assert_eq!(generation.layout, StoreLayout::Generation);
    assert_eq!(generation.generation, 4);
    assert_eq!(generation.base_epoch, 4);
    assert_eq!(generation.document_count(), 3);
    assert_eq!(generation.chunk_count(), 3);
    assert_eq!(generation.raw_text.len(), 3);
    assert!(generation.lexical_tokens.is_empty());

    let store_text = load_store(
        persisted_fixture("v0_4_store_text"),
        INDEX_KEY,
        LoadOptions::default(),
    )
    .unwrap();
    assert_eq!(store_text.document_count(), 3);
    assert_eq!(store_text.raw_text.len(), 3);

    let index_text = load_store(
        persisted_fixture("v0_4_index_text"),
        INDEX_KEY,
        LoadOptions::default(),
    )
    .unwrap();
    assert_eq!(index_text.document_count(), 3);
    assert_eq!(index_text.raw_text.len(), 3);
    assert_eq!(index_text.lexical_tokens.len(), 3);
    assert_eq!(
        index_text.lexical_tokens["doc-alpha"][0],
        [
            "alpha", "launch", "notes", "mention", "error", "code", "e-1001", "and", "a", "blue",
            "widget"
        ]
    );
}

#[test]
fn legacy_top_level_fixture_loads() {
    let legacy = load_store(
        persisted_fixture("v0_4_legacy_top_level_json"),
        INDEX_KEY,
        LoadOptions::default(),
    )
    .unwrap();
    assert_eq!(legacy.layout, StoreLayout::LegacyTopLevelJson);
    assert_eq!(legacy.document_count(), 3);
    assert_eq!(legacy.chunk_count(), 3);
    assert!(legacy.raw_text.is_empty());
}

#[test]
fn read_only_generation_load_ignores_wal_tail() {
    let wal_fixture = persisted_fixture("v0_4_wal");
    let read_only = load_store(&wal_fixture, INDEX_KEY, LoadOptions::default()).unwrap();
    assert_eq!(read_only.document_count(), 0);
    assert!(read_only.wal_records.is_empty());

    let with_wal = load_store(
        &wal_fixture,
        INDEX_KEY,
        LoadOptions {
            read_only: false,
            read_wal: true,
        },
    )
    .unwrap();
    assert_eq!(with_wal.document_count(), 0);
    assert_eq!(with_wal.wal_records.len(), 3);
    assert!(with_wal
        .wal_records
        .iter()
        .all(|record| record.op == "upsert_documents"));
}

#[test]
fn superseded_epoch_sidecar_deltas_validate_and_replay() {
    let fixture = persisted_fixture("v0_4_index_text");
    let manifest = read_commit_manifest(&commit_manifest_path(&fixture, INDEX_KEY))
        .unwrap()
        .unwrap();
    let epoch = 2;
    let g2_json_manifest = read_json_value(
        &base_json_path(&fixture, INDEX_KEY, epoch)
            .with_file_name("g2.json.json-delta/manifest.json"),
    );
    let mut state = state_journal::read_base_payload(
        &base_json_path(&fixture, INDEX_KEY, epoch),
        Some(&g2_json_manifest),
    )
    .unwrap();
    state_journal::replay_onto_payload(
        &mut state,
        &base_json_path(&fixture, INDEX_KEY, epoch),
        &g2_json_manifest,
    )
    .unwrap();
    assert_eq!(state["document_hashes"].as_object().unwrap().len(), 2);
    assert_eq!(state["chunks"].as_array().unwrap().len(), 2);

    let g2_text_manifest = read_json_value(
        &base_tvtext_path(&fixture, INDEX_KEY, epoch)
            .with_file_name("g2.tvtext.tvtext-delta/manifest.json"),
    );
    let texts = text_store::load(
        &base_tvtext_path(&fixture, INDEX_KEY, epoch),
        Some(&g2_text_manifest),
    )
    .unwrap();
    assert_eq!(texts.len(), 2);

    let g2_lex_manifest = read_json_value(
        &base_tvlex_path(&fixture, INDEX_KEY, epoch)
            .with_file_name("g2.tvlex.tvlex-delta/manifest.json"),
    );
    let tokens = lexical_store::load(
        &base_tvlex_path(&fixture, INDEX_KEY, epoch),
        Some(&g2_lex_manifest),
    )
    .unwrap();
    assert_eq!(tokens.len(), 2);

    let g2_tvim_manifest = read_json_value(
        &base_tvim_path(&fixture, INDEX_KEY, epoch)
            .with_file_name("g2.tvim.tvim-delta/manifest.json"),
    );
    let summary = tvim_delta::validate(
        &base_tvim_path(&fixture, INDEX_KEY, epoch),
        Some(&g2_tvim_manifest),
    )
    .unwrap();
    assert_eq!(summary.segment_count, 1);
    assert_eq!(summary.upsert_rows, 1);

    assert_eq!(manifest.index_key(), INDEX_KEY);
}

#[test]
fn corrupt_commit_manifest_rejects() {
    let temp = copy_fixture("v0_4_generation");
    let path = commit_manifest_path(&temp, INDEX_KEY);
    let mut document = read_json_value(&path);
    document["body"]["document_count"] = Value::from(99);
    fs::write(&path, serde_json::to_string(&document).unwrap()).unwrap();
    let error = load_store(&temp, INDEX_KEY, LoadOptions::default()).unwrap_err();
    assert_eq!(error.code(), CoreErrorCode::CorruptStore);
    fs::remove_dir_all(temp).unwrap();
}

#[test]
fn corrupt_sidecar_rejects() {
    let temp = copy_fixture("v0_4_generation");
    let manifest = read_commit_manifest(&commit_manifest_path(&temp, INDEX_KEY))
        .unwrap()
        .unwrap();
    let base_epoch = manifest.base_epoch();
    let text_path = base_tvtext_path(&temp, INDEX_KEY, base_epoch);
    let mut text = fs::read_to_string(&text_path).unwrap();
    text.push(' ');
    fs::write(&text_path, text).unwrap();
    let error = load_store(&temp, INDEX_KEY, LoadOptions::default()).unwrap_err();
    assert_eq!(error.code(), CoreErrorCode::CorruptStore);
    fs::remove_dir_all(temp).unwrap();
}

#[test]
fn wal_corrupt_interior_frame_rejects_and_torn_tail_drops() {
    let fixture = persisted_fixture("v0_4_wal");
    let wal_path = wal::wal_path(&fixture, INDEX_KEY);
    let original = fs::read(&wal_path).unwrap();
    assert_eq!(wal::read_records(&wal_path).unwrap().len(), 3);

    let corrupt_path = unique_temp_dir("wal_corrupt").join("bad.wal");
    fs::create_dir_all(corrupt_path.parent().unwrap()).unwrap();
    let mut corrupt = original.clone();
    corrupt[32] ^= 0x80;
    fs::write(&corrupt_path, corrupt).unwrap();
    let error = wal::read_records(&corrupt_path).unwrap_err();
    assert_eq!(error.code(), CoreErrorCode::CorruptStore);
    fs::remove_dir_all(corrupt_path.parent().unwrap()).unwrap();

    let torn_path = unique_temp_dir("wal_torn").join("torn.wal");
    fs::create_dir_all(torn_path.parent().unwrap()).unwrap();
    let mut torn = original;
    torn.truncate(torn.len() - 11);
    fs::write(&torn_path, torn).unwrap();
    assert_eq!(wal::read_records(&torn_path).unwrap().len(), 2);
    fs::remove_dir_all(torn_path.parent().unwrap()).unwrap();
}

fn read_json_value(path: &Path) -> Value {
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn copy_fixture(name: &str) -> PathBuf {
    let source = persisted_fixture(name);
    let target = unique_temp_dir(name);
    copy_dir_all(&source, &target);
    target
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "lodedb-core-{label}-{}-{nanos}",
        std::process::id()
    ))
}

fn copy_dir_all(source: &Path, target: &Path) {
    fs::create_dir_all(target).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let destination = target.join(entry.file_name());
        if path.is_dir() {
            copy_dir_all(&path, &destination);
        } else {
            fs::copy(&path, &destination).unwrap();
        }
    }
}
