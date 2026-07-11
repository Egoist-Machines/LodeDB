use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use lodedb_core::engine::CoreEngine;
use lodedb_core::stable_uint64_ids_for_chunk_ids;
use lodedb_core::types::{
    CoreAnnOptions, CoreIndexCreateOptions, CoreOpenOptions, CoreRescoreOptions, CoreVectorDocument,
};
use turbovec::IdMapIndex;

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_dir(label: &str) -> PathBuf {
    let nonce = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!(
        "lodedb_operator_{label}_{}_{}_{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        nonce,
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn open_options(path: &PathBuf) -> CoreOpenOptions {
    CoreOpenOptions {
        path: path.to_string_lossy().to_string(),
        read_only: false,
        durability: "relaxed".to_string(),
        commit_mode: "generation".to_string(),
        store_text: false,
        index_text: false,
        compress_text: false,
        chunk_character_limit: 900,
        acquire_writer_lock: false,
    }
}

fn options(ann: bool, rescore: bool) -> CoreIndexCreateOptions {
    let mut options = CoreIndexCreateOptions::native_default("default", 8, 4);
    if ann {
        options.ann = Some(CoreAnnOptions {
            algorithm: CoreAnnOptions::CLUSTER.to_string(),
            clusters: Some(4),
            nprobe: Some(1),
        });
    }
    if rescore {
        options.rescore = Some(CoreRescoreOptions {
            mode: CoreRescoreOptions::ORIGINAL.to_string(),
            dtype: Some("float32".to_string()),
            oversample: Some(2.0),
        });
    }
    options
}

fn clustered_documents() -> Vec<CoreVectorDocument> {
    let mut documents = Vec::new();
    for row in 0..32 {
        for (cluster, axis) in [0, 2, 4, 6].into_iter().enumerate() {
            let mut vector = vec![0.0; 8];
            vector[axis] = 1.0;
            vector[axis + 1] = (row + 1) as f32 * 0.001;
            documents.push(CoreVectorDocument {
                document_id: format!("cluster-{cluster}-row-{row}"),
                vector,
                metadata: BTreeMap::new(),
                text: None,
                patch_matrix: None,
            });
        }
    }
    documents
}

#[test]
fn ann_warm_and_compact_write_a_cluster_contiguous_fresh_base() {
    let path = temp_dir("compact");
    let documents = clustered_documents();
    let mut engine = CoreEngine::open(open_options(&path)).unwrap();
    engine
        .create_index_with_options(options(true, false))
        .unwrap();
    engine.upsert_vectors("default", &documents).unwrap();
    engine.persist().unwrap();
    let before = lodedb_core::storage::load_store(
        &path,
        "default",
        lodedb_core::storage::LoadOptions::default(),
    )
    .unwrap()
    .base_epoch;

    assert!(!engine.ann_cluster_resident("default").unwrap());
    assert!(engine.ann_warm("default").unwrap());
    assert!(engine.ann_cluster_resident("default").unwrap());
    engine.compact("default").unwrap();
    engine.persist().unwrap();

    let loaded = lodedb_core::storage::load_store(
        &path,
        "default",
        lodedb_core::storage::LoadOptions::default(),
    )
    .unwrap();
    assert!(loaded.base_epoch > before);
    let posting_ids = loaded
        .ann
        .as_ref()
        .unwrap()
        .postings
        .iter()
        .flatten()
        .cloned()
        .collect::<Vec<_>>();
    let persisted = IdMapIndex::load(loaded.tvim_path.as_ref().unwrap()).unwrap();
    assert_eq!(
        persisted.reconstruct_all().0,
        stable_uint64_ids_for_chunk_ids(&posting_ids)
    );
    drop(loaded);
    drop(engine);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn stats_report_rescore_and_ann_states_without_warming_them() {
    let mut exact = CoreEngine::new_in_memory();
    exact
        .create_index_with_options(options(false, false))
        .unwrap();
    let exact_stats = exact.stats("default").unwrap();
    assert!(exact_stats.rescore.is_none());
    assert!(exact_stats.ann.is_none());

    let mut engine = CoreEngine::new_in_memory();
    engine
        .create_index_with_options(options(true, true))
        .unwrap();
    engine
        .upsert_vectors("default", &clustered_documents())
        .unwrap();
    let stats = engine.stats("default").unwrap();
    let rescore = stats.rescore.unwrap();
    let ann = stats.ann.unwrap();
    assert_eq!(rescore.dtype, "float32");
    assert_eq!(rescore.oversample, 2.0);
    assert_eq!(rescore.pending_rows, 128);
    assert_eq!(rescore.sidecar_rows, None);
    assert_eq!(rescore.tombstones, None);
    assert_eq!(rescore.corrupt_rows_seen, 0);
    assert!(!rescore.reader_resident);
    assert_eq!(ann.clusters, 4);
    assert_eq!(ann.nprobe_effective, 1);
    assert!(!ann.cluster_resident);

    engine
        .set_session_overrides("default", Some(4), None)
        .unwrap();
    assert_eq!(
        engine
            .stats("default")
            .unwrap()
            .ann
            .unwrap()
            .nprobe_effective,
        4
    );
    assert!(engine.ann_warm("default").unwrap());
    assert!(
        engine
            .stats("default")
            .unwrap()
            .ann
            .unwrap()
            .cluster_resident
    );
}
