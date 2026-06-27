use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use lodedb_core::engine::CoreEngine;
use lodedb_core::types::{
    CoreDocument, CoreIndexCreateOptions, CoreOpenOptions, CoreVectorDocument,
};
use lodedb_core::vector::index::CoreVectorChunk;
use lodedb_core::vector::turbovec::TurboVecNativeIndex;
use serde_json::{json, Value};

const INDEX_KEY: &str = "6f78dec251fa5e544784ac1af95b0ae6530cad714a2d34f8c4615740ecbf8205";
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn metadata(entries: &[(&str, &str)]) -> BTreeMap<String, String> {
    entries
        .iter()
        .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
        .collect()
}

fn doc(id: &str, axis: usize, metadata: BTreeMap<String, String>) -> CoreVectorDocument {
    let mut vector = vec![0.0; 8];
    vector[axis] = 1.0;
    CoreVectorDocument {
        document_id: id.to_string(),
        vector,
        metadata,
        text: None,
    }
}

fn text_doc(id: &str, text: &str, metadata: BTreeMap<String, String>) -> CoreDocument {
    CoreDocument {
        document_id: id.to_string(),
        text: text.to_string(),
        metadata,
    }
}

fn open_options(path: &Path, read_only: bool, commit_mode: &str) -> CoreOpenOptions {
    CoreOpenOptions {
        path: path.to_string_lossy().to_string(),
        read_only,
        durability: "relaxed".to_string(),
        commit_mode: commit_mode.to_string(),
        store_text: true,
        index_text: true,
        chunk_character_limit: 900,
    }
}

fn seeded_engine() -> CoreEngine {
    let mut engine = CoreEngine::new_in_memory();
    engine.create_index("default", 8, 4).unwrap();
    engine
        .upsert_vectors(
            "default",
            &[
                doc("a", 0, metadata(&[("topic", "ops"), ("year", "2024")])),
                doc("b", 1, metadata(&[("topic", "ml"), ("year", "2025")])),
                doc("c", 2, metadata(&[("topic", "ops"), ("year", "2026")])),
            ],
        )
        .unwrap();
    engine
}

#[test]
fn upsert_vectors_is_atomic_when_a_later_row_is_invalid() {
    let mut engine = CoreEngine::new_in_memory();
    engine.create_index("default", 8, 4).unwrap();
    // A valid first row followed by a wrong-dimension row: the whole batch must
    // fail without leaving the first row visible in documents/stats/search.
    let bad = CoreVectorDocument {
        document_id: "b".to_string(),
        vector: vec![0.0; 4],
        metadata: metadata(&[]),
        text: None,
    };
    let result = engine.upsert_vectors("default", &[doc("a", 0, metadata(&[])), bad]);
    assert!(result.is_err(), "wrong-dim batch must error");

    let stats = engine.stats("default").unwrap();
    assert_eq!(stats.document_count, 0, "no row may survive a failed batch");
    let hits = engine
        .query_vector(
            "default",
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            5,
            None,
        )
        .unwrap();
    assert!(hits.hits.is_empty(), "failed batch must not be searchable");

    // A subsequent valid batch still works (the index was left clean).
    engine
        .upsert_vectors("default", &[doc("a", 0, metadata(&[]))])
        .unwrap();
    assert_eq!(engine.stats("default").unwrap().document_count, 1);
}

#[test]
fn vector_query_returns_ranked_hits_with_metadata() {
    let engine = seeded_engine();
    let hits = engine
        .query_vector(
            "default",
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            3,
            None,
        )
        .unwrap();

    assert_eq!(hits.total_considered, 3);
    assert_eq!(hits.hits[0].document_id, "b");
    assert_eq!(hits.hits[0].chunk_id, "b");
    assert_eq!(hits.hits[0].metadata["topic"], "ml");
}

#[test]
fn metadata_and_document_id_filters_match_python_semantics() {
    let engine = seeded_engine();
    let by_metadata = engine
        .query_vector(
            "default",
            &[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            5,
            Some(&json!({"metadata": {"topic": "ops", "year": {"$gte": 2025}}})),
        )
        .unwrap();
    assert_eq!(
        by_metadata
            .hits
            .iter()
            .map(|hit| hit.document_id.as_str())
            .collect::<Vec<_>>(),
        ["c"]
    );

    let by_ids = engine
        .query_vector(
            "default",
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            5,
            Some(&json!({"document_ids": ["a", "b"], "metadata": {"topic": {"$ne": "ml"}}})),
        )
        .unwrap();
    assert_eq!(by_ids.hits[0].document_id, "a");
    assert_eq!(by_ids.total_considered, 1);
}

#[test]
fn batch_queries_preserve_order() {
    let engine = seeded_engine();
    let rows = engine
        .query_vectors_batch(
            "default",
            &[
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            1,
            None,
        )
        .unwrap();
    assert_eq!(rows[0].hits[0].document_id, "a");
    assert_eq!(rows[1].hits[0].document_id, "c");
}

#[test]
fn update_delete_and_stats_are_metrics_only() {
    let mut engine = seeded_engine();
    engine
        .update_document_payload(
            "default",
            "b",
            Some(metadata(&[("topic", "ops"), ("year", "2027")])),
            Some(Some("retained outside redacted stats".to_string())),
        )
        .unwrap();
    let filtered = engine
        .query_vector(
            "default",
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            3,
            Some(&json!({"topic": "ops", "year": {"$gte": 2027}})),
        )
        .unwrap();
    assert_eq!(filtered.hits[0].document_id, "b");

    let deleted = engine
        .delete_documents("default", &["a".to_string(), "missing".to_string()])
        .unwrap();
    assert_eq!(deleted.documents_deleted, 1);
    let stats = engine.stats("default").unwrap();
    assert_eq!(stats.document_count, 2);
    assert_eq!(stats.chunk_count, 2);
    assert_eq!(stats.delete_count, 1);
    assert_eq!(stats.deleted_chunk_count, 1);
    assert!(!stats.raw_payload_text_present);
}

#[test]
fn text_prepare_apply_keeps_embeddings_in_binding_layer() {
    let mut engine = CoreEngine::new_in_memory();
    engine.create_index("text", 8, 4).unwrap();
    let plan = engine
        .prepare_text_upsert(
            "text",
            &[
                text_doc("doc-a", "fault code E1234", metadata(&[("topic", "ops")])),
                text_doc(
                    "doc-b",
                    "quarterly revenue",
                    metadata(&[("topic", "finance")]),
                ),
            ],
            true,
            true,
            100,
        )
        .unwrap();

    assert_eq!(plan.documents.len(), 2);
    assert_eq!(plan.chunks_to_embed.len(), 2);
    assert_eq!(plan.chunks_to_embed[0].text, "fault code E1234");
    assert_eq!(
        plan.documents[0].chunks[0].tokens,
        ["fault", "code", "e1234"]
    );

    let applied = engine
        .apply_text_upsert(
            &plan,
            &[
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            12.5,
        )
        .unwrap();
    assert_eq!(applied.embedded_chunks, 2);
    assert_eq!(applied.reused_chunks, 0);
    assert_eq!(applied.embedding_time_ms, 12.5);
    assert_eq!(engine.stats("text").unwrap().chunk_count, 2);
    assert_eq!(
        engine.document_token_lists("text").unwrap()["doc-a"][0],
        ["fault", "code", "e1234"]
    );
    assert_eq!(
        engine
            .get_document_text("text", "doc-a")
            .unwrap()
            .as_deref(),
        Some("fault code E1234")
    );
    assert_eq!(
        engine
            .get_document_text("text", "missing")
            .unwrap()
            .as_deref(),
        None
    );
    assert_eq!(
        engine
            .get_document_texts("text", &["doc-a".to_string(), "missing".to_string()])
            .unwrap()
            .get("doc-a")
            .map(String::as_str),
        Some("fault code E1234")
    );
    let record = engine
        .get_document("text", "doc-a")
        .unwrap()
        .expect("doc-a should exist");
    assert_eq!(record["document_id"], "doc-a");
    assert_eq!(record["metadata"], json!({"topic": "ops"}));
    assert_eq!(record["chunk_count"], 1);
    assert!(!record["content_hash"].as_str().unwrap().is_empty());
    assert!(!record.as_object().unwrap().contains_key("text"));
    let listed = engine
        .list_documents("text", Some(&json!({"metadata": {"topic": "ops"}})))
        .unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0]["document_id"], "doc-a");

    let query_plan = engine.prepare_query_text("E1234", "vector").unwrap();
    assert!(query_plan.requires_embedding);
    assert_eq!(query_plan.query_tokens, ["e1234"]);
    let hits = engine
        .search_embedded_text(
            "text",
            &query_plan,
            Some(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            2,
            Some(&json!({"metadata": {"topic": "ops"}})),
        )
        .unwrap();
    assert_eq!(hits.hits[0].document_id, "doc-a");

    let lexical_plan = engine.prepare_query_text("revenue", "lexical").unwrap();
    assert!(!lexical_plan.requires_embedding);
    let lexical_hits = engine
        .search_embedded_text("text", &lexical_plan, None, 2, None)
        .unwrap();
    assert_eq!(lexical_hits.hits[0].document_id, "doc-b");

    let hybrid_plan = engine.prepare_query_text("revenue", "hybrid").unwrap();
    assert!(hybrid_plan.requires_embedding);
    let hybrid_hits = engine
        .search_embedded_text(
            "text",
            &hybrid_plan,
            Some(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            2,
            None,
        )
        .unwrap();
    assert_eq!(hybrid_hits.hits[0].document_id, "doc-b");

    let reuse_plan = engine
        .prepare_text_upsert(
            "text",
            &[text_doc(
                "doc-a",
                "fault code E1234",
                metadata(&[("topic", "ops")]),
            )],
            true,
            true,
            100,
        )
        .unwrap();
    assert!(reuse_plan.chunks_to_embed.is_empty());
    let reused = engine.apply_text_upsert(&reuse_plan, &[], 0.0).unwrap();
    assert_eq!(reused.embedded_chunks, 0);
    assert_eq!(reused.reused_chunks, 1);
}

#[test]
fn stale_text_plan_is_rejected() {
    let mut engine = CoreEngine::new_in_memory();
    engine.create_index("text", 8, 4).unwrap();
    let plan = engine
        .prepare_text_upsert(
            "text",
            &[text_doc("doc-a", "alpha", metadata(&[]))],
            true,
            false,
            100,
        )
        .unwrap();
    engine
        .upsert_vectors("text", &[doc("other", 7, metadata(&[]))])
        .unwrap();

    let error = engine
        .apply_text_upsert(&plan, &[vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 1.0)
        .unwrap_err();
    assert_eq!(error.code().as_str(), "PLAN_STALE");
}

#[test]
fn persistent_engine_opens_mutates_persists_and_reopens_readonly() {
    let path = unique_temp_dir("core_persistent");
    let mut engine = CoreEngine::open(open_options(&path, false, "generation")).unwrap();
    engine.create_index("default", 8, 4).unwrap();
    engine
        .upsert_vectors(
            "default",
            &[
                doc("persist-a", 0, metadata(&[("kind", "alpha")])),
                doc("persist-b", 1, metadata(&[("kind", "beta")])),
            ],
        )
        .unwrap();
    assert_eq!(
        engine
            .query_vector(
                "default",
                &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1,
                None,
            )
            .unwrap()
            .hits[0]
            .document_id,
        "persist-a"
    );
    engine.persist().unwrap();
    engine.close().unwrap();

    let mut readonly =
        CoreEngine::open_readonly(&path, open_options(&path, true, "generation")).unwrap();
    assert_eq!(readonly.stats("default").unwrap().document_count, 2);
    assert_eq!(
        readonly
            .query_vector(
                "default",
                &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1,
                None,
            )
            .unwrap()
            .hits[0]
            .document_id,
        "persist-a"
    );
    assert!(readonly.persist().is_err());
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn persistent_engine_writes_python_compatible_vector_metadata() {
    let path = unique_temp_dir("core_python_compatible");
    let mut engine = CoreEngine::open(open_options(&path, false, "generation")).unwrap();
    engine
        .create_index_with_options(CoreIndexCreateOptions {
            index_id: "default".to_string(),
            index_key: INDEX_KEY.to_string(),
            client_id_hash: INDEX_KEY.to_string(),
            name: "lodedb-local".to_string(),
            model: "external".to_string(),
            provider: "external".to_string(),
            task: "vector-only".to_string(),
            route_profile: "vector-only".to_string(),
            storage_profile: "turbovec_direct".to_string(),
            vector_dim: 8,
            bit_width: 4,
        })
        .unwrap();
    engine
        .upsert_vectors(
            "default",
            &[
                doc("persist-a", 0, metadata(&[("kind", "alpha")])),
                doc("persist-b", 1, metadata(&[("kind", "beta")])),
            ],
        )
        .unwrap();
    engine.persist().unwrap();
    assert!(path.join(format!("{INDEX_KEY}.commit.json")).is_file());
    assert!(!path.join("default.commit.json").exists());

    let payload_path = path.join(format!("{INDEX_KEY}.gen/g1.json"));
    let payload: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(payload_path).unwrap()).unwrap();
    assert_eq!(payload["index_id"], "default");
    assert_eq!(payload["index_key"], INDEX_KEY);
    assert_eq!(payload["client_id_hash"], INDEX_KEY);
    assert_eq!(payload["model"], "external");
    assert_eq!(payload["provider"], "external");
    assert_eq!(payload["task"], "vector-only");
    assert_eq!(payload["route_profile"], "vector-only");
    assert_eq!(payload["storage_profile"], "turbovec_direct");

    engine.close().unwrap();
    let readonly =
        CoreEngine::open_readonly(&path, open_options(&path, true, "generation")).unwrap();
    assert_eq!(
        readonly
            .query_vector(
                "default",
                &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1,
                None,
            )
            .unwrap()
            .hits[0]
            .document_id,
        "persist-b"
    );
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn persisted_native_tvim_backed_writable_open_persists_live_index() {
    let path = copy_persisted_fixture("v0_4_store_text");
    let mut writable = CoreEngine::open(open_options(&path, false, "generation")).unwrap();
    assert_eq!(writable.stats("default").unwrap().document_count, 3);
    writable
        .upsert_vectors(
            "default",
            &[doc("vec-delta", 3, metadata(&[("tenant", "new")]))],
        )
        .unwrap();
    writable.persist().unwrap();
    drop(writable);

    let readonly =
        CoreEngine::open_readonly(&path, open_options(&path, true, "generation")).unwrap();
    assert_eq!(readonly.stats("default").unwrap().document_count, 4);
    let hits = readonly
        .query_vector(
            "default",
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            1,
            None,
        )
        .unwrap();
    assert_eq!(hits.hits[0].document_id, "vec-delta");
    drop(readonly);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn persisted_v0_4_tvim_vectors_seed_readonly_queries() {
    let path = copy_persisted_fixture("v0_4_store_text");
    let readonly =
        CoreEngine::open_readonly(&path, open_options(&path, true, "generation")).unwrap();

    let hits = readonly
        .query_vector(
            "default",
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            3,
            None,
        )
        .unwrap();
    assert_eq!(hits.hits[0].document_id, "vec-alpha");
    assert_eq!(hits.total_considered, 3);

    let filtered = readonly
        .query_vector(
            "default",
            &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            3,
            Some(&json!({"metadata": {"tenant": "zen"}})),
        )
        .unwrap();
    assert_eq!(filtered.hits[0].document_id, "vec-beta");
    assert_eq!(filtered.total_considered, 1);
    drop(readonly);

    let mut writable = CoreEngine::open(open_options(&path, false, "generation")).unwrap();
    let mutation = writable
        .upsert_vectors(
            "default",
            &[doc("vec-delta", 3, metadata(&[("tenant", "new")]))],
        )
        .unwrap();
    assert_eq!(mutation.documents_upserted, 1);
    assert_eq!(
        writable
            .query_vector(
                "default",
                &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                1,
                None,
            )
            .unwrap()
            .hits[0]
            .document_id,
        "vec-delta"
    );
    drop(writable);

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn persisted_store_text_rebuilds_lexical_from_raw_text() {
    let path = copy_persisted_fixture("v0_4_store_text");
    let readonly =
        CoreEngine::open_readonly(&path, open_options(&path, true, "generation")).unwrap();
    let query_plan = readonly
        .prepare_query_text("retained payload", "lexical")
        .unwrap();

    let hits = readonly
        .search_embedded_text("default", &query_plan, None, 3, None)
        .unwrap();

    assert_eq!(hits.hits.len(), 3);
    assert!(hits
        .hits
        .iter()
        .all(|hit| hit.document_id.starts_with("vec-")));
    drop(readonly);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn persisted_tvim_delta_replays_into_native_index() {
    let fixture = lodedb_core::storage::fixture_root(
        "tests/fixtures/persisted/v0_4_generation/6f78dec251fa5e544784ac1af95b0ae6530cad714a2d34f8c4615740ecbf8205.gen",
    );
    let manifest = read_json_value(&fixture.join("g2.tvim.tvim-delta/manifest.json"));
    let chunks = vec![
        CoreVectorChunk::new("doc-alpha:6ed29ed824c2:0000", "doc-alpha", vec![0.0; 384]),
        CoreVectorChunk::new("doc-beta:7508a2274b7f:0000", "doc-beta", vec![0.0; 384]),
    ];

    let index = TurboVecNativeIndex::load_with_manifest(
        fixture.join("g2.tvim"),
        Some(&manifest),
        &chunks,
        2,
    )
    .unwrap();

    assert_eq!(index.len(), 2);
}

#[test]
fn persistent_engine_enforces_single_writer_but_readonly_takes_no_lock() {
    let path = unique_temp_dir("core_lock");
    let mut writer = CoreEngine::open(open_options(&path, false, "generation")).unwrap();
    assert!(CoreEngine::open(open_options(&path, false, "generation")).is_err());
    let readonly = CoreEngine::open_readonly(&path, open_options(&path, true, "generation"));
    assert!(readonly.is_ok());
    writer.close().unwrap();
    let reopened = CoreEngine::open(open_options(&path, false, "generation"));
    assert!(reopened.is_ok());
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn writable_open_replays_and_checkpoints_wal_but_readonly_ignores_it() {
    let path = copy_persisted_fixture("rust_wal");
    let readonly = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    assert_eq!(readonly.stats("default").unwrap().document_count, 0);
    assert!(path.join(format!("{INDEX_KEY}.wal")).exists());

    let mut writable = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    assert_eq!(writable.stats("default").unwrap().document_count, 1);
    writable.close().unwrap();
    assert!(!path.join(format!("{INDEX_KEY}.wal")).exists());
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn native_wal_vector_records_replay_and_checkpoint() {
    let path = unique_temp_dir("core_vector_wal");
    let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    engine
        .create_index_with_options(CoreIndexCreateOptions {
            index_id: "default".to_string(),
            index_key: INDEX_KEY.to_string(),
            client_id_hash: INDEX_KEY.to_string(),
            name: "lodedb-local".to_string(),
            model: "external".to_string(),
            provider: "external".to_string(),
            task: "vector-only".to_string(),
            route_profile: "vector-only".to_string(),
            storage_profile: "turbovec_direct".to_string(),
            vector_dim: 8,
            bit_width: 4,
        })
        .unwrap();
    engine.persist().unwrap();
    engine
        .upsert_vectors(
            "default",
            &[CoreVectorDocument {
                document_id: "wal-vec".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata: metadata(&[("kind", "wal")]),
                text: Some("WAL vector text".to_string()),
            }],
        )
        .unwrap();
    assert!(path.join(format!("{INDEX_KEY}.wal")).is_file());
    drop(engine);

    let readonly = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    assert_eq!(readonly.stats("default").unwrap().document_count, 0);
    drop(readonly);

    let writable = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    assert_eq!(writable.stats("default").unwrap().document_count, 1);
    assert_eq!(
        writable
            .query_vector(
                "default",
                &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1,
                None,
            )
            .unwrap()
            .hits[0]
            .document_id,
        "wal-vec"
    );
    assert!(!path.join(format!("{INDEX_KEY}.wal")).exists());
    drop(writable);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn native_wal_text_apply_records_replay_and_checkpoint() {
    let path = unique_temp_dir("core_text_wal");
    let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    engine
        .create_index_with_options(CoreIndexCreateOptions {
            index_id: "default".to_string(),
            index_key: INDEX_KEY.to_string(),
            client_id_hash: INDEX_KEY.to_string(),
            name: "lodedb-local".to_string(),
            model: "external".to_string(),
            provider: "external".to_string(),
            task: "text".to_string(),
            route_profile: "text".to_string(),
            storage_profile: "turbovec_direct".to_string(),
            vector_dim: 8,
            bit_width: 4,
        })
        .unwrap();
    engine.persist().unwrap();
    let plan = engine
        .prepare_text_upsert(
            "default",
            &[text_doc(
                "wal-text",
                "Alpha WAL text mentions E-1001.",
                metadata(&[("kind", "wal")]),
            )],
            true,
            true,
            900,
        )
        .unwrap();
    engine
        .apply_text_upsert(&plan, &[vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], 1.0)
        .unwrap();
    assert!(path.join(format!("{INDEX_KEY}.wal")).is_file());
    drop(engine);

    let writable = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    assert_eq!(writable.stats("default").unwrap().document_count, 1);
    let query_plan = writable.prepare_query_text("E-1001", "lexical").unwrap();
    assert_eq!(
        writable
            .search_embedded_text("default", &query_plan, None, 1, None)
            .unwrap()
            .hits[0]
            .document_id,
        "wal-text"
    );
    assert!(!path.join(format!("{INDEX_KEY}.wal")).exists());
    drop(writable);
    let loaded = lodedb_core::storage::load_store(
        &path,
        INDEX_KEY,
        lodedb_core::storage::LoadOptions::default(),
    )
    .unwrap();
    assert_eq!(
        loaded.raw_text.get("wal-text").map(String::as_str),
        Some("Alpha WAL text mentions E-1001.")
    );
    fs::remove_dir_all(path).unwrap();
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "lodedb-core-{label}-{}-{nanos}-{counter}",
        std::process::id()
    ))
}

fn copy_persisted_fixture(name: &str) -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/fixtures/persisted")
        .join(name);
    let target = unique_temp_dir(name);
    copy_dir_all(&source, &target);
    target
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

fn read_json_value(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
