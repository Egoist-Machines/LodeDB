use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use lodedb_core::engine::{CoreAppender, CoreEngine};
use lodedb_core::types::{
    CoreAnnOptions, CoreDocument, CoreIndexCreateOptions, CoreOpenOptions, CoreSearchResults,
    CoreVectorDocument,
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
        patch_matrix: None,
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
        compress_text: true,
        chunk_character_limit: 900,
        // Most tests open unique temp paths; the dedicated lock-contention test
        // builds its own options with the lock enabled.
        acquire_writer_lock: false,
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
fn list_documents_keyset_cursor_pages_in_id_order() {
    // The `after`/`limit` keyset cursor pages list_documents by stable-id order,
    // composing with a metadata filter.
    let mut engine = CoreEngine::new_in_memory();
    engine.create_index("default", 8, 4).unwrap();
    let docs: Vec<_> = (0..6)
        .map(|i| {
            let topic = if i % 2 == 0 { "a" } else { "b" };
            doc(&format!("d{i:03}"), i % 8, metadata(&[("topic", topic)]))
        })
        .collect();
    engine.upsert_vectors("default", &docs).unwrap();

    let ids = |page: Vec<serde_json::Value>| -> Vec<String> {
        page.iter()
            .map(|record| record["document_id"].as_str().unwrap().to_string())
            .collect()
    };

    let page1 = engine.list_documents("default", None, None, Some(4)).unwrap();
    assert_eq!(ids(page1), ["d000", "d001", "d002", "d003"]);
    let page2 = engine.list_documents("default", None, Some("d003"), Some(4)).unwrap();
    assert_eq!(ids(page2), ["d004", "d005"]);

    // The cursor composes with a filter (topic=a is d000, d002, d004).
    let filter = json!({"metadata": {"topic": "a"}});
    let filtered = engine
        .list_documents("default", Some(&filter), Some("d000"), Some(2))
        .unwrap();
    assert_eq!(ids(filtered), ["d002", "d004"]);
}

#[test]
fn payload_update_respects_store_text_privacy() {
    // store_text=false must keep no raw text in memory or in the WAL, including
    // after a payload-only text update (index_text=false, so no tokens either).
    let path = unique_temp_dir("core_payload_privacy");
    let mut options = open_options(&path, false, "wal");
    options.store_text = false;
    options.index_text = false;
    let mut engine = CoreEngine::open(options).unwrap();
    engine.create_index("default", 8, 4).unwrap();
    engine
        .upsert_vectors(
            "default",
            &[CoreVectorDocument {
                document_id: "d".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata: metadata(&[]),
                text: Some("initial secret caption".to_string()),
                patch_matrix: None,
            }],
        )
        .unwrap();
    engine
        .update_document_payload(
            "default",
            "d",
            None,
            Some(Some("updated secret caption".to_string())),
        )
        .unwrap();
    // No raw text is retained when store_text is off.
    assert!(engine
        .get_document_texts("default", &["d".to_string()])
        .unwrap()
        .is_empty());
    engine.close().unwrap();

    // The on-disk WAL must contain neither the original nor the updated text.
    let wal_bytes: Vec<u8> = fs::read_dir(&path)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "wal"))
        .flat_map(|entry| fs::read(entry.path()).unwrap_or_default())
        .collect();
    let wal_text = String::from_utf8_lossy(&wal_bytes);
    assert!(!wal_text.contains("initial secret caption"));
    assert!(!wal_text.contains("updated secret caption"));
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn native_wal_payload_update_replays_after_crash() {
    let path = unique_temp_dir("core_payload_wal");
    let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    engine.create_index("default", 8, 4).unwrap();
    engine
        .upsert_vectors(
            "default",
            &[CoreVectorDocument {
                document_id: "d".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata: metadata(&[("topic", "old")]),
                text: Some("old text".to_string()),
                patch_matrix: None,
            }],
        )
        .unwrap();
    engine.persist().unwrap();

    // Payload-only update in WAL mode, then a crash (drop without close): the
    // update must be in the WAL and replay on the next writable open.
    engine
        .update_document_payload(
            "default",
            "d",
            Some(metadata(&[("topic", "new")])),
            Some(Some("new text".to_string())),
        )
        .unwrap();
    // The update lives only in the WAL (persist() checkpointed the pre-update
    // generation); dropping without close simulates a crash.
    drop(engine);

    let mut replayed = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    let hits = replayed
        .query_vector(
            "default",
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1,
            None,
        )
        .unwrap();
    assert_eq!(hits.hits[0].metadata["topic"], "new");
    assert_eq!(
        replayed
            .get_document_text("default", "d")
            .unwrap()
            .as_deref(),
        Some("new text")
    );
    replayed.close().unwrap();
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn create_index_rejects_dimensions_turbovec_cannot_serve() {
    let mut engine = CoreEngine::new_in_memory();
    // TurboVec requires a positive multiple of 8; reject at create time rather
    // than accept an index that upserts but cannot build a serving index.
    assert!(engine.create_index("bad", 2, 4).is_err());
    assert!(engine.create_index("bad", 10, 4).is_err());
    // A valid multiple-of-8 dimension still works end to end.
    engine.create_index("ok", 8, 4).unwrap();
    engine
        .upsert_vectors("ok", &[doc("a", 0, metadata(&[]))])
        .unwrap();
    let hits = engine
        .query_vector("ok", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1, None)
        .unwrap();
    assert_eq!(hits.hits[0].document_id, "a");
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
        patch_matrix: None,
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
        .list_documents("text", Some(&json!({"metadata": {"topic": "ops"}})), None, None)
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
fn search_embedded_text_batch_matches_looped_single() {
    // The batched text/hybrid/lexical search must rank identically to looping the
    // single-query path, so search_many can share one vector scan across the batch.
    let mut engine = CoreEngine::new_in_memory();
    engine.create_index("text", 8, 4).unwrap();
    let plan = engine
        .prepare_text_upsert(
            "text",
            &[
                text_doc("doc-a", "fault code E1234", metadata(&[("topic", "ops")])),
                text_doc("doc-b", "quarterly revenue report", metadata(&[("topic", "finance")])),
                text_doc("doc-c", "revenue forecast E1234", metadata(&[("topic", "ops")])),
            ],
            true,
            true,
            100,
        )
        .unwrap();
    engine
        .apply_text_upsert(
            &plan,
            &[
                vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            1.0,
        )
        .unwrap();

    let queries = ["E1234", "revenue"];
    let embeddings = [
        vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    for mode in ["vector", "hybrid", "lexical"] {
        let plans: Vec<_> = queries
            .iter()
            .map(|query| engine.prepare_query_text(query, mode).unwrap())
            .collect();
        let embeds: Option<Vec<Vec<f32>>> =
            if mode == "lexical" { None } else { Some(embeddings.to_vec()) };
        let batch = engine
            .search_embedded_text_batch("text", &plans, embeds.as_deref(), 3, None)
            .unwrap();
        assert_eq!(batch.len(), queries.len());
        for (i, query_plan) in plans.iter().enumerate() {
            let single_embed: Option<&[f32]> =
                if mode == "lexical" { None } else { Some(embeddings[i].as_slice()) };
            let single = engine
                .search_embedded_text("text", query_plan, single_embed, 3, None)
                .unwrap();
            let batch_ids: Vec<&String> =
                batch[i].hits.iter().map(|hit| &hit.document_id).collect();
            let single_ids: Vec<&String> =
                single.hits.iter().map(|hit| &hit.document_id).collect();
            assert_eq!(batch_ids, single_ids, "mode {mode} query {i}: batch != single");
        }
    }
    // Mixed modes in one batch are rejected (search_many applies a single mode).
    let mixed = vec![
        engine.prepare_query_text("E1234", "lexical").unwrap(),
        engine.prepare_query_text("revenue", "hybrid").unwrap(),
    ];
    assert!(engine
        .search_embedded_text_batch("text", &mixed, None, 3, None)
        .is_err());
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
fn wal_replay_preserves_multivector_patches() {
    let path = unique_temp_dir("core_wal_multivector");
    let patch = lodedb_core::storage::multivec_store::MultiVecRecord {
        dtype: "float32".to_string(),
        patch_count: 2,
        bytes: [
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect(),
    };
    let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    engine.create_index("default", 8, 4).unwrap();
    // Checkpoint the (empty) index so a reopen finds it, then upsert the multi-vector
    // document so it lives only in the WAL (not yet in a committed generation).
    engine.persist().unwrap();
    engine
        .upsert_vectors(
            "default",
            &[CoreVectorDocument {
                document_id: "mv".to_string(),
                vector: vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata: BTreeMap::new(),
                text: None,
                patch_matrix: Some(patch),
            }],
        )
        .unwrap();
    // Reopen WITHOUT a checkpoint: the WAL replay must restore the patch matrix, not
    // just the anchor vector, so the late-interaction document still scores via MaxSim.
    drop(engine);
    let reopened = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    let hits = reopened
        .query_multivector("default", &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1, 1, None)
        .unwrap();
    assert_eq!(hits.hits.len(), 1);
    assert_eq!(hits.hits[0].document_id, "mv");
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
            ann: None,
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
#[cfg(unix)]
fn persistent_engine_enforces_single_writer_but_readonly_takes_no_lock() {
    let path = unique_temp_dir("core_lock");
    // Standalone native writers take the shared <dir>/.lodedb.lock; fail fast on
    // contention instead of waiting the default timeout.
    std::env::set_var("LODEDB_PERSIST_LOCK_TIMEOUT", "0");
    let locked = |read_only: bool| {
        let mut options = open_options(&path, read_only, "generation");
        options.acquire_writer_lock = true;
        options
    };
    let mut writer = CoreEngine::open(locked(false)).unwrap();
    // A second writable open contends with the held lock and must fail (a
    // separate descriptor in this process still conflicts with a BSD flock).
    assert!(CoreEngine::open(locked(false)).is_err());
    // A read-only open never takes the writer lock.
    let readonly = CoreEngine::open_readonly(&path, locked(true));
    assert!(readonly.is_ok());
    writer.close().unwrap();
    // Closing the writer releases the lock, so a new writer can acquire it.
    let reopened = CoreEngine::open(locked(false));
    assert!(reopened.is_ok());
    std::env::remove_var("LODEDB_PERSIST_LOCK_TIMEOUT");
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn writable_open_fails_closed_on_python_wal_and_readonly_ignores_it() {
    let path = copy_persisted_fixture("rust_wal");
    // Read-only open ignores the WAL tail (lock-free committed snapshot).
    let readonly = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    assert_eq!(readonly.stats("default").unwrap().document_count, 0);
    assert!(path.join(format!("{INDEX_KEY}.wal")).exists());
    drop(readonly);

    // The fixture WAL holds Python `upsert_documents` records, whose embeddings
    // the logical replay cannot turn into a TurboVec snapshot. A writable native
    // open must fail closed rather than checkpoint a tvim-less generation that
    // Python can no longer read, and it must leave the WAL intact for the Python
    // writer to own.
    assert!(CoreEngine::open(open_options(&path, false, "wal")).is_err());
    assert!(path.join(format!("{INDEX_KEY}.wal")).exists());
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
            ann: None,
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
                patch_matrix: None,
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
fn native_wal_replay_advances_to_a_fresh_generation() {
    // A crash after mutating but before checkpointing leaves the writes only in
    // the WAL. Recovery must fold them into a FRESH generation epoch: the counter
    // doubles as the immutable epoch id, so re-checkpointing must never rewrite
    // the committed base epoch in place (that would break crash-atomicity).
    let path = unique_temp_dir("core_wal_generation");
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
            ann: None,
        })
        .unwrap();
    engine.persist().unwrap();
    let first = engine
        .upsert_vectors(
            "default",
            &[CoreVectorDocument {
                document_id: "vec-a".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata: metadata(&[("kind", "wal")]),
                text: None,
                patch_matrix: None,
            }],
        )
        .unwrap();
    let second = engine
        .upsert_vectors(
            "default",
            &[CoreVectorDocument {
                document_id: "vec-b".to_string(),
                vector: vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata: metadata(&[("kind", "wal")]),
                text: None,
                patch_matrix: None,
            }],
        )
        .unwrap();
    assert_eq!(first.generation, 1);
    assert_eq!(second.generation, 2);
    // Drop without persisting: the WAL holds LSN 1 and 2 while the manifest is
    // still the empty checkpoint at generation 1.
    drop(engine);

    let mut reopened = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    // No loss and no duplication after replay.
    assert_eq!(reopened.stats("default").unwrap().document_count, 2);
    // Recovery advanced the generation past the recovered base rather than pinning
    // it back onto the committed epoch, so the next write's generation is strictly
    // greater than the last pre-crash write's.
    let third = reopened
        .upsert_vectors(
            "default",
            &[CoreVectorDocument {
                document_id: "vec-c".to_string(),
                vector: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                metadata: metadata(&[("kind", "wal")]),
                text: None,
                patch_matrix: None,
            }],
        )
        .unwrap();
    assert!(
        third.generation > second.generation,
        "generation must advance past the pre-crash base, got {}",
        third.generation
    );
    assert_eq!(reopened.stats("default").unwrap().document_count, 3);
    drop(reopened);

    // The recovery checkpoint is a valid committed generation: a clean reopen
    // still sees all three documents.
    let final_open = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    assert_eq!(final_open.stats("default").unwrap().document_count, 3);
    drop(final_open);
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
            ann: None,
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

#[test]
fn query_multivector_ranks_documents_by_maxsim() {
    use lodedb_core::storage::multivec_store::MultiVecRecord;

    fn unit(axis: usize) -> [f32; 8] {
        let mut row = [0.0_f32; 8];
        row[axis] = 1.0;
        row
    }
    fn encode(rows: &[[f32; 8]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in rows {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }
    fn mv_doc(id: &str, pooled_axis: usize, rows: &[[f32; 8]]) -> CoreVectorDocument {
        let mut pooled = vec![0.0_f32; 8];
        pooled[pooled_axis] = 1.0;
        CoreVectorDocument {
            document_id: id.to_string(),
            vector: pooled,
            metadata: metadata(&[]),
            text: None,
            patch_matrix: Some(MultiVecRecord {
                dtype: "float32".to_string(),
                patch_count: rows.len(),
                bytes: encode(rows),
            }),
        }
    }

    let mut engine = CoreEngine::new_in_memory();
    engine.create_index("default", 8, 4).unwrap();
    // doc-a carries patches on axes 0 and 2; doc-b only on axis 1.
    engine
        .upsert_vectors(
            "default",
            &[
                mv_doc("a", 0, &[unit(0), unit(2)]),
                mv_doc("b", 1, &[unit(1)]),
            ],
        )
        .unwrap();

    // A single query patch on axis 0: MaxSim favors doc-a (dot 1.0) over doc-b (0).
    let query = unit(0);
    let results = engine
        .query_multivector("default", &query, 1, 5, None)
        .unwrap();
    assert_eq!(results.total_considered, 2);
    assert_eq!(results.hits[0].document_id, "a");
    assert!((results.hits[0].score - 1.0).abs() < 1e-6);
    assert!(results.hits[0].score > results.hits[1].score);

    // Two query patches (axes 1 and 2): doc-a matches axis 2, doc-b matches axis 1.
    let two = [unit(1), unit(2)].concat();
    let results = engine
        .query_multivector("default", &two, 2, 5, None)
        .unwrap();
    // doc-a: max(axis1)=0 + max(axis2)=1 = 1; doc-b: max(axis1)=1 + max(axis2)=0 = 1.
    assert_eq!(results.hits.len(), 2);
    assert!((results.hits[0].score - 1.0).abs() < 1e-6);
}

#[test]
fn cleared_patch_matrix_does_not_resurrect_on_reload() {
    use lodedb_core::storage::multivec_store::MultiVecRecord;

    fn unit(axis: usize) -> [f32; 8] {
        let mut row = [0.0_f32; 8];
        row[axis] = 1.0;
        row
    }
    fn encode(rows: &[[f32; 8]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in rows {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }
    // The same live document, first with a late-interaction matrix, then without.
    let with_matrix = || CoreVectorDocument {
        document_id: "a".to_string(),
        vector: unit(0).to_vec(),
        metadata: metadata(&[]),
        text: None,
        patch_matrix: Some(MultiVecRecord {
            dtype: "float32".to_string(),
            patch_count: 2,
            bytes: encode(&[unit(0), unit(2)]),
        }),
    };
    let without_matrix = || CoreVectorDocument {
        document_id: "a".to_string(),
        vector: unit(0).to_vec(),
        metadata: metadata(&[]),
        text: None,
        patch_matrix: None,
    };

    let path = unique_temp_dir("core_mv_clear");
    let options = open_options(&path, false, "generation");
    // 1. Persist the matrix: a full base with a `.tvmv` base segment.
    {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.upsert_vectors("default", &[with_matrix()]).unwrap();
        engine.persist().unwrap();
    }
    // 2. Re-upsert the same live document without a matrix. The pooled vector is
    // unchanged, so this folds as an O(changed) delta (not a base rewrite) — the
    // path where an omitted clear would let the base matrix survive.
    {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        engine.upsert_vectors("default", &[without_matrix()]).unwrap();
        engine.persist().unwrap();
    }
    // 3. Reopen: the matrix must be gone. query_multivector skips matrix-less
    // documents, so a resurrected base matrix would make "a" reappear here.
    {
        let engine = CoreEngine::open(options).unwrap();
        let results = engine
            .query_multivector("default", &unit(0), 1, 5, None)
            .unwrap();
        assert!(
            results.hits.iter().all(|hit| hit.document_id != "a"),
            "cleared patch matrix resurrected from the base on reload"
        );
    }

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn deleted_then_readded_document_does_not_resurrect_matrix() {
    use lodedb_core::storage::multivec_store::MultiVecRecord;

    fn unit(axis: usize) -> [f32; 8] {
        let mut row = [0.0_f32; 8];
        row[axis] = 1.0;
        row
    }
    fn encode(rows: &[[f32; 8]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in rows {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }
    let with_matrix = || CoreVectorDocument {
        document_id: "a".to_string(),
        vector: unit(0).to_vec(),
        metadata: metadata(&[]),
        text: None,
        patch_matrix: Some(MultiVecRecord {
            dtype: "float32".to_string(),
            patch_count: 2,
            bytes: encode(&[unit(0), unit(2)]),
        }),
    };
    let without_matrix = || CoreVectorDocument {
        document_id: "a".to_string(),
        vector: unit(0).to_vec(),
        metadata: metadata(&[]),
        text: None,
        patch_matrix: None,
    };

    let path = unique_temp_dir("core_mv_delete_readd");
    let options = open_options(&path, false, "generation");
    // 1. Base with a committed matrix.
    {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.upsert_vectors("default", &[with_matrix()]).unwrap();
        engine.persist().unwrap();
    }
    // 2. Delete then re-add the same document without a matrix, both before persist.
    // The re-add cancels the pending delete, so only a delete-time clear keeps the
    // delta from carrying nothing for the matrix.
    {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        engine
            .delete_documents("default", &["a".to_string()])
            .unwrap();
        engine
            .upsert_vectors("default", &[without_matrix()])
            .unwrap();
        engine.persist().unwrap();
    }
    // 3. Reopen: the base matrix must not resurrect.
    {
        let engine = CoreEngine::open(options).unwrap();
        let results = engine
            .query_multivector("default", &unit(0), 1, 5, None)
            .unwrap();
        assert!(
            results.hits.iter().all(|hit| hit.document_id != "a"),
            "delete + re-add resurrected the base matrix on reload"
        );
    }

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn text_reupsert_clears_a_prior_patch_matrix_on_reload() {
    use lodedb_core::storage::multivec_store::MultiVecRecord;

    fn unit(axis: usize) -> [f32; 8] {
        let mut row = [0.0_f32; 8];
        row[axis] = 1.0;
        row
    }
    fn encode(rows: &[[f32; 8]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in rows {
            for value in row {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes
    }

    let path = unique_temp_dir("core_text_clears_matrix");
    let options = open_options(&path, false, "generation");
    // 1. A vector-in document that carries both a caption and a late-interaction
    // matrix, so its base has `.tvtext`/`.tvlex` (from the caption) alongside the
    // `.tvmv` matrix. The text/lexical bases are what let step 2's text re-add
    // append as a delta instead of forcing a full base rewrite.
    {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        engine.create_index("default", 8, 4).unwrap();
        let mut with_matrix = doc("a", 0, metadata(&[]));
        with_matrix.text = Some("hello world".to_string());
        with_matrix.patch_matrix = Some(MultiVecRecord {
            dtype: "float32".to_string(),
            patch_count: 2,
            bytes: encode(&[unit(0), unit(2)]),
        });
        engine.upsert_vectors("default", &[with_matrix]).unwrap();
        engine.persist().unwrap();
    }
    // 2. Re-add the same id through the TEXT path (which carries no matrix). The
    // chunk embedding equals the base document vector so the calibration
    // fingerprint is unchanged and this folds as an O(changed) delta (not a base
    // rewrite) — the path where an omitted clear leaves the base matrix behind.
    // This is the third replacement site the clear tracking must cover, alongside
    // upsert_vectors and delete.
    {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        let plan = engine
            .prepare_text_upsert("default", &[text_doc("a", "hello world", metadata(&[]))], true, true, 900)
            .unwrap();
        engine.apply_text_upsert(&plan, &[unit(0).to_vec()], 1.0).unwrap();
        engine.persist().unwrap();
    }
    // 3. Reopen: the base matrix must not resurrect (query_multivector skips
    // matrix-less documents, so a resurrected matrix would make "a" reappear).
    {
        let engine = CoreEngine::open(options).unwrap();
        let results = engine
            .query_multivector("default", &unit(0), 1, 5, None)
            .unwrap();
        assert!(
            results.hits.iter().all(|hit| hit.document_id != "a"),
            "a text re-add resurrected the base patch matrix on reload"
        );
    }

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn refresh_keeps_the_last_good_view_when_a_reload_fails() {
    // refresh() rebuilds the view from the base plus the WAL tail. If the rebuild
    // errors partway -- here a WAL record only the Python engine can replay, which
    // overlay_wal_tails rejects -- the reader must keep its last-good view (the base
    // plus the append it already overlaid), not collapse back to just the base or an
    // empty map.
    let path = unique_temp_dir("core_refresh_atomic");
    let mut writer_opts = open_options(&path, false, "wal");
    writer_opts.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(writer_opts).unwrap();
        create_vector_only_index(&mut engine);
        engine
            .upsert_vectors("default", &[doc("keep", 0, metadata(&[("kind", "base")]))])
            .unwrap();
        engine.persist().unwrap();
    }
    // A lock-free read-only handle over the committed base.
    let mut reader = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    // A concurrent appender logs one record beyond the base; the reader overlays it,
    // so its last-good view is base + WAL.
    let mut appender_opts = open_options(&path, false, "wal");
    appender_opts.acquire_writer_lock = true;
    {
        let appender = CoreAppender::open(appender_opts).expect("open appender");
        appender
            .append_vectors(&[doc("overlaid", 1, metadata(&[("kind", "wal")]))])
            .expect("append overlaid record");
    }
    reader.refresh().expect("first refresh overlays the appended record");
    assert_eq!(reader.stats("default").unwrap().document_count, 2);

    // A record only the Python engine can replay now lands in the WAL. The next
    // refresh cannot fold it, so it must fail...
    let wal = path.join(format!("{INDEX_KEY}.wal"));
    lodedb_core::storage::wal::append_record(
        &wal,
        999,
        "upsert_documents",
        json!({ "documents": [] }),
        false,
    )
    .expect("inject a python-only wal record");
    assert!(
        reader.refresh().is_err(),
        "refresh should surface the unreplayable record"
    );
    // ...and leave the last-good overlaid view intact, not roll back to just the
    // base (the pre-swap clear-then-reload would have dropped the overlaid append).
    assert_eq!(
        reader.stats("default").unwrap().document_count,
        2,
        "a failed refresh discarded the reader's last-good overlaid view"
    );

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn refresh_skips_the_reload_when_nothing_changed() {
    // After a refresh, if neither the committed manifest nor the WAL changed, the
    // next refresh must skip the reload entirely. Proven by deleting the base
    // segments a reload needs while leaving the manifest (its token) and the WAL (its
    // length) untouched: the unchanged signature makes the fast path return the
    // still-current view without ever touching the now-missing base.
    let path = unique_temp_dir("core_refresh_fastpath");
    let mut writer_opts = open_options(&path, false, "wal");
    writer_opts.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(writer_opts).unwrap();
        create_vector_only_index(&mut engine);
        engine
            .upsert_vectors("default", &[doc("keep", 0, metadata(&[]))])
            .unwrap();
        engine.persist().unwrap();
    }
    let mut reader = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    // The first refresh reloads and populates the fast-path cache.
    reader.refresh().unwrap();
    assert_eq!(reader.stats("default").unwrap().document_count, 1);

    // Remove the base segments a reload would read; the manifest and WAL are
    // untouched, so the fast-path signature is unchanged.
    fs::remove_dir_all(path.join(format!("{INDEX_KEY}.gen"))).unwrap();

    // The fast path skips the reload, so it never touches the missing base.
    reader.refresh().unwrap();
    assert_eq!(
        reader.stats("default").unwrap().document_count,
        1,
        "the unchanged-signature fast path should have skipped the reload"
    );

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn refresh_fast_path_detects_a_counter_only_change() {
    // The fast-path signature must fold in the appender's counter LSN, not just the
    // WAL byte length: a repaired-then-replaced tail can reuse the exact byte length
    // while the acknowledged LSN advances (the reservation always clears the WAL's
    // true max). Here only the counter LSN moves -- the WAL file and manifest are
    // untouched -- and the base is deleted, so a skip would keep serving the stale
    // view while a correct reload surfaces the now-missing base.
    let path = unique_temp_dir("core_refresh_counter_sig");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine
            .upsert_vectors("default", &[doc("keep", 0, metadata(&[]))])
            .unwrap();
        engine.persist().unwrap();
    }
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender");
        appender
            .append_vectors(&[doc("app1", 1, metadata(&[]))])
            .expect("append");
    }
    let mut reader = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    reader.refresh().unwrap();
    assert_eq!(reader.stats("default").unwrap().document_count, 2);

    // Advance ONLY the counter LSN: the WAL file and manifest stay byte-for-byte the
    // same, so the length- and token-based signals do not move.
    let lsn_file = lodedb_core::storage::lsn::lsn_path(&path, INDEX_KEY);
    let wal_len = fs::metadata(path.join(format!("{INDEX_KEY}.wal")))
        .unwrap()
        .len();
    {
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&lsn_file)
            .unwrap();
        let current = lodedb_core::storage::lsn::read_counter(&mut file)
            .unwrap()
            .unwrap();
        lodedb_core::storage::lsn::write_counter(&mut file, current.lsn + 1, Some(wal_len), false)
            .unwrap();
    }
    // Remove the base a reload would read, so a reload is observable as an error.
    fs::remove_dir_all(path.join(format!("{INDEX_KEY}.gen"))).unwrap();

    // The counter LSN moved, so the fast path must NOT skip; the reload then hits the
    // missing base and surfaces an error rather than silently serving a stale view.
    assert!(
        reader.refresh().is_err(),
        "a counter-only change must defeat the fast-path skip"
    );

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn read_only_refresh_overlays_wal_tail_for_read_your_writes() {
    let path = unique_temp_dir("core_reader_freshness");
    let mut writer_opts = open_options(&path, false, "wal");
    writer_opts.acquire_writer_lock = true;
    // A writer creates and checkpoints an empty index, then closes so an appender
    // (and the lock-free reader) can proceed.
    {
        let mut engine = CoreEngine::open(writer_opts).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }

    // A read-only handle: a stable snapshot of the committed base, no WAL overlay.
    let mut reader = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    let base_lsn = reader.applied_lsn("default").unwrap();
    assert!(
        reader
            .list_documents("default", None, None, None)
            .unwrap()
            .is_empty(),
        "the reader starts at the empty committed base"
    );

    // A separate appender durably logs a vector-in record.
    let mut appender_opts = open_options(&path, false, "wal");
    appender_opts.acquire_writer_lock = true;
    let appended_lsn = {
        let appender = CoreAppender::open(appender_opts).expect("open appender");
        appender
            .append_vectors(&[doc("fresh", 0, metadata(&[("kind", "appended")]))])
            .expect("append")
    };

    // Without a refresh the reader is still the snapshot: it neither sees the record
    // nor advances its applied LSN.
    assert_eq!(reader.applied_lsn("default").unwrap(), base_lsn);
    assert!(reader
        .list_documents("default", None, None, None)
        .unwrap()
        .is_empty());

    // Refresh overlays the current WAL tail in memory (no checkpoint): the append is
    // now visible and applied_lsn has caught up to (at least) the appended LSN.
    reader.refresh().unwrap();
    assert!(
        reader.applied_lsn("default").unwrap() >= appended_lsn,
        "refresh must fold the WAL up to the appended LSN for read-your-writes"
    );
    let docs = reader.list_documents("default", None, None, None).unwrap();
    assert_eq!(docs.len(), 1);
    assert_eq!(docs[0]["document_id"], serde_json::json!("fresh"));

    // The WAL was overlaid, not truncated: a fresh read-only open still sees the
    // record on its own refresh (the appender's record is still durable on disk).
    let mut reader2 = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    reader2.refresh().unwrap();
    assert_eq!(
        reader2.list_documents("default", None, None, None).unwrap().len(),
        1
    );

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn writable_handle_applied_lsn_reflects_the_committed_generation() {
    // A live writable handle's applied_lsn() must report the committed generation
    // (the writer's own LSN), not a stale zero: the persist path advances the
    // in-memory watermark to match the manifest, so the same handle and a fresh
    // reopen agree.
    let path = unique_temp_dir("core_writable_applied_lsn");
    let options = open_options(&path, false, "wal");
    let live = {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine
            .upsert_vectors("default", &[doc("a", 0, metadata(&[]))])
            .unwrap();
        engine.persist().unwrap();
        engine
            .upsert_vectors("default", &[doc("b", 1, metadata(&[]))])
            .unwrap();
        engine.persist().unwrap();
        engine.applied_lsn("default").unwrap()
    };
    assert!(
        live > 0,
        "a committed writable handle must not report a zero applied_lsn"
    );
    // A fresh open reads the durable watermark from the manifest; the live handle
    // must have reported exactly that.
    let reopened = CoreEngine::open(options).unwrap();
    assert_eq!(
        reopened.applied_lsn("default").unwrap(),
        live,
        "the live handle's applied_lsn diverged from the durable manifest watermark"
    );

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn applied_lsn_watermark_survives_a_no_op_folded_append() {
    // A folded record need not advance `generation` one-for-one (an idempotent
    // re-add is a full no-op), so the durable applied-LSN watermark is tracked
    // separately, or a reader's read-your-writes check for the appender's returned
    // LSN would stall after the writer folds and truncates the WAL. The fold also
    // clamps generation up to the watermark so the writer cannot re-mint an
    // acknowledged LSN.
    let path = unique_temp_dir("core_applied_lsn_watermark");
    let mut opts = open_options(&path, false, "wal");
    opts.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(opts.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    // Append the same document twice: the first fold is a real upsert (bumps the
    // generation), the second is an idempotent no-op (does not).
    let (first_lsn, second_lsn) = {
        let appender = CoreAppender::open(opts.clone()).expect("open appender");
        let first = appender
            .append_vectors(&[doc("dup", 0, metadata(&[]))])
            .expect("first append");
        let second = appender
            .append_vectors(&[doc("dup", 0, metadata(&[]))])
            .expect("second append");
        (first, second)
    };
    assert!(second_lsn > first_lsn);
    // A writer folds both records and checkpoints (truncating the WAL).
    let committed_generation = {
        let engine = CoreEngine::open(opts).unwrap();
        engine.stats("default").unwrap().generation
    };
    // The fold clamps generation up to the applied watermark (the no-op second fold
    // advanced applied_lsn but not generation one-per-LSN), so the writer's next
    // mint lands strictly above the acknowledged LSN rather than reusing it.
    assert!(
        committed_generation >= second_lsn,
        "generation must be clamped up to the applied watermark after a fold"
    );

    // The persisted watermark reached the acknowledged LSN, so a fresh read-only
    // reader (no refresh, WAL already truncated) reports read-your-writes for it.
    let reader = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    assert_eq!(
        reader.applied_lsn("default").unwrap(),
        second_lsn,
        "applied_lsn must reach the highest acknowledged append even when folded as a no-op"
    );

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn pure_no_op_fold_persists_watermark_and_truncates() {
    // If the WAL tail folds entirely to no-ops (an appender re-adding an
    // already-committed document), the fold advances no generation but does advance
    // the applied-LSN watermark. The writer must still commit that watermark (a
    // watermark-only epoch) before truncating the WAL, or the acknowledged LSN would
    // be stranded and generation-mode recovery would see a never-emptying WAL.
    let path = unique_temp_dir("core_pure_noop_fold");
    let mut opts = open_options(&path, false, "wal");
    opts.acquire_writer_lock = true;
    // Commit "dup" into the base.
    {
        let mut engine = CoreEngine::open(opts.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.upsert_vectors("default", &[doc("dup", 0, metadata(&[]))]).unwrap();
        engine.persist().unwrap();
    }
    // An appender re-adds the identical document: its fold is a pure no-op.
    let noop_lsn = {
        let appender = CoreAppender::open(opts.clone()).expect("open appender");
        appender.append_vectors(&[doc("dup", 0, metadata(&[]))]).expect("append")
    };
    // A writer opens (folds the no-op, commits the watermark) and closes.
    {
        let engine = CoreEngine::open(opts).unwrap();
        drop(engine);
    }
    // The WAL was truncated: the watermark is durable, so generation-mode recovery
    // (which rejects an unfolded WAL) stays safe.
    let wal = lodedb_core::storage::wal::wal_path(&path, INDEX_KEY);
    assert!(
        lodedb_core::storage::wal::read_records(&wal).unwrap().is_empty(),
        "the no-op fold's watermark should be durable, letting the WAL truncate"
    );
    // The watermark-only commit re-seals just the manifest: it must not append a
    // state-journal delta segment (an empty one would push the base toward premature
    // compaction). The base was authored with no deltas, so the journal stays empty.
    let mut delta_segments = 0;
    for entry in fs::read_dir(path.join(format!("{INDEX_KEY}.gen"))).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir()
            && entry.file_name().to_string_lossy().ends_with(".json-delta")
        {
            for segment in fs::read_dir(entry.path()).unwrap() {
                if segment.unwrap().file_name().to_string_lossy().ends_with(".jsd") {
                    delta_segments += 1;
                }
            }
        }
    }
    assert_eq!(
        delta_segments, 0,
        "a watermark-only commit must not append a state-journal delta segment"
    );
    // A fresh read-only reader reaches read-your-writes for the acknowledged LSN
    // straight from the durable manifest watermark -- no refresh needed.
    let reader = CoreEngine::open_readonly(&path, open_options(&path, true, "wal")).unwrap();
    assert!(
        reader.applied_lsn("default").unwrap() >= noop_lsn,
        "the committed watermark must cover the acknowledged no-op LSN"
    );

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn concurrent_appenders_are_folded_by_the_next_writer() {
    let path = unique_temp_dir("core_appender");
    // Real coordination: writer takes the exclusive lock, appenders the shared one.
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;

    // A writer creates the index and checkpoints an empty base, then closes so the
    // shared appenders can take the lock.
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
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
                ann: None,
            })
            .unwrap();
        engine.persist().unwrap();
    }

    // Many concurrent appenders take the shared lock and log vector-in records.
    let threads = 5_usize;
    let per_thread = 8_usize;
    let handles: Vec<_> = (0..threads)
        .map(|thread| {
            let options = base.clone();
            std::thread::spawn(move || {
                let appender = CoreAppender::open(options).expect("open appender");
                for i in 0..per_thread {
                    let id = format!("doc-{thread}-{i}");
                    appender
                        .append_vectors(&[doc(&id, i % 8, metadata(&[("kind", "appended")]))])
                        .expect("append vectors");
                }
            })
        })
        .collect();
    for handle in handles {
        handle.join().unwrap();
    }

    // The next exclusive writer folds every appended record into the index: no
    // loss, no duplication, no LSN collision across the concurrent appenders.
    let writer = CoreEngine::open(base).unwrap();
    assert_eq!(
        writer.stats("default").unwrap().document_count,
        threads * per_thread
    );
    // An appended vector is queryable, proving the vector and metadata survived the
    // WAL round-trip.
    let mut probe = vec![0.0_f32; 8];
    probe[3] = 1.0;
    let hits = writer.query_vector("default", &probe, 1, None).unwrap().hits;
    assert_eq!(hits.len(), 1);
    assert!(hits[0].document_id.starts_with("doc-"));
    drop(writer);
    fs::remove_dir_all(path).unwrap();
}

fn create_vector_only_index(engine: &mut CoreEngine) {
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
            ann: None,
        })
        .unwrap();
}

#[test]
fn appender_rejects_generation_mode() {
    let path = unique_temp_dir("core_appender_gen");
    let mut wal_opts = open_options(&path, false, "wal");
    wal_opts.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(wal_opts).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    // Generation mode never replays the WAL, so appended records would be
    // acknowledged yet invisible; the appender must refuse to open.
    let mut gen_opts = open_options(&path, false, "generation");
    gen_opts.acquire_writer_lock = true;
    assert!(CoreAppender::open(gen_opts).is_err());
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_repairs_a_torn_wal_tail() {
    let path = unique_temp_dir("core_appender_torn");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    // One good appended record.
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender");
        appender
            .append_vectors(&[doc("before-crash", 0, metadata(&[("kind", "a")]))])
            .expect("append good record");
    }
    // Simulate a crash mid-append: a frame that claims a 5-byte body but carries
    // only two, leaving a torn trailing frame after the good record.
    let wal = path.join(format!("{INDEX_KEY}.wal"));
    {
        use std::io::Write;
        let mut handle = std::fs::OpenOptions::new()
            .append(true)
            .open(&wal)
            .expect("open wal for torn write");
        handle.write_all(&[0, 0, 0, 5, 1, 2]).expect("write torn frame");
    }
    // The next appender repairs the torn tail, so its record lands after the good
    // one rather than after the torn bytes.
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender after torn tail");
        appender
            .append_vectors(&[doc("after-repair", 1, metadata(&[("kind", "b")]))])
            .expect("append after repair");
    }
    // The writer replays both durable records; the torn frame is gone.
    let writer = CoreEngine::open(base).unwrap();
    assert_eq!(writer.stats("default").unwrap().document_count, 2);
    drop(writer);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_repairs_a_zero_byte_wal() {
    let path = unique_temp_dir("core_appender_zero");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    // A crash can leave a zero-byte `<key>.wal` created before its header was
    // written. The appender must drop it rather than append a headerless frame
    // that the next writer would reject as bad magic.
    let wal = path.join(format!("{INDEX_KEY}.wal"));
    std::fs::File::create(&wal).expect("create zero-byte wal");
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender over zero-byte wal");
        appender
            .append_vectors(&[doc("after-zero", 2, metadata(&[("kind", "z")]))])
            .expect("append after zero-byte repair");
    }
    let writer = CoreEngine::open(base).unwrap();
    assert_eq!(writer.stats("default").unwrap().document_count, 1);
    drop(writer);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn append_after_a_crashed_peer_tail_never_reuses_its_lsn() {
    // A crashed peer can leave a complete-but-unacknowledged frame past the
    // counter's watermark. refresh() overlays every CRC-valid frame, so a reader can
    // transiently observe that frame's LSN; if the next append re-minted it for a
    // different record, that reader's applied_lsn would point at the wrong record.
    // The repair must clamp the reservation above the WAL's true max LSN, dropping
    // the unacknowledged tail without reusing its LSN.
    let path = unique_temp_dir("core_appender_no_lsn_reuse");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    let wal = path.join(format!("{INDEX_KEY}.wal"));

    let appender = CoreAppender::open(base.clone()).expect("open appender");
    // One acknowledged append: the counter's watermark now sits at the end of it.
    let first = appender
        .append_vectors(&[doc("acked", 0, metadata(&[("kind", "a")]))])
        .expect("append acknowledged record");
    // Inject a crashed peer's complete frame one LSN past the watermark WITHOUT
    // advancing the shared counter -- the state a peer leaves when it writes a frame
    // and crashes before publishing the watermark.
    lodedb_core::storage::wal::append_record(
        &wal,
        first + 1,
        "upsert_vectors",
        json!({ "vectors": [] }),
        false,
    )
    .expect("inject crashed peer frame");
    // The same appender's next append re-reads the counter (still at `first`) and
    // takes the repair path: it must land two LSNs on, above the orphaned frame, not
    // reuse `first + 1`.
    let second = appender
        .append_vectors(&[doc("after", 1, metadata(&[("kind", "b")]))])
        .expect("append after the crashed peer tail");
    assert_eq!(
        second,
        first + 2,
        "append reused a crashed peer's unacknowledged LSN"
    );
    drop(appender);

    // The unacknowledged frame is dropped; the writer replays only the two
    // acknowledged records.
    let writer = CoreEngine::open(base).unwrap();
    assert_eq!(writer.stats("default").unwrap().document_count, 2);
    drop(writer);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn load_store_metadata_reads_native_dim_from_the_manifest() {
    // The commit manifest records native_dim, so a metadata read does not touch the
    // state base. Corrupt every state-base segment: the read must still succeed and
    // report the right dimension straight from the manifest.
    let path = unique_temp_dir("core_metadata_native_dim");
    {
        let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
        create_vector_only_index(&mut engine);
        engine
            .upsert_vectors("default", &[doc("a", 0, metadata(&[]))])
            .unwrap();
        engine.persist().unwrap();
    }
    let gen_dir = path.join(format!("{INDEX_KEY}.gen"));
    for entry in fs::read_dir(&gen_dir).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_file()
            && entry.file_name().to_string_lossy().ends_with(".json")
        {
            fs::write(entry.path(), b"corrupt").unwrap();
        }
    }
    let meta = lodedb_core::storage::load_store_metadata(&path, INDEX_KEY).unwrap();
    assert_eq!(meta.native_dim, 8);

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_rejects_non_native_wal() {
    let path = unique_temp_dir("core_appender_nonnative");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    // A prior Python text ingest can leave an `upsert_documents` record the native
    // writer cannot replay. Appending native records behind it would strand them,
    // so the appender must refuse to open until a writer recovers the store.
    let wal = lodedb_core::storage::wal::wal_path(&path, INDEX_KEY);
    lodedb_core::storage::wal::append_record(&wal, 1, "upsert_documents", json!({ "documents": [] }), false)
        .expect("write non-native record");
    assert!(CoreAppender::open(base).is_err());
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_repairs_torn_tail_between_appends() {
    let path = unique_temp_dir("core_appender_between");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    let wal = path.join(format!("{INDEX_KEY}.wal"));
    {
        // The appender stays open across a peer's crash: a torn frame injected
        // mid-session must be repaired by the next append, not left to strand it.
        let appender = CoreAppender::open(base.clone()).expect("open appender");
        appender
            .append_vectors(&[doc("first", 0, metadata(&[("n", "1")]))])
            .expect("append first");
        {
            use std::io::Write;
            let mut handle = std::fs::OpenOptions::new()
                .append(true)
                .open(&wal)
                .expect("open wal for torn write");
            handle.write_all(&[0, 0, 0, 5, 1, 2]).expect("write torn frame");
        }
        appender
            .append_vectors(&[doc("second", 1, metadata(&[("n", "2")]))])
            .expect("append second after torn tail");
    }
    let writer = CoreEngine::open(base).unwrap();
    assert_eq!(writer.stats("default").unwrap().document_count, 2);
    drop(writer);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_preserves_a_writers_wal_growth_across_sessions() {
    // A stale cross-session watermark must never make an append truncate records
    // a writer committed to the WAL after the last appender session closed. The
    // O(1) repair trusts the watermark only within a session; `open` re-scans and
    // re-seeds it, so a writer's intervening frames survive.
    let path = unique_temp_dir("core_appender_writer_growth");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    // Appender session 1 records a watermark at the end of its frame, then closes.
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender 1");
        appender
            .append_vectors(&[doc("appended-1", 0, metadata(&[("s", "1")]))])
            .expect("append 1");
    }
    // A writer grows the WAL without touching the LSN counter (the single-writer
    // WAL path uses its in-memory generation, not the shared allocator), so the
    // counter's watermark now sits behind the file. Its LSN stays above the base
    // generation, so replay keeps it.
    let wal = lodedb_core::storage::wal::wal_path(&path, INDEX_KEY);
    lodedb_core::storage::wal::append_record(
        &wal,
        3,
        "upsert_vectors",
        json!({
            "vectors": [{
                "document_id": "writer-doc",
                "vector": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "metadata": {},
                "text": null,
                "tokens": null,
                "patch_matrix": null
            }]
        }),
        false,
    )
    .expect("writer grows the wal");
    // Appender session 2 opens (full-scanning and re-seeding the watermark to
    // include the writer's frame), then appends. The writer's frame must not be
    // truncated back to session 1's stale watermark.
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender 2");
        appender
            .append_vectors(&[doc("appended-2", 4, metadata(&[("s", "2")]))])
            .expect("append 2");
    }
    // The next writer replays all three durable records: both appends and the
    // writer's own growth.
    let writer = CoreEngine::open(base).unwrap();
    assert_eq!(writer.stats("default").unwrap().document_count, 3);
    drop(writer);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_does_not_reuse_an_lsn_after_a_torn_counter() {
    // A long-lived appender opened at floor F=1 (empty WAL). A peer commits a valid
    // frame at LSN 2 and then crashes while rewriting the counter, leaving it torn
    // (valid magic and length, bad CRC -> reads as absent). The appender's next
    // append must scan the WAL and clamp above 2, not reuse 2 from its stale
    // open-time floor, or the WAL would hold two frames at the same LSN.
    let path = unique_temp_dir("core_appender_torn_counter_lsn");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    // Open the appender first, so its floor is captured before the peer's frame.
    let appender = CoreAppender::open(base.clone()).expect("open appender");
    // A peer commits a valid native frame at LSN 2 (the LSN it would have reserved
    // from the empty counter), then its counter rewrite is torn by a crash.
    let wal = lodedb_core::storage::wal::wal_path(&path, INDEX_KEY);
    let peer_lsn = 2_u64;
    lodedb_core::storage::wal::append_record(
        &wal,
        peer_lsn,
        "upsert_vectors",
        json!({
            "vectors": [{
                "document_id": "peer-doc",
                "vector": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "metadata": {},
                "text": null,
                "tokens": null,
                "patch_matrix": null
            }]
        }),
        false,
    )
    .expect("peer commits a frame");
    let lsn_file = path.join(format!("{INDEX_KEY}.lsn"));
    let mut counter_bytes = std::fs::read(&lsn_file).expect("read counter");
    counter_bytes[10] ^= 0xFF; // corrupt the payload so the CRC fails
    std::fs::write(&lsn_file, counter_bytes).expect("torn counter");
    // The appender's next append must land above the peer's LSN, not reuse it.
    let lsn = appender
        .append_vectors(&[doc("appended", 4, metadata(&[("s", "a")]))])
        .expect("append after torn counter");
    assert!(
        lsn > peer_lsn,
        "expected an LSN above the peer's {peer_lsn}, got {lsn}"
    );
    // Both frames replay with distinct LSNs: the peer's is kept, ours is above it.
    drop(appender);
    let writer = CoreEngine::open(base).unwrap();
    assert_eq!(writer.stats("default").unwrap().document_count, 2);
    drop(writer);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_open_reseed_does_not_let_a_peer_reuse_an_lsn() {
    // When one appender's open heals a torn counter, it must publish a truthful
    // LSN. Otherwise a second, still-open appender reads the healed (crc-valid)
    // counter via the O(1) fast path, trusts its stale LSN, and reuses an LSN a
    // peer already committed to the WAL.
    let path = unique_temp_dir("core_appender_reseed_lsn");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    // Long-lived appender X opens first, capturing a floor of 1 (empty WAL).
    let x = CoreAppender::open(base.clone()).expect("open X");
    // A peer commits a valid frame at LSN 2, then its counter rewrite is torn.
    let wal = lodedb_core::storage::wal::wal_path(&path, INDEX_KEY);
    lodedb_core::storage::wal::append_record(
        &wal,
        2,
        "upsert_vectors",
        json!({
            "vectors": [{
                "document_id": "peer-doc",
                "vector": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "metadata": {},
                "text": null,
                "tokens": null,
                "patch_matrix": null
            }]
        }),
        false,
    )
    .expect("peer commits a frame");
    let lsn_file = path.join(format!("{INDEX_KEY}.lsn"));
    let mut counter_bytes = std::fs::read(&lsn_file).expect("read counter");
    counter_bytes[10] ^= 0xFF;
    std::fs::write(&lsn_file, counter_bytes).expect("torn counter");
    // A second appender Y opens, healing the torn counter from its WAL scan. The
    // healed counter must carry an LSN of at least 2, not 0.
    let y = CoreAppender::open(base.clone()).expect("open Y");
    // X, still holding its stale open-time floor, reads the healed counter on the
    // fast path and must still land above the peer's LSN.
    let lsn = x
        .append_vectors(&[doc("x-doc", 4, metadata(&[("s", "x")]))])
        .expect("append X");
    assert!(lsn > 2, "X reused an LSN through the healed counter: got {lsn}");
    drop(x);
    drop(y);
    let writer = CoreEngine::open(base).unwrap();
    assert_eq!(writer.stats("default").unwrap().document_count, 2);
    drop(writer);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_retains_text_only_under_store_text() {
    let path = unique_temp_dir("core_appender_text");

    // store_text on: the appended caption is logged and the next writer retains it.
    let mut with_text = open_options(&path, false, "wal"); // open_options sets store_text = true
    with_text.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(with_text.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    {
        let appender = CoreAppender::open(with_text.clone()).expect("open appender (store_text)");
        let mut document = doc("with-text", 0, metadata(&[("kind", "image")]));
        document.text = Some("a red bicycle by the canal".to_string());
        appender.append_vectors(&[document]).expect("append with text");
    }
    {
        let writer = CoreEngine::open(with_text).unwrap();
        assert_eq!(
            writer
                .get_document_text("default", "with-text")
                .unwrap()
                .as_deref(),
            Some("a red bicycle by the canal")
        );
    }

    // store_text off, index_text on (the privacy mode): no raw text reaches the WAL,
    // but the derived caption tokens do, so replay can rebuild lexical postings
    // without retaining raw text (parity with the engine's upsert_vectors record).
    let mut private_mode = open_options(&path, false, "wal"); // open_options sets index_text = true
    private_mode.acquire_writer_lock = true;
    private_mode.store_text = false;
    {
        let appender =
            CoreAppender::open(private_mode.clone()).expect("open appender (private mode)");
        let mut document = doc("captioned", 1, metadata(&[]));
        document.text = Some("turquoise dragon".to_string());
        appender.append_vectors(&[document]).expect("append captioned");
    }
    // Inspect the appended record before a writer folds and truncates the WAL: it
    // carries derived tokens but no raw text.
    {
        let wal = lodedb_core::storage::wal::wal_path(&path, INDEX_KEY);
        let records = lodedb_core::storage::wal::read_records(&wal).expect("read wal");
        let appended = records.last().expect("an appended record");
        assert_eq!(appended.op, "upsert_vectors");
        let vector = &appended.payload["vectors"][0];
        assert!(vector["text"].is_null(), "no raw text in privacy mode");
        assert!(
            vector["tokens"].is_array(),
            "derived caption tokens must be logged for lexical replay"
        );
    }
    {
        let writer = CoreEngine::open(private_mode).unwrap();
        assert_eq!(
            writer.get_document_text("default", "captioned").unwrap(),
            None
        );
    }

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_and_writer_log_byte_identical_upsert_vectors() {
    // The upsert_vectors WAL record has a single shared builder
    // (`wal_vector_document`), so a concurrent CoreAppender must log a record
    // byte-identical to the one the exclusive writer's `upsert_vectors` writes for
    // the same document (replay resolves both through the same path, so any
    // divergence would silently change what an appended record restores). Guard the
    // parity by running both real paths and comparing the logged payload (the LSN
    // differs by construction and is lifted out of the payload on read).
    let mut document = doc(
        "captioned-parity",
        3,
        metadata(&[("kind", "image"), ("topic", "ops")]),
    );
    document.text = Some("a red bicycle by the canal".to_string());

    let read_last_payload = |path: &Path| -> Value {
        let wal = lodedb_core::storage::wal::wal_path(path, INDEX_KEY);
        let records = lodedb_core::storage::wal::read_records(&wal).expect("read wal");
        let record = records.last().expect("a logged upsert_vectors record");
        assert_eq!(record.op, "upsert_vectors");
        assert!(record.payload.get("lsn").is_none(), "lsn is lifted out on read");
        record.payload.clone()
    };

    // Writer path: create + upsert in WAL mode, then read the record it logged.
    let writer_path = unique_temp_dir("core_wal_parity_writer");
    let mut writer_options = open_options(&writer_path, false, "wal"); // store_text + index_text on
    writer_options.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(writer_options).unwrap();
        create_vector_only_index(&mut engine);
        engine
            .upsert_vectors("default", std::slice::from_ref(&document))
            .unwrap();
    }
    let writer_payload = read_last_payload(&writer_path);

    // Appender path: create + persist an identical store, then append the same doc.
    let appender_path = unique_temp_dir("core_wal_parity_appender");
    let mut appender_options = open_options(&appender_path, false, "wal");
    appender_options.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(appender_options.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    CoreAppender::open(appender_options)
        .expect("open appender")
        .append_vectors(std::slice::from_ref(&document))
        .expect("append");
    let appender_payload = read_last_payload(&appender_path);

    assert_eq!(
        writer_payload, appender_payload,
        "writer and appender must log byte-identical upsert_vectors records"
    );

    fs::remove_dir_all(writer_path).unwrap();
    fs::remove_dir_all(appender_path).unwrap();
}

#[test]
fn appender_open_reads_metadata_not_the_vector_image() {
    // The appender opens O(metadata): it reads the committed generation and the
    // vector dimension from the commit manifest + state base only, never the
    // `.tvim` vectors or the sidecars (an append does not reconstruct them, and a
    // corrupt vector image is the folding writer's problem). So corrupting the
    // committed `.tvim` must not stop an appender from opening and logging.
    let path = unique_temp_dir("core_appender_open_metadata");
    let mut options = open_options(&path, false, "wal");
    options.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine
            .upsert_vectors("default", &[doc("seed", 0, metadata(&[]))])
            .unwrap();
        engine.persist().unwrap();
    }
    // Overwrite every committed `.tvim` base with garbage. A full store load would
    // now fail its checksum; the appender's metadata-only open must not.
    let gen_dir = path.join(format!("{INDEX_KEY}.gen"));
    let mut corrupted = 0;
    for entry in fs::read_dir(&gen_dir).unwrap() {
        let entry_path = entry.unwrap().path();
        if entry_path.extension().and_then(|ext| ext.to_str()) == Some("tvim") {
            fs::write(&entry_path, b"corrupt-tvim-bytes").unwrap();
            corrupted += 1;
        }
    }
    assert!(corrupted > 0, "expected a committed .tvim base to corrupt");

    let appender = CoreAppender::open(options).expect("appender opens over a corrupt vector image");
    let lsn = appender
        .append_vectors(&[doc("appended", 1, metadata(&[]))])
        .expect("append after corrupt-tvim open");
    assert!(lsn >= 1);

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appended_lsn_exceeds_committed_generation() {
    // The exclusive writer uses the generation as its own WAL LSN and can advance
    // it faster than one-per-LSN, so the committed generation can exceed every WAL
    // LSN. An appended LSN must still clear the committed generation, or a later
    // writer's replay would treat the append as already folded and drop it.
    let path = unique_temp_dir("core_appender_gen_clamp");
    let mut options = open_options(&path, false, "wal");
    options.acquire_writer_lock = true;
    let committed_generation;
    {
        let mut engine = CoreEngine::open(options.clone()).unwrap();
        create_vector_only_index(&mut engine);
        // Several upserts advance the generation; the checkpoint truncates the WAL,
        // so at append time the WAL max LSN is 0 while the generation is high.
        for i in 0..5 {
            engine
                .upsert_vectors("default", &[doc(&format!("d{i}"), i % 8, metadata(&[]))])
                .unwrap();
        }
        engine.persist().unwrap();
        committed_generation = engine.stats("default").unwrap().generation;
    }
    assert!(committed_generation >= 1);

    let appender = CoreAppender::open(options).expect("open appender");
    let lsn = appender
        .append_vectors(&[doc("appended", 0, metadata(&[]))])
        .expect("append");
    assert!(
        lsn > committed_generation,
        "appended LSN {lsn} must exceed the committed generation {committed_generation}"
    );

    fs::remove_dir_all(path).unwrap();
}

#[test]
fn appender_privacy_mode_tokens_persist_across_reopen() {
    // Privacy mode (store_text=false, index_text=true): a caption appended onto an
    // existing vector-identical document must survive a checkpoint and reopen. The
    // replay's upsert is a no-op (vector and metadata unchanged, no raw text), so
    // the restored caption tokens have to mark the document pending, or the
    // checkpoint truncates the WAL and drops them.
    let path = unique_temp_dir("core_appender_privacy_tokens");
    let mut base = open_options(&path, false, "wal"); // index_text = true
    base.store_text = false;
    base.acquire_writer_lock = true;
    let generation_before;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        // A pre-existing document with a vector but no caption tokens.
        engine
            .upsert_vectors("default", &[doc("d1", 0, metadata(&[]))])
            .unwrap();
        engine.persist().unwrap();
        generation_before = engine.stats("default").unwrap().generation;
    }
    // The appender adds a caption to the same vector: tokens, no raw text.
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender");
        let mut document = doc("d1", 0, metadata(&[]));
        document.text = Some("turquoise dragon".to_string());
        appender.append_vectors(&[document]).expect("append caption");
    }
    // A writer replays (upsert no-op + token restore) and checkpoints, truncating
    // the WAL. Applying the token restore must advance the generation epoch so
    // generation-based observers see the lexical update.
    {
        let writer = CoreEngine::open(base.clone()).unwrap();
        assert!(
            writer.stats("default").unwrap().generation > generation_before,
            "token restore must advance the generation"
        );
    }
    // After reopen the caption must still be lexically searchable: the tokens were
    // persisted, not dropped with the truncated WAL.
    {
        let writer = CoreEngine::open(base).unwrap();
        let plan = writer.prepare_query_text("dragon", "lexical").unwrap();
        let hits = writer
            .search_embedded_text("default", &plan, None, 1, None)
            .unwrap();
        assert_eq!(hits.hits.len(), 1, "restored caption tokens were dropped at checkpoint");
        assert_eq!(hits.hits[0].document_id, "d1");
    }
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn generation_mode_writable_open_refuses_unfolded_wal_records() {
    // A writable generation-mode open over acknowledged-but-unfolded appends must
    // refuse: its commits would advance the committed generation past the appended
    // LSNs, and the next WAL-mode open would then skip them as already folded and
    // truncate the log, silently destroying the acknowledged records.
    let path = unique_temp_dir("core_appender_generation_guard");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender");
        appender
            .append_vectors(&[doc("d1", 0, metadata(&[]))])
            .expect("append vectors");
    }
    let mut generation = base.clone();
    generation.commit_mode = "generation".to_string();
    let error = match CoreEngine::open(generation.clone()) {
        Ok(_) => panic!("generation-mode open must refuse unfolded WAL records"),
        Err(error) => error,
    };
    assert!(
        error.message().contains("unfolded WAL records"),
        "unexpected error: {}",
        error.message()
    );
    // A WAL-mode open folds the record; generation mode is accepted again after.
    {
        let writer = CoreEngine::open(base).unwrap();
        assert_eq!(writer.stats("default").unwrap().document_count, 1);
    }
    let reopened = CoreEngine::open(generation).unwrap();
    assert_eq!(reopened.stats("default").unwrap().document_count, 1);
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn replayed_empty_caption_matches_a_live_written_store() {
    // A caption that tokenizes to nothing ("///") is logged as an empty token list;
    // its replay must leave the lexical index exactly as a live write does (no
    // zero-token BM25 unit inflating n/avgdl), so scores match a never-crashed
    // store bit-for-bit.
    let live = unique_temp_dir("core_appender_empty_caption_live");
    let replayed = unique_temp_dir("core_appender_empty_caption_replayed");
    let documents = || {
        let mut worded = doc("d1", 0, metadata(&[]));
        worded.text = Some("alpha beta".to_string());
        let mut empty = doc("d2", 1, metadata(&[]));
        empty.text = Some("///".to_string());
        vec![worded, empty]
    };

    // Live store: the writer ingests both captions directly. Keep the session
    // open: the comparison is against the live writer's own lexical state (the
    // reopen rebuild has its own captionless-document shape on both sides).
    let mut base_live = open_options(&live, false, "wal");
    base_live.store_text = false;
    base_live.acquire_writer_lock = true;
    let mut live_engine = CoreEngine::open(base_live.clone()).unwrap();
    create_vector_only_index(&mut live_engine);
    live_engine.upsert_vectors("default", &documents()).unwrap();
    live_engine.persist().unwrap();

    // Replayed store: the same records arrive through an appender and are folded
    // by the next writable open.
    let mut base_replayed = open_options(&replayed, false, "wal");
    base_replayed.store_text = false;
    base_replayed.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base_replayed.clone()).unwrap();
        create_vector_only_index(&mut engine);
        engine.persist().unwrap();
    }
    {
        let appender = CoreAppender::open(base_replayed.clone()).expect("open appender");
        appender.append_vectors(&documents()).expect("append vectors");
    }

    let folded = CoreEngine::open(base_replayed).unwrap();
    let live_plan = live_engine.prepare_query_text("alpha", "lexical").unwrap();
    let live_hits = live_engine
        .search_embedded_text("default", &live_plan, None, 2, None)
        .unwrap();
    let folded_plan = folded.prepare_query_text("alpha", "lexical").unwrap();
    let folded_hits = folded
        .search_embedded_text("default", &folded_plan, None, 2, None)
        .unwrap();
    assert_eq!(live_hits.hits.len(), 1);
    assert_eq!(folded_hits.hits.len(), 1);
    assert_eq!(folded_hits.hits[0].document_id, "d1");
    assert_eq!(
        folded_hits.hits[0].score.to_bits(),
        live_hits.hits[0].score.to_bits(),
        "a replayed empty caption must not inflate the BM25 statistics"
    );
    drop(live_engine);
    drop(folded);
    fs::remove_dir_all(live).unwrap();
    fs::remove_dir_all(replayed).unwrap();
}

#[test]
fn cleared_caption_does_not_resurrect_after_reopen() {
    // Clearing a caption must reach the lexical delta as an explicit delete:
    // absence means "unchanged", so the base's old tokens would resurrect on
    // reload. Cover both clear paths: a live payload update and an appender
    // re-caption folded by WAL replay.
    let path = unique_temp_dir("core_caption_clear");
    let mut base = open_options(&path, false, "wal");
    base.store_text = false;
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        let mut first = doc("d1", 0, metadata(&[]));
        first.text = Some("turquoise dragon".to_string());
        let mut second = doc("d2", 1, metadata(&[]));
        second.text = Some("crimson kraken".to_string());
        engine.upsert_vectors("default", &[first, second]).unwrap();
        engine.persist().unwrap(); // the tvlex base holds both captions
        // Live clear: replace d1's caption with one that tokenizes to nothing.
        engine
            .update_document_payload("default", "d1", None, Some(Some("///".to_string())))
            .unwrap();
        engine.persist().unwrap(); // this delta commit must carry the clear
    }
    // Appender clear: d2 re-captioned to nothing, folded by the next writer.
    {
        let appender = CoreAppender::open(base.clone()).expect("open appender");
        let mut cleared = doc("d2", 1, metadata(&[]));
        cleared.text = Some("///".to_string());
        appender.append_vectors(&[cleared]).expect("append vectors");
    }
    {
        let writer = CoreEngine::open(base.clone()).unwrap();
        drop(writer); // fold + checkpoint truncates the WAL
    }
    let reopened = CoreEngine::open(base).unwrap();
    for term in ["dragon", "kraken"] {
        let plan = reopened.prepare_query_text(term, "lexical").unwrap();
        let hits = reopened
            .search_embedded_text("default", &plan, None, 1, None)
            .unwrap();
        assert!(
            hits.hits.is_empty(),
            "cleared caption resurrected for {term}"
        );
    }
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn text_only_payload_update_survives_the_checkpoint() {
    // A text-only update (no metadata) must mark the document pending: persist()
    // otherwise no-ops and the checkpoint truncates the WAL record carrying the
    // only durable copy. Cover set, replace, and clear against reopens.
    let path = unique_temp_dir("core_text_only_update");
    let mut base = open_options(&path, false, "wal"); // store_text = true
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        let mut captioned = doc("d1", 0, metadata(&[]));
        captioned.text = Some("old caption".to_string());
        engine.upsert_vectors("default", &[captioned]).unwrap();
        engine.persist().unwrap();
        // Replace the caption with a text-only update, then checkpoint.
        engine
            .update_document_payload("default", "d1", None, Some(Some("new caption".to_string())))
            .unwrap();
        engine.persist().unwrap();
    }
    {
        let engine = CoreEngine::open(base.clone()).unwrap();
        assert_eq!(
            engine.get_document_text("default", "d1").unwrap().as_deref(),
            Some("new caption"),
            "a text-only update was truncated away with the WAL"
        );
        let plan = engine.prepare_query_text("caption", "lexical").unwrap();
        let hits = engine
            .search_embedded_text("default", &plan, None, 1, None)
            .unwrap();
        assert_eq!(hits.hits.len(), 1);
    }
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        // Clear the caption entirely; the delta must carry the clear for both the
        // raw-text and lexical sidecars.
        engine
            .update_document_payload("default", "d1", None, Some(None))
            .unwrap();
        engine.persist().unwrap();
    }
    let reopened = CoreEngine::open(base).unwrap();
    assert_eq!(
        reopened.get_document_text("default", "d1").unwrap(),
        None,
        "a cleared raw caption resurrected from the base"
    );
    let plan = reopened.prepare_query_text("caption", "lexical").unwrap();
    let hits = reopened
        .search_embedded_text("default", &plan, None, 1, None)
        .unwrap();
    assert!(hits.hits.is_empty(), "cleared caption tokens resurrected");
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn clear_then_reupsert_still_persists_the_clear() {
    // A replacement upsert runs the same index-removal helper as a delete; an
    // earlier caption clear must survive it, or the delta drops the sidecar
    // delete and the base's old text/tokens resurrect after reopen.
    let path = unique_temp_dir("core_caption_clear_reupsert");
    let mut base = open_options(&path, false, "wal");
    base.acquire_writer_lock = true;
    {
        let mut engine = CoreEngine::open(base.clone()).unwrap();
        create_vector_only_index(&mut engine);
        let mut captioned = doc("d1", 0, metadata(&[]));
        captioned.text = Some("turquoise dragon".to_string());
        engine.upsert_vectors("default", &[captioned]).unwrap();
        engine.persist().unwrap();
        // Clear the caption, then replace the document (still caption-free)
        // before the checkpoint.
        engine
            .update_document_payload("default", "d1", None, Some(None))
            .unwrap();
        engine
            .upsert_vectors("default", &[doc("d1", 2, metadata(&[("kind", "moved")]))])
            .unwrap();
        engine.persist().unwrap();
    }
    let reopened = CoreEngine::open(base).unwrap();
    assert_eq!(
        reopened.get_document_text("default", "d1").unwrap(),
        None,
        "cleared raw caption resurrected past a replacement upsert"
    );
    let plan = reopened.prepare_query_text("dragon", "lexical").unwrap();
    let hits = reopened
        .search_embedded_text("default", &plan, None, 1, None)
        .unwrap();
    assert!(
        hits.hits.is_empty(),
        "cleared caption tokens resurrected past a replacement upsert"
    );
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}

// --- Opt-in ANN (cluster-prune) query path ---
//
// ANN generates candidates; the exact TurboVec scan re-scores them and stays the
// authority. These tests assert the invariants that make the feature safe: exact
// default unchanged, probe-all reproduces exact, hit scores are the exact scores,
// well-separated recall matches exact, new vectors stay findable, tiny corpora
// fall back, and the opt-in config round-trips through persistence.

fn ann_options(clusters: usize, nprobe: usize) -> CoreIndexCreateOptions {
    let mut options = CoreIndexCreateOptions::native_default("default", 8, 4);
    options.ann = Some(CoreAnnOptions {
        algorithm: CoreAnnOptions::CLUSTER.to_string(),
        clusters: Some(clusters),
        nprobe: Some(nprobe),
    });
    options
}

fn vector_doc(id: &str, vector: Vec<f32>) -> CoreVectorDocument {
    CoreVectorDocument {
        document_id: id.to_string(),
        vector,
        metadata: BTreeMap::new(),
        text: None,
        patch_matrix: None,
    }
}

/// Four well-separated blobs on orthogonal axes, five docs each. Each doc leans a
/// little further off its blob's axis (a distinct perpendicular component), so
/// within a blob the cosine to the axis is distinct and there are no score ties.
fn blob_docs() -> Vec<CoreVectorDocument> {
    let axes = [0usize, 2, 4, 6];
    let mut docs = Vec::new();
    for (blob, &axis) in axes.iter().enumerate() {
        for i in 0..5 {
            let mut vector = vec![0.0f32; 8];
            vector[axis] = 1.0;
            vector[axis + 1] = 0.02 * (i as f32 + 1.0);
            docs.push(vector_doc(&format!("b{blob}c{i}"), vector));
        }
    }
    docs
}

fn hit_keys(results: &CoreSearchResults) -> Vec<(String, u32)> {
    results
        .hits
        .iter()
        .map(|hit| (hit.chunk_id.clone(), hit.score.to_bits()))
        .collect()
}

fn axis_query(axis: usize) -> Vec<f32> {
    let mut query = vec![0.0f32; 8];
    query[axis] = 1.0;
    query
}

fn exact_engine(docs: &[CoreVectorDocument]) -> CoreEngine {
    let mut engine = CoreEngine::new_in_memory();
    engine.create_index("default", 8, 4).unwrap();
    engine.upsert_vectors("default", docs).unwrap();
    engine
}

fn ann_engine(docs: &[CoreVectorDocument], clusters: usize, nprobe: usize) -> CoreEngine {
    let mut engine = CoreEngine::new_in_memory();
    engine
        .create_index_with_options(ann_options(clusters, nprobe))
        .unwrap();
    engine.upsert_vectors("default", docs).unwrap();
    engine
}

#[test]
fn ann_probe_all_matches_exact() {
    // nprobe == clusters probes the whole corpus, so ANN must reproduce the exact
    // top-k bit for bit (ids, order, and scores).
    let docs = blob_docs();
    let exact = exact_engine(&docs);
    let ann = ann_engine(&docs, 4, 4);
    for axis in [0usize, 2, 4, 6] {
        let query = axis_query(axis);
        let exact_hits = exact.query_vector("default", &query, 5, None).unwrap();
        let ann_hits = ann.query_vector("default", &query, 5, None).unwrap();
        assert_eq!(
            hit_keys(&exact_hits),
            hit_keys(&ann_hits),
            "probe-all must equal exact for axis {axis}"
        );
    }
}

#[test]
fn ann_recovers_exact_top_k_on_separated_clusters() {
    // With one probe over four well-separated blobs, the query's blob is fully
    // recovered, so ANN returns exactly the exact top-k.
    let docs = blob_docs();
    let exact = exact_engine(&docs);
    let ann = ann_engine(&docs, 4, 1);
    let query = axis_query(0);
    let exact_hits = exact.query_vector("default", &query, 5, None).unwrap();
    let ann_hits = ann.query_vector("default", &query, 5, None).unwrap();
    assert_eq!(hit_keys(&exact_hits), hit_keys(&ann_hits));
    assert!(ann_hits.hits.iter().all(|hit| hit.document_id.starts_with("b0")));
}

#[test]
fn ann_hit_scores_match_exact_rescore() {
    // Every ANN hit carries the exact re-score for its id, proving ANN only picks
    // candidates and never surfaces an approximate centroid score.
    let docs = blob_docs();
    let exact = exact_engine(&docs);
    let ann = ann_engine(&docs, 4, 1);
    let query = axis_query(0);
    let exact_hits = exact.query_vector("default", &query, 5, None).unwrap();
    let exact_scores: BTreeMap<String, u32> = exact_hits
        .hits
        .iter()
        .map(|hit| (hit.chunk_id.clone(), hit.score.to_bits()))
        .collect();
    let ann_hits = ann.query_vector("default", &query, 5, None).unwrap();
    assert!(!ann_hits.hits.is_empty());
    for hit in &ann_hits.hits {
        assert_eq!(
            exact_scores.get(&hit.chunk_id),
            Some(&hit.score.to_bits()),
            "ANN hit {} must have the exact re-score",
            hit.chunk_id
        );
    }
}

#[test]
fn ann_finds_newly_upserted_vector() {
    // A mutation invalidates the cluster cache; the next query rebuilds over the
    // current documents, so a newly-upserted vector is clustered and findable. A
    // stale cache (still holding only the original 20 docs) would leave the new
    // vector out of every posting, so finding it proves the rebuild happened.
    let mut ann = ann_engine(&blob_docs(), 4, 1);
    let query = axis_query(0);
    // First query builds the cluster cache over the original corpus.
    let _ = ann.query_vector("default", &query, 6, None).unwrap();
    // A vector aligned with blob 0; it joins blob 0's cluster on rebuild.
    ann.upsert_vectors("default", &[vector_doc("newtop", axis_query(0))])
        .unwrap();
    let hits = ann.query_vector("default", &query, 6, None).unwrap();
    assert!(
        hits.hits.iter().any(|hit| hit.document_id == "newtop"),
        "a newly-upserted vector must be findable after cache invalidation"
    );
    // And removal drops it again (cache invalidated, and the scan drops absent ids).
    ann.delete_documents("default", &["newtop".to_string()])
        .unwrap();
    let after = ann.query_vector("default", &query, 6, None).unwrap();
    assert!(after.hits.iter().all(|hit| hit.document_id != "newtop"));
}

#[test]
fn ann_returns_full_top_k_when_nearest_cluster_is_small() {
    // Eight orthogonal blobs of five. A single probe reaches only five docs, but
    // top_k=8 must still return eight: the probe set expands to the next cluster
    // rather than clamping to the nearest cluster's size.
    let mut docs = Vec::new();
    for axis in 0..8usize {
        for i in 0..5 {
            let mut vector = vec![0.0f32; 8];
            vector[axis] = 1.0 + 0.01 * i as f32;
            docs.push(vector_doc(&format!("a{axis}c{i}"), vector));
        }
    }
    let ann = ann_engine(&docs, 8, 1);
    let hits = ann.query_vector("default", &axis_query(0), 8, None).unwrap();
    assert_eq!(hits.hits.len(), 8);
}

#[test]
fn ann_batch_queries_match_looping_singles() {
    // With ANN enabled, a batch query must return the same hits as looping the
    // single-query API, for both the struct and the flat-arrays batch paths.
    let docs = blob_docs();
    let ann = ann_engine(&docs, 4, 1);
    let queries: Vec<Vec<f32>> = [0usize, 2, 4, 6].iter().map(|&a| axis_query(a)).collect();

    let batch = ann.query_vectors_batch("default", &queries, 5, None).unwrap();
    for (i, query) in queries.iter().enumerate() {
        let single = ann.query_vector("default", query, 5, None).unwrap();
        assert_eq!(hit_keys(&batch[i]), hit_keys(&single), "batch != single at {i}");
    }

    let flat: Vec<f32> = queries.iter().flatten().copied().collect();
    let arrays = ann
        .query_vectors_batch_arrays("default", &flat, 8, 5, None)
        .unwrap();
    for (i, query) in queries.iter().enumerate() {
        let single = ann.query_vector("default", query, 5, None).unwrap();
        let start = i * arrays.k;
        let ids: Vec<&str> = arrays.document_ids[start..start + single.hits.len()]
            .iter()
            .map(|id| id.as_str())
            .collect();
        let single_ids: Vec<&str> = single
            .hits
            .iter()
            .map(|hit| hit.document_id.as_str())
            .collect();
        assert_eq!(ids, single_ids, "arrays batch != single at {i}");
    }
}

#[test]
fn ann_tiny_corpus_falls_back_to_exact() {
    // Fewer vectors than can form a cluster: ANN must fall back to the exact scan,
    // not panic or drop the single result.
    let mut ann = CoreEngine::new_in_memory();
    ann.create_index_with_options(ann_options(4, 1)).unwrap();
    ann.upsert_vectors("default", &[vector_doc("only", axis_query(0))])
        .unwrap();
    let hits = ann.query_vector("default", &axis_query(0), 3, None).unwrap();
    assert_eq!(hits.hits.len(), 1);
    assert_eq!(hits.hits[0].document_id, "only");
}

#[test]
fn rejects_invalid_ann_options() {
    let cases = [
        ("hnsw", None, None),
        (CoreAnnOptions::CLUSTER, Some(0), None),
        (CoreAnnOptions::CLUSTER, None, Some(0)),
    ];
    for (index, (algorithm, clusters, nprobe)) in cases.into_iter().enumerate() {
        let mut engine = CoreEngine::new_in_memory();
        let mut options =
            CoreIndexCreateOptions::native_default(format!("idx{index}"), 8, 4);
        options.ann = Some(CoreAnnOptions {
            algorithm: algorithm.to_string(),
            clusters,
            nprobe,
        });
        assert!(
            engine.create_index_with_options(options).is_err(),
            "case {index} must be rejected"
        );
    }
}

/// Recursively concatenates every persisted `.json` payload under `dir`.
fn read_persisted_json(dir: &Path) -> String {
    let mut out = String::new();
    let Ok(entries) = fs::read_dir(dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            out.push_str(&read_persisted_json(&path));
        } else if path.extension().is_some_and(|ext| ext == "json") {
            if let Ok(text) = fs::read_to_string(&path) {
                out.push_str(&text);
            }
        }
    }
    out
}

#[test]
fn ann_config_persists_and_survives_reopen() {
    let path = unique_temp_dir("core_ann_reopen");
    let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    let mut options = ann_options(4, 1);
    options.index_key = INDEX_KEY.to_string();
    options.client_id_hash = INDEX_KEY.to_string();
    engine.create_index_with_options(options).unwrap();
    engine.upsert_vectors("default", &blob_docs()).unwrap();
    engine.persist().unwrap();
    // The opt-in config is written to the persisted state header.
    assert!(read_persisted_json(&path).contains("\"ann\""));
    drop(engine);

    let reopened = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    // The load path parsed the persisted ann config and the cluster index
    // rebuilds lazily from the reconstructed rows; the query still recovers the
    // exact blob-0 top-k over this well-separated corpus.
    let hits = reopened
        .query_vector("default", &axis_query(0), 5, None)
        .unwrap();
    assert_eq!(hits.hits.len(), 5);
    assert!(hits.hits.iter().all(|hit| hit.document_id.starts_with("b0")));
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn exact_index_state_header_omits_ann() {
    // An exact-only index must not write the ann key, keeping its state header
    // byte-identical to before the feature.
    let path = unique_temp_dir("core_ann_absent");
    let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    engine.create_index("default", 8, 4).unwrap();
    engine.upsert_vectors("default", &blob_docs()).unwrap();
    engine.persist().unwrap();
    assert!(!read_persisted_json(&path).contains("\"ann\""));
    drop(engine);
    fs::remove_dir_all(path).unwrap();
}

/// Whether any file with `ext` exists anywhere under `dir`.
fn has_file_with_ext(dir: &Path, ext: &str) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    entries.flatten().any(|entry| {
        let path = entry.path();
        if path.is_dir() {
            has_file_with_ext(&path, ext)
        } else {
            path.extension().is_some_and(|found| found == ext)
        }
    })
}

fn ann_durable(path: &Path) -> CoreEngine {
    let mut engine = CoreEngine::open(open_options(path, false, "wal")).unwrap();
    let mut options = ann_options(4, 1);
    options.index_key = INDEX_KEY.to_string();
    options.client_id_hash = INDEX_KEY.to_string();
    engine.create_index_with_options(options).unwrap();
    engine
}

#[test]
fn ann_cluster_index_persists_and_is_adopted_on_reopen() {
    let path = unique_temp_dir("core_ann_sidecar");
    {
        let mut engine = ann_durable(&path);
        engine.upsert_vectors("default", &blob_docs()).unwrap();
        // Query to build the cluster index, then checkpoint it to a `.tvann` base.
        let _ = engine.query_vector("default", &axis_query(0), 5, None).unwrap();
        engine.persist().unwrap();
        assert!(has_file_with_ext(&path, "tvann"), "persist must write a .tvann base");
        drop(engine);
    }
    let reopened = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    // The persisted assignment is adopted on open, before any query rebuilds it.
    assert!(
        reopened.ann_cluster_resident("default").unwrap(),
        "a valid .tvann must be adopted on reopen, skipping the rebuild"
    );
    // And it still serves the correct ANN top-k.
    let hits = reopened
        .query_vector("default", &axis_query(0), 5, None)
        .unwrap();
    assert_eq!(hits.hits.len(), 5);
    assert!(hits.hits.iter().all(|hit| hit.document_id.starts_with("b0")));
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn ingest_only_commit_skips_the_cold_sidecar_build() {
    // A base commit must not force a cold k-means build: an ingest-only session
    // (no ANN query warmed the cache) writes no `.tvann`, and the next reader
    // lazy-builds at its first query instead.
    let path = unique_temp_dir("core_ann_cold_commit");
    {
        let mut engine = ann_durable(&path);
        engine.upsert_vectors("default", &blob_docs()).unwrap();
        engine.persist().unwrap();
        assert!(
            !has_file_with_ext(&path, "tvann"),
            "an ingest-only commit must not build the cluster index"
        );
        drop(engine);
    }
    let reopened = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    assert!(!reopened.ann_cluster_resident("default").unwrap());
    // The first ANN query lazy-builds and still serves the correct top-k.
    let hits = reopened
        .query_vector("default", &axis_query(0), 5, None)
        .unwrap();
    assert_eq!(hits.hits.len(), 5);
    assert!(hits.hits.iter().all(|hit| hit.document_id.starts_with("b0")));
    assert!(reopened.ann_cluster_resident("default").unwrap());
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn ann_reopen_reflects_reembedded_vector() {
    // Regression for stale-clustering adoption: re-embedding a doc under the same
    // chunk id and reopening must serve it at its NEW location. A vector-changing
    // delta invalidates the persisted assignment, so the reopened clustering is
    // rebuilt rather than adopting the stale one.
    let path = unique_temp_dir("core_ann_reembed");
    {
        let mut engine = ann_durable(&path);
        engine.upsert_vectors("default", &blob_docs()).unwrap();
        let _ = engine.query_vector("default", &axis_query(0), 5, None).unwrap();
        engine.persist().unwrap();
        drop(engine);
    }
    {
        let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
        // Move b0c0 from blob 0 (axis 0) to blob 2 (axis 4), same chunk id.
        engine
            .upsert_vectors("default", &[vector_doc("b0c0", axis_query(4))])
            .unwrap();
        engine.persist().unwrap();
        drop(engine);
    }
    let reopened = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    let hits = reopened.query_vector("default", &axis_query(4), 6, None).unwrap();
    assert!(
        hits.hits.iter().any(|hit| hit.document_id == "b0c0"),
        "a re-embedded vector must be served at its new location after reopen"
    );
    fs::remove_dir_all(path).unwrap();
}

#[test]
fn exact_store_writes_no_tvann_sidecar() {
    let path = unique_temp_dir("core_no_tvann");
    let mut engine = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    engine.create_index("default", 8, 4).unwrap();
    engine.upsert_vectors("default", &blob_docs()).unwrap();
    let _ = engine.query_vector("default", &axis_query(0), 5, None).unwrap();
    engine.persist().unwrap();
    assert!(!has_file_with_ext(&path, "tvann"), "an exact store must not write a .tvann");
    drop(engine);
    let reopened = CoreEngine::open(open_options(&path, false, "wal")).unwrap();
    assert!(!reopened.ann_cluster_resident("default").unwrap());
    drop(reopened);
    fs::remove_dir_all(path).unwrap();
}
