use std::collections::BTreeMap;

use lodedb_core::engine::CoreEngine;
use lodedb_core::types::{CoreDocument, CoreVectorDocument};
use serde_json::json;

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
