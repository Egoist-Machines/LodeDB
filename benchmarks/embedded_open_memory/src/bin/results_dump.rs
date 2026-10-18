//! Prints a canonical dump of what a store returns: stats, a fixed set of
//! vector, hybrid and lexical searches (with and without metadata filters,
//! score bits included), stored texts and a document listing. Two builds
//! agree on a store exactly when their dumps are byte-identical.
//!
//! ```text
//! results-dump <store-dir> [--refresh]
//! results-dump <store-dir> --write
//! ```
//!
//! `--write` opens the store writable first (WAL commit mode, as the Swift
//! binding does), upserts one document with text, deletes one, dumps through
//! that handle, persists and closes, then dumps again through a fresh
//! read-only open. Point it at a copy.

use lodedb_core::engine::CoreEngine;
use lodedb_core::types::{CoreOpenOptions, CoreVectorDocument};
use serde_json::{json, Value};

fn options(path: &str, read_only: bool) -> CoreOpenOptions {
    CoreOpenOptions {
        path: path.to_string(),
        read_only,
        durability: "fsync".to_string(),
        commit_mode: "wal".to_string(),
        store_text: true,
        index_text: true,
        compress_text: true,
        chunk_character_limit: 8192,
        acquire_writer_lock: true,
    }
}

fn query_vector(dim: usize, salt: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| (((i + salt) * 7919) % 200) as f32 / 100.0 - 1.0)
        .collect()
}

fn write_once(path: &str) {
    let mut engine = CoreEngine::open(options(path, false)).expect("writable open");
    let index_id = engine.index_ids().into_iter().next().expect("index");
    let dim = engine.stats(&index_id).expect("stats").vector_dim;
    let mut vector = query_vector(dim, 3);
    let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    vector.iter_mut().for_each(|x| *x /= norm);
    let document = CoreVectorDocument {
        document_id: "mem-compat-0001".to_string(),
        vector,
        metadata: [
            ("category".to_string(), "fact".to_string()),
            ("app".to_string(), "Passport".to_string()),
            ("date".to_string(), "2026-10-15T00:00:00Z".to_string()),
        ]
        .into_iter()
        .collect(),
        text: Some("About the Lisbon trip: booked a window seat on morning flights.".to_string()),
        patch_matrix: None,
    };
    engine
        .upsert_vectors(&index_id, &[document])
        .expect("upsert");
    engine
        .delete_documents(&index_id, &["mem-000002".to_string()])
        .expect("delete");
    dump(&engine, "writable");
    engine.persist().expect("persist");
    engine.close().expect("close");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: results-dump <store-dir> [--refresh|--write]");
    let flag = args.next();
    if flag.as_deref() == Some("--write") {
        write_once(&path);
    }
    let mut engine = CoreEngine::open_readonly(&path, options(&path, true)).expect("open_readonly");
    if flag.as_deref() == Some("--refresh") {
        engine.refresh().expect("refresh");
    }
    dump(&engine, "read-only");
}

fn dump(engine: &CoreEngine, label: &str) {
    let index_id = engine.index_ids().into_iter().next().expect("index");
    println!("handle {label}");
    let stats = engine.stats(&index_id).expect("stats");
    println!(
        "stats documents={} chunks={} generation={} dim={} bits={} model={} applied_lsn={}",
        stats.document_count,
        stats.chunk_count,
        stats.generation,
        stats.vector_dim,
        stats.bit_width,
        stats.model,
        engine.applied_lsn(&index_id).expect("applied_lsn"),
    );

    let filters: [(&str, Option<Value>); 5] = [
        ("none", None),
        (
            "category-in",
            Some(json!({"category": {"$in": ["fact", "preference"]}})),
        ),
        ("app-eq", Some(json!({"app": "Passport"}))),
        ("date-gte", Some(json!({"date": {"$gte": "2025-07-01"}}))),
        (
            "and",
            Some(json!({"$and": [{"category": "event"}, {"app": {"$ne": "Gmail"}}]})),
        ),
    ];
    let texts = [
        "the Lisbon trip window seat",
        "dentist appointment Thursday",
        "oat milk",
    ];
    for (salt, text) in texts.iter().enumerate() {
        let query = query_vector(stats.vector_dim, salt);
        for mode in ["vector", "hybrid", "lexical"] {
            let plan = engine.prepare_query_text(text, mode).expect("plan");
            for (name, filter) in &filters {
                for top_k in [5usize, 60] {
                    let results = engine
                        .search_embedded_text(
                            &index_id,
                            &plan,
                            Some(&query),
                            top_k,
                            filter.as_ref(),
                        )
                        .expect("search");
                    let hits = results
                        .hits
                        .iter()
                        .map(|hit| format!("{}:{:08x}", hit.document_id, hit.score.to_bits()))
                        .collect::<Vec<_>>()
                        .join(" ");
                    println!(
                        "search q={salt} mode={mode} filter={name} k={top_k} considered={} hits={hits}",
                        results.total_considered
                    );
                }
            }
        }
    }

    let ids = [
        "mem-000000",
        "mem-000001",
        "mem-000002",
        "mem-002999",
        "mem-compat-0001",
    ]
    .map(String::from);
    for (id, text) in engine.get_document_texts(&index_id, &ids).expect("texts") {
        println!("text {id} {text}");
    }
    let page = engine
        .list_documents(
            &index_id,
            Some(&json!({"category": "project"})),
            None,
            Some(7),
        )
        .expect("list");
    for document in page {
        println!("list {document}");
    }
}
