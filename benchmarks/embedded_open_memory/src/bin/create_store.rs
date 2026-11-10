//! Deterministic synthetic 10k-document, 384-dimensional memory benchmark input.
//! Usage: create_store <new-store-dir>
use lodedb_core::engine::CoreEngine;
use lodedb_core::types::{CoreOpenOptions, CoreVectorDocument};

fn main() {
    let path = std::env::args().nth(1).expect("new store directory");
    assert!(
        !std::path::Path::new(&path).exists(),
        "store already exists"
    );
    let mut engine = CoreEngine::open(CoreOpenOptions {
        path,
        read_only: false,
        durability: "fsync".into(),
        commit_mode: "wal".into(),
        store_text: true,
        index_text: true,
        compress_text: true,
        chunk_character_limit: 8192,
        acquire_writer_lock: true,
    })
    .unwrap();
    engine.create_index("default", 384, 4).unwrap();
    let mut seed = 134_u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 32) as u32
    };
    let phrases = [
        "the Lisbon trip window seat",
        "dentist appointment Thursday",
        "oat milk",
        "morning flights",
        "home office",
        "business day",
    ];
    let documents: Vec<_> = (0..10_000)
        .map(|i| {
            let mut vector: Vec<f32> = (0..384)
                .map(|_| (next() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            vector.iter_mut().for_each(|x| *x /= norm);
            let mut text = phrases[i % phrases.len()].to_string();
            for _ in 0..60 {
                text.push_str(&format!(" term{}", next() % 4096));
            }
            CoreVectorDocument {
                document_id: format!("mem-{i:06}"),
                vector,
                metadata: [(
                    "category".into(),
                    ["fact", "preference", "event", "project"][i % 4].into(),
                )]
                .into_iter()
                .collect(),
                text: Some(text),
                patch_matrix: None,
            }
        })
        .collect();
    engine.upsert_vectors("default", &documents).unwrap();
    engine.persist().unwrap();
    engine.close().unwrap();
}
