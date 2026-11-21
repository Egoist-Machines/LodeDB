//! Off-by-default timing harness for a cold base persist (issue #89).
//!
//! A first persist rewrites every base artifact, so its cost scales with the
//! corpus. This is `#[ignore]`d and skips itself when no corpus size is set,
//! keeping `cargo test` fast. Run it explicitly:
//!
//! ```text
//! LODEDB_PERSIST_BENCH_N=1000000 cargo test --release -p lodedb-core \
//!     --test persist_bench -- --ignored --nocapture cold_base_persist_timing
//! ```
//!
//! The vector dimension is deliberately small by default: the state base's cost
//! is per document, not per dimension, so a narrow vector isolates it from the
//! `.tvim` encode.
//!
//! Tunables: `LODEDB_PERSIST_BENCH_DIM` (default 8), `LODEDB_PERSIST_BENCH_META`
//! (metadata keys per document, default 0, matching the vector-only store the
//! issue measured).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use lodedb_core::engine::CoreEngine;
use lodedb_core::types::{CoreOpenOptions, CoreVectorDocument};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn bench_dir() -> PathBuf {
    let path = std::env::temp_dir().join(format!("lodedb-persist-bench-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn dir_bytes(path: &std::path::Path) -> u64 {
    let mut total = 0;
    for entry in std::fs::read_dir(path).into_iter().flatten().flatten() {
        let meta = entry.metadata().unwrap();
        total += if meta.is_dir() {
            dir_bytes(&entry.path())
        } else {
            meta.len()
        };
    }
    total
}

#[test]
#[ignore = "timing harness; set LODEDB_PERSIST_BENCH_N and run with --ignored"]
fn cold_base_persist_timing() {
    let n = env_usize("LODEDB_PERSIST_BENCH_N", 0);
    if n == 0 {
        eprintln!("LODEDB_PERSIST_BENCH_N unset; skipping");
        return;
    }
    let dim = env_usize("LODEDB_PERSIST_BENCH_DIM", 8);
    let meta_keys = env_usize("LODEDB_PERSIST_BENCH_META", 0);
    let dir = bench_dir();

    let mut engine = CoreEngine::open(CoreOpenOptions {
        path: dir.to_string_lossy().to_string(),
        read_only: false,
        durability: "relaxed".to_string(),
        // Generation mode: a WAL would add an append per batch and is truncated
        // by the persist under test anyway.
        commit_mode: "generation".to_string(),
        store_text: false,
        index_text: false,
        compress_text: false,
        chunk_character_limit: 900,
        acquire_writer_lock: false,
    })
    .unwrap();
    engine.create_index("default", dim, 4).unwrap();

    let ingest = Instant::now();
    let batch = 50_000;
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for start in (0..n).step_by(batch) {
        let documents = (start..(start + batch).min(n))
            .map(|row| {
                let vector = (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect();
                let metadata = (0..meta_keys)
                    .map(|key| (format!("field_{key}"), format!("value-{}", row % 97)))
                    .collect::<BTreeMap<_, _>>();
                CoreVectorDocument {
                    document_id: format!("doc-{row:010}"),
                    vector,
                    metadata,
                    text: None,
                    patch_matrix: None,
                }
            })
            .collect::<Vec<_>>();
        engine.upsert_vectors("default", &documents).unwrap();
    }
    let ingest_seconds = ingest.elapsed().as_secs_f64();

    let persist = Instant::now();
    engine.persist().unwrap();
    let persist_seconds = persist.elapsed().as_secs_f64();

    eprintln!(
        "persist_bench n={n} dim={dim} meta_keys={meta_keys} ingest_s={ingest_seconds:.2} \
         persist_s={persist_seconds:.2} rows_per_s={:.0} store_bytes={}",
        n as f64 / persist_seconds,
        dir_bytes(&dir),
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&dir);
}
