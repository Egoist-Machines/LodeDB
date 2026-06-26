use std::time::Instant;

use lodedb_core::vector::index::CoreVectorChunk;
use lodedb_core::vector::turbovec::TurboVecNativeIndex;
use serde_json::json;

const DIM: usize = 64;
const N: usize = 512;
const NQ: usize = 64;

fn main() {
    let chunks = chunks();

    let started = Instant::now();
    let index = TurboVecNativeIndex::build(&chunks, DIM, 4, 1).expect("build");
    let build_ms = elapsed_ms(started);

    let queries = (0..NQ)
        .map(|i| chunks[(i * 7) % chunks.len()].embedding.clone())
        .collect::<Vec<_>>();

    let started = Instant::now();
    let mut single_checksum = 0u64;
    for query in &queries {
        single_checksum ^= index.search(query, 10, &[]).expect("single")[0].stable_id;
    }
    let single_ms = elapsed_ms(started);

    let allowlist = (0..N)
        .step_by(4)
        .map(|i| chunks[i].chunk_id.clone())
        .collect::<Vec<_>>();
    let started = Instant::now();
    let filtered = index.search(&queries[0], 10, &allowlist).expect("filtered");
    let filtered_ms = elapsed_ms(started);

    let started = Instant::now();
    let batch = index.search_batch(&queries, 10, &[]).expect("batch");
    let batch_ms = elapsed_ms(started);
    let batch_checksum = batch
        .iter()
        .filter_map(|row| row.first())
        .fold(0u64, |acc, hit| acc ^ hit.stable_id);

    let path = std::env::temp_dir().join(format!(
        "lodedb_core_turbovec_smoke_{}.tvim",
        std::process::id()
    ));
    let write = index.write(&path).expect("write");
    let started = Instant::now();
    let loaded = TurboVecNativeIndex::load(&path, &chunks, 1).expect("load");
    let load_ms = elapsed_ms(started);
    std::fs::remove_file(&path).ok();

    println!(
        "{}",
        json!({
            "suite": "native_migration_turbovec_adapter_smoke",
            "vectors": N,
            "queries": NQ,
            "dim": DIM,
            "bit_width": 4,
            "build_ms": build_ms,
            "single_query_total_ms": single_ms,
            "batch_query_total_ms": batch_ms,
            "filtered_query_ms": filtered_ms,
            "filtered_hits": filtered.len(),
            "single_checksum": single_checksum,
            "batch_checksum": batch_checksum,
            "snapshot_bytes": write.snapshot_bytes,
            "load_ms": load_ms,
            "loaded_vectors": loaded.len(),
        })
    );
}

fn chunks() -> Vec<CoreVectorChunk> {
    (0..N)
        .map(|i| {
            CoreVectorChunk::new(
                format!("doc-{i}:chunk-0"),
                format!("doc-{i}"),
                normalized_row(i as u64 | 1),
            )
        })
        .collect()
}

fn normalized_row(mut state: u64) -> Vec<f32> {
    let mut row = vec![0.0f32; DIM];
    for value in &mut row {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *value = ((state >> 40) as f32 / (1u32 << 24) as f32) - 0.5;
    }
    let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut row {
            *value /= norm;
        }
    }
    row
}

fn elapsed_ms(started: Instant) -> f64 {
    (started.elapsed().as_secs_f64() * 100_000.0).round() / 100.0
}
