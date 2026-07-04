//! Off-by-default timing harness for the ANN cluster-index build (issue #71).
//!
//! The build is CPU-bound k-means, so this is `#[ignore]`d and skips itself when
//! no corpus size is set, keeping `cargo test` fast. Run it explicitly:
//!
//! ```text
//! LODEDB_ANN_BENCH_N=200000 cargo test --release -p lodedb-core \
//!     --test ann_build_bench -- --ignored --nocapture ann_kmeans_build_timing
//! ```
//!
//! With `LODEDB_ANN_BENCH_BASELINE=1` it also times a single-threaded,
//! full-corpus reproduction of the pre-#71 build (the O(n^1.5) baseline the issue
//! measured) over the same synthetic corpus, so the speedup is a same-box A/B.
//! Only enable the baseline at the smaller sizes; it is the slow path by design.
//!
//! Tunables: `LODEDB_ANN_BENCH_DIM` (default 384, MiniLM), `LODEDB_ANN_BENCH_BLOBS`
//! (planted cluster count in the synthetic data, default 256).

use std::time::Instant;

use lodedb_core::vector::ann::{ClusterEntry, ClusterIndex};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

/// The engine's default cluster count: `round(sqrt(n))`, clamped to `[1, n]` and
/// capped at 4096 (see `CoreEngine::ann_cluster_count`).
fn default_clusters(n: usize) -> usize {
    ((n as f64).sqrt().round() as usize).clamp(1, n.max(1)).min(4096)
}

/// Deterministic synthetic corpus: `blobs` planted centers, each row a center
/// plus small per-dimension jitter, L2-normalized (real sentence-embedding
/// vectors are ~unit norm and cluster into topics). A tiny LCG gives reproducible
/// pseudo-randomness with no `rand` dependency. Chunk ids are zero-padded so their
/// lexical order matches insertion order (`ClusterIndex::build` wants sorted
/// entries; the engine sorts before calling, so the bench feeds them pre-sorted).
fn synth_corpus(n: usize, dim: usize, blobs: usize) -> (Vec<String>, Vec<f32>) {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        // SplitMix64-ish: cheap, deterministic, decent spread.
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map to [-1, 1).
        (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
    };

    let blobs = blobs.max(1);
    let mut centers = vec![0.0f32; blobs * dim];
    for value in centers.iter_mut() {
        *value = next();
    }

    let mut rows = vec![0.0f32; n * dim];
    let ids: Vec<String> = (0..n).map(|i| format!("c{i:09}")).collect();
    for i in 0..n {
        let center = &centers[(i % blobs) * dim..(i % blobs) * dim + dim];
        let row = &mut rows[i * dim..(i + 1) * dim];
        let mut norm = 0.0f32;
        for (slot, &c) in row.iter_mut().zip(center) {
            let value = c + 0.15 * next();
            *slot = value;
            norm += value * value;
        }
        let norm = norm.sqrt().max(f32::MIN_POSITIVE);
        for slot in row.iter_mut() {
            *slot /= norm;
        }
    }
    (ids, rows)
}

fn as_entries<'a>(ids: &'a [String], rows: &'a [f32], dim: usize) -> Vec<ClusterEntry<'a>> {
    (0..ids.len())
        .map(|i| (ids[i].as_str(), i as u64 + 1, &rows[i * dim..(i + 1) * dim]))
        .collect()
}

// --- Faithful single-threaded reproduction of the pre-#71 build (baseline) ---

fn baseline_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn baseline_nearest(vector: &[f32], centroids: &[f32], k: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for cluster in 0..k {
        let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
        let similarity = baseline_dot(vector, centroid);
        if similarity > best_dot {
            best_dot = similarity;
            best = cluster;
        }
    }
    best
}

/// The old build: single-threaded farthest-first seeding + single-threaded Lloyd
/// over the whole corpus, `round(sqrt(n))` clusters, 10 iterations. Returns the
/// non-empty cluster count so the caller can sanity-check it against the new build.
fn baseline_build(entries: &[ClusterEntry<'_>], dim: usize, requested_k: usize) -> usize {
    const MAX_ITERS: usize = 10;
    let n = entries.len();
    let k = requested_k.clamp(1, n.max(1));

    let mut seeds = vec![0usize];
    let mut seeded = vec![false; n];
    seeded[0] = true;
    let mut nearest = vec![f32::NEG_INFINITY; n];
    let fold = |nearest: &mut [f32], centroid: &[f32]| {
        for (slot, entry) in nearest.iter_mut().zip(entries) {
            let similarity = baseline_dot(entry.2, centroid);
            if similarity > *slot {
                *slot = similarity;
            }
        }
    };
    fold(&mut nearest, entries[0].2);
    while seeds.len() < k {
        let mut pick: Option<usize> = None;
        let mut pick_similarity = f32::INFINITY;
        for (i, &similarity) in nearest.iter().enumerate() {
            if seeded[i] {
                continue;
            }
            if pick.is_none() || similarity < pick_similarity {
                pick_similarity = similarity;
                pick = Some(i);
            }
        }
        let Some(pick) = pick else { break };
        seeds.push(pick);
        seeded[pick] = true;
        fold(&mut nearest, entries[pick].2);
    }

    let cluster_count = seeds.len();
    let mut centroids = vec![0.0f32; cluster_count * dim];
    for (cluster, &row) in seeds.iter().enumerate() {
        centroids[cluster * dim..(cluster + 1) * dim].copy_from_slice(entries[row].2);
    }

    let mut assignment = vec![usize::MAX; n];
    for _ in 0..MAX_ITERS {
        let mut changed = false;
        for (i, entry) in entries.iter().enumerate() {
            let cluster = baseline_nearest(entry.2, &centroids, cluster_count, dim);
            if cluster != assignment[i] {
                assignment[i] = cluster;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        let mut sums = vec![0.0f64; cluster_count * dim];
        let mut counts = vec![0usize; cluster_count];
        for (i, entry) in entries.iter().enumerate() {
            let cluster = assignment[i];
            counts[cluster] += 1;
            let base = cluster * dim;
            for (slot, &value) in sums[base..base + dim].iter_mut().zip(entry.2) {
                *slot += value as f64;
            }
        }
        for (cluster, &count) in counts.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let base = cluster * dim;
            for (slot, &value) in centroids[base..base + dim].iter_mut().zip(&sums[base..base + dim])
            {
                *slot = (value / count as f64) as f32;
            }
        }
    }

    let mut nonempty = vec![false; cluster_count];
    for &cluster in &assignment {
        nonempty[cluster] = true;
    }
    nonempty.iter().filter(|&&live| live).count()
}

#[test]
#[ignore = "CPU-bound k-means timing; run explicitly with LODEDB_ANN_BENCH_N set"]
fn ann_kmeans_build_timing() {
    let n = env_usize("LODEDB_ANN_BENCH_N", 0);
    if n == 0 {
        eprintln!("LODEDB_ANN_BENCH_N unset; skipping the ANN build timing bench.");
        return;
    }
    let dim = env_usize("LODEDB_ANN_BENCH_DIM", 384);
    let blobs = env_usize("LODEDB_ANN_BENCH_BLOBS", 256);
    let run_baseline = env_usize("LODEDB_ANN_BENCH_BASELINE", 0) != 0;
    let k = default_clusters(n);

    let threads = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    eprintln!("ann build bench: n={n} dim={dim} blobs={blobs} clusters(k)={k} threads={threads}");

    let build_start = Instant::now();
    let (ids, rows) = synth_corpus(n, dim, blobs);
    let entries = as_entries(&ids, &rows, dim);
    eprintln!("  corpus generated in {:.2?}", build_start.elapsed());

    let start = Instant::now();
    let index = ClusterIndex::build(&entries, dim, k, None);
    let new_elapsed = start.elapsed();
    eprintln!(
        "  NEW  (parallel):            {new_elapsed:.2?}  -> {} clusters, {} vectors",
        index.num_clusters(),
        index.num_vectors(),
    );
    assert!(index.num_clusters() > 1, "expected a real partition");
    assert_eq!(index.num_vectors(), n, "every row must be clustered exactly once");

    if run_baseline {
        let start = Instant::now();
        let baseline_clusters = baseline_build(&entries, dim, k);
        let baseline_elapsed = start.elapsed();
        eprintln!(
            "  OLD  (single-thread, full): {baseline_elapsed:.2?}  -> {baseline_clusters} clusters"
        );
        let speedup = baseline_elapsed.as_secs_f64() / new_elapsed.as_secs_f64().max(1e-9);
        eprintln!("  speedup: {speedup:.1}x");
    }
}
