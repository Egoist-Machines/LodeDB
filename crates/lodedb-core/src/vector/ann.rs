//! Deterministic in-memory cluster-prune (IVF-style) ANN candidate generation.
//!
//! This is the candidate-generation half of the opt-in ANN path. It partitions
//! the corpus into clusters with a deterministic k-means and, at query time,
//! returns the chunk ids in the `nprobe` nearest clusters. The exact TurboVec
//! scan re-scores those candidates and remains the authority, so this layer only
//! affects *which* rows are scored (recall/latency), never the scores themselves.
//!
//! Everything here is deterministic: no clocks, no RNG (`unsafe_code = forbid`
//! and no `rand` dependency), fixed iteration counts, and stable-order tie
//! breaks. The same corpus always yields the same clustering and the same
//! candidates. Vectors are the TurboVec-reconstructed rows (rotated space), and
//! the index stores the matching rotation so it can rotate a raw query into that
//! space itself before scoring centroids. The metric is dot product to match the
//! exact scan.

use std::cmp::Ordering;

/// Lloyd iterations for the cluster build. Candidate generation needs a decent
/// partition, not Lloyd convergence, so a small fixed cap (with an
/// unchanged-assignment early exit) is plenty and keeps the build deterministic.
const MAX_ITERS: usize = 10;

/// A deterministic cluster partition of the corpus for candidate generation.
#[derive(Debug)]
pub struct ClusterIndex {
    dim: usize,
    /// Row-major `num_clusters * dim` centroids, aligned with `postings`.
    centroids: Vec<f32>,
    /// Sorted chunk ids per cluster; every chunk id appears in exactly one.
    postings: Vec<Vec<String>>,
    num_vectors: usize,
    /// Row-major `dim * dim` rotation matching the centroid space, or `None`
    /// when the rows are already in raw space. A raw query is rotated by this
    /// before scoring centroids so it shares the centroids' coordinate space.
    rotation: Option<Vec<f32>>,
}

impl ClusterIndex {
    /// Builds a cluster index from `(chunk_id, vector)` entries.
    ///
    /// `entries` must be in a deterministic order (the caller sorts by chunk id)
    /// and every vector must have length `dim`, in the same (rotated) space as
    /// `rotation` maps a query into. `requested_k` is the desired cluster count,
    /// clamped to `[1, entries.len()]`. Empty clusters are dropped, so
    /// `num_clusters()` counts only clusters that own at least one chunk.
    pub fn build(
        entries: &[(&str, &[f32])],
        dim: usize,
        requested_k: usize,
        rotation: Option<Vec<f32>>,
    ) -> Self {
        let n = entries.len();
        let k = requested_k.clamp(1, n.max(1));

        // Deterministic farthest-first seeding: start from the first (lowest
        // chunk id) row, then repeatedly take the unseeded row least similar to
        // every seed chosen so far. No RNG. Only unseeded rows are considered, so
        // a zero- or low-norm first row (whose dot with everything is near zero)
        // cannot make the search re-pick an already-seeded index and collapse the
        // corpus to a single cluster.
        let mut seeds: Vec<usize> = Vec::with_capacity(k);
        let mut seeded = vec![false; n];
        seeds.push(0);
        seeded[0] = true;
        let mut nearest = vec![f32::NEG_INFINITY; n];
        update_nearest(&mut nearest, entries, entries[0].1);
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
            let Some(pick) = pick else {
                // Every row is already a seed (k >= number of rows).
                break;
            };
            seeds.push(pick);
            seeded[pick] = true;
            update_nearest(&mut nearest, entries, entries[pick].1);
        }

        let cluster_count = seeds.len();
        let mut centroids = vec![0.0f32; cluster_count * dim];
        for (cluster, &row) in seeds.iter().enumerate() {
            centroids[cluster * dim..(cluster + 1) * dim].copy_from_slice(entries[row].1);
        }

        // Lloyd iterations: assign by max dot product, update centroids to the
        // mean of their members. Means accumulate in f64 in a fixed order so the
        // build is bit-reproducible for the same entries.
        let mut assignment = vec![usize::MAX; n];
        for _ in 0..MAX_ITERS {
            let mut changed = false;
            for (i, entry) in entries.iter().enumerate() {
                let cluster = nearest_centroid(entry.1, &centroids, cluster_count, dim);
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
                for (offset, &value) in entry.1.iter().enumerate() {
                    sums[base + offset] += value as f64;
                }
            }
            for (cluster, &count) in counts.iter().enumerate() {
                if count == 0 {
                    // Keep the prior centroid; the cluster may simply be empty
                    // this round and is dropped below if it stays empty.
                    continue;
                }
                let base = cluster * dim;
                let divisor = count as f64;
                for offset in 0..dim {
                    centroids[base + offset] = (sums[base + offset] / divisor) as f32;
                }
            }
        }

        // Materialize postings and drop empty clusters. An empty cluster owns no
        // chunks, so probing it can only waste a probe slot; dropping it keeps
        // `centroids`/`postings` aligned and every probe productive.
        let mut members: Vec<Vec<String>> = vec![Vec::new(); cluster_count];
        for (i, entry) in entries.iter().enumerate() {
            members[assignment[i]].push(entry.0.to_string());
        }
        let mut kept_centroids = Vec::new();
        let mut postings = Vec::new();
        for cluster in 0..cluster_count {
            if members[cluster].is_empty() {
                continue;
            }
            let mut ids = std::mem::take(&mut members[cluster]);
            ids.sort();
            kept_centroids.extend_from_slice(&centroids[cluster * dim..(cluster + 1) * dim]);
            postings.push(ids);
        }

        Self {
            dim,
            centroids: kept_centroids,
            postings,
            num_vectors: n,
            rotation,
        }
    }

    /// Number of non-empty clusters.
    pub fn num_clusters(&self) -> usize {
        self.postings.len()
    }

    /// Number of chunks the index was built over.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Centroid space dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Sorted chunk ids per cluster, for persistence (centroids are not persisted;
    /// they are recomputed from these plus the reconstructed vectors on load).
    pub fn postings(&self) -> &[Vec<String>] {
        &self.postings
    }

    /// Reassembles a cluster index from a persisted assignment, skipping k-means.
    ///
    /// `postings` is the persisted per-cluster chunk-id membership; `entries` is
    /// the live `(chunk_id, vector)` set (rotated space), sorted by chunk id as
    /// [`build`](Self::build) requires. Centroids are recomputed as the per-cluster
    /// means in entry order, so the result is bit-identical to a fresh `build`
    /// (k-means' final centroids are exactly the means of the final assignment).
    /// `rotation` is the live TurboVec rotation, re-derived on load rather than
    /// persisted. Returns `None` when the persisted assignment does not exactly
    /// cover the live entries (stale sidecar) or is malformed, so the caller
    /// rebuilds instead.
    pub fn from_assignment(
        entries: &[(&str, &[f32])],
        dim: usize,
        postings: Vec<Vec<String>>,
        rotation: Option<Vec<f32>>,
    ) -> Option<Self> {
        if dim == 0 || postings.is_empty() {
            return None;
        }
        if rotation.as_ref().is_some_and(|matrix| matrix.len() != dim * dim) {
            return None;
        }
        let posting_total: usize = postings.iter().map(Vec::len).sum();
        // Exact coverage: the persisted assignment must name every live entry and
        // nothing else, so the reloaded clustering equals a fresh build.
        if posting_total != entries.len() {
            return None;
        }
        let centroids = {
            let cluster_count = postings.len();
            let mut cluster_of: std::collections::HashMap<&str, usize> =
                std::collections::HashMap::with_capacity(posting_total);
            for (cluster, ids) in postings.iter().enumerate() {
                for id in ids {
                    if cluster_of.insert(id.as_str(), cluster).is_some() {
                        return None; // a chunk id in two clusters: corrupt
                    }
                }
            }
            let mut sums = vec![0.0f64; cluster_count * dim];
            let mut counts = vec![0usize; cluster_count];
            for entry in entries {
                if entry.1.len() != dim {
                    return None;
                }
                let cluster = *cluster_of.get(entry.0)?; // an unmapped live entry: stale
                counts[cluster] += 1;
                let base = cluster * dim;
                for (offset, &value) in entry.1.iter().enumerate() {
                    sums[base + offset] += value as f64;
                }
            }
            let mut centroids = vec![0.0f32; cluster_count * dim];
            for (cluster, &count) in counts.iter().enumerate() {
                if count == 0 {
                    return None; // a persisted cluster with no live members
                }
                let base = cluster * dim;
                let divisor = count as f64;
                for offset in 0..dim {
                    centroids[base + offset] = (sums[base + offset] / divisor) as f32;
                }
            }
            centroids
        };
        Some(Self {
            dim,
            centroids,
            postings,
            num_vectors: posting_total,
            rotation,
        })
    }

    /// Returns candidate chunk ids for a raw query.
    ///
    /// `raw_query` is in the pre-rotation space; this rotates it into the
    /// centroid space itself. It probes the `nprobe` nearest clusters (ranked by
    /// dot product to their centroid, ties broken by cluster index) and then
    /// keeps taking the next-nearest cluster until the candidate set holds at
    /// least `min_candidates` chunks or every cluster has been probed. Expanding
    /// to `min_candidates` (the caller passes `top_k`) prevents returning fewer
    /// than the requested results when the nearest clusters are small. The result
    /// is sorted and deduplicated (each chunk lives in exactly one cluster).
    pub fn candidate_chunk_ids(
        &self,
        raw_query: &[f32],
        nprobe: usize,
        min_candidates: usize,
    ) -> Vec<String> {
        let cluster_count = self.num_clusters();
        if cluster_count == 0 {
            return Vec::new();
        }
        let nprobe = nprobe.clamp(1, cluster_count);
        let query = match &self.rotation {
            Some(rotation) => rotate(raw_query, rotation, self.dim),
            None => raw_query.to_vec(),
        };
        let mut scored: Vec<(f32, usize)> = (0..cluster_count)
            .map(|cluster| {
                let centroid = &self.centroids[cluster * self.dim..(cluster + 1) * self.dim];
                (dot(&query, centroid), cluster)
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .0
                .partial_cmp(&left.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });
        let mut candidates = Vec::new();
        for (rank, &(_, cluster)) in scored.iter().enumerate() {
            // Always probe at least `nprobe` clusters; keep going only while the
            // candidate set is still short of `min_candidates`.
            if rank >= nprobe && candidates.len() >= min_candidates {
                break;
            }
            candidates.extend(self.postings[cluster].iter().cloned());
        }
        candidates.sort();
        candidates
    }
}

/// Rotates `query` by a row-major `dim * dim` matrix: `out[o] = Σ q[i]·R[o·dim+i]`.
/// Matches the engine's exact-scan query rotation so centroid scoring shares the
/// scan's coordinate space.
fn rotate(query: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for (o, slot) in out.iter_mut().enumerate() {
        let base = o * dim;
        let mut acc = 0.0f32;
        for (i, &value) in query.iter().enumerate().take(dim) {
            acc += value * rotation[base + i];
        }
        *slot = acc;
    }
    out
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn nearest_centroid(vector: &[f32], centroids: &[f32], cluster_count: usize, dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_dot = f32::NEG_INFINITY;
    for cluster in 0..cluster_count {
        let centroid = &centroids[cluster * dim..(cluster + 1) * dim];
        let similarity = dot(vector, centroid);
        // Strict `>` keeps the lowest cluster index on ties, so assignment is
        // deterministic regardless of centroid order.
        if similarity > best_dot {
            best_dot = similarity;
            best = cluster;
        }
    }
    best
}

fn update_nearest(nearest: &mut [f32], entries: &[(&str, &[f32])], centroid: &[f32]) {
    for (slot, entry) in nearest.iter_mut().zip(entries) {
        let similarity = dot(entry.1, centroid);
        if similarity > *slot {
            *slot = similarity;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ClusterIndex;

    /// Builds entries from owned rows for a test corpus.
    fn entries(rows: &[(String, Vec<f32>)]) -> Vec<(&str, &[f32])> {
        let mut refs: Vec<(&str, &[f32])> = rows
            .iter()
            .map(|(id, vector)| (id.as_str(), vector.as_slice()))
            .collect();
        refs.sort_by(|a, b| a.0.cmp(b.0));
        refs
    }

    #[test]
    fn separates_well_separated_blobs() {
        // Two tight blobs on opposite axes; a query near one blob's axis must
        // recover that blob's members from a single probe.
        let mut rows = Vec::new();
        for i in 0..6 {
            rows.push((format!("a{i}"), vec![1.0, 0.02 * i as f32]));
            rows.push((format!("b{i}"), vec![-1.0, 0.02 * i as f32]));
        }
        let refs = entries(&rows);
        let index = ClusterIndex::build(&refs, 2, 2, None);
        assert_eq!(index.num_clusters(), 2);
        let hits = index.candidate_chunk_ids(&[1.0, 0.0], 1, 1);
        assert!(hits.iter().all(|id| id.starts_with('a')));
        assert_eq!(hits.len(), 6);
    }

    #[test]
    fn probe_all_returns_every_chunk() {
        let rows: Vec<(String, Vec<f32>)> = (0..20)
            .map(|i| (format!("c{i:02}"), vec![(i as f32).cos(), (i as f32).sin()]))
            .collect();
        let refs = entries(&rows);
        let index = ClusterIndex::build(&refs, 2, 4, None);
        let mut all = index.candidate_chunk_ids(&[1.0, 0.0], index.num_clusters(), 0);
        all.sort();
        let mut expected: Vec<String> = rows.iter().map(|(id, _)| id.clone()).collect();
        expected.sort();
        assert_eq!(all, expected);
    }

    #[test]
    fn expands_probes_to_reach_min_candidates() {
        // Two tight blobs of six each. One probe reaches only six candidates, so
        // asking for ten must expand to the second cluster and return all twelve.
        let mut rows = Vec::new();
        for i in 0..6 {
            rows.push((format!("a{i}"), vec![1.0, 0.01 * i as f32]));
            rows.push((format!("b{i}"), vec![-1.0, 0.01 * i as f32]));
        }
        let refs = entries(&rows);
        let index = ClusterIndex::build(&refs, 2, 2, None);
        assert_eq!(index.candidate_chunk_ids(&[1.0, 0.0], 1, 1).len(), 6);
        assert_eq!(index.candidate_chunk_ids(&[1.0, 0.0], 1, 10).len(), 12);
    }

    #[test]
    fn deterministic_across_builds() {
        let rows: Vec<(String, Vec<f32>)> = (0..30)
            .map(|i| {
                let a = (i * 7 % 11) as f32;
                let b = (i * 5 % 13) as f32;
                (format!("c{i:02}"), vec![a, b, 1.0])
            })
            .collect();
        let first = ClusterIndex::build(&entries(&rows), 3, 5, None);
        let second = ClusterIndex::build(&entries(&rows), 3, 5, None);
        let query = [2.0, 3.0, 1.0];
        assert_eq!(
            first.candidate_chunk_ids(&query, 2, 1),
            second.candidate_chunk_ids(&query, 2, 1)
        );
    }

    #[test]
    fn rotation_puts_query_in_centroid_space() {
        // Rows live in a swapped-axis space; the rotation swaps a raw query's
        // axes so it selects the cluster its raw form points away from.
        let rows = vec![
            ("hi".to_string(), vec![0.0f32, 1.0]),
            ("lo".to_string(), vec![0.0f32, -1.0]),
        ];
        let refs = entries(&rows);
        // Swap matrix: out[0]=in[1], out[1]=in[0].
        let swap = vec![0.0f32, 1.0, 1.0, 0.0];
        let index = ClusterIndex::build(&refs, 2, 2, Some(swap));
        // Raw query [1,0] rotates to [0,1], which matches the "hi" row.
        let hits = index.candidate_chunk_ids(&[1.0, 0.0], 1, 1);
        assert_eq!(hits, vec!["hi"]);
    }

    #[test]
    fn seeding_does_not_collapse_with_zero_norm_first_row() {
        // The lowest-id row is a zero vector, so its dot with every row is zero.
        // Seeding must still pick a distinct second seed (not fall back onto the
        // zero row) so the corpus does not collapse to a single cluster.
        let rows = vec![
            ("a".to_string(), vec![0.0f32, 0.0]),
            ("b".to_string(), vec![1.0, 0.0]),
            ("c".to_string(), vec![1.0, 0.1]),
            ("d".to_string(), vec![-1.0, 0.0]),
            ("e".to_string(), vec![-1.0, 0.1]),
        ];
        let refs = entries(&rows);
        let index = ClusterIndex::build(&refs, 2, 2, None);
        assert_eq!(index.num_clusters(), 2);
    }

    #[test]
    fn from_assignment_reproduces_build_and_rejects_stale() {
        // Build, then reassemble from the built postings + the same entries: the
        // reloaded index must yield identical candidates (centroids recomputed
        // from the assignment equal the k-means centroids).
        let rows: Vec<(String, Vec<f32>)> = (0..24)
            .map(|i| {
                let a = (i * 3 % 7) as f32;
                let b = (i * 5 % 11) as f32;
                (format!("c{i:02}"), vec![a, b, 1.0])
            })
            .collect();
        let refs = entries(&rows);
        let built = ClusterIndex::build(&refs, 3, 4, None);
        let postings: Vec<Vec<String>> = built.postings().to_vec();
        let reloaded = ClusterIndex::from_assignment(&refs, 3, postings.clone(), None).unwrap();
        let query = [2.0, 3.0, 1.0];
        assert_eq!(
            built.candidate_chunk_ids(&query, 2, 1),
            reloaded.candidate_chunk_ids(&query, 2, 1)
        );
        // A live entry missing from the persisted assignment (a vector added since
        // the sidecar was written) is rejected so the caller rebuilds.
        let mut extra = rows.clone();
        extra.push(("c99".to_string(), vec![1.0, 1.0, 1.0]));
        assert!(ClusterIndex::from_assignment(&entries(&extra), 3, postings, None).is_none());
    }

    #[test]
    fn handles_k_larger_than_corpus_and_identical_vectors() {
        let rows: Vec<(String, Vec<f32>)> = (0..3)
            .map(|i| (format!("same{i}"), vec![0.5, 0.5]))
            .collect();
        let refs = entries(&rows);
        // Request more clusters than rows, with all-identical vectors.
        let index = ClusterIndex::build(&refs, 2, 8, None);
        assert!(index.num_clusters() >= 1 && index.num_clusters() <= 3);
        let mut hits = index.candidate_chunk_ids(&[0.5, 0.5], index.num_clusters(), 0);
        hits.sort();
        assert_eq!(hits, vec!["same0", "same1", "same2"]);
    }
}
