//! Rerankers — pure ranking functions ported from Graphiti's
//! `graphiti_core/search/search_utils.py` (`rrf`, `maximal_marginal_relevance`,
//! `node_distance_reranker`, `episode_mentions_reranker`).
//!
//! These operate on already-retrieved candidate id lists / scores and are
//! independent of the store, so they port to Rust as pure functions. `v1` of the
//! facade uses hybrid seeds + exact k-hop directly; these are wired in as optional
//! rerankers for `search_subgraph`/`semantic_*` and are the natural extension point
//! for center-node and diversity reranking.
//!
//! IMPLEMENTATION NOTE (Wave 1c): port each function faithfully. `rrf` is
//! reciprocal-rank fusion over several ranked id lists; `maximal_marginal_relevance`
//! trades relevance against diversity given candidate embeddings; the graph-aware
//! rerankers take auxiliary inputs (a center node's neighbour distances, per-id
//! episode-mention counts) the facade supplies.

use std::collections::HashMap;

/// Reciprocal Rank Fusion over several ranked id lists. `k` is the RRF constant
/// (Graphiti uses 1; higher flattens). Returns ids ordered by fused score desc.
///
/// Ported from `search_utils.rrf` (search_utils.py ~L1780):
/// ```text
/// scores[uuid] += 1 / (i + rank_const)   # i is the 0-based rank in each list
/// sort by score descending
/// ```
/// `k` maps to Graphiti's `rank_const` (default 1). Ties preserve first-seen
/// order — Python's `dict` keeps insertion order and its `sort(reverse=True)` is
/// stable, so the first list/position in which an id appears breaks ties.
pub fn rrf(ranked_lists: &[Vec<String>], k: f32) -> Vec<(String, f32)> {
    let mut scores: HashMap<String, f32> = HashMap::new();
    // Track first-seen order to mirror Python dict insertion order for stable ties.
    let mut order: Vec<String> = Vec::new();

    for list in ranked_lists {
        for (i, id) in list.iter().enumerate() {
            let inc = 1.0 / (i as f32 + k);
            if !scores.contains_key(id) {
                order.push(id.clone());
            }
            *scores.entry(id.clone()).or_insert(0.0) += inc;
        }
    }

    let mut scored: Vec<(String, f32)> = order
        .into_iter()
        .map(|id| {
            let s = scores[&id];
            (id, s)
        })
        .collect();

    // Stable sort by fused score descending; equal scores keep first-seen order.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scored
}

/// Maximal Marginal Relevance: reorder `candidates` (id, embedding) trading off
/// relevance against diversity, with `lambda` in [0,1].
///
/// Ported from `search_utils.maximal_marginal_relevance` (search_utils.py ~L1901).
/// Graphiti L2-normalizes each candidate, builds a pairwise cosine-similarity
/// matrix with a **zero diagonal**, then scores each candidate **once** (this is a
/// one-shot scorer, not the iterative textbook MMR):
/// ```text
/// max_sim   = max over the row (includes the self/diagonal 0.0)
/// mmr_score = lambda * dot(query, norm(cand)) + (lambda - 1) * max_sim
///           = lambda * relevance - (1 - lambda) * max_sim
/// sort by mmr_score descending
/// ```
/// Since Graphiti normalizes candidates before the dot, its candidate-candidate
/// term is exactly cosine similarity; we express relevance and pairwise sim through
/// the [`cosine`] helper (identical to Graphiti when the query is unit-length).
pub fn maximal_marginal_relevance(
    query: &[f32],
    candidates: &[(String, Vec<f32>)],
    lambda: f32,
) -> Vec<String> {
    let n = candidates.len();

    // Pairwise similarity matrix with a zero diagonal (matches np.zeros init +
    // only-below-diagonal fill: sim[i][i] stays 0.0).
    let mut sim = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..i {
            let s = cosine(&candidates[i].1, &candidates[j].1);
            sim[i][j] = s;
            sim[j][i] = s;
        }
    }

    let mut scores: Vec<(usize, f32)> = Vec::with_capacity(n);
    for (i, (_, emb)) in candidates.iter().enumerate() {
        // max over the whole row — including the diagonal 0.0.
        let max_sim = sim[i]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let relevance = cosine(query, emb);
        let mmr = lambda * relevance + (lambda - 1.0) * max_sim;
        scores.push((i, mmr));
    }

    // Stable sort by mmr descending; ties keep original (first-seen) order.
    scores.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    scores
        .into_iter()
        .map(|(i, _)| candidates[i].0.clone())
        .collect()
}

/// Rerank `ids` by graph distance from a center node, given precomputed hop
/// distances (id → hops). Nearer ranks higher; ties keep input order.
///
/// Ported from the ranking core of `search_utils.node_distance_reranker`
/// (search_utils.py ~L1798): `filtered_uuids.sort(key=lambda u: scores[u])` where a
/// missing id is assigned `float('inf')`. We decouple the DB fetch — the caller
/// supplies `distances` — and reproduce the ascending, missing-last, stable sort.
pub fn node_distance_reranker(
    ids: &[String],
    distances: &HashMap<String, usize>,
) -> Vec<String> {
    let mut out = ids.to_vec();
    // usize::MAX is the +infinity sentinel for ids with no known distance.
    out.sort_by_key(|id| distances.get(id).copied().unwrap_or(usize::MAX));
    out
}

/// Rerank `ids` by how many episodes mention each (id → count), most-mentioned
/// first. Ties keep input order.
///
/// Decoupled from `search_utils.episode_mentions_reranker` (search_utils.py
/// ~L1860): the caller supplies precomputed `mention_counts` in place of the DB
/// MENTIONS count, and the ranking is by count **descending** (more mentions =
/// better), a missing id treated as `0`, stable for ties.
pub fn episode_mentions_reranker(
    ids: &[String],
    mention_counts: &HashMap<String, usize>,
) -> Vec<String> {
    let mut out = ids.to_vec();
    // Descending by count via a stable sort; missing id → 0 (ranks last).
    out.sort_by(|a, b| {
        let ca = mention_counts.get(a).copied().unwrap_or(0);
        let cb = mention_counts.get(b).copied().unwrap_or(0);
        cb.cmp(&ca)
    });
    out
}

/// Cosine similarity of two vectors: `dot(a, b) / (‖a‖ · ‖b‖)`.
///
/// Returns `0.0` when either vector has zero norm (guards divide-by-zero),
/// mirroring Graphiti's `normalize_l2`, which leaves a zero vector unchanged so a
/// subsequent dot is `0.0`.
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(v: &str) -> String {
        v.to_string()
    }

    #[test]
    fn rrf_rewards_agreement_across_lists_and_is_deterministic() {
        // "a" is rank 0 in both lists; "b" is high only in list 1.
        let list1 = vec![s("a"), s("b"), s("c")];
        let list2 = vec![s("a"), s("c"), s("d")];
        let fused = rrf(&[list1.clone(), list2.clone()], 1.0);

        let score_of = |id: &str| fused.iter().find(|(u, _)| u == id).map(|(_, sc)| *sc).unwrap();

        // Agreement in both lists beats being high in only one.
        assert!(
            score_of("a") > score_of("b"),
            "id high in both lists must beat id high in only one: {fused:?}"
        );
        // a = 1/1 + 1/1 = 2.0 ; c = 1/3 + 1/2 ≈ 0.833 ; b = 1/2 = 0.5 ; d = 1/3 ≈ 0.333
        let order: Vec<&str> = fused.iter().map(|(u, _)| u.as_str()).collect();
        assert_eq!(order, vec!["a", "c", "b", "d"], "fused order: {fused:?}");
        assert!((score_of("a") - 2.0).abs() < 1e-6);

        // Deterministic: same inputs → identical result.
        let again = rrf(&[list1, list2], 1.0);
        assert_eq!(fused, again);
    }

    #[test]
    fn mmr_low_lambda_breaks_up_near_duplicates() {
        // query points along axis 0.
        let query = vec![1.0f32, 0.0, 0.0];
        // c1 & c2 are near-duplicates (both ~axis 0); c3 is orthogonal & diverse.
        let candidates = vec![
            (s("c1"), vec![1.0f32, 0.0, 0.0]),
            (s("c2"), vec![0.98f32, 0.2, 0.0]),
            (s("c3"), vec![0.0f32, 0.0, 1.0]),
        ];

        // Low lambda → diversity dominates.
        let diverse = maximal_marginal_relevance(&query, &candidates, 0.1);

        // The two near-duplicates must NOT occupy the top two slots.
        let top2: std::collections::HashSet<&str> =
            diverse[..2].iter().map(|x| x.as_str()).collect();
        let both_dups: std::collections::HashSet<&str> = ["c1", "c2"].into_iter().collect();
        assert_ne!(top2, both_dups, "low-lambda MMR ranked both near-dups first: {diverse:?}");
        // The diverse candidate is pulled to the front.
        assert_eq!(diverse[0], "c3", "diverse candidate should lead: {diverse:?}");

        // Sanity: high lambda (pure relevance) keeps the two relevant dups first.
        let relevant = maximal_marginal_relevance(&query, &candidates, 1.0);
        assert_eq!(relevant, vec![s("c1"), s("c2"), s("c3")]);
    }

    #[test]
    fn node_distance_reranker_orders_by_ascending_hops_missing_last() {
        let ids = vec![s("a"), s("b"), s("c"), s("d")];
        let mut distances = HashMap::new();
        distances.insert(s("a"), 2usize);
        distances.insert(s("b"), 0usize);
        distances.insert(s("c"), 1usize);
        // "d" is missing → treated as +infinity → last.

        let ranked = node_distance_reranker(&ids, &distances);
        assert_eq!(ranked, vec![s("b"), s("c"), s("a"), s("d")]);
    }

    #[test]
    fn episode_mentions_reranker_orders_by_descending_count_missing_zero() {
        let ids = vec![s("a"), s("b"), s("c")];
        let mut counts = HashMap::new();
        counts.insert(s("a"), 1usize);
        counts.insert(s("b"), 3usize);
        // "c" is missing → treated as 0 → last.

        let ranked = episode_mentions_reranker(&ids, &counts);
        assert_eq!(ranked, vec![s("b"), s("a"), s("c")]);
    }

    #[test]
    fn cosine_guards_zero_norm() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[0.0, 0.0], &[1.0, 0.0]).abs() < 1e-6);
        assert!((cosine(&[1.0, 0.0], &[0.0, 1.0])).abs() < 1e-6);
    }

    #[test]
    fn rrf_ties_preserve_first_seen_order() {
        // x and y each score 1/1 in exactly one list → tie at 1.0.
        // x is seen first (list 0), so it must sort before y.
        let list1 = vec![s("x")];
        let list2 = vec![s("y")];
        let fused = rrf(&[list1, list2], 1.0);
        let order: Vec<&str> = fused.iter().map(|(u, _)| u.as_str()).collect();
        assert_eq!(order, vec!["x", "y"]);
    }
}
