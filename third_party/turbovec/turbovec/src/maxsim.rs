//! Late-interaction MaxSim scoring kernel.
//!
//! Scores a multi-vector query against a set of multi-vector documents with
//! MaxSim: for each document, the sum over query tokens of the maximum dot
//! product against that document's patch vectors,
//!
//! ```text
//! score(query, doc) = sum over query tokens t of  max over doc patches p of  <q_t, d_p>
//! ```
//!
//! Vectors are full precision (`f32`) and assumed already L2-normalized by the
//! caller, so each dot product is a cosine similarity. This is the exact rescore
//! late-interaction retrieval (ColBERT / ColPali) runs over a candidate set; the
//! quantized index supplies the candidates, this scores them.
//!
//! Each document is scored independently and in parallel across documents (rayon):
//! its query-token / patch dot products are a small GEMM (`query @ doc^T`) via
//! faer -- the same primitive the search pipeline uses for batched rotation --
//! reduced (max over patches, sum over query tokens) while the score band is still
//! hot in cache. Per-document tiling keeps the intermediate small (one
//! `n_query x patches` band per document) instead of materializing one
//! `n_query x total_patches` matrix, and parallelizes across cores with no Python
//! overhead between candidates.

use rayon::prelude::*;

/// Computes the MaxSim score of `query` against each document.
///
/// `query` is `n_query * dim` row-major. The candidate documents are
/// concatenated row-major in `docs` (`total_patches * dim`), partitioned by
/// `doc_patch_counts` (patches per document, in order). Returns one score per
/// document, in document order.
///
/// A document with zero patches (or a zero-token query) scores `0.0`. Panics if
/// the slice lengths are inconsistent with the shape arguments; the Python
/// binding validates shapes before calling, so that is a guard, not a path.
pub fn maxsim_scores(
    query: &[f32],
    n_query: usize,
    dim: usize,
    docs: &[f32],
    doc_patch_counts: &[usize],
) -> Vec<f32> {
    assert_eq!(query.len(), n_query * dim, "query length mismatch");
    let total_patches: usize = doc_patch_counts.iter().sum();
    assert_eq!(docs.len(), total_patches * dim, "docs length mismatch");

    // Nothing to score: every document contributes 0.0 (empty query or corpus).
    if n_query == 0 || total_patches == 0 {
        return vec![0.0; doc_patch_counts.len()];
    }

    // Per-document patch offset (in rows of `docs`) so each document scores from
    // an independent, disjoint slice -- safe to process in parallel.
    let mut offsets = Vec::with_capacity(doc_patch_counts.len());
    let mut running = 0usize;
    for &count in doc_patch_counts {
        offsets.push(running);
        running += count;
    }

    doc_patch_counts
        .par_iter()
        .zip(offsets.par_iter())
        .map(|(&count, &offset)| score_one(query, n_query, dim, docs, offset, count))
        .collect()
}

/// Scores one document: GEMM `query @ doc^T` into a local band, then sum over
/// query tokens of the per-token max over patches.
fn score_one(
    query: &[f32],
    n_query: usize,
    dim: usize,
    docs: &[f32],
    patch_offset: usize,
    patch_count: usize,
) -> f32 {
    if patch_count == 0 {
        return 0.0;
    }
    let doc = &docs[patch_offset * dim..(patch_offset + patch_count) * dim];

    // band = query @ doc^T, shape (n_query, patch_count), row-major. Small enough
    // to stay in cache for the reduction that follows. faer runs serially here
    // (Parallelism::None): the outer rayon map already saturates the cores across
    // documents, so a nested pool would only contend.
    let mut band = vec![0.0f32; n_query * patch_count];
    {
        let q_ref = faer::mat::from_row_major_slice::<f32, _, _>(query, n_query, dim);
        let d_ref = faer::mat::from_row_major_slice::<f32, _, _>(doc, patch_count, dim);
        let out = faer::mat::from_row_major_slice_mut::<f32, _, _>(
            &mut band,
            n_query,
            patch_count,
        );
        faer::linalg::matmul::matmul(
            out,
            q_ref,
            d_ref.transpose(),
            None,
            1.0_f32,
            faer::Parallelism::None,
        );
    }

    let mut total = 0.0f32;
    for qi in 0..n_query {
        let row = &band[qi * patch_count..(qi + 1) * patch_count];
        let mut best = f32::NEG_INFINITY;
        for &value in row {
            if value > best {
                best = value;
            }
        }
        total += best;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MaxSim of a tiny hand-checked case matches the closed-form value.
    #[test]
    fn scores_match_reference() {
        let dim = 2;
        // Two query tokens: e0 and e1.
        let query = vec![1.0, 0.0, 0.0, 1.0];
        // doc A patches: e0, e1 -> both tokens match perfectly -> 2.0
        // doc B patches: e0      -> token e0 matches (1.0), token e1 best is 0.0 -> 1.0
        let docs = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let counts = vec![2usize, 1usize];
        let scores = maxsim_scores(&query, 2, dim, &docs, &counts);
        assert!((scores[0] - 2.0).abs() < 1e-6);
        assert!((scores[1] - 1.0).abs() < 1e-6);
    }

    /// An empty document scores zero rather than panicking or leaking the seed.
    #[test]
    fn empty_document_scores_zero() {
        let query = vec![1.0, 0.0];
        let docs: Vec<f32> = vec![];
        let counts = vec![0usize];
        let scores = maxsim_scores(&query, 1, 2, &docs, &counts);
        assert_eq!(scores, vec![0.0]);
    }

    /// A mix of empty and non-empty documents reduces each band independently.
    #[test]
    fn handles_mixed_empty_and_nonempty_docs() {
        let dim = 2;
        let query = vec![1.0, 0.0];
        // doc A: e0 -> 1.0 ; doc B: empty -> 0.0 ; doc C: e1 -> 0.0
        let docs = vec![1.0, 0.0, 0.0, 1.0];
        let counts = vec![1usize, 0usize, 1usize];
        let scores = maxsim_scores(&query, 1, dim, &docs, &counts);
        assert!((scores[0] - 1.0).abs() < 1e-6);
        assert_eq!(scores[1], 0.0);
        assert!((scores[2] - 0.0).abs() < 1e-6);
    }
}
