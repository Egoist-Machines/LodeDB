//! Tests for the local appliance row-reconstruction extension
//! (`reconstruct_rows` / `reconstruct_all` / `rotation_matrix`).
//! See `LOCAL_PATCHES.md`.
//!
//! The score-parity tests pin the coordinate-space contract: an exact
//! inner product between a rotated query (`q @ rotation^T`) and a
//! reconstructed row must reproduce the search kernel's score for that
//! row up to the kernel's uint8 LUT quantization error. GPU-resident
//! exact serving relies on this contract.

use turbovec::{EncodedRowsError, IdMapIndex};

/// Deterministic pseudo-random vectors for one test corpus.
fn vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).max(1);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let value = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32;
        out.push(value / (1u64 << 24) as f32 - 0.5);
    }
    out
}

fn build(n: usize, dim: usize, bit_width: usize) -> IdMapIndex {
    let mut index = IdMapIndex::new(dim, bit_width).unwrap();
    let ids: Vec<u64> = (0..n as u64).collect();
    index.add_with_ids(&vectors(n, dim, 7), &ids).unwrap();
    index
}

/// Exact scores of rotated queries against reconstructed rows:
/// `score[qi][row] = <q_rot[qi], y[row]>` with `q_rot = q @ rotation^T`.
fn exact_scores(
    index: &IdMapIndex,
    queries: &[f32],
    nq: usize,
    dim: usize,
) -> (Vec<u64>, Vec<f64>) {
    let rotation = index.rotation_matrix().expect("dim committed");
    let (ids, rows) = index.reconstruct_all();
    let n = ids.len();
    let mut scores = vec![0.0f64; nq * n];
    for qi in 0..nq {
        let query = &queries[qi * dim..(qi + 1) * dim];
        // q_rot = q @ rotation^T (row-major rotation, matching search()).
        let mut q_rot = vec![0.0f64; dim];
        for (out_d, q_rot_value) in q_rot.iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for in_d in 0..dim {
                acc += f64::from(query[in_d]) * f64::from(rotation[out_d * dim + in_d]);
            }
            *q_rot_value = acc;
        }
        for row in 0..n {
            let y = &rows[row * dim..(row + 1) * dim];
            let mut acc = 0.0f64;
            for d in 0..dim {
                acc += q_rot[d] * f64::from(y[d]);
            }
            scores[qi * n + row] = acc;
        }
    }
    (ids, scores)
}

/// Asserts kernel search scores match exact reconstructed scores within
/// the LUT quantization tolerance, for every returned (id, score) pair.
///
/// The kernel quantizes per-query lookup tables to uint8, so its scores
/// carry a quantization error roughly proportional to `1/sqrt(dim)` of
/// the score scale. Empirically at dim=32 the error sits between 1e-2
/// and 2e-2 of the max score (1e-2 fails, 2e-2 passes); production dims
/// (384/768) are proportionally tighter. The exact reconstructed score
/// is the more faithful estimate of the quantized representation.
fn assert_score_parity(index: &IdMapIndex, dim: usize, k: usize, tolerance: f64) {
    let nq = 4;
    let queries = vectors(nq, dim, 23);
    let (ids, scores) = exact_scores(index, &queries, nq, dim);
    let id_positions: std::collections::HashMap<u64, usize> = ids
        .iter()
        .enumerate()
        .map(|(position, &id)| (id, position))
        .collect();
    let (kernel_scores, kernel_ids) = index.search(&queries, k);
    let score_scale = kernel_scores
        .iter()
        .fold(0.0f64, |acc, &s| acc.max(f64::from(s).abs()))
        .max(1e-6);
    for (slot, &id) in kernel_ids.iter().enumerate() {
        let qi = slot / k;
        let exact = scores[qi * ids.len() + id_positions[&id]];
        let kernel = f64::from(kernel_scores[slot]);
        assert!(
            (kernel - exact).abs() <= tolerance * score_scale,
            "score parity failed for id {id}: kernel {kernel} vs exact {exact} \
             (scale {score_scale}, tolerance {tolerance})"
        );
    }
}

#[test]
fn reconstructed_scores_match_kernel_with_fitted_calibration_4bit() {
    // 1200 rows exceeds the TQ+ sample threshold: fitted calibration.
    let index = build(1200, 32, 4);
    assert!(index.calibration_fitted());
    assert_score_parity(&index, 32, 8, 2e-2);
}

#[test]
fn reconstructed_scores_match_kernel_with_identity_calibration() {
    // 64 rows is below the TQ+ threshold: identity calibration commits.
    let index = build(64, 32, 4);
    assert!(!index.calibration_fitted());
    assert_score_parity(&index, 32, 8, 2e-2);
}

#[test]
fn reconstructed_scores_match_kernel_2bit() {
    let index = build(1200, 32, 2);
    assert_score_parity(&index, 32, 8, 2e-2);
}

#[test]
fn reconstruction_tracks_slot_churn_from_remove_and_upsert() {
    // Removals swap slots and upserts overwrite rows in place; the
    // reconstruction must follow the id mapping, not stale slot order.
    let mut index = build(1200, 32, 4);
    let removed = index.remove_many(&[3, 700, 1199]);
    assert_eq!(removed, 3);
    let replacement = vectors(2, 32, 99);
    let (replaced, appended) = index
        .upsert_with_ids_2d(&replacement, 32, &[10, 5000])
        .unwrap();
    assert_eq!((replaced, appended), (1, 1));
    assert_score_parity(&index, 32, 8, 2e-2);

    // Per-id reconstruction agrees with the all-rows export.
    let (all_ids, all_rows) = index.reconstruct_all();
    let probe_ids = [10u64, 5000u64, 0u64];
    let probe_rows = index.reconstruct_rows(&probe_ids).unwrap();
    for (probe_index, probe_id) in probe_ids.iter().enumerate() {
        let position = all_ids.iter().position(|id| id == probe_id).unwrap();
        assert_eq!(
            &probe_rows[probe_index * 32..(probe_index + 1) * 32],
            &all_rows[position * 32..(position + 1) * 32],
            "per-id reconstruction diverged from reconstruct_all for id {probe_id}"
        );
    }
}

#[test]
fn reconstruct_rows_rejects_unknown_ids() {
    let index = build(16, 16, 4);
    let error = index.reconstruct_rows(&[404]).unwrap_err();
    assert!(matches!(error, EncodedRowsError::UnknownId(404)));
}

#[test]
fn reconstruction_on_uncommitted_index_is_explicit() {
    let index = IdMapIndex::new_lazy(4).unwrap();
    let error = index.reconstruct_rows(&[1]).unwrap_err();
    assert!(matches!(error, EncodedRowsError::DimNotCommitted));
    let (ids, rows) = index.reconstruct_all();
    assert!(ids.is_empty() && rows.is_empty());
    assert!(index.rotation_matrix().is_none());
}
