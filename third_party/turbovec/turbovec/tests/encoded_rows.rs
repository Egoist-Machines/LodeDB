//! Tests for the local appliance encoded-row export/import extension
//! (`export_encoded` / `add_encoded` / `remove_many` /
//! `calibration_fingerprint`). See `LOCAL_PATCHES.md`.

use turbovec::{EncodedRowsError, IdMapIndex};

/// Deterministic pseudo-random unit-ish vectors for one test corpus.
fn vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15).max(1);
    let mut out = Vec::with_capacity(n * dim);
    for _ in 0..n * dim {
        // xorshift64*
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let value = (state.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32;
        out.push(value / (1u64 << 24) as f32 - 0.5);
    }
    out
}

fn build(n: usize, dim: usize) -> IdMapIndex {
    let mut index = IdMapIndex::new(dim, 4).unwrap();
    let ids: Vec<u64> = (0..n as u64).collect();
    index.add_with_ids(&vectors(n, dim, 7), &ids).unwrap();
    index
}

#[test]
fn export_then_add_encoded_reproduces_search_results() {
    let index = build(64, 16);
    let ids: Vec<u64> = (0..64).collect();
    let (codes, scales) = index.export_encoded(&ids).unwrap();
    assert_eq!(codes.len(), 64 * index.bytes_per_vector().unwrap());
    assert_eq!(scales.len(), 64);

    // Replay into a fresh index with the same calibrated coordinate
    // system: same dim/bit width and same frozen TQ+ state, obtained by
    // adding the same first batch then removing everything.
    let mut replay = build(64, 16);
    let removed = replay.remove_many(&ids);
    assert_eq!(removed, 64);
    assert_eq!(replay.len(), 0);
    assert_eq!(
        replay.calibration_fingerprint(),
        index.calibration_fingerprint()
    );
    let (replaced, appended) = replay.add_encoded(&ids, &codes, &scales).unwrap();
    assert_eq!((replaced, appended), (0, 64));
    assert_eq!(replay.len(), 64);

    let queries = vectors(4, 16, 7);
    let (orig_scores, orig_ids) = index.search(&queries[..4 * 16], 5);
    let (replay_scores, replay_ids) = replay.search(&queries[..4 * 16], 5);
    assert_eq!(orig_ids, replay_ids);
    assert_eq!(orig_scores, replay_scores);
}

#[test]
fn add_encoded_overwrites_existing_ids_in_place() {
    let mut index = build(32, 16);
    let source = build(32, 16);
    let (codes, scales) = source.export_encoded(&[3]).unwrap();

    let before_len = index.len();
    let (replaced, appended) = index.add_encoded(&[3], &codes, &scales).unwrap();
    assert_eq!((replaced, appended), (1, 0));
    assert_eq!(index.len(), before_len);

    let (codes_new, scales_new) = source.export_encoded(&[5]).unwrap();
    let (replaced, appended) = index
        .add_encoded(&[1000], &codes_new, &scales_new)
        .unwrap();
    assert_eq!((replaced, appended), (0, 1));
    assert_eq!(index.len(), before_len + 1);
    assert!(index.contains(1000));
}

#[test]
fn add_encoded_validates_before_mutating() {
    let mut index = build(16, 16);
    let (codes, scales) = index.export_encoded(&[0, 1]).unwrap();

    let err = index.add_encoded(&[7, 7], &codes, &scales).unwrap_err();
    assert_eq!(err, EncodedRowsError::DuplicateId(7));
    assert_eq!(index.len(), 16);

    let err = index
        .add_encoded(&[7, 8], &codes[..codes.len() - 1], &scales)
        .unwrap_err();
    assert!(matches!(err, EncodedRowsError::CodesLengthMismatch { .. }));

    let err = index
        .add_encoded(&[7, 8], &codes, &[scales[0]])
        .unwrap_err();
    assert!(matches!(err, EncodedRowsError::ScalesCountMismatch { .. }));

    let err = index
        .add_encoded(&[7, 8], &codes, &[scales[0], f32::NAN])
        .unwrap_err();
    assert!(matches!(err, EncodedRowsError::NonFiniteScale { index: 1, .. }));
    assert_eq!(index.len(), 16);
}

#[test]
fn export_encoded_rejects_unknown_ids() {
    let index = build(8, 16);
    let err = index.export_encoded(&[999]).unwrap_err();
    assert_eq!(err, EncodedRowsError::UnknownId(999));
}

#[test]
fn calibration_fingerprint_distinguishes_different_calibrations() {
    // Below TQPLUS_MIN_SAMPLES (1000) the calibration is identity, so two
    // small same-shape indexes share a fingerprint regardless of data.
    let small_a = build(64, 16);
    let small_b = {
        let mut index = IdMapIndex::new(16, 4).unwrap();
        let ids: Vec<u64> = (0..64).collect();
        index.add_with_ids(&vectors(64, 16, 99), &ids).unwrap();
        index
    };
    assert_eq!(
        small_a.calibration_fingerprint(),
        small_b.calibration_fingerprint()
    );

    // At >= 1000 rows the first batch freezes a data-dependent TQ+
    // calibration: same data agrees, different data differs.
    let big = |seed: u64| {
        let mut index = IdMapIndex::new(16, 4).unwrap();
        let ids: Vec<u64> = (0..1100).collect();
        index.add_with_ids(&vectors(1100, 16, seed), &ids).unwrap();
        index
    };
    let a = big(7);
    let b = big(7);
    let c = big(99);
    assert_eq!(a.calibration_fingerprint(), b.calibration_fingerprint());
    assert_ne!(a.calibration_fingerprint(), c.calibration_fingerprint());
}

#[test]
fn remove_many_counts_only_present_ids() {
    let mut index = build(8, 16);
    let removed = index.remove_many(&[0, 1, 999]);
    assert_eq!(removed, 2);
    assert_eq!(index.len(), 6);
}

#[test]
fn mutated_index_searches_correctly_after_encoded_writes() {
    // The blocked SIMD cache must be invalidated by encoded writes:
    // search (which builds/uses the cache), mutate, search again.
    let mut index = build(64, 16);
    let queries = vectors(2, 16, 11);
    let _ = index.search(&queries, 3);

    let source = build(64, 16);
    let (codes, scales) = source.export_encoded(&[10, 11]).unwrap();
    index.add_encoded(&[200, 201], &codes, &scales).unwrap();
    let (_, ids_after) = index.search(&queries, 64 + 2);
    assert!(ids_after.contains(&200) && ids_after.contains(&201));
}

#[test]
fn upsert_with_ids_overwrites_in_place_and_appends() {
    let mut index = build(64, 16);
    let replacement = vectors(2, 16, 31);

    let (replaced, appended) = index
        .upsert_with_ids_2d(&replacement, 16, &[5, 900])
        .unwrap();
    assert_eq!((replaced, appended), (1, 1));
    assert_eq!(index.len(), 65);
    assert!(index.contains(900));

    // The replaced row must rank first for its own (new) vector.
    let (_, ids) = index.search(&replacement[..16], 1);
    assert_eq!(ids[0], 5);

    // Duplicate ids within one call are rejected before any mutation.
    let err = index
        .upsert_with_ids_2d(&replacement, 16, &[7, 7])
        .unwrap_err();
    assert!(matches!(err, turbovec::AddError::IdAlreadyPresent(7)));
    assert_eq!(index.len(), 65);
}

#[test]
fn upsert_with_ids_on_empty_index_behaves_like_add() {
    let mut index = IdMapIndex::new(16, 4).unwrap();
    let batch = vectors(8, 16, 3);
    let ids: Vec<u64> = (0..8).collect();
    let (replaced, appended) = index.upsert_with_ids_2d(&batch, 16, &ids).unwrap();
    assert_eq!((replaced, appended), (0, 8));
    assert_eq!(index.len(), 8);
}

#[test]
fn calibration_fitted_reflects_identity_versus_fitted_state() {
    // Below TQPLUS_MIN_SAMPLES the first add commits identity calibration.
    let small = build(64, 16);
    assert!(!small.calibration_fitted());

    let mut large = IdMapIndex::new(16, 4).unwrap();
    let ids: Vec<u64> = (0..1100).collect();
    large.add_with_ids(&vectors(1100, 16, 7), &ids).unwrap();
    assert!(large.calibration_fitted());
}
