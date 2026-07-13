//! Correctness tests for `IdMapIndex` — the stable-id wrapper.
//!
//! Invariants exercised:
//!   - `add_with_ids` returns `Err` on bad input (length mismatch, duplicate id).
//!   - `remove` returns true/false and keeps `len` consistent.
//!   - After `remove`, search doesn't return the removed id, and every
//!     remaining id still self-queries to itself.
//!   - Remove then re-add with the same id works.
//!   - Internal `slot_to_id` / `id_to_slot` tables stay consistent after
//!     a swap-and-pop (verified indirectly via search correctness).

use turbovec::IdMapIndex;

fn gaussian_normalized(n: usize, dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut uniform = || {
        let raw = (next() >> 40) as u32 | 1;
        raw as f32 / (1u32 << 24) as f32
    };
    let two_pi = 2.0_f32 * std::f32::consts::PI;
    let mut data = vec![0.0f32; n * dim];
    let mut i = 0;
    while i < data.len() {
        let u1 = uniform().max(1e-7);
        let u2 = uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = two_pi * u2;
        data[i] = r * theta.cos();
        if i + 1 < data.len() {
            data[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    for row_i in 0..n {
        let row = &mut data[row_i * dim..(row_i + 1) * dim];
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            let inv = 1.0 / norm;
            for x in row.iter_mut() {
                *x *= inv;
            }
        }
    }
    data
}

#[test]
fn add_with_ids_updates_len_and_contains() {
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_0000);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &[100, 200, 300, 400, 500]).unwrap();

    assert_eq!(idx.len(), 5);
    assert!(idx.contains(300));
    assert!(!idx.contains(999));
}

#[test]
fn search_returns_ids_not_slots() {
    let dim = 256;
    let data = gaussian_normalized(10, dim, 0xA11D_0001);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let ids: Vec<u64> = (1_000_000..1_000_010).collect();
    idx.add_with_ids(&data, &ids).unwrap();

    // Self-query each vector: expect the matching external id as top-1.
    for (i, &expected_id) in ids.iter().enumerate() {
        let q = &data[i * dim..(i + 1) * dim];
        let (_, got_ids) = idx.search(q, 1);
        assert_eq!(got_ids[0], expected_id);
    }
}

#[test]
fn remove_returns_false_for_missing_id() {
    let dim = 128;
    let data = gaussian_normalized(3, dim, 0xA11D_0002);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &[1, 2, 3]).unwrap();

    assert!(!idx.remove(999));
    assert_eq!(idx.len(), 3);
}

#[test]
fn remove_existing_id_shrinks_and_hides_it() {
    let dim = 256;
    let data = gaussian_normalized(10, dim, 0xA11D_0003);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let ids: Vec<u64> = (0..10).map(|i| i as u64 * 7 + 11).collect();
    idx.add_with_ids(&data, &ids).unwrap();

    // Remove the third vector (id = 25, at slot 2).
    let target_id = ids[2];
    assert!(idx.remove(target_id));
    assert_eq!(idx.len(), 9);
    assert!(!idx.contains(target_id));

    // Its own vector should no longer be returned as a top-1 under its id.
    let q = &data[2 * dim..3 * dim];
    let (_, got_ids) = idx.search(q, 9);
    assert!(!got_ids.contains(&target_id));
}

#[test]
fn remaining_ids_still_self_query_after_mixed_removes() {
    let dim = 384;
    let data = gaussian_normalized(20, dim, 0xA11D_0004);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let ids: Vec<u64> = (0..20).map(|i| i as u64 * 100 + 5).collect();
    idx.add_with_ids(&data, &ids).unwrap();

    // Remove a few ids in different orders — some will trigger
    // swap-and-pop, some will be the last vector (no swap).
    idx.remove(ids[7]);   // middle
    idx.remove(ids[19]);  // last
    idx.remove(ids[0]);   // first

    assert_eq!(idx.len(), 17);
    assert!(!idx.contains(ids[7]));
    assert!(!idx.contains(ids[19]));
    assert!(!idx.contains(ids[0]));

    // Every surviving id still maps back to its own vector.
    for (i, &id) in ids.iter().enumerate() {
        if i == 0 || i == 7 || i == 19 {
            continue;
        }
        let q = &data[i * dim..(i + 1) * dim];
        let (_, got_ids) = idx.search(q, 1);
        assert_eq!(
            got_ids[0], id,
            "id {id} (row {i}) no longer self-queries correctly after remove",
        );
    }
}

#[test]
fn remove_then_re_add_same_id_is_allowed() {
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_0005);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &[1, 2, 3, 4, 5]).unwrap();

    assert!(idx.remove(3));
    assert!(!idx.contains(3));

    // Re-add a new vector with id 3.
    let new_vec = gaussian_normalized(1, dim, 0xA11D_BEEF);
    idx.add_with_ids(&new_vec, &[3]).unwrap();
    assert!(idx.contains(3));
    assert_eq!(idx.len(), 5);
}

#[test]
fn add_with_ids_rejects_duplicate_id() {
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_0006);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data[..2 * dim], &[1, 2]).unwrap();
    // Same id "2" already present.
    let err = idx
        .add_with_ids(&data[2 * dim..3 * dim], &[2])
        .unwrap_err();
    assert_eq!(err, turbovec::AddError::IdAlreadyPresent(2));
}

#[test]
fn add_with_ids_rejects_length_mismatch() {
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_0007);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    // 5 vectors, only 3 ids.
    let err = idx.add_with_ids(&data, &[1, 2, 3]).unwrap_err();
    assert_eq!(
        err,
        turbovec::AddError::IdsCountMismatch {
            expected: 5,
            got: 3,
        },
    );
}

#[test]
fn write_and_load_round_trips() {
    let dim = 256;
    let data = gaussian_normalized(10, dim, 0xA11D_0100);
    let ids: Vec<u64> = (2000..2010).collect();

    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    idx.add_with_ids(&data, &ids).unwrap();

    // Delete a few to exercise non-identity slot_to_id mapping.
    idx.remove(2003);
    idx.remove(2007);

    let tmp = std::env::temp_dir().join(format!("turbovec_idmap_{}.tvim", std::process::id()));
    idx.write(&tmp).expect("write failed");

    let restored = IdMapIndex::load(&tmp).expect("load failed");
    assert_eq!(restored.len(), 8);
    assert!(restored.contains(2000));
    assert!(!restored.contains(2003));
    assert!(!restored.contains(2007));

    // Every surviving id should still self-query to itself on the
    // restored index (exercising packed_codes + scales + slot_to_id
    // all round-trip correctly).
    for (i, &id) in ids.iter().enumerate() {
        if id == 2003 || id == 2007 {
            continue;
        }
        let q = &data[i * dim..(i + 1) * dim];
        let (_, got_ids) = restored.search(q, 1);
        assert_eq!(got_ids[0], id, "id {id} failed to self-query after reload");
    }

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn load_rejects_wrong_magic() {
    let tmp = std::env::temp_dir().join(format!(
        "turbovec_idmap_badmagic_{}.tvim",
        std::process::id()
    ));
    // Write a file that starts with the `.tv` format instead of `TVIM`.
    let dim = 64;
    let data = gaussian_normalized(2, dim, 0xA11D_0101);
    let mut inner = IdMapIndex::new(dim, 4).unwrap();
    inner.add_with_ids(&data, &[1, 2]).unwrap();
    // Use the inner TurboQuantIndex's write to produce a .tv file.
    // We can't do that directly since inner is private; simulate with
    // arbitrary bytes of the right shape.
    std::fs::write(&tmp, b"XXXX\x01").expect("write junk");
    let res = IdMapIndex::load(&tmp);
    assert!(res.is_err(), "load should reject file without TVIM magic");
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn add_with_ids_2d_rolls_back_id_tables_on_inner_dim_mismatch() {
    // Regression test for an audit-found bug: `add_with_ids_2d` used to
    // mutate `id_to_slot` / `slot_to_id` BEFORE calling `inner.add_2d`.
    // If the inner call returned `Err(DimMismatch)` (e.g. caller passed
    // wrong dim on a committed-dim index), the ID tables retained `n`
    // ghost entries pointing at slots that don't exist in the inner
    // index — subsequent `search_with_allowlist` would read those
    // ghosts and corrupt further.
    let dim = 128;
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let initial = gaussian_normalized(3, dim, 0xA11D_0DE0);
    idx.add_with_ids_2d(&initial, dim, &[10, 20, 30]).unwrap();
    assert_eq!(idx.len(), 3);

    // Now try to add with the wrong dim — must return DimMismatch and
    // leave ID tables untouched.
    let wrong = gaussian_normalized(2, 64, 0xA11D_0DE1);
    let err = idx.add_with_ids_2d(&wrong, 64, &[40, 50]).unwrap_err();
    assert_eq!(
        err,
        turbovec::AddError::DimMismatch {
            existing: dim,
            got: 64,
        },
    );

    // ID tables must be untouched — len is still 3, the ids 40/50 must
    // NOT be present (the bug would have left them as ghosts).
    assert_eq!(idx.len(), 3);
    assert!(!idx.contains(40));
    assert!(!idx.contains(50));
    // Original ids still resolve correctly.
    assert!(idx.contains(10));
    assert!(idx.contains(20));
    assert!(idx.contains(30));

    // And a subsequent correctly-dim'd add still works (no leftover
    // ghost entries blocking the slots or colliding with the new ids).
    let extra = gaussian_normalized(2, dim, 0xA11D_0DE2);
    idx.add_with_ids_2d(&extra, dim, &[40, 50]).unwrap();
    assert_eq!(idx.len(), 5);
    assert!(idx.contains(40));
    assert!(idx.contains(50));
}


// ---- IdMapIndex audit-driven coverage ----

#[test]
fn add_with_ids_2d_rejects_non_multiple_buffer() {
    // VectorBufferNotMultipleOfDim — reachable only via `add_with_ids_2d`
    // (the non-2d entry point panics earlier on the same condition).
    let mut idx = IdMapIndex::new_lazy(4).unwrap();
    // 17 floats with dim=8 → 17 % 8 != 0.
    let err = idx
        .add_with_ids_2d(&vec![0.0f32; 17], 8, &[1, 2])
        .unwrap_err();
    assert!(
        matches!(err, turbovec::AddError::VectorBufferNotMultipleOfDim { .. }),
        "expected VectorBufferNotMultipleOfDim, got {err:?}",
    );
}

#[test]
fn add_with_ids_2d_rejects_zero_dim() {
    // Same error variant, dim=0 sub-branch.
    let mut idx = IdMapIndex::new_lazy(4).unwrap();
    let err = idx.add_with_ids_2d(&[], 0, &[]).unwrap_err();
    assert!(
        matches!(err, turbovec::AddError::VectorBufferNotMultipleOfDim { .. }),
        "expected VectorBufferNotMultipleOfDim, got {err:?}",
    );
}

#[test]
fn search_returns_descending_scores_aligned_with_ids() {
    // Pins (1) scores are returned non-empty, (2) length matches ids,
    // (3) sorted descending — none of which is asserted in the existing
    // suite. Same #81-shape regression risk.
    let dim = 128;
    let data = gaussian_normalized(20, dim, 0xA11D_5001);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let ids: Vec<u64> = (0..20).map(|i| i as u64 + 1).collect();
    idx.add_with_ids(&data, &ids).unwrap();

    let q = &data[0..dim];
    let (scores, got_ids) = idx.search(q, 5);

    assert_eq!(scores.len(), 5);
    assert_eq!(scores.len(), got_ids.len());
    assert!(scores.iter().all(|s| s.is_finite()));
    for w in scores.windows(2) {
        assert!(w[0] >= w[1], "scores not in descending order: {scores:?}");
    }
}

#[test]
fn search_multi_query_results_are_row_major() {
    // The docstring promises row-major flattening: result i's results
    // live in qi*k..(qi+1)*k. All existing IdMap tests use single
    // queries; multi-query layout is unverified at this layer.
    let dim = 128;
    let data = gaussian_normalized(20, dim, 0xA11D_5002);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let ids: Vec<u64> = (0..20).map(|i| i as u64 + 1).collect();
    idx.add_with_ids(&data, &ids).unwrap();

    // Build two queries: vec0 and vec5. Each should self-match top-1.
    let k = 3;
    let mut queries = Vec::with_capacity(2 * dim);
    queries.extend_from_slice(&data[0..dim]);
    queries.extend_from_slice(&data[5 * dim..6 * dim]);

    let (scores, got_ids) = idx.search(&queries, k);
    assert_eq!(scores.len(), 2 * k);
    assert_eq!(got_ids.len(), 2 * k);
    // Query 0's results live in indices 0..k; query 1's in k..2k.
    assert_eq!(got_ids[0], ids[0], "query 0 top-1 should be id of vec 0");
    assert_eq!(got_ids[k], ids[5], "query 1 top-1 should be id of vec 5");
}

#[test]
fn remove_keeps_swapped_id_addressable_in_both_tables() {
    // After remove(target), the id that was at the last slot moves into
    // target's slot. Pin that the moved id is still reachable via search
    // AND via `contains` — i.e. both `slot_to_id` and `id_to_slot`
    // stayed consistent. A bug updating only one table could mask
    // itself in self-query and only show up here.
    let dim = 128;
    let data = gaussian_normalized(5, dim, 0xA11D_5003);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let ids = [101u64, 202, 303, 404, 505];
    idx.add_with_ids(&data, &ids).unwrap();

    // Remove the second slot; the last slot's id (505) swaps into slot 1.
    assert!(idx.remove(202));

    // Both tables must reflect the swap: contains() and search() agree.
    assert!(idx.contains(505));
    let q = &data[4 * dim..5 * dim];  // the vector that used to be at slot 4
    let (_, got_ids) = idx.search(q, 1);
    assert_eq!(got_ids[0], 505);
    // The moved id is now at slot 1, and the original slot-1 vector
    // (id=202) is gone.
    assert!(!idx.contains(202));
}

#[test]
fn prepare_does_not_change_search_results() {
    // `prepare` is documented as eagerly populating caches; calling it
    // before search must not change the result.
    let dim = 128;
    let data = gaussian_normalized(10, dim, 0xA11D_5004);
    let mut idx = IdMapIndex::new(dim, 4).unwrap();
    let ids: Vec<u64> = (0..10).collect();
    idx.add_with_ids(&data, &ids).unwrap();

    let q = &data[3 * dim..4 * dim];
    let (s_before, ids_before) = idx.search(q, 5);

    // Fresh index, same data, but prepare() first.
    let mut idx2 = IdMapIndex::new(dim, 4).unwrap();
    idx2.add_with_ids(&data, &ids).unwrap();
    idx2.prepare();
    let (s_after, ids_after) = idx2.search(q, 5);

    assert_eq!(ids_before, ids_after);
    assert_eq!(s_before, s_after);
}

#[test]
fn empty_index_round_trip() {
    let dim = 128;
    let idx = IdMapIndex::new(dim, 4).unwrap();

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_idmap_empty_{}.tvim",
        std::process::id()
    ));
    idx.write(&tmp).expect("write failed");

    let restored = IdMapIndex::load(&tmp).expect("load failed");
    assert_eq!(restored.len(), 0);
    assert_eq!(restored.dim(), dim);
    assert_eq!(restored.bit_width(), 4);
    std::fs::remove_file(&tmp).ok();
}

fn sorted_pairs(scores: Vec<f32>, ids: Vec<u64>) -> Vec<(u64, u32)> {
    let mut pairs: Vec<_> = ids
        .into_iter()
        .zip(scores)
        .map(|(id, score)| (id, score.to_bits()))
        .collect();
    pairs.sort_unstable_by_key(|(id, _)| *id);
    pairs
}

#[test]
fn reorder_to_ids_preserves_searches_round_trip_and_fingerprint() {
    let dim = 64;
    let n = 48;
    let data = gaussian_normalized(n, dim, 0xA11D_7001);
    let ids: Vec<u64> = (0..n).map(|row| 10_000 + row as u64 * 17).collect();
    let mut index = IdMapIndex::new(dim, 4).unwrap();
    index.add_with_ids(&data, &ids).unwrap();

    let mut queries = data[..2 * dim].to_vec();
    queries.extend_from_slice(&data[17 * dim..18 * dim]);
    let allowlist: Vec<u64> = ids.iter().copied().step_by(3).collect();
    let full_before = index.search(&queries, 12);
    let masked_before = index.search_with_allowlist(&queries, 12, Some(&allowlist));
    let fingerprint = index.calibration_fingerprint();

    let mut reordered_ids = ids.clone();
    let mut state = 0xD1CE_CAFE_u64;
    for slot in (1..reordered_ids.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        reordered_ids.swap(slot, (state as usize) % (slot + 1));
    }
    index.reorder_to_ids(&reordered_ids).unwrap();

    assert_eq!(index.reconstruct_all().0, reordered_ids);
    assert_eq!(index.calibration_fingerprint(), fingerprint);
    let full_after = index.search(&queries, 12);
    let masked_after = index.search_with_allowlist(&queries, 12, Some(&allowlist));
    let full_after_pairs = sorted_pairs(full_after.0, full_after.1);
    assert_eq!(
        sorted_pairs(full_before.0, full_before.1),
        full_after_pairs
    );
    assert_eq!(
        sorted_pairs(masked_before.0, masked_before.1),
        sorted_pairs(masked_after.0, masked_after.1)
    );

    let tmp = std::env::temp_dir().join(format!(
        "turbovec_idmap_reorder_{}_{}.tvim",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    index.write(&tmp).unwrap();
    let restored = IdMapIndex::load(&tmp).unwrap();
    assert_eq!(restored.reconstruct_all().0, reordered_ids);
    assert_eq!(restored.calibration_fingerprint(), fingerprint);
    let restored_results = restored.search(&queries, 12);
    assert_eq!(
        full_after_pairs,
        sorted_pairs(restored_results.0, restored_results.1)
    );
    std::fs::remove_file(&tmp).ok();
}

#[test]
fn reorder_to_ids_identity_and_invalid_orders_leave_index_usable() {
    let dim = 64;
    let data = gaussian_normalized(8, dim, 0xA11D_7002);
    let ids: Vec<u64> = (1..=8).collect();
    let mut index = IdMapIndex::new(dim, 4).unwrap();
    index.add_with_ids(&data, &ids).unwrap();
    let query = &data[..dim];
    let expected = index.search(query, 4);
    let order = index.reconstruct_all().0;

    index.reorder_to_ids(&order).unwrap();
    assert_eq!(index.reconstruct_all().0, order);

    let wrong_length = index.reorder_to_ids(&order[..order.len() - 1]).unwrap_err();
    assert_eq!(
        wrong_length,
        turbovec::TurboVecError::IdsCountMismatch {
            expected: order.len(),
            got: order.len() - 1,
        }
    );
    let mut unknown = order.clone();
    unknown[0] = 999;
    assert_eq!(
        index.reorder_to_ids(&unknown).unwrap_err(),
        turbovec::TurboVecError::UnknownId(999)
    );
    let mut duplicate = order.clone();
    duplicate[1] = duplicate[0];
    assert_eq!(
        index.reorder_to_ids(&duplicate).unwrap_err(),
        turbovec::TurboVecError::DuplicateId(duplicate[0])
    );
    assert_eq!(index.reconstruct_all().0, order);
    assert_eq!(index.search(query, 4), expected);
}

#[test]
fn reorder_to_ids_stays_consistent_with_swap_removes() {
    let dim = 64;
    let data = gaussian_normalized(12, dim, 0xA11D_7003);
    let ids: Vec<u64> = (100..112).collect();
    let mut index = IdMapIndex::new(dim, 4).unwrap();
    index.add_with_ids(&data, &ids).unwrap();

    assert!(index.remove(103));
    assert!(index.remove(109));
    let mut order = index.reconstruct_all().0;
    order.reverse();
    index.reorder_to_ids(&order).unwrap();
    assert!(index.remove(order[2]));

    let live = index.reconstruct_all().0;
    for (row, id) in ids.iter().enumerate() {
        if !live.contains(id) {
            continue;
        }
        let query = &data[row * dim..(row + 1) * dim];
        assert_eq!(index.search(query, 1).1, vec![*id]);
    }
}
