"""Unit tests for the backend-agnostic resident-scan helpers."""

from __future__ import annotations

import numpy as np

from lodedb.engine.turbovec_resident import deterministic_topk_order, tile_row_count


def test_tile_row_count_bounds_and_minimum():
    # score_tile = 4000 // (10*4) = 100; cast_tile = 8000 // (100*4) = 20 -> min 20.
    assert tile_row_count(batch_size=10, dim=100, score_tile_bytes=4000, transient_bytes=8000) == 20
    # Never returns 0 even when the budgets are tiny.
    assert tile_row_count(batch_size=10**6, dim=10**6, score_tile_bytes=1, transient_bytes=1) == 1


def test_deterministic_topk_order_descending_score_then_ascending_id():
    stable_ids = np.array([100, 200, 300, 400], dtype=np.uint64)
    host_slots = np.array([[2, 0, 1], [3, 1, 0]], dtype=np.int64)
    host_scores = np.array([[0.5, 0.9, 0.9], [0.1, 0.2, 0.2]], dtype=np.float32)

    ids, scores = deterministic_topk_order(host_slots, host_scores, stable_ids)

    # Row 0: 0.9(->100), 0.9(->200), 0.5(->300); ties broken by ascending id.
    assert list(ids[0]) == [100, 200, 300]
    assert list(scores[0]) == [0.9, 0.9, 0.5]
    # Row 1: 0.2(->200), 0.2(->100), 0.1(->400); ties -> 100 before 200.
    assert list(ids[1]) == [100, 200, 400]
    assert list(scores[1]) == [0.2, 0.2, 0.1]
    assert ids.dtype == np.uint64
    assert scores.dtype == np.float32
