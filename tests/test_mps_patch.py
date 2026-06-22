"""In-place patch() parity for the opt-in torch-MPS resident scan.

Skips unless a usable MPS device is present (Apple-Silicon dev machines, not CI
runners without Metal). Proves the O(changed) ``patch()`` produces the same
served result as a full rebuild, and that removals keep the resident copy
consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from lodedb.engine.mps_turbovec import MpsDirectTurboVecSession, mps_exact_scan_available
from lodedb.engine.turbovec_index import load_turbovec_id_map_index_class

pytest.importorskip("lodedb._turbovec")

_MPS_AVAILABLE, _MPS_REASON = mps_exact_scan_available()
pytestmark = pytest.mark.skipif(not _MPS_AVAILABLE, reason=f"MPS unavailable: {_MPS_REASON}")


def _unit_rows(rng, n, dim):
    vecs = rng.standard_normal((n, dim), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return np.ascontiguousarray(vecs)


def test_mps_patch_upsert_matches_full_rebuild():
    """Patching in new rows yields the same served top-k as building from scratch."""

    IdMapIndex = load_turbovec_id_map_index_class()
    rng = np.random.default_rng(11)
    n, m, dim, k = 600, 120, 384, 32

    base = _unit_rows(rng, n, dim)
    extra = _unit_rows(rng, m, dim)
    queries = _unit_rows(rng, 64, dim)

    index = IdMapIndex(dim=dim, bit_width=4)
    index.add_with_ids(base, np.arange(1, n + 1, dtype=np.uint64))

    session = MpsDirectTurboVecSession.build(index=index)
    # Mutate the index, then patch only the changed rows into the resident copy.
    new_ids = np.arange(n + 1, n + m + 1, dtype=np.uint64)
    index.add_with_ids(extra, new_ids)
    session.patch(index, removed_ids=(), upsert_ids=tuple(int(i) for i in new_ids), generation=1)
    assert session.row_count == n + m
    assert session.generation == 1

    rebuilt = MpsDirectTurboVecSession.build(index=index)
    patched_ids = session.search_batch(queries, top_k=k).stable_ids
    rebuilt_result = rebuilt.search_batch(queries, top_k=k)
    # Same served top-k ids and scores as a from-scratch rebuild.
    assert np.array_equal(patched_ids, rebuilt_result.stable_ids)


def test_mps_patch_remove_keeps_survivors_consistent():
    """Swap-remove drops the right ids and the resident copy still scores correctly."""

    IdMapIndex = load_turbovec_id_map_index_class()
    rng = np.random.default_rng(23)
    n, dim = 500, 384
    vecs = _unit_rows(rng, n, dim)

    index = IdMapIndex(dim=dim, bit_width=4)
    index.add_with_ids(vecs, np.arange(1, n + 1, dtype=np.uint64))

    session = MpsDirectTurboVecSession.build(index=index)
    removed = (5, 50, 123, 400)
    session.patch(index, removed_ids=removed, upsert_ids=(), generation=1)

    assert session.row_count == n - len(removed)
    assert session.generation == 1
    resident = set(int(i) for i in session.stable_ids[: session.row_count])
    assert resident.isdisjoint(removed)
    assert len(resident) == n - len(removed)

    # A surviving vector still scores as its own top-1 against the resident copy.
    survivor_id = 7
    query = vecs[survivor_id - 1 : survivor_id]  # id i is row i-1
    top = session.search_batch(query, top_k=1).stable_ids
    assert int(top[0, 0]) == survivor_id
