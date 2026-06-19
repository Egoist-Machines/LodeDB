"""Recall-parity test for the opt-in torch-MPS exact scan.

Skips unless a usable MPS device is present, so it runs on Apple-Silicon dev
machines but not on CI runners without Metal. The MPS exact scan scores the
dequantized rows in fp32 (no uint8 LUT error), so it agrees with the TurboVec
NEON scan within quantization tolerance and recovers the same top-k set — this
is a correctness/parity check, not a performance test (see
``benchmarks/mps_vs_neon/`` for speed).
"""

from __future__ import annotations

import numpy as np
import pytest

from lodedb.engine.mps_turbovec import MpsDirectTurboVecSession, mps_exact_scan_available
from lodedb.engine.turbovec_index import load_turbovec_id_map_index_class

pytest.importorskip("lodedb._turbovec")  # the bundled patched core provides reconstruct_all

_MPS_AVAILABLE, _MPS_REASON = mps_exact_scan_available()
pytestmark = pytest.mark.skipif(not _MPS_AVAILABLE, reason=f"MPS unavailable: {_MPS_REASON}")


def _recall_at(found_ids: np.ndarray, truth_top1: np.ndarray, k: int) -> float:
    """Fraction of queries whose true top-1 appears in the returned top-k."""

    return float((found_ids[:, :k] == truth_top1[:, None]).any(axis=1).mean())


def test_mps_exact_scan_matches_neon_within_tolerance():
    """MPS exact recall tracks the NEON scan's (>= within LUT tolerance)."""

    IdMapIndex = load_turbovec_id_map_index_class()

    rng = np.random.default_rng(7)
    n, dim, k = 4000, 384, 64
    vecs = rng.standard_normal((n, dim), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    queries = rng.standard_normal((200, dim), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    vecs = np.ascontiguousarray(vecs)
    queries = np.ascontiguousarray(queries)

    index = IdMapIndex(dim=dim, bit_width=4)
    index.add_with_ids(vecs, np.arange(1, n + 1, dtype=np.uint64))

    truth_top1 = (np.argmax(queries @ vecs.T, axis=1) + 1).astype(np.uint64)
    _scores, neon_ids = index.search(queries, k)

    session = MpsDirectTurboVecSession.build(index=index)
    result = session.search_batch(queries, top_k=k)
    mps_ids = result.stable_ids

    assert mps_ids.shape == (queries.shape[0], k)
    assert session.row_count == n and session.dim == dim
    # Exact scan recovers the true nearest neighbour within a small top-k.
    assert _recall_at(mps_ids, truth_top1, 8) >= 0.98
    # Parity: MPS exact recall is at least the NEON scan's, within LUT tolerance.
    for kk in (1, 8, 64):
        assert _recall_at(mps_ids, truth_top1, kk) >= _recall_at(neon_ids, truth_top1, kk) - 0.02
    # MPS and NEON score the same quantized rows, so their top-1 mostly agrees.
    assert float((mps_ids[:, 0] == neon_ids[:, 0]).mean()) >= 0.85


def test_mps_exact_scan_rejects_dim_mismatch():
    """A query batch whose dim != the index dim is rejected."""

    IdMapIndex = load_turbovec_id_map_index_class()

    rng = np.random.default_rng(1)
    vecs = np.ascontiguousarray(rng.standard_normal((256, 64), dtype=np.float32))
    index = IdMapIndex(dim=64, bit_width=4)
    index.add_with_ids(vecs, np.arange(1, 257, dtype=np.uint64))
    session = MpsDirectTurboVecSession.build(index=index)
    with pytest.raises(ValueError, match="dimension"):
        session.search_batch(np.zeros((4, 32), dtype=np.float32), top_k=8)
