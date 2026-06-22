"""Backend-agnostic helpers for resident scan sessions.

Pure NumPy (no cupy / torch): the deterministic top-k ordering and the score-tile
sizing mirror the CUDA resident-scan contract. The MPS session
(:mod:`lodedb.engine.mps_turbovec`) uses them. The shipped CUDA session
(:mod:`lodedb.engine.gpu_turbovec`) keeps its own inline copies for now; refactoring
it to share these needs CUDA-hardware verification and is a separate, Modal/CI-gated
change.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def tile_row_count(
    *,
    batch_size: int,
    dim: int,
    score_tile_bytes: int,
    transient_bytes: int,
) -> int:
    """Returns the row-tile height bounding both the score block and the cast tile.

    Each tile's ``(batch x tile)`` fp32 score block stays under ``score_tile_bytes``
    and its fp16->fp32 row cast under ``transient_bytes``, so the streaming top-k
    never materializes the full ``batch x corpus`` score matrix.
    """

    score_tile = max(1, score_tile_bytes // max(batch_size * 4, 1))
    cast_tile = max(1, transient_bytes // max(dim * 4, 1))
    return max(1, min(score_tile, cast_tile))


def deterministic_topk_order(
    host_slots: NDArray[np.int64],
    host_scores: NDArray[np.float32],
    stable_ids: NDArray[np.uint64],
) -> tuple[NDArray[np.uint64], NDArray[np.float32]]:
    """Maps device slot indices to stable ids and orders each query row deterministically.

    The canonical resident-scan ordering: descending score, ascending stable id on
    ties. ``host_slots`` / ``host_scores`` are ``(batch, k)``; ``stable_ids`` is the
    ``(capacity,)`` slot->id map (only valid slots are indexed).
    """

    stable = stable_ids[host_slots]
    order = np.lexsort((stable, -host_scores), axis=1)
    ordered_scores = np.take_along_axis(host_scores, order, axis=1)
    ordered_ids = np.take_along_axis(stable, order, axis=1)
    return (
        np.ascontiguousarray(ordered_ids, dtype=np.uint64),
        np.ascontiguousarray(ordered_scores, dtype=np.float32),
    )
