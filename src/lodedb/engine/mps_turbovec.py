"""Opt-in Apple-GPU (Metal/MPS) exact batch serving for direct TurboVec routes.

The MPS analog of :mod:`lodedb.engine.gpu_turbovec`: a dequantized fp16 copy of
the index rows is kept resident on the Apple GPU, and a BATCHED query set is
served with an exact tiled matmul plus ``torch.topk``. It is **opt-in and never
the default** — on Apple Silicon the CPU NEON scan is the default, the
single-query path, and (today) faster across batch sizes on the hardware we have
measured; see ``benchmarks/mps_vs_neon/``. ``torch`` is imported lazily, so
importing the engine never requires it.

Same coordinate-space contract as ``gpu_turbovec`` (pinned by the vendored
reconstruction parity tests): ``reconstruct_all()`` exports rows in ROTATED
calibrated space, queries are rotated on the host (``q @ rotation.T``), and the
exact dot product reproduces the CPU kernel's calibrated score *without* its
uint8 LUT quantization error — so recall is ``>=`` the quantized NEON scan. The
final ordering is deterministic on the host (descending score, ascending stable
id on ties).

Resident bytes are fp16 (``rows * dim * 2``) regardless of the index bit width,
drawn from the shared unified-memory pool. This module performs no GPU-memory
admission (unlike the CUDA path): on unified memory there is no separate VRAM to
gate against.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lodedb.engine.gpu_turbovec import turbovec_reconstruction_api_available

MPS_DIRECT_TURBOVEC_BACKEND = "mps_torch_exact_direct"
# Each scoring tile's (batch x tile) fp32 score block stays under this budget, so
# the streaming top-k never materializes the full batch x corpus score matrix.
_TILE_SCORE_BYTES = 64 << 20
# Each fp16->fp32 row-cast tile stays under this budget regardless of corpus size.
_TILE_TRANSIENT_BYTES = 128 << 20


def mps_exact_scan_available() -> tuple[bool, str]:
    """Returns ``(available, reason)`` for the torch-MPS exact scan on this machine."""

    try:
        torch = import_module("torch")
    except ImportError as exc:
        return False, f"torch is not installed: {exc}"
    try:
        if not torch.backends.mps.is_available():
            return False, "torch reports no usable MPS (Metal) device"
    except (AttributeError, RuntimeError) as exc:
        return False, f"torch MPS probe failed: {exc}"
    # `is_available()` can report True on machines (notably virtualized CI
    # runners) where MPS is advertised but cannot actually back allocations,
    # surfacing later as a spurious "MPS backend out of memory" when the resident
    # matrix is uploaded. Confirm a tiny allocation + device op really succeeds so
    # the check stays honest: the parity test skips and production falls back to
    # the CPU scan visibly, instead of crashing mid-build.
    try:
        probe = torch.ones(1, device="mps", dtype=torch.float16)
        float((probe + probe).sum().to("cpu"))
        del probe
    except (RuntimeError, AttributeError) as exc:
        return False, f"MPS device is advertised but unusable: {exc}"
    return True, ""


@dataclass(frozen=True)
class MpsDirectTurboVecBatchResult:
    """Carries one MPS batch's stable-id top-k plus raw-payload-free telemetry."""

    stable_ids: NDArray[np.uint64]
    scores: NDArray[np.float32]
    search_ms: float
    device_to_host_copy_ms: float
    tile_count: int


@dataclass
class MpsDirectTurboVecSession:
    """Keeps one generation's dequantized rows resident on the Apple GPU (MPS)."""

    dim: int
    row_count: int
    stable_ids: NDArray[np.uint64]
    rotation: NDArray[np.float32]
    device_rows: Any  # torch.Tensor (fp16) resident on the mps device
    resident_bytes: int
    upload_build_ms: float

    @classmethod
    def build(cls, *, index: Any) -> MpsDirectTurboVecSession:
        """Reconstructs all rows once and uploads the fp16 matrix to the Apple GPU.

        Raises ``RuntimeError`` when MPS or the patched reconstruction APIs are
        missing, so callers can fall back to the CPU scan visibly.
        """

        available, reason = mps_exact_scan_available()
        if not available:
            raise RuntimeError(reason)
        if not turbovec_reconstruction_api_available(index):
            raise RuntimeError(
                "the loaded TurboVec backend lacks the reconstruction APIs "
                "(reconstruct_all/rotation_matrix); build the patched vendored wheel"
            )
        torch = import_module("torch")
        started = time.perf_counter()
        row_count = int(len(index))
        index_dim = int(index.dim or 0)
        if row_count <= 0 or index_dim <= 0:
            raise RuntimeError("MPS direct TurboVec serving requires a non-empty index")
        ids, rows = index.reconstruct_all()
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] != ids.shape[0]:
            raise RuntimeError("TurboVec reconstruction returned inconsistent shapes")
        rotation = index.rotation_matrix()
        if rotation is None:
            raise RuntimeError("TurboVec rotation matrix is unavailable before first add")
        rotation = np.ascontiguousarray(rotation, dtype=np.float32)
        device_rows = torch.from_numpy(rows).to(device="mps", dtype=torch.float16)
        torch.mps.synchronize()
        return cls(
            dim=int(rotation.shape[0]),
            row_count=int(rows.shape[0]),
            stable_ids=ids,
            rotation=rotation,
            device_rows=device_rows,
            resident_bytes=int(rows.shape[0]) * int(rows.shape[1]) * 2,
            upload_build_ms=float((time.perf_counter() - started) * 1000.0),
        )

    def search_batch(
        self,
        query_matrix: NDArray[np.float32],
        *,
        top_k: int,
    ) -> MpsDirectTurboVecBatchResult:
        """Scores one query batch exactly against the resident rows on the Apple GPU.

        Queries are rotated on the host with the deterministic rotation, scored in
        fp32 against bounded fp16->fp32 row tiles, top-k selected on device with a
        running streaming merge (never materializing the full batch x corpus score
        matrix), then ordered deterministically on the host.
        """

        if query_matrix.ndim != 2 or query_matrix.shape[1] != self.dim:
            raise ValueError("query batch dimension does not match the resident index")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        torch = import_module("torch")
        started = time.perf_counter()
        batch_size = int(query_matrix.shape[0])
        take = min(int(top_k), self.row_count)
        if take == 0:
            return MpsDirectTurboVecBatchResult(
                stable_ids=np.zeros((batch_size, 0), dtype=np.uint64),
                scores=np.zeros((batch_size, 0), dtype=np.float32),
                search_ms=float((time.perf_counter() - started) * 1000.0),
                device_to_host_copy_ms=0.0,
                tile_count=0,
            )
        rotated = np.ascontiguousarray(
            np.asarray(query_matrix, dtype=np.float32) @ self.rotation.T
        )
        rows = self.device_rows
        queries = torch.from_numpy(rotated).to(device=rows.device, dtype=torch.float32)
        score_tile = max(1, _TILE_SCORE_BYTES // max(batch_size * 4, 1))
        cast_tile = max(1, _TILE_TRANSIENT_BYTES // max(self.dim * 4, 1))
        tile_rows = max(1, min(score_tile, cast_tile))
        running_scores: Any | None = None
        running_slots: Any | None = None
        tile_count = 0
        for start in range(0, self.row_count, tile_rows):
            stop = min(start + tile_rows, self.row_count)
            tile = rows[start:stop].to(torch.float32)
            tile_scores = queries @ tile.T
            tile_take = min(int(take), stop - start)
            tile_values, tile_slots = torch.topk(
                tile_scores, tile_take, dim=1, largest=True, sorted=False
            )
            tile_slots = tile_slots + start
            if running_scores is None:
                running_scores, running_slots = tile_values, tile_slots
            else:
                merged_scores = torch.cat((running_scores, tile_values), dim=1)
                merged_slots = torch.cat((running_slots, tile_slots), dim=1)
                merge_take = min(int(take), int(merged_scores.shape[1]))
                running_scores, merge_positions = torch.topk(
                    merged_scores, merge_take, dim=1, largest=True, sorted=False
                )
                running_slots = torch.gather(merged_slots, 1, merge_positions)
            tile_count += 1
        torch.mps.synchronize()
        copy_started = time.perf_counter()
        host_slots = running_slots.to(device="cpu", dtype=torch.int64).numpy()
        host_scores = running_scores.to(device="cpu", dtype=torch.float32).numpy()
        device_to_host_copy_ms = float((time.perf_counter() - copy_started) * 1000.0)
        stable = self.stable_ids[host_slots]
        # Deterministic final ordering: descending score, ascending stable id on ties.
        order = np.lexsort((stable, -host_scores), axis=1)
        ordered_scores = np.take_along_axis(host_scores, order, axis=1)
        ordered_ids = np.take_along_axis(stable, order, axis=1)
        return MpsDirectTurboVecBatchResult(
            stable_ids=np.ascontiguousarray(ordered_ids, dtype=np.uint64),
            scores=np.ascontiguousarray(ordered_scores, dtype=np.float32),
            search_ms=float((time.perf_counter() - started) * 1000.0),
            device_to_host_copy_ms=device_to_host_copy_ms,
            tile_count=tile_count,
        )
