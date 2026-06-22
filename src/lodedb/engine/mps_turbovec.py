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
id on ties), shared with the CUDA path via :mod:`lodedb.engine.turbovec_resident`.

Resident bytes are fp16 (``capacity * dim * 2``) drawn from the shared
unified-memory pool. The resident copy is over-allocated 1.5x so small mutations
patch in place (``patch()``, O(changed) rows) instead of rebuilding. There is no
separate VRAM to gate against, so admission only rejects when an explicit
``LODEDB_MPS_MEMORY_BUDGET_BYTES`` is set and exceeded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lodedb.engine.gpu_turbovec import turbovec_reconstruction_api_available
from lodedb.engine.turbovec_resident import deterministic_topk_order, tile_row_count

MPS_DIRECT_TURBOVEC_BACKEND = "mps_torch_exact_direct"
# Widened (post-filter) top-k requests fall back to the CPU kernel rather than
# sorting huge candidate sets on device and copying them back.
MPS_DIRECT_TURBOVEC_MAX_TOP_K = 4096
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
class MpsResidentMemoryEstimate:
    """Reports raw-payload-free MPS resident memory admission accounting."""

    estimated_bytes: int
    budget_bytes: int | None
    admitted: bool
    reason: str


def estimate_mps_direct_turbovec_memory(
    *,
    row_count: int,
    dim: int,
    max_batch_size: int = 64,
    memory_budget_bytes: int | None = None,
) -> MpsResidentMemoryEstimate:
    """Estimates resident + transient MPS bytes and applies an optional budget.

    Unified memory has no separate pool, so admission only rejects when
    ``memory_budget_bytes`` is set and the estimate exceeds it; otherwise it
    admits (the estimate is still reported for telemetry).
    """

    capacity = max(int(row_count * 1.5), 1024)
    resident = capacity * dim * 2  # fp16 resident rows
    # Transient scratch is bounded by the ACTUAL corpus, not the per-tile budget
    # ceiling. search_batch streams over <= row_count rows in tiles whose height is
    # capped so the (batch x tile) fp32 score block stays under _TILE_SCORE_BYTES and
    # the fp16->fp32 row cast under _TILE_TRANSIENT_BYTES; a small index never fills a
    # full tile, so charging the ceiling would make any sub-ceiling budget reject
    # trivially-small indexes.
    batch = max(int(max_batch_size), 1)
    tile_height = min(
        tile_row_count(
            batch_size=batch,
            dim=dim,
            score_tile_bytes=_TILE_SCORE_BYTES,
            transient_bytes=_TILE_TRANSIENT_BYTES,
        ),
        max(int(row_count), 1),
    )
    score_block = batch * tile_height * 4  # (batch x tile) fp32 scores
    cast_tile = tile_height * dim * 4  # fp16 -> fp32 row cast for one tile
    rotated_queries = batch * dim * 4  # host-rotated query block resident on device
    transient = score_block + cast_tile + rotated_queries
    estimated = int(resident + transient)
    if memory_budget_bytes is None:
        return MpsResidentMemoryEstimate(estimated, None, True, "")
    admitted = estimated <= int(memory_budget_bytes)
    reason = (
        ""
        if admitted
        else f"estimated {estimated} bytes exceeds MPS budget {int(memory_budget_bytes)} bytes"
    )
    return MpsResidentMemoryEstimate(estimated, int(memory_budget_bytes), admitted, reason)


@dataclass(frozen=True)
class MpsDirectTurboVecBatchResult:
    """Carries one MPS batch's stable-id top-k plus raw-payload-free telemetry."""

    stable_ids: NDArray[np.uint64]
    scores: NDArray[np.float32]
    search_ms: float
    device_to_host_copy_ms: float
    tile_count: int
    copy_back_bytes: int


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
    generation: int = 0
    id_to_slot: dict[int, int] = field(default_factory=dict)
    memory_budget_bytes: int | None = None
    estimated_mps_bytes: int = 0

    @classmethod
    def build(
        cls,
        *,
        index: Any,
        generation: int = 0,
        memory_budget_bytes: int | None = None,
        max_batch_size: int = 64,
    ) -> MpsDirectTurboVecSession:
        """Reconstructs all rows once and uploads the fp16 matrix to the Apple GPU.

        Raises ``RuntimeError`` when MPS or the patched reconstruction APIs are
        missing and ``MemoryError`` when an explicit budget rejects the resident
        estimate, so callers can fall back to the CPU scan visibly.
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
        # Admission runs BEFORE the (expensive) full-row reconstruction so a
        # rejecting budget falls back without decoding the whole corpus.
        admission = estimate_mps_direct_turbovec_memory(
            row_count=row_count,
            dim=index_dim,
            max_batch_size=max_batch_size,
            memory_budget_bytes=memory_budget_bytes,
        )
        if not admission.admitted:
            raise MemoryError(admission.reason)
        ids, rows = index.reconstruct_all()
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] != ids.shape[0]:
            raise RuntimeError("TurboVec reconstruction returned inconsistent shapes")
        rotation = index.rotation_matrix()
        if rotation is None:
            raise RuntimeError("TurboVec rotation matrix is unavailable before first add")
        rotation = np.ascontiguousarray(rotation, dtype=np.float32)

        # Over-allocate 1.5x so small adds patch in place instead of rebuilding.
        capacity = max(int(rows.shape[0] * 1.5), 1024)
        ids_alloc = np.empty(capacity, dtype=np.uint64)
        if rows.shape[0] > 0:
            ids_alloc[: rows.shape[0]] = ids
        id_to_slot = {int(uid): slot for slot, uid in enumerate(ids_alloc[: rows.shape[0]])}

        device_rows = torch.empty(
            (capacity, int(rotation.shape[0])), dtype=torch.float16, device="mps"
        )
        if rows.shape[0] > 0:
            device_rows[: rows.shape[0]] = torch.from_numpy(rows).to(
                device="mps", dtype=torch.float16
            )
        torch.mps.synchronize()
        return cls(
            dim=int(rotation.shape[0]),
            row_count=int(rows.shape[0]),
            stable_ids=ids_alloc,
            rotation=rotation,
            device_rows=device_rows,
            resident_bytes=capacity * int(rotation.shape[0]) * 2,
            upload_build_ms=float((time.perf_counter() - started) * 1000.0),
            generation=int(generation),
            id_to_slot=id_to_slot,
            memory_budget_bytes=memory_budget_bytes,
            estimated_mps_bytes=int(admission.estimated_bytes),
        )

    def patch(
        self,
        index: Any,
        removed_ids: tuple[int, ...],
        upsert_ids: tuple[int, ...],
    ) -> None:
        """Applies incremental mutations in-place to avoid a full O(N) rebuild.

        Mirrors the CUDA path: swap-remove then a batched upsert into the
        over-allocated capacity, all in O(changed) rows. The session keeps its own
        ``slot -> stable id`` map (``stable_ids`` / ``id_to_slot``) independent of
        the CPU ``IdMapIndex``'s internal slots; the only invariant maintained is
        that ``device_rows[slot]`` stays paired with ``stable_ids[slot]``, so the
        served top-k matches a from-scratch rebuild (asserted by the patch parity
        tests). Raises ``MemoryError`` if the upsert would exceed capacity, so the
        caller evicts the session and the next batch rebuilds.
        """

        torch = import_module("torch")
        if removed_ids:
            for id_to_remove in removed_ids:
                uid = int(id_to_remove)
                slot = self.id_to_slot.get(uid)
                if slot is not None:
                    last = self.row_count - 1
                    if slot != last:
                        last_uid = int(self.stable_ids[last])
                        self.stable_ids[slot] = last_uid
                        # Clone the source row first to match CUDA's element
                        # assignment exactly and avoid any aliasing on-device.
                        self.device_rows[slot] = self.device_rows[last].clone()
                        self.id_to_slot[last_uid] = slot
                    del self.id_to_slot[uid]
                    self.row_count -= 1

        if upsert_ids:
            upsert_arr = np.asarray(upsert_ids, dtype=np.uint64)
            upsert_rows = np.ascontiguousarray(index.reconstruct_rows(upsert_arr), dtype=np.float32)
            upsert_rows_fp16 = torch.from_numpy(upsert_rows).to(
                device=self.device_rows.device, dtype=torch.float16
            )
            target_slots: list[int] = []
            for target_id in upsert_ids:
                uid = int(target_id)
                slot = self.id_to_slot.get(uid)
                if slot is None:
                    if self.row_count >= self.device_rows.shape[0]:
                        raise MemoryError("MPS resident memory capacity exceeded during patch")
                    slot = self.row_count
                    self.stable_ids[slot] = np.uint64(uid)
                    self.id_to_slot[uid] = slot
                    self.row_count += 1
                target_slots.append(slot)
            if target_slots:
                slot_index = torch.tensor(
                    target_slots, device=self.device_rows.device, dtype=torch.long
                )
                self.device_rows[slot_index] = upsert_rows_fp16
            torch.mps.synchronize()

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
                copy_back_bytes=0,
            )
        rotated = np.ascontiguousarray(
            np.asarray(query_matrix, dtype=np.float32) @ self.rotation.T
        )
        rows = self.device_rows
        queries = torch.from_numpy(rotated).to(device=rows.device, dtype=torch.float32)
        tile_rows = tile_row_count(
            batch_size=batch_size,
            dim=self.dim,
            score_tile_bytes=_TILE_SCORE_BYTES,
            transient_bytes=_TILE_TRANSIENT_BYTES,
        )
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
        ordered_ids, ordered_scores = deterministic_topk_order(
            host_slots, host_scores, self.stable_ids
        )
        return MpsDirectTurboVecBatchResult(
            stable_ids=ordered_ids,
            scores=ordered_scores,
            search_ms=float((time.perf_counter() - started) * 1000.0),
            device_to_host_copy_ms=device_to_host_copy_ms,
            tile_count=tile_count,
            copy_back_bytes=int(host_slots.nbytes + host_scores.nbytes),
        )
