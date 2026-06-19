"""GPU-resident exact batch serving for direct TurboVec route profiles.

Mirrors the V1 GPU exact stage-one pattern for direct routes: a dequantized
fp16 copy of the index's rows stays resident on GPU and eligible BATCHED
queries are served with an exact tiled matmul plus device top-k. The CPU
TurboVec kernel remains the source of truth, the single-query path, and the
visible fallback.

Coordinate-space contract (pinned by the vendored parity tests in
`third_party/turbovec/turbovec/tests/reconstruction.rs`): the vendored
`reconstruct_all`/`reconstruct_rows` APIs export rows in ROTATED calibrated
space scaled by the stored per-row scale, so

``score(q, row) = <q @ rotation_matrix().T, reconstructed_row>``

reproduces the CPU kernel's calibrated score without its uint8 LUT
quantization error (the GPU score is the more faithful estimate of the same
quantized representation; agreement is within the LUT tolerance, roughly
``1/sqrt(dim)``-scaled). Queries are rotated on the host with the
deterministic rotation — exactly the GEMM the native search path runs.

No custom CUDA kernels: scoring is a tiled fp16→fp32 cast plus cuBLAS
matmul with a streaming top-k — each row tile's (batch x tile) scores are
selected with `torch.topk` and merged into a running per-query top-k, so the
full batch x corpus score matrix is never materialized (bounding GPU memory
regardless of batch/corpus). It falls back to a full `cupy.argpartition` when
torch is unavailable. The final ordering is deterministic on the host
(descending score, ascending stable id on ties; the CPU kernel breaks exact
ties by slot order instead, so tie ordering may differ while id sets and
scores agree).

Resident bytes are fp16 (`rows * dim * 2`) regardless of the index bit
width, so 2-bit routes get the same GPU path for free. Small mutations patch
the resident copy in place via `patch()` in O(changed) rows (swap-remove plus
a batched upsert); if a patch cannot be applied (e.g. capacity exceeded) the
resident copy is invalidated by generation and the next batch re-uploads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray

GPU_DIRECT_TURBOVEC_BACKEND = "gpu_cupy_exact_direct"
# Widened (post-filter) top-k requests fall back to the CPU kernel rather
# than sorting huge candidate sets on device and copying them back.
GPU_DIRECT_TURBOVEC_MAX_TOP_K = 4096
# Each scoring tile casts at most this many fp16 row bytes to fp32, keeping
# the transient allocation bounded regardless of corpus size.
_TILE_TRANSIENT_BYTES = 128 << 20
# Each scoring tile's (batch x tile) fp32 score block stays under this budget,
# so the streaming top-k never materializes the full batch x corpus score
# matrix (the old full-matrix path needed batch*corpus*4 bytes — ~3.5 GB at
# batch 1024 x 860K rows, which exceeded GPU memory and forced a CPU fallback).
_TILE_SCORE_BYTES = 64 << 20


@dataclass(frozen=True)
class GpuResidentMemoryEstimate:
    """Reports raw-payload-free GPU resident memory admission accounting."""

    estimated_bytes: int
    budget_bytes: int | None
    available_bytes: int | None
    admitted: bool
    reason: str

    def to_result_fields(self) -> dict[str, int | str | bool]:
        """Serializes memory accounting into safe query telemetry fields."""

        return {
            "gpu_estimated_bytes": self.estimated_bytes,
            "gpu_budget_bytes": int(self.budget_bytes or 0),
            "gpu_available_bytes": int(self.available_bytes or 0),
            "gpu_memory_admitted": self.admitted,
            "gpu_fallback_reason": self.reason,
        }


@dataclass(frozen=True)
class GpuDirectTurboVecDependencies:
    """Stores the lazily imported CuPy runtime for direct GPU serving."""

    available: bool
    cupy: Any | None
    unavailable_reason: str


def _gpu_available_memory_bytes(
    dependencies: GpuDirectTurboVecDependencies | None,
) -> int | None:
    """Returns current free GPU bytes when CuPy exposes CUDA memory information."""

    if dependencies is None or dependencies.cupy is None:
        return None
    try:
        free_bytes, _total_bytes = dependencies.cupy.cuda.runtime.memGetInfo()
    except AttributeError:
        return None
    except Exception:  # noqa: BLE001 - availability probing must not break auto fallback.
        return None
    return int(free_bytes)


def gpu_direct_turbovec_dependencies() -> GpuDirectTurboVecDependencies:
    """Lazily imports CuPy for the direct GPU path (cuVS is not required).

    Tests may install a fake module via the `_override` attribute, mirroring
    the V1 stage-one dependency hook.
    """

    override = getattr(gpu_direct_turbovec_dependencies, "_override", None)
    if override is not None:
        return override
    cached = getattr(gpu_direct_turbovec_dependencies, "_cache", None)
    if cached is not None:
        return cached
    try:
        cupy = import_module("cupy")
    except ImportError as exc:
        dependencies = GpuDirectTurboVecDependencies(
            available=False,
            cupy=None,
            unavailable_reason=f"GPU direct TurboVec dependencies are unavailable: {exc}",
        )
    else:
        dependencies = GpuDirectTurboVecDependencies(
            available=True,
            cupy=cupy,
            unavailable_reason="",
        )
    gpu_direct_turbovec_dependencies._cache = dependencies
    return dependencies


def turbovec_reconstruction_api_available(index: Any) -> bool:
    """Returns whether the loaded TurboVec build exposes the reconstruction APIs.

    PyPI ``turbovec==0.8.0`` lacks them; only the patched vendored wheel
    (see `third_party/turbovec/LOCAL_PATCHES.md`) can feed the GPU path.
    """

    return all(
        hasattr(index, name)
        for name in ("reconstruct_all", "reconstruct_rows", "rotation_matrix")
    )


def estimate_gpu_direct_turbovec_memory(
    *,
    row_count: int,
    dim: int,
    max_batch_size: int = 64,
    dependencies: GpuDirectTurboVecDependencies | None = None,
    memory_budget_bytes: int | None = None,
) -> GpuResidentMemoryEstimate:
    """Estimates resident and transient GPU bytes and applies the shared budget.

    Accounts the fp16 resident rows plus the transient peak of the path that
    will run. The tiled ``torch.topk`` scan (whenever torch is importable) never
    materializes the full batch x corpus score matrix: its transient is one
    bounded score tile, one bounded fp32 cast tile, the running/merged per-query
    top-k buffers, and the rotated query upload. The CuPy fallback (torch
    absent) still builds the full fp32 score matrix and its argpartition index
    buffer, so it is estimated that way. Shares
    ``LODEDB_GPU_MEMORY_BUDGET_BYTES`` admission semantics with the V1
    stage-one estimate.
    """

    if row_count < 0 or dim <= 0:
        raise ValueError("row_count must be non-negative and dim must be positive")
    capacity = max(int(int(row_count) * 1.5), 1024)
    resident_bytes = capacity * int(dim) * 2
    query_bytes = int(max_batch_size) * int(dim) * 4
    cast_tile_bytes = min(_TILE_TRANSIENT_BYTES, max(int(row_count) * int(dim) * 4, 1))
    if _maybe_import_torch() is not None:
        # Tiled streaming top-k: one bounded (batch x tile) score block, one
        # fp32 cast tile, and the running + merged top-k buffers (int64 slots +
        # fp32 scores, doubled by the merge concat). top_k is unknown at
        # admission, so bound it by the max post-filter cap.
        score_tile_bytes = min(_TILE_SCORE_BYTES, int(max_batch_size) * int(row_count) * 4)
        take = min(GPU_DIRECT_TURBOVEC_MAX_TOP_K, int(row_count))
        topk_buffer_bytes = int(max_batch_size) * int(take) * (8 + 4) * 2
        transient_bytes = score_tile_bytes + cast_tile_bytes + topk_buffer_bytes
    else:
        score_matrix_bytes = int(max_batch_size) * int(row_count) * 4
        argpartition_bytes = int(max_batch_size) * int(row_count) * 8
        transient_bytes = score_matrix_bytes + argpartition_bytes + cast_tile_bytes
    estimated_bytes = int(resident_bytes + transient_bytes + query_bytes)
    available_bytes = _gpu_available_memory_bytes(dependencies)
    budget = memory_budget_bytes if memory_budget_bytes is not None else available_bytes
    if budget is not None and estimated_bytes > int(budget):
        return GpuResidentMemoryEstimate(
            estimated_bytes=estimated_bytes,
            budget_bytes=int(budget),
            available_bytes=available_bytes,
            admitted=False,
            reason="gpu_direct_turbovec_memory_budget_exceeded",
        )
    return GpuResidentMemoryEstimate(
        estimated_bytes=estimated_bytes,
        budget_bytes=int(budget) if budget is not None else None,
        available_bytes=available_bytes,
        admitted=True,
        reason="",
    )


@dataclass(frozen=True)
class GpuDirectTurboVecBatchResult:
    """Carries one GPU batch's stable-id top-k plus raw-payload-free telemetry."""

    stable_ids: NDArray[np.uint64]
    scores: NDArray[np.float32]
    search_ms: float
    device_to_host_copy_ms: float
    copy_back_bytes: int
    tile_count: int


@dataclass
class GpuDirectTurboVecSession:
    """Keeps one generation's dequantized rows resident on GPU for batch search."""

    dependencies: GpuDirectTurboVecDependencies
    generation: int
    dim: int
    row_count: int
    stable_ids: NDArray[np.uint64]
    rotation: NDArray[np.float32]
    device_rows: Any
    id_to_slot: dict[int, int]
    estimated_gpu_bytes: int
    memory_budget_bytes: int | None
    upload_build_ms: float

    @classmethod
    def build(
        cls,
        *,
        index: Any,
        generation: int,
        dependencies: GpuDirectTurboVecDependencies | None = None,
        memory_budget_bytes: int | None = None,
        max_batch_size: int = 64,
    ) -> GpuDirectTurboVecSession:
        """Reconstructs all rows once and uploads the fp16 matrix to the GPU.

        Raises ``RuntimeError`` when dependencies or the patched
        reconstruction APIs are missing and ``MemoryError`` when admission
        rejects the resident estimate, so callers can fall back visibly.
        """

        dependencies = dependencies or gpu_direct_turbovec_dependencies()
        if not dependencies.available or dependencies.cupy is None:
            raise RuntimeError(
                dependencies.unavailable_reason
                or "GPU direct TurboVec dependencies are unavailable"
            )
        if not turbovec_reconstruction_api_available(index):
            raise RuntimeError(
                "the loaded TurboVec backend lacks the reconstruction APIs "
                "(reconstruct_all/rotation_matrix); build the patched vendored wheel"
            )
        started = time.perf_counter()
        row_count = int(len(index))
        index_dim = int(index.dim or 0)
        if row_count <= 0 or index_dim <= 0:
            raise RuntimeError("GPU direct TurboVec serving requires a non-empty index")
        # Admission runs BEFORE the (expensive) full-row reconstruction so a
        # rejecting budget falls back in microseconds, not after decoding the
        # whole corpus on every batch.
        admission = estimate_gpu_direct_turbovec_memory(
            row_count=row_count,
            dim=index_dim,
            max_batch_size=max_batch_size,
            dependencies=dependencies,
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

        capacity = max(int(rows.shape[0] * 1.5), 1024)
        ids_alloc = np.empty(capacity, dtype=np.uint64)
        if rows.shape[0] > 0:
            ids_alloc[: rows.shape[0]] = ids
        id_to_slot = {int(uid): slot for slot, uid in enumerate(ids_alloc[: rows.shape[0]])}

        cupy = dependencies.cupy
        device_rows = cupy.empty((capacity, rotation.shape[0]), dtype=cupy.float16)
        if rows.shape[0] > 0:
            device_rows[: rows.shape[0]] = cupy.asarray(rows.astype(np.float16, copy=False))
        _sync_device(cupy)

        return cls(
            dependencies=dependencies,
            generation=int(generation),
            dim=int(rotation.shape[0]),
            row_count=int(rows.shape[0]),
            stable_ids=ids_alloc,
            rotation=rotation,
            device_rows=device_rows,
            id_to_slot=id_to_slot,
            estimated_gpu_bytes=admission.estimated_bytes,
            memory_budget_bytes=memory_budget_bytes,
            upload_build_ms=float((time.perf_counter() - started) * 1000.0),
        )

    def patch(
        self,
        index: Any,
        removed_ids: tuple[int, ...],
        upsert_ids: tuple[int, ...],
    ) -> None:
        """Applies incremental mutations in-place to avoid a full O(N) rebuild.

        Simulates swap-remove semantics to match the CPU IdMapIndex slots exactly,
        and dynamically appends into over-allocated capacity in O(changed) time.
        """

        if removed_ids:
            for id_to_remove in removed_ids:
                uid = int(id_to_remove)
                slot = self.id_to_slot.get(uid)
                if slot is not None:
                    last = self.row_count - 1
                    if slot != last:
                        last_uid = int(self.stable_ids[last])
                        self.stable_ids[slot] = last_uid
                        self.device_rows[slot] = self.device_rows[last]
                        self.id_to_slot[last_uid] = slot
                    del self.id_to_slot[uid]
                    self.row_count -= 1

        if upsert_ids:
            upsert_arr = np.asarray(upsert_ids, dtype=np.uint64)
            upsert_rows = index.reconstruct_rows(upsert_arr)
            cupy = self.dependencies.cupy
            upsert_rows_fp16 = cupy.asarray(upsert_rows.astype(np.float16, copy=False))
            
            target_slots = []
            for target_id in upsert_ids:
                uid = int(target_id)
                slot = self.id_to_slot.get(uid)
                if slot is None:
                    if self.row_count >= self.device_rows.shape[0]:
                        raise MemoryError("GPU resident memory capacity exceeded during patch")
                    slot = self.row_count
                    self.stable_ids[slot] = np.uint64(uid)
                    self.id_to_slot[uid] = slot
                    self.row_count += 1
                target_slots.append(slot)
                
            # Batch the GPU write to avoid kernel-launch overhead per row
            if target_slots:
                self.device_rows[target_slots] = upsert_rows_fp16

    def search_batch(
        self,
        query_matrix: NDArray[np.float32],
        *,
        top_k: int,
    ) -> GpuDirectTurboVecBatchResult:
        """Scores one query batch exactly against the resident rows.

        Queries are rotated on the host with the deterministic rotation
        (cheap at batch scale), scored in fp32 against bounded fp16→fp32
        row tiles, top-k selected on device, and ordered deterministically
        on host by descending score with ascending-stable-id tie-break.
        """

        if query_matrix.ndim != 2 or query_matrix.shape[1] != self.dim:
            raise ValueError("query batch dimension does not match the resident index")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        started = time.perf_counter()
        batch_size = int(query_matrix.shape[0])
        take = min(int(top_k), self.row_count)
        if take == 0:
            return GpuDirectTurboVecBatchResult(
                stable_ids=np.zeros((batch_size, 0), dtype=np.uint64),
                scores=np.zeros((batch_size, 0), dtype=np.float32),
                search_ms=float((time.perf_counter() - started) * 1000.0),
                device_to_host_copy_ms=0.0,
                copy_back_bytes=0,
                tile_count=0,
            )
        rotated = np.ascontiguousarray(
            np.asarray(query_matrix, dtype=np.float32) @ self.rotation.T
        )
        if _maybe_import_torch() is not None:
            host_slots, host_scores, tile_count, device_to_host_copy_ms = (
                self._tiled_top_k_torch(rotated, take=take, batch_size=batch_size)
            )
        else:
            host_slots, host_scores, tile_count, device_to_host_copy_ms = (
                self._full_top_k_cupy(rotated, take=take, batch_size=batch_size)
            )
        copy_back_bytes = int(host_slots.nbytes + host_scores.nbytes)
        stable = self.stable_ids[host_slots]
        # Deterministic final ordering: descending score, ascending stable
        # id on exact ties (lexsort keys are applied last-key-major).
        order = np.lexsort((stable, -host_scores), axis=1)
        ordered_scores = np.take_along_axis(host_scores, order, axis=1)
        ordered_ids = np.take_along_axis(stable, order, axis=1)
        return GpuDirectTurboVecBatchResult(
            stable_ids=np.ascontiguousarray(ordered_ids, dtype=np.uint64),
            scores=np.ascontiguousarray(ordered_scores, dtype=np.float32),
            search_ms=float((time.perf_counter() - started) * 1000.0),
            device_to_host_copy_ms=device_to_host_copy_ms,
            copy_back_bytes=copy_back_bytes,
            tile_count=tile_count,
        )

    def _tiled_top_k_torch(
        self,
        rotated: NDArray[np.float32],
        *,
        take: int,
        batch_size: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], int, float]:
        """Streams a memory-bounded exact top-k with torch, never building full B×N scores.

        Scores the corpus in row tiles sized so each (batch x tile) fp32 score
        block and each fp32 row-cast tile stay under fixed budgets, keeps a
        running per-query top-k, and merges each tile's local top-k into it with
        ``torch.topk``. Returns host ``(slots, scores)``, the tile count, and the
        device->host copy time. The running merge is exact for the top-k set: a
        global top-k row is necessarily within its own tile's top-k, so the union
        of per-tile top-k contains the global top-k. Exact-score ties at the
        boundary resolve arbitrarily (as the full path does); the caller applies
        the deterministic host tie-break.
        """

        torch = import_module("torch")
        rows = torch.from_dlpack(self.device_rows)
        device = rows.device
        queries = torch.from_numpy(rotated).to(device=device, dtype=torch.float32)
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
        copy_started = time.perf_counter()
        host_slots = running_slots.to(device="cpu", dtype=torch.int64).numpy()
        host_scores = running_scores.to(device="cpu", dtype=torch.float32).numpy()
        device_to_host_copy_ms = float((time.perf_counter() - copy_started) * 1000.0)
        _log_top_k_backend("torch_tiled")
        return host_slots, host_scores, tile_count, device_to_host_copy_ms

    def _full_top_k_cupy(
        self,
        rotated: NDArray[np.float32],
        *,
        take: int,
        batch_size: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.float32], int, float]:
        """Fallback top-k: full B×N CuPy score matrix + argpartition (torch absent only)."""

        cupy = self.dependencies.cupy
        device_queries = cupy.asarray(rotated)
        scores = cupy.empty((batch_size, self.row_count), dtype=cupy.float32)
        tile_rows = max(1, _TILE_TRANSIENT_BYTES // max(self.dim * 4, 1))
        tile_count = 0
        for start in range(0, self.row_count, tile_rows):
            stop = min(start + tile_rows, self.row_count)
            tile = self.device_rows[start:stop].astype(cupy.float32)
            scores[:, start:stop] = device_queries @ tile.T
            tile_count += 1
        _sync_device(cupy)
        copy_started = time.perf_counter()
        top_slots = cupy.argpartition(-scores, int(take) - 1, axis=1)[:, : int(take)]
        top_scores = cupy.take_along_axis(scores, top_slots, axis=1)
        host_slots = cupy.asnumpy(top_slots).astype(np.int64, copy=False)
        host_scores = cupy.asnumpy(top_scores).astype(np.float32, copy=False)
        device_to_host_copy_ms = float((time.perf_counter() - copy_started) * 1000.0)
        _log_top_k_backend("cupy_argpartition")
        return host_slots, host_scores, tile_count, device_to_host_copy_ms


def _sync_device(cupy: Any) -> None:
    """Synchronizes the active GPU device when the runtime exposes that hook."""

    try:
        cupy.cuda.runtime.deviceSynchronize()
    except AttributeError:
        return


_TOP_K_BACKEND_LOGGED = False


def _maybe_import_torch() -> Any | None:
    """Returns the torch module if importable, else None (CuPy-only fallback path)."""

    try:
        return import_module("torch")
    except ImportError:
        return None


def _log_top_k_backend(name: str) -> None:
    """Logs the chosen device top-k backend once for raw-payload-free diagnostics."""

    global _TOP_K_BACKEND_LOGGED
    if not _TOP_K_BACKEND_LOGGED:
        print(f"gpu_direct_top_k_backend={name}", flush=True)
        _TOP_K_BACKEND_LOGGED = True
