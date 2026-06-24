"""Runtime selection policies and defaults for the LodeDB engine."""

from __future__ import annotations

import os
from collections.abc import Mapping
from enum import StrEnum


class TvimDeltaPersistencePolicy(StrEnum):
    """Names direct TurboVec `.tvim` delta persistence policies."""

    OFF = "off"
    AUTO = "auto"


class CommitMode(StrEnum):
    """Names the engine's per-mutation commit strategy.

    ``wal`` (the default) appends one framed record to ``<key>.wal`` per mutation
    and checkpoints into a generation periodically; it trades the lock-free reader
    snapshot (unnecessary for single-process writers) for a much cheaper per-write
    commit, and replays the WAL crash-atomically on open. ``generation`` is the
    classic path that publishes a new crash-atomic, MVCC-readable generation on
    every mutation via the root-manifest swap (pick it when many out-of-process
    readers must see every write the instant it commits, with no checkpoint lag).

    Both modes load and recover the same way: the durable on-disk base is always a
    committed generation, and any WAL tail is folded into a fresh generation on
    open, so a handle always opens onto a consistent generation regardless of mode.
    """

    GENERATION = "generation"
    WAL = "wal"


class GpuDirectTurboVecPolicy(StrEnum):
    """Names GPU-resident exact batch serving policies for direct TurboVec."""

    AUTO = "auto"
    OFF = "off"
    REQUIRED = "required"


class MpsDirectTurboVecPolicy(StrEnum):
    """Names Apple-GPU (MPS) exact batch serving policies for direct TurboVec.

    Defaults to ``off``: unlike CUDA (which beats the CPU kernel at every batch
    >= 2), the MPS scan was slower than the NEON CPU scan at every batch size on
    the Apple hardware measured (see ``benchmarks/mps_vs_neon/``), so NEON stays
    the default. ``auto`` opts a host into the MPS scan for eligible batches;
    ``required`` fails closed.
    """

    AUTO = "auto"
    OFF = "off"
    REQUIRED = "required"


# There is no default GPU-direct batch cap. An earlier crossover (CPU faster
# from ~batch 32 up) was an artifact of the slow ``cupy.argpartition`` top-k;
# after the 2026-06-13 ``torch.topk`` swap the GPU-resident scan amortizes and
# beats the CPU kernel at every batch >= 2 across GovReport5K/10K/15K (minilm +
# bge). So batch size is no longer a reason to flip — memory admission is the
# only gate — and the ``auto`` policy serves every eligible batch on the GPU.
# Operators may still set ``LODEDB_GPU_DIRECT_MAX_BATCH`` to force a CPU
# flip above some batch (e.g. for memory headroom).


def tvim_delta_persistence_policy_from_env(
    env: Mapping[str, str] | None = None,
) -> TvimDeltaPersistencePolicy:
    """Returns the direct TurboVec delta persistence policy from the environment.

    Defaults to ``auto`` (delta appends persist O(changed rows) `.tvd` + `.jsd`
    segments with byte-exact restart equality and no query/load
    regressions); ``off`` keeps the legacy full `.tvim`/JSON rewrites.
    ``auto`` falls back visibly to full base rewrites for cold builds,
    missing manifests, unavailable patched vendored APIs, or compaction.
    """

    source = os.environ if env is None else env
    return _parse_tvim_delta_persistence_policy(
        source.get("LODEDB_TVIM_DELTA_PERSISTENCE", "auto")
    )


def commit_mode_from_env(env: Mapping[str, str] | None = None) -> CommitMode:
    """Returns the per-mutation commit mode from the environment.

    Defaults to ``wal``: a single-writer write-ahead-log commit path (one framed
    append per mutation, periodic checkpoint into a generation, crash-atomic WAL
    replay on open) that makes durable single adds roughly an order of magnitude
    cheaper than the per-mutation generation publish. ``LODEDB_COMMIT_MODE=generation``
    selects the classic path that publishes a crash-atomic, MVCC-readable
    generation on every mutation. Used only when no explicit ``commit_mode=`` is
    passed.
    """

    source = os.environ if env is None else env
    return parse_commit_mode(source.get("LODEDB_COMMIT_MODE", "wal"))


def parse_commit_mode(value: str | None) -> CommitMode:
    """Parses and validates a commit-mode string (shared by the env + SDK paths)."""

    try:
        return CommitMode(str(value or "wal").strip().lower())
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in CommitMode)
        raise ValueError(f"commit_mode must be one of: {allowed}") from exc


def gpu_direct_turbovec_policy_from_env(
    env: Mapping[str, str] | None = None,
) -> GpuDirectTurboVecPolicy:
    """Returns the GPU direct TurboVec batch serving policy from the environment.

    Semantics mirror ``LODEDB_GPU_EXACT_STAGE_ONE``: ``auto`` serves
    eligible batched direct-route queries from a GPU-resident dequantized
    copy with visible CPU fallback (single queries, unavailable CuPy or
    patched reconstruction APIs, memory rejection, or runtime failure),
    ``required`` fails closed, and ``off`` always uses the CPU kernel.
    """

    source = os.environ if env is None else env
    return _parse_gpu_direct_turbovec_policy(
        source.get("LODEDB_GPU_DIRECT_TURBOVEC", "auto")
    )


def gpu_direct_turbovec_max_batch_from_env(env: Mapping[str, str] | None = None) -> int | None:
    """Returns the optional GPU-direct ``auto`` batch cap, or None for no cap.

    Unset (the default) means no cap: GPU-direct serves every eligible batch
    (>= 2) and memory admission is the only fallback gate, because the
    ``torch.topk`` top-k swap made the GPU-resident scan faster than the CPU
    kernel at all batch sizes. A positive ``LODEDB_GPU_DIRECT_MAX_BATCH``
    flips batches above it to the CPU kernel; it only bounds the ``auto``
    policy, so ``required`` still forces the GPU path at any eligible batch.
    """

    source = os.environ if env is None else env
    value = source.get("LODEDB_GPU_DIRECT_MAX_BATCH")
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("LODEDB_GPU_DIRECT_MAX_BATCH must be an integer") from exc
    if parsed <= 0:
        raise ValueError("LODEDB_GPU_DIRECT_MAX_BATCH must be positive")
    return parsed


def gpu_memory_budget_bytes_from_env(env: Mapping[str, str] | None = None) -> int | None:
    """Returns a positive GPU memory budget in bytes, or None when not configured."""

    source = os.environ if env is None else env
    value = source.get("LODEDB_GPU_MEMORY_BUDGET_BYTES")
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("LODEDB_GPU_MEMORY_BUDGET_BYTES must be an integer") from exc
    if parsed <= 0:
        raise ValueError("LODEDB_GPU_MEMORY_BUDGET_BYTES must be positive")
    return parsed


def gpu_direct_turbovec_should_use(
    *,
    policy: GpuDirectTurboVecPolicy,
    dependency_available: bool,
    query_batch_size: int,
    minimum_batch_size: int = 2,
    maximum_batch_size: int | None = None,
) -> bool:
    """Returns whether the GPU-resident direct TurboVec path should handle a batch.

    Single queries stay on the CPU kernel (upload/orchestration overhead
    dominates) without raising even under ``required``, matching the V1
    stage-one policy; ``required`` fails closed for eligible batches when
    dependencies are missing.

    ``maximum_batch_size`` bounds the GPU-favorable window for the ``auto``
    policy only: batches above it flip to the CPU kernel (whose shared-top-k
    scan amortizes better at large batch). ``required`` deliberately ignores
    the bound so it can force — and benchmark — the GPU path at any batch size.
    """

    if query_batch_size < minimum_batch_size:
        return False
    if policy == GpuDirectTurboVecPolicy.OFF:
        return False
    if (
        policy == GpuDirectTurboVecPolicy.AUTO
        and maximum_batch_size is not None
        and query_batch_size > maximum_batch_size
    ):
        return False
    if dependency_available:
        return True
    if policy == GpuDirectTurboVecPolicy.REQUIRED:
        raise RuntimeError("GPU direct TurboVec serving is required but unavailable")
    return False


def _parse_tvim_delta_persistence_policy(value: str | None) -> TvimDeltaPersistencePolicy:
    """Parses and validates a `.tvim` delta persistence policy string."""

    try:
        return TvimDeltaPersistencePolicy(str(value or "auto"))
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in TvimDeltaPersistencePolicy)
        raise ValueError(
            f"LODEDB_TVIM_DELTA_PERSISTENCE must be one of: {allowed}"
        ) from exc


def _parse_gpu_direct_turbovec_policy(value: str | None) -> GpuDirectTurboVecPolicy:
    """Parses and validates a GPU direct TurboVec serving policy string."""

    try:
        return GpuDirectTurboVecPolicy(str(value or "auto"))
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in GpuDirectTurboVecPolicy)
        raise ValueError(
            f"LODEDB_GPU_DIRECT_TURBOVEC must be one of: {allowed}"
        ) from exc


def mps_direct_turbovec_policy_from_env(
    env: Mapping[str, str] | None = None,
) -> MpsDirectTurboVecPolicy:
    """Returns the Apple-GPU (MPS) direct TurboVec serving policy from the environment.

    Defaults to ``off`` (NEON is the default and faster on measured Apple
    hardware). ``LODEDB_MPS_DIRECT_TURBOVEC=auto`` opts eligible batches into the
    MPS scan with visible CPU fallback; ``required`` fails closed.
    """

    source = os.environ if env is None else env
    return _parse_mps_direct_turbovec_policy(
        source.get("LODEDB_MPS_DIRECT_TURBOVEC", "off")
    )


def mps_direct_turbovec_max_batch_from_env(env: Mapping[str, str] | None = None) -> int | None:
    """Returns the optional MPS-direct ``auto`` batch cap, or None for no cap."""

    source = os.environ if env is None else env
    value = source.get("LODEDB_MPS_DIRECT_MAX_BATCH")
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("LODEDB_MPS_DIRECT_MAX_BATCH must be an integer") from exc
    if parsed <= 0:
        raise ValueError("LODEDB_MPS_DIRECT_MAX_BATCH must be positive")
    return parsed


def mps_memory_budget_bytes_from_env(env: Mapping[str, str] | None = None) -> int | None:
    """Returns a positive MPS resident memory budget in bytes, or None when unset.

    Unified memory has no separate VRAM pool, so there is no auto budget: a
    resident copy is admitted unless ``LODEDB_MPS_MEMORY_BUDGET_BYTES`` is set and
    the estimate exceeds it.
    """

    source = os.environ if env is None else env
    value = source.get("LODEDB_MPS_MEMORY_BUDGET_BYTES")
    if value is None or not str(value).strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError("LODEDB_MPS_MEMORY_BUDGET_BYTES must be an integer") from exc
    if parsed <= 0:
        raise ValueError("LODEDB_MPS_MEMORY_BUDGET_BYTES must be positive")
    return parsed


def mps_direct_turbovec_should_use(
    *,
    policy: MpsDirectTurboVecPolicy,
    dependency_available: bool,
    query_batch_size: int,
    minimum_batch_size: int = 2,
    maximum_batch_size: int | None = None,
) -> bool:
    """Returns whether the MPS-resident direct TurboVec path should handle a batch.

    Single queries stay on the CPU kernel without raising even under
    ``required``; ``required`` fails closed for eligible batches when MPS is
    unavailable. ``maximum_batch_size`` bounds the ``auto`` window only.
    """

    if query_batch_size < minimum_batch_size:
        return False
    if policy == MpsDirectTurboVecPolicy.OFF:
        return False
    if (
        policy == MpsDirectTurboVecPolicy.AUTO
        and maximum_batch_size is not None
        and query_batch_size > maximum_batch_size
    ):
        return False
    if dependency_available:
        return True
    if policy == MpsDirectTurboVecPolicy.REQUIRED:
        raise RuntimeError("MPS direct TurboVec serving is required but unavailable")
    return False


def _parse_mps_direct_turbovec_policy(value: str | None) -> MpsDirectTurboVecPolicy:
    """Parses and validates an MPS direct TurboVec serving policy string."""

    try:
        return MpsDirectTurboVecPolicy(str(value or "off"))
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in MpsDirectTurboVecPolicy)
        raise ValueError(
            f"LODEDB_MPS_DIRECT_TURBOVEC must be one of: {allowed}"
        ) from exc


