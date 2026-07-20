"""Measurement library for vanilla TurboVec vs the augmented GPU TurboVec.

Compares the *same* vendored 4-bit TurboVec index two ways, with NO FAISS. Both
scan the same 4-bit codes; the difference is the scoring step. The CPU sums a
uint8 LUT (ADC), the GPU reconstructs to fp16 and does a full GEMM dot product.
So "exact" below means exact over the 4-bit reconstruction, not fp32:

- **vanilla**: TurboVec's native CPU SIMD scan, ``index.search(queries, k)``,
  which sums a uint8 lookup table (ADC-style) over the codes. This is exactly the
  kernel the upstream repo benchmarks; the local patches add APIs
  (reconstruction/upsert/encoded-rows) but do not change this scan.
- **augmented**: the GPU-resident fp16-reconstruction scan
  (:class:`lodedb.engine.gpu_turbovec.GpuDirectTurboVecSession`):
  all rows are reconstructed once to fp16 on the GPU and batches are scored with a
  rotated-query GEMM + streaming device top-k. Because it scores the reconstructed
  vectors with full fp16 arithmetic (not the uint8 LUT), it avoids the uint8-LUT
  rounding error the LUT scan accumulates, so its recall is >= vanilla while
  throughput scales on the GPU, at higher resident memory (fp16 rows vs compact
  2/4-bit codes). Both scans carry the same irreducible 4-bit code-quantization loss.

Axes (all metrics only, no payloads):
  speed:   ms/query + queries/sec, vanilla uint8-LUT scan (ST + MT) vs augmented fp16 scan.
  recall:  R@1-within-top-k vs exact fp32 ground truth, vanilla vs augmented.
  memory:  bytes/vector compact (vanilla, in RAM) vs fp16 resident (augmented GPU).
  update:  incremental update + persist cost, vanilla full rewrite (O(N)) vs the
             augmented delta export (O(changed rows)).

Thread count for the CPU scan is controlled by ``RAYON_NUM_THREADS`` and must be set
*before* ``turbovec`` is first used, so ST/MT speed cells are run as fresh
subprocesses (see ``run_cell`` / the ``__main__`` dispatcher).

The GPU axis degrades gracefully: with no CUDA/CuPy it records ``{"skipped": ...}``
so the CPU axes still run on a laptop; the full run happens on a CUDA host.

Run a single cell directly (used by the orchestrator's subprocess pattern)::

    python -m turbovec_vva_bench '{"axis": "memory", "dim": 1536, "bit_width": 4, "n": 100000}'
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

K_GRID = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_K = 64


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def make_vectors(n: int, dim: int, *, seed: int) -> NDArray[np.float32]:
    """Returns ``n`` unit-norm Gaussian vectors (the data-oblivious speed regime)."""

    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(vecs / norms, dtype=np.float32)


def _unit(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns row-wise unit-normalized vectors (TurboVec ranks by cosine).

    Applied to loaded datasets so the exact ground truth is cosine top-k, the same
    metric TurboVec's scan and the GPU reconstruction rank by (both strip the
    per-vector norm). Idempotent on already-unit data (synthetic, most OpenAI).
    """

    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


def build_index(vectors: NDArray[np.float32], *, bit_width: int) -> Any:
    """Builds a vendored TurboVec ``IdMapIndex`` over ``vectors`` (ids = 1..N)."""

    import turbovec

    dim = int(vectors.shape[1])
    index = turbovec.IdMapIndex(dim=dim, bit_width=bit_width)
    ids = np.arange(1, vectors.shape[0] + 1, dtype=np.uint64)
    index.add_with_ids(vectors, ids)
    return index


# --------------------------------------------------------------------------- #
# Exact ground truth + recall
# --------------------------------------------------------------------------- #
def exact_topk_ids(
    vectors: NDArray[np.float32], queries: NDArray[np.float32], k: int
) -> NDArray[np.uint64]:
    """Returns exact top-k stable ids (1-based) by inner product, in fp32.

    Computed in row tiles so the full (queries x corpus) score matrix is never
    materialized, which keeps the CPU ground truth feasible at large corpus sizes.
    """

    n = int(vectors.shape[0])
    qn = int(queries.shape[0])
    take = min(k, n)
    best_scores = np.full((qn, take), -np.inf, dtype=np.float32)
    best_ids = np.zeros((qn, take), dtype=np.uint64)
    tile = max(1, (1 << 26) // max(1, qn))  # ~64M score floats per tile
    for start in range(0, n, tile):
        block = vectors[start : start + tile]
        scores = queries @ block.T  # (qn, tile)
        block_ids = np.arange(start + 1, start + 1 + block.shape[0], dtype=np.uint64)
        merged_scores = np.concatenate([best_scores, scores], axis=1)
        merged_ids = np.concatenate(
            [best_ids, np.broadcast_to(block_ids, (qn, block_ids.shape[0]))], axis=1
        )
        part = np.argpartition(-merged_scores, take - 1, axis=1)[:, :take]
        best_scores = np.take_along_axis(merged_scores, part, axis=1)
        best_ids = np.take_along_axis(merged_ids, part, axis=1)
    order = np.argsort(-best_scores, axis=1)
    return np.take_along_axis(best_ids, order, axis=1)


def recall_curve(found_ids: NDArray[np.uint64], truth_ids: NDArray[np.uint64]) -> dict[str, float]:
    """Returns R@1-within-top-k for each k in :data:`K_GRID` (upstream's recall metric).

    For each k: the fraction of queries whose *true* nearest neighbour (exact
    top-1) appears in the returned top-k. ``found_ids`` is (queries, >=max(K_GRID)).
    """

    truth_top1 = truth_ids[:, 0]
    curve: dict[str, float] = {}
    for k in K_GRID:
        if found_ids.shape[1] < k:
            continue
        hit = (found_ids[:, :k] == truth_top1[:, None]).any(axis=1)
        curve[str(k)] = float(hit.mean())
    return curve


# --------------------------------------------------------------------------- #
# Speed
# --------------------------------------------------------------------------- #
def time_vanilla_search(
    index: Any, queries: NDArray[np.float32], *, k: int, repeats: int
) -> dict[str, Any]:
    """Times the native CPU SIMD scan over the full query set; returns ids + timings."""

    # Warm up (kernel + rayon pool), then take the median of `repeats` full passes.
    scores, ids = index.search(queries, k)
    pass_ms: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        scores, ids = index.search(queries, k)
        pass_ms.append((time.perf_counter() - t0) * 1000.0)
    qn = int(queries.shape[0])
    median_pass = float(np.median(pass_ms))
    return {
        "ms_per_query": median_pass / qn,
        "queries_per_sec": qn / (median_pass / 1000.0) if median_pass > 0 else 0.0,
        "pass_ms_median": median_pass,
        "query_count": qn,
        "k": int(k),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "default(all)"),
        "ids": ids,
    }


def time_augmented_gpu_search(
    index: Any, queries: NDArray[np.float32], *, k: int, repeats: int, batch_size: int
) -> dict[str, Any]:
    """Times the GPU-resident exact scan (build once, batched search). GPU-optional."""

    from lodedb.engine.gpu_turbovec import (
        GpuDirectTurboVecSession,
        gpu_direct_turbovec_dependencies,
    )

    deps = gpu_direct_turbovec_dependencies()
    if not deps.available:
        return {"skipped": deps.unavailable_reason or "no CUDA/CuPy GPU available"}

    session = GpuDirectTurboVecSession.build(
        index=index, generation=1, dependencies=deps, max_batch_size=batch_size
    )
    qn = int(queries.shape[0])
    batches = [queries[s : s + batch_size] for s in range(0, qn, batch_size)]

    def one_pass() -> tuple[float, NDArray[np.uint64]]:
        rows: list[NDArray[np.uint64]] = []
        search_ms = 0.0
        for batch in batches:
            res = session.search_batch(batch, top_k=k)
            search_ms += res.search_ms
            rows.append(res.stable_ids)
        return search_ms, np.concatenate(rows, axis=0)

    _, ids = one_pass()  # warm (resident upload already done in build)
    pass_ms: list[float] = []
    for _ in range(max(1, repeats)):
        ms, ids = one_pass()
        pass_ms.append(ms)
    median_pass = float(np.median(pass_ms))
    return {
        "ms_per_query": median_pass / qn,
        "queries_per_sec": qn / (median_pass / 1000.0) if median_pass > 0 else 0.0,
        "pass_ms_median": median_pass,
        "query_count": qn,
        "k": int(k),
        "batch_size": int(batch_size),
        "gpu_resident_upload_build_ms": session.upload_build_ms,
        "gpu_resident_bytes": int(session.row_count) * int(session.dim) * 2,
        "ids": ids,
    }


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def memory_profile(index: Any, *, row_count: int, dim: int) -> dict[str, Any]:
    """Returns compact (vanilla, RAM) vs fp16-resident (augmented GPU) bytes."""

    compact_bytes_per_vector = int(index.bytes_per_vector())  # packed codes (dim*bit/8)
    fp16_resident_per_vector = dim * 2
    fp32_per_vector = dim * 4
    return {
        "row_count": int(row_count),
        "dim": int(dim),
        "bit_width": int(index.bit_width),
        "vanilla_compact_bytes_per_vector": compact_bytes_per_vector,
        "augmented_fp16_resident_bytes_per_vector": fp16_resident_per_vector,
        "fp32_bytes_per_vector": fp32_per_vector,
        "vanilla_compact_total_mb": compact_bytes_per_vector * row_count / 1e6,
        "augmented_fp16_resident_total_mb": fp16_resident_per_vector * row_count / 1e6,
        "fp32_total_mb": fp32_per_vector * row_count / 1e6,
        "compression_vs_fp32_x": fp32_per_vector / compact_bytes_per_vector,
        "augmented_resident_vs_vanilla_x": fp16_resident_per_vector / compact_bytes_per_vector,
    }


# --------------------------------------------------------------------------- #
# Update / persist  (the CPU-side augmentation)
# --------------------------------------------------------------------------- #
def time_update_persist(
    vectors: NDArray[np.float32], *, bit_width: int, update_count: int, tmp_dir: str, seed: int
) -> dict[str, Any]:
    """Vanilla full-rewrite persist (O(N)) vs augmented delta export (O(changed)).

    Applies ``update_count`` in-place value updates to an existing index and
    measures the persist cost each way:

    - **vanilla**: no in-place upsert in stock TurboVec, so re-add the changed ids
      and write the *whole* ``.tvim`` (the only stock durability primitive).
    - **augmented**: ``upsert_with_ids`` (in-place slot replace) + ``export_encoded``
      of just the changed rows (the delta the ``.tvd``/``.tvim`` delta store ships).
    """

    n, dim = int(vectors.shape[0]), int(vectors.shape[1])
    rng = np.random.default_rng(seed + 1)
    upd_idx = rng.choice(n, size=min(update_count, n), replace=False)
    upd_ids = (upd_idx + 1).astype(np.uint64)
    upd_vecs = make_vectors(len(upd_idx), dim, seed=seed + 2)

    # Vanilla: apply updates by remove+add, then FULL write of the whole index.
    vi = build_index(vectors, bit_width=bit_width)
    t0 = time.perf_counter()
    vi.remove_many(upd_ids)
    vi.add_with_ids(upd_vecs, upd_ids)
    vanilla_apply_ms = (time.perf_counter() - t0) * 1000.0
    full_path = os.path.join(tmp_dir, "vanilla_full.tvim")
    t0 = time.perf_counter()
    vi.write(full_path)
    vanilla_full_write_ms = (time.perf_counter() - t0) * 1000.0
    full_bytes = os.path.getsize(full_path)

    # Augmented: in-place upsert, then export ONLY the changed encoded rows (delta).
    ai = build_index(vectors, bit_width=bit_width)
    t0 = time.perf_counter()
    ai.upsert_with_ids(upd_vecs, upd_ids)
    augmented_apply_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    delta = ai.export_encoded(upd_ids)  # O(changed rows) encoded delta
    augmented_delta_export_ms = (time.perf_counter() - t0) * 1000.0
    delta_bytes = int(getattr(delta, "nbytes", 0)) or _encoded_delta_bytes(delta)

    return {
        "row_count": n,
        "dim": dim,
        "bit_width": int(bit_width),
        "update_count": int(len(upd_idx)),
        "vanilla_apply_ms": vanilla_apply_ms,
        "vanilla_full_write_ms": vanilla_full_write_ms,
        "vanilla_full_write_bytes": int(full_bytes),
        "augmented_apply_upsert_ms": augmented_apply_ms,
        "augmented_delta_export_ms": augmented_delta_export_ms,
        "augmented_delta_bytes": int(delta_bytes),
        "persist_speedup_x": (
            vanilla_full_write_ms / augmented_delta_export_ms
            if augmented_delta_export_ms > 0
            else 0.0
        ),
    }


def _encoded_delta_bytes(delta: Any) -> int:
    """Best-effort byte count of an exported encoded delta of unknown shape."""

    if isinstance(delta, (tuple, list)):
        return int(sum(getattr(part, "nbytes", 0) for part in delta))
    return int(getattr(delta, "nbytes", 0))


# --------------------------------------------------------------------------- #
# Cell runner (subprocess-friendly: one config -> one JSON on stdout)
# --------------------------------------------------------------------------- #
def machine_info() -> dict[str, Any]:
    """Returns recorded machine/runtime facts for provenance."""

    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", "default(all)"),
    }
    try:
        import turbovec  # noqa: F401

        from lodedb.engine.turbovec_index import turbovec_capability

        cap = turbovec_capability()
        info["turbovec_native_backend"] = getattr(cap, "native_backend", None)
        info["turbovec_native_used"] = getattr(cap, "native_used", None)
    except Exception as exc:  # noqa: BLE001
        info["turbovec_capability_error"] = repr(exc)
    return info


def run_cell(cfg: dict[str, Any]) -> dict[str, Any]:
    """Runs one benchmark cell described by ``cfg`` and returns a JSON-able dict.

    ``cfg`` keys: ``axis`` (speed|recall|memory|update), ``dim``, ``bit_width``,
    ``n`` (corpus), ``queries``, ``k``, ``repeats``, ``batch_size``, ``seed``,
    ``include_gpu`` (bool), ``vectors_path``/``queries_path`` (optional .npy for
    real datasets; otherwise synthetic).
    """

    axis = cfg["axis"]
    dim = int(cfg["dim"])
    bit_width = int(cfg.get("bit_width", 4))
    n = int(cfg["n"])
    seed = int(cfg.get("seed", 0))
    k = int(cfg.get("k", DEFAULT_K))
    repeats = int(cfg.get("repeats", 5))
    include_gpu = bool(cfg.get("include_gpu", True))

    if cfg.get("vectors_path"):
        vectors = np.load(cfg["vectors_path"]).astype(np.float32)[:n]
        dim = int(vectors.shape[1])
        n = int(vectors.shape[0])
    else:
        vectors = make_vectors(n, dim, seed=seed)
    vectors = _unit(vectors)  # exact ground truth ranks by cosine, like TurboVec

    result: dict[str, Any] = {"config": {**cfg, "dim": dim, "n": n}, "machine": machine_info()}

    if axis == "memory":
        index = build_index(vectors, bit_width=bit_width)
        result["memory"] = memory_profile(index, row_count=n, dim=dim)
        return result

    if axis == "update":
        import tempfile

        with tempfile.TemporaryDirectory(prefix="tv-update-") as td:
            result["update"] = time_update_persist(
                vectors,
                bit_width=bit_width,
                update_count=int(cfg.get("update_count", 1000)),
                tmp_dir=td,
                seed=seed,
            )
        return result

    # speed / recall both need queries.
    qn = int(cfg.get("queries", 1000))
    if cfg.get("queries_path"):
        queries = np.load(cfg["queries_path"]).astype(np.float32)[:qn]
    else:
        queries = make_vectors(qn, dim, seed=seed + 7)
    queries = _unit(queries)
    index = build_index(vectors, bit_width=bit_width)

    if axis == "speed":
        which = cfg.get("which", "both")  # both | vanilla | augmented
        if which in ("both", "vanilla"):
            van = time_vanilla_search(index, queries, k=k, repeats=repeats)
            van.pop("ids", None)
            result["speed_vanilla"] = van
        if which in ("both", "augmented") and include_gpu:
            aug = time_augmented_gpu_search(
                index, queries, k=k, repeats=repeats, batch_size=int(cfg.get("batch_size", 64))
            )
            aug.pop("ids", None)
            result["speed_augmented"] = aug
        return result

    if axis == "recall":
        truth = exact_topk_ids(vectors, queries, max(K_GRID))
        van = time_vanilla_search(index, queries, k=max(K_GRID), repeats=1)
        result["recall_vanilla"] = recall_curve(van["ids"], truth)
        if include_gpu:
            aug = time_augmented_gpu_search(
                index, queries, k=max(K_GRID), repeats=1, batch_size=int(cfg.get("batch_size", 64))
            )
            if "skipped" in aug:
                result["recall_augmented"] = {"skipped": aug["skipped"]}
            else:
                result["recall_augmented"] = recall_curve(aug["ids"], truth)
        return result

    raise ValueError(f"unknown axis: {axis!r}")


def main(argv: list[str]) -> int:
    """Subprocess entry: ``python bench.py '<json-config>'`` -> JSON result on stdout."""

    if len(argv) != 2:
        print("usage: bench.py '<json-config>'", file=sys.stderr)
        return 2
    cfg = json.loads(argv[1])
    result = run_cell(cfg)
    sys.stdout.write(json.dumps(result, default=_json_default))
    return 0


def _json_default(obj: Any) -> Any:
    """Serializes numpy scalars/arrays left in a result by accident."""

    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
