"""Vector-only throughput sweep: the native GPU default vs plain TurboVec CPU.

Builds one vector-only LodeDB (`store_text`/`index_text` off, so this measures the
vector path only, no text storage or BM25), then measures batched vector-search
throughput (queries/sec) across batch sizes for two serving configs, both driven
through the public `search_many_by_vector` API so it exercises exactly the native
path an application gets:

  - ``cpu``       : ``LODEDB_GPU_DIRECT_TURBOVEC=off`` -> plain TurboVec CPU SIMD scan
  - ``gpu``       : GPU-resident scan with the current defaults (fused two-stage
                    top-k on) -> the new default on a CUDA host

`LODEDB_GPU_DIRECT_TURBOVEC` is read by the Rust core per scan, so both configs
share one built index (no rebuild). Corpus and queries are synthetic unit vectors:
timing is content-independent, and recall parity between the CPU and GPU scans is
covered by the crate's parity tests, so this run measures speed only.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from lodedb.local import LodeDB

_GPU_ENV = "LODEDB_GPU_DIRECT_TURBOVEC"

DEFAULT_BATCH_SIZES = (1, 16, 64, 256, 1024)


@contextmanager
def _gpu_scan(enabled: bool):
    """Toggles the native GPU-resident scan for the block (read per scan)."""

    previous = os.environ.get(_GPU_ENV)
    os.environ[_GPU_ENV] = "auto" if enabled else "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_GPU_ENV, None)
        else:
            os.environ[_GPU_ENV] = previous


def _unit_vectors(count: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic L2-normalized f32 vectors."""

    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal((count, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors


def run_vector_search_native(
    *,
    output_dir: str | Path,
    dim: int = 1536,
    n: int = 100_000,
    bit_width: int = 4,
    batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES,
    query_count: int = 1024,
    top_k: int = 10,
    repeats: int = 5,
    add_chunk: int = 10_000,
) -> dict[str, Any]:
    """Builds one vector-only index and sweeps CPU vs GPU throughput by batch size."""

    if query_count < max(batch_sizes):
        raise ValueError("query_count must be at least the largest batch size")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    corpus = _unit_vectors(n, dim, 0x00C0_FFEE)
    queries = [row.tolist() for row in _unit_vectors(query_count, dim, 0x5EED_5EED)]

    tmp = tempfile.TemporaryDirectory(prefix="lodedb-vec-native-")
    db = LodeDB.open_vector_store(
        path=tmp.name,
        vector_dim=dim,
        bit_width=bit_width,
        store_text=False,
        index_text=False,
    )
    try:
        started = time.perf_counter()
        for start in range(0, n, add_chunk):
            chunk = corpus[start : start + add_chunk]
            db.add_vectors_many(
                [{"vector": row.tolist(), "id": f"v{start + i}"} for i, row in enumerate(chunk)],
                normalize=False,  # already unit vectors
            )
        build_seconds = time.perf_counter() - started

        rows: list[dict[str, Any]] = []
        for gpu in (False, True):
            for size in batch_sizes:
                batches = [queries[i : i + size] for i in range(0, len(queries), size)]
                with _gpu_scan(gpu):
                    db.search_many_by_vector(batches[0], k=top_k, normalize=False)  # warm
                    total_queries = 0
                    clock = time.perf_counter()
                    for _ in range(max(1, repeats)):
                        for batch in batches:
                            db.search_many_by_vector(batch, k=top_k, normalize=False)
                            total_queries += len(batch)
                    elapsed = time.perf_counter() - clock
                rows.append(
                    {
                        "series": "gpu" if gpu else "cpu",
                        "batch_size": int(size),
                        "queries_per_sec": total_queries / elapsed,
                        "ms_per_query": elapsed / total_queries * 1000.0,
                        "queries_timed": total_queries,
                    }
                )
                label = "gpu" if gpu else "cpu"
                print(
                    f"[vector-search-native] {label:>3} batch={size:>4} "
                    f"{rows[-1]['queries_per_sec']:>10.1f} q/s  "
                    f"{rows[-1]['ms_per_query']:.4f} ms/query"
                )

        summary = {
            "artifact_type": "lodedb_vector_search_native",
            "dim": dim,
            "n": n,
            "bit_width": bit_width,
            "top_k": top_k,
            "query_count": query_count,
            "repeats": repeats,
            "build_seconds": build_seconds,
            "rows": rows,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary
    finally:
        db.close()
        tmp.cleanup()
