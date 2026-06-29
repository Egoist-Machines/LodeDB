#!/usr/bin/env python3
"""Batched-retrieval throughput: ``LodeDB.search_many`` queries/sec vs batch size.

This benchmarks the **public SDK path** ``LodeDB.search_many(queries, k=...)`` — the
batched entry point that lets CUDA hosts serve a query batch from the GPU-resident exact
scan. Single-query ``search`` never takes that path, so batched ``search_many`` is the only
way the GPU-batch story shows up through the supported API. It reports **queries/sec**
across batch sizes for the native CPU kernel and, when a CUDA driver is present, the native
GPU-resident path — so the batch crossover is visible end to end. The native scan reads
``LODEDB_GPU_DIRECT_TURBOVEC`` per scan (``off`` forces the CPU baseline, any other value
leaves the GPU path eligible), toggled here without rebuilding the index.

Embedding is intentionally excluded: queries are embedded by a trivial local hash backend
so the number isolates **retrieval** (the stage the GPU accelerates), the same property a
batched-retrieval integration would lean on. Metrics only — counts, batch sizes, timings,
throughput, and a CPU-vs-GPU overlap; never documents, queries, or embeddings.

    uv run python benchmarks/batched_retrieval/run.py --docs 50000 --queries 1024

On a non-CUDA host this reports the CPU-kernel curve only. Render charts with
``benchmarks/batched_retrieval/diagrams.py`` (needs ``matplotlib``).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.native_adapter import NativeCoreAdapter
from lodedb.local.db import LodeDB
from lodedb.local.presets import resolve_preset

DEFAULT_BATCH_SIZES = "1,16,64,256,1024"

# The Rust core reads this per scan; "off" disables the GPU-resident scan, any
# other value (or unset) leaves it eligible, still subject to the CUDA driver and
# the batch/corpus gates in the engine.
_GPU_SCAN_ENV = "LODEDB_GPU_DIRECT_TURBOVEC"

_TOPICS = (
    "machine learning models train on large corpora of text and images",
    "the central bank adjusted interest rates amid inflation concerns",
    "photosynthesis converts sunlight carbon dioxide and water into glucose",
    "the marathon runner paced steadily through the final mile",
    "quantum entanglement links the states of distant particles",
    "the chef reduced the sauce and plated the seared scallops",
    "tectonic plates shift slowly producing earthquakes over time",
    "the orchestra tuned before the conductor raised the baton",
)


@contextmanager
def _gpu_scan(enabled: bool):
    """Toggles the native GPU-resident scan for the duration of the block.

    Flips ``LODEDB_GPU_DIRECT_TURBOVEC`` (the Rust core reads it per scan), so the
    CPU and GPU rows share one built index without a rebuild, and restores the
    prior value on exit.
    """

    previous = os.environ.get(_GPU_SCAN_ENV)
    os.environ[_GPU_SCAN_ENV] = "auto" if enabled else "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_GPU_SCAN_ENV, None)
        else:
            os.environ[_GPU_SCAN_ENV] = previous


def _synthetic_documents(count: int) -> list[dict[str, Any]]:
    """Builds deterministic, varied synthetic documents (one chunk each)."""

    docs: list[dict[str, Any]] = []
    for i in range(count):
        base = _TOPICS[i % len(_TOPICS)]
        docs.append(
            {
                "text": f"{base} (record {i}, variant {i % 17})",
                "id": f"bench-{i}",
                "metadata": {"bucket": str(i % 8)},
            }
        )
    return docs


def _synthetic_queries(count: int) -> list[str]:
    """Builds ``count`` distinct query strings (varied so vectors differ)."""

    return [f"{_TOPICS[i % len(_TOPICS)]} query {i} facet {i % 13}" for i in range(count)]


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    """Parses and validates comma-separated positive batch sizes."""

    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    return sizes


def _mean_overlap(left: list[list[str]], right: list[list[str]]) -> float:
    """Returns mean top-k set overlap for paired result rows."""

    if not left:
        return 1.0
    scores: list[float] = []
    for a, b in zip(left, right, strict=True):
        denom = max(1, min(len(a), len(b)))
        scores.append(len(set(a).intersection(b)) / denom)
    return float(statistics.fmean(scores)) if scores else 1.0


def _measure(
    db: LodeDB,
    *,
    queries_pool: list[str],
    batch_size: int,
    k: int,
    repeats: int,
    gpu_enabled: bool,
    label: str,
) -> dict[str, Any]:
    """Times ``repeats`` ``search_many`` calls at one batch size / GPU setting."""

    batch = [queries_pool[i % len(queries_pool)] for i in range(batch_size)]

    with _gpu_scan(gpu_enabled):
        db.search_many(batch, k=k)  # warm up (resident upload / kernel caches)

        per_call_ms: list[float] = []
        elapsed = 0.0
        served: list[list[str]] = []
        for repeat in range(max(1, repeats)):
            started = time.perf_counter()
            results = db.search_many(batch, k=k)
            dt = time.perf_counter() - started
            elapsed += dt
            per_call_ms.append(dt * 1000.0)
            if repeat == 0:
                served = [[hit.id for hit in hits] for hits in results]

    total_queries = batch_size * max(1, repeats)
    qps = total_queries / elapsed if elapsed > 0 else 0.0
    return {
        "label": label,
        "gpu_enabled": bool(gpu_enabled),
        "batch_size": int(batch_size),
        "repeats": int(max(1, repeats)),
        "k": int(k),
        "queries_per_second": round(qps, 1),
        "per_call_ms_p50": round(statistics.median(per_call_ms), 4),
        "per_query_ms": round((elapsed * 1000.0) / total_queries, 5),
        "_served": served,
    }


def run(
    *,
    doc_count: int,
    query_count: int,
    model: str,
    k: int,
    batch_sizes: tuple[int, ...],
    repeats: int,
) -> dict[str, Any]:
    """Builds one index and sweeps ``search_many`` throughput over batch sizes."""

    preset = resolve_preset(model)
    backend = HashEmbeddingBackend(native_dim=preset.native_dim)
    pool_size = max(query_count, max(batch_sizes))
    gpu_available = bool(NativeCoreAdapter().cuda_runtime_available())

    with contextlib.redirect_stdout(sys.stderr), tempfile.TemporaryDirectory() as tmp:
        db = LodeDB(path=tmp, model=model, _embedding_backend=backend)
        print(
            f"[batched-retrieval] building {doc_count} docs (dim={preset.native_dim}, "
            f"hash backend) ...",
            file=sys.stderr,
        )
        started = time.perf_counter()
        db.add_many(_synthetic_documents(doc_count))
        build_seconds = time.perf_counter() - started
        db.persist()
        queries_pool = _synthetic_queries(pool_size)

        rows: list[dict[str, Any]] = []
        for batch_size in batch_sizes:
            cpu = _measure(
                db,
                queries_pool=queries_pool,
                batch_size=batch_size,
                k=k,
                repeats=repeats,
                gpu_enabled=False,
                label=f"cpu_batch_{batch_size}",
            )
            cpu_served = cpu.pop("_served")
            rows.append(cpu)
            print(
                f"[batched-retrieval] cpu  batch={batch_size:>4}  "
                f"{cpu['queries_per_second']:>12,.1f} q/s",
                file=sys.stderr,
            )
            if gpu_available:
                gpu = _measure(
                    db,
                    queries_pool=queries_pool,
                    batch_size=batch_size,
                    k=k,
                    repeats=repeats,
                    gpu_enabled=True,
                    label=f"gpu_batch_{batch_size}",
                )
                gpu_served = gpu.pop("_served")
                cpu_qps = cpu["queries_per_second"]
                gpu["speedup_vs_cpu"] = (
                    round(gpu["queries_per_second"] / cpu_qps, 2) if cpu_qps else None
                )
                gpu["gpu_vs_cpu_top_k_overlap"] = round(_mean_overlap(cpu_served, gpu_served), 4)
                rows.append(gpu)
                print(
                    f"[batched-retrieval] gpu  batch={batch_size:>4}  "
                    f"{gpu['queries_per_second']:>12,.1f} q/s  "
                    f"({gpu.get('speedup_vs_cpu')}x, overlap {gpu['gpu_vs_cpu_top_k_overlap']})",
                    file=sys.stderr,
                )
            else:
                cpu.pop("_served", None)

        stats = db.stats()
        db.close()

    return {
        "artifact_type": "lodedb_batched_retrieval_throughput",
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "config": {
            "model": model,
            "native_dim": preset.native_dim,
            "embedding_backend": "hash (embedding excluded; isolates retrieval)",
            "doc_count": doc_count,
            "query_count": query_count,
            "k": k,
            "repeats": repeats,
            "batch_sizes": list(batch_sizes),
        },
        "gpu_available": gpu_available,
        "build_seconds": round(build_seconds, 4),
        "document_count": stats.get("document_count"),
        "chunk_count": stats.get("chunk_count"),
        "rows": rows,
        "raw_payload_text_present": False,
    }


def main() -> None:
    """CLI entry point; writes a metrics-only results JSON."""

    parser = argparse.ArgumentParser(description="LodeDB batched-retrieval throughput")
    parser.add_argument("--docs", type=int, default=50000)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--model", default="minilm", choices=["minilm", "bge"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results" / "results.json"),
    )
    args = parser.parse_args()

    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    if args.queries < max(batch_sizes):
        raise SystemExit("--queries must be at least the largest --batch-sizes value")

    summary = run(
        doc_count=args.docs,
        query_count=args.queries,
        model=args.model,
        k=args.k,
        batch_sizes=batch_sizes,
        repeats=args.repeats,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[batched-retrieval] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
