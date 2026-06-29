"""LodeDB filtered batch-search benchmark payload (runs locally or on Modal GPU).

Measures the *filtering asymmetry* in the multi-query path through the public
``LodeDB.search_many`` API. Filtered and unfiltered batches are timed on the
native CPU scan and, where a CUDA driver is present, the native GPU-resident
scan, so any filtered-vs-unfiltered cliff shows up end to end on the path an
application actually uses. The native scan reads ``LODEDB_GPU_DIRECT_TURBOVEC``
per scan: ``off`` forces the CPU baseline, any other value (the default) leaves
the GPU path eligible, still subject to the CUDA driver and the engine's
batch/corpus gates.

For each (gpu, batch_size, condition) it records latency, per-query latency, and
a CPU-vs-GPU top-k overlap so the GPU scan is shown to rank like the CPU scan
rather than only timed. The host CPU ISA (AVX2 vs AVX-512, which Modal varies
per run) is captured alongside the inferred kernel backend. Output is
raw-payload-free: only counts, batch sizes, timings, and overlap metrics.
"""

from __future__ import annotations

import os
import platform
import statistics
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.turbovec_index import turbovec_capability
from lodedb.local import LodeDB
from lodedb.local.presets import resolve_preset

# The Rust core reads this per scan; "off" disables the GPU-resident scan, any
# other value (or unset) leaves it eligible, still subject to the CUDA driver and
# the batch/corpus gates in the engine.
_GPU_SCAN_ENV = "LODEDB_GPU_DIRECT_TURBOVEC"


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


def _cpu_info() -> dict[str, Any]:
    """Detects host CPU model + SIMD ISA (Modal hands out AVX2 and AVX-512 hosts)."""

    info: dict[str, Any] = {
        "machine": platform.machine(),
        "model": "",
        "avx2": False,
        "avx512": False,
        "simd_flags": "",
    }
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("model name") and not info["model"]:
                info["model"] = line.split(":", 1)[1].strip()
            if line.startswith("flags") and not info["simd_flags"]:
                flags = line.split(":", 1)[1].split()
                info["avx2"] = "avx2" in flags
                info["avx512"] = any(flag.startswith("avx512") for flag in flags)
                info["simd_flags"] = " ".join(
                    flag
                    for flag in flags
                    if flag.startswith("avx") or flag in {"sse4_2", "f16c", "fma"}
                )
    except OSError:
        pass
    return info


def _make_docs(n_docs: int) -> list[dict[str, Any]]:
    """Builds n_docs with metadata controlling filter selectivity."""

    docs: list[dict[str, Any]] = []
    for i in range(n_docs):
        docs.append(
            {
                "text": f"document number {i} lorem ipsum dolor sit amet token{i % 997}",
                "id": f"d{i}",
                "metadata": {
                    "sel": "hit" if i < n_docs // 100 else "miss",  # ~1% selective
                    "half": "a" if i % 2 == 0 else "b",  # ~50% non-selective
                },
            }
        )
    return docs


def _result_ids(db: LodeDB, texts: list[str], *, top_k: int, filt: dict | None) -> list[list[str]]:
    """Runs one batch through the public search_many API; returns per-query id lists."""

    results = db.search_many(texts, k=int(top_k), filter=filt)
    return [[hit.id for hit in hits] for hits in results]


def _median_ms(fn: Callable[[], Any], repeat: int) -> float:
    """Times ``repeat`` calls of ``fn`` and returns the median in milliseconds."""

    fn()  # warmup: first call builds + uploads the GPU-resident session
    samples: list[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _mean_overlap(left: list[list[str]], right: list[list[str]]) -> float:
    """Returns mean top-k set overlap for paired result rows."""

    if not left:
        return 1.0
    scores: list[float] = []
    for a, b in zip(left, right, strict=True):
        denom = max(1, min(len(a), len(b)))
        scores.append(len(set(a).intersection(b)) / denom)
    return float(statistics.fmean(scores)) if scores else 1.0


def run_filtered_batch_bench(
    *,
    model: str = "bge",
    n_docs: int = 200_000,
    batch_sizes: tuple[int, ...] = (8, 32, 128),
    top_k: int = 10,
    repeat: int = 5,
    device: str = "cuda",
) -> dict[str, Any]:
    """Builds one index, then sweeps (gpu x batch_size x filter-condition)."""

    native_dim = resolve_preset(model).native_dim
    persistence_dir = Path(tempfile.mkdtemp())
    db = LodeDB(
        path=persistence_dir,
        model=model,
        device=device,
        _embedding_backend=HashEmbeddingBackend(native_dim=native_dim),
    )
    try:
        started = time.perf_counter()
        db.add_many(_make_docs(n_docs))
        build_seconds = time.perf_counter() - started
        db.persist()

        capability = turbovec_capability()

        conditions = [
            ("unfiltered", None),
            ("filtered_selective_1pct", {"metadata": {"sel": "hit"}}),
            ("filtered_nonselective_50pct", {"metadata": {"half": "a"}}),
        ]

        rows: list[dict[str, Any]] = []
        for gpu_label, gpu_enabled in (("auto", True), ("off", False)):
            for batch_size in batch_sizes:
                texts = [
                    f"query about token{(q * 7) % 997} and document {q}"
                    for q in range(batch_size)
                ]
                for name, filt in conditions:
                    with _gpu_scan(gpu_enabled):
                        latency = _median_ms(
                            lambda t=texts, f=filt: _result_ids(db, t, top_k=top_k, filt=f),
                            repeat,
                        )
                        served = _result_ids(db, texts, top_k=top_k, filt=filt)
                    rows.append(
                        {
                            "gpu": gpu_label,
                            "gpu_enabled": bool(gpu_enabled),
                            "batch_size": batch_size,
                            "condition": name,
                            "latency_ms": round(latency, 3),
                            "per_query_ms": round(latency / batch_size, 4),
                            "result_count": sum(len(ids) for ids in served),
                            "_served": served,
                        }
                    )

        # Pair each GPU row with its CPU twin (same batch_size + condition) and
        # record the top-k overlap, so the GPU scan is shown to rank like the CPU
        # scan, not just timed. The served ids never leave this process.
        cpu_by_key = {
            (row["batch_size"], row["condition"]): row["_served"]
            for row in rows
            if not row["gpu_enabled"]
        }
        for row in rows:
            served = row.pop("_served")
            if row["gpu_enabled"]:
                cpu_served = cpu_by_key.get((row["batch_size"], row["condition"]), served)
                row["gpu_vs_cpu_top_k_overlap"] = round(_mean_overlap(cpu_served, served), 4)
    finally:
        db.close()

    return {
        "artifact_type": "lodedb_filtered_batch",
        "model": model,
        "n_docs": n_docs,
        "native_dim": native_dim,
        "top_k": top_k,
        "reconstruction_api_available": bool(capability.reconstruction_available),
        "build_seconds": round(build_seconds, 2),
        "device": device,
        "cpu": _cpu_info(),
        "turbovec_native_backend": capability.native_backend,
        "rows": rows,
        "raw_payload_text_present": False,
    }


if __name__ == "__main__":  # local smoke (CPU): proves the public-API plumbing
    import json

    print(
        json.dumps(
            run_filtered_batch_bench(
                model="minilm", n_docs=20_000, batch_sizes=(8, 32), repeat=3, device="cpu"
            ),
            indent=2,
        )
    )
