"""LodeDB filtered batch-search benchmark payload (runs locally or on Modal GPU).

Measures the *filtering asymmetry* in the multi-query path. Before the
allowlist-pushdown fix, an unfiltered ``search_many`` rode the GPU-resident
scan but a *filtered* one widened the effective top_k to ``len(chunks)``, which
tripped the resident top_k cap (``GPU_DIRECT_TURBOVEC_MAX_TOP_K = 4096``) and
silently bypassed the GPU to the CPU SIMD kernel + an O(N) Python post-filter.
Run it from a checkout before and after the fix to see the cliff close.

For each (gpu_policy, batch_size, condition) it records latency and the redacted
``query_batch_completed`` telemetry — crucially ``gpu_stage_one_status`` and
``gpu_fallback_reason`` — so the cliff is *proven*, not just timed:
unfiltered+auto → ``used``; filtered+auto (corpus > 4096) → ``bypassed`` /
``*_top_k_exceeds_limit``. The ``off`` policy rows give the pure CPU-kernel
baseline, and the host CPU ISA (AVX2 vs AVX-512, which Modal varies per run) is
captured alongside the kernel's own backend label. Output is raw-payload-free.
"""

from __future__ import annotations

import platform
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.runtime_policy import GpuDirectTurboVecPolicy
from lodedb.local import LodeDB
from lodedb.local.presets import resolve_preset


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


def _query_batch(db: LodeDB, texts: list[str], *, top_k: int, filt: dict | None) -> dict[str, Any]:
    """Runs one redacted engine query batch, optionally filtered."""

    items: list[dict[str, Any]] = []
    for text in texts:
        item: dict[str, Any] = {"query": text, "top_k": int(top_k)}
        if filt is not None:
            item["filter"] = filt
        items.append(item)
    return db._index.query_batch(items)


def _last_event(db: LodeDB) -> dict[str, Any]:
    """Returns the most recent redacted query_batch_completed audit event."""

    for event in reversed(db._engine.audit_events):
        if event.get("event") == "query_batch_completed":
            return dict(event)
    return {}


def _median_ms(fn, repeat: int) -> float:
    fn()  # warmup: first call builds + uploads the GPU-resident session
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def run_filtered_batch_bench(
    *,
    model: str = "bge",
    n_docs: int = 200_000,
    batch_sizes: tuple[int, ...] = (8, 32, 128),
    top_k: int = 10,
    repeat: int = 5,
    device: str = "cuda",
) -> dict[str, Any]:
    """Builds one index, then sweeps (gpu_policy x batch_size x filter-condition)."""

    from lodedb.engine.gpu_turbovec import turbovec_reconstruction_api_available

    native_dim = resolve_preset(model).native_dim
    persistence_dir = Path(tempfile.mkdtemp())
    db = LodeDB(
        path=persistence_dir,
        model=model,
        device=device,
        _embedding_backend=HashEmbeddingBackend(native_dim=native_dim),
    )
    started = time.perf_counter()
    db.add_many(_make_docs(n_docs))
    build_seconds = time.perf_counter() - started

    serving = db._engine._turbovec_index_for_state(next(iter(db._engine._indexes.values())))
    reconstruction_available = bool(turbovec_reconstruction_api_available(serving.index))

    conditions = [
        ("unfiltered", None),
        ("filtered_selective_1pct", {"metadata": {"sel": "hit"}}),
        ("filtered_nonselective_50pct", {"metadata": {"half": "a"}}),
    ]
    policies = [("auto", GpuDirectTurboVecPolicy.AUTO), ("off", GpuDirectTurboVecPolicy.OFF)]

    rows: list[dict[str, Any]] = []
    for policy_name, policy in policies:
        db._engine.gpu_direct_turbovec_policy = policy
        for batch_size in batch_sizes:
            texts = [
                f"query about token{(q * 7) % 997} and document {q}" for q in range(batch_size)
            ]
            for name, filt in conditions:
                latency = _median_ms(
                    lambda t=texts, f=filt: _query_batch(db, t, top_k=top_k, filt=f), repeat
                )
                event = _last_event(db)
                rows.append(
                    {
                        "gpu_policy": policy_name,
                        "batch_size": batch_size,
                        "condition": name,
                        "latency_ms": round(latency, 3),
                        "per_query_ms": round(latency / batch_size, 4),
                        "gpu_stage_one_status": str(event.get("gpu_stage_one_status", "")),
                        "gpu_fallback_reason": str(event.get("gpu_fallback_reason", "")),
                        "native_query_used": bool(event.get("native_query_used", False)),
                        "native_backend": str(event.get("native_backend", "")),
                        "stage_one_backend": str(event.get("stage_one_backend", "")),
                        "gpu_stage_one_search_ms": float(
                            event.get("gpu_stage_one_search_ms", 0.0) or 0.0
                        ),
                    }
                )

    db.close()
    return {
        "model": model,
        "n_docs": n_docs,
        "native_dim": native_dim,
        "top_k": top_k,
        "gpu_resident_cap": 4096,
        "reconstruction_api_available": reconstruction_available,
        "build_seconds": round(build_seconds, 2),
        "device": device,
        "cpu": _cpu_info(),
        "rows": rows,
    }


if __name__ == "__main__":  # local smoke (CPU): proves telemetry plumbing
    import json

    print(
        json.dumps(
            run_filtered_batch_bench(
                model="minilm", n_docs=20_000, batch_sizes=(8, 32), repeat=3, device="cpu"
            ),
            indent=2,
        )
    )
