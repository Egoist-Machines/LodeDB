"""Reproducible local benchmark: embedding throughput + CPU scan latency.

Honest scope (constraint #5 + #7): on Apple Silicon, embedding is the only
accelerated stage (MPS); the TurboVec vector scan runs on the CPU kernel.
This benchmark therefore reports two distinct numbers and makes no
GPU-vector-search-on-Mac claim:

1. **Embedding throughput**: docs/sec for the selected backend/device.
2. **CPU vector-scan latency**: per-query p50/p95 *search* latency (the native
   TurboVec scan + result assembly), measured with embedding excluded by timing
   the query embedding and the native vector scan separately.

Reuses :class:`LodeDB` (hence the native core + TurboVec storage); it does not
reimplement search.
"""

from __future__ import annotations

import platform
import time
from pathlib import Path
from typing import Any

from lodedb.local.backends import is_apple_silicon, resolve_local_device
from lodedb.local.db import LodeDB


def _percentile(values: list[float], pct: float) -> float:
    """Returns the ``pct`` percentile (0-100) via nearest-rank on sorted values."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[rank])


def _synthetic_documents(count: int) -> list[dict[str, Any]]:
    """Builds deterministic, varied synthetic documents for indexing."""

    topics = (
        "machine learning models train on large corpora of text and images",
        "the central bank adjusted interest rates amid inflation concerns",
        "photosynthesis converts sunlight carbon dioxide and water into glucose",
        "the marathon runner paced steadily through the final mile",
        "quantum entanglement links the states of distant particles",
        "the chef reduced the sauce and plated the seared scallops",
        "tectonic plates shift slowly producing earthquakes over time",
        "the orchestra tuned before the conductor raised the baton",
    )
    docs: list[dict[str, Any]] = []
    for i in range(count):
        base = topics[i % len(topics)]
        docs.append(
            {
                "text": f"{base} (record {i}, variant {i % 17})",
                "id": f"bench-{i}",
                "metadata": {"bucket": str(i % 8)},
            }
        )
    return docs


def run_local_benchmark(
    *,
    path: str | Path,
    model: str = "minilm",
    device: str = "auto",
    embedding_runtime: str = "auto",
    doc_count: int = 2000,
    query_count: int = 200,
    top_k: int = 10,
    embed_batch_size: int = 64,
) -> dict[str, Any]:
    """Runs the embedding + CPU-scan benchmark and returns a redacted summary."""

    effective_device = resolve_local_device(device)
    db = LodeDB(
        path=path,
        model=model,
        device=device,
        embedding_runtime=embedding_runtime,
        batch_size=embed_batch_size,
    )

    documents = _synthetic_documents(doc_count)

    # Warm up the embedding backend (load weights once) so the timed throughput
    # below is warm steady state, not the one-time model load. Without this the
    # figure is dominated by cold start and depends on call ordering.
    db._embedding_backend.embed_documents((documents[0]["text"],))

    embed_start = time.perf_counter()
    db.add_many(documents)
    embed_seconds = time.perf_counter() - embed_start
    docs_per_second = doc_count / embed_seconds if embed_seconds > 0 else 0.0

    db.persist()

    # Query latency: time the query embedding and the native vector scan separately
    # so the search-only figure excludes embedding. Embedding the query and then
    # calling search_by_vector is exactly what db.search does internally (embed, then
    # scan), so the end-to-end figure is their sum and the search-only figure is the
    # native scan + result assembly that an application pays beyond embedding.
    queries = [documents[i % doc_count]["text"] for i in range(query_count)]
    backend = db._embedding_backend
    total_latencies: list[float] = []
    search_latencies: list[float] = []
    for text in queries:
        start = time.perf_counter()
        vector = backend.embed_query(text)
        embedded = time.perf_counter()
        db.search_by_vector(vector, k=top_k)
        finished = time.perf_counter()
        total_latencies.append((finished - start) * 1000.0)
        search_latencies.append((finished - embedded) * 1000.0)

    stats = db.stats()
    storage = stats.get("storage", {}) if isinstance(stats.get("storage"), dict) else {}
    db.close()

    return {
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "apple_silicon": is_apple_silicon(),
        },
        "config": {
            "model": model,
            "requested_device": device,
            "effective_device": effective_device,
            "embedding_backend": db.embedding_resolution.backend_name,
            "fallback_used": db.embedding_resolution.fallback_used,
            "fallback_reason": db.embedding_resolution.fallback_reason,
            "doc_count": doc_count,
            "query_count": query_count,
            "top_k": top_k,
            "embed_batch_size": embed_batch_size,
        },
        "embedding": {
            "indexed_docs": doc_count,
            "embed_seconds": round(embed_seconds, 4),
            "docs_per_second": round(docs_per_second, 2),
        },
        "cpu_vector_scan": {
            "note": (
                "TurboVec scan runs on the CPU kernel on Apple Silicon; no GPU "
                "vector search on Mac."
            ),
            "search_only_ms_p50": round(_percentile(search_latencies, 50), 4),
            "search_only_ms_p95": round(_percentile(search_latencies, 95), 4),
            "end_to_end_query_ms_p50": round(_percentile(total_latencies, 50), 4),
            "end_to_end_query_ms_p95": round(_percentile(total_latencies, 95), 4),
        },
        "storage": {
            key: storage.get(key)
            for key in (
                "tvim_base_bytes",
                "tvim_delta_bytes",
                "json_base_bytes",
                "json_delta_bytes",
                "snapshot_bytes",
            )
            if key in storage
        },
        "document_count": stats.get("document_count"),
        "chunk_count": stats.get("chunk_count"),
        "raw_payload_text_present": stats.get("raw_payload_text_present", False),
    }
