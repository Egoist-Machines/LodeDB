"""Commit-overhead and reopen benchmark for the persistent BM25 postings store.

Quantifies the opt-in persistent lexical index (``index_text=True``, the
durable postings store) against the default vector-only flow
(``index_text=False``):

- commit latency per incremental write (does enabling it slow the write path?),
- query latency per mode (vector vs hybrid vs lexical) on one corpus,
- reopen/load time and on-disk bytes (does hybrid survive a reopen with no
  retained raw text?),
- exact-token recall (vector vs hybrid), the motivating signal.

Metrics-only and payload-free: it records counts, bytes, and latencies, never
document text, tokens, or query strings. Runs the same locally or on Modal; the
GPU-resident batch scan serves ``search_many`` automatically when CuPy is present.
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

# Exact-token shapes the embedding cannot encode but the lexical ranker matches.
_TOKEN_SHAPES = ("E{:04d}", "ABC-{:03d}", "2024-{:02d}-{:02d}")
_PROSE = (
    "the overnight maintenance log records that the auxiliary system reported a "
    "transient condition before the unit recovered and returned to nominal load "
    "across every monitored region during the reporting window under review "
)


def _planted_token(index: int) -> str:
    """Returns a deterministic exact token for one carrier document."""

    shape = _TOKEN_SHAPES[index % len(_TOKEN_SHAPES)]
    if "{:02d}-{:02d}" in shape:
        return shape.format(1 + index % 12, 1 + index % 28)
    return shape.format(index % 9000)


def _make_documents(scale: int, *, plant_every: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Builds deterministic synthetic docs; every ``plant_every``-th carries a token."""

    documents: list[dict[str, Any]] = []
    planted: dict[str, str] = {}
    for i in range(scale):
        body = f"{_PROSE}document {i} segment notes and miscellaneous unrelated asides"
        if i % plant_every == 0:
            token = _planted_token(i)
            body = f"controller reported fault {token} during {body}"
            planted[token] = f"doc-{i}"
        documents.append({"text": body, "id": f"doc-{i}", "metadata": {"shard": str(i % 8)}})
    return documents, planted


def _summary(samples_ms: list[float]) -> dict[str, float]:
    """Returns count/mean/p50/p95 for a list of millisecond samples."""

    if not samples_ms:
        return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(samples_ms)
    return {
        "count": len(samples_ms),
        "mean_ms": float(statistics.fmean(samples_ms)),
        "p50_ms": float(ordered[len(ordered) // 2]),
        "p95_ms": float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]),
    }


def _dir_bytes(path: Path, *, suffix: str | None = None) -> int:
    """Returns total bytes of files under ``path`` (optionally filtered by suffix)."""

    total = 0
    for file in path.rglob("*"):
        if file.is_file() and (suffix is None or file.name.endswith(suffix)):
            total += file.stat().st_size
    return total


def _served_backend(db: Any) -> str:
    """Best-effort: reports which scan backend last served a query (telemetry only)."""

    try:
        metrics = db._index.engine.metrics  # noqa: SLF001 - benchmark introspection (a property)
    except Exception:  # noqa: BLE001
        return "unknown"
    for metric in reversed(metrics):
        backend = metric.get("stage_one_backend") or metric.get("native_backend")
        if backend:
            return str(backend)
    return "none"


def _measure_config(
    *,
    make_db,
    documents: list[dict[str, Any]],
    planted: dict[str, str],
    ingest_batch: int,
    incremental: int,
    query_batch: int,
    query_count: int,
    top_k: int,
    lexical_capable: bool,
    post_mutation_rounds: int = 15,
) -> dict[str, Any]:
    """Builds one corpus, then measures commit, query, storage, and reopen costs."""

    db = make_db()
    # Cold build (batched commits).
    build_started = time.perf_counter()
    for offset in range(0, len(documents), ingest_batch):
        db.add_many(documents[offset : offset + ingest_batch])
    build_ms = (time.perf_counter() - build_started) * 1000.0

    # Incremental single-document commits (each one commits atomically).
    commit_samples: list[float] = []
    base = len(documents)
    for i in range(incremental):
        started = time.perf_counter()
        db.add(f"{_PROSE} incremental delta {i}", id=f"delta-{i}", metadata={"shard": "9"})
        commit_samples.append((time.perf_counter() - started) * 1000.0)
    del base

    # Query latency per available mode (batched -> GPU when present).
    probes = sorted(planted)[:query_batch] or ["system recovered nominal load"]
    modes = ["vector"] + (["hybrid", "lexical"] if lexical_capable else [])
    query_latency: dict[str, Any] = {}
    backends: dict[str, str] = {}
    recall: dict[str, float] = {}
    for mode in modes:
        samples: list[float] = []
        for _ in range(query_count):
            started = time.perf_counter()
            db.search_many(probes, k=top_k, mode=mode)
            samples.append((time.perf_counter() - started) * 1000.0)
        query_latency[mode] = _summary(samples)
        backends[mode] = _served_backend(db)
        rows = db.search_many(probes, k=top_k, mode=mode)
        hits = sum(
            1
            for token, row in zip(probes, rows, strict=True)
            if planted.get(token) in {h.id for h in row}
        )
        recall[mode] = hits / len(probes)

    # Follow-up B signal: the cost of the first hybrid query right after a single
    # mutation. The incremental path folds the one added chunk into the cached
    # BM25 index in place; the forced path clears the cache so the same query
    # pays a full O(corpus) rebuild. The gap is the incremental-maintenance win.
    incremental_samples: list[float] = []
    rebuild_samples: list[float] = []
    probe_slice = probes[: min(8, len(probes))]
    engine = getattr(getattr(db, "_index", None), "engine", None)
    can_force = engine is not None and hasattr(engine, "_lexical_indexes")
    for i in range(post_mutation_rounds):
        db.add(f"{_PROSE} post-mutation probe {i}", id=f"pm-{i}", metadata={"shard": "9"})
        started = time.perf_counter()
        db.search_many(probe_slice, k=top_k, mode="hybrid")
        incremental_samples.append((time.perf_counter() - started) * 1000.0)
        if can_force:
            engine._lexical_indexes.clear()  # noqa: SLF001 - force a full rebuild to compare
            started = time.perf_counter()
            db.search_many(probe_slice, k=top_k, mode="hybrid")
            rebuild_samples.append((time.perf_counter() - started) * 1000.0)

    persist_dir = Path(db.path)
    total_bytes = _dir_bytes(persist_dir)
    lexical_bytes = _dir_bytes(persist_dir, suffix=".tvlex")
    lexical_bytes += _dir_bytes(persist_dir, suffix=".lxd")
    db.close()

    # Reopen: load-on-open replays the journals, so the constructor time is the
    # reopen cost. Then confirm hybrid still works (on the no-raw-text config,
    # this proves it works with no retained text at all).
    reopen_started = time.perf_counter()
    reopened = make_db()
    reopen_ms = (time.perf_counter() - reopen_started) * 1000.0
    post_recall = None
    if lexical_capable:
        rows = reopened.search_many(probes, k=top_k, mode="hybrid")
        post_recall = sum(
            1
            for token, row in zip(probes, rows, strict=True)
            if planted.get(token) in {h.id for h in row}
        ) / len(probes)
    reopened.close()

    return {
        "document_count": len(documents) + incremental + post_mutation_rounds,
        "cold_build_ms": build_ms,
        "incremental_commit": _summary(commit_samples),
        "query_latency_ms": query_latency,
        "query_backend": backends,
        "recall_at_k": recall,
        "post_mutation_first_query_incremental_ms": _summary(incremental_samples),
        "post_mutation_first_query_full_rebuild_ms": _summary(rebuild_samples),
        "reopen_load_ms": reopen_ms,
        "post_reopen_hybrid_recall_at_k": post_recall,
        "bytes_total": total_bytes,
        "bytes_lexical_sidecar": lexical_bytes,
    }


def run_persist_bench(
    *,
    scale: int = 20_000,
    plant_every: int = 50,
    ingest_batch: int = 2_000,
    incremental: int = 200,
    query_batch: int = 64,
    query_count: int = 20,
    dim: int = 384,
    top_k: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Runs the persistence benchmark across the default and index_text configs."""

    import platform

    import numpy as np

    from lodedb.engine.embedding_backends import HashEmbeddingBackend
    from lodedb.local import LodeDB

    del seed  # documents are deterministic by construction
    np.random.seed(0)
    documents, planted = _make_documents(scale, plant_every=plant_every)

    root = Path(tempfile.mkdtemp(prefix="lodedb-persist-bench-"))

    def make_db(name: str, *, store_text: bool, index_text: bool):
        return lambda: LodeDB(
            path=root / name,
            store_text=store_text,
            index_text=index_text,
            _embedding_backend=HashEmbeddingBackend(native_dim=dim),
        )

    # Three configs. The first two hold ``store_text=True`` constant and toggle
    # only ``index_text``, so their commit-latency delta is the pure additive
    # cost of journaling the postings on top of the default flow. The third drops
    # raw text entirely to show hybrid still works after a reopen from the
    # persisted tokens alone.
    make_baseline = make_db("baseline", store_text=True, index_text=False)
    make_with_index = make_db("with_index_text", store_text=True, index_text=True)
    make_no_raw = make_db("index_text_no_store_text", store_text=False, index_text=True)

    bundle: dict[str, Any] = {
        "config": {
            "scale": scale,
            "plant_every": plant_every,
            "ingest_batch": ingest_batch,
            "incremental": incremental,
            "query_batch": query_batch,
            "query_count": query_count,
            "dim": dim,
            "top_k": top_k,
            "planted_token_count": len(planted),
        },
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    try:
        import torch  # noqa: PLC0415

        bundle["machine"]["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            bundle["machine"]["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        bundle["machine"]["cuda_available"] = False

    common = {
        "documents": documents,
        "planted": planted,
        "ingest_batch": ingest_batch,
        "incremental": incremental,
        "query_batch": query_batch,
        "query_count": query_count,
        "top_k": top_k,
    }
    bundle["baseline_store_text"] = _measure_config(
        make_db=make_baseline, lexical_capable=True, **common
    )
    bundle["with_index_text"] = _measure_config(
        make_db=make_with_index, lexical_capable=True, **common
    )
    bundle["index_text_no_store_text"] = _measure_config(
        make_db=make_no_raw, lexical_capable=True, **common
    )

    off = bundle["baseline_store_text"]["incremental_commit"]["mean_ms"]
    on = bundle["with_index_text"]["incremental_commit"]["mean_ms"]
    bundle["commit_overhead"] = {
        "note": "store_text held True; the delta is the additive cost of index_text",
        "baseline_mean_ms": off,
        "with_index_text_mean_ms": on,
        "absolute_ms": on - off,
        "relative_pct": (100.0 * (on - off) / off) if off else 0.0,
    }
    return bundle
