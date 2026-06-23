"""Shared helpers for the memory-integration benchmark.

The benchmark compares the LodeDB adapters for LangChain, LlamaIndex, and mem0
against each framework's own default and common vector-store backends, driving
every backend through the framework's vector-store interface with **one fixed
embedding model** held constant across all of them. Embeddings are computed once
here, so the per-store ingest/query timings isolate the store, not the embedder.

All artifacts are **metrics-only** (counts, bytes, latency, recall, backend
labels). No raw documents, queries, payloads, or embeddings are written to the
result bundle, matching the repo's benchmark provenance rules
(``benchmarks/README.md``).
"""

from __future__ import annotations

import logging
import os
import random
import time
import uuid
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

# A long but single-chunk-friendly cap. LodeDB chunks text past its chunk limit
# into several vectors; capping each document to one chunk keeps the document and
# its single precomputed vector one-to-one, so recall@k against the brute-force
# ground truth is an apples-to-apples id comparison rather than a chunk-vs-doc
# mismatch.
DOC_CHAR_CAP = 900

_TOPICS = ("ml", "bio", "law", "econ", "physics", "history", "art", "med", "eng", "geo")
_CATEGORIES = ("preference", "fact", "event", "task", "profile")

# Always take at least this many durable single-add samples, even if a slow
# full-rewrite store blows the per-backend time budget on the first few.
_MIN_INCREMENTAL_SAMPLES = 3

# chromadb rejects a single add larger than its max batch size (SQLite-bound,
# commonly ~5.4k). Bulk ingest into Chroma is chunked under this; the other
# backends accept the whole batch in one call.
CHROMA_BATCH = 4000


def quiet_logging() -> None:
    """Silences embedding/telemetry chatter so benchmark stdout stays clean."""

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb posthog opt-out
    for name in ("lodedb", "sentence_transformers", "transformers", "httpx", "chromadb"):
        logging.getLogger(name).setLevel(logging.ERROR)


def percentile(values: list[float], pct: float) -> float:
    """Returns the ``pct`` percentile (0..100) of ``values`` (nearest-rank)."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[rank]


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    """Summarizes a list of per-call latencies (ms)."""

    if not samples_ms:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}
    return {
        "count": len(samples_ms),
        "p50_ms": round(median(samples_ms), 4),
        "p95_ms": round(percentile(samples_ms, 95), 4),
        "mean_ms": round(sum(samples_ms) / len(samples_ms), 4),
    }


def dir_bytes(path: str | Path) -> int:
    """Returns the total on-disk size (bytes) of everything under ``path``."""

    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(f.stat().st_size for f in root.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #


def load_corpus(
    dataset_name: str,
    max_documents: int,
    query_count: int,
) -> tuple[list[str], list[str]]:
    """Loads ``(documents, queries)``; ``synthetic`` needs no network.

    Documents are capped to :data:`DOC_CHAR_CAP` characters (one LodeDB chunk),
    so every document maps to exactly one stored vector across all backends.
    """

    if dataset_name == "synthetic":
        rng = random.Random(1234)
        docs = [
            f"Document {i} concerns {rng.choice(_TOPICS)} and "
            + " ".join(rng.choice(_TOPICS) for _ in range(20))
            for i in range(max_documents)
        ]
        queries = [f"information about {_TOPICS[i % len(_TOPICS)]}" for i in range(query_count)]
        return docs, queries

    from datasets import load_dataset  # noqa: PLC0415 - optional heavy dep, Modal-only

    stream = load_dataset("ccdv/govreport-summarization", split="train", streaming=True)
    docs: list[str] = []
    queries: list[str] = []
    for row in stream:
        if len(docs) >= max_documents:
            break
        report = str(row.get("report", "")).strip()
        if not report:
            continue
        docs.append(report[:DOC_CHAR_CAP])
        if len(queries) < query_count:
            summary = str(row.get("summary", "")).strip()
            if summary:
                queries.append(summary[:400])
    if not queries:
        queries = [f"information about {t}" for t in _TOPICS]
    return docs, queries


def doc_ids(n: int) -> list[str]:
    """Stable document ids ``d0..d{n-1}`` shared by every backend in a suite."""

    return [f"d{i}" for i in range(n)]


def uuid_ids(n: int, salt: str) -> list[str]:
    """Deterministic UUID ids (Qdrant/some stores require UUID or int point ids)."""

    namespace = uuid.uuid5(uuid.NAMESPACE_DNS, f"lodedb-memory-bench-{salt}")
    return [str(uuid.uuid5(namespace, str(i))) for i in range(n)]


def rag_metadata(index: int) -> dict[str, Any]:
    """Deterministic filterable metadata for RAG document ``index``."""

    return {"topic": _TOPICS[index % len(_TOPICS)], "year": 2000 + (index % 26)}


def memory_payload(index: int, n_users: int) -> dict[str, Any]:
    """Deterministic mem0-style agent-memory payload for memory ``index``.

    Carries the scalar scoping fields mem0 filters on (``user_id`` / ``agent_id``
    / ``run_id`` / ``category``) plus a ``data`` text field (the memory itself).
    """

    user = index % n_users
    return {
        "user_id": f"u{user}",
        "agent_id": f"agent{user % 3}",
        "run_id": f"run{index % 7}",
        "category": _CATEGORIES[index % len(_CATEGORIES)],
        "data": f"memory {index} about {_TOPICS[index % len(_TOPICS)]}",
    }


# --------------------------------------------------------------------------- #
# shared embedding (one model, computed once, reused by every backend)
# --------------------------------------------------------------------------- #


class Embedded:
    """Precomputed embeddings + a query embedder, shared across backends.

    ``doc_vectors`` / ``query_vectors`` are float32, L2-normalized, and indexed by
    position (``doc_vectors[i]`` is the embedding of document ``i``). ``embed_text``
    re-embeds an arbitrary string with the same model (used to fairly time the
    query-embedding constant that every text-path retrieval pays).
    """

    def __init__(
        self,
        doc_vectors: np.ndarray,
        query_vectors: np.ndarray,
        *,
        native_dim: int,
        effective_device: str,
        doc_embed_ms: float,
        query_embed_ms: float,
        backend: Any,
    ) -> None:
        self.doc_vectors = doc_vectors
        self.query_vectors = query_vectors
        self.native_dim = native_dim
        self.effective_device = effective_device
        self.doc_embed_ms = doc_embed_ms
        self.query_embed_ms = query_embed_ms
        self._backend = backend
        self._by_text: dict[str, np.ndarray] | None = None

    def attach_texts(self, texts: list[str]) -> None:
        """Builds a text->vector lookup so framework adapters can fetch by content."""

        self._by_text = {text: self.doc_vectors[i] for i, text in enumerate(texts)}

    def vector_for_text(self, text: str) -> list[float]:
        """Returns the precomputed vector for a known document text (cache hit)."""

        if self._by_text is not None and text in self._by_text:
            return self._by_text[text].tolist()
        return [float(v) for v in self._backend.embed_query(text)]

    def embed_text(self, text: str) -> list[float]:
        """Embeds an arbitrary string with the shared model (query path)."""

        return [float(v) for v in self._backend.embed_query(text)]


def embed_corpus(
    documents: list[str],
    queries: list[str],
    *,
    model: str,
    device: str,
) -> Embedded:
    """Embeds the corpus + queries once with LodeDB's own backend.

    Using LodeDB's backend (not a separate sentence-transformers call) makes the
    precomputed vectors byte-identical to what LodeDB's text-path adapters produce
    internally, so the only thing the comparison varies is the store.
    """

    from lodedb.local.backends import build_local_embedding_backend  # noqa: PLC0415
    from lodedb.local.presets import resolve_preset  # noqa: PLC0415

    preset = resolve_preset(model)
    backend, resolution = build_local_embedding_backend(preset, device=device)

    # Warm up so the timed embedding is warm steady state, not the one-time model
    # load. This matters because LodeDB's text-path adapters embed internally with
    # their own (also warmed) backend, and the runner subtracts this embed time to
    # get their store-only ingest figure; an unwarmed measurement here would carry
    # the model-load cost and make that subtraction wrong.
    backend.embed_documents((documents[0],))

    t0 = time.perf_counter()
    doc_raw = backend.embed_documents(tuple(documents))
    doc_embed_ms = (time.perf_counter() - t0) * 1000.0
    doc_vectors = _as_unit_f32(doc_raw)

    t0 = time.perf_counter()
    query_raw = [backend.embed_query(q) for q in queries]
    query_embed_ms = (time.perf_counter() - t0) * 1000.0
    query_vectors = _as_unit_f32(query_raw)

    return Embedded(
        doc_vectors,
        query_vectors,
        native_dim=preset.native_dim,
        effective_device=resolution.effective_device,
        doc_embed_ms=doc_embed_ms,
        query_embed_ms=query_embed_ms,
        backend=backend,
    )


def _as_unit_f32(vectors: Any) -> np.ndarray:
    """Returns a contiguous float32 array of L2-normalized row vectors."""

    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return np.ascontiguousarray(arr / norms, dtype=np.float32)


# --------------------------------------------------------------------------- #
# brute-force ground truth + recall
# --------------------------------------------------------------------------- #


def exact_topk(
    ids: list[str],
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    *,
    allowed: np.ndarray | None = None,
) -> list[set[str]]:
    """Exact top-k id sets per query by brute-force cosine (dot on unit vectors).

    ``allowed`` is an optional boolean mask over documents (for filtered recall,
    e.g. the within-user subset); only those rows are eligible.
    """

    sims = query_vectors @ doc_vectors.T  # (Q, N) cosine on unit vectors
    if allowed is not None:
        sims = np.where(allowed[None, :], sims, -np.inf)
    truth: list[set[str]] = []
    eligible = int(allowed.sum()) if allowed is not None else len(ids)
    kk = min(k, max(1, eligible))
    for row in sims:
        top = np.argpartition(-row, kk - 1)[:kk] if kk < len(ids) else np.arange(len(ids))
        truth.append({ids[i] for i in top if np.isfinite(row[i])})
    return truth


def recall_at_k(returned: list[list[str]], truth: list[set[str]], k: int) -> float:
    """Mean recall@k of ``returned`` id lists against ``truth`` id sets."""

    if not truth:
        return 0.0
    scores: list[float] = []
    for got, want in zip(returned, truth, strict=True):
        if not want:
            continue
        hit = len(set(got[:k]) & want)
        scores.append(hit / min(k, len(want)))
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def backend_skeleton(name: str, role: str) -> dict[str, Any]:
    """A uniform per-backend result stub (``role`` is ``lodedb`` or ``baseline``)."""

    return {"backend": name, "role": role}


# --------------------------------------------------------------------------- #
# generic per-store workflow
# --------------------------------------------------------------------------- #


class StoreDriver:
    """Protocol every per-backend driver implements (duck-typed, not enforced).

    A driver adapts one vector store (the LodeDB adapter or a baseline) to a
    uniform set of operations so :func:`run_core_phases` can time them
    identically. Ingest takes both ``texts`` and precomputed ``vectors``; a
    text-path store (LodeDB's LangChain/LlamaIndex adapters) uses the text and
    re-embeds, a vector-path store uses the precomputed vectors. Query and
    incremental-add always use precomputed vectors, so neither phase is charged
    for query embedding and the numbers isolate the store.
    """

    name: str
    role: str
    # text-path stores embed during ingest; subtract the shared embed time to get
    # a store-only ingest figure comparable to the vector-path baselines.
    embeds_on_ingest: bool = False
    supports_reopen: bool = True
    # True => one durable add is O(changed); False => durability needs a full
    # O(corpus) rewrite (the in-memory dump-to-disk stores).
    incremental_is_delta: bool = True

    def warmup(self) -> None: ...  # one-time setup excluded from the ingest timer (model load)
    def ingest(self, ids, texts, vectors, metadatas) -> None: ...
    def query_one(self, qvec, k) -> list[str]: ...
    def persist(self) -> None: ...  # force a durability checkpoint (full dump for in-RAM stores)
    def footprint_bytes(self) -> int: ...
    def reopen(self) -> int: ...  # returns surviving document count, or -1 if unsupported
    def incremental_add(self, doc_id, text, vector, metadata) -> None: ...
    def close(self) -> None: ...


def run_core_phases(
    driver: StoreDriver,
    embedded: Embedded,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]],
    truth: list[set[str]],
    *,
    k: int,
    incremental_count: int,
    incremental_ids: list[str] | None = None,
    incremental_time_budget_s: float = 60.0,
    extra_phases: Any = None,
) -> dict[str, Any]:
    """Times ingest, query (+recall), footprint, reopen, and incremental adds.

    ``extra_phases``, if given, is called as ``extra_phases(driver, result)``
    after the query phase and before incremental/reopen (while the store is still
    open and full), to record framework-specific phases such as mem0's filtered
    search and update. ``incremental_ids`` overrides the ids used for the
    one-at-a-time durable adds (mem0/Qdrant need UUID ids).
    """

    result = backend_skeleton(driver.name, driver.role)
    n = len(ids)

    # --- ingest (bulk) ---
    vectors = [embedded.doc_vectors[i].tolist() for i in range(n)]
    driver.warmup()  # model load / handle open, excluded from the ingest timing
    t0 = time.perf_counter()
    driver.ingest(ids, texts, vectors, metadatas)
    ingest_ms = (time.perf_counter() - t0) * 1000.0
    store_only_ms = ingest_ms - (embedded.doc_embed_ms if driver.embeds_on_ingest else 0.0)
    store_only_ms = max(store_only_ms, 0.0)
    result["ingest"] = {
        "documents": n,
        "total_ms": round(ingest_ms, 2),
        "store_only_ms": round(store_only_ms, 2),
        "store_only_docs_per_s": round(n / (store_only_ms / 1000.0), 1) if store_only_ms else 0.0,
        "embeds_on_ingest": driver.embeds_on_ingest,
    }

    # --- query (per-query latency + recall@k) ---
    latencies: list[float] = []
    returned: list[list[str]] = []
    for row in embedded.query_vectors:
        qv = row.tolist()
        s = time.perf_counter()
        hits = driver.query_one(qv, k)
        latencies.append((time.perf_counter() - s) * 1000.0)
        returned.append(hits)
    result["query"] = {
        **latency_summary(latencies),
        "k": k,
        "recall_at_k": recall_at_k(returned, truth, k),
    }

    # --- persist (force durability) + footprint (durable, on disk) ---
    # In-RAM stores (InMemoryVectorStore, FAISS) only reach disk here; stores that
    # persist on every write (LodeDB, Chroma, Qdrant) treat this as a checkpoint.
    t0 = time.perf_counter()
    driver.persist()
    persist_ms = (time.perf_counter() - t0) * 1000.0
    result["persist"] = {"full_dump_ms": round(persist_ms, 2)}
    result["footprint_bytes"] = driver.footprint_bytes()

    # --- framework-specific phases (e.g. mem0 filtered search + update) ---
    if extra_phases is not None:
        extra_phases(driver, result)

    # --- incremental durable adds (agent-memory accrual: one memory at a time) ---
    # For a full-rewrite store (the in-memory defaults, FAISS) one durable add is
    # O(corpus): ~tens of seconds at 50k. The O(N)-vs-O(1) gap is the finding, so a
    # few samples settle the median; cap the wall-clock so a slow store does not
    # eat the run. Fast (delta) stores take the full sample count cheaply.
    inc_latencies: list[float] = []
    add_n = min(incremental_count, n)
    budget_ms = incremental_time_budget_s * 1000.0
    cumulative_ms = 0.0
    for j in range(add_n):
        doc_id = incremental_ids[j] if incremental_ids else f"inc{j}"
        vec = embedded.doc_vectors[j].tolist()
        s = time.perf_counter()
        driver.incremental_add(doc_id, texts[j], vec, metadatas[j])
        dt = (time.perf_counter() - s) * 1000.0
        inc_latencies.append(dt)
        cumulative_ms += dt
        if cumulative_ms > budget_ms and len(inc_latencies) >= _MIN_INCREMENTAL_SAMPLES:
            break
    added = len(inc_latencies)
    result["incremental_add"] = {
        **latency_summary(inc_latencies),
        "is_delta_persistence": driver.incremental_is_delta,
        "requested": add_n,
        "stopped_early": added < add_n,
        "footprint_bytes_after": driver.footprint_bytes(),
    }

    # --- reopen (durability across a process-like restart) ---
    if driver.supports_reopen:
        s = time.perf_counter()
        surviving = driver.reopen()
        reopen_ms = (time.perf_counter() - s) * 1000.0
        result["reopen"] = {
            "reopen_ms": round(reopen_ms, 2),
            "surviving_documents": surviving,
            "expected_documents": n + added,
            "durable": surviving >= n,  # adds may or may not be flushed by some stores
        }
    else:
        result["reopen"] = {"supported": False}

    driver.close()
    return result
