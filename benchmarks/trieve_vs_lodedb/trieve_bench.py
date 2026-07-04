"""Trieve side of the LodeDB-vs-Trieve retrieval benchmark (pure measurement).

Given a booted Trieve stack (base_url + dataset_id), this ingests the SAME
pre-chunked corpus the LodeDB side builds and runs the two comparison axes,
emitting a results JSON whose keys mirror ``lodedb_bench.py`` so the two can be
compared directly.

- Axis A (scale/latency): GovReport streamed + chunked (``chunk_text``,
  chunk_character_limit=360) to ~2M chunks, ingested via ``POST /api/chunk`` in
  batches of 120 (concurrent), polled until the ingestion queue drains, then
  semantic (dense) single-query p50/p95 and batched throughput. Per-phase latency
  comes from Trieve's ``Server-Timing`` header (dense-embed / qdrant / rerank),
  so the vector-store phase (Qdrant) is comparable to LodeDB's scan.
- Axis B (quality): MLDR English with real qrels. Each corpus chunk carries its
  source docid in both ``tracking_id`` and ``metadata.docid``; per query the top-k
  chunks map back to a deduped docid ranking, giving doc-level recall@{10,100} and
  nDCG@10 for Trieve ``semantic`` and Trieve ``hybrid`` (dense + SPLADE + rerank).

No Modal import here: the dataset builders reuse ``lodedb_bench`` (which reuses
``lodedb.engine.core.chunk_text``), and the HTTP client is stdlib. The
``Server-Timing`` parser and the docid/quality math are pure functions with a
synthetic ``self_test`` so they validate locally without a running Trieve.
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

# Mirror the LodeDB side's dataset ids + chunk size so the corpus is byte-identical.
GOVREPORT_DATASET = "ccdv/govreport-summarization"
MLDR_DATASET = "Shitao/MLDR"
DEFAULT_CHUNK_CHARACTER_LIMIT = 360
BATCH_CHUNK_LIMIT = 120  # Trieve's per-request array cap (env BATCH_CHUNK_LIMIT default)


# -- shared metric helpers (kept identical to lodedb_bench) ------------------


def _summary_ms(samples_ms: list[float]) -> dict[str, float]:
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


def _dcg_at_k(relevances: list[int], k: int) -> float:
    """Returns the discounted cumulative gain of a binary relevance list at k."""

    total = 0.0
    for rank, rel in enumerate(relevances[:k], start=1):
        if rel:
            total += 1.0 / math.log2(rank + 1)
    return total


def _ndcg_at_k(ranked_relevant: list[int], num_relevant: int, k: int) -> float:
    """Returns nDCG@k for a binary-relevance ranking against an ideal ranking."""

    if num_relevant <= 0:
        return 0.0
    ideal = _dcg_at_k([1] * num_relevant, k)
    if ideal == 0.0:
        return 0.0
    return _dcg_at_k(ranked_relevant, k) / ideal


# -- Server-Timing parsing --------------------------------------------------

# The simple_server_timing_header crate sanitizes every label with
# replace(|c| !c.is_alphanumeric(), "_"), so "fetched from qdrant" is emitted as
# "fetched_from_qdrant". Durations are per-segment integer milliseconds. We bucket
# the sanitized phase names into the three comparison phases.
_QDRANT_PHASES = frozenset(
    {"fetched_from_qdrant", "searched_within_qdrant", "fetching_from_qdrant"}
)
_EMBED_PHASES = frozenset(
    {
        "computed_dense_embedding",
        "computed_query_vector",
        "computed_sparse_and_dense_embeddings",
    }
)
_RERANK_PHASES = frozenset({"reranking"})


def parse_server_timing(header_value: str) -> dict[str, float]:
    """Parses a Server-Timing header into {phase_name: total_ms} plus bucketed sums.

    Splits on ``", "`` then ``";dur="`` exactly like Trieve's own parser, returning
    every phase by its sanitized name and additionally ``embed_ms`` / ``qdrant_ms`` /
    ``rerank_ms`` / ``total_ms`` roll-ups for the apples-to-apples comparison.
    """

    phases: dict[str, float] = {}
    if header_value:
        for entry in header_value.split(", "):
            parts = entry.split(";dur=")
            if len(parts) != 2:
                continue
            name = parts[0].strip()
            try:
                phases[name] = phases.get(name, 0.0) + float(parts[1])
            except ValueError:
                continue
    embed_ms = sum(value for name, value in phases.items() if name in _EMBED_PHASES)
    qdrant_ms = sum(value for name, value in phases.items() if name in _QDRANT_PHASES)
    rerank_ms = sum(value for name, value in phases.items() if name in _RERANK_PHASES)
    return {
        "phases": phases,
        "embed_ms": float(embed_ms),
        "qdrant_ms": float(qdrant_ms),
        "rerank_ms": float(rerank_ms),
        "total_ms": float(sum(phases.values())),
    }


# -- docid ranking + quality (mirrors lodedb_bench semantics) ---------------


def rank_docids(hits: list[dict[str, Any]]) -> list[str]:
    """Maps Trieve search hits to a deduped docid ranking, preserving hit order.

    Each hit is normalized to ``{docid, score}`` by ``_extract_hits``; the docid is
    read from tracking_id/metadata.docid. First-seen order is preserved so the ranking
    matches the LodeDB side's ``_rank_docids``.
    """

    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        docid = str(hit.get("docid", ""))
        if not docid or docid in seen:
            continue
        seen.add(docid)
        ranked.append(docid)
    return ranked


def quality_metrics(
    ranked_docids: list[str], relevant: set[str], *, recall_ks: tuple[int, ...], ndcg_k: int
) -> dict[str, float]:
    """Returns doc-level recall@k (each k) and nDCG@k for one query (LodeDB parity)."""

    metrics: dict[str, float] = {}
    for k in recall_ks:
        retrieved = set(ranked_docids[:k])
        hit_count = len(retrieved & relevant)
        metrics[f"recall@{k}"] = hit_count / len(relevant) if relevant else 0.0
    binary = [1 if docid in relevant else 0 for docid in ranked_docids[:ndcg_k]]
    metrics[f"ndcg@{ndcg_k}"] = _ndcg_at_k(binary, len(relevant), ndcg_k)
    return metrics


# -- Trieve HTTP client -----------------------------------------------------


class TrieveClient:
    """Thin stdlib HTTP client for the Trieve endpoints this benchmark uses.

    Auth is ``Authorization: Bearer <api_key>``. The dataset is scoped via the
    ``TR-Dataset`` header (a UUID, so no ``TR-Organization`` is needed for chunk /
    search / usage / queue calls). ``X-API-Version: V2`` pins the search response
    shape to ``{chunks: [{chunk, score}], total_pages}`` regardless of org age.
    """

    def __init__(self, base_url: str, dataset_id: str, api_key: str = "admin") -> None:
        self.base_url = base_url.rstrip("/")
        self.dataset_id = dataset_id
        self.api_key = api_key

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "TR-Dataset": self.dataset_id,
        }
        if extra:
            headers.update(extra)
        return headers

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: Any = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float = 120.0,
    ) -> tuple[int, dict[str, str], Any]:
        """Issues one request; returns (status, response_headers, parsed_json_or_none)."""

        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=self._headers(extra_headers),
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw) if raw else None
                # Lowercase header keys so lookups are case-insensitive (HTTP header
                # names are case-insensitive; a plain dict lookup is not).
                headers = {key.lower(): value for key, value in response.headers.items()}
                return response.status, headers, parsed
        except urllib.error.HTTPError as exc:
            # Read the server error body and re-raise a PICKLABLE error: the raw
            # HTTPError holds an unpicklable BufferedReader that breaks Modal's remote
            # exception propagation and hides the actual Trieve message.
            try:
                body = exc.read().decode("utf-8", "replace")
            except Exception:
                body = "<unreadable>"
            raise RuntimeError(
                f"{method} {request.full_url} -> HTTP {exc.code}: {body[:2000]}"
            ) from None

    def ingest_batch(self, chunks: list[dict[str, Any]]) -> int:
        """POSTs one batch of chunks; returns the HTTP status (200 = queued)."""

        status, _headers, _body = self._request("POST", "/api/chunk", payload=chunks, timeout=180.0)
        return status

    def queue_length(self) -> int:
        """Returns the current ingestion chunk_queue_length (0 = drained)."""

        _status, _headers, body = self._request(
            "GET", "/api/dataset/get_dataset_queue_lengths", timeout=30.0
        )
        if not isinstance(body, dict):
            return -1
        return int(body.get("chunk_queue_length", -1))

    def chunk_count(self) -> int:
        """Returns the dataset's committed chunk_count (Postgres-side usage counter)."""

        _status, _headers, body = self._request(
            "GET", f"/api/dataset/usage/{self.dataset_id}", timeout=30.0
        )
        if not isinstance(body, dict):
            return -1
        return int(body.get("chunk_count", -1))

    def search(
        self, query: str, *, search_type: str, page_size: int, timed: bool = True
    ) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
        """Runs one search; returns (normalized_hits, server_timing, end_to_end_ms).

        ``slim_chunks`` keeps the payload small while still returning tracking_id and
        metadata (only content_only drops metadata); highlighting is disabled to avoid
        its latency. Uses the V2 response shape.
        """

        payload = {
            "query": query,
            "search_type": search_type,
            "page_size": page_size,
            "slim_chunks": True,
            "highlight_options": {"highlight_results": False},
        }
        started = time.perf_counter()
        _status, headers, body = self._request(
            "POST", "/api/chunk/search", payload=payload,
            extra_headers={"X-API-Version": "V2"}, timeout=120.0,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        raw_timing = headers.get("server-timing", "")
        timing = parse_server_timing(raw_timing) if timed else {}
        if timing:
            timing["_raw_header"] = raw_timing
            timing["_header_keys"] = sorted(headers)  # debug: confirm what Trieve emits
        return _extract_hits(body), timing, elapsed_ms


def _extract_hits(body: Any) -> list[dict[str, Any]]:
    """Normalizes a V2 (or V1) search response into [{docid, tracking_id, score}].

    V2 hits live under ``chunks[].chunk`` with a sibling ``score``; V1 under
    ``score_chunks[].metadata[0]``. The docid is read from ``tracking_id`` first, then
    ``metadata.docid`` (we store it in both on ingest).
    """

    hits: list[dict[str, Any]] = []
    if not isinstance(body, dict):
        return hits
    raw_hits: list[Any] = []
    if isinstance(body.get("chunks"), list):  # V2
        for entry in body["chunks"]:
            chunk = entry.get("chunk") if isinstance(entry, dict) else None
            score = entry.get("score") if isinstance(entry, dict) else None
            if isinstance(chunk, dict):
                raw_hits.append((chunk, score))
    elif isinstance(body.get("score_chunks"), list):  # V1
        for entry in body["score_chunks"]:
            metadata_list = entry.get("metadata") if isinstance(entry, dict) else None
            score = entry.get("score") if isinstance(entry, dict) else None
            if isinstance(metadata_list, list) and metadata_list:
                raw_hits.append((metadata_list[0], score))
    for chunk, score in raw_hits:
        tracking_id = chunk.get("tracking_id")
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        # Recover the DOC-level id: metadata.docid is authoritative; tracking_id is the
        # per-chunk id "{docid}::c{n}", so fall back to its prefix (never the whole
        # tracking_id, which would never match the doc-level qrels).
        if metadata.get("docid"):
            docid = str(metadata["docid"])
        elif tracking_id:
            docid = str(tracking_id).split("::", 1)[0]
        else:
            docid = ""
        hits.append({"docid": docid, "tracking_id": tracking_id, "score": score})
    return hits


# -- ingest + poll ----------------------------------------------------------


def _chunk_payload(chunk_html: str, tracking_id: str, docid: str) -> dict[str, Any]:
    """Builds one Trieve chunk payload storing the docid in tracking_id + metadata.

    ``convert_html_to_text`` is set false because the corpus is already plain text
    (chunked by ``chunk_text``); leaving Trieve's HTML->text on would be a no-op for
    tag-free text but false avoids any entity-unescaping surprises so the embedded
    text byte-matches the LodeDB side.
    """

    return {
        "chunk_html": chunk_html,
        "tracking_id": tracking_id,
        "metadata": {"docid": docid},
        "convert_html_to_text": False,
        "upsert_by_tracking_id": True,
    }


def ingest_corpus(
    client: TrieveClient,
    payloads: list[dict[str, Any]],
    *,
    batch_size: int = BATCH_CHUNK_LIMIT,
    max_workers: int = 16,
) -> dict[str, Any]:
    """Ingests all chunk payloads in concurrent batches; returns ingest timings.

    Batches are capped at Trieve's BATCH_CHUNK_LIMIT (120) and submitted across a
    thread pool (the enqueue call is I/O bound). Returns wall seconds, request count,
    and any non-200 batch statuses.
    """

    batches = [
        payloads[start : start + batch_size] for start in range(0, len(payloads), batch_size)
    ]
    started = time.perf_counter()
    statuses: list[int] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(client.ingest_batch, batch) for batch in batches]
        for index, future in enumerate(concurrent.futures.as_completed(futures)):
            statuses.append(future.result())
            if (index + 1) % 200 == 0:
                print(
                    f"[trieve-bench] enqueued {index + 1}/{len(batches)} batches",
                    flush=True,
                )
    enqueue_seconds = time.perf_counter() - started
    bad = [status for status in statuses if status != 200]
    return {
        "enqueue_seconds": round(enqueue_seconds, 3),
        "batch_count": len(batches),
        "non_200_batches": len(bad),
        "enqueue_throughput_per_s": round(len(payloads) / enqueue_seconds, 1)
        if enqueue_seconds > 0
        else 0.0,
    }


def wait_until_indexed(
    client: TrieveClient, expected_count: int, *, timeout: float = 7200.0, interval: float = 5.0
) -> dict[str, Any]:
    """Polls queue length + chunk_count until the ingestion queue drains.

    Waits for ``chunk_queue_length == 0`` AND ``chunk_count >= expected_count`` (the
    queue draining plus the usage counter catching up means the chunks are committed to
    Postgres and upserted to Qdrant). Returns the drain wall seconds and final counts.
    """

    started = time.perf_counter()
    deadline = started + timeout
    last_count = -1
    while time.perf_counter() < deadline:  # monotonic clock (matches `started`/`deadline`)
        queue = client.queue_length()
        count = client.chunk_count()
        if count != last_count:
            print(
                f"[trieve-bench] indexing: chunk_count={count}/{expected_count} queue={queue}",
                flush=True,
            )
            last_count = count
        if queue == 0 and count >= expected_count:
            return {
                "index_wait_seconds": round(time.perf_counter() - started, 2),
                "final_chunk_count": count,
                "expected_count": expected_count,
            }
        time.sleep(interval)
    return {
        "index_wait_seconds": round(time.perf_counter() - started, 2),
        "final_chunk_count": client.chunk_count(),
        "expected_count": expected_count,
        "timed_out": True,
    }


# -- corpus builders (reuse the LodeDB side's chunking) ---------------------


def _govreport_payloads(max_corpus: int, n_query: int, chunk_character_limit: int) -> tuple[
    list[dict[str, Any]], list[str]
]:
    """Builds GovReport chunk payloads + query strings, identical to the LodeDB side."""

    import lodedb_bench

    corpus_texts, query_texts = lodedb_bench.load_govreport_chunks(
        max_corpus=max_corpus, n_query=n_query, chunk_character_limit=chunk_character_limit
    )
    payloads = [
        _chunk_payload(text, tracking_id=f"c{index}", docid=f"c{index}")
        for index, text in enumerate(corpus_texts)
    ]
    return payloads, query_texts


def _mldr_payloads(max_docs: int, chunk_character_limit: int) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    """Builds MLDR chunk payloads (docid in tracking_id+metadata) + qrel queries."""

    import lodedb_bench

    corpus_chunks, queries = lodedb_bench.load_mldr_en(
        max_docs=max_docs, chunk_character_limit=chunk_character_limit
    )
    payloads = [
        _chunk_payload(chunk["text"], tracking_id=chunk["chunk_id"], docid=chunk["docid"])
        for chunk in corpus_chunks
    ]
    return payloads, queries


# -- Axis A: scale / latency ------------------------------------------------


def run_axis_a_govreport(
    base_url: str,
    dataset_id: str,
    *,
    max_corpus: int,
    n_query: int = 1000,
    k: int = 10,
    chunk_character_limit: int = DEFAULT_CHUNK_CHARACTER_LIMIT,
    api_key: str = "admin",
    latency_iters: int = 1000,
    warmup: int = 50,
    batch_sizes: tuple[int, ...] = (1, 16, 64, 256),
    payloads: list[dict[str, Any]] | None = None,
    query_texts: list[str] | None = None,
) -> dict[str, Any]:
    """Ingests GovReport into Trieve then measures semantic latency + throughput.

    Mirrors ``lodedb_bench.run_axis_a_govreport`` keys: ``corpus_count``,
    ``ingest_seconds`` (queue-drain-inclusive), ``single_query_latency_ms``,
    ``batched_throughput``, plus a Trieve-specific ``server_timing`` roll-up (dense
    embed / qdrant / rerank) and ``index_wait``. No recall here (Trieve is the system
    under test, not an oracle); the quality axis handles retrieval quality.
    """

    client = TrieveClient(base_url, dataset_id, api_key=api_key)
    out: dict[str, Any] = {"axis": "A", "system": "trieve", "dataset": GOVREPORT_DATASET, "k": k}

    if payloads is None or query_texts is None:
        payloads, query_texts = _govreport_payloads(max_corpus, n_query, chunk_character_limit)
    out["corpus_count"] = len(payloads)
    out["chunk_character_limit"] = chunk_character_limit

    enqueue = ingest_corpus(client, payloads)
    indexed = wait_until_indexed(client, len(payloads))
    out["enqueue"] = enqueue
    out["index_wait"] = indexed
    # Total ingest time = enqueue + queue drain, comparable to LodeDB ingest+persist.
    out["ingest_seconds"] = round(
        enqueue["enqueue_seconds"] + indexed["index_wait_seconds"], 3
    )
    out["count_after_ingest"] = indexed["final_chunk_count"]
    out["ingest_throughput_per_s"] = (
        round(len(payloads) / out["ingest_seconds"], 1) if out["ingest_seconds"] > 0 else 0.0
    )

    if not query_texts:
        raise ValueError("axis A needs at least one query string")

    # Warm the semantic path (loads models + qdrant caches).
    for warm_index in range(min(warmup, len(query_texts))):
        client.search(query_texts[warm_index], search_type="semantic", page_size=k, timed=False)

    single_samples_ms: list[float] = []
    embed_ms: list[float] = []
    qdrant_ms: list[float] = []
    timing_sample: dict[str, Any] = {}
    for iteration in range(latency_iters):
        query = query_texts[iteration % len(query_texts)]
        _hits, timing, elapsed_ms = client.search(query, search_type="semantic", page_size=k)
        single_samples_ms.append(elapsed_ms)
        if timing:
            if not timing_sample and timing.get("phases"):
                timing_sample = timing
            embed_ms.append(timing["embed_ms"])
            qdrant_ms.append(timing["qdrant_ms"])
    out["single_query_latency_ms"] = _summary_ms(single_samples_ms)
    out["server_timing"] = {
        "embed_ms": _summary_ms(embed_ms),
        "qdrant_ms": _summary_ms(qdrant_ms),
        # Raw phase names + header from the first query so the true Trieve labels are
        # captured; lets us re-bucket in post if the frozenset labels ever drift.
        "phases_sample": timing_sample.get("phases", {}),
        "raw_header_sample": timing_sample.get("_raw_header", ""),
        "note": "per-phase from Trieve Server-Timing; qdrant_ms is the vector-store phase",
    }

    # Batched throughput: Trieve's search API is one-query-per-request, so we drive
    # concurrent requests and report achieved qps at each concurrency level.
    batch_results: list[dict[str, Any]] = []
    for concurrency in batch_sizes:
        queries = [query_texts[i % len(query_texts)] for i in range(concurrency)]
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(
                pool.map(
                    lambda q: client.search(q, search_type="semantic", page_size=k, timed=False),
                    queries,
                )
            )
        elapsed = time.perf_counter() - started
        batch_results.append(
            {
                "concurrency": concurrency,
                "qps": round(concurrency / elapsed, 1) if elapsed > 0 else 0.0,
                "per_query_ms": round((elapsed / concurrency) * 1000.0, 4) if concurrency else 0.0,
            }
        )
    out["batched_throughput"] = batch_results
    return out


# -- Axis B: MLDR-en quality ------------------------------------------------


def _score_mode(
    client: TrieveClient,
    queries: list[dict[str, Any]],
    *,
    search_type: str,
    k: int,
    recall_ks: tuple[int, ...],
    ndcg_k: int,
) -> dict[str, Any]:
    """Runs one Trieve search mode over all queries and averages the quality metrics.

    ``search_type="semantic"`` is dense-only; ``"hybrid"`` is dense + SPLADE + cross
    -encoder rerank. Returns the averaged recall/nDCG plus latency and the mean
    per-phase Server-Timing (so the rerank cost of hybrid is visible).
    """

    per_query: list[dict[str, float]] = []
    latencies_ms: list[float] = []
    embed_ms: list[float] = []
    qdrant_ms: list[float] = []
    rerank_ms: list[float] = []
    for query in queries:
        relevant = set(query["relevant_docids"])
        hits, timing, elapsed_ms = client.search(
            query["query"], search_type=search_type, page_size=k
        )
        latencies_ms.append(elapsed_ms)
        if timing:
            embed_ms.append(timing["embed_ms"])
            qdrant_ms.append(timing["qdrant_ms"])
            rerank_ms.append(timing["rerank_ms"])
        ranked = rank_docids(hits)
        per_query.append(quality_metrics(ranked, relevant, recall_ks=recall_ks, ndcg_k=ndcg_k))
    keys = sorted({key for row in per_query for key in row})
    averaged = {
        key: round(float(statistics.fmean([row[key] for row in per_query])), 4) for key in keys
    }
    return {
        "mode": search_type,
        "query_count": len(queries),
        "metrics": averaged,
        "query_latency_ms": _summary_ms(latencies_ms),
        "server_timing": {
            "embed_ms": _summary_ms(embed_ms),
            "qdrant_ms": _summary_ms(qdrant_ms),
            "rerank_ms": _summary_ms(rerank_ms),
        },
    }


def run_axis_b_mldr(
    base_url: str,
    dataset_id: str,
    *,
    max_docs: int,
    k: int = 100,
    recall_ks: tuple[int, ...] = (10, 100),
    ndcg_k: int = 10,
    chunk_character_limit: int = DEFAULT_CHUNK_CHARACTER_LIMIT,
    api_key: str = "admin",
    payloads: list[dict[str, Any]] | None = None,
    queries: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Ingests MLDR-en into Trieve then scores semantic + hybrid quality (LodeDB keys).

    Each chunk carries its docid in tracking_id + metadata, so top-k chunks map back to
    a deduped docid ranking. Mirrors ``lodedb_bench.run_axis_b_mldr``:
    ``corpus_chunk_count``, ``corpus_doc_count``, ``query_count``, and ``vector`` /
    ``hybrid`` blocks (here keyed by Trieve's ``semantic`` / ``hybrid`` search types).
    """

    client = TrieveClient(base_url, dataset_id, api_key=api_key)
    out: dict[str, Any] = {"axis": "B", "system": "trieve", "dataset": f"{MLDR_DATASET}/en", "k": k}

    if payloads is None or queries is None:
        payloads, queries = _mldr_payloads(max_docs, chunk_character_limit)
    out["corpus_chunk_count"] = len(payloads)
    out["corpus_doc_count"] = len({payload["metadata"]["docid"] for payload in payloads})
    out["query_count"] = len(queries)
    out["chunk_character_limit"] = chunk_character_limit

    enqueue = ingest_corpus(client, payloads)
    indexed = wait_until_indexed(client, len(payloads))
    out["enqueue"] = enqueue
    out["index_wait"] = indexed
    out["ingest_seconds"] = round(enqueue["enqueue_seconds"] + indexed["index_wait_seconds"], 3)
    out["count_after_ingest"] = indexed["final_chunk_count"]

    # "vector" mirrors the LodeDB dense key; "hybrid" is Trieve's dense+SPLADE+rerank.
    out["vector"] = _score_mode(
        client, queries, search_type="semantic", k=k, recall_ks=recall_ks, ndcg_k=ndcg_k
    )
    out["hybrid"] = _score_mode(
        client, queries, search_type="hybrid", k=k, recall_ks=recall_ks, ndcg_k=ndcg_k
    )
    return out


# -- local self-test (no Trieve, no Modal, no dataset) ----------------------


def _self_test() -> int:
    """Validates the Server-Timing parser, docid ranking, and quality math offline."""

    failures = 0

    def check(name: str, condition: bool) -> None:
        nonlocal failures
        if condition:
            print(f"[trieve-bench self-test] PASS {name}")
        else:
            failures += 1
            print(f"[trieve-bench self-test] FAIL {name}")

    # Server-Timing: sanitized names, per-segment ms, correct bucketing.
    header = (
        "start_correcting_query;dur=1, computed_sparse_and_dense_embeddings;dur=8, "
        "searched_within_qdrant;dur=5, fetched_metadata_from_postgres;dur=2, "
        "reranking;dur=12, search_chunks;dur=0"
    )
    timing = parse_server_timing(header)
    check("server_timing embed bucket", math.isclose(timing["embed_ms"], 8.0))
    check("server_timing qdrant bucket", math.isclose(timing["qdrant_ms"], 5.0))
    check("server_timing rerank bucket", math.isclose(timing["rerank_ms"], 12.0))
    check("server_timing total", math.isclose(timing["total_ms"], 28.0))
    check("server_timing empty", parse_server_timing("")["total_ms"] == 0.0)

    # Dense semantic header (no rerank): rerank bucket must be zero.
    dense_header = "computed_dense_embedding;dur=4, fetched_from_qdrant;dur=6"
    dense_timing = parse_server_timing(dense_header)
    check("dense embed", math.isclose(dense_timing["embed_ms"], 4.0))
    check("dense qdrant", math.isclose(dense_timing["qdrant_ms"], 6.0))
    check("dense no rerank", dense_timing["rerank_ms"] == 0.0)

    # V2 hit extraction: docid from tracking_id, then metadata.docid; dedup order.
    v2_body = {
        "chunks": [
            {"chunk": {"tracking_id": "D1::c0", "metadata": {"docid": "D1"}}, "score": 0.9},
            {"chunk": {"tracking_id": None, "metadata": {"docid": "D2"}}, "score": 0.5},
            {"chunk": {"tracking_id": "D1::c1", "metadata": {"docid": "D1"}}, "score": 0.4},
        ]
    }
    hits = _extract_hits(v2_body)
    check("v2 extract count", len(hits) == 3)
    check("v2 first docid via tracking", hits[0]["docid"] == "D1::c0")
    check("v2 metadata docid fallback", hits[1]["docid"] == "D2")

    # rank_docids should dedup on the docid field (set by the axis-B mapping).
    mapped = [{"docid": "D1"}, {"docid": "D2"}, {"docid": "D1"}, {"docid": "D3"}]
    check("rank_docids dedup order", rank_docids(mapped) == ["D1", "D2", "D3"])

    # V1 fallback extraction.
    v1_body = {
        "score_chunks": [
            {"metadata": [{"tracking_id": "X", "metadata": {"docid": "X"}}], "score": 1.0}
        ]
    }
    check("v1 extract", _extract_hits(v1_body)[0]["docid"] == "X")

    # Quality math: perfect ranking = recall 1.0 and nDCG 1.0.
    perfect = quality_metrics(["A", "B", "C"], {"A", "B"}, recall_ks=(2,), ndcg_k=2)
    check("recall perfect", math.isclose(perfect["recall@2"], 1.0))
    check("ndcg perfect", math.isclose(perfect["ndcg@2"], 1.0))
    miss = quality_metrics(["Z", "Y"], {"A"}, recall_ks=(2,), ndcg_k=2)
    check("recall miss", miss["recall@2"] == 0.0)

    # _chunk_payload stores docid in both places and disables html conversion.
    payload = _chunk_payload("hello", tracking_id="D1::c0", docid="D1")
    check("payload tracking", payload["tracking_id"] == "D1::c0")
    check("payload metadata docid", payload["metadata"]["docid"] == "D1")
    check("payload no html convert", payload["convert_html_to_text"] is False)

    print(f"[trieve-bench self-test] {'OK' if not failures else 'FAILURES: ' + str(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
