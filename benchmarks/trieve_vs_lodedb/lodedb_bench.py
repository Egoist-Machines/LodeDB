"""LodeDB side of the LodeDB-vs-Trieve retrieval benchmark (pure measurement).

Two measurement axes over a shared dense model (all-MiniLM-L6-v2, 384-d):

- Axis A (scale/latency/footprint): GovReport streamed + chunked to ~2M chunks,
  ingested vector-in, then build throughput, on-disk footprint (total +
  per-extension), peak process RSS, single-query p50/p95, batched qps across
  batch sizes, and index-fidelity recall@10 vs an fp32 brute-force top-10.
- Axis B (quality): MLDR English with real qrels. Corpus docs are chunked and
  ingested with a ``docid`` in metadata; per query the top-k chunks map back to
  docids, so doc-level recall@{10,100} and nDCG@10 are computed for both LodeDB
  vector (search_by_vector) and LodeDB hybrid (search mode="hybrid").

No Modal import here: every function runs locally and returns JSON-able dicts,
so signatures can be validated with the tiny synthetic ``__main__`` smoke without
Modal or any dataset download. Metrics-only and payload-free (counts, bytes,
latencies, recall/nDCG), matching the repo's benchmark provenance rules.
"""

from __future__ import annotations

import math
import resource
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GOVREPORT_DATASET = "ccdv/govreport-summarization"
MLDR_DATASET = "Shitao/MLDR"
# MiniLM chunk size: the repo notes 480 chars yield ~1.1M GovReport chunks, so
# 360 chars (~90 tokens) reaches ~2M before the max_corpus cap.
DEFAULT_CHUNK_CHARACTER_LIMIT = 360
MINILM_DIM = 384


# -- shared helpers ---------------------------------------------------------


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


def _peak_rss_bytes() -> int:
    """Returns peak process RSS in bytes (getrusage; Linux reports KiB, macOS bytes)."""

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is KiB on Linux and bytes on macOS; normalize to bytes.
    return int(peak) if sys.platform == "darwin" else int(peak) * 1024


def _dir_footprint(path: Path) -> dict[str, Any]:
    """Returns total on-disk bytes under ``path`` plus a per-extension breakdown."""

    total = 0
    by_ext: dict[str, int] = {}
    file_count = 0
    for file in Path(path).rglob("*"):
        if not file.is_file():
            continue
        size = file.stat().st_size
        total += size
        file_count += 1
        ext = file.suffix or "<none>"
        by_ext[ext] = by_ext.get(ext, 0) + size
    return {
        "total_bytes": int(total),
        "file_count": int(file_count),
        "by_extension_bytes": {ext: int(size) for ext, size in sorted(by_ext.items())},
    }


def _l2_normalize(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Returns a row-wise L2-normalized copy (zero rows left unchanged)."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


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


# -- embedding --------------------------------------------------------------


def _load_embedder(model_name: str = DEFAULT_EMBED_MODEL, *, device: str = "cuda") -> Any:
    """Loads the shared sentence-transformers model (import kept lazy)."""

    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


def _encode_blocks(
    model: Any,
    texts: list[str],
    *,
    batch_size: int = 512,
    block: int = 250_000,
    label: str = "corpus",
) -> NDArray[np.float32]:
    """Encodes texts in blocks (cosine-normalized fp32), printing progress.

    Mirrors turbovec_govreport_scale._encode_blocks: same batching,
    normalize_embeddings=True, and fp32 contiguous output.
    """

    if not texts:
        return np.zeros((0, int(model.get_sentence_embedding_dimension())), dtype=np.float32)
    parts: list[NDArray[np.float32]] = []
    for start in range(0, len(texts), block):
        emb = model.encode(
            texts[start : start + block],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        parts.append(np.asarray(emb, dtype=np.float32))
        print(
            f"[trieve-vs-lodedb] embedded {label} {min(start + block, len(texts))}/{len(texts)}",
            flush=True,
        )
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


# -- dataset loading --------------------------------------------------------


def load_govreport_chunks(
    *, max_corpus: int, n_query: int, chunk_character_limit: int
) -> tuple[list[str], list[str]]:
    """Streams GovReport and chunks each report into corpus + query (summary) texts.

    Reuses LodeDB's ``chunk_text`` and streams ``ccdv/govreport-summarization``
    over train/validation/test until the corpus reaches ``max_corpus`` chunks.
    Returns (corpus_chunk_texts, query_summary_texts).
    """

    from datasets import load_dataset

    from lodedb.engine.core import chunk_text

    corpus_texts: list[str] = []
    query_texts: list[str] = []
    for split in ("train", "validation", "test"):
        if len(corpus_texts) >= max_corpus:
            break
        rows = load_dataset(GOVREPORT_DATASET, split=split, streaming=True)
        for row in rows:
            if len(corpus_texts) >= max_corpus:
                break
            report = str(row.get("report", "")).strip()
            summary = str(row.get("summary", "")).strip()
            if not report or not summary:
                continue
            chunks = chunk_text(report, chunk_character_limit)
            if not chunks:
                continue
            for body in chunks:
                if len(corpus_texts) >= max_corpus:
                    break
                corpus_texts.append(body)
            if len(query_texts) < n_query:
                query_texts.append(summary)
    return corpus_texts, query_texts


def load_mldr_en(
    *, max_docs: int, chunk_character_limit: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Loads MLDR English corpus chunks + queries with qrels.

    Queries: ``load_dataset("Shitao/MLDR","en",split="test")`` (query_id, query,
    positive_passages[{docid,text}]); a query's relevant docids are its positive
    passages. To keep recall FAIR, every relevant doc is ALWAYS indexed -- its full
    text ships inside the positive passage, so no corpus scan is needed to find it --
    and the streamed ``corpus-en`` split then supplies distractor docs up to
    ``max_docs`` total (relevant + distractors). ``max_docs <= 0`` streams the whole
    corpus as distractors (expensive: ~200k docs). Each doc is chunked with
    ``chunk_text``; a corpus chunk is ``{"chunk_id","docid","text"}`` and a query is
    ``{"query_id","query","relevant_docids"}``.
    """

    from datasets import load_dataset

    from lodedb.engine.core import chunk_text

    # 1) Queries + qrels first; capture each relevant doc's text from its positive
    # passage so it can be indexed directly (guarantees the doc is retrievable).
    queries: list[dict[str, Any]] = []
    relevant_text: dict[str, str] = {}
    query_rows = load_dataset(MLDR_DATASET, "en", split="test", trust_remote_code=True)
    for row in query_rows:
        query_id = str(row.get("query_id", "")).strip()
        query = str(row.get("query", "")).strip()
        if not query_id or not query:
            continue
        relevant: set[str] = set()
        for passage in row.get("positive_passages") or []:
            docid = str(passage.get("docid", "")).strip()
            text = str(passage.get("text", "")).strip()
            if not docid:
                continue
            relevant.add(docid)
            if text and docid not in relevant_text:
                relevant_text[docid] = text
        if not relevant:
            continue
        queries.append(
            {"query_id": query_id, "query": query, "relevant_docids": sorted(relevant)}
        )

    corpus_chunks: list[dict[str, Any]] = []
    kept_docids: set[str] = set()

    def _add_doc(docid: str, text: str) -> None:
        chunks = chunk_text(text, chunk_character_limit)
        if not chunks:
            return
        kept_docids.add(docid)
        for index, body in enumerate(chunks):
            corpus_chunks.append(
                {"chunk_id": f"{docid}::c{index}", "docid": docid, "text": body}
            )

    # 2) Index relevant docs (fair recall). A smoke bounds the corpus via max_docs, so
    # index at most max_docs relevant docs (relevant-first) and keep only queries whose
    # relevant docs are all indexed; the full run (max_docs >> #relevant, or 0) keeps all.
    all_relevant = list(relevant_text)
    cap = max_docs if max_docs > 0 else len(all_relevant)
    indexed_relevant = set(all_relevant[:cap])
    for docid in all_relevant:
        if docid in indexed_relevant:
            _add_doc(docid, relevant_text[docid])
    queries = [q for q in queries if set(q["relevant_docids"]) <= indexed_relevant]

    # 3) Fill with distractor docs from the streamed corpus (skip already-indexed),
    # up to the total-doc cap (relevant + distractors).
    distractor_budget = None if max_docs <= 0 else max(0, max_docs - len(kept_docids))
    if distractor_budget is None or distractor_budget > 0:
        added = 0
        corpus_rows = load_dataset(
            MLDR_DATASET, "corpus-en", split="corpus", streaming=True, trust_remote_code=True
        )
        for row in corpus_rows:
            if distractor_budget is not None and added >= distractor_budget:
                break
            docid = str(row.get("docid", "")).strip()
            text = str(row.get("text", "")).strip()
            if not docid or not text or docid in kept_docids:
                continue
            _add_doc(docid, text)
            added += 1

    return corpus_chunks, queries


# -- Axis A: scale / latency / footprint ------------------------------------


def _ingest_vectors(
    db: Any,
    ids: list[str],
    vectors: NDArray[np.float32],
    texts: list[str],
    metadatas: list[dict[str, str]] | None,
    *,
    batch: int,
) -> float:
    """Ingests precomputed vectors via add_vectors_many; returns wall seconds.

    Vectors are already unit-norm (encoded with normalize_embeddings=True), so
    normalize=True here is a cheap no-op that keeps cosine scores comparable with
    the text/hybrid path. ``text`` is retained so hybrid BM25 has a lexical source.
    """

    started = time.perf_counter()
    for start in range(0, len(ids), batch):
        stop = min(start + batch, len(ids))
        payload: list[dict[str, Any]] = []
        for index in range(start, stop):
            item: dict[str, Any] = {
                "id": ids[index],
                "vector": vectors[index].tolist(),
                "text": texts[index],
            }
            if metadatas is not None:
                item["metadata"] = metadatas[index]
            payload.append(item)
        db.add_vectors_many(payload, normalize=True)
    return time.perf_counter() - started


def _brute_force_recall_at_k(
    corpus: NDArray[np.float32],
    query_vectors: NDArray[np.float32],
    served_ids: list[list[str]],
    corpus_ids: list[str],
    *,
    k: int,
) -> float:
    """Returns index-fidelity recall@k of the served ids vs fp32 brute-force top-k.

    Computes the exact cosine top-k per query with numpy against the full corpus
    matrix (both sides L2-normalized, so a dot product is cosine) and measures the
    average overlap fraction with the ids LodeDB returned. This is index fidelity,
    not retrieval quality: the ground truth is the exact scan over the same vectors.
    """

    if not served_ids:
        return 0.0
    id_by_row = np.asarray(corpus_ids, dtype=object)
    normalized_corpus = _l2_normalize(corpus)
    normalized_queries = _l2_normalize(query_vectors)
    overlap_sum = 0.0
    counted = 0
    for query_index in range(normalized_queries.shape[0]):
        scores = normalized_corpus @ normalized_queries[query_index]
        top = np.argpartition(-scores, min(k, scores.shape[0] - 1))[:k]
        truth = set(id_by_row[top].tolist())
        served = set(served_ids[query_index][:k])
        if truth:
            overlap_sum += len(truth & served) / len(truth)
            counted += 1
    return overlap_sum / counted if counted else 0.0


def _measure_ann_config(
    ann_path: str,
    config: dict[str, Any],
    *,
    corpus_ids: list[str],
    corpus_vectors: NDArray[np.float32],
    corpus_texts: list[str],
    query_vectors: NDArray[np.float32],
    k: int,
    ingest_batch: int,
    latency_iters: int,
    warmup: int,
    recall_sample: int,
) -> dict[str, Any]:
    """Builds one ANN (cluster-prune) store over the already-embedded corpus and
    measures its single-query latency, index-fidelity recall, and footprint.

    ANN is a create-time choice in LodeDB (the ``clusters``/``nprobe`` tuning is
    persisted with the store), so every config is a fresh store built from the same
    precomputed vectors rather than a reopen of the exact store. Returned scores stay
    exact -- the exact TurboVec scan re-scores the probed candidates -- so the only
    approximation is set membership: a true neighbor in an unprobed cluster is missed,
    which is the recall/latency trade this measures against the exact scan. Recall uses
    the *same* fp32 brute-force ground truth as the exact row, so the two are directly
    comparable and the drop is the cluster-pruning cost (on top of 4-bit quantization).
    """

    from lodedb.local.db import LodeDB

    algorithm = str(config.get("algorithm", "cluster"))
    clusters = config.get("clusters")
    nprobe = config.get("nprobe")
    result: dict[str, Any] = {
        "label": str(config.get("label", algorithm)),
        "algorithm": algorithm,
        "clusters": clusters,
        "nprobe": nprobe,
    }
    # store_text=True / index_text=False mirrors the exact store (vector path only, no
    # unused lexical index), so the footprint is directly comparable and the only structural
    # difference is ANN's own metadata. The cluster-prune partition is built in memory on the
    # first query (a warmup cost, not charged to the timed loop).
    db = LodeDB(
        ann_path,
        model="minilm",
        store_text=True,
        index_text=False,
        ann=algorithm,
        ann_clusters=clusters,
        ann_nprobe=nprobe,
    )
    try:
        ingest_seconds = _ingest_vectors(
            db, corpus_ids, corpus_vectors, corpus_texts, None, batch=ingest_batch
        )
        persist_started = time.perf_counter()
        db.persist()
        result["ingest_seconds"] = round(ingest_seconds, 3)
        result["persist_seconds"] = round(time.perf_counter() - persist_started, 3)
        result["footprint"] = _dir_footprint(Path(ann_path))

        query_list = [row.tolist() for row in query_vectors]
        if not query_list:
            raise ValueError("ANN measurement needs at least one held-out query vector")
        # The first query builds the k-means cluster index. That build is O(n * clusters * dim)
        # and dominates ANN setup at scale (with the default ~sqrt(n) clusters it is impractical
        # past ~1M vectors), so time it explicitly and print around it -- otherwise the run looks
        # hung during the build. Charged as cluster_build_seconds, not to the query latency below.
        print(
            f"[trieve-vs-lodedb] ANN {result['label']}: building cluster index over "
            f"{len(corpus_ids)} vectors (clusters={clusters}, first-query k-means)...",
            flush=True,
        )
        build_started = time.perf_counter()
        db.search_by_vector(query_list[0], k=k)
        result["cluster_build_seconds"] = round(time.perf_counter() - build_started, 2)
        print(
            f"[trieve-vs-lodedb] ANN {result['label']}: cluster index built in "
            f"{result['cluster_build_seconds']}s",
            flush=True,
        )
        for warm_index in range(min(warmup, len(query_list))):
            db.search_by_vector(query_list[warm_index], k=k)
        samples_ms: list[float] = []
        for iteration in range(latency_iters):
            vector = query_list[iteration % len(query_list)]
            started = time.perf_counter()
            db.search_by_vector(vector, k=k)
            samples_ms.append((time.perf_counter() - started) * 1000.0)
        result["single_query_latency_ms"] = _summary_ms(samples_ms)

        sample = min(recall_sample, len(query_list))
        sample_vectors = query_vectors[:sample]
        served_ids = [
            [hit.id for hit in db.search_by_vector(sample_vectors[index].tolist(), k=k)]
            for index in range(sample)
        ]
        result["index_recall_at_k"] = {
            "k": k,
            "query_sample": sample,
            "recall": round(
                _brute_force_recall_at_k(
                    corpus_vectors, sample_vectors, served_ids, corpus_ids, k=k
                ),
                4,
            ),
            "note": "ANN result set vs fp32 brute-force top-k (same metric as the exact row)",
        }
    finally:
        db.close()
    return result


def run_axis_a_govreport(
    db_path: str,
    *,
    max_corpus: int,
    n_query: int = 1000,
    k: int = 10,
    chunk_character_limit: int = DEFAULT_CHUNK_CHARACTER_LIMIT,
    ingest_batch: int = 4096,
    batch_sizes: tuple[int, ...] = (1, 16, 64, 256),
    latency_iters: int = 1000,
    warmup: int = 50,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str = "cuda",
    embed_batch_size: int = 512,
    recall_sample: int = 1000,
    ann_configs: tuple[dict[str, Any], ...] = (),
    corpus_vectors: NDArray[np.float32] | None = None,
    corpus_texts: list[str] | None = None,
    query_vectors: NDArray[np.float32] | None = None,
) -> dict[str, Any]:
    """Runs the GovReport scale/latency/footprint axis and returns a JSON-able dict.

    Held-out query vectors drive both the latency and recall measurements; the
    corpus/query embeddings may be passed in (for the local smoke) or embedded here
    from streamed GovReport text. ``recall_sample`` bounds the brute-force ground
    truth to a query subsample (the full corpus matrix is scanned per query).

    ``ann_configs`` opts into the approximate-search comparison: each entry (a dict of
    optional ``label``/``clusters``/``nprobe``) builds a separate ANN cluster-prune store
    over the same precomputed vectors and reports its single-query latency and index
    recall under ``out["ann"]``, alongside the exact scan. Empty by default (exact only).
    """

    from lodedb.local.db import LodeDB

    out: dict[str, Any] = {"axis": "A", "dataset": GOVREPORT_DATASET, "k": k}

    if corpus_vectors is None or corpus_texts is None or query_vectors is None:
        corpus_texts, query_texts = load_govreport_chunks(
            max_corpus=max_corpus, n_query=n_query, chunk_character_limit=chunk_character_limit
        )
        model = _load_embedder(model_name, device=device)
        embed_started = time.perf_counter()
        corpus_vectors = _encode_blocks(
            model, corpus_texts, batch_size=embed_batch_size, label="corpus"
        )
        query_vectors = _encode_blocks(
            model, query_texts, batch_size=embed_batch_size, label="query"
        )
        out["embed_seconds"] = round(time.perf_counter() - embed_started, 2)

    corpus_count = int(corpus_vectors.shape[0])
    dim = int(corpus_vectors.shape[1])
    out["corpus_count"] = corpus_count
    out["dim"] = dim
    out["chunk_character_limit"] = chunk_character_limit
    corpus_ids = [f"c{index}" for index in range(corpus_count)]

    # index_text=False: axis A is the vector-scale/latency/footprint path and never runs
    # hybrid/lexical search, so skip the durable BM25 lexical index. Since v1.2.0 index_text
    # defaults to store_text (True here), leaving it unset would build and persist an unused
    # ~100 MB .tvlex sidecar, inflating ingest, persist, and footprint and making the vector
    # comparison unfair. store_text stays True to retain the text payload (footprint parity).
    db = LodeDB(db_path, model="minilm", store_text=True, index_text=False)
    try:
        ingest_seconds = _ingest_vectors(
            db, corpus_ids, corpus_vectors, corpus_texts, None, batch=ingest_batch
        )
        persist_started = time.perf_counter()
        db.persist()
        persist_seconds = time.perf_counter() - persist_started
        out["count_after_ingest"] = int(db.count())
        out["ingest_seconds"] = round(ingest_seconds, 3)
        out["persist_seconds"] = round(persist_seconds, 3)
        out["ingest_throughput_per_s"] = (
            round(corpus_count / ingest_seconds, 1) if ingest_seconds > 0 else 0.0
        )
        out["footprint"] = _dir_footprint(Path(db_path))
        out["peak_rss_bytes"] = _peak_rss_bytes()
        print(
            f"[trieve-vs-lodedb] axis A exact: ingested {out['count_after_ingest']} "
            f"({out['ingest_throughput_per_s']}/s), persisted in {out['persist_seconds']}s; "
            f"measuring latency/recall...",
            flush=True,
        )

        query_list = [row.tolist() for row in query_vectors]
        if not query_list:
            raise ValueError("axis A needs at least one held-out query vector")

        # (4) single-query latency p50/p95 over the held-out query vectors (warm up first).
        for warm_index in range(min(warmup, len(query_list))):
            db.search_by_vector(query_list[warm_index], k=k)
        single_samples_ms: list[float] = []
        for iteration in range(latency_iters):
            vector = query_list[iteration % len(query_list)]
            started = time.perf_counter()
            db.search_by_vector(vector, k=k)
            single_samples_ms.append((time.perf_counter() - started) * 1000.0)
        out["single_query_latency_ms"] = _summary_ms(single_samples_ms)

        # (5) batched throughput qps via search_many_by_vector at each batch size.
        batch_results: list[dict[str, Any]] = []
        for batch_size in batch_sizes:
            batch = query_list[:batch_size]
            if len(batch) < batch_size:
                batch = (batch * (batch_size // max(1, len(batch)) + 1))[:batch_size]
            db.search_many_by_vector(batch, k=k)  # warm the batched path
            repeats = max(1, min(20, latency_iters // max(1, batch_size)))
            started = time.perf_counter()
            for _ in range(repeats):
                db.search_many_by_vector(batch, k=k)
            elapsed = time.perf_counter() - started
            served = batch_size * repeats
            batch_results.append(
                {
                    "batch_size": batch_size,
                    "repeats": repeats,
                    "qps": round(served / elapsed, 1) if elapsed > 0 else 0.0,
                    "per_query_ms": round((elapsed / served) * 1000.0, 4) if served else 0.0,
                }
            )
        out["batched_throughput"] = batch_results

        # (6) index-fidelity recall@k vs fp32 brute force on a query subsample.
        sample = min(recall_sample, len(query_list))
        sample_vectors = query_vectors[:sample]
        served_ids = [
            [hit.id for hit in db.search_by_vector(sample_vectors[index].tolist(), k=k)]
            for index in range(sample)
        ]
        out["index_recall_at_k"] = {
            "k": k,
            "query_sample": sample,
            "recall": round(
                _brute_force_recall_at_k(
                    corpus_vectors, sample_vectors, served_ids, corpus_ids, k=k
                ),
                4,
            ),
            "note": "index fidelity vs fp32 brute-force top-k over the same vectors",
        }

        # Checkpoint the exact-scan result to stdout before the (potentially very slow) ANN
        # cluster build, so the essential baseline survives even if the ANN phase times out or
        # the client disconnects (recoverable from Modal's worker logs). See issue #71: the
        # k-means build is impractical at 2M, so the exact baseline must not depend on it.
        import json as _json

        print(f"LODEDB_RESULT axis_a_exact {_json.dumps(out, sort_keys=True)}", flush=True)

        # (7) opt-in ANN (cluster-prune) latency/recall trade vs the exact scan above.
        # Each config is a separate store (ANN tuning is create-time), built from the
        # same precomputed vectors, so this isolates the approximate-search win at scale:
        # the exact single-query latency grows with the corpus, ANN stays sub-linear.
        if ann_configs:
            out["ann"] = [
                _measure_ann_config(
                    f"{db_path}-ann{index}",
                    config,
                    corpus_ids=corpus_ids,
                    corpus_vectors=corpus_vectors,
                    corpus_texts=corpus_texts,
                    query_vectors=query_vectors,
                    k=k,
                    ingest_batch=ingest_batch,
                    latency_iters=latency_iters,
                    warmup=warmup,
                    recall_sample=recall_sample,
                )
                for index, config in enumerate(ann_configs)
            ]
    finally:
        db.close()
    return out


# -- Axis B: MLDR-en quality ------------------------------------------------


def _rank_docids(hits: list[Any]) -> list[str]:
    """Maps chunk hits to a deduped docid ranking, preserving first-seen order."""

    ranked: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        docid = str(hit.metadata.get("docid", "")) if hit.metadata else ""
        if not docid or docid in seen:
            continue
        seen.add(docid)
        ranked.append(docid)
    return ranked


def _quality_metrics(
    ranked_docids: list[str], relevant: set[str], *, recall_ks: tuple[int, ...], ndcg_k: int
) -> dict[str, float]:
    """Returns doc-level recall@k (for each k) and nDCG@k for one query."""

    metrics: dict[str, float] = {}
    for k in recall_ks:
        retrieved = set(ranked_docids[:k])
        hit_count = len(retrieved & relevant)
        metrics[f"recall@{k}"] = hit_count / len(relevant) if relevant else 0.0
    binary = [1 if docid in relevant else 0 for docid in ranked_docids[:ndcg_k]]
    metrics[f"ndcg@{ndcg_k}"] = _ndcg_at_k(binary, len(relevant), ndcg_k)
    return metrics


def _score_mode(
    db: Any,
    queries: list[dict[str, Any]],
    query_vectors: NDArray[np.float32],
    *,
    mode: str,
    k: int,
    recall_ks: tuple[int, ...],
    ndcg_k: int,
) -> dict[str, Any]:
    """Runs one retrieval mode over all queries and averages the quality metrics.

    ``mode="vector"`` uses search_by_vector with the precomputed MiniLM query
    vector; ``mode="hybrid"``/``"lexical"`` use the text path
    search(query, mode=mode), which embeds the query with MiniLM internally and
    (hybrid) fuses BM25 over stored text with RRF, or (lexical) runs BM25 alone.
    """

    # Warm up so the one-time lexical index build (first hybrid/lexical query after
    # open) is not charged to the measured latencies.
    if queries:
        if mode == "vector":
            db.search_by_vector(query_vectors[0].tolist(), k=k)
        else:
            db.search(queries[0]["query"], k=k, mode=mode)

    per_query: list[dict[str, float]] = []
    latencies_ms: list[float] = []
    for index, query in enumerate(queries):
        relevant = set(query["relevant_docids"])
        started = time.perf_counter()
        if mode == "vector":
            hits = db.search_by_vector(query_vectors[index].tolist(), k=k)
        else:
            hits = db.search(query["query"], k=k, mode=mode)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        ranked = _rank_docids(hits)
        per_query.append(
            _quality_metrics(ranked, relevant, recall_ks=recall_ks, ndcg_k=ndcg_k)
        )
    keys = sorted({key for row in per_query for key in row})
    averaged = {
        key: round(float(statistics.fmean([row[key] for row in per_query])), 4)
        for key in keys
    }
    return {
        "mode": mode,
        "query_count": len(queries),
        "metrics": averaged,
        "query_latency_ms": _summary_ms(latencies_ms),
    }


def run_axis_b_mldr(
    db_path: str,
    *,
    max_docs: int,
    k: int = 100,
    recall_ks: tuple[int, ...] = (10, 100),
    ndcg_k: int = 10,
    chunk_character_limit: int = DEFAULT_CHUNK_CHARACTER_LIMIT,
    ingest_batch: int = 4096,
    model_name: str = DEFAULT_EMBED_MODEL,
    device: str = "cuda",
    embed_batch_size: int = 512,
    corpus_chunks: list[dict[str, Any]] | None = None,
    queries: list[dict[str, Any]] | None = None,
    corpus_vectors: NDArray[np.float32] | None = None,
    query_vectors: NDArray[np.float32] | None = None,
) -> dict[str, Any]:
    """Runs the MLDR-en quality axis for both vector and hybrid; returns a dict.

    Chunks + qrels may be passed in (local smoke) or loaded and embedded here. Each
    corpus chunk is ingested with ``metadata={"docid": docid}`` and its text, so a
    top-k chunk retrieval maps back to a deduped docid ranking; a doc is relevant if
    its docid is in that query's positive-passage docids.
    """

    from lodedb.local.db import LodeDB

    out: dict[str, Any] = {"axis": "B", "dataset": f"{MLDR_DATASET}/en", "k": k}

    if corpus_chunks is None or queries is None:
        corpus_chunks, queries = load_mldr_en(
            max_docs=max_docs, chunk_character_limit=chunk_character_limit
        )
    if corpus_vectors is None or query_vectors is None:
        model = _load_embedder(model_name, device=device)
        embed_started = time.perf_counter()
        corpus_vectors = _encode_blocks(
            model,
            [chunk["text"] for chunk in corpus_chunks],
            batch_size=embed_batch_size,
            label="mldr-corpus",
        )
        query_vectors = _encode_blocks(
            model,
            [query["query"] for query in queries],
            batch_size=embed_batch_size,
            label="mldr-query",
        )
        out["embed_seconds"] = round(time.perf_counter() - embed_started, 2)

    out["corpus_chunk_count"] = len(corpus_chunks)
    out["corpus_doc_count"] = len({chunk["docid"] for chunk in corpus_chunks})
    out["query_count"] = len(queries)
    out["chunk_character_limit"] = chunk_character_limit

    # index_text=True captures per-chunk lexical terms at add time (durable .tvlex
    # postings), which is what populates BM25 for vector-in docs; store_text alone
    # rebuilds BM25 only from text-path adds, leaving vector-in hybrid BM25 empty.
    db = LodeDB(db_path, model="minilm", store_text=True, index_text=True)
    try:
        ids = [chunk["chunk_id"] for chunk in corpus_chunks]
        texts = [chunk["text"] for chunk in corpus_chunks]
        metadatas = [{"docid": chunk["docid"]} for chunk in corpus_chunks]
        ingest_seconds = _ingest_vectors(
            db, ids, corpus_vectors, texts, metadatas, batch=ingest_batch
        )
        db.persist()
        out["ingest_seconds"] = round(ingest_seconds, 3)
        out["count_after_ingest"] = int(db.count())
        out["footprint"] = _dir_footprint(Path(db_path))

        out["vector"] = _score_mode(
            db, queries, query_vectors, mode="vector", k=k, recall_ks=recall_ks, ndcg_k=ndcg_k
        )
        out["hybrid"] = _score_mode(
            db, queries, query_vectors, mode="hybrid", k=k, recall_ks=recall_ks, ndcg_k=ndcg_k
        )
        # BM25-only diagnostic: reveals whether the lexical index is populated
        # (near-zero recall here means BM25 has no content, so "hybrid" would just
        # be vector; substantial recall means BM25 works and agrees with vector).
        out["lexical"] = _score_mode(
            db, queries, query_vectors, mode="lexical", k=k, recall_ks=recall_ks, ndcg_k=ndcg_k
        )
    finally:
        db.close()
    return out


# -- local smoke ------------------------------------------------------------


def _synthetic_axis_a(db_path: str) -> dict[str, Any]:
    """Axis A smoke over a synthetic 200-chunk corpus + 5 held-out queries."""

    rng = np.random.default_rng(0)
    corpus_count = 200
    corpus_vectors = _l2_normalize(
        rng.standard_normal((corpus_count, MINILM_DIM)).astype(np.float32)
    )
    corpus_texts = [
        f"synthetic government report chunk number {index}" for index in range(corpus_count)
    ]
    # Held-out queries are near a few corpus rows so recall is meaningfully nonzero.
    query_vectors = _l2_normalize(
        corpus_vectors[:5] + 0.05 * rng.standard_normal((5, MINILM_DIM)).astype(np.float32)
    )
    return run_axis_a_govreport(
        db_path,
        max_corpus=corpus_count,
        n_query=5,
        k=10,
        batch_sizes=(1, 16, 64),
        latency_iters=50,
        warmup=5,
        recall_sample=5,
        # A tiny cluster count keeps the synthetic 200-row corpus well above the
        # per-cluster minimum so the ANN path actually engages in the smoke.
        ann_configs=({"label": "cluster-default", "clusters": 8, "nprobe": 4},),
        corpus_vectors=corpus_vectors,
        corpus_texts=corpus_texts,
        query_vectors=query_vectors,
    )


def _synthetic_axis_b(db_path: str) -> dict[str, Any]:
    """Axis B smoke over a synthetic 40-doc corpus (~200 chunks) + 5 qrel queries."""

    rng = np.random.default_rng(1)
    corpus_chunks: list[dict[str, Any]] = []
    doc_count = 40
    for doc_index in range(doc_count):
        docid = f"D{doc_index}"
        for chunk_index in range(5):  # 40 docs x 5 chunks = 200 chunks
            corpus_chunks.append(
                {
                    "chunk_id": f"{docid}::c{chunk_index}",
                    "docid": docid,
                    "text": f"document {docid} passage {chunk_index} about topic {doc_index % 7}",
                }
            )
    corpus_vectors = _l2_normalize(
        rng.standard_normal((len(corpus_chunks), MINILM_DIM)).astype(np.float32)
    )
    # Each of the 5 queries is relevant to one doc and sits near that doc's first chunk.
    queries: list[dict[str, Any]] = []
    query_rows: list[NDArray[np.float32]] = []
    for query_index in range(5):
        docid = f"D{query_index}"
        anchor_row = query_index * 5  # first chunk of that doc
        queries.append(
            {
                "query_id": f"q{query_index}",
                "query": f"find document {docid} about topic {query_index % 7}",
                "relevant_docids": [docid],
            }
        )
        query_rows.append(
            corpus_vectors[anchor_row]
            + 0.02 * rng.standard_normal(MINILM_DIM).astype(np.float32)
        )
    query_vectors = _l2_normalize(np.vstack(query_rows).astype(np.float32))
    return run_axis_b_mldr(
        db_path,
        max_docs=doc_count,
        k=50,
        recall_ks=(10, 50),
        ndcg_k=10,
        corpus_chunks=corpus_chunks,
        queries=queries,
        corpus_vectors=corpus_vectors,
        query_vectors=query_vectors,
    )


def smoke() -> dict[str, Any]:
    """Runs both axes on tiny synthetic data (no Modal, no dataset download).

    Exercises ingest, vector search, hybrid search, footprint, and recall/nDCG so
    the LodeDB API calls and their signatures are validated end to end locally.
    """

    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        axis_a = _synthetic_axis_a(str(Path(tmp) / "axis_a"))
        axis_b = _synthetic_axis_b(str(Path(tmp) / "axis_b"))
    bundle = {"smoke": True, "axis_a": axis_a, "axis_b": axis_b}
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return bundle


if __name__ == "__main__":
    smoke()
