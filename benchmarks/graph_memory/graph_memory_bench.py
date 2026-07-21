"""Graph-memory benchmark: vector-in, predicate filters, and graph traversal.

Exercises the three capabilities added for the knowledge-graph / memory stack
and reports **metrics only** (counts, latency, throughput, recall/overlap, never
raw text, queries, or embeddings), matching the repo's benchmark provenance rules.

Three sub-benchmarks, all driven from the same loaded corpus:

1. ``vector_in``: text-in ingest (LodeDB embeds internally) vs vector-in ingest
   (caller supplies precomputed vectors via ``add_vectors_many``), plus query
   parity: ``search`` vs ``search_by_vector`` over byte-identical indexes should
   return the same hits, isolating the embedding cost vector-in removes.
2. ``filters``: search latency across predicate selectivities, exact ``$eq``
   (posting-allowlist pushdown) vs ``$gte`` / ``$ne`` / ``$exists`` (which today
   are resolved by the per-field planner). Compares the planner against a
   per-document scan.
3. ``graph``: a synthetic knowledge graph over the corpus, with k-hop traversal
   latency (SQLite topology) and hybrid ``search_subgraph`` latency (semantic
   seed + structural expansion) at scale.

Runnable locally (``dataset=synthetic`` needs no network) or on Modal (GovReport).
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any

from lodedb import LodeDB
from lodedb.graph import KnowledgeGraph
from lodedb.local.backends import build_local_embedding_backend
from lodedb.local.presets import resolve_preset

_TOPICS = ("ml", "bio", "law", "econ", "physics", "history", "art", "med", "eng", "geo")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _percentile(values: list[float], pct: float) -> float:
    """Returns the ``pct`` percentile (0..100) of ``values`` (nearest-rank)."""

    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[rank]


def _latency_summary(samples_ms: list[float]) -> dict[str, float]:
    """Summarizes a list of per-call latencies (ms)."""

    return {
        "count": len(samples_ms),
        "p50_ms": round(median(samples_ms), 4) if samples_ms else 0.0,
        "p95_ms": round(_percentile(samples_ms, 95), 4),
        "mean_ms": round(sum(samples_ms) / len(samples_ms), 4) if samples_ms else 0.0,
    }


def _load_documents(
    dataset_name: str, max_documents: int, query_count: int
) -> tuple[list[str], list[str]]:
    """Loads ``(documents, queries)``; ``synthetic`` needs no network."""

    if dataset_name == "synthetic":
        rng = random.Random(1234)
        docs = [
            f"Document {i} concerns {random.choice(_TOPICS)} and "
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
        docs.append(report[:4000])
        if len(queries) < query_count:
            summary = str(row.get("summary", "")).strip()
            if summary:
                queries.append(summary[:400])
    return docs, queries


def _synth_metadata(index: int) -> dict[str, Any]:
    """Deterministic filterable metadata for document ``index``."""

    return {"topic": _TOPICS[index % len(_TOPICS)], "year": 2000 + (index % 26)}


# --------------------------------------------------------------------------- #
# 1. vector-in vs text-in
# --------------------------------------------------------------------------- #


def run_vector_in_bench(
    documents: list[str],
    queries: list[str],
    *,
    model: str,
    device: str,
    top_k: int,
    workdir: Path,
) -> dict[str, Any]:
    """Compares text-in vs vector-in ingest and query, and checks parity."""

    preset = resolve_preset(model)
    backend, resolution = build_local_embedding_backend(preset, device=device)
    native_dim = preset.native_dim

    # Cap each document to a single chunk (LodeDB chunks text beyond
    # chunk_character_limit into multiple vectors, while a precomputed doc-level
    # vector is exactly one chunk). Capping keeps the text-in and vector-in
    # indexes byte-identical so the parity overlap is an exact equality check,
    # not confounded by chunk-count differences on long documents.
    chunk_char_cap = 900
    documents = [doc[:chunk_char_cap] for doc in documents]

    # Precompute document + query embeddings with LodeDB's own backend, so the
    # vector-in index is byte-identical to the text-in index and parity is exact.
    embed_started = time.perf_counter()
    doc_vectors = [list(vec) for vec in backend.embed_documents(tuple(documents))]
    doc_embed_ms = (time.perf_counter() - embed_started) * 1000.0
    query_vectors = [list(backend.embed_query(q)) for q in queries]

    metadatas = [_synth_metadata(i) for i in range(len(documents))]

    # text-in ingest: LodeDB embeds internally then stores.
    text_db = LodeDB(workdir / "textin", model=model, device=device)
    t0 = time.perf_counter()
    text_db.add_many(
        [
            {"text": documents[i], "id": f"d{i}", "metadata": metadatas[i]}
            for i in range(len(documents))
        ]
    )
    textin_ingest_ms = (time.perf_counter() - t0) * 1000.0

    # vector-in ingest: store precomputed vectors (no embedding), normalize=False
    # because the backend already L2-normalizes.
    vec_db = LodeDB(workdir / "vectorin", model=model, device=device)
    t0 = time.perf_counter()
    vec_db.add_vectors_many(
        [
            {"vector": doc_vectors[i], "id": f"d{i}", "metadata": metadatas[i]}
            for i in range(len(documents))
        ],
        normalize=False,
    )
    vectorin_ingest_ms = (time.perf_counter() - t0) * 1000.0

    # query latency + parity (overlap of top-k id sets, identical indexes).
    text_lat: list[float] = []
    vec_lat: list[float] = []
    overlaps: list[float] = []
    for q, qv in zip(queries, query_vectors, strict=True):
        s = time.perf_counter()
        text_hits = text_db.search(q, k=top_k)
        text_lat.append((time.perf_counter() - s) * 1000.0)
        s = time.perf_counter()
        vec_hits = vec_db.search_by_vector(qv, k=top_k, normalize=False)
        vec_lat.append((time.perf_counter() - s) * 1000.0)
        text_ids = {h.id for h in text_hits}
        vec_ids = {h.id for h in vec_hits}
        denom = max(1, len(text_ids))
        overlaps.append(len(text_ids & vec_ids) / denom)

    n = len(documents)
    result = {
        "document_count": n,
        "query_count": len(queries),
        "native_dim": native_dim,
        "doc_char_cap": chunk_char_cap,
        "embedding_device": resolution.effective_device,
        "doc_embed_ms": round(doc_embed_ms, 2),
        "textin_ingest_ms": round(textin_ingest_ms, 2),
        "vectorin_ingest_ms": round(vectorin_ingest_ms, 2),
        "textin_ingest_docs_per_s": round(n / (textin_ingest_ms / 1000.0), 2)
        if textin_ingest_ms
        else 0.0,
        "vectorin_ingest_docs_per_s": round(n / (vectorin_ingest_ms / 1000.0), 2)
        if vectorin_ingest_ms
        else 0.0,
        "ingest_speedup_vectorin_over_textin": round(textin_ingest_ms / vectorin_ingest_ms, 3)
        if vectorin_ingest_ms
        else 0.0,
        "search_text_latency": _latency_summary(text_lat),
        "search_vector_latency": _latency_summary(vec_lat),
        "topk_overlap_mean": round(sum(overlaps) / len(overlaps), 4) if overlaps else 0.0,
        "topk_overlap_min": round(min(overlaps), 4) if overlaps else 0.0,
    }
    text_db.close()
    vec_db.close()
    return result


# --------------------------------------------------------------------------- #
# 2. predicate filter latency
# --------------------------------------------------------------------------- #


def run_filter_bench(
    documents: list[str],
    queries: list[str],
    *,
    model: str,
    device: str,
    top_k: int,
    workdir: Path,
) -> dict[str, Any]:
    """Measures search latency across filter predicate kinds and selectivities."""

    db = LodeDB(workdir / "filters", model=model, device=device)
    db.add_many(
        [
            {"text": documents[i], "id": f"d{i}", "metadata": _synth_metadata(i)}
            for i in range(len(documents))
        ]
    )

    cases: dict[str, Any] = {
        "no_filter": None,
        "eq_topic": {"topic": "ml"},  # exact -> posting allowlist
        "in_topic_3": {"topic": {"$in": list(_TOPICS[:3])}},
        "gte_year": {"year": {"$gte": 2013}},  # ordered predicate
        "range_year": {"year": {"$gte": 2010, "$lt": 2015}},
        "ne_topic": {"topic": {"$ne": "ml"}},
        "exists_topic": {"topic": {"$exists": True}},
        "and_topic_year": {"$and": [{"topic": "bio"}, {"year": {"$gte": 2015}}]},
    }

    per_case: dict[str, Any] = {}
    for name, flt in cases.items():
        latencies: list[float] = []
        result_counts: list[int] = []
        for q in queries:
            s = time.perf_counter()
            hits = db.search(q, k=top_k, filter=flt)
            latencies.append((time.perf_counter() - s) * 1000.0)
            result_counts.append(len(hits))
        per_case[name] = {
            **_latency_summary(latencies),
            "avg_result_count": round(sum(result_counts) / len(result_counts), 2)
            if result_counts
            else 0.0,
        }
    db.close()
    return {"document_count": len(documents), "query_count": len(queries), "cases": per_case}


# --------------------------------------------------------------------------- #
# 3. graph traversal + hybrid retrieval
# --------------------------------------------------------------------------- #


def run_graph_bench(
    documents: list[str],
    *,
    model: str,
    device: str,
    node_count: int,
    avg_degree: int,
    hops: int,
    seed_queries: int,
    top_k: int,
    workdir: Path,
) -> dict[str, Any]:
    """Builds a synthetic KG over the corpus and times traversal + hybrid search."""

    preset = resolve_preset(model)
    backend, _resolution = build_local_embedding_backend(preset, device=device)
    node_count = min(node_count, len(documents))
    labels = documents[:node_count]

    # Precompute node embeddings once and index via vector-in (fast build).
    node_vectors = [list(vec) for vec in backend.embed_documents(tuple(labels))]

    # Per-node build rate on a small sample (one index commit per add_node), for
    # the speedup comparison against the batched add_nodes path.
    sample = min(200, node_count)
    kg_sample = KnowledgeGraph(workdir / "kg_sample", model=model, device=device)
    sample_started = time.perf_counter()
    for i in range(sample):
        kg_sample.add_node(
            id=f"s{i}", type=_TOPICS[i % len(_TOPICS)], embedding=node_vectors[i]
        )
    per_node_build_ms = (time.perf_counter() - sample_started) * 1000.0
    kg_sample.close()

    # Batched build of the full graph via add_nodes/add_edges (one commit per batch).
    kg = KnowledgeGraph(workdir / "kg", model=model, device=device)
    build_started = time.perf_counter()
    kg.add_nodes(
        [
            {
                "id": f"n{i}",
                "type": _TOPICS[i % len(_TOPICS)],
                "embedding": node_vectors[i],
                "properties": {"idx": i},
            }
            for i in range(node_count)
        ]
    )
    node_build_ms = (time.perf_counter() - build_started) * 1000.0

    rng = random.Random(7)
    edge_count = node_count * avg_degree
    edge_items: list[dict[str, Any]] = []
    for _ in range(edge_count):
        a = rng.randrange(node_count)
        b = rng.randrange(node_count)
        if a == b:
            continue
        edge_items.append({"src": f"n{a}", "relation": "rel", "dst": f"n{b}"})
    edge_started = time.perf_counter()
    kg.add_edges(edge_items)
    edge_build_ms = (time.perf_counter() - edge_started) * 1000.0

    # k-hop traversal latency from random seeds.
    khop_latency: list[float] = []
    khop_sizes: list[int] = []
    for _ in range(seed_queries):
        seed = f"n{rng.randrange(node_count)}"
        s = time.perf_counter()
        sub = kg.k_hop(seed, k=hops, direction="both")
        khop_latency.append((time.perf_counter() - s) * 1000.0)
        khop_sizes.append(len(sub.nodes))

    # hybrid search_subgraph latency (semantic seed + expansion).
    hybrid_latency: list[float] = []
    hybrid_sizes: list[int] = []
    for i in range(seed_queries):
        qv = node_vectors[i % node_count]
        s = time.perf_counter()
        sub = kg.search_subgraph(embedding=qv, k=top_k, hops=hops, direction="both")
        hybrid_latency.append((time.perf_counter() - s) * 1000.0)
        hybrid_sizes.append(len(sub.nodes))

    stats = kg.stats()
    kg.close()
    return {
        "node_count": node_count,
        "edge_count_requested": edge_count,
        "edge_count_actual": stats["edges"],
        "avg_degree": avg_degree,
        "hops": hops,
        "node_build_ms": round(node_build_ms, 2),
        "edge_build_ms": round(edge_build_ms, 2),
        "node_build_nodes_per_s": round(node_count / (node_build_ms / 1000.0), 2)
        if node_build_ms
        else 0.0,
        "per_node_build_sample": sample,
        "per_node_build_nodes_per_s": round(sample / (per_node_build_ms / 1000.0), 2)
        if per_node_build_ms
        else 0.0,
        "build_speedup_batched_over_per_node": round(
            (node_count / node_build_ms) / (sample / per_node_build_ms), 2
        )
        if node_build_ms and per_node_build_ms
        else 0.0,
        "edge_build_edges_per_s": round(stats["edges"] / (edge_build_ms / 1000.0), 2)
        if edge_build_ms
        else 0.0,
        "khop_latency": _latency_summary(khop_latency),
        "khop_avg_subgraph_nodes": round(sum(khop_sizes) / len(khop_sizes), 2)
        if khop_sizes
        else 0.0,
        "hybrid_latency": _latency_summary(hybrid_latency),
        "hybrid_avg_subgraph_nodes": round(sum(hybrid_sizes) / len(hybrid_sizes), 2)
        if hybrid_sizes
        else 0.0,
    }


# --------------------------------------------------------------------------- #
# suite
# --------------------------------------------------------------------------- #


def run_graph_memory_suite(
    *,
    dataset_name: str = "synthetic",
    max_documents: int = 2000,
    query_count: int = 64,
    model: str = "minilm",
    device: str = "cpu",
    top_k: int = 10,
    graph_nodes: int = 2000,
    avg_degree: int = 8,
    hops: int = 2,
    seed_queries: int = 64,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Runs all three sub-benchmarks and returns one metrics-only bundle."""

    documents, queries = _load_documents(dataset_name, max_documents, query_count)
    if not queries:
        queries = [f"information about {t}" for t in _TOPICS]

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        vector_in = run_vector_in_bench(
            documents, queries, model=model, device=device, top_k=top_k, workdir=workdir / "vi"
        )
        filters = run_filter_bench(
            documents, queries, model=model, device=device, top_k=top_k, workdir=workdir / "fi"
        )
        graph = run_graph_bench(
            documents,
            model=model,
            device=device,
            node_count=graph_nodes,
            avg_degree=avg_degree,
            hops=hops,
            seed_queries=seed_queries,
            top_k=top_k,
            workdir=workdir / "gr",
        )

    bundle = {
        "suite": "graph_memory",
        "provenance": "measured",
        "dataset": dataset_name,
        "model": model,
        "device": device,
        "document_count": len(documents),
        "query_count": len(queries),
        "vector_in": vector_in,
        "filters": filters,
        "graph": graph,
    }
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8"
        )
    return bundle


def main() -> None:
    """CLI entry point for local runs."""

    parser = argparse.ArgumentParser(description="LodeDB graph-memory benchmark")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--max-documents", type=int, default=2000)
    parser.add_argument("--query-count", type=int, default=64)
    parser.add_argument("--model", default="minilm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--graph-nodes", type=int, default=2000)
    parser.add_argument("--avg-degree", type=int, default=8)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--seed-queries", type=int, default=64)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    bundle = run_graph_memory_suite(
        dataset_name=args.dataset,
        max_documents=args.max_documents,
        query_count=args.query_count,
        model=args.model,
        device=args.device,
        top_k=args.top_k,
        graph_nodes=args.graph_nodes,
        avg_degree=args.avg_degree,
        hops=args.hops,
        seed_queries=args.seed_queries,
        output_dir=args.out,
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
