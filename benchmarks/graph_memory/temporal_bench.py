"""Temporal knowledge-graph benchmark — metrics only (counts, latency, throughput,
never payloads).

Exercises the bi-temporal path of ``lodedb.graph.TemporalKnowledgeGraph``: ingesting
entities and facts, invalidating facts as contradictions arrive, and querying "as of"
past instants plus full history. Self-contained and offline — it uses a deterministic
hash embedder, so it needs no model download and runs anywhere.

    python benchmarks/graph_memory/temporal_bench.py --entities 2000 --facts 4000
"""

from __future__ import annotations

import argparse
import json
import math
import time

from lodedb.graph import TemporalKnowledgeGraph


class HashEmbedder:
    """Deterministic offline embedder: bucket bytes into ``dim`` bins, L2-normalize."""

    def __init__(self, dim: int = 64) -> None:
        self.dimension = dim

    def embed(self, texts, role):
        dim = self.dimension
        out = []
        for text in texts:
            vec = [0.0] * dim
            for byte in text.lower().encode("utf-8"):
                vec[byte % dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            out.append([x / norm for x in vec] if norm else [1.0] + [0.0] * (dim - 1))
        return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _latency_ms(samples_ms: list[float]) -> dict[str, float]:
    return {
        "count": len(samples_ms),
        "p50_ms": round(_percentile(samples_ms, 50), 4),
        "p95_ms": round(_percentile(samples_ms, 95), 4),
        "p99_ms": round(_percentile(samples_ms, 99), 4),
    }


def run(entities: int, facts: int, contradiction_rate: float, query_count: int) -> dict:
    dim = 64
    kg = TemporalKnowledgeGraph(embedder=HashEmbedder(dim))  # in-memory

    # 1. Ingest entities.
    t0 = time.perf_counter()
    for i in range(entities):
        kg.upsert_entity(f"e{i}", "Thing", f"entity number {i} thing widget")
    entity_ingest_s = time.perf_counter() - t0

    # 2. Ingest facts; a fraction contradict (and thus invalidate) a prior fact on the
    #    same (src, relation), so the invalidation path is exercised at scale.
    last_fact_for: dict[tuple[int, str], str] = {}
    invalidations = 0
    fact_latencies: list[float] = []
    t0 = time.perf_counter()
    for j in range(facts):
        src = j % entities
        dst = (j * 7 + 1) % entities
        relation = "rel"
        valid_at = 1000 + j
        key = (src, relation)
        prior = last_fact_for.get(key)
        contradict = prior is not None and (j % max(1, int(1 / max(contradiction_rate, 1e-9)))) == 0
        invalidates = [prior] if contradict else []
        s = time.perf_counter()
        fid = kg.add_fact(
            f"e{src}", relation, f"e{dst}", f"e{src} rel e{dst} at {valid_at}",
            valid_at=valid_at, invalidates=invalidates,
        )
        fact_latencies.append((time.perf_counter() - s) * 1000.0)
        if contradict:
            invalidations += 1
        last_fact_for[key] = fid
    fact_ingest_s = time.perf_counter() - t0

    mid = 1000 + facts // 2  # an "as of" instant in the middle of the timeline

    # 3. As-of neighbor queries.
    neighbor_ms: list[float] = []
    for q in range(query_count):
        s = time.perf_counter()
        kg.neighbors(f"e{q % entities}", direction="out", relation="rel", as_of=mid)
        neighbor_ms.append((time.perf_counter() - s) * 1000.0)

    # 4. As-of semantic subgraph queries.
    subgraph_ms: list[float] = []
    for q in range(query_count):
        s = time.perf_counter()
        kg.search_subgraph("entity thing widget", k=5, hops=1, as_of=mid)
        subgraph_ms.append((time.perf_counter() - s) * 1000.0)

    # 5. History (all frames) for a sample of entities.
    history_ms: list[float] = []
    for q in range(min(query_count, entities)):
        s = time.perf_counter()
        kg.history(f"e{q % entities}")
        history_ms.append((time.perf_counter() - s) * 1000.0)

    stats = kg.stats()
    return {
        "config": {
            "entities": entities,
            "facts": facts,
            "contradiction_rate": contradiction_rate,
            "query_count": query_count,
            "vector_dim": dim,
        },
        "ingest": {
            "entities_per_s": round(entities / entity_ingest_s, 1) if entity_ingest_s else 0.0,
            "facts_per_s": round(facts / fact_ingest_s, 1) if fact_ingest_s else 0.0,
            "invalidations": invalidations,
            "fact_add_latency": _latency_ms(fact_latencies),
        },
        "query": {
            "neighbors_as_of": _latency_ms(neighbor_ms),
            "search_subgraph_as_of": _latency_ms(subgraph_ms),
            "history": _latency_ms(history_ms),
        },
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LodeDB bi-temporal graph benchmark (metrics only)")
    parser.add_argument("--entities", type=int, default=2000)
    parser.add_argument("--facts", type=int, default=4000)
    parser.add_argument("--contradiction-rate", type=float, default=0.25)
    parser.add_argument("--query-count", type=int, default=64)
    args = parser.parse_args()

    result = run(args.entities, args.facts, args.contradiction_rate, args.query_count)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
