"""LodeDB hybrid-search recall benchmark payload (runs locally; no GPU needed).

Measures the exact-token recall win that hybrid retrieval exists to deliver.
Each probe is an exact token (an error code, a hyphenated serial, or an ISO
date) planted in exactly one document's body, surrounded by distractor
documents that share no token with it. Pure vector search over a content-blind
embedding cannot rank the carrier; the lexical BM25 ranker isolates it, and
Reciprocal Rank Fusion lifts it into the top-k.

For each mode (``vector``, ``hybrid``, ``lexical``) it records recall@k and the
mean reciprocal rank of the carrier across all probes, plus the per-query
latency. The deterministic hash embedding backend is used on purpose: it makes
the "embedding cannot see the literal token" failure mode reproducible without
downloading a model, so the lexical contribution is isolated rather than masked
by a model that happens to encode some character-level signal. Output is
raw-payload-free (counts, ratios, latency only — never tokens or terms).

Run::

    python benchmarks/hybrid/hybrid_recall.py
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local import LodeDB

# Exact tokens the embedding cannot capture but the lexical ranker matches
# verbatim: an error code, a hyphenated serial, and an ISO date.
_PROBES = [
    "E1234",
    "ABC-123",
    "2024-01-15",
    "X9-ALPHA",
    "2023-11-30",
]

_DISTRACTORS = [
    "Quick brown foxes and lazy dogs wander the meadow at noon under a warm sky.",
    "Quarterly revenue grew while operating costs declined across every region.",
    "The committee reviewed the proposal and deferred the vote to next session.",
    "Rainfall totals exceeded the seasonal average across the northern districts.",
    "A short note on gardening, compost rotation, and seasonal planting schedules.",
]


def _build_corpus(db: LodeDB, *, distractor_copies: int) -> dict[str, str]:
    """Seeds one carrier per probe plus many shared distractors; returns probe->carrier id."""

    carriers: dict[str, str] = {}
    for index, token in enumerate(_PROBES):
        carrier_id = f"carrier-{index}"
        db.add(
            "The overnight maintenance log records that the auxiliary unit reported "
            f"reference {token} before operations resumed as normal.",
            id=carrier_id,
        )
        carriers[token] = carrier_id
    counter = 0
    for _ in range(distractor_copies):
        for distractor in _DISTRACTORS:
            db.add(f"{distractor} (note {counter})", id=f"distractor-{counter}")
            counter += 1
    return carriers


def _measure(db: LodeDB, carriers: dict[str, str], *, mode: str, k: int) -> dict[str, float]:
    """Runs every probe under one mode and returns recall@k, MRR, and mean latency."""

    hits_at_k = 0
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    for token, carrier_id in carriers.items():
        started = time.perf_counter()
        results = db.search(token, k=k, mode=mode)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        ids = [hit.id for hit in results]
        if carrier_id in ids:
            hits_at_k += 1
            reciprocal_ranks.append(1.0 / (ids.index(carrier_id) + 1))
        else:
            reciprocal_ranks.append(0.0)
    probe_count = len(carriers)
    return {
        "recall_at_k": hits_at_k / probe_count if probe_count else 0.0,
        "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "mean_latency_ms": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
    }


def run(*, k: int = 5, distractor_copies: int = 20, dim: int = 384) -> dict[str, object]:
    """Builds the corpus once and benchmarks vector, hybrid, and lexical recall."""

    with tempfile.TemporaryDirectory() as path:
        db = LodeDB(
            path=path,
            store_text=True,
            _embedding_backend=HashEmbeddingBackend(native_dim=dim),
        )
        carriers = _build_corpus(db, distractor_copies=distractor_copies)
        document_count = db.count()
        modes = {
            mode: _measure(db, carriers, mode=mode, k=k)
            for mode in ("vector", "hybrid", "lexical")
        }
        db.close()
    return {
        "k": k,
        "probe_count": len(carriers),
        "document_count": document_count,
        "modes": modes,
    }


def main() -> None:
    """Prints the raw-payload-free recall summary as JSON."""

    print(json.dumps(run(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
