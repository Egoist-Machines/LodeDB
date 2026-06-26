#!/usr/bin/env python3
"""Late-interaction (MaxSim) vs single-vector page embeddings: recall and latency.

This is stage 2 of issue #25: measure whether the late-interaction prototype
(``LodeLateInteractionIndex``, MaxSim over a set of patch vectors per page)
recovers the true ranking better than a single mean-pooled vector per page, and
at what query-latency cost.

Both indexes are fed the **same** synthetic multi-vector documents, so this
measures storage and scan behaviour, not an encoder. Ground truth is the exact
brute-force MaxSim top-k over every document's full-precision patches -- the
metric ColPali / ColQwen optimise -- and both indexes are scored against it:

- late interaction: ``LodeLateInteractionIndex`` (one row per document, exact
  MaxSim over the resident patch matrix; ``--storage`` selects float16 / float32 /
  int8 precision);
- single vector: each page mean-pooled to one unit vector in a plain vector-only
  ``LodeDB``, the cheap baseline late interaction is meant to beat.

The synthetic generator plants latent "concepts": each page draws a few concept
directions (shared across pages) plus background patches, and each query is a
handful of a target page's concept directions with noise. That structure is what
makes MaxSim meaningful -- a query token matches the one page patch that carries
its concept, which mean-pooling dilutes. Real ViDoRe numbers need a bring-your-own
ColPali encoder; pass precomputed patches with ``--vectors`` (see ``--help``) to
run the identical comparison on real embeddings.

    uv run python benchmarks/late_interaction/run.py
    uv run python benchmarks/late_interaction/run.py --docs 2000 --queries 200 --dim 128

All output is metrics-only: counts, bytes, latency, recall; never vectors.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from lodedb import LodeDB, LodeLateInteractionIndex


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalizes each row of a 2-D matrix to unit length."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


def _synthetic_corpus(
    *,
    docs: int,
    patches_per_doc: int,
    dim: int,
    concepts: int,
    concepts_per_doc: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Builds ``docs`` patch matrices over a shared concept pool.

    Returns the list of ``(patches_per_doc, dim)`` matrices and the
    ``(docs, concepts_per_doc)`` integer table of each doc's concept ids (used to
    synthesize queries that target a specific page).
    """

    rng = np.random.default_rng(seed)
    concept_pool = _unit_rows(rng.standard_normal((concepts, dim)))
    doc_concepts = np.empty((docs, concepts_per_doc), dtype=np.int64)
    matrices: list[np.ndarray] = []
    for d in range(docs):
        chosen = rng.choice(concepts, size=concepts_per_doc, replace=False)
        doc_concepts[d] = chosen
        patches = rng.standard_normal((patches_per_doc, dim)).astype(np.float32) * 0.35
        # Plant each chosen concept onto one patch (the rest stay background).
        for slot, concept_id in enumerate(chosen):
            patches[slot % patches_per_doc] += concept_pool[concept_id]
        matrices.append(_unit_rows(patches))
    return matrices, doc_concepts


def _synthetic_queries(
    matrices: list[np.ndarray],
    doc_concepts: np.ndarray,
    *,
    queries: int,
    tokens_per_query: int,
    dim: int,
    concepts: int,
    seed: int,
) -> list[np.ndarray]:
    """Builds query token matrices, each derived from a random target page.

    A query takes ``tokens_per_query`` of its target page's concept directions
    (rebuilt from the shared pool) plus noise, so the target page is the natural
    MaxSim winner and concept-sharing pages trail it.
    """

    rng = np.random.default_rng(seed + 1)
    concept_pool = _unit_rows(
        np.random.default_rng(seed).standard_normal((concepts, dim))
    )
    out: list[np.ndarray] = []
    for _ in range(queries):
        target = int(rng.integers(0, len(matrices)))
        chosen = doc_concepts[target]
        take = min(tokens_per_query, len(chosen))
        picks = rng.choice(chosen, size=take, replace=False)
        tokens = concept_pool[picks] + rng.standard_normal((take, dim)).astype(
            np.float32
        ) * 0.25
        out.append(_unit_rows(tokens))
    return out


def _maxsim(query: np.ndarray, document: np.ndarray) -> float:
    """Exact MaxSim: sum over query tokens of the max patch dot-product."""

    return float((query @ document.T).max(axis=1).sum())


def _brute_force_topk(
    query: np.ndarray, matrices: list[np.ndarray], k: int
) -> list[int]:
    """Returns the exact MaxSim top-k document indices for one query."""

    scores = np.fromiter(
        (_maxsim(query, doc) for doc in matrices), dtype=np.float64, count=len(matrices)
    )
    return np.argsort(-scores)[:k].tolist()


def _dir_bytes(path: Path) -> int:
    """Returns the total size in bytes of a directory tree."""

    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _recall_at_k(predicted: list[str], truth_ids: set[str]) -> float:
    """Fraction of the ground-truth ids recovered in the predicted list."""

    if not truth_ids:
        return 1.0
    return len(set(predicted) & truth_ids) / len(truth_ids)


def run(args: argparse.Namespace) -> dict:
    """Runs the benchmark and returns a metrics-only result dict."""

    dim = args.dim
    if dim % 8 != 0:
        raise SystemExit("--dim must be a multiple of 8 (TurboVec requirement)")

    matrices, doc_concepts = _synthetic_corpus(
        docs=args.docs,
        patches_per_doc=args.patches,
        dim=dim,
        concepts=args.concepts,
        concepts_per_doc=args.concepts_per_doc,
        seed=args.seed,
    )
    queries = _synthetic_queries(
        matrices,
        doc_concepts,
        queries=args.queries,
        tokens_per_query=args.query_tokens,
        dim=dim,
        concepts=args.concepts,
        seed=args.seed,
    )
    doc_ids = [f"page-{i:06d}" for i in range(args.docs)]

    # Ground truth: exact brute-force MaxSim top-k over full-precision patches.
    truth = [
        {doc_ids[i] for i in _brute_force_topk(q, matrices, args.k)} for q in queries
    ]

    workdir = Path(tempfile.mkdtemp(prefix="li_bench_"))
    li_path = workdir / "late_interaction"
    sv_path = workdir / "single_vector"
    try:
        # -- late-interaction index ----------------------------------------
        li = LodeLateInteractionIndex(li_path, dim=dim, storage=args.storage)
        t0 = time.perf_counter()
        li.add_documents(
            [
                {"id": doc_id, "patches": matrix}
                for doc_id, matrix in zip(doc_ids, matrices, strict=True)
            ],
            normalize=False,
        )
        li.persist()
        li_ingest = time.perf_counter() - t0
        li_bytes = _dir_bytes(li_path)

        # -- single-vector (mean-pooled) baseline --------------------------
        sv = LodeDB.open_vector_store(sv_path, vector_dim=dim)
        pooled = [
            _unit_rows(matrix.mean(axis=0, keepdims=True))[0] for matrix in matrices
        ]
        t0 = time.perf_counter()
        sv.add_vectors_many(
            [
                {"id": doc_id, "vector": vec.tolist()}
                for doc_id, vec in zip(doc_ids, pooled, strict=True)
            ],
            normalize=False,
        )
        sv.persist()
        sv_ingest = time.perf_counter() - t0
        sv_bytes = _dir_bytes(sv_path)

        # Warm up: the first late-interaction query builds the in-memory resident
        # patch matrix (a one-time cost, amortized over the session). Time it
        # separately so the per-query latency reflects steady state.
        t0 = time.perf_counter()
        li.search(queries[0], k=args.k, normalize=False)
        li_build = time.perf_counter() - t0

        # -- query: latency + recall vs exact MaxSim ground truth ----------
        li_recall, sv_recall = [], []
        li_latency, sv_latency = [], []
        for q, gt in zip(queries, truth, strict=True):
            t0 = time.perf_counter()
            li_hits = li.search(q, k=args.k, normalize=False)
            li_latency.append((time.perf_counter() - t0) * 1000.0)
            li_recall.append(_recall_at_k([h.id for h in li_hits], gt))

            pooled_q = _unit_rows(q.mean(axis=0, keepdims=True))[0]
            t0 = time.perf_counter()
            sv_hits = sv.search_by_vector(pooled_q.tolist(), k=args.k, normalize=False)
            sv_latency.append((time.perf_counter() - t0) * 1000.0)
            sv_recall.append(_recall_at_k([h.id for h in sv_hits], gt))

        return {
            "config": {
                "docs": args.docs,
                "patches_per_doc": args.patches,
                "dim": dim,
                "concepts": args.concepts,
                "concepts_per_doc": args.concepts_per_doc,
                "query_tokens": args.query_tokens,
                "queries": args.queries,
                "k": args.k,
                "storage": args.storage,
                "seed": args.seed,
            },
            "late_interaction": {
                "storage": args.storage,
                "ingest_seconds": round(li_ingest, 3),
                "patch_rows": li.patch_count(),
                "disk_bytes": li_bytes,
                "resident_build_seconds": round(li_build, 3),
                "mean_query_ms": round(float(np.mean(li_latency)), 3),
                "p95_query_ms": round(float(np.percentile(li_latency, 95)), 3),
                "recall_at_k": round(float(np.mean(li_recall)), 4),
            },
            "single_vector": {
                "ingest_seconds": round(sv_ingest, 3),
                "rows": args.docs,
                "disk_bytes": sv_bytes,
                "mean_query_ms": round(float(np.mean(sv_latency)), 3),
                "p95_query_ms": round(float(np.percentile(sv_latency, 95)), 3),
                "recall_at_k": round(float(np.mean(sv_recall)), 4),
            },
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> None:
    """Parses args, runs the benchmark, and prints metrics-only JSON."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=1000)
    parser.add_argument("--patches", type=int, default=64, help="patches per page")
    parser.add_argument("--dim", type=int, default=128, help="multiple of 8")
    parser.add_argument("--concepts", type=int, default=256)
    parser.add_argument("--concepts-per-doc", type=int, default=6)
    parser.add_argument("--query-tokens", type=int, default=8)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--storage",
        choices=("float16", "float32", "int8"),
        default="float16",
        help="patch-matrix storage precision",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to write the JSON result to",
    )
    args = parser.parse_args()

    result = run(args)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
