"""Image-vector storage benchmark: LodeDB vs optional Chroma / Qdrant.

Every store is fed the **same** precomputed CLIP-dimension vectors (512-d by
default), so this measures storage and scan, not the encoder. It reports ingest
time, on-disk footprint, mean query latency, and recall@k against the exact
brute-force top-k (so the metric is identical across stores).

Run (LodeDB only, no extra installs):

    uv run python benchmarks/multimodal_image/run.py

With competitors, when installed:

    uv pip install chromadb qdrant-client
    uv run python benchmarks/multimodal_image/run.py --n 5000 --queries 200

The real end-to-end image path (CLIP encode + ``add_image`` + cross-modal text
search) is shown in ``examples/multimodal_clip.py``; this benchmark deliberately
uses fixed vectors so the comparison is apples-to-apples across stores.

All output is metrics-only: counts, bytes, latency, recall, backend labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

DIM = 512  # sentence-transformers/clip-ViT-B-32 embedding dimension


def unit_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    """Returns ``n`` L2-normalized float32 vectors (a stand-in for image embeddings)."""

    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((n, dim)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix


def footprint_bytes(path: Path) -> int:
    """Returns the total on-disk size of every file under a store directory."""

    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def recall_at_k(approx_ids: list[int], truth_ids: list[int], k: int) -> float:
    """Returns the fraction of the true top-``k`` neighbors the store returned."""

    if not truth_ids:
        return 0.0
    return len(set(approx_ids[:k]) & set(truth_ids[:k])) / float(min(k, len(truth_ids)))


def brute_force_topk(corpus: np.ndarray, queries: np.ndarray, k: int) -> list[list[int]]:
    """Returns the exact top-``k`` neighbor ids per query (cosine == dot on unit vectors)."""

    scores = queries @ corpus.T
    top = np.argsort(-scores, axis=1)[:, :k]
    return top.tolist()


def _summary(
    backend: str,
    ingest_s: float,
    bytes_on_disk: int,
    latencies_ms: list[float],
    recalls: list[float],
) -> dict[str, object]:
    """Builds one store's metrics-only result row."""

    return {
        "backend": backend,
        "ingest_seconds": round(ingest_s, 4),
        "footprint_mb": round(bytes_on_disk / 1e6, 2),
        "mean_query_ms": round(float(np.mean(latencies_ms)), 3),
        "p95_query_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "recall_at_k": round(float(np.mean(recalls)), 4),
    }


def bench_lodedb(corpus, queries, truth, k, workdir) -> dict[str, object]:
    """Benchmarks LodeDB's bring-your-own-vectors path on the shared vectors."""

    from lodedb import LodeDB

    path = workdir / "lodedb"
    db = LodeDB.open_vector_store(path, vector_dim=corpus.shape[1])
    docs = [{"vector": corpus[i].tolist(), "id": str(i)} for i in range(corpus.shape[0])]
    start = time.perf_counter()
    db.add_vectors_many(docs)
    db.persist()
    ingest_s = time.perf_counter() - start

    latencies, recalls = [], []
    for qi in range(queries.shape[0]):
        s = time.perf_counter()
        hits = db.search_by_vector(queries[qi].tolist(), k=k)
        latencies.append((time.perf_counter() - s) * 1000.0)
        recalls.append(recall_at_k([int(h.id) for h in hits], truth[qi], k))
    db.close()
    return _summary("lodedb", ingest_s, footprint_bytes(path), latencies, recalls)


def bench_chroma(corpus, queries, truth, k, workdir) -> dict[str, object] | None:
    """Benchmarks Chroma (persistent) on the shared vectors, or None if not installed."""

    try:
        import chromadb
    except ImportError:
        return None

    path = workdir / "chroma"
    client = chromadb.PersistentClient(path=str(path))
    collection = client.create_collection("images", metadata={"hnsw:space": "cosine"})
    ids = [str(i) for i in range(corpus.shape[0])]
    start = time.perf_counter()
    # Chroma rejects very large single adds, so batch the upload.
    for lo in range(0, corpus.shape[0], 5000):
        hi = min(lo + 5000, corpus.shape[0])
        collection.add(ids=ids[lo:hi], embeddings=corpus[lo:hi].tolist())
    ingest_s = time.perf_counter() - start

    latencies, recalls = [], []
    for qi in range(queries.shape[0]):
        s = time.perf_counter()
        res = collection.query(query_embeddings=[queries[qi].tolist()], n_results=k)
        latencies.append((time.perf_counter() - s) * 1000.0)
        returned = [int(x) for x in res["ids"][0]]
        recalls.append(recall_at_k(returned, truth[qi], k))
    return _summary("chroma", ingest_s, footprint_bytes(path), latencies, recalls)


def bench_qdrant(corpus, queries, truth, k, workdir) -> dict[str, object] | None:
    """Benchmarks Qdrant (embedded) on the shared vectors, or None if not installed."""

    try:
        from qdrant_client import QdrantClient, models
    except ImportError:
        return None

    path = workdir / "qdrant"
    client = QdrantClient(path=str(path))
    client.create_collection(
        "images",
        vectors_config=models.VectorParams(
            size=corpus.shape[1], distance=models.Distance.COSINE
        ),
    )
    points = [
        models.PointStruct(id=i, vector=corpus[i].tolist()) for i in range(corpus.shape[0])
    ]
    start = time.perf_counter()
    client.upsert("images", points=points)
    ingest_s = time.perf_counter() - start

    latencies, recalls = [], []
    for qi in range(queries.shape[0]):
        s = time.perf_counter()
        res = client.query_points("images", query=queries[qi].tolist(), limit=k).points
        latencies.append((time.perf_counter() - s) * 1000.0)
        recalls.append(recall_at_k([int(p.id) for p in res], truth[qi], k))
    client.close()
    return _summary("qdrant", ingest_s, footprint_bytes(path), latencies, recalls)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=2000, help="Number of corpus vectors.")
    parser.add_argument("--queries", type=int, default=200, help="Number of query vectors.")
    parser.add_argument("--k", type=int, default=10, help="top-k per query.")
    parser.add_argument("--dim", type=int, default=DIM, help="Embedding dimension.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed.")
    args = parser.parse_args()

    corpus = unit_vectors(args.n, args.dim, args.seed)
    queries = unit_vectors(args.queries, args.dim, args.seed + 1)
    truth = brute_force_topk(corpus, queries, args.k)

    import tempfile

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        rows.append(bench_lodedb(corpus, queries, truth, args.k, workdir))
        for runner, name in ((bench_chroma, "chroma"), (bench_qdrant, "qdrant")):
            row = runner(corpus, queries, truth, args.k, workdir)
            if row is None:
                print(f"# {name} not installed; skipping (pip install it to compare)")
            else:
                rows.append(row)

    config = {"n": args.n, "queries": args.queries, "k": args.k, "dim": args.dim}
    print(json.dumps({"config": config, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
