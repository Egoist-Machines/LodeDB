"""Side-by-side Python-oracle vs Rust-core migration benchmark.

The output is metrics-only: counts, timings, checksums, ratios, and backend labels.
It intentionally does not write raw documents, queries, chunks, embeddings, or paths.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import math
import os
import random
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from lodedb import LodeDB
from lodedb.engine.core import EngineDocument, EngineVectorDocument
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.native_adapter import NativeCoreAdapter

_INDEX_ID = "default"


def _measure(label: str, func: Callable[[], int]) -> dict[str, Any]:
    started = time.perf_counter()
    checksum = int(func())
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "name": label,
        "elapsed_ms": round(elapsed_ms, 3),
        "checksum": checksum,
    }


def _ratio(rust_ms: float, python_ms: float) -> float | None:
    return round(rust_ms / python_ms, 3) if python_ms else None


def _comparison(name: str, python_row: dict[str, Any], rust_row: dict[str, Any]) -> dict[str, Any]:
    ratio = _ratio(float(rust_row["elapsed_ms"]), float(python_row["elapsed_ms"]))
    out = {
        "name": name,
        "python_elapsed_ms": python_row["elapsed_ms"],
        "rust_elapsed_ms": rust_row["elapsed_ms"],
        "rust_to_python_elapsed_ratio": ratio,
        "python_checksum": python_row["checksum"],
        "rust_checksum": rust_row["checksum"],
        "checksums_match": python_row["checksum"] == rust_row["checksum"],
    }
    if ratio is not None:
        out["rust_speedup"] = round(1.0 / ratio, 3) if ratio else None
    return out


def _vector(seed: int, dim: int) -> list[float]:
    rng = random.Random(seed)
    values = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(value * value for value in values))
    return [value / norm for value in values]


def _metadata(i: int) -> dict[str, str]:
    return {
        "tenant": "acme" if i % 3 else "zen",
        "kind": "incident" if i % 5 == 0 else "note",
        "bucket": str(i % 16),
    }


def _vectors(documents: int, dim: int) -> list[EngineVectorDocument]:
    return [
        EngineVectorDocument(
            f"vec-{i:06d}",
            _vector(i, dim),
            metadata=_metadata(i),
            text=None,
        )
        for i in range(documents)
    ]


def _queries(queries: int, dim: int) -> list[list[float]]:
    return [_vector(10_000 + i, dim) for i in range(queries)]


def _text_documents(documents: int) -> list[EngineDocument]:
    return [
        EngineDocument(
            f"doc-{i:06d}",
            (
                f"record {i} tenant {_metadata(i)['tenant']} bucket {i % 16} "
                f"error E-{1000 + (i % 97)} local recovery vector search"
            ),
            metadata=_metadata(i),
        )
        for i in range(documents)
    ]


def _ids_checksum(ids: Iterable[str]) -> int:
    total = 0
    for item in ids:
        for char in item:
            total = (total * 131 + ord(char)) % 1_000_000_007
    return total


def _hit_checksum(rows: Iterable[Iterable[str]]) -> int:
    total = 0
    for ids in rows:
        total = (total * 131 + _ids_checksum(ids)) % 1_000_000_007
    return total


def _require_native_adapter() -> NativeCoreAdapter:
    extension_override = os.environ.get("LODEDB_NATIVE_CORE_EXTENSION_PATH")
    if extension_override:
        _load_extension_override(Path(extension_override))
    adapter = NativeCoreAdapter()
    if not adapter.available:
        raise RuntimeError("native core extension is not available")
    return adapter


def _load_extension_override(path: Path) -> None:
    import sys

    import lodedb  # noqa: F401 - ensure the package parent exists

    spec = importlib.util.spec_from_file_location("lodedb._turbovec", path)
    if spec is None or spec.loader is None:
        loader = importlib.machinery.ExtensionFileLoader("lodedb._turbovec", str(path))
        spec = importlib.util.spec_from_file_location(
            "lodedb._turbovec",
            path,
            loader=loader,
        )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load native extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lodedb._turbovec"] = module
    spec.loader.exec_module(module)


def _python_vector_bench(
    documents: list[EngineVectorDocument],
    queries: list[list[float]],
    *,
    dim: int,
    k: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="lodedb-py-vector-bench-") as tmp:
        db = LodeDB.open_vector_store(
            Path(tmp) / "vectors",
            vector_dim=dim,
            commit_mode="generation",
        )
        try:
            upsert = _measure(
                "python_vector_upsert",
                lambda: (
                    db.add_vectors_many(
                        [
                            {
                                "id": document.document_id,
                                "vector": document.vector,
                                "metadata": document.metadata,
                            }
                            for document in documents
                        ],
                        normalize=False,
                    )
                    and len(documents)
                ),
            )
            unfiltered_search = _measure(
                "python_vector_search_unfiltered",
                lambda: _hit_checksum(
                    [
                        [
                            hit.id
                            for hit in db.search_by_vector(
                                query,
                                k=k,
                                normalize=False,
                            )
                        ]
                        for query in queries
                    ]
                ),
            )
            filtered_search = _measure(
                "python_vector_search_filtered",
                lambda: _hit_checksum(
                    [
                        [
                            hit.id
                            for hit in db.search_by_vector(
                                query,
                                k=k,
                                filter={"bucket": str(position % 16)},
                                normalize=False,
                            )
                        ]
                        for position, query in enumerate(queries)
                    ]
                ),
            )
            batch_search = _measure(
                "python_vector_search_batch",
                lambda: _hit_checksum(
                    [
                        [
                            hit.id
                            for hit in db.search_by_vector(
                                query,
                                k=k,
                                normalize=False,
                            )
                        ]
                        for query in queries
                    ]
                ),
            )
            stats = db.stats()
            summary = {
                "document_count": int(stats.get("document_count", 0) or 0),
                "chunk_count": int(stats.get("chunk_count", 0) or 0),
            }
        finally:
            db.close()
    return upsert, unfiltered_search, filtered_search, batch_search, summary


def _rust_vector_bench(
    adapter: NativeCoreAdapter,
    documents: list[EngineVectorDocument],
    queries: list[list[float]],
    *,
    dim: int,
    k: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    engine = adapter.new_engine()
    engine.create_index(_INDEX_ID, vector_dim=dim, bit_width=4)
    try:
        upsert = _measure(
            "rust_vector_upsert",
            lambda: (
                engine.upsert_vectors(_INDEX_ID, documents)
                and len(documents)
            ),
        )
        unfiltered_search = _measure(
            "rust_vector_search_unfiltered",
            lambda: _hit_checksum(
                [
                    [
                        str(hit["document_id"])
                        for hit in engine.query_vector(
                            _INDEX_ID,
                            query,
                            top_k=k,
                            filter=None,
                        ).get("hits", [])
                    ]
                    for query in queries
                ]
            ),
        )
        filtered_search = _measure(
            "rust_vector_search_filtered",
            lambda: _hit_checksum(
                [
                    [
                        str(hit["document_id"])
                        for hit in engine.query_vector(
                            _INDEX_ID,
                            query,
                            top_k=k,
                            filter={"metadata": {"bucket": str(position % 16)}},
                        ).get("hits", [])
                    ]
                    for position, query in enumerate(queries)
                ]
            ),
        )
        batch_search = _measure(
            "rust_vector_search_batch",
            lambda: _hit_checksum(
                [
                    [str(hit["document_id"]) for hit in row.get("hits", [])]
                    for row in engine.query_vectors_batch(
                        _INDEX_ID,
                        queries,
                        top_k=k,
                        filter=None,
                    )
                ]
            ),
        )
        stats = engine.stats(_INDEX_ID)
        summary = {
            "document_count": int(stats.get("document_count", 0) or 0),
            "chunk_count": int(stats.get("chunk_count", 0) or 0),
        }
    finally:
        engine.close()
    return upsert, unfiltered_search, filtered_search, batch_search, summary


def _python_reopen_bench(
    documents: list[EngineVectorDocument],
    queries: list[list[float]],
    *,
    dim: int,
    k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="lodedb-py-reopen-bench-") as tmp:
        path = Path(tmp) / "vectors"
        db = LodeDB.open_vector_store(
            path,
            vector_dim=dim,
            commit_mode="generation",
        )
        try:
            db.add_vectors_many(
                [
                    {
                        "id": document.document_id,
                        "vector": document.vector,
                        "metadata": document.metadata,
                    }
                    for document in documents
                ],
                normalize=False,
            )
        finally:
            db.close()

        def reopen_and_query() -> int:
            reopened = LodeDB.open_vector_store(
                path,
                vector_dim=dim,
                commit_mode="generation",
                read_only=True,
            )
            try:
                return _hit_checksum(
                    [
                        [
                            hit.id
                            for hit in reopened.search_by_vector(
                                query,
                                k=k,
                                normalize=False,
                            )
                        ]
                        for query in queries
                    ]
                )
            finally:
                reopened.close()

        reopen = _measure("python_persisted_reopen_query", reopen_and_query)
        summary = {"document_count": len(documents), "chunk_count": len(documents)}
    return reopen, summary


def _rust_reopen_bench(
    adapter: NativeCoreAdapter,
    documents: list[EngineVectorDocument],
    queries: list[list[float]],
    *,
    dim: int,
    k: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with tempfile.TemporaryDirectory(prefix="lodedb-rust-reopen-bench-") as tmp:
        path = Path(tmp) / "vectors"
        engine = adapter.open_engine(
            path=path,
            read_only=False,
            durability="relaxed",
            commit_mode="generation",
            store_text=True,
            index_text=True,
        )
        try:
            engine.create_index(_INDEX_ID, vector_dim=dim, bit_width=4)
            engine.upsert_vectors(_INDEX_ID, documents)
            engine.persist()
        finally:
            engine.close()

        def reopen_and_query() -> int:
            reopened = adapter.open_readonly_engine(
                path,
                durability="relaxed",
                commit_mode="generation",
                store_text=True,
                index_text=True,
            )
            try:
                return _hit_checksum(
                    [
                        [
                            str(hit["document_id"])
                            for hit in reopened.query_vector(
                                _INDEX_ID,
                                query,
                                top_k=k,
                                filter=None,
                            ).get("hits", [])
                        ]
                        for query in queries
                    ]
                )
            finally:
                reopened.close()

        reopen = _measure("rust_persisted_reopen_query", reopen_and_query)
        summary = {"document_count": len(documents), "chunk_count": len(documents)}
    return reopen, summary


def _python_text_bench(documents: list[EngineDocument], queries: list[str], *, k: int) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    with tempfile.TemporaryDirectory(prefix="lodedb-py-text-bench-") as tmp:
        db = LodeDB(
            Path(tmp) / "text",
            index_text=True,
            commit_mode="generation",
            _embedding_backend=HashEmbeddingBackend(native_dim=384),
        )
        try:
            upsert = _measure(
                "python_text_upsert",
                lambda: (
                    db.add_many(
                        [
                            {
                                "id": document.document_id,
                                "text": document.text,
                                "metadata": document.metadata,
                            }
                            for document in documents
                        ]
                    )
                    and len(documents)
                ),
            )
            lexical = _measure(
                "python_text_lexical_search",
                lambda: _hit_checksum(
                    [
                        [hit.id for hit in db.search(query, k=k, mode="lexical")]
                        for query in queries
                    ]
                ),
            )
            hybrid = _measure(
                "python_text_hybrid_search",
                lambda: _hit_checksum(
                    [
                        [hit.id for hit in db.search(query, k=k, mode="hybrid")]
                        for query in queries
                    ]
                ),
            )
            stats = db.stats()
            summary = {
                "document_count": int(stats.get("document_count", 0) or 0),
                "chunk_count": int(stats.get("chunk_count", 0) or 0),
            }
        finally:
            db.close()
    return upsert, lexical, hybrid, summary


def _rust_text_bench(
    adapter: NativeCoreAdapter,
    documents: list[EngineDocument],
    queries: list[str],
    *,
    k: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    engine = adapter.new_engine()
    engine.create_index(_INDEX_ID, vector_dim=384, bit_width=4)
    embedder = HashEmbeddingBackend(native_dim=384)
    try:
        upsert = _measure(
            "rust_text_prepare_apply",
            lambda: _rust_apply_text_documents(engine, embedder, documents),
        )
        lexical = _measure(
            "rust_text_lexical_search",
            lambda: _hit_checksum(
                [
                    [
                        str(hit["document_id"])
                        for hit in engine.search_text(
                            _INDEX_ID,
                            query,
                            "lexical",
                            None,
                            top_k=k,
                        ).get("hits", [])
                    ]
                    for query in queries
                ]
            ),
        )
        hybrid = _measure(
            "rust_text_hybrid_search",
            lambda: _hit_checksum(
                [
                    [
                        str(hit["document_id"])
                        for hit in engine.search_text(
                            _INDEX_ID,
                            query,
                            "hybrid",
                            embedder.embed_query(query),
                            top_k=k,
                        ).get("hits", [])
                    ]
                    for query in queries
                ]
            ),
        )
        stats = engine.stats(_INDEX_ID)
        summary = {
            "document_count": int(stats.get("document_count", 0) or 0),
            "chunk_count": int(stats.get("chunk_count", 0) or 0),
        }
    finally:
        engine.close()
    return upsert, lexical, hybrid, summary


def _rust_apply_text_documents(
    engine: Any,
    embedder: HashEmbeddingBackend,
    documents: list[EngineDocument],
) -> int:
    plan = engine.prepare_text_upsert(
        _INDEX_ID,
        documents,
        store_text=True,
        index_text=True,
        chunk_character_limit=8192,
    )
    chunk_texts = [str(chunk.get("text", "")) for chunk in plan.get("chunks_to_embed", [])]
    embeddings = embedder.embed_documents(chunk_texts) if chunk_texts else ()
    engine.apply_text_upsert(plan, embeddings, embedding_time_ms=0.0)
    return len(documents)


def run(
    output: Path | None = None,
    *,
    documents: int = 2_000,
    queries: int = 200,
    dim: int = 64,
    k: int = 8,
) -> dict[str, Any]:
    adapter = _require_native_adapter()
    vector_documents = _vectors(documents, dim)
    query_vectors = _queries(queries, dim)
    text_documents = _text_documents(documents)
    text_queries = [f"E-{1000 + (i % 97)} vector recovery" for i in range(queries)]

    (
        py_vector_upsert,
        py_vector_unfiltered_search,
        py_vector_filtered_search,
        py_vector_batch_search,
        py_vector_stats,
    ) = _python_vector_bench(
        vector_documents,
        query_vectors,
        dim=dim,
        k=k,
    )
    (
        rust_vector_upsert,
        rust_vector_unfiltered_search,
        rust_vector_filtered_search,
        rust_vector_batch_search,
        rust_vector_stats,
    ) = _rust_vector_bench(
        adapter,
        vector_documents,
        query_vectors,
        dim=dim,
        k=k,
    )
    py_text_upsert, py_text_search, py_text_hybrid, py_text_stats = _python_text_bench(
        text_documents,
        text_queries,
        k=k,
    )
    rust_text_upsert, rust_text_search, rust_text_hybrid, rust_text_stats = _rust_text_bench(
        adapter,
        text_documents,
        text_queries,
        k=k,
    )
    py_reopen, py_reopen_stats = _python_reopen_bench(
        vector_documents,
        query_vectors,
        dim=dim,
        k=k,
    )
    rust_reopen, rust_reopen_stats = _rust_reopen_bench(
        adapter,
        vector_documents,
        query_vectors,
        dim=dim,
        k=k,
    )

    payload = {
        "suite": "native_migration_rust_vs_python",
        "parameters": {
            "documents": documents,
            "queries": queries,
            "dim": dim,
            "k": k,
        },
        "native_core": {
            "version": adapter.version,
            "abi_version": adapter.abi_version,
            "extension_override": bool(os.environ.get("LODEDB_NATIVE_CORE_EXTENSION_PATH")),
        },
        "comparisons": [
            _comparison("vector_upsert", py_vector_upsert, rust_vector_upsert),
            _comparison(
                "vector_search_unfiltered",
                py_vector_unfiltered_search,
                rust_vector_unfiltered_search,
            ),
            _comparison(
                "vector_search_filtered",
                py_vector_filtered_search,
                rust_vector_filtered_search,
            ),
            _comparison("vector_search_batch", py_vector_batch_search, rust_vector_batch_search),
            _comparison("text_upsert_hash_embedder", py_text_upsert, rust_text_upsert),
            _comparison("text_lexical_search", py_text_search, rust_text_search),
            _comparison("text_hybrid_search", py_text_hybrid, rust_text_hybrid),
            _comparison("persisted_reopen_query", py_reopen, rust_reopen),
        ],
        "summaries": {
            "python_vector": py_vector_stats,
            "rust_vector": rust_vector_stats,
            "python_text": py_text_stats,
            "rust_text": rust_text_stats,
            "python_reopen": py_reopen_stats,
            "rust_reopen": rust_reopen_stats,
        },
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--documents", type=int, default=2_000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output,
                documents=args.documents,
                queries=args.queries,
                dim=args.dim,
                k=args.k,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
