"""Memory-integration benchmark: LodeDB vs framework default/common stores.

Compares the LodeDB adapters for LangChain, LlamaIndex, and mem0 against each
framework's own vector-store backends (its in-memory default plus FAISS, Chroma,
and Qdrant) on realistic workflows (RAG ingest/retrieval for LangChain and
LlamaIndex, agent-memory accrual for mem0), measuring ingest throughput, query
latency, recall@k, durable on-disk footprint, the cost of a durable single-memory
add, and reopen. One fixed embedding model is held constant across every backend,
so the numbers isolate the store. Emits **metrics-only** JSON.

Local (synthetic, no network):

    python benchmarks/memory_integrations/run.py --max-documents 2000 --device cpu

Modal (GovReport at scale) is driven by ``modal_bench.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from common import embed_corpus, load_corpus, quiet_logging

_FRAMEWORKS = ("langchain", "llamaindex", "mem0")


def run_memory_integrations_suite(
    *,
    dataset_name: str = "synthetic",
    max_documents: int = 2000,
    query_count: int = 128,
    model: str = "minilm",
    device: str = "cpu",
    top_k: int = 10,
    incremental_count: int = 50,
    n_users: int = 20,
    batch_size: int = 64,
    frameworks: tuple[str, ...] = _FRAMEWORKS,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Runs the requested framework suites and returns one metrics-only bundle."""

    quiet_logging()
    documents, queries = load_corpus(dataset_name, max_documents, query_count)

    suites: dict[str, Any] = {}
    # The engine prints build/query progress to stdout; keep it off our result
    # stream (matches lodedb.local.benchmark). Timing is unaffected.
    with contextlib.redirect_stdout(sys.stderr):
        embedded = embed_corpus(documents, queries, model=model, device=device)
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            if "langchain" in frameworks:
                from bench_langchain import run_langchain_suite  # noqa: PLC0415

                suites["langchain"] = run_langchain_suite(
                    embedded,
                    documents,
                    model=model,
                    device=device,
                    k=top_k,
                    incremental_count=incremental_count,
                    batch_size=batch_size,
                    workdir=workdir / "langchain",
                )
            if "llamaindex" in frameworks:
                from bench_llamaindex import run_llamaindex_suite  # noqa: PLC0415

                suites["llamaindex"] = run_llamaindex_suite(
                    embedded,
                    documents,
                    model=model,
                    device=device,
                    k=top_k,
                    incremental_count=incremental_count,
                    batch_size=batch_size,
                    workdir=workdir / "llamaindex",
                )
            if "mem0" in frameworks:
                from bench_mem0 import run_mem0_suite  # noqa: PLC0415

                suites["mem0"] = run_mem0_suite(
                    embedded,
                    n_users=n_users,
                    k=top_k,
                    incremental_count=incremental_count,
                    batch_size=batch_size,
                    workdir=workdir / "mem0",
                )

    bundle = {
        "suite": "memory_integrations",
        "provenance": "measured",
        "dataset": dataset_name,
        "model": model,
        "device": device,
        "document_count": len(documents),
        "query_count": len(queries),
        "top_k": top_k,
        "incremental_count": incremental_count,
        "batch_size": batch_size,
        "embedding": {
            "native_dim": embedded.native_dim,
            "effective_device": embedded.effective_device,
            "doc_embed_ms_warm": round(embedded.doc_embed_ms, 2),
            "query_embed_ms_warm": round(embedded.query_embed_ms, 2),
            "note": "one fixed model across all backends; baselines receive these vectors, "
            "LodeDB text-path adapters re-embed and the runner subtracts this time for store-only",
        },
        "suites": suites,
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

    parser = argparse.ArgumentParser(description="LodeDB memory-integration benchmark")
    parser.add_argument("--dataset", default="synthetic")
    parser.add_argument("--max-documents", type=int, default=2000)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--model", default="minilm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--incremental-count", type=int, default=50)
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--frameworks",
        default=",".join(_FRAMEWORKS),
        help="comma-separated subset of: langchain,llamaindex,mem0",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    bundle = run_memory_integrations_suite(
        dataset_name=args.dataset,
        max_documents=args.max_documents,
        query_count=args.query_count,
        model=args.model,
        device=args.device,
        top_k=args.top_k,
        incremental_count=args.incremental_count,
        n_users=args.n_users,
        batch_size=args.batch_size,
        frameworks=tuple(f.strip() for f in args.frameworks.split(",") if f.strip()),
        output_dir=args.out,
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
