"""LodeDB native GPU-resident batch sweep.

Builds one local LodeDB index, then compares the native CPU TurboVec scan against
the optional native CUDA GPU-resident scan across query batch sizes. The native
core serves an unfiltered batch from the GPU once the batch and corpus clear the
engine's thresholds and a CUDA driver loads; `LODEDB_GPU_DIRECT_TURBOVEC=off`
forces the CPU scan, any other value (the default) leaves the GPU path eligible.
Queries run through the public `LodeDB` API, so this drives exactly the path an
application gets. The output is raw-payload-free: only ids, counts, timings,
recall/overlap metrics, and a redacted audit status are written.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from lodedb.engine.core import audit_persisted_index_snapshots
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.turbovec_index import turbovec_capability
from lodedb.local import LodeDB
from lodedb.local.presets import resolve_preset

# The Rust core reads this per scan; "off" disables the GPU-resident scan, any
# other value (or unset) leaves it eligible, still subject to the CUDA driver and
# the batch/corpus gates in the engine.
_GPU_SCAN_ENV = "LODEDB_GPU_DIRECT_TURBOVEC"

DEFAULT_BATCH_SIZES = "1,2,4,8,16,32,64,128,256,512,1024"
GOVREPORT_DATASET = "ccdv/govreport-summarization"
GOVREPORT_CHUNK_CHARACTER_LIMIT = 480
QREL_RECALL_PARITY_TOLERANCE = 0.002


@contextmanager
def _gpu_scan(enabled: bool):
    """Toggles the native GPU-resident scan for the duration of the block.

    Flips ``LODEDB_GPU_DIRECT_TURBOVEC`` (the Rust core reads it per scan), so the
    CPU and GPU rows share one built index without a rebuild, and restores the
    prior value on exit.
    """

    previous = os.environ.get(_GPU_SCAN_ENV)
    os.environ[_GPU_SCAN_ENV] = "auto" if enabled else "off"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_GPU_SCAN_ENV, None)
        else:
            os.environ[_GPU_SCAN_ENV] = previous


def run_direct_gpu_sweep(
    *,
    output_dir: str | Path,
    dataset_name: str = "GovReport5K",
    model: str = "minilm",
    max_documents: int | None = None,
    query_count: int = 64,
    top_k: int = 100,
    batch_sizes: str = DEFAULT_BATCH_SIZES,
    query_repeats: int = 5,
    device: str = "cuda",
    use_hash_backend: bool = False,
    persistence_root: str | Path | None = None,
    artifact_checkpoint: Callable[[], None] | None = None,
    dataset_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Runs the native CPU/GPU batch sweep and writes a JSON summary."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    batch_size_values = _parse_batch_sizes(batch_sizes)
    if query_count < max(batch_size_values):
        raise ValueError("query_count must be at least the largest requested batch size")

    dataset = (
        _normalize_dataset_override(dataset_override)
        if dataset_override is not None
        else _load_govreport_dataset(
            dataset_name=dataset_name,
            max_documents=max_documents,
            query_count=query_count,
        )
    )
    if len(dataset["queries"]) < query_count:
        raise ValueError("dataset did not provide enough queries")
    documents = list(dataset["documents"])
    queries = list(dataset["queries"][:query_count])

    cleanup_root: tempfile.TemporaryDirectory[str] | None = None
    if persistence_root is None:
        cleanup_root = tempfile.TemporaryDirectory(prefix="lodedb-direct-gpu-sweep-")
        persistence_dir = Path(cleanup_root.name)
    else:
        persistence_dir = Path(persistence_root)
    try:
        summary = _run_sweep(
            output=output,
            dataset_name=str(dataset["name"]),
            documents=documents,
            queries=queries,
            model=model,
            device=device,
            persistence_dir=persistence_dir,
            top_k=top_k,
            batch_size_values=batch_size_values,
            query_repeats=query_repeats,
            use_hash_backend=use_hash_backend,
            artifact_checkpoint=artifact_checkpoint,
        )
    finally:
        if cleanup_root is not None:
            cleanup_root.cleanup()
    return summary


def _run_sweep(
    *,
    output: Path,
    dataset_name: str,
    documents: list[dict[str, Any]],
    queries: list[dict[str, str]],
    model: str,
    device: str,
    persistence_dir: Path,
    top_k: int,
    batch_size_values: tuple[int, ...],
    query_repeats: int,
    use_hash_backend: bool,
    artifact_checkpoint: Callable[[], None] | None,
) -> dict[str, Any]:
    """Builds one LodeDB and executes all sweep rows through the public API."""

    persistence_dir.mkdir(parents=True, exist_ok=True)
    preset = resolve_preset(model)
    backend = HashEmbeddingBackend(native_dim=preset.native_dim) if use_hash_backend else None
    db = LodeDB(
        path=persistence_dir,
        model=model,
        device=device,
        chunk_character_limit=GOVREPORT_CHUNK_CHARACTER_LIMIT,
        _embedding_backend=backend,
    )
    try:
        started = time.perf_counter()
        db.add_many(documents)
        build_seconds = time.perf_counter() - started
        db.persist()
        stats = db.stats()
        chunk_count = int(stats.get("chunk_count", 0) or 0)

        rows: list[dict[str, Any]] = []
        cpu_results_by_batch: dict[int, list[list[str]]] = {}

        def checkpoint(row: dict[str, Any]) -> None:
            rows.append(row)
            rows_dir = output / "rows"
            rows_dir.mkdir(parents=True, exist_ok=True)
            (rows_dir / f"{len(rows):02d}-{row['row']}.json").write_text(
                json.dumps(row, indent=2, sort_keys=True), encoding="utf-8"
            )
            if artifact_checkpoint is not None:
                artifact_checkpoint()

        for batch_size in batch_size_values:
            cpu_row = _query_row(
                db,
                queries=queries,
                batch_size=batch_size,
                top_k=top_k,
                repeats=query_repeats,
                gpu=False,
                label=f"cpu_batch_{batch_size}",
            )
            cpu_results_by_batch[batch_size] = cpu_row.pop("_served_document_ids")
            checkpoint(cpu_row)

            gpu_row = _query_row(
                db,
                queries=queries,
                batch_size=batch_size,
                top_k=top_k,
                repeats=query_repeats,
                gpu=True,
                label=f"gpu_batch_{batch_size}",
            )
            gpu_served = gpu_row.pop("_served_document_ids")
            gpu_row["gpu_vs_cpu_top_k_overlap"] = _mean_overlap(
                cpu_results_by_batch[batch_size], gpu_served
            )
            recall_gap = abs(
                float(gpu_row["document_recall_at_top_k"])
                - float(cpu_row["document_recall_at_top_k"])
            )
            gpu_row["document_recall_gap_vs_cpu"] = recall_gap
            # The GPU scan must rank identically to the CPU scan (it reads back the
            # same reconstructed rows); a divergence is a correctness bug, not noise.
            if recall_gap > QREL_RECALL_PARITY_TOLERANCE:
                raise RuntimeError(
                    f"GPU document recall diverged from CPU at batch {batch_size}: "
                    f"gap {recall_gap:.6f} exceeds {QREL_RECALL_PARITY_TOLERANCE}"
                )
            checkpoint(gpu_row)

        audit = audit_persisted_index_snapshots(persistence_dir)
        summary = {
            "artifact_type": "lodedb_direct_turbovec_gpu_sweep",
            "dataset_name": dataset_name,
            "model": model,
            "document_count": len(documents),
            "chunk_count": chunk_count,
            "query_count": len(queries),
            "top_k": int(top_k),
            "batch_sizes": list(batch_size_values),
            "build_seconds": float(build_seconds),
            "turbovec_capability": turbovec_capability().to_dict(),
            "row_count": len(rows),
            "rows": rows,
            "audit_status": audit["status"],
            "raw_payload_text_present": False,
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        if artifact_checkpoint is not None:
            artifact_checkpoint()
        return summary
    finally:
        db.close()


def _query_row(
    db: LodeDB,
    *,
    queries: list[dict[str, str]],
    batch_size: int,
    top_k: int,
    repeats: int,
    gpu: bool,
    label: str,
) -> dict[str, Any]:
    """Measures one CPU-or-GPU/batch-size cell over the evaluation query set."""

    batches = [queries[start : start + batch_size] for start in range(0, len(queries), batch_size)]
    batch_ms: list[float] = []
    served: list[list[str]] = []
    with _gpu_scan(gpu):
        _run_query_batch(db, batches[0], top_k=top_k)  # warm the layout / resident upload
        for repeat in range(max(1, repeats)):
            repeat_served: list[list[str]] = []
            for batch in batches:
                started = time.perf_counter()
                results = _run_query_batch(db, batch, top_k=top_k)
                batch_ms.append((time.perf_counter() - started) * 1000.0)
                if repeat == 0:
                    repeat_served.extend([hit.id for hit in hits] for hits in results)
            if repeat == 0:
                served = repeat_served
    return {
        "row": label,
        "gpu_enabled": bool(gpu),
        "batch_size": int(batch_size),
        "repeats": int(max(1, repeats)),
        "query_count": len(queries),
        "batch_p50_ms": float(np.percentile(batch_ms, 50)),
        "batch_p95_ms": float(np.percentile(batch_ms, 95)),
        "document_recall_at_top_k": _document_recall(queries, served),
        "_served_document_ids": served,
    }


def _run_query_batch(
    db: LodeDB, queries: list[dict[str, str]], *, top_k: int
) -> list[list[Any]]:
    """Runs one batch of vector queries through the public batched search API."""

    return db.search_many([query["text"] for query in queries], k=int(top_k), mode="vector")


def _document_recall(queries: list[dict[str, str]], served: list[list[str]]) -> float:
    """Returns query-document recall against source-document labels."""

    if not queries:
        return 0.0
    hits = 0
    for query, result_ids in zip(queries, served, strict=True):
        if query["document_id"] in set(result_ids):
            hits += 1
    return hits / len(queries)


def _mean_overlap(left: list[list[str]], right: list[list[str]]) -> float:
    """Returns mean top-k set overlap for paired result rows."""

    if not left:
        return 0.0
    scores: list[float] = []
    for a, b in zip(left, right, strict=True):
        denom = max(1, min(len(a), len(b)))
        scores.append(len(set(a).intersection(b)) / denom)
    return float(np.mean(scores))


def _parse_batch_sizes(value: str) -> tuple[int, ...]:
    """Parses and validates comma-separated positive batch sizes."""

    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("batch_sizes must contain positive integers")
    return sizes


def _normalize_dataset_override(dataset: Mapping[str, Any]) -> dict[str, Any]:
    """Validates the offline test dataset shape."""

    documents = [dict(item) for item in dataset.get("documents", [])]
    queries = [dict(item) for item in dataset.get("queries", [])]
    if not documents or not queries:
        raise ValueError("dataset_override needs non-empty documents and queries")
    return {
        "name": str(dataset.get("name", "override")),
        "documents": documents,
        "queries": queries,
    }


def _load_govreport_dataset(
    *,
    dataset_name: str,
    max_documents: int | None,
    query_count: int,
) -> dict[str, Any]:
    """Loads a GovReport-shaped dataset with source-document relevance labels."""

    from datasets import load_dataset

    limit = max(int(max_documents or _documents_from_dataset_name(dataset_name)), query_count)
    documents: list[dict[str, Any]] = []
    queries: list[dict[str, str]] = []
    seen = 0
    for split in ("train", "validation", "test"):
        if len(documents) >= limit and len(queries) >= query_count:
            break
        rows = load_dataset(GOVREPORT_DATASET, split=split, streaming=True)
        for row in rows:
            if len(documents) >= limit and len(queries) >= query_count:
                break
            report = str(row.get("report", "")).strip()
            summary = str(row.get("summary", "")).strip()
            if not report or not summary:
                continue
            document_id = f"govreport-{seen:06d}"
            seen += 1
            if len(documents) < limit:
                documents.append(
                    {
                        "id": document_id,
                        "text": report,
                        "metadata": {"dataset": dataset_name},
                    }
                )
            if len(queries) < query_count:
                queries.append(
                    {
                        "id": f"query-{len(queries):06d}",
                        "text": summary,
                        "document_id": document_id,
                    }
                )
    return {"name": dataset_name, "documents": documents, "queries": queries}


def _documents_from_dataset_name(name: str) -> int:
    """Infers a GovReport document limit from labels like GovReport5K."""

    label = name.strip().lower().removeprefix("govreport")
    if label.endswith("k") and label[:-1].isdigit():
        return int(label[:-1]) * 1000
    if label.isdigit():
        return int(label)
    return 5000


def main() -> None:
    """CLI entry point for direct CUDA hosts."""

    import argparse

    parser = argparse.ArgumentParser(description="Run LodeDB native GPU sweep")
    parser.add_argument("--out", default="benchmarks/direct_gpu_sweep/results/results.json")
    parser.add_argument("--dataset", default="GovReport5K")
    parser.add_argument("--model", default="minilm", choices=["minilm", "bge"])
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--query-count", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-sizes", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--query-repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary_path = Path(args.out)
    summary = run_direct_gpu_sweep(
        output_dir=summary_path.parent,
        dataset_name=args.dataset,
        model=args.model,
        max_documents=args.max_documents,
        query_count=args.query_count,
        top_k=args.top_k,
        batch_sizes=args.batch_sizes,
        query_repeats=args.query_repeats,
        device=args.device,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
