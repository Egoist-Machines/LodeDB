"""Engine-independent measurement harness for wiki_dpr_e5 disk-rescore runs.

The corpus and exact fp32 reference are prepared by :mod:`data_prep`.  This
module never recomputes brute-force truth while serving.  It drives only the
vector-in LodeDB API, so the measured path contains neither model loading nor
document text retention.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import GT_K, canonical_sha256, dataset_arrays
except ImportError:  # Direct execution from this directory.
    from common import GT_K, canonical_sha256, dataset_arrays  # type: ignore[no-redef]


class EngineFeatureRequired(RuntimeError):
    """Raised when an opt-in benchmark switch needs the companion engine branch."""


def _requires_engine(feature: str) -> EngineFeatureRequired:
    return EngineFeatureRequired("installed LodeDB build does not provide: " + feature)


_STORE_CONFIG_NAME = "benchmark_store_config.json"
_STORE_CONFIG_VERSION = 2
_RESCORE_DTYPES = {"fp16": "float16", "fp32": "float32"}
_EXACT_LAYOUT_ID = "turbovec-exact-v1"
_CLUSTER_LAYOUT_ID = "cluster-contiguous-v1"


def _summary_ms(samples_ms: list[float]) -> dict[str, float | int]:
    """Returns count, p50, p95, and mean for millisecond samples."""

    if not samples_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "count": 0}
    ordered = sorted(samples_ms)
    return {
        "p50_ms": float(ordered[len(ordered) // 2]),
        "p95_ms": float(ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]),
        "mean_ms": float(statistics.fmean(samples_ms)),
        "count": len(samples_ms),
    }


def _dir_footprint(path: Path) -> dict[str, Any]:
    """Returns total bytes and a per-extension breakdown below ``path``."""

    total = 0
    by_extension: dict[str, int] = {}
    for file in path.rglob("*"):
        if not file.is_file():
            continue
        if file.name == _STORE_CONFIG_NAME:
            continue
        size = int(file.stat().st_size)
        total += size
        suffix = file.suffix or "<none>"
        by_extension[suffix] = by_extension.get(suffix, 0) + size
    return {
        "total_bytes": total,
        "by_extension_bytes": dict(sorted(by_extension.items())),
    }


def _store_config(
    *,
    bit_width: int,
    ann_clusters: int | None,
    ann_nprobe: int | None,
    rescore: str,
    oversample: float,
    rows: int,
    dim: int,
    corpus_id: str,
    builder_git_sha: str,
    builder_lodedb_version: str,
    layout_id: str,
) -> dict[str, Any]:
    """Returns the benchmark-level configuration persisted alongside a store."""

    config: dict[str, Any] = {
        "schema_version": _STORE_CONFIG_VERSION,
        "corpus_id": corpus_id,
        "create": {
            "bit_width": bit_width,
            "ann_clusters": ann_clusters,
            "ann_nprobe": ann_nprobe,
            "rescore": rescore,
            "oversample": oversample,
            "rows": rows,
            "dim": dim,
        },
        "builder": {
            "git_sha": builder_git_sha,
            "lodedb_version": builder_lodedb_version,
            "layout_id": layout_id,
        },
    }
    config["store_id"] = canonical_sha256(config)
    return config


def _requested_store_create(
    *,
    bit_width: int,
    ann_clusters: int | None,
    ann_nprobe: int | None,
    rescore: str,
    oversample: float,
    rows: int,
    dim: int,
) -> dict[str, Any]:
    return {
        "bit_width": bit_width,
        "ann_clusters": ann_clusters,
        "ann_nprobe": ann_nprobe,
        "rescore": rescore,
        "oversample": oversample,
        "rows": rows,
        "dim": dim,
    }


def _load_store_config(store_dir: Path) -> dict[str, Any]:
    """Loads the configuration required to safely reuse an existing store."""

    config_path = store_dir / _STORE_CONFIG_NAME
    try:
        loaded = json.loads(config_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(
            f"existing store is missing {config_path}; rebuild the store before benchmarking"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid benchmark store configuration: {config_path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"benchmark store configuration must be an object: {config_path}")
    if loaded.get("schema_version") != _STORE_CONFIG_VERSION:
        raise ValueError(f"unsupported benchmark store config at {config_path}; rebuild the store")
    if not isinstance(loaded.get("create"), dict) or not isinstance(loaded.get("builder"), dict):
        raise ValueError(f"benchmark store provenance is incomplete: {config_path}")
    stored_id = loaded.get("store_id")
    without_id = {key: value for key, value in loaded.items() if key != "store_id"}
    if not isinstance(stored_id, str) or stored_id != canonical_sha256(without_id):
        raise ValueError(f"benchmark store config identity is invalid: {config_path}")
    if not isinstance(loaded.get("corpus_id"), str):
        raise ValueError(f"benchmark store corpus identity is missing: {config_path}")
    builder = loaded["builder"]
    for field in ("git_sha", "lodedb_version", "layout_id"):
        if not isinstance(builder.get(field), str) or not builder[field]:
            raise ValueError(f"benchmark store builder.{field} is missing: {config_path}")
    return loaded


def _supports_session_overrides() -> bool:
    """Returns whether this LodeDB build can apply reopen-time session knobs."""

    from lodedb.local.db import LodeDB

    # The companion engine branch adds rescore_oversample at the same time as
    # applying both knobs to the native handle on reopen. Older releases accept
    # ann_nprobe but ignore it for an existing store, so using this capability
    # signal keeps an override benchmark from producing mislabeled results.
    return "rescore_oversample" in inspect.signature(LodeDB).parameters


def _guard_existing_store_config(
    store_dir: Path,
    requested_create: dict[str, Any],
    *,
    corpus_id: str,
    serve_nprobe: int | None,
    serve_oversample: float | None,
    expected_layout_id: str | None = None,
    expected_builder_git_sha: str | None = None,
) -> dict[str, Any]:
    """Rejects a store built for a different corpus or create-time configuration."""

    persisted = _load_store_config(store_dir)

    if persisted["corpus_id"] != corpus_id:
        raise ValueError(
            f"existing store {store_dir} belongs to corpus {persisted['corpus_id']!r}, "
            f"requested {corpus_id!r}; rebuild the store"
        )
    create = persisted["create"]
    for field, requested in requested_create.items():
        if type(create.get(field)) is not type(requested) or create.get(field) != requested:
            raise ValueError(
                f"existing store {store_dir} has {field}={create.get(field)!r}, "
                f"requested {requested!r}; rebuild the store"
            )
    builder = persisted["builder"]
    if expected_layout_id is not None and builder["layout_id"] != expected_layout_id:
        raise ValueError(
            f"existing store {store_dir} has layout {builder['layout_id']!r}, "
            f"expected {expected_layout_id!r}"
        )
    if expected_builder_git_sha is not None and builder["git_sha"] != expected_builder_git_sha:
        raise ValueError(
            f"existing store {store_dir} was built by {builder['git_sha']!r}, "
            f"expected {expected_builder_git_sha!r}"
        )
    effective_nprobe = create["ann_nprobe"] if serve_nprobe is None else serve_nprobe
    if effective_nprobe != create["ann_nprobe"] and not _supports_session_overrides():
        raise _requires_engine("open-time ann_nprobe override")
    effective_oversample = (
        create["oversample"] if serve_oversample is None else serve_oversample
    )
    if effective_oversample != create["oversample"] and not _supports_session_overrides():
        raise _requires_engine("open-time rescore oversample override")
    return persisted


def _write_store_config(store_dir: Path, config: dict[str, Any]) -> None:
    """Writes the benchmark configuration after a successful store build."""

    config_path = store_dir / _STORE_CONFIG_NAME
    temporary = store_dir / f".{_STORE_CONFIG_NAME}.tmp"
    temporary.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    temporary.replace(config_path)


def _git_sha() -> str | None:
    """Returns the checked-out source revision when Git metadata is available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _environment() -> dict[str, Any]:
    """Collects small, non-payload environment details for result provenance."""

    import lodedb

    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "lodedb_version": getattr(lodedb, "__version__", "unknown"),
        "git_sha": _git_sha(),
        "numpy": np.__version__,
    }


def _open_store(
    path: Path,
    *,
    dim: int,
    bit_width: int,
    ann_clusters: int | None,
    ann_nprobe: int | None,
    rescore: str,
    oversample: float,
    rows: int,
    corpus_id: str,
    serve_nprobe: int | None = None,
    serve_oversample: float | None = None,
    expected_layout_id: str | None = None,
    expected_builder_git_sha: str | None = None,
) -> Any:
    """Opens a pure vector LodeDB store, guarding future rescore keywords."""

    from lodedb.local.db import LodeDB

    if (ann_clusters is None) != (ann_nprobe is None):
        raise ValueError("--ann-clusters and --ann-nprobe must be supplied together")
    kwargs: dict[str, Any] = {
        "vector_dim": dim,
        "bit_width": bit_width,
        "store_text": False,
        "index_text": False,
    }
    if ann_clusters is not None:
        kwargs.update(
            {
                "ann": "cluster",
                "ann_clusters": ann_clusters,
                "ann_nprobe": ann_nprobe,
            }
        )
    if rescore != "none":
        # These are intentionally feature-gated. The engine branch adds both
        # create-time options; current releases reject the keywords cleanly.
        kwargs["rescore"] = "original"
        kwargs["rescore_dtype"] = _RESCORE_DTYPES[rescore]
        kwargs["rescore_oversample"] = oversample
    existing_store = path.exists() and any(path.iterdir())
    requested_create = _requested_store_create(
        bit_width=bit_width,
        ann_clusters=ann_clusters,
        ann_nprobe=ann_nprobe,
        rescore=rescore,
        oversample=oversample,
        rows=rows,
        dim=dim,
    )
    if existing_store:
        _guard_existing_store_config(
            path,
            requested_create,
            corpus_id=corpus_id,
            serve_nprobe=serve_nprobe,
            serve_oversample=serve_oversample,
            expected_layout_id=expected_layout_id,
            expected_builder_git_sha=expected_builder_git_sha,
        )
        if serve_nprobe is not None:
            kwargs["ann_nprobe"] = serve_nprobe
        if serve_oversample is not None:
            kwargs["rescore_oversample"] = serve_oversample
    try:
        db = LodeDB(path, **kwargs)
    except TypeError as exc:
        if rescore != "none":
            raise _requires_engine("rescore dtype and oversample options") from exc
        raise
    return db


def _ingest_vectors(db: Any, vectors: np.ndarray, *, batch: int) -> tuple[float, str]:
    """Ingests normalized memmap rows without converting vectors to Python lists."""

    if batch < 1:
        raise ValueError("ingest batch must be positive")
    import hashlib

    digest = hashlib.sha256()
    started = time.perf_counter()
    for start in range(0, vectors.shape[0], batch):
        stop = min(start + batch, vectors.shape[0])
        matrix = np.ascontiguousarray(vectors[start:stop], dtype="<f4")
        digest.update(matrix.tobytes(order="C"))
        payload = [
            {"id": str(index), "vector": matrix[index - start]}
            for index in range(start, stop)
        ]
        db.add_vectors_many(payload, normalize=False)
    return time.perf_counter() - started, digest.hexdigest()


def _hit_indices(hits: list[Any]) -> set[int]:
    """Converts benchmark's numeric document ids back into corpus row indices."""

    try:
        return {int(hit.id) for hit in hits}
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("LodeDB returned a non-numeric benchmark document id") from exc


def _recall_at_100(db: Any, queries: np.ndarray, gt_indices: np.ndarray, *, k: int) -> float:
    """Measures overlap with precomputed fp32 top-100, never recomputing truth."""

    total = 0.0
    for index, query in enumerate(queries):
        truth = set(int(value) for value in gt_indices[index, :GT_K])
        served = _hit_indices(db.search_by_vector(query, k=k, normalize=False)[:GT_K])
        total += len(truth & served) / GT_K
    return total / len(queries) if len(queries) else 0.0


def _sequential_latency(db: Any, queries: np.ndarray, *, k: int) -> dict[str, float | int]:
    """Measures each prepared query once after a 10-query warmup."""

    for query in queries[: min(10, len(queries))]:
        db.search_by_vector(query, k=k, normalize=False)
    samples: list[float] = []
    for query in queries:
        started = time.perf_counter()
        db.search_by_vector(query, k=k, normalize=False)
        samples.append((time.perf_counter() - started) * 1000.0)
    return _summary_ms(samples)


def _closed_loop(
    db: Any,
    queries: np.ndarray,
    *,
    k: int,
    seconds: float,
    concurrency: int,
    seed: int,
) -> dict[str, Any]:
    """Runs shuffled, closed-loop request streams over one shared store handle."""

    if seconds <= 0 or concurrency < 1:
        raise ValueError("loop seconds and loop concurrency must be positive")
    if not len(queries):
        raise ValueError("closed-loop measurement needs at least one query")
    barrier = threading.Barrier(concurrency + 1)
    samples_by_worker: list[list[float]] = [[] for _ in range(concurrency)]
    worker_errors: list[BaseException] = []
    worker_errors_lock = threading.Lock()

    def worker(worker_index: int) -> None:
        try:
            rng = np.random.default_rng(seed + worker_index + 1)
            order = rng.permutation(len(queries))
            position = 0
            barrier.wait()
            while time.perf_counter() < deadline:
                query = queries[order[position]]
                position += 1
                if position == len(order):
                    order = rng.permutation(len(queries))
                    position = 0
                started = time.perf_counter()
                db.search_by_vector(query, k=k, normalize=False)
                samples_by_worker[worker_index].append((time.perf_counter() - started) * 1000.0)
        except BaseException as exc:
            with worker_errors_lock:
                worker_errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(concurrency)]
    for thread in threads:
        thread.start()
    deadline = time.perf_counter() + seconds
    started = deadline - seconds
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        pass
    for thread in threads:
        thread.join()
    if worker_errors:
        raise worker_errors[0]
    elapsed = time.perf_counter() - started
    samples = [sample for worker_samples in samples_by_worker for sample in worker_samples]
    summary = _summary_ms(samples)
    return {
        "concurrency": concurrency,
        "seconds": elapsed,
        "qps": len(samples) / elapsed if elapsed else 0.0,
        "latency_ms": {"p50_ms": summary["p50_ms"], "p95_ms": summary["p95_ms"]},
    }


def _batched_qps(db: Any, queries: np.ndarray, *, k: int) -> list[dict[str, float | int]]:
    """Measures array-returning batch throughput at the requested comparison widths."""

    results: list[dict[str, float | int]] = []
    for batch_size in (64, 256, 1000):
        repeat = int(math.ceil(batch_size / len(queries)))
        batch = np.ascontiguousarray(np.tile(queries, (repeat, 1))[:batch_size], dtype=np.float32)
        db.search_many_by_vector_arrays(batch, k=k, normalize=False)
        repeats = max(1, 1024 // batch_size)
        started = time.perf_counter()
        for _ in range(repeats):
            db.search_many_by_vector_arrays(batch, k=k, normalize=False)
        elapsed = time.perf_counter() - started
        total = batch_size * repeats
        results.append({"batch_size": batch_size, "qps": total / elapsed if elapsed else 0.0})
    return results


def _block_skip_api() -> tuple[Any, Any]:
    """Finds the feature-branch counter exported through the extension module."""

    try:
        from lodedb import _turbovec
    except ImportError as exc:
        raise _requires_engine("block-skip counter export") from exc
    reset = getattr(_turbovec, "reset_blocks_skipped_by_mask", None)
    read = getattr(_turbovec, "blocks_skipped_by_mask", None)
    if not callable(reset) or not callable(read):
        raise _requires_engine("block-skip counter export")
    return reset, read


def validate_result_schema(result: dict[str, Any]) -> None:
    """Performs a dependency-free check of this benchmark's result JSON contract."""

    if result.get("schema_version") != 2:
        raise ValueError("result uses an unsupported or legacy-unverified schema")
    for key in ("label", "env", "dataset", "store", "measurement", "run_id", "serve"):
        if key not in result:
            raise ValueError(f"result is missing {key!r}")
    dataset = result["dataset"]
    for key in ("corpus_id", "source_revision"):
        if not isinstance(dataset.get(key), str) or not dataset[key]:
            raise ValueError(f"result dataset is missing {key!r}")
    evaluation_id = dataset.get("evaluation_id")
    if evaluation_id is None:
        if dataset.get("gt") is not None:
            raise ValueError("build-only result ground truth and evaluation_id must agree")
    elif not isinstance(evaluation_id, str) or dataset.get("gt") != "fp32-exact-top100":
        raise ValueError("result evaluation_id must identify the fp32 exact top-100 reference")
    store = result["store"]
    for key in (
        "bit_width",
        "ann",
        "rescore",
        "serve_overrides",
        "layout",
        "build",
        "footprint",
        "open",
        "provenance",
    ):
        if key not in store:
            raise ValueError(f"result store is missing {key!r}")
    serve = result["serve"]
    if serve is not None:
        if dataset.get("gt") != "fp32-exact-top100" or not isinstance(
            dataset.get("evaluation_id"), str
        ):
            raise ValueError("serve results require a committed fp32 exact top-100 evaluation")
        for key in (
            "effective_nprobe",
            "effective_oversample",
            "recall_at_100",
            "sequential_latency_ms",
            "closed_loop",
            "batched",
            "block_skip",
        ):
            if key not in serve:
                raise ValueError(f"result serve is missing {key!r}")
    provenance = store["provenance"]
    if not isinstance(provenance, dict) or not isinstance(provenance.get("store_id"), str):
        raise ValueError("result store provenance is missing its store_id")
    run_spec = {
        "label": result["label"],
        "evaluation_id": dataset["evaluation_id"],
        "store_id": provenance["store_id"],
        "measurement": result["measurement"],
        "serve_overrides": store["serve_overrides"],
    }
    if result["run_id"] != canonical_sha256(run_spec):
        raise ValueError("result run_id does not match its dataset, store, and measurement")


def load_result_for_resume(
    path: str | Path,
    *,
    label: str,
    evaluation_id: str,
    store_id: str,
    measurement: dict[str, Any],
    serve_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Loads one result only when it exactly matches the requested run."""

    path = Path(path)
    try:
        result = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"result cannot be resumed: {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"result root must be an object: {path}")
    validate_result_schema(result)
    expected = canonical_sha256(
        {
            "label": label,
            "evaluation_id": evaluation_id,
            "store_id": store_id,
            "measurement": measurement,
            "serve_overrides": serve_overrides,
        }
    )
    if result["run_id"] != expected:
        raise ValueError(f"result provenance does not match the requested run: {path}")
    return result


def write_result_atomic(path: str | Path, result: dict[str, Any]) -> None:
    """Validates and atomically publishes one complete benchmark result."""

    validate_result_schema(result)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_benchmark(
    *,
    data_dir: str | Path,
    store_dir: str | Path,
    label: str,
    bit_width: int = 4,
    ann_clusters: int | None = None,
    ann_nprobe: int | None = None,
    loop_seconds: float = 20.0,
    loop_concurrency: int = 4,
    k: int = 100,
    ingest_batch: int = 8192,
    build: bool = True,
    serve: bool = True,
    compact: bool = False,
    rescore: str = "none",
    oversample: float = 4.0,
    serve_nprobe: int | None = None,
    serve_oversample: float | None = None,
    report_block_skips: bool = False,
    builder_git_sha: str | None = None,
    layout_id: str | None = None,
    buildable: bool = True,
    expected_builder_git_sha: str | None = None,
) -> dict[str, Any]:
    """Builds and/or serves one exact or cluster-pruned vector store configuration."""

    if k < GT_K:
        raise ValueError("--k must be at least 100 because this harness reports recall@100")
    if rescore not in {"none", "fp16", "fp32"}:
        raise ValueError("rescore must be none, fp16, or fp32")
    if oversample <= 0:
        raise ValueError("oversample must be positive")
    if serve_nprobe is not None:
        if ann_clusters is None:
            raise ValueError("serve_nprobe requires an ANN store")
        if serve_nprobe < 1:
            raise ValueError("serve_nprobe must be positive")
    if serve_oversample is not None:
        if rescore == "none":
            raise ValueError("serve_oversample requires a rescore store")
        if serve_oversample <= 0:
            raise ValueError("serve_oversample must be positive")
    if compact and not build:
        raise _requires_engine("compact() is a build-time operation")
    if build and not buildable:
        raise ValueError(
            f"store {label!r} is a historical artifact and cannot be built by the current engine"
        )
    manifest, base, queries, gt_indices, _ = dataset_arrays(data_dir, require_gt=serve)
    if serve and gt_indices is None:
        raise ValueError("serve measurements need exact ground truth")
    store_dir = Path(store_dir)
    corpus_id = str(manifest["corpus_id"])
    evaluation_id = manifest.get("evaluation_id")
    if serve and not isinstance(evaluation_id, str):
        raise ValueError("serve measurements require a committed evaluation_id")
    layout_id = layout_id or (
        _EXACT_LAYOUT_ID if ann_clusters is None else _CLUSTER_LAYOUT_ID
    )
    environment = _environment()
    ann = (
        None
        if ann_clusters is None
        else {"algorithm": "cluster", "clusters": ann_clusters, "nprobe": ann_nprobe}
    )
    effective_nprobe = serve_nprobe if serve_nprobe is not None else ann_nprobe
    if rescore == "none":
        effective_oversample = None
    elif serve_oversample is not None:
        effective_oversample = serve_oversample
    else:
        effective_oversample = oversample
    result: dict[str, Any] = {
        "schema_version": 2,
        "label": label,
        "env": environment,
        "dataset": {
            "rows": int(manifest["rows"]),
            "dim": int(manifest["dim"]),
            "n_queries": int(manifest["n_queries"]),
            "seed": int(manifest["seed"]),
            "gt": "fp32-exact-top100" if evaluation_id is not None else None,
            "corpus_id": corpus_id,
            "evaluation_id": evaluation_id,
            "source_revision": manifest["source"]["revision"],
        },
        "measurement": {
            "k": k,
            "loop_seconds_requested": loop_seconds,
            "loop_concurrency": loop_concurrency,
            "query_count": int(manifest["n_queries"]),
        },
        "run_id": None,
        "store": {
            "bit_width": bit_width,
            "ann": ann,
            "rescore": (
                None if rescore == "none" else {"dtype": rescore, "oversample": oversample}
            ),
            "serve_overrides": {
                "ann_nprobe": serve_nprobe,
                "oversample": serve_oversample,
            },
            "layout": {"compacted": bool(compact), "id": layout_id},
            "provenance": None,
            "build": None,
            "footprint": None,
            "open": {"open_plus_first_query_seconds": None},
        },
        "serve": None,
    }
    if build:
        if store_dir.exists() and any(store_dir.iterdir()):
            raise ValueError(
                f"build target {store_dir} is not empty; validate it for resume or use a new path"
            )
        actual_builder_sha = builder_git_sha or environment.get("git_sha")
        if (
            not isinstance(actual_builder_sha, str)
            or len(actual_builder_sha) != 40
            or any(character not in "0123456789abcdef" for character in actual_builder_sha)
        ):
            raise ValueError("store builds require an explicit 40-character builder_git_sha")
        store_dir.mkdir(parents=True, exist_ok=True)
        db = _open_store(
            store_dir,
            dim=int(manifest["dim"]),
            bit_width=bit_width,
            ann_clusters=ann_clusters,
            ann_nprobe=ann_nprobe,
            rescore=rescore,
            oversample=oversample,
            rows=int(manifest["rows"]),
            corpus_id=corpus_id,
        )
        try:
            ingest_seconds, ingested_sha256 = _ingest_vectors(db, base, batch=ingest_batch)
            if ingested_sha256 != manifest["sha256"]["base"]:
                raise ValueError(
                    "ingested corpus SHA-256 does not match the committed dataset manifest"
                )
            print(f"[wiki-dpr] {label}: ingest done in {ingest_seconds:.1f}s", flush=True)
            cluster_build_seconds: float | None = None
            compact_seconds: float | None = None
            warm_path = "query"
            if ann is not None:
                compact_fn = getattr(db, "compact", None)
                started = time.perf_counter()
                if callable(compact_fn):
                    # compact() warms the cluster, rewrites the base, and
                    # persists; its duration is compaction, not k-means alone,
                    # so it must not masquerade as cluster_build_seconds.
                    print(f"[wiki-dpr] {label}: warming via compact", flush=True)
                    compact_fn()
                    warm_path = "compact"
                    compact_seconds = time.perf_counter() - started
                else:
                    print(f"[wiki-dpr] {label}: warm query", flush=True)
                    db.search_by_vector(queries[0], k=k, normalize=False)
                    cluster_build_seconds = time.perf_counter() - started
            else:
                print(f"[wiki-dpr] {label}: warm query", flush=True)
                db.search_by_vector(queries[0], k=k, normalize=False)
            if compact and warm_path != "compact":
                compact_fn = getattr(db, "compact", None)
                if not callable(compact_fn):
                    raise _requires_engine("compact()")
                started = time.perf_counter()
                compact_fn()
                compact_seconds = time.perf_counter() - started
            print(f"[wiki-dpr] {label}: {warm_path} warm done, persisting", flush=True)
            started = time.perf_counter()
            db.persist()
            persist_seconds = time.perf_counter() - started
            print(f"[wiki-dpr] {label}: persist done in {persist_seconds:.1f}s", flush=True)
            result["store"]["build"] = {
                "ingest_seconds": ingest_seconds,
                "ingest_rows_per_s": base.shape[0] / ingest_seconds if ingest_seconds else 0.0,
                "cluster_build_seconds": cluster_build_seconds,
                "warm_path": warm_path,
                "compact_seconds": compact_seconds,
                "persist_seconds": persist_seconds,
            }
        finally:
            db.close()
        print(f"[wiki-dpr] {label}: store closed", flush=True)
        result["store"]["footprint"] = _dir_footprint(store_dir)
        persisted_config = _store_config(
            bit_width=bit_width,
            ann_clusters=ann_clusters,
            ann_nprobe=ann_nprobe,
            rescore=rescore,
            oversample=oversample,
            rows=int(manifest["rows"]),
            dim=int(manifest["dim"]),
            corpus_id=corpus_id,
            builder_git_sha=actual_builder_sha,
            builder_lodedb_version=str(environment["lodedb_version"]),
            layout_id=layout_id,
        )
        _write_store_config(store_dir, persisted_config)
        result["store"]["provenance"] = persisted_config
    if serve:
        persisted_config = _guard_existing_store_config(
            store_dir,
            _requested_store_create(
                bit_width=bit_width,
                ann_clusters=ann_clusters,
                ann_nprobe=ann_nprobe,
                rescore=rescore,
                oversample=oversample,
                rows=int(manifest["rows"]),
                dim=int(manifest["dim"]),
            ),
            corpus_id=corpus_id,
            serve_nprobe=serve_nprobe,
            serve_oversample=serve_oversample,
            expected_layout_id=layout_id,
            expected_builder_git_sha=expected_builder_git_sha,
        )
        result["store"]["provenance"] = persisted_config
        started = time.perf_counter()
        db = _open_store(
            store_dir,
            dim=int(manifest["dim"]),
            bit_width=bit_width,
            ann_clusters=ann_clusters,
            ann_nprobe=ann_nprobe,
            rescore=rescore,
            oversample=oversample,
            rows=int(manifest["rows"]),
            corpus_id=corpus_id,
            serve_nprobe=serve_nprobe,
            serve_oversample=serve_oversample,
            expected_layout_id=layout_id,
            expected_builder_git_sha=expected_builder_git_sha,
        )
        try:
            db.search_by_vector(queries[0], k=k, normalize=False)
            result["store"]["open"]["open_plus_first_query_seconds"] = time.perf_counter() - started
            if result["store"]["footprint"] is None:
                result["store"]["footprint"] = _dir_footprint(store_dir)
            reset = read = None
            if report_block_skips:
                reset, read = _block_skip_api()
                reset()
                counter_before = int(read())
            recall = _recall_at_100(db, queries, gt_indices, k=k)
            block_skip = None
            if report_block_skips:
                counter_delta = int(read()) - counter_before
                total_blocks = int(len(queries) * math.ceil(base.shape[0] / 32))
                candidate_fraction = (
                    1.0 if ann is None else float(effective_nprobe) / float(ann_clusters)
                )
                block_skip = {
                    "fraction": counter_delta / total_blocks if total_blocks else 0.0,
                    "counter_delta": counter_delta,
                    "total_blocks": total_blocks,
                    "candidate_fraction_f": candidate_fraction,
                }
            result["serve"] = {
                "effective_nprobe": effective_nprobe,
                "effective_oversample": effective_oversample,
                "recall_at_100": recall,
                "sequential_latency_ms": _sequential_latency(db, queries, k=k),
                "closed_loop": _closed_loop(
                    db,
                    queries,
                    k=k,
                    seconds=loop_seconds,
                    concurrency=loop_concurrency,
                    seed=int(manifest["seed"]),
                ),
                "batched": _batched_qps(db, queries, k=k),
                "block_skip": block_skip,
            }
        finally:
            db.close()
    provenance = result["store"]["provenance"]
    if not isinstance(provenance, dict):
        raise ValueError("benchmark did not establish store provenance")
    result["run_id"] = canonical_sha256(
        {
            "label": label,
            "evaluation_id": evaluation_id,
            "store_id": provenance["store_id"],
            "measurement": result["measurement"],
            "serve_overrides": result["store"]["serve_overrides"],
        }
    )
    validate_result_schema(result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Builds the benchmark CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--bit-width", type=int, default=4, choices=(2, 4))
    parser.add_argument("--ann-clusters", type=int)
    parser.add_argument("--ann-nprobe", type=int)
    parser.add_argument("--loop-seconds", type=float, default=20.0)
    parser.add_argument("--loop-concurrency", type=int, default=4)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--ingest-batch", type=int, default=8192)
    parser.add_argument("--serve-only", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--rescore", choices=("none", "fp16", "fp32"), default="none")
    parser.add_argument("--oversample", type=float, default=4.0)
    parser.add_argument("--serve-nprobe", type=int)
    parser.add_argument("--serve-oversample", type=float)
    parser.add_argument("--report-block-skips", action="store_true")
    parser.add_argument(
        "--builder-git-sha",
        help="exact source commit used to build the store (defaults to the checkout HEAD)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/wiki_dpr.json"))
    return parser


def main(argv: list[str] | None = None) -> None:
    """Runs one benchmark configuration and writes its metrics-only JSON result."""

    args = build_parser().parse_args(argv)
    if args.serve_only and args.build_only:
        raise SystemExit("--serve-only and --build-only cannot be used together")
    try:
        result = run_benchmark(
            data_dir=args.data,
            store_dir=args.store,
            label=args.out.stem,
            bit_width=args.bit_width,
            ann_clusters=args.ann_clusters,
            ann_nprobe=args.ann_nprobe,
            loop_seconds=args.loop_seconds,
            loop_concurrency=args.loop_concurrency,
            k=args.k,
            ingest_batch=args.ingest_batch,
            build=not args.serve_only,
            serve=not args.build_only,
            compact=args.compact,
            rescore=args.rescore,
            oversample=args.oversample,
            serve_nprobe=args.serve_nprobe,
            serve_oversample=args.serve_oversample,
            report_block_skips=args.report_block_skips,
            builder_git_sha=args.builder_git_sha,
        )
    except EngineFeatureRequired as exc:
        raise SystemExit(str(exc)) from exc
    write_result_atomic(args.out, result)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
