"""Engine-independent measurement harness for wiki_dpr_e5 disk-rescore runs.

The corpus and exact fp32 reference are prepared by :mod:`data_prep`.  This
module never recomputes brute-force truth while serving.  It drives only the
vector-in LodeDB API, so the measured path contains neither model loading nor
document text retention.
"""

from __future__ import annotations

import argparse
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
    from .common import GT_K, dataset_arrays
except ImportError:  # Direct execution from this directory.
    from common import GT_K, dataset_arrays  # type: ignore[no-redef]


class EngineFeatureRequired(RuntimeError):
    """Raised when an opt-in benchmark switch needs the companion engine branch."""


def _requires_engine(feature: str) -> EngineFeatureRequired:
    return EngineFeatureRequired(
        "requires engine branch feat/cluster-layout-rescore: " + feature
    )


_STORE_CONFIG_NAME = "benchmark_store_config.json"


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
) -> dict[str, int | float | str | None]:
    """Returns the benchmark-level configuration persisted alongside a store."""

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
    expected_fields = set(
        _store_config(
            bit_width=0,
            ann_clusters=None,
            ann_nprobe=None,
            rescore="none",
            oversample=0.0,
            rows=0,
            dim=0,
        )
    )
    missing = sorted(expected_fields - set(loaded))
    if missing:
        raise ValueError(
            f"benchmark store configuration is missing {missing}: {config_path}; rebuild the store"
        )
    return loaded


def _guard_existing_store_config(store_dir: Path, requested: dict[str, Any]) -> None:
    """Rejects a store built for a different corpus or create-time configuration."""

    persisted = _load_store_config(store_dir)

    def matches(field: str) -> bool:
        return (
            type(persisted[field]) is type(requested[field])
            and persisted[field] == requested[field]
        )

    create_time_fields = ("bit_width", "ann_clusters", "rescore", "rows", "dim")
    for field in create_time_fields:
        if not matches(field):
            raise ValueError(
                f"existing store {store_dir} has {field}={persisted[field]!r}, "
                f"requested {requested[field]!r}; rebuild the store"
            )
    if not matches("ann_nprobe"):
        raise _requires_engine("open-time ann_nprobe override")
    if not matches("oversample"):
        raise _requires_engine("open-time rescore oversample override")


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
        kwargs["rescore_dtype"] = rescore
        kwargs["rescore_oversample"] = oversample
    existing_store = path.exists() and any(path.iterdir())
    requested_config = _store_config(
        bit_width=bit_width,
        ann_clusters=ann_clusters,
        ann_nprobe=ann_nprobe,
        rescore=rescore,
        oversample=oversample,
        rows=rows,
        dim=dim,
    )
    if existing_store:
        _guard_existing_store_config(path, requested_config)
    try:
        db = LodeDB(path, **kwargs)
    except TypeError as exc:
        if rescore != "none":
            raise _requires_engine("rescore dtype and oversample options") from exc
        raise
    return db


def _ingest_vectors(db: Any, vectors: np.ndarray, *, batch: int) -> float:
    """Ingests normalized memmap rows without converting vectors to Python lists."""

    if batch < 1:
        raise ValueError("ingest batch must be positive")
    started = time.perf_counter()
    for start in range(0, vectors.shape[0], batch):
        stop = min(start + batch, vectors.shape[0])
        payload = [
            {"id": str(index), "vector": vectors[index]}
            for index in range(start, stop)
        ]
        db.add_vectors_many(payload, normalize=False)
    return time.perf_counter() - started


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

    for key in ("label", "env", "dataset", "store", "serve"):
        if key not in result:
            raise ValueError(f"result is missing {key!r}")
    dataset = result["dataset"]
    if dataset.get("gt") != "fp32-exact-top100":
        raise ValueError("result dataset must identify the fp32 exact top-100 reference")
    store = result["store"]
    for key in ("bit_width", "ann", "rescore", "layout", "build", "footprint", "open"):
        if key not in store:
            raise ValueError(f"result store is missing {key!r}")
    serve = result["serve"]
    if serve is not None:
        for key in (
            "recall_at_100",
            "sequential_latency_ms",
            "closed_loop",
            "batched",
            "block_skip",
        ):
            if key not in serve:
                raise ValueError(f"result serve is missing {key!r}")


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
    report_block_skips: bool = False,
) -> dict[str, Any]:
    """Builds and/or serves one exact or cluster-pruned vector store configuration."""

    if k < GT_K:
        raise ValueError("--k must be at least 100 because this harness reports recall@100")
    if rescore not in {"none", "fp16", "fp32"}:
        raise ValueError("rescore must be none, fp16, or fp32")
    if oversample <= 0:
        raise ValueError("oversample must be positive")
    if compact and not build:
        raise _requires_engine("compact() is a build-time operation")
    manifest, base, queries, gt_indices, _ = dataset_arrays(data_dir, require_gt=serve)
    if serve and gt_indices is None:
        raise ValueError("serve measurements need exact ground truth")
    store_dir = Path(store_dir)
    ann = (
        None
        if ann_clusters is None
        else {"algorithm": "cluster", "clusters": ann_clusters, "nprobe": ann_nprobe}
    )
    result: dict[str, Any] = {
        "label": label,
        "env": _environment(),
        "dataset": {
            "rows": int(manifest["rows"]),
            "dim": int(manifest["dim"]),
            "n_queries": int(manifest["n_queries"]),
            "seed": int(manifest["seed"]),
            "gt": "fp32-exact-top100",
        },
        "store": {
            "bit_width": bit_width,
            "ann": ann,
            "rescore": None if rescore == "none" else {"dtype": rescore, "oversample": oversample},
            "layout": {"compacted": bool(compact)},
            "build": None,
            "footprint": None,
            "open": {"open_plus_first_query_seconds": None},
        },
        "serve": None,
    }
    if build:
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
        )
        try:
            ingest_seconds = _ingest_vectors(db, base, batch=ingest_batch)
            cluster_build_seconds: float | None = None
            if ann is not None:
                started = time.perf_counter()
                db.search_by_vector(queries[0], k=k, normalize=False)
                cluster_build_seconds = time.perf_counter() - started
            else:
                db.search_by_vector(queries[0], k=k, normalize=False)
            compact_seconds: float | None = None
            if compact:
                compact_fn = getattr(db, "compact", None)
                if not callable(compact_fn):
                    raise _requires_engine("compact()")
                started = time.perf_counter()
                compact_fn()
                compact_seconds = time.perf_counter() - started
            started = time.perf_counter()
            db.persist()
            persist_seconds = time.perf_counter() - started
            result["store"]["build"] = {
                "ingest_seconds": ingest_seconds,
                "ingest_rows_per_s": base.shape[0] / ingest_seconds if ingest_seconds else 0.0,
                "cluster_build_seconds": cluster_build_seconds,
                "compact_seconds": compact_seconds,
                "persist_seconds": persist_seconds,
            }
        finally:
            db.close()
        result["store"]["footprint"] = _dir_footprint(store_dir)
        _write_store_config(
            store_dir,
            _store_config(
                bit_width=bit_width,
                ann_clusters=ann_clusters,
                ann_nprobe=ann_nprobe,
                rescore=rescore,
                oversample=oversample,
                rows=int(manifest["rows"]),
                dim=int(manifest["dim"]),
            ),
        )
    if serve:
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
                    1.0 if ann is None else float(ann_nprobe) / float(ann_clusters)
                )
                block_skip = {
                    "fraction": counter_delta / total_blocks if total_blocks else 0.0,
                    "counter_delta": counter_delta,
                    "total_blocks": total_blocks,
                    "candidate_fraction_f": candidate_fraction,
                }
            result["serve"] = {
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
    parser.add_argument("--report-block-skips", action="store_true")
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
            report_block_skips=args.report_block_skips,
        )
    except EngineFeatureRequired as exc:
        raise SystemExit(str(exc)) from exc
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
