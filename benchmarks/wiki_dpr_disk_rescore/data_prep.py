"""Prepare wiki_dpr_e5 vectors for disk-rescore benchmarking.

Queries are sampled corpus rows and remain in the corpus.  The exact fp32
top-100 reference therefore includes each query's own row.  This self-retrieval
convention makes the reference reproducible without retaining document text.

Parquet input is streamed with ``ParquetFile.iter_batches``.  No full shard set
is materialized in memory.
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .common import (
        BASE_NAME,
        MANIFEST_VERSION,
        QUERIES_NAME,
        _atomic_save_npy,
        compute_exact_ground_truth,
        load_manifest,
        sha256_file,
        validate_dataset,
        write_manifest,
    )
except ImportError:  # Direct execution from this directory.
    from common import (  # type: ignore[no-redef]
        BASE_NAME,
        MANIFEST_VERSION,
        QUERIES_NAME,
        _atomic_save_npy,
        compute_exact_ground_truth,
        load_manifest,
        sha256_file,
        validate_dataset,
        write_manifest,
    )


def _pyarrow() -> Any:
    """Imports PyArrow only for parquet preparation."""

    import pyarrow as pa
    import pyarrow.parquet as pq

    return pa, pq


def _discover_embedding_column(shard: Path) -> tuple[str, int]:
    """Finds a float fixed-size-list or list embedding column in the first shard."""

    pa, pq = _pyarrow()
    schema = pq.ParquetFile(shard).schema_arrow
    candidates: list[tuple[int, str, int]] = []
    for field in schema:
        field_type = field.type
        is_fixed = pa.types.is_fixed_size_list(field_type)
        is_list = pa.types.is_list(field_type) or pa.types.is_large_list(field_type)
        if not (is_fixed or is_list):
            continue
        value_type = field_type.value_type
        if not pa.types.is_floating(value_type):
            continue
        names = {"embedding", "embeddings", "vector", "vectors"}
        priority = 0 if field.name.lower() in names else 1
        dim = int(field_type.list_size) if is_fixed else 0
        candidates.append((priority, field.name, dim))
    if not candidates:
        raise ValueError(f"no floating list embedding column found in {shard}")
    candidates.sort()
    _, name, dim = candidates[0]
    if dim:
        return name, dim
    first_batch = next(pq.ParquetFile(shard).iter_batches(batch_size=1, columns=[name]), None)
    if first_batch is None or first_batch.num_rows == 0:
        raise ValueError(f"cannot infer list embedding width from empty shard: {shard}")
    values = first_batch.column(0).to_pylist()[0]
    if not isinstance(values, list) or not values:
        raise ValueError(f"embedding column {name!r} has an empty first row")
    return name, len(values)


def _batch_matrix(column: Any, dim: int) -> np.ndarray:
    """Converts one Arrow list-array batch to a float32 matrix."""

    _, _ = _pyarrow()
    if column.null_count:
        raise ValueError("embedding column contains null rows")
    # flatten() honors slice offsets; .values would return the whole buffer.
    values = np.asarray(column.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
    expected = len(column) * dim
    if values.size != expected:
        raise ValueError(f"embedding rows are not uniformly {dim}-dimensional")
    return np.ascontiguousarray(values.reshape(len(column), dim), dtype=np.float32)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalizes a batch in fp32 and rejects zero vectors."""

    if not np.all(np.isfinite(matrix)):
        raise ValueError("embedding input contains NaN or infinite coordinates")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0.0):
        raise ValueError("embedding input contains a zero vector")
    return np.ascontiguousarray(matrix / norms, dtype=np.float32)


def prepare_dataset(
    shards_dir: str | Path,
    out_dir: str | Path,
    *,
    target_rows: int,
    n_queries: int = 1000,
    seed: int = 42,
    skip_gt: bool = False,
    source_revision: str,
) -> dict[str, Any]:
    """Streams parquet shards into a normalized raw-f32 corpus and query file."""

    if target_rows < 100:
        raise ValueError("target_rows must be at least 100 for top-100 ground truth")
    if not 1 <= n_queries <= target_rows:
        raise ValueError("n_queries must be in [1, target_rows]")
    if (
        len(source_revision) not in (40, 64)
        or any(character not in "0123456789abcdef" for character in source_revision)
    ):
        raise ValueError("source_revision must be a 40- or 64-character immutable commit digest")
    shards = sorted(Path(shards_dir).glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no parquet shards found below {shards_dir}")
    column_name, dim = _discover_embedding_column(shards[0])
    print(
        f"[wiki-dpr] embedding column={column_name!r}, dim={dim}, shards={len(shards)}",
        flush=True,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generation = secrets.token_hex(8)
    base_name = f"{Path(BASE_NAME).stem}-{generation}.f32"
    query_name = f"{Path(QUERIES_NAME).stem}-{generation}.npy"
    base_path = out_dir / base_name
    base_temporary = out_dir / f".{base_name}.tmp"
    base = np.memmap(base_temporary, dtype="<f4", mode="w+", shape=(target_rows, dim))
    base_sha = hashlib.sha256()
    _, pq = _pyarrow()
    written = 0
    for shard in shards:
        if written >= target_rows:
            break
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=65_536, columns=[column_name]):
            if written >= target_rows:
                break
            matrix = _batch_matrix(batch.column(0), dim)
            take = min(target_rows - written, matrix.shape[0])
            normalized = _normalize_rows(matrix[:take])
            base[written : written + take] = normalized
            base_sha.update(np.ascontiguousarray(normalized, dtype="<f4").tobytes(order="C"))
            written += take
            print(f"[wiki-dpr] wrote {written}/{target_rows} rows", flush=True)
    if written != target_rows:
        raise ValueError(f"parquet shards held {written} rows, target_rows is {target_rows}")
    base.flush()
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(target_rows, size=n_queries, replace=False).astype(np.int64)
    queries = np.asarray(base[query_indices], dtype=np.float32)
    del base
    base_temporary.replace(base_path)
    query_path = out_dir / query_name
    _atomic_save_npy(query_path, queries)
    manifest: dict[str, Any] = {
        "version": MANIFEST_VERSION,
        "rows": int(target_rows),
        "dim": int(dim),
        "n_queries": int(n_queries),
        "seed": int(seed),
        "query_row_indices": [int(index) for index in query_indices],
        "normalized": True,
        "source": {
            "dataset": "kenhktsui/wiki_dpr_e5",
            "revision": source_revision,
            "shards": [
                {"name": shard.name, "size": int(shard.stat().st_size)} for shard in shards
            ],
        },
        "files": {
            "base": base_name,
            "queries": query_name,
            # compute_exact_ground_truth installs these names after writing both files.
            "gt_indices": None,
            "gt_scores": None,
        },
        "sha256": {
            "base": base_sha.hexdigest(),
            "queries": sha256_file(query_path),
            "gt_indices": None,
            "gt_scores": None,
        },
        "created_by": "benchmarks/wiki_dpr_disk_rescore/data_prep.py",
    }
    write_manifest(out_dir, manifest)
    if skip_gt:
        validate_dataset(out_dir, require_gt=False)
    else:
        compute_exact_ground_truth(out_dir)
    return load_manifest(out_dir)


def build_parser() -> argparse.ArgumentParser:
    """Builds the standalone preparation CLI parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards-dir", type=Path, help="directory containing data/*.parquet")
    parser.add_argument("--out", required=True, type=Path, help="prepared dataset directory")
    parser.add_argument("--target-rows", type=int, help="number of corpus rows to retain")
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--source-revision",
        help="immutable source dataset revision (for example a Hugging Face commit SHA)",
    )
    parser.add_argument("--skip-gt", action="store_true", help="write corpus and queries only")
    parser.add_argument(
        "--gt-only", action="store_true", help="compute ground truth from --out only"
    )
    parser.add_argument("--gt-block-rows", type=int, default=100_000)
    return parser


def main(argv: list[str] | None = None) -> None:
    """Runs the dataset preparation command."""

    args = build_parser().parse_args(argv)
    if args.gt_only:
        if args.skip_gt:
            raise SystemExit("--gt-only and --skip-gt cannot be used together")
        compute_exact_ground_truth(args.out, block_rows=args.gt_block_rows)
        return
    if args.shards_dir is None or args.target_rows is None or args.source_revision is None:
        raise SystemExit(
            "--shards-dir, --target-rows, and --source-revision are required "
            "unless --gt-only is used"
        )
    prepare_dataset(
        args.shards_dir,
        args.out,
        target_rows=args.target_rows,
        n_queries=args.n_queries,
        seed=args.seed,
        skip_gt=args.skip_gt,
        source_revision=args.source_revision,
    )


if __name__ == "__main__":
    main()
