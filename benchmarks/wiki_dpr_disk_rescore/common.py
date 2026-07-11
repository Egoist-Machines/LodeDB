"""Shared dataset manifest and synthetic-data helpers for the wiki_dpr benchmark.

The prepared dataset contains only vectors and numeric ground truth.  It is deliberately
separate from a LodeDB store, so every engine can use the same corpus and fp32 reference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST_NAME = "manifest.json"
BASE_NAME = "base.f32"
QUERIES_NAME = "queries.npy"
GT_INDICES_NAME = "gt_top100_indices.npy"
GT_SCORES_NAME = "gt_top100_scores.npy"
GT_K = 100


def _path(data_dir: str | Path, name: str | None) -> Path | None:
    """Resolves an optional manifest file name below a dataset directory."""

    return None if name is None else Path(data_dir) / name


def _require_int(manifest: dict[str, Any], key: str, *, minimum: int = 1) -> int:
    """Reads a positive integer manifest field with a useful validation error."""

    value = manifest.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"manifest {key!r} must be an integer >= {minimum}")
    return int(value)


def load_manifest(data_dir: str | Path) -> dict[str, Any]:
    """Loads and validates the JSON manifest without loading the vector payload."""

    path = Path(data_dir) / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing dataset manifest: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid dataset manifest JSON: {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    if manifest.get("version") != 1:
        raise ValueError(f"unsupported manifest version: {manifest.get('version')!r}")
    rows = _require_int(manifest, "rows", minimum=GT_K)
    _require_int(manifest, "dim")
    n_queries = _require_int(manifest, "n_queries")
    if manifest.get("normalized") is not True:
        raise ValueError("manifest must declare normalized=true")
    indices = manifest.get("query_row_indices")
    if not isinstance(indices, list) or len(indices) != n_queries:
        raise ValueError("manifest query_row_indices must have n_queries entries")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < rows
        for index in indices
    ):
        raise ValueError("manifest query_row_indices contains an invalid row")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("manifest files must be an object")
    for key in ("base", "queries", "gt_indices", "gt_scores"):
        if key not in files:
            raise ValueError(f"manifest files missing {key!r}")
        if files[key] is not None and (not isinstance(files[key], str) or not files[key]):
            raise ValueError(f"manifest files.{key} must be a non-empty string or null")
    if not isinstance(manifest.get("shards_fingerprint"), (list, type(None))):
        raise ValueError("manifest shards_fingerprint must be a list or null")
    if not isinstance(manifest.get("created_by"), str):
        raise ValueError("manifest created_by must be a string")
    return manifest


def write_manifest(data_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Validates then atomically writes a manifest and returns its path."""

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    temporary = data_dir / f".{MANIFEST_NAME}.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(data_dir / MANIFEST_NAME)
    load_manifest(data_dir)
    return data_dir / MANIFEST_NAME


def validate_dataset(data_dir: str | Path, *, require_gt: bool = True) -> dict[str, Any]:
    """Checks every declared payload's shape, dtype, and required byte count.

    ``base.f32`` is raw float32 data and is therefore checked exactly against
    ``rows * dim * 4``.  The NPY payloads carry headers, so they are opened as
    memory maps and checked by dtype and shape rather than raw byte equality.
    """

    data_dir = Path(data_dir)
    manifest = load_manifest(data_dir)
    rows = int(manifest["rows"])
    dim = int(manifest["dim"])
    n_queries = int(manifest["n_queries"])
    files = manifest["files"]
    base_path = _path(data_dir, files["base"])
    query_path = _path(data_dir, files["queries"])
    if base_path is None or query_path is None:
        raise ValueError("manifest needs base and queries files")
    expected_base_bytes = rows * dim * np.dtype(np.float32).itemsize
    if not base_path.is_file() or base_path.stat().st_size != expected_base_bytes:
        actual = base_path.stat().st_size if base_path.exists() else None
        raise ValueError(f"base file size is {actual}, expected {expected_base_bytes}: {base_path}")
    queries = np.load(query_path, mmap_mode="r")
    if queries.dtype != np.float32 or queries.shape != (n_queries, dim):
        raise ValueError("queries file must be float32 with shape (n_queries, dim)")
    gt_indices_path = _path(data_dir, files["gt_indices"])
    gt_scores_path = _path(data_dir, files["gt_scores"])
    if (gt_indices_path is None) != (gt_scores_path is None):
        raise ValueError("ground-truth index and score files must be both present or both null")
    if require_gt and gt_indices_path is None:
        raise ValueError("this operation needs ground truth, run data_prep without --skip-gt")
    if gt_indices_path is not None and gt_scores_path is not None:
        indices = np.load(gt_indices_path, mmap_mode="r")
        scores = np.load(gt_scores_path, mmap_mode="r")
        expected = (n_queries, GT_K)
        if indices.dtype != np.int64 or indices.shape != expected:
            raise ValueError("ground-truth indices must be int64 with shape (n_queries, 100)")
        if scores.dtype != np.float32 or scores.shape != expected:
            raise ValueError("ground-truth scores must be float32 with shape (n_queries, 100)")
    return manifest


def dataset_arrays(
    data_dir: str | Path, *, require_gt: bool = True
) -> tuple[dict[str, Any], np.memmap, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Opens prepared arrays as memory maps after validating their manifest."""

    data_dir = Path(data_dir)
    manifest = validate_dataset(data_dir, require_gt=require_gt)
    files = manifest["files"]
    base = np.memmap(
        data_dir / files["base"],
        dtype=np.float32,
        mode="r",
        shape=(int(manifest["rows"]), int(manifest["dim"])),
    )
    queries = np.load(data_dir / files["queries"], mmap_mode="r")
    indices_name = files["gt_indices"]
    scores_name = files["gt_scores"]
    indices = None if indices_name is None else np.load(data_dir / indices_name, mmap_mode="r")
    scores = None if scores_name is None else np.load(data_dir / scores_name, mmap_mode="r")
    return manifest, base, queries, indices, scores


def compute_exact_ground_truth(
    data_dir: str | Path, *, block_rows: int = 100_000
) -> tuple[Path, Path]:
    """Writes exact fp32 top-100 self-retrieval ground truth in bounded memory."""

    if block_rows < GT_K:
        raise ValueError(f"block_rows must be at least {GT_K}")
    data_dir = Path(data_dir)
    manifest, base, queries, _, _ = dataset_arrays(data_dir, require_gt=False)
    rows = int(manifest["rows"])
    n_queries = int(manifest["n_queries"])
    if rows < GT_K:
        raise ValueError(f"exact top-{GT_K} requires at least {GT_K} rows")
    query_matrix = np.ascontiguousarray(queries, dtype=np.float32)
    best_scores = np.full((n_queries, GT_K), -np.inf, dtype=np.float32)
    best_indices = np.full((n_queries, GT_K), -1, dtype=np.int64)
    for start in range(0, rows, block_rows):
        stop = min(start + block_rows, rows)
        scores = np.asarray(base[start:stop]) @ query_matrix.T
        local_k = min(GT_K, stop - start)
        local_order = np.argpartition(scores, scores.shape[0] - local_k, axis=0)[-local_k:, :]
        local_scores = np.take_along_axis(scores, local_order, axis=0).T
        local_indices = (local_order.T + start).astype(np.int64, copy=False)
        candidate_scores = np.concatenate((best_scores, local_scores), axis=1)
        candidate_indices = np.concatenate((best_indices, local_indices), axis=1)
        selected = np.argpartition(-candidate_scores, GT_K - 1, axis=1)[:, :GT_K]
        best_scores = np.take_along_axis(candidate_scores, selected, axis=1)
        best_indices = np.take_along_axis(candidate_indices, selected, axis=1)
    order = np.argsort(-best_scores, axis=1)
    best_scores = np.take_along_axis(best_scores, order, axis=1).astype(np.float32, copy=False)
    best_indices = np.take_along_axis(best_indices, order, axis=1).astype(np.int64, copy=False)
    indices_path = data_dir / GT_INDICES_NAME
    scores_path = data_dir / GT_SCORES_NAME
    np.save(indices_path, best_indices)
    np.save(scores_path, best_scores)
    manifest["files"]["gt_indices"] = GT_INDICES_NAME
    manifest["files"]["gt_scores"] = GT_SCORES_NAME
    write_manifest(data_dir, manifest)
    validate_dataset(data_dir, require_gt=True)
    return indices_path, scores_path


def make_synthetic_dataset(
    out_dir: str | Path, rows: int, dim: int, n_queries: int, seed: int
) -> dict[str, Any]:
    """Creates a deterministic normalized vector dataset and exact top-100 reference.

    This helper is intentionally local and download-free, making it suitable for
    unit and end-to-end smoke tests.
    """

    if rows < GT_K:
        raise ValueError(f"rows must be at least {GT_K}")
    if dim < 1 or n_queries < 1 or n_queries > rows:
        raise ValueError("dim and n_queries must be positive, with n_queries <= rows")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base = np.memmap(out_dir / BASE_NAME, dtype=np.float32, mode="w+", shape=(rows, dim))
    # Correlated random unit vectors make this a stable retrieval smoke rather
    # than a top-100 tie/noise test of a very small 4-bit corpus. Each group has
    # exactly 100 distinct random rows around a random unit centre, so a query's
    # expected fp32 neighbours are its group. Fall back to independent random
    # unit rows for sizes that cannot be divided into 100-row groups.
    if rows % GT_K == 0:
        centres = rng.standard_normal((rows // GT_K, dim), dtype=np.float32)
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        jitter = rng.standard_normal((rows, dim), dtype=np.float32) * np.float32(0.03)
        vectors = np.repeat(centres, GT_K, axis=0) + jitter
    else:
        vectors = rng.standard_normal((rows, dim), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    base[:] = vectors / np.where(norms == 0.0, 1.0, norms)
    base.flush()
    query_indices = rng.choice(rows, size=n_queries, replace=False).astype(np.int64)
    queries = np.asarray(base[query_indices], dtype=np.float32)
    np.save(out_dir / QUERIES_NAME, queries)
    manifest: dict[str, Any] = {
        "version": 1,
        "rows": int(rows),
        "dim": int(dim),
        "n_queries": int(n_queries),
        "seed": int(seed),
        "query_row_indices": [int(index) for index in query_indices],
        "normalized": True,
        "shards_fingerprint": None,
        "files": {
            "base": BASE_NAME,
            "queries": QUERIES_NAME,
            "gt_indices": None,
            "gt_scores": None,
        },
        "created_by": "benchmarks/wiki_dpr_disk_rescore/common.py synthetic",
    }
    write_manifest(out_dir, manifest)
    compute_exact_ground_truth(out_dir)
    return load_manifest(out_dir)
