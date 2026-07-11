"""CPU-only Modal launcher for the wiki_dpr_e5 disk-rescore benchmark.

This image deliberately has no torch or CUDA dependency.  The workload starts
from precomputed E5 vectors and uses only the LodeDB vector-in API.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import modal

_REMOTE_SRC = "/root/lodedb-src"
_REMOTE_BENCH = "/root/benchmarks/wiki_dpr_disk_rescore"
_VOLUME_PATH = "/vol"
_SERVE_CPU = 4.0
_SERVE_MEMORY_MB = 131072


def _build_image() -> modal.Image:
    """Builds a Linux CPU image with OpenBLAS and the full local LodeDB workspace."""

    image = (
        modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
        .apt_install("build-essential", "curl", "pkg-config", "libopenblas-dev")
        .pip_install(
            "numpy>=2.0.0,<3.0.0",
            "typer>=0.12.0",
            "pyyaml>=6.0.0",
            "pyarrow>=16.0.0",
            "huggingface-hub>=0.24.0",
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal"
        )
    )
    # Modal imports this module in the container too. Local paths exist only while
    # submitting, so source-copy and non-copy bench mount steps must be local-only.
    if modal.is_local():
        repo_root = Path(__file__).resolve().parents[2]
        for rel in ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "Cargo.toml", "Cargo.lock"):
            image = image.add_local_file(
                str(repo_root / rel), remote_path=f"{_REMOTE_SRC}/{rel}", copy=True
            )
        for rel in ("crates", "third_party/turbovec", "src"):
            image = image.add_local_dir(
                str(repo_root / rel),
                remote_path=f"{_REMOTE_SRC}/{rel}",
                copy=True,
                ignore=["**/target/**", "**/__pycache__/**", "**/*.pyc", "**/*.so", "**/*.dylib"],
            )
        image = image.run_commands(
            'PATH="$HOME/.cargo/bin:$PATH" python -m pip install --no-deps /root/lodedb-src'
        ).env({"PYTHONPATH": _REMOTE_BENCH})
        image = image.add_local_dir(
            str(Path(__file__).resolve().parent),
            remote_path=_REMOTE_BENCH,
            copy=False,
            ignore=["__pycache__", "results", "*.pyc"],
        )
    return image


IMAGE = _build_image()
app = modal.App("lodedb-wiki-dpr-bench", image=IMAGE)
VOLUME = modal.Volume.from_name("wiki-dpr-e5", create_if_missing=True)


def _prepared_dir(target_rows: int) -> str:
    """Returns the canonical volume path for a full or subset prepared corpus."""

    if target_rows == 21_015_300:
        return f"{_VOLUME_PATH}/prepared/full_21m"
    return f"{_VOLUME_PATH}/prepared/subset_{target_rows}"


def _store_dir(target_rows: int, label: str) -> str:
    """Returns an isolated store directory for one corpus size and benchmark label."""

    return f"{_VOLUME_PATH}/stores/rows_{target_rows}/{label}"


def _log_result(result: dict[str, Any]) -> None:
    """Logs result JSON worker-side so it survives a disconnected local client."""

    print(f"LODEDB_RESULT {json.dumps(result, sort_keys=True)}", flush=True)


@app.function(cpu=4.0, memory=16384, timeout=14_400, volumes={_VOLUME_PATH: VOLUME})
def download() -> dict[str, Any]:
    """Downloads parquet shards once into the persistent volume."""

    from huggingface_hub import snapshot_download

    shards_dir = Path(_VOLUME_PATH) / "shards" / "data"
    marker = Path(_VOLUME_PATH) / "shards" / ".download_complete"
    existing = sorted(shards_dir.glob("*.parquet"))
    try:
        completed_shards = int(marker.read_text().strip())
    except (FileNotFoundError, ValueError):
        completed_shards = None
    if completed_shards == len(existing) and completed_shards > 0:
        return {"downloaded": False, "shards": len(existing), "path": str(shards_dir)}
    snapshot_download(
        repo_id="kenhktsui/wiki_dpr_e5",
        repo_type="dataset",
        allow_patterns=["data/*.parquet"],
        local_dir=f"{_VOLUME_PATH}/shards",
    )
    shards = sorted(shards_dir.glob("*.parquet"))
    if not shards:
        raise RuntimeError("Hugging Face snapshot completed without data/*.parquet")
    marker.write_text(f"{len(shards)}\n")
    VOLUME.commit()
    return {"downloaded": True, "shards": len(shards), "path": str(shards_dir)}


@app.function(cpu=16.0, memory=65536, timeout=21_600, volumes={_VOLUME_PATH: VOLUME})
def prepare(target_rows: int = 21_015_300, n_queries: int = 1000, seed: int = 42) -> dict[str, Any]:
    """Streams source shards to the normalized corpus and query files, without GT."""

    from common import load_manifest
    from data_prep import prepare_dataset

    out = Path(_prepared_dir(target_rows))
    try:
        existing = load_manifest(out)
        if (
            int(existing["rows"]) == target_rows
            and int(existing["n_queries"]) == n_queries
            and existing.get("seed") == seed
        ):
            return {"prepared": False, "path": str(out), "rows": target_rows}
    except FileNotFoundError:
        pass
    manifest = prepare_dataset(
        Path(_VOLUME_PATH) / "shards" / "data",
        out,
        target_rows=target_rows,
        n_queries=n_queries,
        seed=seed,
        skip_gt=True,
    )
    VOLUME.commit()
    return {"prepared": True, "path": str(out), "rows": manifest["rows"], "dim": manifest["dim"]}


@app.function(cpu=32.0, memory=131072, timeout=14_400, volumes={_VOLUME_PATH: VOLUME})
def ground_truth(target_rows: int = 21_015_300, block_rows: int = 100_000) -> dict[str, Any]:
    """Computes the one-pass fp32 top-100 reference if it is not already present."""

    from common import compute_exact_ground_truth, load_manifest, validate_dataset

    out = Path(_prepared_dir(target_rows))
    manifest = load_manifest(out)
    if int(manifest["rows"]) != target_rows:
        raise ValueError(f"prepared row count is {manifest['rows']}, expected {target_rows}")
    try:
        validate_dataset(out, require_gt=True)
        return {"ground_truth": False, "path": str(out), "rows": target_rows}
    except (FileNotFoundError, ValueError):
        pass
    indices, scores = compute_exact_ground_truth(out, block_rows=block_rows)
    VOLUME.commit()
    return {"ground_truth": True, "indices": str(indices), "scores": str(scores)}


def _build_store_impl(
    store_label: str,
    target_rows: int,
    git_sha: str | None,
) -> dict[str, Any]:
    """Builds one named reusable store directly on the volume, without serving."""

    from lodedb_bench import run_benchmark
    from sweep import STORES

    try:
        store = STORES[store_label]
    except KeyError as exc:
        raise ValueError(f"unknown store label: {store_label}") from exc
    store_dir = Path(_store_dir(target_rows, store_label))
    config_path = store_dir / "benchmark_store_config.json"
    if config_path.exists():
        return {"label": store_label, "resumed": True, "store": {"build": None}}
    if store_dir.exists() and any(store_dir.iterdir()):
        # A store directory without its config file is a crashed earlier build;
        # it cannot be trusted or served, so rebuild it from scratch.
        print(f"[wiki-dpr] wiping incomplete store {store_dir}", flush=True)
        shutil.rmtree(store_dir)
        VOLUME.commit()
    kwargs = dict(store)
    kwargs.pop("requires_engine", None)
    result = run_benchmark(
        data_dir=_prepared_dir(target_rows),
        store_dir=store_dir,
        label=store_label,
        build=True,
        serve=False,
        **kwargs,
    )
    if git_sha is not None:
        result["env"]["git_sha"] = git_sha
    VOLUME.commit()
    _log_result(result)
    return result


@app.function(cpu=32.0, memory=131072, timeout=86_400, volumes={_VOLUME_PATH: VOLUME})
def build_store(
    store_label: str,
    target_rows: int = 21_015_300,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Builds one named reusable store directly on the volume, without serving."""

    return _build_store_impl(store_label, target_rows, git_sha)


@app.function(cpu=32.0, memory=131072, timeout=86_400, volumes={_VOLUME_PATH: VOLUME})
def build_many(
    store_labels: list[str],
    target_rows: int = 21_015_300,
    git_sha: str | None = None,
) -> list[dict[str, Any]]:
    """Builds several stores sequentially inside one remote container.

    Running the loop server-side makes the whole batch survive a lost local
    client: with a spawned call, nothing after the submission depends on the
    submitting machine's connectivity.
    """

    results = []
    for label in store_labels:
        print(f"[wiki-dpr] building {label} rows={target_rows}", flush=True)
        results.append(_build_store_impl(label, target_rows, git_sha))
    return results


def _serve_from_local_copy(
    spec: dict[str, Any],
    target_rows: int,
    *,
    data_dir: str | Path | None = None,
    store_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copies one durable store to local SSD, then runs all serving measurements."""

    from lodedb_bench import run_benchmark

    label = str(spec["label"])
    store_label = str(spec.get("store", label))
    source = Path(_store_dir(target_rows, store_label))
    if not source.exists():
        raise FileNotFoundError(f"store is not built: {source}")
    local_root = Path(tempfile.mkdtemp(prefix=f"wiki-dpr-{label}-"))
    local_store = local_root / "store"
    started = time.perf_counter()
    shutil.copytree(source, local_store)
    copy_seconds = time.perf_counter() - started
    if store_kwargs is None:
        from sweep import STORES

        try:
            store_kwargs = STORES[store_label]
        except KeyError as exc:
            raise ValueError(f"unknown store label: {store_label}") from exc
    kwargs = dict(store_kwargs)
    kwargs.pop("requires_engine", None)
    overrides = dict(spec.get("serve_overrides", {}))
    try:
        result = run_benchmark(
            data_dir=_prepared_dir(target_rows) if data_dir is None else data_dir,
            store_dir=local_store,
            label=label,
            build=False,
            serve=True,
            serve_nprobe=overrides.get("ann_nprobe"),
            serve_oversample=overrides.get("oversample"),
            **kwargs,
        )
    finally:
        shutil.rmtree(local_root, ignore_errors=True)
    result["volume_copy_seconds"] = copy_seconds
    return result


@app.function(
    cpu=_SERVE_CPU,
    memory=_SERVE_MEMORY_MB,
    timeout=14_400,
    volumes={_VOLUME_PATH: VOLUME},
)
def serve(
    spec: dict[str, Any],
    target_rows: int = 21_015_300,
    git_sha: str | None = None,
) -> dict[str, Any]:
    """Serves a volume-built store from container-local disk and logs the result."""

    result = _serve_from_local_copy(spec, target_rows)
    result["env"].update(
        {
            "requested_cpu": _SERVE_CPU,
            "requested_memory_mb": _SERVE_MEMORY_MB,
        }
    )
    if git_sha is not None:
        result["env"]["git_sha"] = git_sha
    _log_result(result)
    return result


def _local_git_sha() -> str | None:
    """Returns the local checkout revision for a remote container result."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


@app.local_entrypoint()
def build(
    target_rows: int = 21_015_300,
    labels: str = "exact_bw4,ann1000",
    parallel: bool = False,
) -> None:
    """Builds the requested reusable stores as durable stores on the volume.

    Sequential by default: concurrent containers committing large files to one
    volume can collide, killing a build between its persist and its config write.
    """

    from sweep import STORES

    wanted = [label.strip() for label in labels.split(",") if label.strip()]
    unknown = sorted(set(wanted) - set(STORES))
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    git_sha = _local_git_sha()
    if parallel:
        results = list(build_store.starmap((label, target_rows, git_sha) for label in wanted))
        for label, result in zip(wanted, results, strict=False):
            build_seconds = (result.get("store") or {}).get("build")
            print(f"built {label} rows={target_rows}: {build_seconds}")
        return
    # Fire-and-forget: the sequential loop runs inside one remote container, so
    # a dropped local connection cannot interrupt the batch. Progress lands on
    # the volume (benchmark_store_config.json per completed store).
    call = build_many.spawn(wanted, target_rows, git_sha)
    print(f"spawned build_many {wanted} rows={target_rows}: {call.object_id}")


@app.local_entrypoint()
def main(target_rows: int = 21_015_300, labels: str = "exact_bw4,ann1000_np16") -> None:
    """Collects serving results from stores that were built with ``build_store``."""

    from sweep import SERVE_CONFIGS

    wanted = {label.strip() for label in labels.split(",") if label.strip()}
    specs = [config for config in SERVE_CONFIGS if config["label"] in wanted]
    if len(specs) != len(wanted):
        unknown = sorted(wanted - {config["label"] for config in SERVE_CONFIGS})
        raise ValueError(f"unknown labels: {unknown}")
    git_sha = _local_git_sha()
    results = [
        serve.remote(spec, target_rows=target_rows, git_sha=git_sha) for spec in specs
    ]
    out = Path(__file__).resolve().parent / "results" / "modal_serve_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


@app.function(cpu=4.0, memory=16384, timeout=14_400, volumes={_VOLUME_PATH: VOLUME})
def smoke() -> dict[str, Any]:
    """Runs a download-free 50k x 256d exact and ANN pipeline check for cost control."""

    from common import make_synthetic_dataset
    from lodedb_bench import run_benchmark

    data = Path(_VOLUME_PATH) / "prepared" / "smoke_50k"
    make_synthetic_dataset(data, rows=50_000, dim=256, n_queries=50, seed=42)
    specs = (
        {
            "label": "smoke_exact",
            "store": "smoke_exact",
            "serve_overrides": {},
            "store_kwargs": {"bit_width": 4},
        },
        {
            "label": "smoke_ann",
            "store": "smoke_ann",
            "serve_overrides": {},
            "store_kwargs": {"bit_width": 4, "ann_clusters": 64, "ann_nprobe": 8},
        },
    )
    results: list[dict[str, Any]] = []
    for spec in specs:
        store = Path(_store_dir(50_000, str(spec["label"])))
        kwargs = dict(spec["store_kwargs"])
        label = str(spec["label"])
        built = run_benchmark(
            data_dir=data,
            store_dir=store,
            label=label,
            loop_seconds=2.0,
            build=True,
            serve=False,
            **kwargs,
        )
        _log_result(built)
        served = _serve_from_local_copy(
            spec,
            target_rows=50_000,
            data_dir=data,
            store_kwargs=kwargs,
        )
        if float(served["serve"]["recall_at_100"]) <= 0.9:
            recall = served["serve"]["recall_at_100"]
            raise AssertionError(f"{label} recall did not clear 0.9: {recall}")
        results.append(served)
        _log_result(served)
    VOLUME.commit()
    return {"smoke": True, "results": results}
