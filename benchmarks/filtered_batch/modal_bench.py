"""Run the LodeDB filtered batch-search benchmark on Modal (A10 / L40S).

Proves the filtering asymmetry on real CUDA hardware: an unfiltered
``search_many`` uses the GPU-resident scan, but a *filtered* one widens
top_k to the corpus size, trips the resident 4096 cap, and bypasses to CPU.

The image is built from the local ``src/`` tree, so it tests whatever branch
is checked out (run from a worktree with the fix to measure the fix).

Launch from the repo root:

    modal run benchmarks/filtered_batch/modal_bench.py::smoke
    modal run benchmarks/filtered_batch/modal_bench.py::a10
    modal run benchmarks/filtered_batch/modal_bench.py::l40s

Image recipe mirrors benchmarks/gpu_patch/modal_bench.py (CUDA torch base +
cupy + maturin-built vendored TurboVec + LodeDB from local src).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

_LODEDB_RUNTIME_DEPENDENCIES = (
    "numpy>=2.0.0",
    "typer>=0.12.0",
    "sentence-transformers>=3.0.0",
    "pyyaml>=6.0.0",
)
_CUPY_DEPENDENCY = "cupy-cuda12x>=13.0.0"
_REMOTE_BENCH_DIR = "/root/filtered_batch"


def _build_image() -> modal.Image:
    """CUDA image with LodeDB compiled from local src (maturin + vendored crate).

    ``/root/lodedb-src`` must mirror the repo layout (pyproject + readme/license
    + ``src/`` + the full ``third_party/turbovec/`` workspace) so the single
    maturin ``pip install`` can compile ``lodedb._turbovec`` against the sibling
    crate, exactly as ``uv sync`` sees it locally and in CI.
    """

    repo_root = Path(__file__).resolve().parents[2]
    image = (
        modal.Image.from_registry(
            "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
            add_python="3.11",
        )
        .apt_install("build-essential", "curl", "libopenblas-dev")
        .pip_install(*_LODEDB_RUNTIME_DEPENDENCIES, _CUPY_DEPENDENCY)
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal"
        )
        .add_local_file(
            str(repo_root / "pyproject.toml"),
            remote_path="/root/lodedb-src/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            str(repo_root / "README.md"), remote_path="/root/lodedb-src/README.md", copy=True
        )
        .add_local_file(
            str(repo_root / "LICENSE"), remote_path="/root/lodedb-src/LICENSE", copy=True
        )
        .add_local_file(str(repo_root / "NOTICE"), remote_path="/root/lodedb-src/NOTICE", copy=True)
        .add_local_dir(
            str(repo_root / "third_party" / "turbovec"),
            remote_path="/root/lodedb-src/third_party/turbovec",
            copy=True,
            ignore=["**/target/**", "**/__pycache__/**", "**/*.so", "**/*.pyd", "**/*.dylib"],
        )
        .add_local_dir(
            str(repo_root / "src"),
            remote_path="/root/lodedb-src/src",
            copy=True,
            ignore=["**/*.so", "**/*.pyd", "**/*.dylib", "**/__pycache__/**", "**/*.pyc"],
        )
        .run_commands(
            'PATH="$HOME/.cargo/bin:$PATH" python -m pip install --no-deps /root/lodedb-src'
        )
        .env({"PYTHONPATH": _REMOTE_BENCH_DIR})
    )
    return image.add_local_dir(
        str(Path(__file__).resolve().parent),
        remote_path=_REMOTE_BENCH_DIR,
        ignore=["**/__pycache__/**", "**/*.pyc", "results/**"],
    )


IMAGE = _build_image()
app = modal.App("lodedb-filtered-batch-bench", image=IMAGE)


@app.function(gpu="A10", cpu=16.0, memory=65536, timeout=7200)
def run_a10(spec: dict) -> dict:
    """Runs the filtered batch sweep on a Modal A10 (24 GB)."""

    from filtered_batch import run_filtered_batch_bench

    return run_filtered_batch_bench(**spec)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=7200)
def run_l40s(spec: dict) -> dict:
    """Runs the filtered batch sweep on a Modal L40S (48 GB)."""

    from filtered_batch import run_filtered_batch_bench

    return run_filtered_batch_bench(**spec)


def _print(result: dict) -> None:
    print(json.dumps(result, indent=2, sort_keys=True))
    cpu = result.get("cpu", {})
    isa = (
        "AVX512" if cpu.get("avx512") else ("AVX2" if cpu.get("avx2") else cpu.get("machine", "?"))
    )
    rows = result.get("rows") or []
    kernel = rows[0].get("native_backend") if rows else "?"
    print(
        f"\nhost CPU: {cpu.get('model', '?')}  [{isa}]   "
        f"model={result.get('model')} dim={result.get('native_dim')} "
        f"n_docs={result.get('n_docs')} kernel_backend={kernel}"
    )
    print(
        "\n  policy  batch  condition                       "
        " latency_ms  per_q_ms  gpu_status   backend  reason"
    )
    print("  " + "-" * 104)
    for row in rows:
        print(
            f"  {row['gpu_policy']:<6}  {row['batch_size']:>5}  {row['condition']:<30}  "
            f"{row['latency_ms']:>9.2f}  {row['per_query_ms']:>7.3f}  "
            f"{row['gpu_stage_one_status']:<11}  {row['stage_one_backend']:<7}  "
            f"{row['gpu_fallback_reason']}"
        )


@app.local_entrypoint()
def smoke() -> None:
    """Tiny A10 validation: corpus just over the 4096 cap."""

    spec = {"model": "minilm", "n_docs": 50_000, "batch_sizes": (8, 32, 128), "repeat": 5}
    _print(run_a10.remote(spec))


@app.local_entrypoint()
def a10() -> None:
    """Full A10 sweep (bge / 768-dim)."""

    spec = {"model": "bge", "n_docs": 200_000, "batch_sizes": (8, 32, 128, 256), "repeat": 7}
    _print(run_a10.remote(spec))


@app.local_entrypoint()
def l40s() -> None:
    """Full L40S sweep (bge / 768-dim, larger corpus)."""

    spec = {"model": "bge", "n_docs": 500_000, "batch_sizes": (8, 32, 128, 256), "repeat": 7}
    _print(run_l40s.remote(spec))
