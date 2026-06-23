"""Run the hybrid persistence benchmark on Modal (A10) against real CUDA.

The image is built from the local ``src/`` tree, so launch it from the checkout
that has the persistent-postings change (``index_text``). The GPU-resident scan
serves ``search_many`` automatically when CuPy is present, so the vector path is
exercised on real hardware while the lexical pass and commit journaling run on
CPU.

Launch from the repo root:

    modal run benchmarks/hybrid/modal_bench.py::smoke   # tiny validation
    modal run benchmarks/hybrid/modal_bench.py::a10     # full A10 run

The image recipe mirrors benchmarks/gpu_patch/modal_bench.py: a CUDA PyTorch
base plus CuPy plus the maturin-built vendored TurboVec wheel and LodeDB from
local src.
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
_REMOTE_BENCH_DIR = "/root/hybrid"


def _build_image() -> modal.Image:
    """Builds a CUDA image with LodeDB compiled from local src (maturin layout)."""

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
        .add_local_file(
            str(repo_root / "NOTICE"), remote_path="/root/lodedb-src/NOTICE", copy=True
        )
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
app = modal.App("lodedb-hybrid-persist-bench", image=IMAGE)


@app.function(gpu="A10", cpu=16.0, memory=65536, timeout=7200)
def run_persist_a10(spec: dict) -> dict:
    """Runs the persistence benchmark on a Modal A10 (24 GB)."""

    from persist_bench import run_persist_bench

    return run_persist_bench(**spec)


def _full_spec() -> dict:
    """Full run: 20K-doc corpus, 200 incremental commits, batched GPU queries."""

    return {
        "scale": 20_000,
        "plant_every": 50,
        "ingest_batch": 2_000,
        "incremental": 200,
        "query_batch": 64,
        "query_count": 20,
        "dim": 384,
        "top_k": 10,
    }


def _smoke_spec() -> dict:
    """Tiny validation run to exercise the CUDA image and APIs end to end."""

    return {
        "scale": 1_000,
        "plant_every": 25,
        "ingest_batch": 500,
        "incremental": 20,
        "query_batch": 16,
        "query_count": 5,
        "dim": 384,
        "top_k": 10,
    }


def _write(bundle: dict, out: str) -> None:
    """Writes the metrics-only bundle locally and prints a one-line summary."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    overhead = bundle.get("commit_overhead", {})
    machine = bundle.get("machine", {})
    print(
        f"[hybrid-persist-bench] wrote {path} | gpu={machine.get('gpu_name')} "
        f"commit_overhead={overhead.get('relative_pct'):.1f}%"
    )


@app.local_entrypoint()
def smoke(out: str = "benchmarks/hybrid/results/persist_smoke.json") -> None:
    """Tiny A10 validation run before the full corpus."""

    _write(run_persist_a10.remote(_smoke_spec()), out)


@app.local_entrypoint()
def a10(out: str = "benchmarks/hybrid/results/persist_a10.json") -> None:
    """Full persistence benchmark on an A10."""

    _write(run_persist_a10.remote(_full_spec()), out)
