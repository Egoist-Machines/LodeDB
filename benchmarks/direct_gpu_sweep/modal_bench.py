"""Run the LodeDB direct TurboVec GPU sweep on Modal.

Launch from the repo root:

    modal run benchmarks/direct_gpu_sweep/modal_bench.py::smoke
    modal run benchmarks/direct_gpu_sweep/modal_bench.py::smoke_a10
    modal run benchmarks/direct_gpu_sweep/modal_bench.py::main
    modal run benchmarks/direct_gpu_sweep/modal_bench.py::main_a10
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
_REMOTE_BENCH_DIR = "/root/direct_gpu_sweep"


def _build_image() -> modal.Image:
    """Builds a self-contained CUDA image for the direct GPU sweep."""

    repo_root = Path(__file__).resolve().parents[2]
    image = (
        modal.Image.from_registry(
            "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
            add_python="3.11",
        )
        .apt_install("build-essential", "curl", "libopenblas-dev")
        .pip_install(
            *_LODEDB_RUNTIME_DEPENDENCIES,
            _CUPY_DEPENDENCY,
            "datasets>=3.0.0",
        )
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
        # LodeDB's build backend is maturin: it compiles the vendored crate
        # third_party/turbovec into the lodedb._turbovec extension, and
        # turbovec-python depends on the sibling turbovec core via path = "../turbovec".
        # So the full workspace must live under the build dir (/root/lodedb-src),
        # exactly as `uv sync` sees it locally and in CI; otherwise maturin errors
        # with "manifest path third_party/turbovec/turbovec-python/Cargo.toml does not exist".
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
        # One build, as CI does: LodeDB plus the bundled lodedb._turbovec extension.
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
app = modal.App("lodedb-direct-gpu-sweep", image=IMAGE)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=7200)
def run_sweep_l40s(spec: dict) -> dict:
    """Runs the direct GPU sweep in the Modal L40S CUDA image."""

    from direct_gpu_sweep import run_direct_gpu_sweep

    return run_direct_gpu_sweep(**spec)


@app.function(gpu="A10", cpu=16.0, memory=65536, timeout=7200)
def run_sweep_a10(spec: dict) -> dict:
    """Runs the direct GPU sweep in the Modal A10 CUDA image."""

    from direct_gpu_sweep import run_direct_gpu_sweep

    return run_direct_gpu_sweep(**spec)


def _full_spec(output_dir: str) -> dict:
    """Returns the GovReport5K launch sweep spec."""

    return {
        "output_dir": output_dir,
        "dataset_name": "GovReport5K",
        "model": "minilm",
        "query_count": 1024,
        "top_k": 100,
        "batch_sizes": "1,2,4,8,16,32,64,128,256,512,1024",
        "query_repeats": 5,
        "device": "cuda",
        "expect_gpu_rows": True,
    }


def _smoke_spec(output_dir: str) -> dict:
    """Returns the small CUDA validation spec."""

    return {
        "output_dir": output_dir,
        "dataset_name": "GovReport64",
        "model": "minilm",
        "max_documents": 64,
        "query_count": 16,
        "top_k": 16,
        "batch_sizes": "1,2,4,8,16",
        "query_repeats": 2,
        "device": "cuda",
        "expect_gpu_rows": True,
    }


def _write(bundle: dict, out: str) -> None:
    """Writes the results bundle locally."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[direct-gpu-sweep] wrote {path} | rows={bundle.get('row_count')} "
        f"docs={bundle.get('document_count')} queries={bundle.get('query_count')}"
    )


@app.local_entrypoint()
def main(out: str = "benchmarks/direct_gpu_sweep/results/results_l40s.json") -> None:
    """Full LodeDB-owned direct GPU sweep on GovReport5K."""

    _write(run_sweep_l40s.remote(_full_spec("/root/direct-gpu-sweep-results-l40s")), out)


@app.local_entrypoint()
def main_a10(out: str = "benchmarks/direct_gpu_sweep/results/results_a10.json") -> None:
    """Full LodeDB-owned direct GPU sweep on GovReport5K using an A10."""

    _write(run_sweep_a10.remote(_full_spec("/root/direct-gpu-sweep-results-a10")), out)


@app.local_entrypoint()
def smoke(out: str = "benchmarks/direct_gpu_sweep/results/results_smoke.json") -> None:
    """Small L40S CUDA validation run before the full GovReport5K sweep."""

    _write(
        run_sweep_l40s.remote(_smoke_spec("/root/direct-gpu-sweep-smoke-l40s")),
        out,
    )


@app.local_entrypoint()
def smoke_a10(out: str = "benchmarks/direct_gpu_sweep/results/results_smoke_a10.json") -> None:
    """Small A10 CUDA validation run before the full GovReport5K sweep."""

    _write(
        run_sweep_a10.remote(_smoke_spec("/root/direct-gpu-sweep-smoke-a10")),
        out,
    )
