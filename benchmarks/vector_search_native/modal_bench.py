"""Run the native vector-only throughput sweep (GPU default vs plain TurboVec) on Modal.

Launch from the repo root:

    modal run benchmarks/vector_search_native/modal_bench.py::smoke_a10
    modal run benchmarks/vector_search_native/modal_bench.py::main_a10
    modal run benchmarks/vector_search_native/modal_bench.py::main_l40s

`smoke_*` is a tiny CUDA validation (small dim/corpus). `main_*` runs the full
sweep (dim 1536, 100,000 vectors, 4-bit) matching the vanilla-vs-augmented graph's
config, but over the native path with the current defaults.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# A CUDA -devel image ships libnvrtc + libcublas (dlopened by the native cudarc
# scan); the driver comes from the Modal GPU host. libopenblas-dev is required at
# build time because the vendored TurboVec crate's build.rs links system OpenBLAS.
_CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
_REMOTE_SRC = "/root/lodedb-src"
_REMOTE_BENCH = "/root/vector_search_native"

# Base runtime deps for a vector-only LodeDB (no embedding model needed).
_RUNTIME_DEPS = ("numpy>=2.0.0", "typer>=0.12.0", "pyyaml>=6.0.0")


def _build_image() -> modal.Image:
    """Builds a CUDA image with the Rust toolchain and the LodeDB wheel."""

    image = (
        modal.Image.from_registry(_CUDA_IMAGE, add_python="3.11")
        .apt_install("build-essential", "curl", "pkg-config", "libopenblas-dev")
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal"
        )
        .pip_install(*_RUNTIME_DEPS)
    )
    # Copy the workspace and build the lodedb wheel (maturin compiles
    # third_party/turbovec/turbovec-python, which path-depends on crates/lodedb-core
    # -> crates/lodedb-gpu, so the whole workspace must be present). Guarded with
    # modal.is_local(): Modal re-imports this module in the container to locate the
    # function, where the local tree is absent.
    if modal.is_local():
        repo_root = Path(__file__).resolve().parents[2]
        # Cargo.toml + Cargo.lock are the ROOT workspace manifest: crates/lodedb-core
        # inherits `edition`/`version` from `[workspace.package]`, so the maturin build
        # of turbovec-python (which path-depends on lodedb-core) needs them present.
        for rel in ("pyproject.toml", "Cargo.toml", "Cargo.lock", "README.md", "LICENSE", "NOTICE"):
            image = image.add_local_file(
                str(repo_root / rel), remote_path=f"{_REMOTE_SRC}/{rel}", copy=True
            )
        for rel in ("crates", "third_party/turbovec", "src"):
            image = image.add_local_dir(
                str(repo_root / rel),
                remote_path=f"{_REMOTE_SRC}/{rel}",
                copy=True,
                ignore=["**/target/**", "**/__pycache__/**", "**/*.so", "**/*.pyd", "**/*.dylib"],
            )
    image = image.run_commands(
        f'PATH="$HOME/.cargo/bin:$PATH" python -m pip install --no-deps {_REMOTE_SRC}'
    ).env(
        # Belt-and-suspenders: the -devel image already ldconfigs the CUDA libs, but
        # pin the loader path so the native scan reliably dlopens nvrtc/cublas.
        {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64", "PYTHONPATH": _REMOTE_BENCH}
    )
    return image.add_local_dir(
        str(Path(__file__).resolve().parent),
        remote_path=_REMOTE_BENCH,
        ignore=["**/__pycache__/**", "**/*.pyc", "results/**"],
    )


IMAGE = _build_image()
app = modal.App("lodedb-vector-search-native", image=IMAGE)


def _run(spec: dict) -> dict:
    """Runs the sweep in the container and returns its summary."""

    import os

    os.environ.setdefault("LODEDB_GPU_DEBUG", "1")
    from vector_search_native import run_vector_search_native

    return run_vector_search_native(**spec)


@app.function(gpu="A10", cpu=8.0, memory=32768, timeout=3600)
def run_a10(spec: dict) -> dict:
    return _run(spec)


@app.function(gpu="L40S", cpu=8.0, memory=49152, timeout=3600)
def run_l40s(spec: dict) -> dict:
    return _run(spec)


def _full_spec(output_dir: str) -> dict:
    return {
        "output_dir": output_dir,
        "dim": 1536,
        "n": 100_000,
        "bit_width": 4,
        "batch_sizes": (1, 16, 64, 256, 1024),
        "query_count": 1024,
        "top_k": 10,
        "repeats": 5,
    }


def _smoke_spec(output_dir: str) -> dict:
    return {
        "output_dir": output_dir,
        "dim": 128,
        "n": 2000,
        "bit_width": 4,
        "batch_sizes": (1, 16, 64),
        "query_count": 64,
        "top_k": 10,
        "repeats": 2,
    }


def _write(bundle: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[vector-search-native] wrote {path} | rows={len(bundle.get('rows', []))}")


@app.local_entrypoint()
def smoke_a10(out: str = "benchmarks/vector_search_native/results/results_smoke_a10.json") -> None:
    _write(run_a10.remote(_smoke_spec("/root/vsn-smoke-a10")), out)


@app.local_entrypoint()
def main_a10(out: str = "benchmarks/vector_search_native/results/results_a10.json") -> None:
    _write(run_a10.remote(_full_spec("/root/vsn-a10")), out)


@app.local_entrypoint()
def main_l40s(out: str = "benchmarks/vector_search_native/results/results_l40s.json") -> None:
    _write(run_l40s.remote(_full_spec("/root/vsn-l40s")), out)
