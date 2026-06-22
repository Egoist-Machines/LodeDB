"""Run the LodeDB graph-memory benchmark on Modal.

Launch from the repo root:

    modal run benchmarks/graph_memory/modal_bench.py::smoke      # tiny synthetic (A10)
    modal run benchmarks/graph_memory/modal_bench.py::main_a10   # GovReport @ scale (A10)
    modal run benchmarks/graph_memory/modal_bench.py::main_l40s  # GovReport @ scale (L40S)

Measures vector-in vs text-in ingest/query, predicate-filter latency, and graph
traversal + hybrid retrieval. Emits a metrics-only JSON bundle.
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
_REMOTE_BENCH_DIR = "/root/graph_memory"


def _build_image() -> modal.Image:
    """Builds a self-contained CUDA image that compiles this branch's LodeDB."""

    repo_root = Path(__file__).resolve().parents[2]
    image = (
        modal.Image.from_registry(
            "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
            add_python="3.11",
        )
        .apt_install("build-essential", "curl", "libopenblas-dev")
        .pip_install(*_LODEDB_RUNTIME_DEPENDENCIES, _CUPY_DEPENDENCY, "datasets>=3.0.0")
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
        # maturin needs the full vendored turbovec workspace under the build dir,
        # exactly as `uv sync`/CI see it, or it errors that the manifest path
        # third_party/turbovec/turbovec-python/Cargo.toml does not exist.
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
app = modal.App("lodedb-graph-memory-bench", image=IMAGE)


@app.function(gpu="A10", cpu=16.0, memory=65536, timeout=7200)
def run_suite_a10(spec: dict) -> dict:
    """Runs the graph-memory suite in the Modal A10 CUDA image."""

    from graph_memory_bench import run_graph_memory_suite

    return run_graph_memory_suite(**spec)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=7200)
def run_suite_l40s(spec: dict) -> dict:
    """Runs the graph-memory suite in the Modal L40S CUDA image."""

    from graph_memory_bench import run_graph_memory_suite

    return run_graph_memory_suite(**spec)


def _full_spec(output_dir: str) -> dict:
    """GovReport-backed suite at scale (CUDA embedding)."""

    return {
        "output_dir": output_dir,
        "dataset_name": "govreport",
        "model": "minilm",
        "max_documents": 50000,
        "query_count": 256,
        "device": "cuda",
        "top_k": 10,
        "graph_nodes": 50000,
        "avg_degree": 16,
        "hops": 2,
        "seed_queries": 256,
    }


def _smoke_spec(output_dir: str) -> dict:
    """Tiny synthetic validation spec (no dataset download)."""

    return {
        "output_dir": output_dir,
        "dataset_name": "synthetic",
        "model": "minilm",
        "max_documents": 500,
        "query_count": 32,
        "device": "cuda",
        "top_k": 10,
        "graph_nodes": 500,
        "avg_degree": 8,
        "hops": 2,
        "seed_queries": 32,
    }


def _write(bundle: dict, out: str) -> None:
    """Writes the metrics bundle locally and prints headline numbers."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    vector_in = bundle.get("vector_in", {})
    graph = bundle.get("graph", {})
    print(
        f"[graph-memory] wrote {path} | docs={bundle.get('document_count')} "
        f"ingest_speedup_vectorin={vector_in.get('ingest_speedup_vectorin_over_textin')}x "
        f"overlap={vector_in.get('topk_overlap_mean')} "
        f"khop_p50={graph.get('khop_latency', {}).get('p50_ms')}ms "
        f"hybrid_p50={graph.get('hybrid_latency', {}).get('p50_ms')}ms"
    )


@app.local_entrypoint()
def smoke(out: str = "benchmarks/graph_memory/results/results_smoke.json") -> None:
    """Tiny synthetic A10 validation run before the full GovReport suite."""

    _write(run_suite_a10.remote(_smoke_spec("/root/graph-memory-smoke")), out)


@app.local_entrypoint()
def main_a10(out: str = "benchmarks/graph_memory/results/results_a10.json") -> None:
    """Full GovReport graph-memory suite on an A10."""

    _write(run_suite_a10.remote(_full_spec("/root/graph-memory-a10")), out)


@app.local_entrypoint()
def main_l40s(out: str = "benchmarks/graph_memory/results/results_l40s.json") -> None:
    """Full GovReport graph-memory suite on an L40S."""

    _write(run_suite_l40s.remote(_full_spec("/root/graph-memory-l40s")), out)
