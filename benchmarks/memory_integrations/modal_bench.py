"""Run the LodeDB memory-integration benchmark on Modal.

Launch from the repo root:

    modal run benchmarks/memory_integrations/modal_bench.py::smoke      # tiny synthetic (A10)
    modal run benchmarks/memory_integrations/modal_bench.py::main_a10   # GovReport @ scale (A10)
    modal run benchmarks/memory_integrations/modal_bench.py::main_l40s  # GovReport @ scale (L40S)

Compares the LodeDB LangChain / LlamaIndex / mem0 adapters against each
framework's default and common vector stores (in-memory default, FAISS, Chroma,
Qdrant) on realistic workflows. Emits a metrics-only JSON bundle.

The image compiles the working tree's LodeDB (vendored TurboVec via maturin), so
it benchmarks this branch's code, and installs the three frameworks plus their
baseline stores at the versions this benchmark was validated against.
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

# Frameworks + their default/common vector stores, pinned to the versions this
# benchmark was validated against locally (see README). faiss-cpu is the baseline
# reference scan; embedding still runs on the GPU.
_FRAMEWORK_DEPENDENCIES = (
    "langchain==1.3.11",
    "langchain-community==0.4.2",
    "langchain-chroma==1.1.0",
    "langchain-qdrant==1.1.0",
    "llama-index-core==0.14.22",
    "llama-index-vector-stores-faiss==0.6.0",
    "llama-index-vector-stores-chroma==0.5.5",
    "llama-index-vector-stores-qdrant==0.10.1",
    "mem0ai==2.0.7",
    "qdrant-client==1.18.0",
    "chromadb==1.5.9",
    "faiss-cpu==1.14.3",
)
_REMOTE_BENCH_DIR = "/root/memory_integrations"


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
        .pip_install(*_FRAMEWORK_DEPENDENCIES)
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
        .env({"PYTHONPATH": _REMOTE_BENCH_DIR, "TOKENIZERS_PARALLELISM": "false"})
    )
    return image.add_local_dir(
        str(Path(__file__).resolve().parent),
        remote_path=_REMOTE_BENCH_DIR,
        ignore=["**/__pycache__/**", "**/*.pyc", "results/**"],
    )


IMAGE = _build_image()
app = modal.App("lodedb-memory-integrations-bench", image=IMAGE)


@app.function(gpu="A10", cpu=16.0, memory=65536, timeout=7200)
def run_suite_a10(spec: dict) -> dict:
    """Runs the memory-integration suite in the Modal A10 CUDA image."""

    from run import run_memory_integrations_suite

    return run_memory_integrations_suite(**spec)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=7200)
def run_suite_l40s(spec: dict) -> dict:
    """Runs the memory-integration suite in the Modal L40S CUDA image."""

    from run import run_memory_integrations_suite

    return run_memory_integrations_suite(**spec)


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
        "incremental_count": 30,
        "n_users": 50,
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
        "incremental_count": 20,
        "n_users": 10,
    }


def _write(bundle: dict, out: str) -> None:
    """Writes the metrics bundle locally and prints headline comparisons."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[memory-integrations] wrote {path} | docs={bundle.get('document_count')} "
        f"embed_device={bundle.get('embedding', {}).get('effective_device')}"
    )
    for framework, suite in bundle.get("suites", {}).items():
        default = suite.get("default_backend")
        by_name = {
            b["backend"]: b
            for b in suite.get("backends", [])
            if not b.get("skipped") and not b.get("failed")
        }
        lode = by_name.get("lodedb")
        base = by_name.get(default)
        if not lode:
            continue
        line = (
            f"  {framework}: lodedb recall={lode['query']['recall_at_k']} "
            f"footprint={lode['footprint_bytes'] / 1024:.0f}KB "
            f"durable_add_p50={lode['incremental_add']['p50_ms']}ms"
        )
        if base:
            ratio = base["footprint_bytes"] / max(1, lode["footprint_bytes"])
            line += (
                f" | {default} footprint={base['footprint_bytes'] / 1024:.0f}KB "
                f"durable_add_p50={base['incremental_add']['p50_ms']}ms "
                f"(lodedb {ratio:.1f}x smaller)"
            )
        print(line)


@app.local_entrypoint()
def smoke(out: str = "benchmarks/memory_integrations/results/results_smoke.json") -> None:
    """Tiny synthetic A10 validation run before the full GovReport suite."""

    _write(run_suite_a10.remote(_smoke_spec("/root/mi-smoke")), out)


@app.local_entrypoint()
def main_a10(out: str = "benchmarks/memory_integrations/results/results_a10.json") -> None:
    """Full GovReport memory-integration suite on an A10."""

    _write(run_suite_a10.remote(_full_spec("/root/mi-a10")), out)


@app.local_entrypoint()
def main_l40s(out: str = "benchmarks/memory_integrations/results/results_l40s.json") -> None:
    """Full GovReport memory-integration suite on an L40S."""

    _write(run_suite_l40s.remote(_full_spec("/root/mi-l40s")), out)
