"""Benchmark OpenKnowledge's two production embedding-provider choices on Modal.

Launch this module from the repository root. Its local benchmark import relies on
the repository-root ``benchmarks`` namespace.

    modal run benchmarks/openknowledge_embeddings/modal_bench.py::bench --docs 500
    modal run benchmarks/openknowledge_embeddings/modal_bench.py::bench --provider lodedb
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import modal

from benchmarks.openknowledge_embeddings.bench_core import run_benchmark

_REMOTE_SRC = "/root/lodedb-src"
_REMOTE_BENCHMARK_DIR = "/root/benchmarks/openknowledge_embeddings"
_KUBERNETES_WEBSITE = "/root/k8s-website"
KUBERNETES_WEBSITE_REVISION = "71d23f81e3479361befc94564e2b955860c03164"
_LODEDB_RUNTIME_DEPENDENCIES = (
    "numpy>=2.0.0,<3",
    "typer>=0.12.0",
    "pyyaml>=6.0.0",
    "onnxruntime>=1.20.0,<2",
    "transformers>=4.40.0,<5",
)


def _build_image() -> modal.Image:
    """Builds the local LodeDB source with its CPU ONNX embedding dependencies."""

    image = (
        modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
        .apt_install("build-essential", "curl", "git", "pkg-config", "libopenblas-dev")
        # transformers resolves tokenizers and huggingface-hub for MiniLM download and tokenization.
        .pip_install(*_LODEDB_RUNTIME_DEPENDENCIES)
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal"
        )
        .env({"PYTHONPATH": "/root"})
    )

    # Modal imports this module again in the container. These local paths only exist while
    # `modal run` constructs the image, so leave the source-copy steps behind this guard.
    if modal.is_local():
        repo_root = Path(__file__).resolve().parents[2]
        for relative_path in (
            "pyproject.toml",
            "README.md",
            "LICENSE",
            "NOTICE",
            "Cargo.toml",
            "Cargo.lock",
        ):
            image = image.add_local_file(
                str(repo_root / relative_path),
                remote_path=f"{_REMOTE_SRC}/{relative_path}",
                copy=True,
            )
        image = image.add_local_dir(
            str(repo_root / "third_party" / "turbovec"),
            remote_path=f"{_REMOTE_SRC}/third_party/turbovec",
            copy=True,
            ignore=["**/target/**", "**/__pycache__/**", "**/*.so", "**/*.pyd", "**/*.dylib"],
        )
        # The Python extension links lodedb-core by relative path. Its manifest inherits
        # package fields from the root Cargo workspace and lodedb-core links lodedb-gpu.
        image = image.add_local_dir(
            str(repo_root / "crates"),
            remote_path=f"{_REMOTE_SRC}/crates",
            copy=True,
            ignore=["**/target/**", "**/__pycache__/**"],
        )
        image = image.add_local_dir(
            str(repo_root / "src"),
            remote_path=f"{_REMOTE_SRC}/src",
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc", "**/*.so", "**/*.pyd", "**/*.dylib"],
        )
        image = image.add_local_dir(
            str(Path(__file__).resolve().parent),
            remote_path=_REMOTE_BENCHMARK_DIR,
            copy=True,
            ignore=["**/__pycache__/**", "**/*.pyc", "results/**"],
        )

    return image.run_commands(
        'PATH="$HOME/.cargo/bin:$PATH" python -m pip install --no-deps /root/lodedb-src',
        f"git init {_KUBERNETES_WEBSITE}",
        f"git -C {_KUBERNETES_WEBSITE} remote add origin https://github.com/kubernetes/website",
        f"git -C {_KUBERNETES_WEBSITE} fetch --depth 1 origin {KUBERNETES_WEBSITE_REVISION}",
        f"git -C {_KUBERNETES_WEBSITE} checkout --detach FETCH_HEAD",
    )


IMAGE = _build_image()
app = modal.App("lodedb-openknowledge-embeddings", image=IMAGE)
OPENAI_SECRET = modal.Secret.from_name("openai-embeddings-bench")


@app.function(cpu=8.0, memory=16384, timeout=3600, secrets=[OPENAI_SECRET])
def run_remote(docs: int, queries: int, provider: str) -> dict[str, Any]:
    """Runs selected providers in one CPU container, with LodeDB ordered first."""

    return run_benchmark(
        corpus_root=Path(_KUBERNETES_WEBSITE),
        docs=docs,
        query_count=queries,
        provider=provider,
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
        lodedb_store_path=Path("/root/okbench-store"),
        cpu_count=os.cpu_count(),
        corpus_revision=KUBERNETES_WEBSITE_REVISION,
    )


def _print_results(results: dict[str, Any]) -> None:
    print("BEGIN_RESULTS_JSON")
    print(json.dumps(results, indent=2, sort_keys=True))
    print("END_RESULTS_JSON")


@app.local_entrypoint()
def bench(docs: int = 0, queries: int = 100, provider: str = "both") -> None:
    """Runs the benchmark. ``docs=0`` selects the full sorted Kubernetes Markdown corpus."""

    _print_results(run_remote.remote(docs=docs, queries=queries, provider=provider))
