"""Run the LodeDB side of the LodeDB-vs-Trieve retrieval benchmark on a Modal GPU.

Two axes over the shared MiniLM-L6-v2 (384-d) dense model:

- Axis A: GovReport streamed + chunked to ~2M chunks, ingested vector-in, then
  build throughput, on-disk footprint, peak RSS, single-query p50/p95, batched
  qps, and index-fidelity recall@10 vs fp32 brute force.
- Axis B: MLDR English with real qrels, doc-level recall@{10,100} + nDCG@10 for
  both LodeDB vector and LodeDB hybrid.

The image is self-contained (patched TurboVec wheel + lodedb + sentence-transformers
+ datasets) and mirrors the govreport_scale recipe verbatim. Measurement lives in
``lodedb_bench.py`` (no Modal import), so the pipeline validates locally with its
synthetic smoke.

Launch from the repo root:

    # cheap pipeline validation (GovReport ~40K chunks + MLDR ~2000 docs):
    modal run benchmarks/trieve_vs_lodedb/modal_bench.py::smoke
    # full run (GovReport 2M chunks + full MLDR-en) on an L40S:
    modal run benchmarks/trieve_vs_lodedb/modal_bench.py::main
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# lodedb runtime deps must be present before `pip install --no-deps lodedb`, so installing
# lodedb never pulls a stock turbovec that would clobber the patched vendored wheel.
_LODEDB_RUNTIME_DEPENDENCIES = (
    "numpy>=2.0.0",
    "typer>=0.12.0",
    "sentence-transformers>=3.0.0",
    "pyyaml>=6.0.0",
)
_CUPY_DEPENDENCY = "cupy-cuda12x>=13.0.0"
_REMOTE_BENCH_ROOT = "/root/benchmarks"
_TRIEVE_DIR = f"{_REMOTE_BENCH_ROOT}/trieve_vs_lodedb"


def _build_image() -> modal.Image:
    """Self-contained CUDA image: patched TurboVec wheel + lodedb + this bench dir.

    Copied from benchmarks/govreport_scale/modal_bench.py verbatim (the
    patched-TurboVec-wheel + local-lodedb build with its maturin/third_party
    gotcha). Only the final mount targets this benchmark dir instead of the
    govreport pair.
    """

    repo_root = Path(__file__).resolve().parents[2]
    bench_root = Path(__file__).resolve().parent.parent
    mount_ignore = ["**/__pycache__/**", "**/*.pyc", "results/**", "docs/**"]
    image = (
        modal.Image.from_registry(
            "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
            add_python="3.11",
        )
        .apt_install("build-essential", "curl", "libopenblas-dev")
        .pip_install(
            *_LODEDB_RUNTIME_DEPENDENCIES,
            _CUPY_DEPENDENCY,
            "datasets>=2.19.0,<3.0.0",  # <3.0 keeps script-based MLDR loadable (trust_remote_code)
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal"
        )
        # Mirror the FULL repo workspace under /root/lodedb-src so maturin's manifest
        # (third_party/turbovec/turbovec-python) resolves its path deps:
        # ../../../crates/lodedb-core (native core; itself deps lodedb-gpu + turbovec)
        # and ../turbovec. lodedb-core inherits from the root [workspace], so the root
        # Cargo.toml + Cargo.lock + every workspace member under crates/ must be present
        # (the older recipes copied only third_party/turbovec and now fail). One
        # pip install then builds lodedb + the bundled _turbovec extension together.
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
        .add_local_file(
            str(repo_root / "Cargo.toml"), remote_path="/root/lodedb-src/Cargo.toml", copy=True
        )
        .add_local_file(
            str(repo_root / "Cargo.lock"), remote_path="/root/lodedb-src/Cargo.lock", copy=True
        )
        .add_local_dir(
            str(repo_root / "crates"),
            remote_path="/root/lodedb-src/crates",
            copy=True,
            ignore=["**/target/**", "**/__pycache__/**", "**/*.so", "**/*.pyd", "**/*.dylib"],
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
        .env({"PYTHONPATH": _TRIEVE_DIR})
    )
    # Final, non-copy mount (must come after all build steps): this benchmark dir,
    # so lodedb_bench is importable on the remote worker.
    return image.add_local_dir(
        str(bench_root / "trieve_vs_lodedb"), remote_path=_TRIEVE_DIR, ignore=mount_ignore
    )


IMAGE = _build_image()
app = modal.App("lodedb-trieve-bench", image=IMAGE)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=14400)
def run_axis_a(spec: dict) -> dict:
    """Embeds + ingests GovReport on the GPU, then runs the scale/latency/footprint axis."""

    from lodedb_bench import run_axis_a_govreport

    return run_axis_a_govreport(
        "/root/data/govreport-store",
        max_corpus=int(spec["max_corpus"]),
        n_query=int(spec.get("n_query", 1000)),
        k=int(spec.get("k", 10)),
        chunk_character_limit=int(spec.get("chunk_character_limit", 360)),
        ingest_batch=int(spec.get("ingest_batch", 4096)),
        batch_sizes=tuple(int(size) for size in spec.get("batch_sizes", (1, 16, 64, 256))),
        latency_iters=int(spec.get("latency_iters", 1000)),
        recall_sample=int(spec.get("recall_sample", 1000)),
        device="cuda",
    )


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=14400)
def run_axis_b(spec: dict) -> dict:
    """Embeds + ingests MLDR-en on the GPU, then runs the quality axis (vector + hybrid)."""

    from lodedb_bench import run_axis_b_mldr

    return run_axis_b_mldr(
        "/root/data/mldr-store",
        max_docs=int(spec["max_docs"]),
        k=int(spec.get("k", 100)),
        recall_ks=tuple(int(rk) for rk in spec.get("recall_ks", (10, 100))),
        ndcg_k=int(spec.get("ndcg_k", 10)),
        chunk_character_limit=int(spec.get("chunk_character_limit", 360)),
        ingest_batch=int(spec.get("ingest_batch", 4096)),
        device="cuda",
    )


def _write(bundle: dict, out: str) -> None:
    """Writes the results bundle locally and prints a one-line summary."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    axis_a = bundle.get("axis_a", {})
    axis_b = bundle.get("axis_b", {})
    print(
        f"[trieve-vs-lodedb] wrote {path} | "
        f"A: corpus={axis_a.get('corpus_count')} "
        f"ingest={axis_a.get('ingest_seconds')}s "
        f"p50={axis_a.get('single_query_latency_ms', {}).get('p50_ms')}ms | "
        f"B: docs={axis_b.get('corpus_doc_count')} "
        f"vector={axis_b.get('vector', {}).get('metrics')} "
        f"hybrid={axis_b.get('hybrid', {}).get('metrics')}"
    )


@app.local_entrypoint()
def main(out: str = "benchmarks/trieve_vs_lodedb/results/results.json") -> None:
    """Matched head-to-head vs Trieve: GovReport 200k chunks + MLDR ~1500 docs, both axes."""

    axis_a = run_axis_a.remote(
        {
            "max_corpus": 200_000,  # matched to the Trieve side (Trieve ingests ~70 chunks/s)
            "n_query": 1000,
            "k": 10,
            "chunk_character_limit": 360,
            "batch_sizes": (1, 16, 64, 256),
            "latency_iters": 1000,
            "recall_sample": 1000,
        }
    )
    axis_b = run_axis_b.remote(
        {
            "max_docs": 1500,  # all ~800 qrel-relevant docs + distractors to 1500 (~150k chunks)
            "k": 100,
            "recall_ks": (10, 100),
            "ndcg_k": 10,
            "chunk_character_limit": 360,
        }
    )
    _write({"smoke": False, "axis_a": axis_a, "axis_b": axis_b}, out)


@app.local_entrypoint()
def scale2m(out: str = "benchmarks/trieve_vs_lodedb/results/results_lodedb_2m.json") -> None:
    """LodeDB-only scale headline: GovReport at 2M chunks (axis A only)."""

    axis_a = run_axis_a.remote(
        {
            "max_corpus": 2_000_000,
            "n_query": 1000,
            "k": 10,
            "chunk_character_limit": 360,
            "batch_sizes": (1, 16, 64, 256),
            "latency_iters": 1000,
            "recall_sample": 1000,
        }
    )
    _write({"scale_2m": True, "axis_a": axis_a}, out)


@app.local_entrypoint()
def smoke(out: str = "benchmarks/trieve_vs_lodedb/results/results_smoke.json") -> None:
    """Cheap pipeline validation: GovReport ~40K chunks + MLDR ~2000 docs."""

    axis_a = run_axis_a.remote(
        {
            "max_corpus": 40_000,
            "n_query": 500,
            "k": 10,
            "chunk_character_limit": 360,
            "batch_sizes": (1, 16, 64),
            "latency_iters": 200,
            "recall_sample": 200,
        }
    )
    axis_b = run_axis_b.remote(
        {
            "max_docs": 2000,
            "k": 100,
            "recall_ks": (10, 100),
            "ndcg_k": 10,
            "chunk_character_limit": 360,
        }
    )
    _write({"smoke": True, "axis_a": axis_a, "axis_b": axis_b}, out)
