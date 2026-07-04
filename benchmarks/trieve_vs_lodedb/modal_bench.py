"""Run the LodeDB side of the LodeDB-vs-Trieve retrieval benchmark on a Modal GPU.

Two axes over the shared MiniLM-L6-v2 (384-d) dense model:

- Axis A: GovReport streamed + chunked to ~2M chunks, ingested vector-in, then
  build throughput, on-disk footprint, peak RSS, single-query p50/p95, batched
  qps, index-fidelity recall@10 vs fp32 brute force, and (v1.2.0) the opt-in ANN
  cluster-prune latency/recall trade against that exact scan.
- Axis B: MLDR English with real qrels, doc-level recall@{10,100} + nDCG@10 for
  both LodeDB vector and LodeDB hybrid (the hybrid/lexical path now serves through
  the v1.2.0 MaxScore BM25 index).

The image is self-contained (locally built lodedb 1.2.0 + sentence-transformers +
datasets) and mirrors the govreport_scale recipe verbatim. Measurement lives in
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

# As of v1.2.0 the built-in embedding runtime is opt-in (the [embeddings]/[torch] extras);
# the base install is a slim vector store (numpy/typer/pyyaml). This image builds lodedb from
# source with `pip install --no-deps` and installs the runtime deps itself, adding
# sentence-transformers as the embedding runtime so the "minilm" preset resolves. ONNX Runtime
# is deliberately not installed, so LodeDB's internal query embedding (axis B hybrid/lexical)
# takes the sentence-transformers fallback -- the same weights this bench uses to embed the
# corpus -- which keeps the re-run isolating the v1.2.0 code changes (MaxScore BM25, opt-in ANN)
# rather than an embedding-runtime swap. sentence-transformers is capped at <5 to match the
# shipped dependency cap (a 5.x memory regression, see docs/deployment-and-performance.md).
_LODEDB_RUNTIME_DEPENDENCIES = (
    "numpy>=2.0.0,<3",
    "typer>=0.12.0",
    "sentence-transformers>=3.0.0,<5",
    # Pin transformers below the next major too: sentence-transformers 4.x is not
    # compatible with transformers 5.x (the PreTrainedModel lazy-import moved), and
    # leaving it transitive lets pip resolve 5.x and break the ST import at encode time.
    # This mirrors the shipped [embeddings] cap in pyproject.toml.
    "transformers>=4.40.0,<5",
    "pyyaml>=6.0.0",
)
_CUPY_DEPENDENCY = "cupy-cuda12x>=13.0.0"
# NOTE on the vector scan: LodeDB's GPU-resident scan (crates/lodedb-gpu) JIT-compiles its
# top-k kernel through NVRTC (cudarc). The pytorch *runtime* base image ships libcublas but
# NOT libnvrtc, so cudarc's lazy dlopen panics (caught) and the scan falls back to the CPU
# SIMD kernel. That fallback is byte-identical in result, so every number here is exact; it
# just means the vector scan runs on the CPU on this image (the batched-throughput figures are
# CPU, not GPU). This is unchanged from the 1.1.0 run: crates/lodedb-gpu is identical across the
# two, so its "GPU scan" numbers were the same CPU fallback. Landing libnvrtc on the loader path
# turned out to disturb torch's own CUDA libraries (breaking the sentence-transformers import),
# so enabling the GPU scan here is left as a separate task on a libnvrtc-provisioned image.
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


def _log_result(axis: str, result: dict) -> None:
    """Prints the full result JSON to the worker's stdout under a grep-able prefix.

    The result is otherwise only delivered to the local client (via ``.remote()``'s return),
    so a dropped client connection loses it. Logging it worker-side means it survives in Modal's
    server logs and can be recovered with ``modal app logs <app>`` even if the client dies.
    """

    print(f"LODEDB_RESULT {axis} {json.dumps(result, sort_keys=True)}", flush=True)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=7200)
def run_axis_a(spec: dict) -> dict:
    """Embeds + ingests GovReport on the GPU, then runs the scale/latency/footprint axis."""

    from lodedb_bench import run_axis_a_govreport

    result = run_axis_a_govreport(
        "/root/data/govreport-store",
        max_corpus=int(spec["max_corpus"]),
        n_query=int(spec.get("n_query", 1000)),
        k=int(spec.get("k", 10)),
        chunk_character_limit=int(spec.get("chunk_character_limit", 360)),
        ingest_batch=int(spec.get("ingest_batch", 4096)),
        batch_sizes=tuple(int(size) for size in spec.get("batch_sizes", (1, 16, 64, 256))),
        latency_iters=int(spec.get("latency_iters", 1000)),
        recall_sample=int(spec.get("recall_sample", 1000)),
        ann_configs=tuple(dict(config) for config in spec.get("ann_configs", ())),
        device="cuda",
    )
    _log_result("axis_a", result)
    return result


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=7200)
def run_axis_b(spec: dict) -> dict:
    """Embeds + ingests MLDR-en on the GPU, then runs the quality axis (vector + hybrid)."""

    from lodedb_bench import run_axis_b_mldr

    result = run_axis_b_mldr(
        "/root/data/mldr-store",
        max_docs=int(spec["max_docs"]),
        k=int(spec.get("k", 100)),
        recall_ks=tuple(int(rk) for rk in spec.get("recall_ks", (10, 100))),
        ndcg_k=int(spec.get("ndcg_k", 10)),
        chunk_character_limit=int(spec.get("chunk_character_limit", 360)),
        ingest_batch=int(spec.get("ingest_batch", 4096)),
        device="cuda",
    )
    _log_result("axis_b", result)
    return result


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
            # Opt-in ANN (v1.2.0): core-default tuning (~sqrt(n) clusters, ~sqrt(clusters)
            # probes) plus a higher-recall point that probes more clusters, to trace the
            # latency/recall trade against the exact scan at the matched scale.
            "ann_configs": (
                {"label": "cluster-default"},
                {"label": "cluster-highrecall", "nprobe": 64},
            ),
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
            # ANN at 2M uses a small, explicit cluster count. The k-means build is single-threaded
            # and grows with n*clusters (issue #71): the default ~sqrt(n)=1414 clusters projects to
            # ~2.3 h, and even clusters=256 measured ~78 min (both hit the timeout). clusters=32
            # keeps the build tractable (~10 min, extrapolated from the 256 point) at the cost of
            # coarser pruning: nprobe=8 probes 8/32 of the clusters (~1/4 of the corpus per query),
            # so this is the honest, buildable ANN data point at 2M rather than a number that never
            # finishes. The exact scan is checkpointed before this, so it lands regardless.
            "ann_configs": ({"label": "cluster-32", "clusters": 32, "nprobe": 8},),
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
            "ann_configs": ({"label": "cluster-default"},),
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
