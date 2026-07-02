"""Run the Trieve side of the LodeDB-vs-Trieve benchmark on one Modal GPU container.

Modal's docker-in-docker is CPU-only, so we do NOT use docker-compose. The whole
Trieve stack (Postgres, Redis, Qdrant, the GPU model server, trieve-server and the
ingestion-worker) runs as native child processes inside one @app.function on an
L40S. The image builds the two Trieve binaries from source at the pinned SHA with
only ``--features runtime-env`` (dropping ``hallucination-detection`` so no libtorch
is needed), downloads the standalone Qdrant v1.12.2 binary, and installs Postgres,
Redis, and the Python model-serving + dataset deps.

The heavy, cached layer is the Rust build (~30-40 min the first time). Everything
after it (env, boot, bootstrap, ingest, query) lives in ``orchestrator.py`` /
``trieve_bench.py`` so it can be validated locally without Modal.

Launch from the repo root:

    # smoke: boot stack + ingest ~1k chunks + semantic/hybrid search + Server-Timing
    modal run benchmarks/trieve_vs_lodedb/trieve_modal.py::smoke
    # full: GovReport 2M chunks + full MLDR-en
    modal run benchmarks/trieve_vs_lodedb/trieve_modal.py::main
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# Pinned Trieve commit (last MIT version; see PLAN.md).
TRIEVE_REPO = "https://github.com/devflowinc/trieve.git"
TRIEVE_SHA = "a99b21e23f21025757b44efb594676c4a7b7495f"
QDRANT_VERSION = "v1.12.2"
QDRANT_URL = (
    f"https://github.com/qdrant/qdrant/releases/download/{QDRANT_VERSION}/"
    "qdrant-x86_64-unknown-linux-gnu.tar.gz"
)
RUST_TOOLCHAIN = "1.87.0"

_REMOTE_BENCH_ROOT = "/root/benchmarks"
_TRIEVE_DIR = f"{_REMOTE_BENCH_ROOT}/trieve_vs_lodedb"
_TRIEVE_CLONE = "/opt/trieve"
_TRIEVE_SERVER_BIN = f"{_TRIEVE_CLONE}/server/target/release/trieve-server"
_INGESTION_WORKER_BIN = f"{_TRIEVE_CLONE}/server/target/release/ingestion-worker"
_TRIEVE_SERVER_DIR = f"{_TRIEVE_CLONE}/server"


def _stack_env() -> dict[str, str]:
    """Returns the full single-container env block trieve-server + workers read.

    Every ``get_env!`` var trieve touches on the chunk/search path is set. Model URLs
    point at the localhost model server; sentinel-triggering values (empty
    EMBEDDING_BASE_URL, k8s reranker sentinel without RERANKER_SERVER_ORIGIN) are
    avoided. OIDC points at the model server's discovery stub. ClickHouse/analytics
    off, S3 dummy (not on the chunk path).
    """

    model_origin = "http://localhost:7070"
    return {
        # Core datastores.
        "DATABASE_URL": "postgres://postgres:password@localhost:5432/trieve",
        "REDIS_URL": "redis://:thisredispasswordisverysecureandcomplex@localhost:6379",
        "REDIS_PASSWORD": "thisredispasswordisverysecureandcomplex",
        "REDIS_CONNECTIONS": "10",
        "QDRANT_URL": "http://localhost:6334",
        "QDRANT_API_KEY": "qdrant_pass",
        "CREATE_QDRANT_COLLECTIONS": "true",
        "QUANTIZE_VECTORS": "false",  # Qdrant at f32 (PLAN: report the honest quant trade)
        "REPLICATION_FACTOR": "1",  # single node
        "QDRANT_SHARD_COUNT": "1",
        "VECTOR_SIZES": "384,512,768,1024,1536,3072",
        # Admin bootstrap + limits.
        "ADMIN_API_KEY": "admin",
        "UNLIMITED": "true",
        "USE_ANALYTICS": "false",
        "BATCH_CHUNK_LIMIT": "120",
        # Model servers (dense/sparse/rerank all served by our one GPU process).
        "OPENAI_API_KEY": "sk-local-not-used",
        "OPENAI_BASE_URL": model_origin,
        "EMBEDDING_SERVER_ORIGIN": model_origin,
        "SPARSE_SERVER_DOC_ORIGIN": model_origin,
        "SPARSE_SERVER_QUERY_ORIGIN": model_origin,
        "RERANKER_SERVER_ORIGIN": model_origin,
        "BM25_ACTIVE": "false",
        # Crypto + server url.
        "SECRET_KEY": "0123401234012340123401234012340123401234012340123401234012340123",
        "SALT": "goodsaltisveryyummy",
        "BASE_SERVER_URL": "http://localhost:8090",
        # OIDC discovery stub (model server serves /.well-known/openid-configuration).
        "OIDC_ISSUER_URL": model_origin,
        "OIDC_CLIENT_ID": "bench",
        "OIDC_CLIENT_SECRET": "bench-secret",
        "OIDC_AUTH_REDIRECT_URL": f"{model_origin}/authorize",
        # S3 dummy (not exercised by the chunk path).
        "S3_ENDPOINT": "http://localhost:9000",
        "S3_ACCESS_KEY": "minioadmin",
        "S3_SECRET_KEY": "minioadmin",
        "S3_BUCKET": "trieve",
        "AWS_REGION": "",
        # Orchestrator-side hints for locating binaries.
        "TRIEVE_SERVER_BIN": _TRIEVE_SERVER_BIN,
        "INGESTION_WORKER_BIN": _INGESTION_WORKER_BIN,
        "TRIEVE_CLONE_DIR": _TRIEVE_SERVER_DIR,
        "QDRANT_BIN": "/usr/local/bin/qdrant",
        "RUST_LOG": "info",
        "PYTHONUNBUFFERED": "1",
    }


def _build_image() -> modal.Image:
    """Builds the CUDA image: torch base + rust-built Trieve bins + qdrant + pg/redis.

    Base is pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime so the model server's torch
    has CUDA. The Rust build of both Trieve bins is the big cached layer; it uses only
    ``--features runtime-env`` (no hallucination-detection, hence no libtorch).
    """

    bench_root = Path(__file__).resolve().parent.parent
    mount_ignore = ["**/__pycache__/**", "**/*.pyc", "results/**", "docs/**"]
    return (
        modal.Image.from_registry(
            "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
            add_python="3.11",
        )
        # Postgres pulls tzdata, which prompts interactively and hangs the build unless
        # apt runs non-interactively with a preset timezone.
        .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
        .apt_install(
            # Trieve build deps (from server/Dockerfile.server).
            "pkg-config",
            "libssl-dev",
            "libpq-dev",
            "g++",
            "build-essential",
            "curl",
            "git",
            "ca-certificates",
            # Native datastores.
            "postgresql",
            "postgresql-contrib",
            "redis-server",
            "redis-tools",
        )
        # Rust toolchain pinned to 1.87 (matches server/Dockerfile.server FROM rust:1.87).
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            f"sh -s -- -y --default-toolchain {RUST_TOOLCHAIN} --profile minimal",
        )
        .env({"PATH": "/root/.cargo/bin:/usr/local/bin:/usr/bin:/bin"})
        # Standalone Qdrant binary (v1.12.2) -> /usr/local/bin/qdrant.
        .run_commands(
            f"curl -L -o /tmp/qdrant.tar.gz {QDRANT_URL}",
            "tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin",
            "chmod +x /usr/local/bin/qdrant",
            "rm /tmp/qdrant.tar.gz",
        )
        # Clone Trieve at the pinned SHA and build both bins with runtime-env only.
        # This is the expensive, cached layer.
        .run_commands(
            f"git clone {TRIEVE_REPO} {_TRIEVE_CLONE}",
            f"cd {_TRIEVE_CLONE} && git checkout {TRIEVE_SHA}",
            f"cd {_TRIEVE_SERVER_DIR} && cargo build --release --features runtime-env "
            "--bin trieve-server --bin ingestion-worker",
            gpu=None,
        )
        # Python deps for the model server + dataset parity. torch/cuda already in base.
        .pip_install(
            "fastapi>=0.110.0",
            "uvicorn>=0.29.0",
            "transformers>=4.40.0",
            "sentence-transformers>=3.0.0",
            "numpy>=1.26.0",
            "datasets>=2.19.0,<3.0.0",  # <3.0 keeps script-based MLDR loadable (trust_remote_code)
            "lodedb",  # for lodedb.engine.core.chunk_text (identical chunking)
        )
        .env({"PYTHONPATH": _TRIEVE_DIR})
        .add_local_dir(
            str(bench_root / "trieve_vs_lodedb"), remote_path=_TRIEVE_DIR, ignore=mount_ignore
        )
    )


IMAGE = _build_image()
app = modal.App("lodedb-trieve-trieveside-bench", image=IMAGE)


def _apply_stack_env() -> None:
    """Sets the stack env vars into os.environ for the child processes to inherit."""

    import os

    os.environ.update(_stack_env())


def _dump_service_logs() -> None:
    """Prints the tail of each service log so a remote failure shows the server side."""

    from pathlib import Path as _Path

    log_dir = _Path("/var/lib/trieve/logs")
    for name in ("trieve-server", "ingestion-worker", "model_server", "qdrant", "postgres"):
        path = log_dir / f"{name}.log"
        if not path.exists():
            continue
        try:
            tail = "\n".join(path.read_text(errors="replace").splitlines()[-50:])
        except Exception as exc:  # noqa: BLE001
            tail = f"<unreadable: {exc}>"
        print(f"\n===== {name}.log (tail) =====\n{tail}", flush=True)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=14400)
def run_trieve_bench(spec: dict) -> dict:
    """Boots the Trieve stack, bootstraps, ingests + queries both axes, returns results.

    ``spec`` mirrors the LodeDB side: ``axis_a`` (max_corpus, n_query, k, ...) and
    ``axis_b`` (max_docs, k, recall_ks, ndcg_k). Whichever blocks are present are run.
    Ingest uses the same corpus builders (GovReport + MLDR via ``lodedb_bench``), so the
    corpus is byte-identical to the LodeDB run.
    """

    _apply_stack_env()

    from trieve_stack import orchestrator
    import trieve_bench

    handle = orchestrator.boot_and_bootstrap(embedding_size=384, model_name="all-MiniLM-L6-v2")
    base_url = handle["base_url"]
    dataset_id = handle["dataset_id"]

    results: dict = {"handle": handle, "trieve_sha": TRIEVE_SHA}

    try:
        axis_a = spec.get("axis_a")
        if axis_a:
            results["axis_a"] = trieve_bench.run_axis_a_govreport(
                base_url,
                dataset_id,
                max_corpus=int(axis_a["max_corpus"]),
                n_query=int(axis_a.get("n_query", 1000)),
                k=int(axis_a.get("k", 10)),
                chunk_character_limit=int(axis_a.get("chunk_character_limit", 360)),
                latency_iters=int(axis_a.get("latency_iters", 1000)),
                batch_sizes=tuple(int(size) for size in axis_a.get("batch_sizes", (1, 16, 64, 256))),
            )

        axis_b = spec.get("axis_b")
        if axis_b:
            # Axis B needs its own dataset (different corpus). Create a second dataset under
            # the same org so its qrel docids do not collide with GovReport tracking_ids.
            second = orchestrator.bootstrap_org_and_dataset(
                embedding_size=384, model_name="all-MiniLM-L6-v2"
            )
            results["handle_axis_b"] = second
            results["axis_b"] = trieve_bench.run_axis_b_mldr(
                second["base_url"],
                second["dataset_id"],
                max_docs=int(axis_b["max_docs"]),
                k=int(axis_b.get("k", 100)),
                recall_ks=tuple(int(rk) for rk in axis_b.get("recall_ks", (10, 100))),
                ndcg_k=int(axis_b.get("ndcg_k", 10)),
                chunk_character_limit=int(axis_b.get("chunk_character_limit", 360)),
            )
    except BaseException:
        _dump_service_logs()
        raise

    return results


def _write(bundle: dict, out: str) -> None:
    """Writes the results bundle locally and prints a one-line summary."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    axis_a = bundle.get("axis_a", {})
    axis_b = bundle.get("axis_b", {})
    print(
        f"[trieve-bench] wrote {path} | "
        f"A: corpus={axis_a.get('corpus_count')} "
        f"ingest={axis_a.get('ingest_seconds')}s "
        f"p50={axis_a.get('single_query_latency_ms', {}).get('p50_ms')}ms "
        f"qdrant_p50={axis_a.get('server_timing', {}).get('qdrant_ms', {}).get('p50_ms')}ms | "
        f"B: docs={axis_b.get('corpus_doc_count')} "
        f"semantic={axis_b.get('vector', {}).get('metrics')} "
        f"hybrid={axis_b.get('hybrid', {}).get('metrics')}"
    )


@app.local_entrypoint()
def main(out: str = "benchmarks/trieve_vs_lodedb/results/trieve_results.json") -> None:
    """Full run: GovReport at 2M chunks + full MLDR-en, both axes on an L40S."""

    bundle = run_trieve_bench.remote(
        {
            "axis_a": {
                "max_corpus": 200_000,  # matched head-to-head size (Trieve ingests ~70 chunks/s)
                "n_query": 1000,
                "k": 10,
                "chunk_character_limit": 360,
                "batch_sizes": (1, 16, 64, 256),
                "latency_iters": 1000,
            },
            "axis_b": {
                "max_docs": 1500,  # all ~800 qrel-relevant docs + distractors to 1500 (~150k chunks)
                "k": 100,
                "recall_ks": (10, 100),
                "ndcg_k": 10,
                "chunk_character_limit": 360,
            },
        }
    )
    _write({"smoke": False, **bundle}, out)


@app.local_entrypoint()
def smoke(out: str = "benchmarks/trieve_vs_lodedb/results/trieve_results_smoke.json") -> None:
    """Cheap pipeline validation: boot + ingest ~1k GovReport chunks + MLDR ~200 docs."""

    bundle = run_trieve_bench.remote(
        {
            "axis_a": {
                "max_corpus": 1_000,
                "n_query": 50,
                "k": 10,
                "chunk_character_limit": 360,
                "batch_sizes": (1, 8, 16),
                "latency_iters": 50,
            },
            "axis_b": {
                "max_docs": 30,
                "k": 100,
                "recall_ks": (10, 100),
                "ndcg_k": 10,
                "chunk_character_limit": 360,
            },
        }
    )
    _write({"smoke": True, **bundle}, out)
