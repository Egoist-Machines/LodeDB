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
    # additional embedded / local vector stores (LangChain suite)
    "lancedb==0.33.0",
    "sqlite-vec==0.1.9",
    "langchain-postgres==0.0.17",
    "psycopg==3.3.4",
    "pgserver==0.1.4",  # embedded Postgres + pgvector, no separate service
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
        # The root Cargo workspace manifest + lock. crates/lodedb-core inherits
        # version/edition/lints from [workspace.package]/[workspace.lints] here, so
        # cargo must find this workspace root when maturin resolves the
        # turbovec-python -> crates/lodedb-core path dependency (otherwise the
        # build fails with "failed to find a workspace root").
        .add_local_file(
            str(repo_root / "Cargo.toml"), remote_path="/root/lodedb-src/Cargo.toml", copy=True
        )
        .add_local_file(
            str(repo_root / "Cargo.lock"), remote_path="/root/lodedb-src/Cargo.lock", copy=True
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
        # The native Rust core: turbovec-python's Cargo.toml depends on
        # crates/lodedb-core (path = ../../../crates/lodedb-core) and registers the
        # native CoreEngine onto the _turbovec extension, so the maturin build
        # needs the crates workspace present or it errors on the missing path dep.
        .add_local_dir(
            str(repo_root / "crates"),
            remote_path="/root/lodedb-src/crates",
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


def _register_cuda_libs() -> None:
    """Makes the image's bundled CUDA libraries resolvable by cudarc's dlopen.

    torch and the nvidia-*-cu12 wheels ship libcublas/libnvrtc as versioned
    sonames under site-packages, which are not on the loader's default search
    path; cudarc dlopens them by soname and would otherwise fall back to the CPU
    kernel. Discovering the dirs from the running interpreter (so the paths match
    whichever python Modal uses), registering them with ldconfig, and refreshing
    the cache before the native core's first GPU touch lets the GPU path engage.
    Best effort: any failure simply leaves the native core on its CPU kernel."""

    import glob
    import os
    import subprocess

    dirs: set[str] = set()
    try:
        import nvidia  # nvidia-*-cu12 wheels expose a `nvidia` namespace package

        for base in nvidia.__path__:
            for lib_dir in glob.glob(os.path.join(base, "*", "lib")):
                if glob.glob(os.path.join(lib_dir, "lib*.so*")):
                    dirs.add(lib_dir)
    except Exception as exc:  # noqa: BLE001 - discovery is best effort
        print(f"[gpu-libs] nvidia wheels not found: {exc}")
    try:
        import torch

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            dirs.add(torch_lib)
    except Exception as exc:  # noqa: BLE001 - discovery is best effort
        print(f"[gpu-libs] torch libs not found: {exc}")

    if not dirs:
        print("[gpu-libs] no CUDA lib dirs discovered; native core stays on CPU")
        return
    try:
        with open("/etc/ld.so.conf.d/lodedb-cuda.conf", "w", encoding="utf-8") as handle:
            handle.write("\n".join(sorted(dirs)) + "\n")
        subprocess.run(["ldconfig"], check=False)
        print(f"[gpu-libs] registered CUDA lib dirs: {sorted(dirs)}")
    except Exception as exc:  # noqa: BLE001 - never fail the run over this
        print(f"[gpu-libs] ldconfig registration failed: {exc}")


def _log_native_core() -> None:
    """Confirms the freshly built wheel exposes the native Rust core, so the run
    reflects the native offering (not a stale-extension Python fallback)."""

    import lodedb._turbovec as turbovec

    present = hasattr(turbovec, "CoreEngine")
    version = turbovec.native_core_version() if hasattr(turbovec, "native_core_version") else "?"
    print(f"[native-core] CoreEngine present={present} version={version}")
    if not present:
        raise RuntimeError("native CoreEngine missing from the built extension")


def _run_suite(spec: dict) -> dict:
    """Runs the suite, honoring an optional ``native_core`` spec key that forces
    LodeDB's engine on (Rust) or off (Python) for this container."""

    import os

    _register_cuda_libs()
    _log_native_core()
    # One-time stderr confirmation that the native GPU scan actually engaged on the
    # rust container (a no-op for the Python container, which logs its own backend).
    os.environ["LODEDB_GPU_DEBUG"] = "1"
    spec = dict(spec)
    mode = spec.pop("native_core", None)
    if mode is not None:
        os.environ["LODEDB_NATIVE_CORE"] = mode
        # Default WAL commit mode both sides (the fast path). With write-through the
        # rust container is the sole writer: native appends its own durable WAL
        # record per add and skips the Python engine entirely, while the generation
        # publish is deferred to checkpoint/close.
        os.environ.pop("LODEDB_COMMIT_MODE", None)
        if mode == "on":
            os.environ["LODEDB_NATIVE_CORE_WRITE"] = "on"
        else:
            os.environ.pop("LODEDB_NATIVE_CORE_WRITE", None)
    from run import run_memory_integrations_suite

    return run_memory_integrations_suite(**spec)


@app.function(gpu="A10", cpu=16.0, memory=65536, timeout=7200)
def run_suite_a10(spec: dict) -> dict:
    """Runs the memory-integration suite in the Modal A10 CUDA image."""

    return _run_suite(spec)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=7200)
def run_suite_l40s(spec: dict) -> dict:
    """Runs the memory-integration suite in the Modal L40S CUDA image."""

    return _run_suite(spec)


def _gpu_smoke(n: int = 17500, dim: int = 384, nq: int = 64, k: int = 10) -> dict:
    """Validates the native GPU scan end to end: engagement, GPU-vs-CPU parity,
    and a rough timing, on synthetic precomputed vectors (no dataset/model).

    Native vector reads are authoritative on; LODEDB_GPU_DEBUG=1 makes the Rust
    core print a one-time "GPU scan engaged" line, so the function logs prove the
    GPU path served rather than silently falling back to the CPU kernel."""

    import os
    import tempfile
    import time

    import numpy as np

    _register_cuda_libs()
    _log_native_core()
    os.environ["LODEDB_NATIVE_CORE"] = "on"
    os.environ.pop("LODEDB_NATIVE_CORE_WRITE", None)
    os.environ["LODEDB_GPU_DEBUG"] = "1"

    from lodedb import LodeDB

    rng = np.random.default_rng(7)
    vectors = rng.standard_normal((n, dim)).astype("float32")
    queries = list(rng.standard_normal((nq, dim)).astype("float32"))

    tmp = tempfile.mkdtemp()
    db = LodeDB(path=os.path.join(tmp, "lode"), vector_dim=dim, device="cpu")
    db.add_vectors_many(
        [{"vector": vectors[i], "id": str(i)} for i in range(n)],
        normalize=False,
    )

    def run_batch() -> list[list[str]]:
        return [[h.id for h in hits] for hits in db.search_many_by_vector(queries, k=k)]

    def timed(label: str) -> tuple[list[list[str]], float]:
        run_batch()  # warm
        start = time.perf_counter()
        result: list[list[str]] = []
        for _ in range(100):
            result = run_batch()
        ms_per_query = (time.perf_counter() - start) / 100 / nq * 1000.0
        print(f"[gpu-smoke] {label}: {ms_per_query:.4f} ms/query")
        return result, ms_per_query

    os.environ["LODEDB_GPU_DIRECT_TURBOVEC"] = "auto"
    gpu_res, gpu_ms = timed("gpu-on ")
    os.environ["LODEDB_GPU_DIRECT_TURBOVEC"] = "off"
    cpu_res, cpu_ms = timed("gpu-off")

    # The GPU path scores exact-reconstructed rows while the CPU kernel scores the
    # uint8 LUT quantization of the same rows, so the two legitimately diverge at
    # the top-k boundary. Recall against the brute-force float ground truth is the
    # meaningful check: the GPU path should recall at least as well as the CPU
    # kernel (it is the more faithful estimator), and both should stay high.
    queries_mat = np.stack(queries)
    truth_idx = np.argsort(-(queries_mat @ vectors.T), axis=1)[:, :k]
    truth = [{str(int(slot)) for slot in row} for row in truth_idx]

    def recall(result: list[list[str]]) -> float:
        hit = sum(len(set(r) & t) for r, t in zip(result, truth, strict=True))
        return hit / (nq * k)

    out = {
        "n": n,
        "dim": dim,
        "nq": nq,
        "k": k,
        "gpu_ms_per_query": round(gpu_ms, 4),
        "cpu_ms_per_query": round(cpu_ms, 4),
        "gpu_recall_vs_float": round(recall(gpu_res), 4),
        "cpu_recall_vs_float": round(recall(cpu_res), 4),
        "gpu_cpu_overlap": round(
            sum(len(set(a) & set(b)) for a, b in zip(gpu_res, cpu_res, strict=True)) / (nq * k),
            4,
        ),
    }
    print(f"[gpu-smoke] result: {out}")
    return out


@app.function(gpu="L40S", cpu=8.0, memory=32768, timeout=1800)
def run_gpu_smoke_l40s() -> dict:
    """Runs the native GPU-scan smoke on a Modal L40S."""

    return _gpu_smoke()


def _textstore_repro(n: int = 20000) -> str:
    """Runs the LodeDB LangChain backend over the real GovReport corpus in
    isolation (native write-through, CUDA, ingest -> query -> persist ->
    incremental -> reopen). The reopen re-reads and checksums the native-written
    document text base; real document text exercises the non-ASCII canonical-JSON
    path that synthetic ASCII corpora never reach."""

    import json
    import os
    import tempfile
    import traceback
    from pathlib import Path

    _register_cuda_libs()
    os.environ["LODEDB_NATIVE_CORE"] = "on"
    os.environ["LODEDB_NATIVE_CORE_WRITE"] = "on"

    from bench_langchain import _LodeDBDriver, _make_cached_embeddings
    from common import (
        embed_corpus,
        exact_topk,
        load_corpus,
        rag_metadata,
        run_core_phases,
        uuid_ids,
    )

    docs, queries = load_corpus("govreport", n, 64)
    embedded = embed_corpus(docs, queries, model="minilm", device="cuda")
    cache = {text: embedded.doc_vectors[i].tolist() for i, text in enumerate(docs)}
    emb = _make_cached_embeddings(cache, embedded)
    ids = uuid_ids(len(docs), "langchain")
    inc_ids = uuid_ids(30, "langchain-inc")
    metadatas = [rag_metadata(i) for i in range(len(docs))]
    truth = exact_topk(ids, embedded.doc_vectors, embedded.query_vectors, 10)
    workdir = Path(tempfile.mkdtemp())
    driver = _LodeDBDriver("lodedb", workdir, emb, model="minilm", device="cuda")
    try:
        metrics = run_core_phases(
            driver,
            embedded,
            ids,
            docs,
            metadatas,
            truth,
            k=10,
            incremental_count=30,
            incremental_ids=inc_ids,
            batch_size=64,
        )
        out = "OK reopen=" + json.dumps(metrics.get("reopen"))
    except Exception:
        out = "EXCEPTION:\n" + traceback.format_exc()
    print(out)
    return out


@app.function(gpu="L40S", cpu=8.0, memory=65536, timeout=1800)
def run_textstore_repro_l40s() -> str:
    """Runs the GovReport text-store checksum repro on a Modal L40S."""

    return _textstore_repro()


@app.function(gpu="L40S", cpu=8.0, memory=32768, timeout=1800)
def run_gpu_unit_tests_l40s() -> str:
    """Runs the lodedb-gpu cargo tests on a real GPU.

    The parity test compares the GPU top-k against an exact CPU reference over raw
    rows (no TurboVec quantization), so it isolates the GEMM + top-k kernel math
    from the quantization divergence the CPU serving kernel carries."""

    import os
    import subprocess

    _register_cuda_libs()
    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.cargo/bin") + ":" + env.get("PATH", "")
    proc = subprocess.run(
        ["cargo", "test", "-p", "lodedb-gpu", "--release", "--", "--nocapture"],
        cwd="/root/lodedb-src",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    print(proc.stdout)
    print(proc.stderr)
    return f"returncode={proc.returncode}\n{proc.stdout}\n{proc.stderr}"


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
        "batch_size": 64,
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
        "batch_size": 64,
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


def _compare_spec(output_dir: str) -> dict:
    """GovReport suite sized to engage the CUDA GPU-resident scan within a
    reasonable two-pass (Rust + Python) run."""

    return {
        "output_dir": output_dir,
        "dataset_name": "govreport",
        "model": "minilm",
        "max_documents": 20000,
        "query_count": 256,
        "device": "cuda",
        "top_k": 10,
        "incremental_count": 30,
        "n_users": 50,
        "batch_size": 64,
    }


def _write_compare(bundle: dict, out: str) -> None:
    """Writes the {rust, python} bundle and prints the LodeDB Rust-vs-Python and
    LodeDB-vs-baseline headline per framework."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True), encoding="utf-8")
    rust = bundle.get("rust", {})
    python = bundle.get("python", {})
    print(f"[compare] wrote {path} | docs={rust.get('document_count')}")

    def backends(b, framework):
        return {
            row["backend"]: row
            for row in b.get("suites", {}).get(framework, {}).get("backends", [])
            if not row.get("skipped") and not row.get("failed")
        }

    for framework in rust.get("suites", {}):
        r = backends(rust, framework).get("lodedb")
        p = backends(python, framework).get("lodedb")
        if not r or not p:
            continue
        print(f"  [{framework}] LodeDB Rust vs Python:")
        for metric, path_keys, unit in (
            ("single-query p50", ("query", "p50_ms"), "ms"),
            ("batch per-query", ("query_batch", "mean_per_query_ms"), "ms"),
            ("durable add p50", ("incremental_add", "p50_ms"), "ms"),
        ):
            rv = r.get(path_keys[0], {}).get(path_keys[1])
            pv = p.get(path_keys[0], {}).get(path_keys[1])
            if rv is not None and pv is not None:
                speedup = pv / rv if rv else float("inf")
                print(
                    f"    {metric:18s}: rust={rv:.4f}{unit} "
                    f"python={pv:.4f}{unit} ({speedup:.2f}x)"
                )
        print(
            f"    batch path         : rust={r.get('query_batch', {}).get('path')!r} "
            f"python={p.get('query_batch', {}).get('path')!r}"
        )
        baselines = backends(rust, framework)
        others = ", ".join(
            f"{name} batch/q="
            f"{baselines[name].get('query_batch', {}).get('mean_per_query_ms'):.4f}ms"
            for name in baselines
            if name != "lodedb"
        )
        if others:
            print(f"    baselines          : {others}")


@app.local_entrypoint()
def gpu_smoke_l40s() -> None:
    """Confirms the native GPU scan engages and matches the CPU kernel on an L40S."""

    result = run_gpu_smoke_l40s.remote()
    print(f"[gpu-smoke] {result}")


@app.local_entrypoint()
def gpu_unit_tests_l40s() -> None:
    """Runs the lodedb-gpu cargo parity tests on a real L40S GPU."""

    print(run_gpu_unit_tests_l40s.remote())


@app.local_entrypoint()
def textstore_repro_l40s() -> None:
    """Runs the GovReport text-store write-through + reopen repro on an L40S."""

    print(run_textstore_repro_l40s.remote())


@app.local_entrypoint()
def compare_l40s(
    out: str = "benchmarks/memory_integrations/results/results_compare_l40s.json",
) -> None:
    """LodeDB Rust (native on) vs Python (native off) with baselines.

    Each engine runs in its OWN fresh L40S container, so cold-start warmup (GPU
    context, cupy, embedding model) is paid independently and neither engine gets
    a second-run advantage. This is the fair counterpart to a single-container
    two-pass run, where the engine that runs second inherits a warm GPU."""

    rust = run_suite_l40s.remote(dict(_compare_spec("/root/mi-rust"), native_core="on"))
    python = run_suite_l40s.remote(dict(_compare_spec("/root/mi-python"), native_core="off"))
    _write_compare({"rust": rust, "python": python}, out)


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
