"""Run the vanilla-vs-augmented TurboVec benchmark on a Modal GPU (A10 / L40S).

The augmented series needs CUDA (CuPy + the patched TurboVec reconstruction APIs),
so the whole matrix runs in one GPU container: the vanilla CPU SIMD scan on its
vCPUs, the augmented GPU-resident exact scan on its GPU. Real OpenAI-DBpedia
embeddings are streamed for the recall axis (100K corpus + 1K held-out queries);
speed/memory/update use synthetic vectors.

The image is built inline and is self-contained: it builds the vendored TurboVec
wheel from ``third_party/turbovec/turbovec-python`` (rustup + maturin), installs
``cupy-cuda12x``, installs the local ``lodedb`` package, and mounts this benchmark
directory so the measurement core (``turbovec_vva_bench`` / ``turbovec_vva_runner``,
dev-only sibling scripts) is importable. The benchmark core also runs directly on
any CUDA host without Modal (see ``turbovec_vva_runner.py``).

Launch from the repo root (relative image paths resolve there):

    # Full matrix on Modal A10:
    modal run benchmarks/gpu_vanilla_vs_augmented/modal_bench.py \
        --out benchmarks/gpu_vanilla_vs_augmented/results/results_a10.json

    # GPU-ceiling variant on an L40S (48 GB):
    modal run benchmarks/gpu_vanilla_vs_augmented/modal_bench.py::ceiling \
        --out benchmarks/gpu_vanilla_vs_augmented/results/results_l40s.json

Then render diagrams locally:

    python benchmarks/gpu_vanilla_vs_augmented/diagrams.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# Runtime libraries the benchmark core needs in the container. ``lodedb.engine``
# is imported through the top-level ``lodedb`` package, so its declared runtime
# deps must be present before lodedb is installed with ``--no-deps`` (below).
# Installing lodedb's deps via PyPI would otherwise pull a stock ``turbovec`` and
# clobber the patched vendored wheel.
_LODEDB_RUNTIME_DEPENDENCIES = (
    "numpy>=2.0.0",
    "typer>=0.12.0",
    "sentence-transformers>=3.0.0",
    "pyyaml>=6.0.0",
)
_CUPY_DEPENDENCY = "cupy-cuda12x>=13.0.0"
_REMOTE_BENCH_DIR = "/root/gpu_vanilla_vs_augmented"


def _build_image() -> modal.Image:
    """Builds a self-contained CUDA image: patched TurboVec wheel + CuPy + lodedb.

    Mirrors a from-source install of the vendored crate: the ``turbovec-python``
    crate has a path dependency on the sibling ``turbovec`` core crate, so the
    whole ``third_party/turbovec`` tree is copied and the wheel is built with
    maturin (via rustup). The crate's ``build.rs`` links system OpenBLAS on Linux,
    so ``libopenblas-dev`` must be present at build time (and its runtime ``.so``
    afterwards). All build steps (``copy=True`` adds + ``run_commands``) come
    before the final non-copy ``add_local_dir`` mounts, which Modal requires to be
    last.
    """

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
            "datasets>=3.0.0",  # stream OpenAI-DBpedia embeddings for the recall axis
            "h5py>=3.0.0",      # read the ann-benchmarks GloVe hdf5
        )
        # Build + install the patched vendored TurboVec wheel from source. Copy the
        # whole crate tree (the python crate path-depends on the sibling core crate).
        .add_local_dir(
            str(repo_root / "third_party" / "turbovec"),
            remote_path="/root/turbovec-src",
            copy=True,
            ignore=["**/target/**", "**/__pycache__/**", "**/*.so", "**/*.pyd", "**/*.dylib"],
        )
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal",
            'PATH="$HOME/.cargo/bin:$PATH" python -m pip install '
            "--no-deps /root/turbovec-src/turbovec-python",
        )
        # Install the local lodedb package WITHOUT deps (already pip-installed above)
        # so the patched turbovec wheel is never replaced by a PyPI build.
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
            str(repo_root / "src"),
            remote_path="/root/lodedb-src/src",
            copy=True,
            ignore=["**/*.so", "**/*.pyd", "**/*.dylib", "**/__pycache__/**", "**/*.pyc"],
        )
        .run_commands("python -m pip install --no-deps /root/lodedb-src")
        .env({"PYTHONPATH": _REMOTE_BENCH_DIR})
    )
    # Final, non-copy mount of the benchmark core (must come after all build steps).
    return image.add_local_dir(
        str(Path(__file__).resolve().parent),
        remote_path=_REMOTE_BENCH_DIR,
        ignore=["**/__pycache__/**", "**/*.pyc", "results/**", "docs/**"],
    )


IMAGE = _build_image()

app = modal.App("turbovec-vanilla-vs-augmented", image=IMAGE)


def _prep_openai(dim: int, *, n_corpus: int, n_query: int, out_dir: str) -> dict | None:
    """Streams n_corpus+n_query OpenAI-DBpedia embeddings to .npy; returns a dataset spec."""

    import numpy as np
    from datasets import load_dataset

    name = f"Qdrant/dbpedia-entities-openai3-text-embedding-3-large-{dim}-1M"
    col = f"text-embedding-3-large-{dim}-embedding"
    need = n_corpus + n_query
    print(f"[modal_bench] streaming {need} rows from {name}", flush=True)
    rows = []
    try:
        ds = load_dataset(name, split="train", streaming=True)
        for rec in ds:
            rows.append(np.asarray(rec[col], dtype=np.float32))
            if len(rows) >= need:
                break
    except Exception as exc:  # noqa: BLE001
        print(f"[modal_bench] dataset {name} failed: {exc!r}", flush=True)
        return None
    if len(rows) < need:
        print(f"[modal_bench] only got {len(rows)}/{need} rows for d={dim}", flush=True)
        if len(rows) < n_corpus + 100:
            return None
        n_query = len(rows) - n_corpus
    arr = np.stack(rows)
    vec_path = str(Path(out_dir) / f"openai-{dim}-corpus.npy")
    qry_path = str(Path(out_dir) / f"openai-{dim}-queries.npy")
    np.save(vec_path, arr[:n_corpus])
    np.save(qry_path, arr[n_corpus : n_corpus + n_query])
    print(f"[modal_bench] saved openai-{dim}: corpus={n_corpus} queries={n_query}", flush=True)
    return {
        "name": f"openai-{dim}",
        "dim": dim,
        "vectors_path": vec_path,
        "queries_path": qry_path,
        "bit_widths": [2, 4],
    }


def _prep_glove(*, n_corpus: int, n_query: int, out_dir: str) -> dict | None:
    """Downloads ann-benchmarks GloVe-200 (angular) hdf5 -> .npy; returns a dataset spec.

    GloVe d=200 is the low-dimensional regime where the uint8-LUT quantization error
    is largest, so it is where the augmented exact-reconstruction scan can recover
    recall the vanilla quantized scan loses. Vectors are normalized at load time
    (cosine / "angular"); ground truth is computed exactly in the cell runner.
    """

    import shutil
    import urllib.request

    import h5py
    import numpy as np

    path = str(Path(out_dir) / "glove-200-angular.hdf5")
    # ann-benchmarks.com serves browsers/curl but 403s the default python-urllib
    # User-Agent (verified), so send a browser UA. Try https then http.
    urls = (
        "https://ann-benchmarks.com/glove-200-angular.hdf5",
        "http://ann-benchmarks.com/glove-200-angular.hdf5",
    )
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126"}
    try:
        if not Path(path).exists():
            downloaded = False
            for url in urls:
                try:
                    print(f"[modal_bench] downloading GloVe hdf5 from {url}", flush=True)
                    req = urllib.request.Request(url, headers=headers)
                    with urllib.request.urlopen(req, timeout=600) as resp, open(path, "wb") as fh:
                        shutil.copyfileobj(resp, fh)
                    downloaded = True
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[modal_bench] GloVe url failed ({url}): {exc!r}", flush=True)
            if not downloaded:
                return None
        with h5py.File(path, "r") as f:
            train = np.asarray(f["train"][:n_corpus], dtype=np.float32)
            test = np.asarray(f["test"][:n_query], dtype=np.float32)
    except Exception as exc:  # noqa: BLE001
        print(f"[modal_bench] GloVe read failed: {exc!r}", flush=True)
        return None
    vec_path = str(Path(out_dir) / "glove-corpus.npy")
    qry_path = str(Path(out_dir) / "glove-queries.npy")
    np.save(vec_path, train)
    np.save(qry_path, test)
    print(f"[modal_bench] saved glove-200: corpus={train.shape} queries={test.shape}", flush=True)
    return {
        "name": "glove-200",
        "dim": 200,
        "vectors_path": vec_path,
        "queries_path": qry_path,
        "bit_widths": [2, 4],
    }


def _prep_and_run(spec: dict, require_backend: str) -> dict:
    """Shared container body: optional host-CPU guard, prep real datasets, run matrix.

    ``require_backend`` (e.g. ``"avx512bw"``) makes the call **bail in seconds** if the
    host's TurboVec CPU kernel is not the requested one. Modal assigns the host CPU, so
    the launcher can relaunch for a fresh host without paying for a full run on the wrong
    baseline. ``""`` or ``"any"`` accepts whatever host.
    """

    from turbovec_vva_bench import machine_info
    from turbovec_vva_runner import run_all

    info = machine_info()
    backend = info.get("turbovec_native_backend")
    if require_backend and require_backend != "any" and backend != require_backend:
        print(
            f"[modal_bench] host backend={backend!r}, need {require_backend!r}; bailing cheap",
            flush=True,
        )
        return {"_wrong_host": True, "backend": backend, "machine": info}

    data_dir = "/root/data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    datasets = []
    for dim in (1536, 3072):
        ds = _prep_openai(dim, n_corpus=spec["parity_n"], n_query=spec["queries"], out_dir=data_dir)
        if ds is not None:
            datasets.append(ds)
    glove = _prep_glove(n_corpus=spec["parity_n"], n_query=spec["queries"], out_dir=data_dir)
    if glove is not None:
        datasets.append(glove)
    if datasets:
        spec["datasets"] = datasets  # recall uses real embeddings; else synthetic fallback
    names = [d["name"] for d in datasets]
    print(f"[modal_bench] backend={backend} recall datasets: {names}", flush=True)
    return run_all(spec)


@app.function(gpu="A10", cpu=16.0, memory=65536, timeout=3600)
def run_benchmark(spec: dict, require_backend: str = "") -> dict:
    """Full matrix on an A10 (the default; CPU baseline + augmented GPU on one host)."""

    return _prep_and_run(spec, require_backend)


@app.function(gpu="L40S", cpu=16.0, memory=65536, timeout=3600)
def run_benchmark_l40s(spec: dict, require_backend: str = "") -> dict:
    """GPU-ceiling variant on an L40S (48 GB). Shows where the exact-GEMM path pulls ahead."""

    return _prep_and_run(spec, require_backend)


@app.function(gpu="A10", cpu=8.0, memory=32768, timeout=1800)
def run_glove_recall(spec: dict) -> dict:
    """Preps GloVe-200 and runs ONLY its recall cells (vanilla vs augmented).

    Recall is CPU-arch-independent and the GPU is the A10 either way, so this runs
    on any host and merges into an AVX-512 full-run results file without disturbing
    its speed/memory/update numbers.
    """

    from turbovec_vva_bench import machine_info, run_cell

    data_dir = "/root/data"
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    glove = _prep_glove(n_corpus=spec["parity_n"], n_query=spec["queries"], out_dir=data_dir)
    if glove is None:
        return {"_glove_failed": True, "machine": machine_info()}
    rows = []
    for bit in glove["bit_widths"]:
        cell = run_cell({
            "axis": "recall", "dim": glove["dim"], "bit_width": bit, "n": spec["parity_n"],
            "queries": spec["queries"], "k": spec["k"], "seed": spec.get("seed", 0),
            "batch_size": spec["batch_size"], "include_gpu": True,
            "vectors_path": glove["vectors_path"], "queries_path": glove["queries_path"],
        })
        rows.append({
            "dataset": glove["name"], "dim": glove["dim"], "bit_width": bit,
            "vanilla": cell.get("recall_vanilla"), "augmented": cell.get("recall_augmented"),
        })
    return {"recall": rows, "machine": machine_info()}


@app.local_entrypoint()
def glove(out: str = "benchmarks/gpu_vanilla_vs_augmented/results/results_a10.json") -> None:
    """Runs GloVe-200 recall on A10 and merges it into an existing results file."""

    from turbovec_vva_runner import full_spec

    res = run_glove_recall.remote(full_spec())
    if res.get("_glove_failed"):
        print("[modal_bench] GloVe prep failed (download blocked); results file unchanged")
        return
    path = Path(out)
    bundle = json.loads(path.read_text())
    recall = [
        r for r in bundle["axes"]["recall"]
        if not str(r.get("dataset", "")).startswith("glove")
    ]
    recall.extend(res["recall"])
    bundle["axes"]["recall"] = recall
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    print(f"[modal_bench] merged {len(res['recall'])} GloVe recall rows into {path}")


@app.local_entrypoint()
def main(
    out: str = "benchmarks/gpu_vanilla_vs_augmented/results/results_a10.json",
    require_backend: str = "avx512bw",
) -> None:
    """Runs the A10 benchmark and writes the results bundle locally.

    ``require_backend`` defaults to ``avx512bw`` for a representative AVX-512 server
    CPU baseline; if the assigned A10 host lacks it the function bails cheap and this
    prints WRONG_HOST (leaving any existing results file untouched) so the run can be
    relaunched for a fresh host. Pass ``--require-backend any`` to accept any host.
    """

    from turbovec_vva_runner import full_spec

    spec = full_spec()
    bundle = run_benchmark.remote(spec, require_backend=require_backend)
    if bundle.get("_wrong_host"):
        print(
            f"[modal_bench] WRONG_HOST backend={bundle.get('backend')} "
            f"(wanted {require_backend}); relaunch for a fresh host"
        )
        return
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    machine = bundle.get("machine", {})
    print(f"[modal_bench] wrote {out_path} in {bundle.get('wall_seconds', 0):.0f}s")
    print(f"[modal_bench] backend={machine.get('turbovec_native_backend')} gpu-run complete")


@app.local_entrypoint()
def ceiling(
    out: str = "benchmarks/gpu_vanilla_vs_augmented/results/results_l40s.json",
    require_backend: str = "any",
) -> None:
    """Runs the full matrix on an L40S (GPU ceiling) and writes a separate results file.

    Accepts any host CPU by default (the point is the GPU ceiling, not the CPU baseline).
    """

    from turbovec_vva_runner import full_spec

    bundle = run_benchmark_l40s.remote(full_spec(), require_backend=require_backend)
    if bundle.get("_wrong_host"):
        print(f"[modal_bench] WRONG_HOST backend={bundle.get('backend')}; relaunch")
        return
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    machine = bundle.get("machine", {})
    print(
        f"[modal_bench] wrote {out_path} gpu=L40S backend={machine.get('turbovec_native_backend')} "
        f"wall={bundle.get('wall_seconds', 0):.0f}s"
    )
