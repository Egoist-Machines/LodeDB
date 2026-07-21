"""Run the GovReport-at-scale vanilla-vs-augmented TurboVec benchmark on a Modal GPU.

Embeds GovReport chunks (MiniLM) up to ~1M vectors on the GPU, then measures recall@k vs
fp32 brute force (the vanilla **uint8-LUT** scan + the augmented **fp16-reconstruction**
scan, both over the same 4-bit index) and the CPU scan's throughput ceiling (vanilla
single-/all-threads vs the augmented GPU path) across a 100K -> 1M corpus-size sweep, plus a
batch sweep at 1M.

The image is self-contained (patched TurboVec wheel + CuPy + lodedb + sentence-transformers +
datasets) and mounts BOTH this benchmark dir and the sibling ``gpu_vanilla_vs_augmented`` dir,
whose measurement core (``turbovec_vva_bench`` / ``turbovec_vva_runner``) this benchmark reuses.

Launch from the repo root:

    # validate the pipeline cheaply first (~40K chunks):
    modal run benchmarks/govreport_scale/modal_bench.py::smoke
    # full 100K/500K/1M run on an L40S:
    modal run benchmarks/govreport_scale/modal_bench.py::main
    # render charts from the results JSON:
    python benchmarks/govreport_scale/diagrams.py
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
_GOVREPORT_DIR = f"{_REMOTE_BENCH_ROOT}/govreport_scale"
_VVA_DIR = f"{_REMOTE_BENCH_ROOT}/gpu_vanilla_vs_augmented"


def _build_image() -> modal.Image:
    """Self-contained CUDA image: patched TurboVec wheel + CuPy + lodedb + both bench dirs."""

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
            "datasets>=3.0.0",  # stream the GovReport corpus
            "h5py>=3.0.0",  # kept identical to the gpu_vanilla_vs_augmented image so the
            # expensive patched-TurboVec build layer is shared from Modal's cache
        )
        # Build + install the patched vendored TurboVec wheel from source (the python crate
        # path-depends on the sibling core crate, so copy the whole tree). build.rs links
        # system OpenBLAS on Linux.
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
        # Install the local lodedb package WITHOUT deps (already installed above).
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
        .env({"PYTHONPATH": f"{_GOVREPORT_DIR}:{_VVA_DIR}"})
    )
    # Final, non-copy mounts (must come after all build steps): both bench dirs, so the shared
    # cell runner in gpu_vanilla_vs_augmented is importable as a sibling of this benchmark.
    return image.add_local_dir(
        str(bench_root / "govreport_scale"), remote_path=_GOVREPORT_DIR, ignore=mount_ignore
    ).add_local_dir(
        str(bench_root / "gpu_vanilla_vs_augmented"), remote_path=_VVA_DIR, ignore=mount_ignore
    )


IMAGE = _build_image()
app = modal.App("turbovec-govreport-scale", image=IMAGE)


@app.function(gpu="L40S", cpu=16.0, memory=131072, timeout=7200)
def run_govreport(spec: dict) -> dict:
    """Embeds GovReport on the GPU, then runs the recall + speed size sweep."""

    from turbovec_govreport_scale import embed_govreport, run_govreport_scale

    spec["dataset"] = embed_govreport(
        max_corpus=int(spec["max_corpus"]),
        n_query=int(spec["queries"]),
        out_dir="/root/data",
        chunk_character_limit=int(spec.get("chunk_character_limit", 480)),
    )
    return run_govreport_scale(spec)


def _write(bundle: dict, out: str) -> None:
    """Writes the results bundle locally and prints a one-line summary."""

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    dataset = bundle.get("dataset", {})
    print(
        f"[govreport-scale] wrote {path} | corpus={dataset.get('corpus_count')} "
        f"embed={dataset.get('embed_seconds')}s wall={bundle.get('wall_seconds', 0):.0f}s"
    )


@app.local_entrypoint()
def main(out: str = "benchmarks/govreport_scale/results/results.json") -> None:
    """Full run: GovReport at 100K/500K/1M on an L40S."""

    from turbovec_govreport_scale import govreport_scale_spec

    _write(run_govreport.remote(govreport_scale_spec()), out)


@app.local_entrypoint()
def smoke(out: str = "benchmarks/govreport_scale/results/results_smoke.json") -> None:
    """Tiny pipeline-validation run (~40K chunks) before the full 1M matrix."""

    from turbovec_govreport_scale import govreport_scale_smoke_spec

    _write(run_govreport.remote(govreport_scale_smoke_spec()), out)
