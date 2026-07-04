"""Time the ANN cluster-index build on a many-core host via Modal (issue #71).

The build is single-threaded, super-linear k-means in the Rust core and is not
reachable from Python, so it is timed by running the crate's own ignored timing
test (``ann_kmeans_build_timing``) on a CPU host. The test builds a deterministic
synthetic corpus and times the new parallel + bounded-sample build; with the
baseline flag it also times a single-threaded, full-corpus reproduction of the
pre-#71 build for a same-box A/B.

Launch from the repo root:

    modal run benchmarks/ann_kmeans_build/modal_bench.py::bench --n 200000 --baseline
    modal run benchmarks/ann_kmeans_build/modal_bench.py::bench --n 2000000

``--baseline`` also runs the (slow, single-threaded) old build; only pair it with
the smaller sizes. ``--n`` is the corpus size; ``--dim`` defaults to 384 (MiniLM).
"""

from __future__ import annotations

from pathlib import Path

import modal

_REMOTE_SRC = "/root/lodedb-src"


def _build_image() -> modal.Image:
    """Rust toolchain + the whole LodeDB workspace (cargo parses every member)."""

    # `libopenblas-dev`: TurboVec's build.rs emits `-lopenblas` on Linux (ndarray's
    # BLAS feature), so building lodedb-core needs OpenBLAS at link time; the -dev
    # package also pulls in the runtime `.so`.
    image = (
        modal.Image.from_registry("ubuntu:22.04", add_python="3.11")
        .apt_install("build-essential", "curl", "pkg-config", "libopenblas-dev")
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal"
        )
    )
    # The local paths exist only during `modal run`; Modal re-imports this module
    # inside the container, where they are absent. Guard the local-only copy steps
    # with modal.is_local() so the container re-import is safe (the files are
    # already baked into the image built locally).
    if modal.is_local():
        repo_root = Path(__file__).resolve().parents[2]
        for rel in ("Cargo.toml", "Cargo.lock"):
            image = image.add_local_file(
                str(repo_root / rel), remote_path=f"{_REMOTE_SRC}/{rel}", copy=True
            )
        for rel in ("crates", "third_party/turbovec"):
            image = image.add_local_dir(
                str(repo_root / rel),
                remote_path=f"{_REMOTE_SRC}/{rel}",
                copy=True,
                ignore=["**/target/**", "**/__pycache__/**"],
            )
    return image


IMAGE = _build_image()
app = modal.App("lodedb-ann-kmeans-build", image=IMAGE)


def _run(n: int, dim: int, blobs: int, baseline: bool) -> str:
    import os
    import subprocess

    env = dict(os.environ)
    env["PATH"] = "/root/.cargo/bin:" + env.get("PATH", "")
    env["LODEDB_ANN_BENCH_N"] = str(n)
    env["LODEDB_ANN_BENCH_DIM"] = str(dim)
    env["LODEDB_ANN_BENCH_BLOBS"] = str(blobs)
    if baseline:
        env["LODEDB_ANN_BENCH_BASELINE"] = "1"

    cmd = [
        "cargo",
        "test",
        "--release",
        "-p",
        "lodedb-core",
        "--test",
        "ann_build_bench",
        "--",
        "--ignored",
        "--nocapture",
        "ann_kmeans_build_timing",
    ]
    result = subprocess.run(
        cmd, cwd=_REMOTE_SRC, env=env, capture_output=True, text=True, check=False
    )
    output = result.stdout + "\n" + result.stderr
    print(output)
    if result.returncode != 0:
        raise RuntimeError(f"cargo test failed (exit {result.returncode})")
    return output


# 16 vCPUs matches the "16-core hosts" the issue cites for the benchmark runs.
@app.function(cpu=16.0, memory=65536, timeout=3600)
def run(n: int, dim: int, blobs: int, baseline: bool) -> str:
    return _run(n, dim, blobs, baseline)


@app.local_entrypoint()
def bench(
    n: int = 200_000,
    dim: int = 384,
    blobs: int = 256,
    baseline: bool = False,
) -> None:
    run.remote(n=n, dim=dim, blobs=blobs, baseline=baseline)
