"""Validate the native ``lodedb-gpu`` CUDA scan on real hardware via Modal.

The GPU-resident scan lives in the Rust core (cudarc) and cannot be exercised
from Python, so its correctness and performance are validated by running the
crate's own ``cargo test`` on a CUDA host. This is the vehicle for the opt-in
fused two-stage top-k (``LODEDB_GPU_FUSED_TOPK``): the parity test
``gpu_fused_topk_matches_two_pass`` proves it returns exactly what the default
cuBLAS + ``topk_argmax`` path returns, and ``gpu_fused_topk_timing`` measures the
speedup across batch sizes.

Launch from the repo root:

    modal run benchmarks/gpu_fused_topk/modal_rust_gpu_test.py::tests_l40s
    modal run benchmarks/gpu_fused_topk/modal_rust_gpu_test.py::tests_a10
    modal run benchmarks/gpu_fused_topk/modal_rust_gpu_test.py::bench_l40s
    modal run benchmarks/gpu_fused_topk/modal_rust_gpu_test.py::bench_a10

``tests_*`` gate correctness (all crate GPU tests, including patch + scan + the
fused parity test). ``bench_*`` additionally run the timing comparison; pass a
corpus size with ``--corpus`` (default 1,000,000, the size where the fused path's
removal of the k-pass score re-read is expected to matter).
"""

from __future__ import annotations

from pathlib import Path

import modal

# A CUDA -devel image ships libnvrtc + libcublas, which cudarc's dynamic-loading
# dlopens at runtime; the driver (libcuda) is provided by the Modal GPU host.
_CUDA_IMAGE = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
_REMOTE_SRC = "/root/lodedb-src"


def _build_image() -> modal.Image:
    """Builds a CUDA image with the Rust toolchain and the LodeDB workspace."""

    image = (
        modal.Image.from_registry(_CUDA_IMAGE, add_python="3.11")
        .apt_install("build-essential", "curl", "pkg-config")
        .run_commands(
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | "
            "sh -s -- -y --default-toolchain stable --profile minimal"
        )
    )
    # The workspace copy reads local paths that exist only during the local
    # `modal run`. Modal re-imports this module inside the container to locate the
    # function, where those paths are absent and __file__ resolves to /root (too few
    # parents for parents[2]); guard the local-only steps with modal.is_local() so
    # the container re-import is safe. The function still runs in the image built
    # locally, which has the files baked in.
    #
    # `cargo test -p lodedb-gpu` compiles only lodedb-gpu (deps: cudarc), but cargo
    # must parse every workspace member and its path deps, so ship the whole
    # workspace: manifests, lockfile, all crates, and the vendored turbovec that
    # lodedb-core path-depends on.
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
app = modal.App("lodedb-gpu-fused-topk", image=IMAGE)


def _run(bench: bool, corpus: int) -> str:
    """Runs the crate's GPU tests, optionally including the timing comparison."""

    import os
    import subprocess

    env = dict(os.environ)
    env["LODEDB_GPU_DEBUG"] = "1"
    # rustup put cargo under /root/.cargo/bin; the CUDA -devel image keeps
    # libnvrtc/libcublas under /usr/local/cuda/lib64 (dlopened by cudarc). Make both
    # discoverable to the cargo test subprocess.
    env["PATH"] = "/root/.cargo/bin:/usr/local/cuda/bin:" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + env.get("LD_LIBRARY_PATH", "")
    # `cargo test` in the crate: the GPU tests self-gate on a CUDA driver, so on a
    # GPU host they run for real. --nocapture surfaces the timing/debug eprintln!,
    # and --test-threads=1 serializes the device work so the timing is uncontended
    # and concurrent CUDA contexts cannot flake.
    cmd = [
        "cargo",
        "test",
        "--release",
        "-p",
        "lodedb-gpu",
        "--",
        "--nocapture",
        "--test-threads=1",
    ]
    if bench:
        env["LODEDB_GPU_BENCH"] = "1"
        env["LODEDB_GPU_BENCH_N"] = str(corpus)
    else:
        # Correctness only: skip the (slow, large-corpus) timing test by name.
        cmd += ["--skip", "gpu_fused_topk_timing"]

    result = subprocess.run(
        cmd, cwd=_REMOTE_SRC, env=env, capture_output=True, text=True, check=False
    )
    output = result.stdout + "\n" + result.stderr
    print(output)
    if result.returncode != 0:
        raise RuntimeError(f"cargo test failed (exit {result.returncode})")
    return output


@app.function(gpu="L40S", cpu=8.0, memory=32768, timeout=3600)
def run_l40s(bench: bool, corpus: int) -> str:
    return _run(bench, corpus)


@app.function(gpu="A10", cpu=8.0, memory=32768, timeout=3600)
def run_a10(bench: bool, corpus: int) -> str:
    return _run(bench, corpus)


@app.local_entrypoint()
def tests_l40s() -> None:
    """Correctness: run every lodedb-gpu GPU test on an L40S."""

    run_l40s.remote(bench=False, corpus=0)


@app.local_entrypoint()
def tests_a10() -> None:
    """Correctness: run every lodedb-gpu GPU test on an A10."""

    run_a10.remote(bench=False, corpus=0)


@app.local_entrypoint()
def bench_l40s(corpus: int = 1_000_000) -> None:
    """Correctness + timing (default vs fused top-k) on an L40S."""

    run_l40s.remote(bench=True, corpus=corpus)


@app.local_entrypoint()
def bench_a10(corpus: int = 1_000_000) -> None:
    """Correctness + timing (default vs fused top-k) on an A10."""

    run_a10.remote(bench=True, corpus=corpus)
