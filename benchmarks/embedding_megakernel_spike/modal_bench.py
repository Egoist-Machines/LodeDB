"""Run the embedding megakernel spike (issue #67) on a CUDA GPU via Modal.

The laptop numbers in the issue are the motivation; the megakernel would target a server
GPU, so this runs the same attribution / compiler-comparison / roofline spike on CUDA and
adds the ORT CUDA Graph and torch.compile bars (see spike_cuda). Launch from the repo root:

    modal run benchmarks/embedding_megakernel_spike/modal_bench.py::smoke_a10
    modal run benchmarks/embedding_megakernel_spike/modal_bench.py::a10
    modal run benchmarks/embedding_megakernel_spike/modal_bench.py::l40s

Only the pure-Python embedding path is exercised, so the image does not build the native
lodedb._turbovec extension: src/lodedb is added to PYTHONPATH and the base runtime deps are
installed. Results are metrics-only (latency, counts, cosine).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

_REMOTE_SPIKE_DIR = "/root/embedding_megakernel_spike"
_REMOTE_SRC = "/root/lodedb-src/src"

# LodeDB base runtime deps (the engine.core import chain the embedding presets reach) plus
# the embedding + analysis stacks. torch/CUDA come from the base image; onnxruntime-gpu is
# pinned to a CUDA-12 build to match the base's CUDA 12.4 (the latest wheels want CUDA 13,
# whose libcudart the image lacks, so the CUDA EP silently fails to load). The nvidia-*-cu12
# wheels supply the exact CUDA-12 / cuDNN-9 split libs onnxruntime-gpu 1.20 dlopens
# (libcudnn_adv.so.9 etc.), independent of torch's bundle discovery.
_PIP_DEPENDENCIES = (
    "numpy>=2.0.0",
    "pyyaml>=6.0.0",
    "typer>=0.12.0",
    "onnxruntime-gpu==1.20.1",
    "transformers>=4.40,<5",
    "sentence-transformers>=3.0.0,<5",
    "onnx>=1.16",
    "huggingface_hub>=0.24",
    "nvidia-cudnn-cu12>=9,<10",
    "nvidia-cublas-cu12",
    "nvidia-cuda-runtime-cu12",
)

# Register every nvidia-*-cu12 lib dir (plus torch's) with ldconfig so onnxruntime-gpu's CUDA
# EP resolves cuDNN/cuBLAS/cudart at load time. The final grep fails the *build* (not a silent
# runtime CPU fallback) if libcudnn_adv is still missing.
_REGISTER_CUDA_LIBS = (
    'SP="$(python -c \'import site;print(site.getsitepackages()[0])\')" && '
    'find "$SP/nvidia" -maxdepth 2 -name lib -type d > /etc/ld.so.conf.d/spike-cuda.conf && '
    'python -c \'import os,torch;'
    'print(os.path.join(os.path.dirname(torch.__file__),"lib"))\' '
    '>> /etc/ld.so.conf.d/spike-cuda.conf && '
    "ldconfig && ldconfig -p | grep -q libcudnn_adv.so.9"
)


def _build_image() -> modal.Image:
    """Builds a CUDA image with the embedding runtimes + the spike modules (no maturin)."""

    repo_root = Path(__file__).resolve().parents[2]
    image = (
        modal.Image.from_registry(
            "pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime",
            add_python="3.11",
        )
        .apt_install("build-essential")
        .pip_install(*_PIP_DEPENDENCIES)
        # Make onnxruntime-gpu's CUDA EP resolvable (see _REGISTER_CUDA_LIBS). Without this the
        # EP loads but fails at session creation and onnx-cuda silently falls back to the CPU.
        .run_commands(_REGISTER_CUDA_LIBS)
        # TensorRT EP is best-effort: if the wheel/libs are unavailable the onnx-tensorrt
        # baseline is simply skipped at runtime (build_named_backend returns None).
        .run_commands("python -m pip install 'tensorrt-cu12>=10,<11' || true")
        .add_local_dir(
            str(repo_root / "src"),
            remote_path=_REMOTE_SRC,
            copy=True,
            ignore=["**/*.so", "**/*.pyd", "**/*.dylib", "**/__pycache__/**", "**/*.pyc"],
        )
        .env(
            {
                "PYTHONPATH": f"{_REMOTE_SPIKE_DIR}:{_REMOTE_SRC}",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
    )
    return image.add_local_dir(
        str(Path(__file__).resolve().parent),
        remote_path=_REMOTE_SPIKE_DIR,
        ignore=["**/__pycache__/**", "**/*.pyc", "results/**"],
    )


IMAGE = _build_image()
app = modal.App("lodedb-embedding-megakernel-spike", image=IMAGE)


@app.function(gpu="A10", cpu=8.0, memory=32768, timeout=3600)
def run_spike_a10(spec: dict) -> dict:
    """Runs the CUDA spike on an A10."""

    from spike_cuda import run_cuda_spike

    return run_cuda_spike(**spec)


@app.function(gpu="L40S", cpu=8.0, memory=32768, timeout=3600)
def run_spike_l40s(spec: dict) -> dict:
    """Runs the CUDA spike on an L40S."""

    from spike_cuda import run_cuda_spike

    return run_cuda_spike(**spec)


def _write(bundle: dict, out: str) -> None:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    verdict = bundle.get("verdict", {})
    print(f"[megakernel-spike] wrote {path}")
    print(f"[megakernel-spike] forward share: {verdict.get('forward_pass_share_of_e2e')}")


@app.local_entrypoint()
def a10(out: str = "benchmarks/embedding_megakernel_spike/results/spike_a10.json") -> None:
    """Full A10 CUDA spike."""

    _write(run_spike_a10.remote({"iters": 200, "warmup": 30, "seq_len": 16}), out)


@app.local_entrypoint()
def l40s(out: str = "benchmarks/embedding_megakernel_spike/results/spike_l40s.json") -> None:
    """Full L40S CUDA spike."""

    _write(run_spike_l40s.remote({"iters": 200, "warmup": 30, "seq_len": 16}), out)


_SMOKE_A10_OUT = "benchmarks/embedding_megakernel_spike/results/spike_smoke_a10.json"
_SMOKE_L40S_OUT = "benchmarks/embedding_megakernel_spike/results/spike_smoke_l40s.json"


@app.local_entrypoint()
def smoke_a10(out: str = _SMOKE_A10_OUT) -> None:
    """Small A10 validation run before the full spike."""

    _write(run_spike_a10.remote({"iters": 20, "warmup": 5, "seq_len": 16}), out)


@app.local_entrypoint()
def smoke_l40s(out: str = _SMOKE_L40S_OUT) -> None:
    """Small L40S validation run before the full spike."""

    _write(run_spike_l40s.remote({"iters": 20, "warmup": 5, "seq_len": 16}), out)
