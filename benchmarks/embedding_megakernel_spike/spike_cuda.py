"""CUDA spike logic (issue #67), imported by the Modal worker on the remote GPU.

Kept separate from ``modal_bench.py`` so importing it remotely does not re-run the image
builder. It runs the same attribution / compiler-comparison / roofline spike as ``run.py``
on the CUDA baselines, then adds the two lowest-maintenance fusion paths a hand-written
megakernel would have to beat by a multiplicative margin to be worth owning:

- **ORT CUDA Graph replay** - a fixed-shape session with ``enable_cuda_graph``, replayed
  through I/O binding. This removes per-op kernel-launch overhead with zero custom kernel.
- **torch.compile(reduce-overhead)** - Inductor fusion + CUDA graphs, again no hand kernel.

KernelBench-Mega flags both CUDA graphs and torch.compile as *not* genuine single-kernel
fusion, which is exactly why they are the bar here: whatever latency they reach is what a
hand megakernel must clear to justify its maintenance cost.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import spike_backends as sb  # noqa: E402
import spike_measure as sm  # noqa: E402
from run import run_spike  # noqa: E402

CUDA_BASELINES = ("onnx-cuda", "torch-cuda", "onnx-tensorrt")


def run_cuda_spike(*, iters: int, warmup: int, seq_len: int = 16) -> dict:
    """Runs the CUDA spike and augments it with the CUDA-graph / torch.compile bars."""

    result = run_spike(CUDA_BASELINES, iters=iters, warmup=warmup)
    result["spike"] = "embedding_megakernel_gate_cuda"
    result["config"]["fixed_seq_len"] = seq_len

    low_maint: dict[str, object] = {}
    low_maint["ort_cuda_graph"] = _safe(
        measure_ort_cuda_graph, iters=iters, warmup=warmup, seq_len=seq_len
    )
    low_maint["torch_compile"] = _safe(
        measure_torch_compile, iters=iters, warmup=warmup, seq_len=seq_len
    )
    result["compiler"]["low_maintenance_fusion"] = low_maint
    result["verdict"]["low_maintenance_fusion_note"] = (
        "ort_cuda_graph and torch_compile are the best no-custom-kernel fusion paths; the "
        "megakernel bar is a durable multiplicative win over the fastest of these AND the "
        "roofline floor. Both use CUDA graphs, which KernelBench-Mega does not count as a "
        "genuine single kernel, so they are the bar, not the megakernel itself."
    )
    return result


def _safe(fn, **kwargs) -> dict:
    try:
        return fn(**kwargs)
    except Exception as exc:  # noqa: BLE001 - record, keep the spike alive
        return {"error": f"{type(exc).__name__}: {exc}"}


def _fixed_tokens_np(backend, seq_len: int) -> dict:
    """Tokenizes one fixture query to a fixed [1, seq_len] shape (for CUDA graph capture)."""

    tokenizer = backend._load_tokenizer()
    query = f"{backend.query_prefix}{sb.FIXTURE_QUERIES[3]}"
    enc = tokenizer(
        [query], padding="max_length", truncation=True, max_length=seq_len, return_tensors="np"
    )
    return {k: np.asarray(v) for k, v in dict(enc).items()}


def measure_ort_cuda_graph(*, iters: int, warmup: int, seq_len: int) -> dict:
    """Times ORT CUDA Graph replay of the forward pass at a fixed sequence length."""

    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return {"skipped": "no CUDAExecutionProvider"}

    backend = sb.onnx_backend(("CUDAExecutionProvider", "CPUExecutionProvider"))
    hidden = backend.native_dim
    feeds = _fixed_tokens_np(backend, seq_len)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(backend.onnx_model_path),
        sess_options=options,
        providers=[
            ("CUDAExecutionProvider", {"enable_cuda_graph": True}),
            "CPUExecutionProvider",
        ],
    )
    input_names = {i.name for i in session.get_inputs()}
    output_name = session.get_outputs()[0].name

    binding = session.io_binding()
    input_ovs = {}
    for name in input_names:
        if name not in feeds:
            continue
        ov = ort.OrtValue.ortvalue_from_numpy(feeds[name], "cuda", 0)
        input_ovs[name] = ov  # keep a ref so the fixed address stays valid for replay
        binding.bind_ortvalue_input(name, ov)
    out_ov = ort.OrtValue.ortvalue_from_shape_and_type([1, seq_len, hidden], np.float32, "cuda", 0)
    binding.bind_ortvalue_output(output_name, out_ov)

    import torch

    for _ in range(warmup + 2):  # first replay captures the graph
        session.run_with_iobinding(binding)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        session.run_with_iobinding(binding)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)

    stats = sm.percentiles(samples)
    stats["note"] = (
        "ORT CUDA Graph replay at fixed seq_len; per-op launch overhead removed with no "
        "custom kernel. Not a genuine single kernel (KernelBench-Mega)."
    )
    stats["seq_len"] = seq_len
    return stats


def measure_torch_compile(*, iters: int, warmup: int, seq_len: int) -> dict:
    """Times eager vs torch.compile(reduce-overhead) forward at a fixed sequence length."""

    import torch

    if not torch.cuda.is_available():
        return {"skipped": "no CUDA torch device"}

    manual = sb.ManualTorchMiniLM("cuda", max_seq_length=seq_len)
    tokens = manual.tokenizer(
        [f"{sb.FIXTURE_QUERIES[3]}"],
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    ).to("cuda")

    def _run(model):
        with torch.no_grad():
            return model(**tokens)

    # Eager forward at the same fixed shape (fair baseline for the compile speedup).
    for _ in range(warmup + 3):
        _run(manual.model)
    torch.cuda.synchronize()
    eager = []
    for _ in range(iters):
        start = time.perf_counter()
        _run(manual.model)
        torch.cuda.synchronize()
        eager.append((time.perf_counter() - start) * 1000.0)

    compiled_model = torch.compile(manual.model, mode="reduce-overhead")
    for _ in range(warmup + 5):  # compile + cudagraph capture
        _run(compiled_model)
    torch.cuda.synchronize()
    compiled = []
    for _ in range(iters):
        start = time.perf_counter()
        _run(compiled_model)
        torch.cuda.synchronize()
        compiled.append((time.perf_counter() - start) * 1000.0)

    eager_stats = sm.percentiles(eager)
    compiled_stats = sm.percentiles(compiled)
    speedup = None
    if eager_stats.get("p50_ms") and compiled_stats.get("p50_ms"):
        speedup = round(eager_stats["p50_ms"] / compiled_stats["p50_ms"], 3)
    return {
        "eager_forward": eager_stats,
        "compiled_forward": compiled_stats,
        "compile_speedup": speedup,
        "seq_len": seq_len,
        "note": (
            "torch.compile(reduce-overhead) = Inductor fusion + CUDA graphs; no custom "
            "kernel. Not a genuine single kernel (KernelBench-Mega)."
        ),
    }
