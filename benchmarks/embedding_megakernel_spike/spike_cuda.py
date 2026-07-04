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

CUDA_BASELINES = ("onnx-cuda", "torch-cuda", "onnx-tensorrt", "torch-compile-cuda")


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
    """Eager vs torch.compile(reduce-overhead) over VARYING queries at a fixed pad length.

    Uses different fixture queries each call (production-realistic: cudagraphs must handle
    changing input values, not one repeated tensor) padded to a static ``seq_len`` shape.
    Reports the forward-only speedup and the full-embed speedup (tokenize + forward + pool +
    normalize + device->host), so the end-to-end number is not overstated by the forward.
    """

    import torch

    if not torch.cuda.is_available():
        return {"skipped": "no CUDA torch device"}

    manual = sb.ManualTorchMiniLM("cuda", max_seq_length=seq_len)
    queries = list(sb.FIXTURE_QUERIES)
    # Fixed-pad each query to [1, seq_len]: static shape (cudagraph-safe), varying content.
    fixed_tokens = {
        q: manual.tokenizer(
            [q], padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt"
        ).to("cuda")
        for q in queries
    }
    compiled_model = torch.compile(manual.model, mode="reduce-overhead")

    def forward(model, q):
        with torch.no_grad():
            return model(**fixed_tokens[q])

    def embed(model, q):
        tokens = manual.tokenizer(
            [q], padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt"
        ).to("cuda")
        with torch.no_grad():
            outputs = model(**tokens)
        vec = manual.pool_normalize(outputs, tokens["attention_mask"])
        return vec[0].detach().to("cpu").numpy()

    def _cycle_time(fn, model) -> list[float]:
        for i in range(warmup + 5):  # compile + cudagraph capture on the compiled model
            fn(model, queries[i % len(queries)])
        torch.cuda.synchronize()
        samples = []
        for i in range(iters):
            q = queries[i % len(queries)]
            start = time.perf_counter()
            fn(model, q)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000.0)
        return samples

    fwd_eager = sm.percentiles(_cycle_time(forward, manual.model))
    fwd_compiled = sm.percentiles(_cycle_time(forward, compiled_model))
    embed_eager = sm.percentiles(_cycle_time(embed, manual.model))
    embed_compiled = sm.percentiles(_cycle_time(embed, compiled_model))

    def _speedup(a, b):
        if a.get("p50_ms") and b.get("p50_ms"):
            return round(a["p50_ms"] / b["p50_ms"], 3)
        return None

    return {
        "eager_forward": fwd_eager,
        "compiled_forward": fwd_compiled,
        "compile_speedup": _speedup(fwd_eager, fwd_compiled),
        "eager_embed": embed_eager,
        "compiled_embed": embed_compiled,
        "embed_speedup": _speedup(embed_eager, embed_compiled),
        "seq_len": seq_len,
        "note": (
            "Varying queries, fixed pad length. compile_speedup is forward-only; "
            "embed_speedup includes tokenize + pool + normalize + device->host copy and is "
            "the realistic end-to-end figure. torch.compile(reduce-overhead) = Inductor fusion "
            "+ CUDA graphs, no custom kernel; not a genuine single kernel (KernelBench-Mega)."
        ),
    }
