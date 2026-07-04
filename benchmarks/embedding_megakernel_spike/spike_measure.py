"""Timing, stage attribution, and compiler-comparison measurement for the spike.

Three things, all metrics-only (latency, counts, cosine; never text/embeddings):

- **Stage attribution** - split warm single-query latency into tokenize, forward
  (``session.run`` / torch ``model()``), pool, normalize, and tuple conversion, so gate
  step 1 (is the forward pass the majority?) is answered per runtime.
- **ORT profiler node analysis** - the ONNX Runtime profiler's per-operator times and
  graph-node count. Reported honestly: the summed node time is *profiled operator time*,
  not device kernel time, and ``model_run - summed_nodes`` is unprofiled span (scheduler,
  allocation, sync, profiling overhead), not clean launch overhead. Node count is graph
  nodes executed, not GPU kernel launches. Raw traces are parsed then deleted.
- **ORT optimization-level sweep** - the same graph at each ORT graph-optimization level,
  reporting forward latency and the optimized-graph node/fused-op counts. This quantifies
  gate step 2: how much fusion the tuned compiler already does with zero custom kernel.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np

from lodedb.engine.embedding_backends import _l2_normalize_rows, _pool_onnx_output


def percentiles(samples_ms: list[float]) -> dict[str, float]:
    """p50/p90/p95/p99 + IQR-based spread for a latency sample (milliseconds)."""

    if not samples_ms:
        return {}
    arr = np.asarray(samples_ms, dtype=np.float64)
    q25, q50, q75 = (float(np.percentile(arr, p)) for p in (25, 50, 75))
    return {
        "p50_ms": round(q50, 5),
        "p90_ms": round(float(np.percentile(arr, 90)), 5),
        "p95_ms": round(float(np.percentile(arr, 95)), 5),
        "p99_ms": round(float(np.percentile(arr, 99)), 5),
        "iqr_ms": round(q75 - q25, 5),
        "min_ms": round(float(arr.min()), 5),
        "samples": int(arr.size),
    }


def _time_calls(fn, args_iter, *, iters: int, warmup: int, sync=None) -> list[float]:
    """Times ``fn(arg)`` for each arg over ``iters`` passes after ``warmup`` passes."""

    args = list(args_iter)
    for _ in range(warmup):
        for arg in args:
            fn(arg)
    if sync is not None:
        sync()
    samples: list[float] = []
    for _ in range(iters):
        for arg in args:
            start = time.perf_counter()
            fn(arg)
            if sync is not None:
                sync()
            samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def attribute_onnx(backend, queries: tuple[str, ...], *, iters: int, warmup: int) -> dict:
    """Splits ONNX single-query latency into tokenize / session.run / pool / norm / tuple."""

    prefix = backend.query_prefix
    texts = [f"{prefix}{q}" for q in queries]
    tokenized = {t: backend._tokenize((t,)) for t in texts}
    ran = {t: backend._session_run(tokenized[t]) for t in texts}

    def _pool(t):
        return _pool_onnx_output(
            ran[t],
            attention_mask=np.asarray(tokenized[t].get("attention_mask")),
            pooling=backend.pooling,
            output_name=backend.output_name,
        )

    pooled = {t: _pool(t) for t in texts}

    e2e = _time_calls(backend.embed_query, queries, iters=iters, warmup=warmup)
    tok = _time_calls(lambda t: backend._tokenize((t,)), texts, iters=iters, warmup=warmup)
    run = _time_calls(lambda t: backend._session_run(tokenized[t]), texts,
                      iters=iters, warmup=warmup)
    pool = _time_calls(_pool, texts, iters=iters, warmup=warmup)
    norm = _time_calls(lambda t: _l2_normalize_rows(pooled[t]), texts, iters=iters, warmup=warmup)
    # Time only the tuple conversion on already-normalized rows, so this stage does not
    # re-charge normalization (already measured above).
    normalized = {t: _l2_normalize_rows(pooled[t]) for t in texts}
    tup = _time_calls(
        lambda t: tuple(tuple(float(v) for v in row) for row in normalized[t]),
        texts, iters=iters, warmup=warmup,
    )
    return {
        "end_to_end": percentiles(e2e),
        "tokenize": percentiles(tok),
        "session_run": percentiles(run),
        "pool": percentiles(pool),
        "normalize": percentiles(norm),
        "to_tuple": percentiles(tup),
    }


def attribute_torch(manual, queries: tuple[str, ...], *, iters: int, warmup: int) -> dict:
    """Splits torch single-query latency into tokenize / forward / pool+normalize."""

    tokenized = {q: manual.tokenize(q) for q in queries}
    outputs = {q: manual.forward(tokenized[q]) for q in queries}
    manual.sync()

    e2e = _time_calls(manual.embed, queries, iters=iters, warmup=warmup, sync=manual.sync)
    tok = _time_calls(manual.tokenize, queries, iters=iters, warmup=warmup, sync=manual.sync)
    fwd = _time_calls(lambda q: manual.forward(tokenized[q]), queries,
                      iters=iters, warmup=warmup, sync=manual.sync)
    pool = _time_calls(
        lambda q: manual.pool_normalize(outputs[q], tokenized[q]["attention_mask"]),
        queries, iters=iters, warmup=warmup, sync=manual.sync,
    )
    return {
        "end_to_end": percentiles(e2e),
        "tokenize": percentiles(tok),
        "forward": percentiles(fwd),
        "pool_normalize": percentiles(pool),
    }


def _resolve_trace_file(trace_name: str | None, tmp: str) -> Path | None:
    """Locates the ORT profiler JSON (name may be relative to CWD or the tmp prefix)."""

    candidates: list[Path] = []
    if trace_name:
        name = Path(trace_name)
        candidates.extend([name, Path(tmp) / name.name, Path.cwd() / name.name])
    candidates.extend(sorted(Path(tmp).glob("*.json")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def ort_profile_nodes(backend, queries: tuple[str, ...], *, runs: int = 30) -> dict:
    """ORT profiler node/operator aggregates for the ONNX forward pass (metrics-only).

    Builds a profiled session, runs the forward pass ``runs`` times, then parses and
    deletes the trace. Returns per-run graph-node count and summed profiled operator
    time, honestly labeled (see module docstring).
    """

    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmp:
        options = ort.SessionOptions()
        options.enable_profiling = True
        # Prefix with a path inside tmp so the trace is contained, not dropped in the CWD.
        options.profile_file_prefix = str(Path(tmp) / "spike")
        # Profile with the same providers as the measured backend.
        session = ort.InferenceSession(
            str(backend.onnx_model_path),
            sess_options=options,
            providers=list(backend.providers),
        )
        input_names = {i.name for i in session.get_inputs()}
        output_names = [o.name for o in session.get_outputs()]
        # Reuse the backend tokenizer so shapes match the real path.
        feeds = []
        for q in queries:
            tok = backend._tokenize((f"{backend.query_prefix}{q}",))
            feeds.append({k: v for k, v in tok.items() if k in input_names})
        for _ in range(3):  # warmup; a handful of extra forwards vs runs*queries, averaged in
            session.run(output_names, feeds[0])
        for _ in range(runs):
            for feed in feeds:
                session.run(output_names, feed)
        trace_name = session.end_profiling()
        trace_file = _resolve_trace_file(trace_name, tmp)
        events = json.loads(trace_file.read_text(encoding="utf-8")) if trace_file else []

    events = events if isinstance(events, list) else []
    node_kernel_us = 0.0
    node_kernel_events = 0
    model_run_us = 0.0
    model_run_events = 0
    node_names: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        cat = ev.get("cat")
        name = str(ev.get("name", ""))
        dur = float(ev.get("dur", 0) or 0)
        if cat == "Node" and name.endswith("_kernel_time"):
            node_kernel_us += dur
            node_kernel_events += 1
            node_names.add(name[: -len("_kernel_time")])
        elif cat == "Session" and name == "model_run":
            model_run_us += dur
            model_run_events += 1

    total_forward_events = max(model_run_events, 1)
    nodes_per_run = node_kernel_events / total_forward_events if total_forward_events else 0.0
    summed_node_us_per_run = node_kernel_us / total_forward_events if total_forward_events else 0.0
    model_run_us_per_run = model_run_us / total_forward_events if total_forward_events else 0.0
    return {
        "note": (
            "ORT CPU-profiler aggregates. summed_profiled_operator_us is profiled operator "
            "time (not device kernel time); model_run_minus_nodes_us is unprofiled span "
            "(scheduler/alloc/sync/profiling overhead), NOT clean launch overhead; "
            "graph_nodes_per_run counts graph nodes executed, not GPU kernel launches."
        ),
        "distinct_graph_nodes": len(node_names),
        "graph_nodes_per_run": round(nodes_per_run, 1),
        "summed_profiled_operator_us": round(summed_node_us_per_run, 2),
        "model_run_us": round(model_run_us_per_run, 2),
        "model_run_minus_nodes_us": round(model_run_us_per_run - summed_node_us_per_run, 2),
        "forward_runs_profiled": total_forward_events,
    }


# ONNX Runtime contrib fusions worth surfacing: their presence in the optimized graph is
# the "free" fusion the tuned compiler already applies (gate step 2).
_FUSED_OP_TYPES = (
    "Attention",
    "MultiHeadAttention",
    "SkipLayerNormalization",
    "EmbedLayerNormalization",
    "LayerNormalization",
    "FusedMatMul",
    "FusedGemm",
    "BiasGelu",
    "FastGelu",
    "Gelu",
    "QuickGelu",
)


def ort_optimization_sweep(backend, queries: tuple[str, ...], *, iters: int, warmup: int) -> dict:
    """Times the forward pass at each ORT graph-optimization level + counts fused ops.

    The node-count drop and the appearance of fused contrib ops from DISABLE_ALL to
    ENABLE_ALL is how much the tuned ONNX compiler fuses with no custom kernel.
    """

    import onnxruntime as ort

    levels = {
        "disable_all": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "enable_basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "enable_extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "enable_all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    # One fixed, representative tokenized input (median-length fixture).
    median_q = sorted(queries, key=len)[len(queries) // 2]
    base_tok = backend._tokenize((f"{backend.query_prefix}{median_q}",))
    # ORT can only serialize the optimized graph for CPU sessions; accelerator EPs
    # (CoreML/CUDA/TensorRT) compile nodes and refuse the dump, so node/fusion counts
    # are collected for CPU-only providers and omitted (latency only) otherwise.
    dump_graph = tuple(backend.providers) == ("CPUExecutionProvider",)

    results: dict[str, object] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for label, level in levels.items():
            options = ort.SessionOptions()
            options.graph_optimization_level = level
            opt_path = Path(tmp) / f"opt_{label}.onnx"
            if dump_graph:
                options.optimized_model_filepath = str(opt_path)
            session = ort.InferenceSession(
                str(backend.onnx_model_path),
                sess_options=options,
                providers=list(backend.providers),
            )
            input_names = {i.name for i in session.get_inputs()}
            output_names = [o.name for o in session.get_outputs()]
            feed = {k: v for k, v in base_tok.items() if k in input_names}
            samples = _time_calls(
                lambda _f, s=session, o=output_names, fd=feed: s.run(o, fd),
                [None], iters=iters, warmup=warmup,
            )
            node_count, fused = _optimized_graph_stats(opt_path) if dump_graph else (None, {})
            results[label] = {
                "forward": percentiles(samples),
                "optimized_node_count": node_count,
                "fused_op_counts": fused,
            }
    return results


def _optimized_graph_stats(opt_path: Path) -> tuple[int | None, dict[str, int]]:
    """Counts nodes + fused-op occurrences in an ORT-dumped optimized graph."""

    if not opt_path.is_file():
        return None, {}
    try:
        import onnx

        model = onnx.load(str(opt_path), load_external_data=False)
    except Exception:  # noqa: BLE001 - optional analysis dependency
        return None, {}
    counts: dict[str, int] = {}
    for node in model.graph.node:
        if node.op_type in _FUSED_OP_TYPES:
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return len(model.graph.node), counts
