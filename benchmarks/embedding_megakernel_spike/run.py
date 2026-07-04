#!/usr/bin/env python3
"""Local spike harness for issue #67 (fused single-query MiniLM embedding megakernel).

Answers the issue's spike gate on this machine, metrics-only:

1. Attribution - where the warm single-query embedding latency goes (tokenize vs forward
   vs pool vs normalize), per runtime/provider baseline.
2. Compiler comparison - how much the tuned ONNX compiler already fuses for free (ORT
   graph-optimization-level sweep: forward latency + optimized node/fused-op counts).
3. Ceiling - the memory-bandwidth roofline floor for a batch-1 forward pass, which bounds
   the best speedup any megakernel could deliver (derived, see spike_roofline).

    python benchmarks/embedding_megakernel_spike/run.py           # -> results/spike_<machine>.json
    python benchmarks/embedding_megakernel_spike/run.py --iters 100

Baselines are named so the issue's two M1 numbers land on the right stack: the 5.73 ms
"CPU embed" is onnx-cpu; the 8.42 ms "MPS embed" is torch-mps (an ONNX mps request routes
to the CPU provider here, so it is not an ONNX number). Unavailable baselines are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spike_backends as sb  # noqa: E402
import spike_measure as sm  # noqa: E402
import spike_roofline as sr  # noqa: E402

# Which baselines to try locally; unavailable ones (no MPS, no Core ML) are skipped.
LOCAL_BASELINES = ("onnx-cpu", "onnx-coreml", "torch-cpu", "torch-mps")
ONNX_BASELINES = ("onnx-cpu", "onnx-coreml", "onnx-cuda", "onnx-tensorrt")


def _torch_device_for(name: str) -> str:
    return name.split("-", 1)[1]


def run_spike(baselines: tuple[str, ...], *, iters: int, warmup: int) -> dict:
    """Runs attribution + compiler comparison + roofline; returns a metrics-only dict."""

    machine = sb.machine_info()
    queries = sb.FIXTURE_QUERIES

    built: dict[str, sb.NamedBackend] = {}
    for name in baselines:
        try:
            backend = sb.build_named_backend(name)
        except Exception as exc:  # noqa: BLE001 - a missing runtime should skip, not crash
            print(f"[spike] skip {name}: {exc}", file=sys.stderr)
            continue
        if backend is None:
            print(f"[spike] skip {name}: runtime/device unavailable", file=sys.stderr)
            continue
        built[name] = backend
        print(f"[spike] built {name} (providers={backend.providers or 'n/a'})", file=sys.stderr)

    reference = _reference_vectors(built, queries)

    baseline_results: dict[str, object] = {}
    attribution: dict[str, object] = {}
    compiler: dict[str, object] = {}

    for name, backend in built.items():
        print(f"[spike] measuring {name} ...", file=sys.stderr)
        e2e = sm._time_calls(
            backend.embed_query, queries, iters=iters, warmup=warmup,
            sync=_sync_for(backend),
        )
        baseline_results[name] = {
            "runtime": backend.runtime,
            "device": backend.device,
            "providers": list(backend.providers),
            "embed_query": sm.percentiles(e2e),
            "parity_cosine_vs_onnx_cpu": _parity(backend, queries, reference),
        }

    # Attribution: ONNX via the backend seams; torch via the parity-checked HF pipeline.
    for name, backend in built.items():
        print(f"[spike] attributing {name} ...", file=sys.stderr)
        try:
            if backend.runtime == "onnx":
                attribution[name] = sm.attribute_onnx(
                    backend.backend, queries, iters=iters, warmup=warmup
                )
            else:
                device = _torch_device_for(name)
                manual = sb.ManualTorchMiniLM(device)
                parity = min(
                    sb.cosine(manual.embed(q), reference[q]) for q in queries
                ) if reference else None
                attr = sm.attribute_torch(manual, queries, iters=iters, warmup=warmup)
                attr["manual_pipeline_min_parity_cosine"] = (
                    round(parity, 6) if parity is not None else None
                )
                attribution[name] = attr
        except Exception as exc:  # noqa: BLE001 - record, do not abort the whole spike
            attribution[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[spike] attribution failed for {name}: {exc}", file=sys.stderr)

    # Compiler comparison: ORT optimization sweep + profiler node aggregates per ONNX path.
    for name, backend in built.items():
        if backend.runtime != "onnx":
            continue
        print(f"[spike] compiler comparison {name} ...", file=sys.stderr)
        entry: dict[str, object] = {}
        try:
            entry["optimization_sweep"] = sm.ort_optimization_sweep(
                backend.backend, queries, iters=max(iters, 30), warmup=warmup
            )
        except Exception as exc:  # noqa: BLE001 - record, keep going
            entry["optimization_sweep"] = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            # Fixed, modest run count: the profiler is for node/op counts and relative
            # timing, and ORT's event buffer (~1M events) overflows well before iters=200
            # (nodes x queries x runs), which would truncate the trace.
            entry["profiler_nodes"] = sm.ort_profile_nodes(backend.backend, queries, runs=20)
        except Exception as exc:  # noqa: BLE001 - record, keep going
            entry["profiler_nodes"] = {"error": f"{type(exc).__name__}: {exc}"}
        compiler[name] = entry

    roofline = _roofline_section(built)
    verdict = _verdict(attribution, compiler, roofline)

    return {
        "spike": "embedding_megakernel_gate",
        "issue": 67,
        "model": sb.MINILM_MODEL,
        "machine": machine,
        "config": {"iters": iters, "warmup": warmup, "query_count": len(queries)},
        "baselines": baseline_results,
        "attribution": attribution,
        "compiler": compiler,
        "roofline": roofline,
        "verdict": verdict,
    }


def _sync_for(backend: sb.NamedBackend):
    if backend.runtime != "torch":
        return None
    try:
        import torch
    except Exception:  # noqa: BLE001
        return None
    if backend.device == "mps":
        return torch.mps.synchronize
    if backend.device == "cuda":
        return torch.cuda.synchronize
    return None


def _reference_vectors(built: dict[str, sb.NamedBackend], queries: tuple[str, ...]) -> dict:
    """Builds the onnx-cpu parity reference (the CPU correctness bar).

    Always an onnx-cpu embedding, even when onnx-cpu is not one of the measured baselines
    (the CUDA path builds only GPU baselines), so ``parity_cosine_vs_onnx_cpu`` is a genuine
    cross-runtime check against the CPU embedding rather than a GPU-vs-itself comparison.
    """

    ref_backend = built.get("onnx-cpu")
    if ref_backend is None:
        try:
            ref_backend = sb.build_named_backend("onnx-cpu")
        except Exception as exc:  # noqa: BLE001 - no CPU runtime: skip parity, do not crash
            print(f"[spike] parity reference unavailable: {exc}", file=sys.stderr)
            return {}
    if ref_backend is None:
        return {}
    return {q: ref_backend.embed_query(q) for q in queries}


def _parity(backend: sb.NamedBackend, queries: tuple[str, ...], reference: dict) -> dict:
    if not reference:
        return {}
    cosines = [sb.cosine(backend.embed_query(q), reference[q]) for q in queries]
    return {
        "min": round(min(cosines), 6),
        "mean": round(sum(cosines) / len(cosines), 6),
    }


def _roofline_section(built: dict[str, sb.NamedBackend]) -> dict:
    """Weight budget + per-device floors, from whichever ONNX artifact is materialized."""

    onnx_backend = next((b for b in built.values() if b.runtime == "onnx"), None)
    if onnx_backend is None:
        onnx_backend_obj = sb.onnx_backend(("CPUExecutionProvider",))
        onnx_path = onnx_backend_obj.onnx_model_path
    else:
        onnx_path = onnx_backend.backend.onnx_model_path
    try:
        budget = sr.analyze_weight_budget(onnx_path)
    except ImportError:
        return {"note": "roofline needs the onnx package (pip install onnx); skipped."}
    floor = sr.roofline_floor(budget, seq_len=16)
    return {"weight_budget": budget.to_dict(), "floor": floor.to_dict()}


def _verdict(attribution: dict, compiler: dict, roofline: dict) -> dict:
    """Derives the gate answer from the measured sections (see README for the rule)."""

    forward_share = {}
    for name, attr in attribution.items():
        e2e = attr.get("end_to_end", {}).get("p50_ms")
        fwd_key = "session_run" if "session_run" in attr else "forward"
        fwd = attr.get(fwd_key, {}).get("p50_ms")
        if e2e and fwd:
            forward_share[name] = round(fwd / e2e, 3)

    # ORT free-fusion effect: best (lowest) optimized latency vs the unoptimized level.
    fusion_speedup = {}
    for name, comp in compiler.items():
        sweep = comp.get("optimization_sweep", {})
        base = sweep.get("disable_all", {}).get("forward", {}).get("p50_ms")
        best = min(
            (lvl.get("forward", {}).get("p50_ms", float("inf")) for lvl in sweep.values()),
            default=None,
        )
        if base and best and best != float("inf"):
            fusion_speedup[name] = round(base / best, 3)

    return {
        "forward_pass_share_of_e2e": forward_share,
        "ort_free_fusion_speedup": fusion_speedup,
        "note": (
            "forward_pass_share_of_e2e answers gate step 1 (is the forward pass the "
            "majority). ort_free_fusion_speedup is how much the tuned ONNX compiler already "
            "gains from graph fusion with no custom kernel (gate step 2). The megakernel bar "
            "is a durable multiplicative win over the BEST compiler path AND the roofline floor "
            "(gate step 3). See README for the decision rule and the recommendation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding megakernel spike (issue #67)")
    parser.add_argument("--iters", type=int, default=50, help="timed passes per stage")
    parser.add_argument("--warmup", type=int, default=10, help="warmup passes per stage")
    parser.add_argument(
        "--baselines",
        default=",".join(LOCAL_BASELINES),
        help="comma-separated baseline names to try",
    )
    parser.add_argument("--out", default=None, help="output JSON path")
    args = parser.parse_args()

    baselines = tuple(b.strip() for b in args.baselines.split(",") if b.strip())
    result = run_spike(baselines, iters=args.iters, warmup=args.warmup)

    machine_tag = "m1" if result["machine"].get("apple_silicon") else "local"
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results" / f"spike_{machine_tag}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[spike] wrote {out}", file=sys.stderr)
    _print_summary(result)


def _print_summary(result: dict) -> None:
    print("\n=== embedding megakernel spike: summary ===", file=sys.stderr)
    for name, base in result["baselines"].items():
        eq = base["embed_query"]
        parity = base.get("parity_cosine_vs_onnx_cpu", {})
        print(
            f"  {name:14s} embed_query p50={eq.get('p50_ms')}ms p95={eq.get('p95_ms')}ms "
            f"parity_min={parity.get('min')}",
            file=sys.stderr,
        )
    print("  forward-pass share of e2e:", result["verdict"]["forward_pass_share_of_e2e"],
          file=sys.stderr)
    print("  ORT free-fusion speedup:", result["verdict"]["ort_free_fusion_speedup"],
          file=sys.stderr)
    floor = result["roofline"].get("floor", {}).get("floors_ms", {})
    if floor:
        print("  roofline floor (fp16, ms):",
              {d: v.get("floor_ms_fp16") for d, v in floor.items()}, file=sys.stderr)


if __name__ == "__main__":
    main()
