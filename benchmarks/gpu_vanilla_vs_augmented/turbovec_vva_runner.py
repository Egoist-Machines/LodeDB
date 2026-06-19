"""Orchestrate the vanilla-vs-augmented TurboVec benchmark across all four axes.

This module and its measurement core (``turbovec_vva_bench``) are dev-only
benchmark scripts that live beside this file — they are NOT part of the shipped
``lodedb`` package. Vanilla single- vs multi-threaded speed is measured in fresh
subprocesses with ``RAYON_NUM_THREADS`` pinned (rayon reads it once per process,
so the cell module is invoked via ``python -m`` per thread setting); the GPU path
and the recall/memory/update axes run in-process.

Run directly on any host (``lodedb`` must be importable — e.g. installed, or with
this benchmark dir on ``sys.path`` so ``import turbovec_vva_bench`` resolves; the
file inserts its own directory automatically)::

    # Local CPU-only smoke run (no GPU; the augmented series records "skipped"):
    python benchmarks/gpu_vanilla_vs_augmented/turbovec_vva_runner.py --smoke

    # Full matrix on a CUDA host (CuPy present): from this directory,
    python turbovec_vva_runner.py --out results/results_a10.json

The full GPU matrix is most easily run on Modal via
``benchmarks/gpu_vanilla_vs_augmented/modal_bench.py`` (which calls
:func:`run_all`).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# These benchmark scripts run as plain sibling modules (benchmarks/ is dev-only,
# not part of the shipped package), so make this directory importable for both the
# in-process import below and the ``python -m turbovec_vva_bench`` subprocess cells.
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from turbovec_vva_bench import (  # noqa: E402, F401
    K_GRID,
    machine_info,
    run_cell,
)

_CELL_MODULE = "turbovec_vva_bench"


def _subprocess_cell(cfg: dict[str, Any], *, threads: int) -> dict[str, Any]:
    """Runs one bench cell in a fresh process with RAYON_NUM_THREADS pinned.

    Inherits the caller's environment and prepends this benchmark directory to
    ``PYTHONPATH`` so the child's ``python -m turbovec_vva_bench`` resolves the
    sibling cell module regardless of the working directory.
    """

    child_pythonpath = os.pathsep.join(
        p for p in (_THIS_DIR, os.environ.get("PYTHONPATH", "")) if p
    )
    env = {**os.environ, "RAYON_NUM_THREADS": str(threads), "PYTHONPATH": child_pythonpath}
    proc = subprocess.run(
        [sys.executable, "-m", _CELL_MODULE, json.dumps(cfg)],
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"bench cell failed (threads={threads}, cfg={cfg}):\n{proc.stderr[-2000:]}"
        )
    return json.loads(proc.stdout)


def run_all(spec: dict[str, Any]) -> dict[str, Any]:
    """Runs the full matrix described by ``spec`` and returns an aggregated bundle."""

    dims: list[int] = spec["dims"]
    bits: list[int] = spec["bit_widths"]
    parity_n: int = spec["parity_n"]
    queries: int = spec["queries"]
    k: int = spec["k"]
    repeats: int = spec["repeats"]
    batch_size: int = spec["batch_size"]
    include_gpu: bool = spec["include_gpu"]
    scaling_ns: list[int] = spec["scaling_ns"]
    scaling_dim: int = spec["scaling_dim"]
    scaling_bit: int = spec["scaling_bit"]
    update_ns: list[int] = spec["update_ns"]
    update_count: int = spec["update_count"]
    datasets: list[dict[str, Any]] = spec.get("datasets", [])
    ncpu = int(os.cpu_count() or 1)
    seed = int(spec.get("seed", 0))
    started = time.perf_counter()

    out: dict[str, Any] = {
        "machine": machine_info(),
        "spec": spec,
        "axes": {
            "speed_parity": [], "speed_scaling": [], "speed_batch": [],
            "recall": [], "memory": [], "update": [],
        },
    }

    def log(msg: str) -> None:
        print(f"[tv-vva] {msg}", flush=True)

    # ---- Speed (parity): vanilla ST + MT (subprocess) + augmented GPU (in-process)
    for dim in dims:
        for bit in bits:
            base = {
                "axis": "speed", "dim": dim, "bit_width": bit, "n": parity_n,
                "queries": queries, "k": k, "repeats": repeats, "seed": seed,
                "batch_size": batch_size,
            }
            log(f"speed parity dim={dim} bit={bit} n={parity_n}")
            st = _subprocess_cell({**base, "which": "vanilla", "include_gpu": False}, threads=1)
            mt = _subprocess_cell({**base, "which": "vanilla", "include_gpu": False}, threads=ncpu)
            aug = run_cell({**base, "which": "augmented", "include_gpu": include_gpu})
            out["axes"]["speed_parity"].append({
                "dim": dim, "bit_width": bit, "n": parity_n, "k": k, "queries": queries,
                "vanilla_st": st.get("speed_vanilla"),
                "vanilla_mt": mt.get("speed_vanilla"),
                "augmented": aug.get("speed_augmented"),
            })

    # ---- Speed (scaling): corpus-size sweep, vanilla MT vs augmented GPU
    for n in scaling_ns:
        base = {
            "axis": "speed", "dim": scaling_dim, "bit_width": scaling_bit, "n": n,
            "queries": queries, "k": k, "repeats": repeats, "seed": seed, "batch_size": batch_size,
        }
        log(f"speed scaling n={n} dim={scaling_dim} bit={scaling_bit}")
        mt = _subprocess_cell({**base, "which": "vanilla", "include_gpu": False}, threads=ncpu)
        aug = run_cell({**base, "which": "augmented", "include_gpu": include_gpu})
        out["axes"]["speed_scaling"].append({
            "n": n, "dim": scaling_dim, "bit_width": scaling_bit,
            "vanilla_mt": mt.get("speed_vanilla"),
            "augmented": aug.get("speed_augmented"),
        })

    # ---- Speed (batch sweep): augmented GPU throughput vs batch size.
    # The GPU exact path scores via a batched GEMM, so its throughput climbs with
    # batch while the CPU SIMD scan is batch-insensitive (flat reference line).
    bs_n = int(spec.get("batch_sweep_n", 0))
    if bs_n and include_gpu:
        bbase = {
            "axis": "speed", "dim": scaling_dim, "bit_width": scaling_bit, "n": bs_n,
            "queries": queries, "k": k, "repeats": repeats, "seed": seed,
        }
        ref = _subprocess_cell(
            {**bbase, "which": "vanilla", "include_gpu": False, "batch_size": 64}, threads=ncpu
        )
        vanilla_mt = ref.get("speed_vanilla")
        for bs in spec.get("batch_sweep_sizes", [1, 16, 64, 256, 1024]):
            log(f"speed batch sweep bs={bs} n={bs_n} dim={scaling_dim} bit={scaling_bit}")
            aug = run_cell({**bbase, "which": "augmented", "include_gpu": True, "batch_size": bs})
            out["axes"]["speed_batch"].append({
                "batch_size": bs, "n": bs_n, "dim": scaling_dim, "bit_width": scaling_bit,
                "augmented": aug.get("speed_augmented"),
                "vanilla_mt": vanilla_mt,
            })

    # ---- Recall: vanilla quantized vs augmented exact-reconstruction (real or synthetic)
    recall_targets = datasets or [
        {"name": f"synthetic_d{d}", "dim": d, "bit_widths": bits} for d in dims
    ]
    for ds in recall_targets:
        for bit in ds.get("bit_widths", bits):
            cfg = {
                "axis": "recall", "dim": ds["dim"], "bit_width": bit, "n": parity_n,
                "queries": queries, "k": k, "seed": seed, "batch_size": batch_size,
                "include_gpu": include_gpu,
            }
            if ds.get("vectors_path"):
                cfg["vectors_path"] = ds["vectors_path"]
            if ds.get("queries_path"):
                cfg["queries_path"] = ds["queries_path"]
            log(f"recall dataset={ds['name']} dim={ds['dim']} bit={bit}")
            cell = run_cell(cfg)
            out["axes"]["recall"].append({
                "dataset": ds["name"], "dim": ds["dim"], "bit_width": bit,
                "vanilla": cell.get("recall_vanilla"),
                "augmented": cell.get("recall_augmented"),
            })

    # ---- Memory: compact (vanilla) vs fp16 resident (augmented)
    for dim in dims:
        for bit in bits:
            log(f"memory dim={dim} bit={bit}")
            cell = run_cell(
                {"axis": "memory", "dim": dim, "bit_width": bit, "n": parity_n, "seed": seed}
            )
            out["axes"]["memory"].append(cell["memory"])

    # ---- Update/persist: vanilla full rewrite vs augmented delta export
    for n in update_ns:
        log(f"update n={n} updates={update_count}")
        cell = run_cell({
            "axis": "update", "dim": scaling_dim, "bit_width": scaling_bit, "n": n,
            "update_count": update_count, "seed": seed,
        })
        out["axes"]["update"].append(cell["update"])

    out["wall_seconds"] = time.perf_counter() - started
    return out


def smoke_spec() -> dict[str, Any]:
    """Tiny synthetic, CPU-only matrix for local validation (no GPU, seconds)."""

    return {
        "dims": [256, 384], "bit_widths": [2, 4], "parity_n": 4000, "queries": 200,
        "k": 64, "repeats": 2, "batch_size": 64, "include_gpu": True,
        "scaling_ns": [2000, 4000, 8000], "scaling_dim": 256, "scaling_bit": 4,
        "batch_sweep_n": 4000, "batch_sweep_sizes": [1, 16, 64, 256],
        "update_ns": [4000, 16000], "update_count": 500, "seed": 0,
    }


def full_spec() -> dict[str, Any]:
    """Upstream-parity (100K, k=64) + a GPU corpus-scaling sweep."""

    return {
        "dims": [1536, 3072], "bit_widths": [2, 4], "parity_n": 100_000, "queries": 1000,
        "k": 64, "repeats": 5, "batch_size": 64, "include_gpu": True,
        "scaling_ns": [100_000, 250_000, 500_000, 1_000_000],
        "scaling_dim": 1536, "scaling_bit": 4,
        "batch_sweep_n": 100_000, "batch_sweep_sizes": [1, 16, 64, 256, 1024],
        "update_ns": [100_000, 500_000, 1_000_000], "update_count": 1000, "seed": 0,
    }


def main(argv: list[str]) -> int:
    """CLI entry: build a spec (smoke|full), run, write results JSON."""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="tiny synthetic CPU-only matrix")
    ap.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results" / "results_a10.json"),
    )
    ap.add_argument("--no-gpu", action="store_true", help="skip the augmented GPU series")
    ap.add_argument("--datasets-json", default=None, help="real-dataset spec for recall")
    args = ap.parse_args(argv[1:])

    spec = smoke_spec() if args.smoke else full_spec()
    if args.no_gpu:
        spec["include_gpu"] = False
    if args.datasets_json:
        spec["datasets"] = json.loads(Path(args.datasets_json).read_text())

    bundle = run_all(spec)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True))
    print(f"[tv-vva] wrote {out_path} ({bundle['wall_seconds']:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
