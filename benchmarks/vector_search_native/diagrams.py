"""Render the native vector-only throughput sweep as a throughput-vs-batch chart.

Reads the A10 and L40S result JSONs and plots queries/sec vs batch size for the
GPU-resident scan (current default) against the plain TurboVec CPU scan.

    python benchmarks/vector_search_native/diagrams.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"


def _series(summary: dict, name: str) -> tuple[list[int], list[float]]:
    """Returns (batch_sizes, queries_per_sec) for one series, sorted by batch."""

    points = sorted(
        (int(r["batch_size"]), float(r["queries_per_sec"]))
        for r in summary["rows"]
        if r["series"] == name
    )
    return [b for b, _ in points], [q for _, q in points]


def _load(name: str) -> dict | None:
    path = _RESULTS / name
    if not path.exists():
        print(f"skip: {path} not found")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    a10 = _load("results_a10.json")
    l40s = _load("results_l40s.json")
    if a10 is None and l40s is None:
        raise SystemExit("no results found; run modal_bench.py::main_a10 / main_l40s first")

    reference = a10 or l40s
    dim = reference["dim"]
    n = reference["n"]
    bit_width = reference["bit_width"]
    top_k = reference["top_k"]

    fig, ax = plt.subplots(figsize=(9, 6))
    # Both series are end-to-end through the public API, so the GPU-vs-CPU gap is a
    # like-for-like comparison. The vanilla-vs-augmented CPU band is deliberately not
    # drawn: it is a raw-scan number (API bypassed), a different regime.
    series_defs = [
        ("a10", a10, {"color": "#d62728", "marker": "s"}),
        ("l40s", l40s, {"color": "#7b3ff2", "marker": "o"}),
    ]
    for label, summary, style in series_defs:
        if summary is None:
            continue
        gpu_x, gpu_y = _series(summary, "gpu")
        cpu_x, cpu_y = _series(summary, "cpu")
        ax.plot(
            gpu_x, gpu_y, label=f"{label.upper()} GPU (native default)", linewidth=2, **style
        )
        ax.plot(
            cpu_x,
            cpu_y,
            label=f"{label.upper()} CPU (native, augmented)",
            linewidth=1.3,
            linestyle="--",
            alpha=0.7,
            **style,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 16, 64, 256, 1024])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("query batch size")
    ax.set_ylabel("throughput (queries / sec)")
    ax.set_title(
        f"Native vector search, end-to-end public API: GPU default vs augmented CPU\n"
        f"(d={dim}, {bit_width}-bit, n={n:,}, k={top_k})"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out = _HERE / "docs" / "throughput_batch.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
