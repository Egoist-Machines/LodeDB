#!/usr/bin/env python3
"""Render the batched-retrieval throughput chart from a results JSON (matplotlib).

    python benchmarks/batched_retrieval/diagrams.py \
        --results benchmarks/batched_retrieval/results/laptop_m1.json \
        --out docs

Plots ``search_many`` queries/sec vs query batch size: one line for the CPU kernel and,
when the results include GPU rows, one for the GPU-resident path. matplotlib is a dev-only
dependency (not part of the lodedb runtime set); install it separately to render charts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_CPU_COLOR = "#55a868"
_GPU_COLOR = "#4c72b0"


def _series(rows: list[dict], policy_values: set[str]) -> tuple[list[int], list[float]]:
    """Returns (batch_sizes, queries_per_second) for rows matching the policies, sorted."""

    points = sorted(
        (r["batch_size"], r["queries_per_second"]) for r in rows if r["policy"] in policy_values
    )
    return [b for b, _ in points], [q for _, q in points]


def _annotate(ax, xs: list[int], ys: list[float]) -> None:
    """Labels each point with its throughput."""

    for x, y in zip(xs, ys, strict=True):
        ax.annotate(
            f"{y:,.0f}",
            xy=(x, y),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )


def render_throughput(data: dict, outdir: Path) -> None:
    """Line chart: search_many queries/sec vs batch size (CPU, and GPU when present)."""

    rows = data["rows"]
    cfg = data["config"]
    cpu_x, cpu_y = _series(rows, {"off"})
    gpu_x, gpu_y = _series(rows, {"auto", "required"})

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(cpu_x, cpu_y, marker="o", color=_CPU_COLOR, label="CPU kernel")
    _annotate(ax, cpu_x, cpu_y)
    if gpu_x:
        ax.plot(gpu_x, gpu_y, marker="s", color=_GPU_COLOR, label="GPU-resident")
        _annotate(ax, gpu_x, gpu_y)

    ax.set_xscale("log", base=2)
    ax.set_xticks(cpu_x)
    ax.set_xticklabels([str(b) for b in cpu_x])
    ax.set_xlabel("query batch size (search_many)")
    ax.set_ylabel("queries / second (higher is better)")
    gpu_note = "" if gpu_x else "  ·  GPU not available on this host"
    ax.set_title(
        f"Batched retrieval throughput — {cfg['model']}, "
        f"{cfg['doc_count']:,} docs, dim {cfg['native_dim']}{gpu_note}"
    )
    ax.grid(True, which="both", axis="y", alpha=0.3)
    ax.legend()
    ax.margins(y=0.18)

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"throughput_batch.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)


def main() -> None:
    """Renders the batched-retrieval chart from a results JSON."""

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Render batched-retrieval charts")
    parser.add_argument("--results", default=str(here / "results" / "laptop_m1.json"))
    parser.add_argument("--out", default=str(here / "docs"))
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    outdir = Path(args.out)
    render_throughput(data, outdir)
    print(f"wrote charts to {outdir}")


if __name__ == "__main__":
    main()
