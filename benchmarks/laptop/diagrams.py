#!/usr/bin/env python3
"""Render LodeDB laptop-benchmark charts from a results JSON (matplotlib).

    python benchmarks/laptop/diagrams.py \
        --results benchmarks/laptop/results/laptop_m1.json \
        --out docs

Produces two charts (PNG + SVG): embedding throughput by device, and end-to-end
query latency (p50/p95) by device. matplotlib is a dev-only dependency; it is not
part of the lodedb runtime set, so install it separately to render charts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DEVICE_LABEL = {"mps": "MPS", "cpu": "CPU"}
_DEVICE_COLOR = {"mps": "#4c72b0", "cpu": "#55a868"}


def _device_labels(runs: list[dict]) -> tuple[list[str], list[str]]:
    """Returns (display labels, bar colors) for each device run."""

    labels, colors = [], []
    for run in runs:
        dev = run["config"]["requested_device"]
        label = _DEVICE_LABEL.get(dev, dev)
        if run["config"].get("fallback_used"):
            label += f"\n(→{run['config']['effective_device']})"
        labels.append(label)
        colors.append(_DEVICE_COLOR.get(dev, "#888888"))
    return labels, colors


def _save(fig, outdir: Path, name: str) -> None:
    """Writes a figure as both PNG and SVG into the output directory."""

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)


def _bar_labels(ax, bars, fmt: str) -> None:
    """Annotates each bar with its value."""

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def render_embed_throughput(data: dict, outdir: Path) -> None:
    """Bar chart: bulk-index embedding throughput (docs/s) per device."""

    runs = data["runs"]
    labels, colors = _device_labels(runs)
    values = [run["embedding"]["docs_per_second"] for run in runs]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, values, color=colors, width=0.6)
    _bar_labels(ax, bars, "{:.0f}")
    ax.set_ylabel("documents / second (higher is better)")
    ax.set_title(
        f"Embedding throughput — {data['model']}, {data['doc_count']:,} docs "
        f"(warm, batch 64)"
    )
    ax.margins(y=0.15)
    _save(fig, outdir, "embed_throughput")


def render_query_latency(data: dict, outdir: Path) -> None:
    """Grouped bar chart: end-to-end query latency p50/p95 per device."""

    runs = data["runs"]
    labels, colors = _device_labels(runs)
    p50 = [run["cpu_vector_scan"]["end_to_end_query_ms_p50"] for run in runs]
    p95 = [run["cpu_vector_scan"]["end_to_end_query_ms_p95"] for run in runs]

    x = range(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    b1 = ax.bar([i - width / 2 for i in x], p50, width, label="p50", color="#4c72b0")
    b2 = ax.bar([i + width / 2 for i in x], p95, width, label="p95", color="#c44e52")
    _bar_labels(ax, b1, "{:.1f}")
    _bar_labels(ax, b2, "{:.1f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("end-to-end query latency, ms (lower is better)")
    ax.set_title("Query latency = embed one query + CPU TurboVec scan")
    ax.legend()
    ax.margins(y=0.15)
    _save(fig, outdir, "query_latency")


def main() -> None:
    """Renders all laptop-benchmark charts from a results JSON."""

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Render LodeDB laptop-benchmark charts")
    parser.add_argument("--results", default=str(here / "results" / "laptop_m1.json"))
    parser.add_argument("--out", default=str(here / "docs"))
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    outdir = Path(args.out)
    render_embed_throughput(data, outdir)
    render_query_latency(data, outdir)
    print(f"wrote charts to {outdir}")


if __name__ == "__main__":
    main()
