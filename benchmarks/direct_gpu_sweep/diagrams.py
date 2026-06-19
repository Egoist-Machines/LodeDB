#!/usr/bin/env python3
"""Render direct CUDA GPU sweep charts from LodeDB result JSONs.

    python benchmarks/direct_gpu_sweep/diagrams.py \
        --out benchmarks/direct_gpu_sweep/docs

By default this reads the measured full-run artifacts:
``results/results_a10.json`` and ``results/results_l40s.json``.

Produces PNG + SVG charts for the GPU batch scan, per-query search latency,
end-to-end batch latency, recall parity, and GPU memory/copy accounting.
matplotlib is a dev-only tool, not a LodeDB runtime dependency.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

_GPU_COLORS = {
    "A10": "#4c72b0",
    "L40S": "#c44e52",
}
_CPU_COLORS = {
    "A10": "#9ecae1",
    "L40S": "#f2a6a3",
}
_GRID_ALPHA = 0.3


@dataclass(frozen=True)
class SweepRun:
    """One measured direct GPU sweep artifact."""

    label: str
    path: Path
    data: dict[str, Any]
    rows: dict[str, dict[str, Any]]

    @property
    def batches(self) -> list[int]:
        """Returns the measured batch sizes in ascending order."""

        return [int(batch) for batch in self.data["batch_sizes"]]

    def row(self, prefix: str, batch_size: int) -> dict[str, Any]:
        """Returns a row by prefix and batch size."""

        return self.rows[f"{prefix}_direct_batch_{batch_size}"]


def _km_fmt(value: float, _pos: object = None) -> str:
    """Compact axis labels for bytes/counts."""

    value = float(value)
    if abs(value) >= 1e9:
        return f"{value / 1e9:.1f}G"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.0f}K"
    return f"{value:g}"


KM = FuncFormatter(_km_fmt)


def _save(fig: Any, outdir: Path, name: str) -> None:
    """Writes a figure as PNG + SVG."""

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)


def _label_from_path(path: Path) -> str:
    """Infers a display label from a result file name."""

    stem = path.stem.lower()
    if "l40s" in stem:
        return "L40S"
    if "a10" in stem:
        return "A10"
    return path.stem.replace("results_", "").upper()


def _load_run(path: Path) -> SweepRun:
    """Loads one result artifact and indexes rows by row name."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("artifact_type") != "lodedb_direct_turbovec_gpu_sweep":
        raise ValueError(f"{path} is not a LodeDB direct GPU sweep artifact")
    rows = {str(row["row"]): row for row in data.get("rows", [])}
    if not rows:
        raise ValueError(f"{path} does not contain rows")
    return SweepRun(label=_label_from_path(path), path=path, data=data, rows=rows)


def _batch_ge_2(run: SweepRun) -> list[int]:
    """Returns batch sizes that are expected to use GPU."""

    return [batch for batch in run.batches if batch >= 2]


def _plot_gpu_scan_ms(runs: list[SweepRun], outdir: Path) -> None:
    """Plots the actual GPU scan event duration by batch size."""

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for run in runs:
        batches = _batch_ge_2(run)
        values = [
            float(run.row("gpu", batch)["gpu_stage_one_search_ms"])
            for batch in batches
        ]
        ax.plot(
            batches,
            values,
            "o-",
            color=_GPU_COLORS.get(run.label, "#4c72b0"),
            label=f"{run.label} GPU scan",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(_batch_ge_2(runs[0]))
    ax.set_xticklabels([str(batch) for batch in _batch_ge_2(runs[0])])
    ax.set_xlabel("query batch size")
    ax.set_ylabel("GPU stage-one scan time (ms per batch)")
    ax.set_title("Direct CUDA scan time by batch size")
    ax.legend()
    ax.grid(True, alpha=_GRID_ALPHA, which="both")
    _save(fig, outdir, "gpu_scan_ms")


def _plot_search_p50_ms(runs: list[SweepRun], outdir: Path) -> None:
    """Plots per-query search p50 for CPU policy vs GPU policy."""

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for run in runs:
        batches = run.batches
        cpu = [float(run.row("cpu", batch)["search_p50_ms"]) for batch in batches]
        gpu = [float(run.row("gpu", batch)["search_p50_ms"]) for batch in batches]
        ax.plot(
            batches,
            cpu,
            "o--",
            color=_CPU_COLORS.get(run.label, "#9ecae1"),
            label=f"{run.label} CPU per-query p50",
        )
        ax.plot(
            batches,
            gpu,
            "s-",
            color=_GPU_COLORS.get(run.label, "#4c72b0"),
            label=f"{run.label} GPU-policy per-query p50",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(runs[0].batches)
    ax.set_xticklabels([str(batch) for batch in runs[0].batches])
    ax.set_xlabel("query batch size")
    ax.set_ylabel("per-query search p50 (ms, log)")
    ax.set_title("Per-query search latency: CPU policy vs GPU policy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=_GRID_ALPHA, which="both")
    _save(fig, outdir, "search_p50_ms")


def _plot_batch_latency_ms(runs: list[SweepRun], outdir: Path) -> None:
    """Plots end-to-end query-batch p50 latency, including embedding and hydration."""

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for run in runs:
        batches = run.batches
        cpu = [float(run.row("cpu", batch)["batch_p50_ms"]) for batch in batches]
        gpu = [float(run.row("gpu", batch)["batch_p50_ms"]) for batch in batches]
        ax.plot(
            batches,
            cpu,
            "o--",
            color=_CPU_COLORS.get(run.label, "#9ecae1"),
            label=f"{run.label} CPU policy",
        )
        ax.plot(
            batches,
            gpu,
            "s-",
            color=_GPU_COLORS.get(run.label, "#4c72b0"),
            label=f"{run.label} GPU policy",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(runs[0].batches)
    ax.set_xticklabels([str(batch) for batch in runs[0].batches])
    ax.set_xlabel("query batch size")
    ax.set_ylabel("end-to-end batch p50 (ms, log)")
    ax.set_title("End-to-end batch latency, including query embedding")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=_GRID_ALPHA, which="both")
    _save(fig, outdir, "batch_latency_ms")


def _plot_recall_parity(runs: list[SweepRun], outdir: Path) -> None:
    """Plots CPU/GPU document-recall parity and recall gap."""

    fig, (ax_recall, ax_gap) = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
    for run in runs:
        batches = run.batches
        cpu_recall = [
            float(run.row("cpu", batch)["document_recall_at_top_k"])
            for batch in batches
        ]
        gpu_recall = [
            float(run.row("gpu", batch)["document_recall_at_top_k"])
            for batch in batches
        ]
        gaps = [
            float(run.row("gpu", batch).get("document_recall_gap_vs_cpu", 0.0))
            for batch in batches
        ]
        color = _GPU_COLORS.get(run.label, "#4c72b0")
        ax_recall.plot(batches, cpu_recall, "o--", color=_CPU_COLORS.get(run.label, color),
                       label=f"{run.label} CPU recall")
        ax_recall.plot(batches, gpu_recall, "s-", color=color, label=f"{run.label} GPU recall")
        ax_gap.plot(batches, gaps, "o-", color=color, label=f"{run.label} recall gap")
    ax_recall.set_ylabel("document recall@top-k")
    ax_recall.set_title("CPU/GPU document-recall parity")
    ax_recall.legend(fontsize=8)
    ax_recall.grid(True, alpha=_GRID_ALPHA)
    ax_recall.set_ylim(0.99, 1.001)
    ax_gap.axhline(0.002, color="#6b7280", ls="--", label="launch tolerance (0.002)")
    ax_gap.set_xscale("log", base=2)
    ax_gap.set_xticks(runs[0].batches)
    ax_gap.set_xticklabels([str(batch) for batch in runs[0].batches])
    ax_gap.set_xlabel("query batch size")
    ax_gap.set_ylabel("absolute recall gap")
    ax_gap.set_ylim(0.0, 0.0022)
    ax_gap.legend(fontsize=8)
    ax_gap.grid(True, alpha=_GRID_ALPHA, which="both")
    _save(fig, outdir, "recall_parity")


def _plot_memory_copy(runs: list[SweepRun], outdir: Path) -> None:
    """Plots resident memory estimate and device-to-host copy bytes."""

    fig, (ax_memory, ax_copy) = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
    for run in runs:
        batches = _batch_ge_2(run)
        estimated_mb = [
            float(run.row("gpu", batch)["gpu_estimated_bytes"]) / (1024 * 1024)
            for batch in batches
        ]
        copy_mb = [
            float(run.row("gpu", batch)["gpu_copy_back_bytes"]) / (1024 * 1024)
            for batch in batches
        ]
        color = _GPU_COLORS.get(run.label, "#4c72b0")
        ax_memory.plot(batches, estimated_mb, "o-", color=color, label=f"{run.label} estimate")
        ax_copy.plot(batches, copy_mb, "s-", color=color, label=f"{run.label} copy-back")
    ax_memory.set_ylabel("estimated GPU bytes (MiB)")
    ax_memory.set_title("GPU memory admission and copy-back accounting")
    ax_memory.legend(fontsize=8)
    ax_memory.grid(True, alpha=_GRID_ALPHA)
    ax_copy.set_xscale("log", base=2)
    ax_copy.set_yscale("log")
    ax_copy.set_xticks(_batch_ge_2(runs[0]))
    ax_copy.set_xticklabels([str(batch) for batch in _batch_ge_2(runs[0])])
    ax_copy.set_xlabel("query batch size")
    ax_copy.set_ylabel("device-to-host copy-back (MiB, log)")
    ax_copy.legend(fontsize=8)
    ax_copy.grid(True, alpha=_GRID_ALPHA, which="both")
    _save(fig, outdir, "memory_copy")


def _print_summary(runs: list[SweepRun], outdir: Path) -> None:
    """Prints a compact launch-proof summary."""

    labels = ", ".join(run.label for run in runs)
    artifact_list = ", ".join(str(run.path) for run in runs)
    print(f"wrote charts to {outdir} ({labels})")
    print(f"source artifacts: {artifact_list}")


def main() -> None:
    """Renders all direct GPU sweep charts."""

    here = Path(__file__).resolve().parent
    default_results = [
        here / "results" / "results_a10.json",
        here / "results" / "results_l40s.json",
    ]
    parser = argparse.ArgumentParser(description="Render direct CUDA GPU sweep charts")
    parser.add_argument(
        "--results",
        action="append",
        default=None,
        help="Result JSON to plot; repeat for multiple GPUs.",
    )
    parser.add_argument("--out", default=str(here / "docs"))
    args = parser.parse_args()

    result_paths = [Path(path) for path in (args.results or default_results)]
    runs = [_load_run(path) for path in result_paths]
    runs.sort(
        key=lambda run: (
            ("L40S", "A10").index(run.label) if run.label in {"L40S", "A10"} else 99
        )
    )
    outdir = Path(args.out)
    _plot_gpu_scan_ms(runs, outdir)
    _plot_search_p50_ms(runs, outdir)
    _plot_batch_latency_ms(runs, outdir)
    _plot_recall_parity(runs, outdir)
    _plot_memory_copy(runs, outdir)
    _print_summary(runs, outdir)


if __name__ == "__main__":
    main()
