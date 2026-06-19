#!/usr/bin/env python3
"""Render GovReport-scale charts from the results JSON(s) (matplotlib).

    python benchmarks/govreport_scale/diagrams.py \
        --results benchmarks/govreport_scale/results/results.json --out docs

Three charts: recall@1 vs corpus size (vanilla uint8-LUT scan vs augmented
fp16-reconstruction scan), throughput vs corpus size, and the batch sweep at the top size.

The CPU scan's throughput depends on the host CPU kernel (Modal assigns it), so the speed
and batch charts overlay BOTH measured hosts when an AVX-512 results file is present
(``--results-avx512``, default ``results/results_avx512.json``): the GPU/augmented line is
host-independent (it's the L40S either way), while the CPU ceiling shifts with the host.
Recall is host-independent, so that chart uses the primary run only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HOST_LABELS = {"avx512bw": "AVX-512", "avx512": "AVX-512", "avx2": "AVX2", "neon": "NEON"}


def _km(value: float) -> str:
    """Formats a count as K/M for compact axis labels."""

    if value >= 1e6:
        return f"{value / 1e6:.1f}M"
    if value >= 1e3:
        return f"{value / 1e3:.0f}K"
    return str(int(value))


def _host_label(data: dict) -> str:
    """Returns a display label for the run's host CPU kernel (AVX-512 / AVX2 / ...)."""

    backend = (data.get("machine") or {}).get("turbovec_native_backend") or "cpu"
    return _HOST_LABELS.get(backend, str(backend))


def _qps(row: dict, key: str) -> float:
    """Returns queries/sec for a speed sub-result, or 0.0 when absent."""

    value = row.get(key)
    return float(value["queries_per_sec"]) if value else 0.0


def _save(fig, outdir: Path, name: str) -> None:
    """Writes a figure as PNG + SVG."""

    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)


def recall_vs_scale(data: dict, outdir: Path) -> None:
    """R@1-in-top-1 vs corpus size: vanilla uint8-LUT vs augmented fp16-reconstruction.

    Recall is CPU-arch-independent, so a single run suffices.
    """

    rows = sorted(data["recall"], key=lambda r: r["n"])
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(ns, [r["vanilla"]["1"] for r in rows], "o-", color="#55a868",
            label="vanilla — uint8-LUT scan")
    ax.plot(ns, [r["augmented"]["1"] for r in rows], "s-", color="#4c72b0",
            label="augmented — fp16 reconstruction")
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([_km(n) for n in ns])
    ax.set_xlabel("corpus size (vectors)")
    ax.set_ylabel("R@1-in-top-1 vs fp32 brute force")
    ax.set_title("Recall vs scale — GovReport, MiniLM 384-d, 4-bit (host-independent)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, outdir, "recall_vs_scale")


def speed_vs_scale(data: dict, data512: dict | None, outdir: Path) -> None:
    """Throughput vs corpus size: vanilla ST/MT (CPU ceiling, per host) vs augmented GPU.

    Overlays both measured CPU hosts; the augmented GPU line (L40S) is host-independent.
    """

    rows = sorted(data["speed"], key=lambda r: r["n"])
    ns = [r["n"] for r in rows]
    host = _host_label(data)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    # Stronger CPU host (AVX-512) drawn solid + darker; primary run (AVX2) dashed + lighter.
    if data512 is not None:
        rows512 = sorted(data512["speed"], key=lambda r: r["n"])
        ns512 = [r["n"] for r in rows512]
        host512 = _host_label(data512)
        ax.plot(ns512, [_qps(r, "vanilla_mt") for r in rows512], "o-", color="#2f7d3a",
                label=f"vanilla all-threads — {host512}")
        ax.plot(ns512, [_qps(r, "vanilla_st") for r in rows512], "^-", color="#6f6f6f",
                label=f"vanilla 1-thread — {host512}")
    ax.plot(ns, [_qps(r, "vanilla_mt") for r in rows], "o--", color="#7bc486",
            label=f"vanilla all-threads — {host}")
    ax.plot(ns, [_qps(r, "vanilla_st") for r in rows], "^--", color="#b0b0b0",
            label=f"vanilla 1-thread — {host}")
    ax.plot(ns, [_qps(r, "augmented") for r in rows], "s-", color="#4c72b0",
            label="augmented — fp16 reconstruction (L40S, batch 64)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([_km(n) for n in ns])
    ax.set_xlabel("corpus size (vectors)")
    ax.set_ylabel("throughput (queries / sec, log)")
    ax.set_title("Search throughput vs scale — GovReport 384-d, 4-bit (CPU host varies)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, outdir, "speed_vs_scale")


def batch_sweep(data: dict, data512: dict | None, outdir: Path) -> None:
    """Augmented GPU throughput vs batch at the top corpus size, vs each host's CPU ceiling."""

    rows = sorted(data.get("batch", []), key=lambda r: r["batch_size"])
    if not rows:
        return
    batches = [r["batch_size"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(batches, [r["augmented"]["queries_per_sec"] for r in rows], "s-",
            color="#4c72b0", label="augmented — fp16 reconstruction (L40S)")

    if data512 is not None and data512.get("batch"):
        rows512 = sorted(data512["batch"], key=lambda r: r["batch_size"])
        mt512 = float(rows512[0]["vanilla_mt"]["queries_per_sec"])
        ax.axhline(mt512, ls="--", color="#2f7d3a",
                   label=f"CPU all-threads — {_host_label(data512)} ≈ {mt512:,.0f} q/s")
    mt = float(rows[0]["vanilla_mt"]["queries_per_sec"])
    ax.axhline(mt, ls="--", color="#7bc486",
               label=f"CPU all-threads — {_host_label(data)} ≈ {mt:,.0f} q/s")

    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("query batch size")
    ax.set_ylabel("throughput (queries / sec)")
    ax.set_title(f"GPU throughput vs batch — GovReport {_km(rows[0]['n'])} vectors, 4-bit")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save(fig, outdir, "batch_sweep")


def main() -> None:
    """Renders all GovReport-scale charts (overlaying both CPU hosts where available)."""

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Render GovReport-scale charts")
    parser.add_argument("--results", default=str(here / "results" / "results.json"))
    parser.add_argument("--results-avx512", default=str(here / "results" / "results_avx512.json"))
    parser.add_argument("--out", default=str(here / "docs"))
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    avx512_path = Path(args.results_avx512)
    data512 = json.loads(avx512_path.read_text()) if avx512_path.exists() else None
    outdir = Path(args.out)
    recall_vs_scale(data, outdir)
    speed_vs_scale(data, data512, outdir)
    batch_sweep(data, data512, outdir)
    hosts = _host_label(data) + ("" if data512 is None else f" + {_host_label(data512)}")
    print(f"wrote charts to {outdir} (hosts: {hosts})")


if __name__ == "__main__":
    main()
