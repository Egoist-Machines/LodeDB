#!/usr/bin/env python3
"""Render the MPS-vs-NEON throughput chart from a results JSON (matplotlib).

    python benchmarks/mps_vs_neon/diagrams.py \
        --results benchmarks/mps_vs_neon/results/mps_vs_neon_m1.json --out docs

matplotlib is a dev-only tool, not a LodeDB runtime dependency — install it
separately (``uv pip install matplotlib``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    """Renders a NEON-vs-MPS throughput-by-batch line chart (PNG + SVG)."""

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Render the MPS-vs-NEON chart")
    parser.add_argument("--results", default=str(here / "results" / "mps_vs_neon_m1.json"))
    parser.add_argument("--out", default=str(here / "docs"))
    args = parser.parse_args()

    data = json.loads(Path(args.results).read_text())
    speed = data["speed"]
    cfg = data["config"]
    batches = [s["batch"] for s in speed]

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(batches, [s["neon_qps"] for s in speed], "o-", color="#55a868",
            label="NEON CPU scan (default)")
    ax.plot(batches, [s["mps_qps"] for s in speed], "s-", color="#4c72b0",
            label="MPS exact scan (opt-in)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("query batch size")
    ax.set_ylabel("throughput (queries / sec, higher is better)")
    ax.set_title(
        f"MPS exact vs. NEON CPU scan — d={cfg['dim']}, "
        f"{cfg['n']:,} vectors, {cfg['bit_width']}-bit"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(outdir / f"speed_batch.{ext}", bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"wrote charts to {outdir}")


if __name__ == "__main__":
    main()
