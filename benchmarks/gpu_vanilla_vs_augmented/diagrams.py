"""Render vanilla-vs-augmented TurboVec benchmark diagrams from a results JSON.

Plots the vanilla uint8-LUT CPU SIMD scan against the augmented
fp16-reconstruction GPU scan on each axis (recall curves, search speed, memory,
update). No FAISS. Both scan the same 4-bit index; the difference is the scoring
step (the CPU sums a uint8 LUT (ADC), the GPU does a full fp16 GEMM dot product),
so "exact" here means exact over the 4-bit reconstruction, not fp32. One figure
per axis, written as both PNG and SVG.

  python diagrams.py --results results/results_a10.json --out docs/

Robust to a missing/skipped augmented series (e.g. a CPU-only laptop run): it
plots whatever is present and annotates the rest as unavailable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

VANILLA_C = "#2563eb"   # blue: vanilla CPU
VANILLA2_C = "#93c5fd"  # light blue: vanilla single-thread
AUG_C = "#dc2626"       # red: augmented GPU
FP32_C = "#94a3b8"      # grey: fp32 reference
K_GRID = (1, 2, 4, 8, 16, 32, 64)


def _km_fmt(x: float, _pos: object = None) -> str:
    """Compact axis label: 1_000_000 -> '1M', 250_000 -> '250K', 7_487 -> '7.5K'."""

    x = float(x)
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f}".rstrip("0").rstrip(".") + "K"
    return f"{x:g}"


KM = FuncFormatter(_km_fmt)


def _augmented_ran(series: Any) -> bool:
    """Returns whether an augmented series carries real data (not skipped/None)."""

    return isinstance(series, dict) and "skipped" not in series and bool(series)


def _save(fig: Any, out: Path, name: str) -> list[str]:
    """Saves a figure as PNG + SVG and returns the written paths."""

    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "svg"):
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=140)
        written.append(str(p))
    plt.close(fig)
    return written


def plot_speed_parity(rows: list[dict], out: Path) -> list[str]:
    """Grouped bars: vanilla ST, vanilla MT, augmented GPU q/s per (dim, bit)."""

    if not rows:
        return []
    labels = [f"d={r['dim']}\n{r['bit_width']}-bit" for r in rows]
    st = [(_g(r, "vanilla_st", "queries_per_sec")) for r in rows]
    mt = [(_g(r, "vanilla_mt", "queries_per_sec")) for r in rows]
    aug = [
        _g(r, "augmented", "queries_per_sec") if _augmented_ran(r.get("augmented")) else 0.0
        for r in rows
    ]
    x = range(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(rows)), 4.6))
    ax.bar([i - w for i in x], st, w, label="vanilla — uint8-LUT scan (1 thread)", color=VANILLA2_C)
    ax.bar(list(x), mt, w, label="vanilla — uint8-LUT scan (all threads)", color=VANILLA_C)
    if any(a > 0 for a in aug):
        ax.bar([i + w for i in x], aug, w, label="augmented — fp16 reconstruction", color=AUG_C)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("throughput (queries / sec)")
    n_lbl, k_lbl = rows[0].get("n", "?"), rows[0].get("k", "?")
    ax.set_title(f"Search speed — uint8-LUT scan vs fp16 reconstruction (n={n_lbl:,}, k={k_lbl})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(KM)
    return _save(fig, out, "speed_parity")


def plot_speed_scaling(rows: list[dict], out: Path) -> list[str]:
    """Line plot: q/s vs corpus size, vanilla MT vs augmented GPU (log-x)."""

    if not rows:
        return []
    ns = [r["n"] for r in rows]
    van = [_g(r, "vanilla_mt", "queries_per_sec") for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ns, van, "o-", color=VANILLA_C, label="vanilla — uint8-LUT scan (all threads)")
    aug = [
        _g(r, "augmented", "queries_per_sec") if _augmented_ran(r.get("augmented")) else None
        for r in rows
    ]
    if any(a for a in aug):
        ax.plot(
            [n for n, a in zip(ns, aug, strict=True) if a],
            [a for a in aug if a],
            "s-", color=AUG_C, label="augmented — fp16 reconstruction",
        )
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.minorticks_off()
    ax.xaxis.set_major_formatter(KM)
    ax.yaxis.set_major_formatter(KM)
    ax.set_xlabel("corpus size (vectors)")
    ax.set_ylabel("throughput (queries / sec)")
    d, b = rows[0]["dim"], rows[0]["bit_width"]
    ax.set_title(f"Search throughput vs corpus size (d={d}, {b}-bit)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, out, "speed_scaling")


L40S_C = "#7c3aed"  # violet: augmented GPU, L40S


def plot_speed_batch(
    rows: list[dict], out: Path, rows_l40s: list[dict] | None = None
) -> list[str]:
    """fp16-reconstruction GPU q/s vs batch for A10 (+ L40S when given); the vanilla
    uint8-LUT all-threads CPU scan is a shaded reference band (one baseline per host)."""

    rows = [r for r in rows if _augmented_ran(r.get("augmented"))]
    if not rows:
        return []
    bs = [r["batch_size"] for r in rows]
    aug = [_g(r, "augmented", "queries_per_sec") for r in rows]
    rows_b = [r for r in (rows_l40s or []) if _augmented_ran(r.get("augmented"))]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    # CPU reference band: the A10 and L40S hosts have different CPUs, so their
    # all-threads baselines bound a band rather than a single line.
    cpu = [_g(rows[0], "vanilla_mt", "queries_per_sec")]
    if rows_b:
        cpu.append(_g(rows_b[0], "vanilla_mt", "queries_per_sec"))
    cpu = [v for v in cpu if v > 0]
    if cpu:
        lo, hi = min(cpu), max(cpu)
        ax.axhspan(
            lo, hi, color=FP32_C, alpha=0.2,
            label=f"vanilla TurboVec CPU, all threads ({lo / 1e3:.1f} to {hi / 1e3:.1f}k q/s)",
        )

    ax.plot(bs, aug, "s-", color=AUG_C, label="A10 GPU (fp16 reconstruction)")
    if rows_b:
        bs2 = [r["batch_size"] for r in rows_b]
        aug2 = [_g(r, "augmented", "queries_per_sec") for r in rows_b]
        ax.plot(bs2, aug2, "o-", color=L40S_C, label="L40S GPU (fp16 reconstruction)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(bs)
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("query batch size")
    ax.set_ylabel("throughput (queries / sec)")
    d, b, n = rows[0]["dim"], rows[0]["bit_width"], rows[0]["n"]
    title = "GPU throughput vs batch size" + (": A10 vs L40S" if rows_b else "")
    ax.set_title(f"{title} (d={d}, {b}-bit, n={n:,})")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    ax.yaxis.set_major_formatter(KM)
    return _save(fig, out, "speed_batch")


def plot_recall(rows: list[dict], out: Path) -> list[str]:
    """R@1-within-top-k curves vs k, vanilla vs augmented, one panel per (dataset, bit)."""

    panels = [r for r in rows if r.get("vanilla")]
    if not panels:
        return []
    cols = min(3, len(panels))
    import math

    nrows = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.4 * cols, 3.6 * nrows), squeeze=False)
    for i, r in enumerate(panels):
        ax = axes[i // cols][i % cols]
        ks = [k for k in K_GRID if str(k) in r["vanilla"]]
        ax.plot(
            ks, [r["vanilla"][str(k)] for k in ks],
            "o-", color=VANILLA_C, label="vanilla — uint8-LUT scan",
        )
        if _augmented_ran(r.get("augmented")):
            ax.plot(
                ks, [r["augmented"][str(k)] for k in ks if str(k) in r["augmented"]],
                "s-", color=AUG_C, label="augmented — fp16 reconstruction",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_ylim(min(0.3, min(r["vanilla"].values()) - 0.05), 1.01)
        ax.set_title(f"{r['dataset']} · {r['bit_width']}-bit")
        ax.set_xlabel("k (returned)")
        ax.set_ylabel("R@1 within top-k")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for j in range(len(panels), nrows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("Recall — vanilla uint8-LUT scan vs augmented fp16 reconstruction", y=1.02)
    fig.tight_layout()
    return _save(fig, out, "recall")


def plot_memory(rows: list[dict], out: Path) -> list[str]:
    """Bytes/vector: vanilla compact vs augmented fp16 resident vs fp32 reference."""

    if not rows:
        return []
    labels = [f"d={r['dim']}\n{r['bit_width']}-bit" for r in rows]
    van = [r["vanilla_compact_bytes_per_vector"] for r in rows]
    aug = [r["augmented_fp16_resident_bytes_per_vector"] for r in rows]
    fp32 = [r["fp32_bytes_per_vector"] for r in rows]
    x = range(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(rows)), 4.6))
    ax.bar([i - w for i in x], van, w, label="vanilla compact (RAM)", color=VANILLA_C)
    ax.bar(list(x), aug, w, label="augmented fp16 (GPU resident)", color=AUG_C)
    ax.bar([i + w for i in x], fp32, w, label="fp32 reference", color=FP32_C)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("bytes / vector")
    ax.set_title("Memory per vector — compact codes vs fp16 GPU-resident")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(KM)
    for i, (v, a) in enumerate(zip(van, aug, strict=True)):
        ax.text(i - w, v, f"{v}", ha="center", va="bottom", fontsize=8)
        ax.text(i, a, f"{a}", ha="center", va="bottom", fontsize=8)
    return _save(fig, out, "memory")


def plot_update(rows: list[dict], out: Path) -> list[str]:
    """Persist cost vs corpus: vanilla full rewrite (O(N)) vs augmented delta (O(changed))."""

    if not rows:
        return []
    ns = [r["row_count"] for r in rows]
    full = [r["vanilla_full_write_ms"] for r in rows]
    delta = [r["augmented_delta_export_ms"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ns, full, "o-", color=VANILLA_C, label="vanilla full rewrite")
    ax.plot(
        ns, delta, "s-", color=AUG_C,
        label=f"augmented delta export ({rows[0]['update_count']} changed)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(KM)
    ax.set_xlabel("corpus size (vectors)")
    ax.set_ylabel("persist time (ms, log)")
    ax.set_title("Incremental update persist — full rewrite vs delta export")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    for n, f, d in zip(ns, full, delta, strict=True):
        if d > 0:
            ax.annotate(
                f"{f/d:.0f}×", (n, f), textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8, color=VANILLA_C,
            )
    return _save(fig, out, "update")


def _g(row: dict, series_key: str, field: str) -> float:
    """Safely reads a numeric field from a possibly-missing nested series."""

    series = row.get(series_key)
    if not isinstance(series, dict):
        return 0.0
    return float(series.get(field, 0.0) or 0.0)


def main(argv: list[str]) -> int:
    """Reads a results JSON and writes one PNG+SVG per axis to --out."""

    ap = argparse.ArgumentParser(description=__doc__)
    _here = Path(__file__).resolve().parent
    ap.add_argument("--results", default=str(_here / "results" / "results_a10.json"))
    ap.add_argument("--out", default=str(_here / "docs"))
    args = ap.parse_args(argv[1:])

    bundle = json.loads(Path(args.results).read_text())
    axes = bundle["axes"]
    out = Path(args.out)

    # Overlay the sibling L40S run on the throughput chart when it exists.
    results_path = Path(args.results)
    l40s_path = results_path.parent / "results_l40s.json"
    l40s_speed = None
    if l40s_path.exists() and l40s_path != results_path:
        l40s_speed = json.loads(l40s_path.read_text()).get("axes", {}).get("speed_batch", [])

    written: list[str] = []
    written += plot_speed_parity(axes.get("speed_parity", []), out)
    written += plot_speed_scaling(axes.get("speed_scaling", []), out)
    written += plot_speed_batch(axes.get("speed_batch", []), out, rows_l40s=l40s_speed)
    written += plot_recall(axes.get("recall", []), out)
    written += plot_memory(axes.get("memory", []), out)
    written += plot_update(axes.get("update", []), out)
    for p in written:
        print(f"[diagrams] wrote {p}")
    print(f"[diagrams] {len(written)} files from {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
