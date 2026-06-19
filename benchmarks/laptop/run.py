#!/usr/bin/env python3
"""Laptop benchmark: embedding throughput + CPU TurboVec scan latency, per device.

Runs the same metrics-only measurement behind ``lodedb benchmark`` for each available
embedding device (mps / cpu) and writes one combined results JSON. Metrics only —
counts, bytes, and latency; never document or query text.

    python benchmarks/laptop/run.py --docs 20000 --queries 200

Then render charts with ``benchmarks/laptop/diagrams.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from lodedb.local.backends import torch_mps_available
from lodedb.local.benchmark import run_local_benchmark


def _available_devices() -> list[str]:
    """Returns the embedding devices to benchmark, fastest-acceleration first."""

    devices: list[str] = []
    if torch_mps_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def main() -> None:
    """Runs the per-device benchmark and writes the combined results JSON."""

    parser = argparse.ArgumentParser(description="LodeDB laptop benchmark")
    parser.add_argument("--docs", type=int, default=20000)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--model", default="minilm")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parent / "results" / "laptop_m1.json"),
    )
    args = parser.parse_args()

    runs = []
    for device in _available_devices():
        print(
            f"[laptop-bench] device={device} model={args.model} "
            f"docs={args.docs} queries={args.queries}",
            file=sys.stderr,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runs.append(
                run_local_benchmark(
                    path=tmp,
                    model=args.model,
                    device=device,
                    doc_count=args.docs,
                    query_count=args.queries,
                )
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "machine": runs[0]["machine"],
                "model": args.model,
                "doc_count": args.docs,
                "query_count": args.queries,
                "runs": runs,
            },
            indent=2,
        )
    )
    print(f"[laptop-bench] wrote {out} ({len(runs)} device runs)", file=sys.stderr)


if __name__ == "__main__":
    main()
