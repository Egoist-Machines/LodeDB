"""Render metrics-only wiki_dpr benchmark JSON files as a Markdown comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _number(value: Any, digits: int = 3) -> str:
    """Formats an optional numeric field for a compact Markdown cell."""

    return "-" if value is None else f"{float(value):.{digits}f}"


def _batch_qps(result: dict[str, Any], batch_size: int = 256) -> Any:
    """Extracts one batched-QPS row when it was recorded."""

    serve = result.get("serve") or {}
    for row in serve.get("batched", []):
        if row.get("batch_size") == batch_size:
            return row.get("qps")
    return None


def render_report(results_dir: str | Path) -> str:
    """Reads ``*.json`` result files and returns a Markdown table plus parity footer."""

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*.json")):
        if path.name == "sweep_summary.json":
            continue
        try:
            result = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict) and "store" in result:
            rows.append(result)
    lines = [
        "| config | layout | rescore | recall@100 | seq p50 ms | CL-4 QPS | batch-256 QPS | "
        "blocks skipped | f |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in rows:
        store = result["store"]
        serve = result.get("serve") or {}
        ann = store.get("ann")
        layout = "compact" if store.get("layout", {}).get("compacted") else "base"
        if ann:
            layout += f", ann{ann.get('clusters')}/np{ann.get('nprobe')}"
        rescore = store.get("rescore")
        rescore_cell = (
            "none" if not rescore else f"{rescore.get('dtype')} x{rescore.get('oversample')}"
        )
        sequential = serve.get("sequential_latency_ms", {})
        closed_loop = serve.get("closed_loop", {})
        block_skip = serve.get("block_skip") or {}
        lines.append(
            "| {label} | {layout} | {rescore} | {recall} | {seq} | {cl} | {batch} | "
            "{skip} | {fraction} |".format(
                label=result.get("label", "unknown"),
                layout=layout,
                rescore=rescore_cell,
                recall=_number(serve.get("recall_at_100"), 4),
                seq=_number(sequential.get("p50_ms"), 3),
                cl=_number(closed_loop.get("qps"), 2),
                batch=_number(_batch_qps(result), 2),
                skip=_number(block_skip.get("fraction"), 4),
                fraction=_number(block_skip.get("candidate_fraction_f"), 4),
            )
        )
    lines.extend(
        [
            "",
            "Published per-node parity reference, not rerun:",
            "",
            "| system | aggregate QPS | per-node QPS | recall@100 | deployment | status |",
            "|---|---:|---:|---:|---|---|",
            "| Qdrant | 111.9 | 37.3 | 0.9596 | 3 x 4vCPU/16GB | published, not rerun |",
            "| Elastic DiskBBQ | 32.4 | 10.8 | 0.9600 | 3 x 7vCPU/26GB | published, not rerun |",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Writes or prints the Markdown report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    report = render_report(args.results)
    if args.out is None:
        print(report)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")


if __name__ == "__main__":
    main()
