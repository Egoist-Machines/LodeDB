"""Render metrics-only wiki_dpr benchmark JSON files as a Markdown comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .lodedb_bench import validate_result_schema
except ImportError:  # Direct execution from this directory.
    from lodedb_bench import validate_result_schema  # type: ignore[no-redef]


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
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid result JSON: {path}") from exc
        if isinstance(result, dict) and "store" in result:
            validate_result_schema(result)
            rows.append(result)
    lines = [
        "| config | rows | queries | eval id | layout | rescore | effective np | effective ov | "
        "recall@100 | seq p50 ms | closed-loop QPS | concurrency | measured sec | batch-256 QPS |",
        "|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in rows:
        store = result["store"]
        serve = result.get("serve") or {}
        ann = store.get("ann")
        layout = str(store.get("layout", {}).get("id", "unknown"))
        if store.get("layout", {}).get("compacted"):
            layout += ", compacted"
        if ann:
            layout += f", ann{ann.get('clusters')}/np{ann.get('nprobe')}"
        rescore = store.get("rescore")
        rescore_cell = (
            "none" if not rescore else f"{rescore.get('dtype')} x{rescore.get('oversample')}"
        )
        sequential = serve.get("sequential_latency_ms", {})
        closed_loop = serve.get("closed_loop", {})
        dataset = result["dataset"]
        lines.append(
            "| {label} | {rows} | {queries} | {evaluation} | {layout} | {rescore} | "
            "{effective_nprobe} | {effective_oversample} | {recall} | {seq} | {cl} | "
            "{concurrency} | {seconds} | {batch} |".format(
                label=result.get("label", "unknown"),
                rows=dataset["rows"],
                queries=dataset["n_queries"],
                evaluation=(dataset["evaluation_id"] or "-")[:12],
                layout=layout,
                rescore=rescore_cell,
                effective_nprobe=_number(serve.get("effective_nprobe"), 0),
                effective_oversample=_number(serve.get("effective_oversample"), 1),
                recall=_number(serve.get("recall_at_100"), 4),
                seq=_number(sequential.get("p50_ms"), 3),
                cl=_number(closed_loop.get("qps"), 2),
                concurrency=_number(closed_loop.get("concurrency"), 0),
                seconds=_number(closed_loop.get("seconds"), 2),
                batch=_number(_batch_qps(result), 2),
            )
        )
    identities = {result["dataset"]["evaluation_id"] for result in rows}
    populations = {
        (result["dataset"]["rows"], result["dataset"]["n_queries"]) for result in rows
    }
    lines.append("")
    if len(identities) > 1 or len(populations) > 1:
        lines.append(
            "**Not a parity table:** rows span different corpus/query populations or evaluation "
            "identities. Compare only rows with the same eval id, row count, and query count."
        )
    else:
        lines.append(
            "All rows above share one corpus/query population and evaluation identity. External "
            "published systems are intentionally omitted unless rerun on that exact query set."
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
