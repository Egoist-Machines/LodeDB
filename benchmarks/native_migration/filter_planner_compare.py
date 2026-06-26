"""Compare Python and Rust metadata filter planners on pinned oracle fixtures."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lodedb.engine._filter_plan import build_field_indexes, resolve
from lodedb.engine._predicate import coerce_sdk_filter

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "native_core_predicate" / "predicate.json"


def _measure(label: str, iterations: int, func: Callable[[], int]) -> dict[str, Any]:
    start = time.perf_counter()
    checksum = 0
    for _ in range(iterations):
        checksum += func()
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return {"name": label, "iterations": iterations, "elapsed_ms": elapsed_ms, "checksum": checksum}


def _python_set_planner(iterations: int) -> dict[str, Any]:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    metadata_by_id = {document["id"]: document["metadata"] for document in fixture["documents"]}
    filters = [coerce_sdk_filter(case["filter"]) for case in fixture["cases"]]
    fields, all_docs = build_field_indexes(metadata_by_id)

    def run_once() -> int:
        return sum(len(resolve(filter_expr, fields, all_docs)) for filter_expr in filters)

    row = _measure("python_set_planner", iterations, run_once)
    row.update({"cases": len(filters), "documents": len(metadata_by_id)})
    return row


def _rust_planner(iterations: int) -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "cargo",
            "run",
            "--quiet",
            "-p",
            "lodedb-core",
            "--example",
            "filter_planner_bench",
            "--",
            "--iterations",
            str(iterations),
        ],
        cwd=ROOT,
        text=True,
    )
    return json.loads(output)


def run(output: Path | None = None, iterations: int = 10_000) -> dict[str, Any]:
    python_row = _python_set_planner(iterations)
    rust_row = _rust_planner(iterations)
    payload = {
        "suite": "native_migration_filter_planner_compare",
        "fixture": str(FIXTURE_PATH.relative_to(ROOT)),
        "measurements": [python_row, rust_row],
        "rust_to_python_elapsed_ratio": round(
            rust_row["elapsed_ms"] / python_row["elapsed_ms"], 3
        )
        if python_row["elapsed_ms"]
        else None,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=10_000)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.iterations), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
