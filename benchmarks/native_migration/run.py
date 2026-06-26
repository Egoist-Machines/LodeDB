"""Runs all native-core migration baseline suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.native_migration import (
    baseline_engine,
    baseline_filters,
    baseline_hybrid,
    baseline_storage,
    baseline_wal,
)

SUITES = (
    baseline_engine,
    baseline_storage,
    baseline_hybrid,
    baseline_filters,
    baseline_wal,
)


def run(output: Path | None = None) -> dict[str, Any]:
    payload = {"suite": "native_migration", "results": [suite.run() for suite in SUITES]}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
