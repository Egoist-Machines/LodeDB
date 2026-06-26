"""Metadata filter baseline for the native-core migration."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.native_migration.corpora import FILTERS, VECTOR_DOCUMENTS, VECTOR_QUERIES
from lodedb.local.db import LodeDB


def _measure(label: str, func: Callable[[], Any]) -> dict[str, Any]:
    start = time.perf_counter()
    result = func()
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return {"name": label, "elapsed_ms": elapsed_ms, "result": result}


def run(output: Path | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lodedb-native-filters-") as tmp:
        path = Path(tmp) / "vectors"
        db = LodeDB.open_vector_store(path, vector_dim=8)
        try:
            for document in VECTOR_DOCUMENTS:
                db.add_vectors(
                    document["vector"],
                    id=document["id"],
                    metadata=document["metadata"],
                    text=document["text"],
                )
            rows = [
                _measure(
                    f"filter_{idx}",
                    lambda filter_expr=filter_expr: [
                        hit.id
                        for hit in db.search_by_vector(
                            VECTOR_QUERIES[0], k=3, filter=filter_expr
                        )
                    ],
                )
                for idx, filter_expr in enumerate(FILTERS)
            ]
        finally:
            db.close()
    payload = {"suite": "native_migration_filters", "measurements": rows}
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
