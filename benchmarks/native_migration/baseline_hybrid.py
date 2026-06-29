"""Lexical and hybrid-search baseline for the native-core migration."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.native_migration.corpora import TEXT_DOCUMENTS, TEXT_QUERIES
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB


def _measure(label: str, func: Callable[[], Any]) -> dict[str, Any]:
    start = time.perf_counter()
    result = func()
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return {"name": label, "elapsed_ms": elapsed_ms, "result": result}


def run(output: Path | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lodedb-native-hybrid-") as tmp:
        db = LodeDB(
            Path(tmp) / "hybrid",
            index_text=True,
            _embedding_backend=HashEmbeddingBackend(native_dim=384),
        )
        try:
            for document in TEXT_DOCUMENTS:
                db.add(document["text"], id=document["id"], metadata=document["metadata"])
            rows = []
            for mode in ("lexical", "hybrid"):
                for query in TEXT_QUERIES:
                    rows.append(
                        _measure(
                            f"{mode}_{query}",
                            lambda mode=mode, query=query: [
                                hit.id for hit in db.search(query, k=3, mode=mode)
                            ],
                        )
                    )
        finally:
            db.close()
    payload = {"suite": "native_migration_hybrid", "measurements": rows}
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
