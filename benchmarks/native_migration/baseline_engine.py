"""Import/open baseline for the native-core migration."""

from __future__ import annotations

import argparse
import importlib
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmarks.native_migration.corpora import TEXT_DOCUMENTS
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB


def _measure(label: str, func: Callable[[], Any]) -> dict[str, Any]:
    start = time.perf_counter()
    result = func()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {"name": label, "elapsed_ms": round(elapsed_ms, 3), "result": result}


def run(output: Path | None = None) -> dict[str, Any]:
    """Runs a lightweight engine-overhead baseline and optionally writes JSON."""

    rows: list[dict[str, Any]] = []
    rows.append(_measure("import_lodedb", lambda: importlib.import_module("lodedb").__version__))
    with tempfile.TemporaryDirectory(prefix="lodedb-native-engine-") as tmp:
        path = Path(tmp) / "store"
        rows.append(
            _measure(
                "open_empty_db",
                lambda: _open_close(path),
            )
        )
        rows.append(
            _measure(
                "seed_text_db",
                lambda: _seed_text(path),
            )
        )
        rows.append(
            _measure(
                "reopen_persisted_db",
                lambda: _open_count(path),
            )
        )
    payload = {"suite": "native_migration_engine", "measurements": rows}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _open_close(path: Path) -> int:
    db = LodeDB(path, _embedding_backend=HashEmbeddingBackend(native_dim=384))
    try:
        return db.count()
    finally:
        db.close()


def _seed_text(path: Path) -> int:
    db = LodeDB(path, _embedding_backend=HashEmbeddingBackend(native_dim=384))
    try:
        for document in TEXT_DOCUMENTS:
            db.add(document["text"], id=document["id"], metadata=document["metadata"])
        db.persist()
        return db.count()
    finally:
        db.close()


def _open_count(path: Path) -> int:
    db = LodeDB(path, _embedding_backend=HashEmbeddingBackend(native_dim=384))
    try:
        return db.count()
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
