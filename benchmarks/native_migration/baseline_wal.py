"""WAL append/replay baseline for the native-core migration."""

from __future__ import annotations

import argparse
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
    elapsed_ms = round((time.perf_counter() - start) * 1000.0, 3)
    return {"name": label, "elapsed_ms": elapsed_ms, "result": result}


def run(output: Path | None = None) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="lodedb-native-wal-") as tmp:
        path = Path(tmp) / "wal"
        rows = [
            _measure("wal_append_no_fsync", lambda: _seed(path, durability="fast")),
            _measure("wal_reopen", lambda: _open_count(path)),
        ]
        fsync_path = Path(tmp) / "wal-fsync"
        rows.append(_measure("wal_append_fsync", lambda: _seed(fsync_path, durability="fsync")))
    payload = {"suite": "native_migration_wal", "measurements": rows}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _seed(path: Path, *, durability: str) -> int:
    db = LodeDB(
        path,
        commit_mode="wal",
        durability=durability,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        for document in TEXT_DOCUMENTS:
            db.add(document["text"], id=document["id"], metadata=document["metadata"])
        return db.count()
    finally:
        db.close()


def _open_count(path: Path) -> int:
    db = LodeDB(path, commit_mode="wal", _embedding_backend=HashEmbeddingBackend(native_dim=384))
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
