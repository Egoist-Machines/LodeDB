"""Persistence and WAL baseline for the native-core migration."""

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
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="lodedb-native-storage-") as tmp:
        root = Path(tmp)
        rows.append(_measure("wal_append_text_batch", lambda: _seed(root / "wal", "wal")))
        rows.append(
            _measure(
                "generation_commit_text_batch",
                lambda: _seed(root / "gen", "generation"),
            )
        )
        rows.append(
            _measure(
                "wal_reopen_replay_or_checkpoint",
                lambda: _open_count(root / "wal", "wal"),
            )
        )
        rows.append(_measure("generation_reopen", lambda: _open_count(root / "gen", "generation")))
    payload = {"suite": "native_migration_storage", "measurements": rows}
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _seed(path: Path, commit_mode: str) -> int:
    db = LodeDB(
        path,
        commit_mode=commit_mode,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        for document in TEXT_DOCUMENTS:
            db.add(document["text"], id=document["id"], metadata=document["metadata"])
        db.persist()
        return db.count()
    finally:
        db.close()


def _open_count(path: Path, commit_mode: str) -> int:
    db = LodeDB(
        path,
        commit_mode=commit_mode,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
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
