"""Generate pinned persisted-store fixtures for native-core migration tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from benchmarks.native_migration.corpora import TEXT_DOCUMENTS, VECTOR_DOCUMENTS
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB

FIXTURE_SCHEMA_VERSION = 1


def generate(root: Path) -> None:
    """Rebuilds all persisted fixtures below ``root`` using current Python code."""

    root.mkdir(parents=True, exist_ok=True)
    _reset(root / "v0_4_generation")
    _seed_text_fixture(root / "v0_4_generation", commit_mode="generation")

    _reset(root / "v0_4_wal")
    _seed_text_fixture(root / "v0_4_wal", commit_mode="wal", checkpoint=False)

    _reset(root / "v0_4_store_text")
    _seed_vector_fixture(root / "v0_4_store_text", store_text=True)

    _reset(root / "v0_4_index_text")
    _seed_text_fixture(root / "v0_4_index_text", commit_mode="generation", index_text=True)

    _reset(root / "v0_4_legacy_top_level_json")
    _seed_legacy_fixture(root / "v0_4_legacy_top_level_json")
    for lock_path in root.glob("v0_4_*/.lodedb.lock"):
        lock_path.unlink(missing_ok=True)


def _reset(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _seed_text_fixture(
    path: Path,
    *,
    commit_mode: str,
    index_text: bool = False,
    checkpoint: bool = True,
) -> None:
    db = LodeDB(
        path,
        commit_mode=commit_mode,
        index_text=index_text,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        for document in TEXT_DOCUMENTS:
            db.add(document["text"], id=document["id"], metadata=document["metadata"])
        if checkpoint:
            db.persist()
        _write_manifest(path, mode="text", commit_mode=commit_mode, index_text=index_text)
    finally:
        if checkpoint:
            db.close()


def _seed_vector_fixture(path: Path, *, store_text: bool) -> None:
    db = LodeDB.open_vector_store(
        path,
        vector_dim=8,
        commit_mode="generation",
        store_text=store_text,
    )
    try:
        for document in VECTOR_DOCUMENTS:
            db.add_vectors(
                document["vector"],
                id=document["id"],
                metadata=document["metadata"],
                text=document["text"],
            )
        db.persist()
        _write_manifest(path, mode="vector", commit_mode="generation", store_text=store_text)
    finally:
        db.close()


def _seed_legacy_fixture(path: Path) -> None:
    db = LodeDB(
        path,
        commit_mode="generation",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        for document in TEXT_DOCUMENTS:
            db.add(document["text"], id=document["id"], metadata=document["metadata"])
        db.persist()
    finally:
        db.close()
    commit_manifests = list(path.glob("*.commit.json"))
    if len(commit_manifests) != 1:
        raise RuntimeError("expected one committed manifest to derive legacy fixture")
    body = json.loads(commit_manifests[0].read_text(encoding="utf-8"))["body"]
    base_epoch = int(body["base_epoch"])
    index_key = body["index_key"]
    source = path / f"{index_key}.gen" / f"g{base_epoch}.json"
    target = path / f"{index_key}.json"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    tvim_source = path / f"{index_key}.gen" / f"g{base_epoch}.tvim"
    shutil.copy2(tvim_source, path / f"{index_key}.tvim")
    shutil.rmtree(path / f"{index_key}.gen")
    commit_manifests[0].unlink()
    _write_manifest(path, mode="legacy_top_level_json", commit_mode="generation")


def _write_manifest(path: Path, **metadata: Any) -> None:
    payload = {
        "fixture_schema_version": FIXTURE_SCHEMA_VERSION,
        "lodedb_version": "0.4.0",
        "metadata": metadata,
    }
    (path / "fixture_manifest.txt").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent)
