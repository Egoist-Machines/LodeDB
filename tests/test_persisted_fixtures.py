from __future__ import annotations

import shutil
from pathlib import Path

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "persisted"


def _copy_fixture(name: str, tmp_path: Path) -> Path:
    source = FIXTURE_ROOT / name
    target = tmp_path / name
    shutil.copytree(source, target)
    return target


def test_v0_4_generation_fixture_opens(tmp_path: Path) -> None:
    path = _copy_fixture("v0_4_generation", tmp_path)
    db = LodeDB(
        path,
        commit_mode="generation",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        assert db.count() == 3
        assert db.get("doc-alpha") is not None
    finally:
        db.close()


def test_v0_4_wal_fixture_replays(tmp_path: Path) -> None:
    path = _copy_fixture("v0_4_wal", tmp_path)
    db = LodeDB(path, commit_mode="wal", _embedding_backend=HashEmbeddingBackend(native_dim=384))
    try:
        assert db.count() == 3
        assert db.get("doc-beta") == "Beta incident report for serial AX-42 on 2024-06-13."
    finally:
        db.close()


def test_v0_4_store_text_fixture_opens(tmp_path: Path) -> None:
    path = _copy_fixture("v0_4_store_text", tmp_path)
    db = LodeDB.open_vector_store(path, vector_dim=8, commit_mode="generation", store_text=True)
    try:
        assert db.count() == 3
        assert db.get("vec-alpha") == "Vector alpha retained payload."
    finally:
        db.close()


def test_v0_4_index_text_fixture_supports_lexical_after_reopen(tmp_path: Path) -> None:
    path = _copy_fixture("v0_4_index_text", tmp_path)
    db = LodeDB(
        path,
        commit_mode="generation",
        index_text=True,
        store_text=True,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        assert db.count() == 3
        assert [hit.id for hit in db.search("AX-42", k=1, mode="lexical")] == ["doc-beta"]
    finally:
        db.close()


def test_v0_4_legacy_top_level_json_fixture_opens(tmp_path: Path) -> None:
    path = _copy_fixture("v0_4_legacy_top_level_json", tmp_path)
    db = LodeDB(
        path,
        commit_mode="generation",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        assert db.count() == 3
    finally:
        db.close()


def test_rust_generation_fixture_opens_in_python(tmp_path: Path) -> None:
    path = _copy_fixture("rust_generation_empty", tmp_path)
    db = LodeDB(
        path,
        commit_mode="generation",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        assert db.count() == 0
    finally:
        db.close()
