from __future__ import annotations

import shutil
from pathlib import Path

import pytest

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


def test_v0_4_text_wal_fixture_fails_closed(tmp_path: Path) -> None:
    """A leftover text-ingest WAL from a pre-native writer fails closed on open.

    The fixture's WAL holds ``upsert_documents`` (raw text + metadata) records,
    which only re-embedding could replay; the native-only core cannot re-embed
    during recovery, so it leaves the WAL untouched and fails the open rather than
    silently dropping the logged writes. A clean store (committed generation with
    no leftover text WAL) opens normally; this is the pre-native crash boundary.
    """

    path = _copy_fixture("v0_4_wal", tmp_path)
    with pytest.raises(RuntimeError, match="WAL containing non-native records"):
        LodeDB(path, commit_mode="wal", _embedding_backend=HashEmbeddingBackend(native_dim=384))


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


def test_text_wal_fixture_fails_closed_in_python(tmp_path: Path) -> None:
    """A leftover text-ingest WAL fails closed (no re-embed during native recovery).

    The fixture's WAL holds an ``upsert_documents`` (raw text) record. Replaying it
    would require re-embedding, which the native-only recovery path does not do, so
    the open fails closed and leaves the WAL on disk instead of losing the write.
    """

    path = _copy_fixture("rust_wal", tmp_path)
    with pytest.raises(RuntimeError, match="WAL containing non-native records"):
        LodeDB(path, commit_mode="wal", _embedding_backend=HashEmbeddingBackend(native_dim=384))
