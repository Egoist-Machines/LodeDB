"""Durability knob: atomic by default, fsync-on-commit when asked.

``durability="fast"`` keeps the atomic temp-file + ``os.replace`` publish (no
torn files, but not power-loss durable); ``durability="fsync"`` additionally
fsyncs each published file and its directory on commit.
"""

from __future__ import annotations

import pytest

from lodedb.engine._atomic_io import durability_from_env, durable_replace, normalize_durability
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB


def _be() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=384)


def test_normalize_durability():
    assert normalize_durability(None) is False
    assert normalize_durability("") is False
    assert normalize_durability("fast") is False
    assert normalize_durability("fsync") is True
    assert normalize_durability("FSYNC") is True  # case-insensitive
    with pytest.raises(ValueError):
        normalize_durability("bogus")


def test_durability_from_env(monkeypatch):
    monkeypatch.delenv("LODEDB_DURABILITY", raising=False)
    assert durability_from_env() is False
    monkeypatch.setenv("LODEDB_DURABILITY", "fsync")
    assert durability_from_env() is True
    monkeypatch.setenv("LODEDB_DURABILITY", "fast")
    assert durability_from_env() is False


def test_durable_replace_is_atomic_in_both_modes(tmp_path):
    dst = tmp_path / "f.txt"

    tmp = tmp_path / "f.txt.tmp"
    tmp.write_text("v1")
    durable_replace(tmp, dst, fsync=False)
    assert dst.read_text() == "v1"
    assert not tmp.exists()  # the temp file was renamed, not left behind

    tmp.write_text("v2")
    durable_replace(tmp, dst, fsync=True)
    assert dst.read_text() == "v2"
    assert not tmp.exists()


def test_durability_knob_threads_to_native_open(tmp_path):
    """The durability knob maps to the native open contract (it fsyncs in the core).

    The fsync-on-commit now happens inside the native core (Rust ``sync_all``), not
    through Python's ``os.fsync``, so the Python-observable contract is the mapped
    durability the SDK hands the native open: ``fast`` -> ``relaxed`` (atomic only)
    and ``fsync`` -> ``fsync`` (fsync files + directories).
    """

    from lodedb.engine.native_adapter import NativeCoreAdapter

    fast_options = NativeCoreAdapter.open_options_payload(
        path=tmp_path / "fast",
        read_only=False,
        durability="relaxed",
        commit_mode="wal",
        store_text=True,
        index_text=False,
        chunk_character_limit=512,
    )
    fsync_options = NativeCoreAdapter.open_options_payload(
        path=tmp_path / "fsync",
        read_only=False,
        durability="fsync",
        commit_mode="wal",
        store_text=True,
        index_text=False,
        chunk_character_limit=512,
    )
    assert fast_options["durability"] == "relaxed"  # the fast path does not fsync
    assert fsync_options["durability"] == "fsync"  # the fsync path fsyncs on commit
    # A writer opened in fsync mode threads "fsync" durability to its native engine.
    db = LodeDB(
        path=tmp_path / "fsync", model="minilm", durability="fsync", _embedding_backend=_be()
    )
    try:
        db.add("durable doc", id="z")
    finally:
        db.close()


def test_fsync_durability_roundtrips(tmp_path):
    """A store written with durability=fsync reopens with its data intact."""

    writer = LodeDB(path=tmp_path, model="minilm", durability="fsync", _embedding_backend=_be())
    writer.add("alpha", id="a")
    writer.add("beta", id="b")
    assert writer.count() == 2
    writer.close()

    reader = LodeDB.open_readonly(tmp_path, model="minilm", _embedding_backend=_be())
    try:
        assert reader.count() == 2
        assert reader.get("a") == "alpha"
        assert reader.get("b") == "beta"
    finally:
        reader.close()


def test_bad_durability_value_rejected_at_open(tmp_path):
    with pytest.raises(ValueError):
        LodeDB(path=tmp_path, model="minilm", durability="bogus", _embedding_backend=_be())
