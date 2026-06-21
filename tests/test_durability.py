"""Durability knob: atomic by default, fsync-on-commit when asked.

``durability="fast"`` keeps the atomic temp-file + ``os.replace`` publish (no
torn files, but not power-loss durable); ``durability="fsync"`` additionally
fsyncs each published file and its directory on commit.
"""

from __future__ import annotations

import os

import pytest

from lodedb.engine._atomic_io import durability_from_env, durable_replace, normalize_durability
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB

# Capture the real os.fsync once, before any test patches it.
_REAL_FSYNC = os.fsync


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


def _count_fsync_on_commit(path, monkeypatch, durability: str) -> int:
    """Opens a writer with the given durability, adds one doc, counts os.fsync calls."""

    calls = {"n": 0}

    def counting_fsync(fd):
        calls["n"] += 1
        _REAL_FSYNC(fd)

    monkeypatch.setattr(os, "fsync", counting_fsync)
    db = LodeDB(path=path, model="minilm", durability=durability, _embedding_backend=_be())
    db.add("durable doc", id="z")
    db.close()
    return calls["n"]


def test_fsync_mode_syncs_on_commit_fast_mode_does_not(tmp_path, monkeypatch):
    fast = _count_fsync_on_commit(tmp_path / "fast", monkeypatch, "fast")
    fsync = _count_fsync_on_commit(tmp_path / "fsync", monkeypatch, "fsync")
    assert fast == 0  # the fast path never fsyncs
    assert fsync > 0  # the fsync path fsyncs files + directories


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
