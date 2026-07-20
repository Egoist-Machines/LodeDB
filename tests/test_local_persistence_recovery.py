"""Persistence crash-recovery / fail-closed tests for LodeDB.

LodeDB reuses the engine's ``.tvim``/``.tvd``/``.jsd`` persistence, which is
documented to **fail closed** on restart when a sidecar is missing or corrupt for
a non-empty index, rather than silently returning an empty or partial index.
These tests corrupt/remove the on-disk sidecars and assert that reopen raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB


def _seed(path, *, commit_mode: str | None = None) -> None:
    """Creates a 2-document on-disk index and closes it."""

    db = LodeDB(
        path=path,
        model="minilm",
        commit_mode=commit_mode,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    db.add("alpha document", id="a", metadata={"k": "a"})
    db.add("beta document", id="b", metadata={"k": "b"})
    db.persist()
    assert db.count() == 2
    db.close()


def _reopen(path, *, commit_mode: str | None = None) -> LodeDB:
    """Reopens the on-disk index with a matching hash backend."""

    return LodeDB(
        path=path,
        model="minilm",
        commit_mode=commit_mode,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )


def test_clean_reopen_preserves_data(tmp_path):
    """The control: an untouched on-disk index reloads all documents."""

    _seed(tmp_path)
    db = _reopen(tmp_path)
    assert db.count() == 2
    db.close()


def test_corrupt_tvim_base_fails_closed(tmp_path):
    """A garbled .tvim base raises on reopen instead of silently losing data."""

    _seed(tmp_path)
    # The committed vector base lives under the per-index <key>.gen/ directory.
    bases = [p for p in Path(tmp_path).glob("**/*.tvim") if p.is_file()]
    assert bases, "expected a .tvim base sidecar"
    bases[0].write_bytes(b"corrupt-not-a-real-tvim-sidecar")
    with pytest.raises(RuntimeError):
        _reopen(tmp_path)


def test_missing_tvim_base_fails_closed(tmp_path):
    """A missing .tvim base for a non-empty index raises on reopen."""

    _seed(tmp_path)
    bases = [p for p in Path(tmp_path).glob("**/*.tvim") if p.is_file()]
    bases[0].unlink()
    with pytest.raises(RuntimeError):
        _reopen(tmp_path)


def test_corrupt_jsd_delta_fails_closed(tmp_path):
    """A garbled .jsd journal delta raises on reopen."""

    # O(changed) .jsd journal deltas are written by the generation commit path;
    # the WAL default buffers writes and folds them into a base at checkpoint.
    _seed(tmp_path, commit_mode="generation")
    jsds = list(Path(tmp_path).glob("**/*.jsd"))
    assert jsds, "expected a .jsd journal delta"
    jsds[0].write_bytes(b"\x00\x01 not valid journal lines \xff")
    with pytest.raises(RuntimeError):
        _reopen(tmp_path, commit_mode="generation")
