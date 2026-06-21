"""Read-only LodeDB handles: single writer, many readers, per path.

A ``read_only=True`` handle takes no writer lock, so it can read a committed
snapshot while a writer holds the path; it serves search/get/stats and rejects
every mutating call with :class:`ReadOnlyError`.
"""

from __future__ import annotations

import pytest

from lodedb.engine.core import LodeEngine
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB, ReadOnlyError


def _be() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=384)


def _writer(path) -> LodeDB:
    return LodeDB(path=path, model="minilm", _embedding_backend=_be())


def _reader(path) -> LodeDB:
    return LodeDB(path=path, model="minilm", read_only=True, _embedding_backend=_be())


def test_read_only_reads_committed_state(tmp_path):
    """A read-only handle loads and serves the writer's committed snapshot."""

    writer = _writer(tmp_path)
    writer.add("the quick brown fox", id="x")
    writer.close()

    reader = _reader(tmp_path)
    try:
        assert reader.count() == 1
        assert reader.get("x") == "the quick brown fox"
        assert reader.search("fox", k=1)[0].id == "x"
    finally:
        reader.close()


def test_open_readonly_classmethod_matches_kwarg(tmp_path):
    """``open_readonly`` is sugar for ``read_only=True`` and reads the same."""

    writer = _writer(tmp_path)
    writer.add("hello", id="h")
    writer.close()

    reader = LodeDB.open_readonly(tmp_path, model="minilm", _embedding_backend=_be())
    try:
        assert reader.read_only is True
        assert reader.count() == 1
    finally:
        reader.close()


def test_read_only_rejects_every_mutation(tmp_path):
    """Mutating verbs raise ReadOnlyError before touching the engine."""

    writer = _writer(tmp_path)
    writer.add("a doc", id="a")
    writer.close()

    reader = _reader(tmp_path)
    try:
        with pytest.raises(ReadOnlyError):
            reader.add("new")
        with pytest.raises(ReadOnlyError):
            reader.add_many([{"text": "new"}])
        with pytest.raises(ReadOnlyError):
            reader.remove("a")
        # The store is untouched by the rejected mutations.
        assert reader.count() == 1
    finally:
        reader.close()


def test_read_only_open_does_not_block_live_writer(tmp_path):
    """A reader opens and reads while a writer still holds the single-writer lock.

    The pre-read-only behavior was that any second open blocked on the writer's
    exclusive lock and then failed with ConcurrentWriterError; a read-only open
    must take no lock and succeed immediately.
    """

    writer = _writer(tmp_path)
    writer.add("doc", id="d")  # persisted on add
    try:
        reader = _reader(tmp_path)  # would block/raise if it took the writer lock
        try:
            assert reader.count() == 1
            assert reader.get("d") == "doc"
        finally:
            reader.close()
    finally:
        writer.close()


def test_read_only_missing_path_raises(tmp_path):
    """Opening read-only on a non-existent path is a clear error, not a silent empty DB."""

    with pytest.raises(FileNotFoundError):
        LodeDB(
            path=tmp_path / "does-not-exist",
            model="minilm",
            read_only=True,
            _embedding_backend=_be(),
        )


def test_read_only_load_surfaces_corruption(tmp_path, monkeypatch):
    """A read-only open fails closed on a corrupt load (no silent empty index).

    Stage 2 reads the consistent generation named by the atomic commit manifest,
    so there is no torn-read retry that could mask a genuine load failure.
    """

    writer = _writer(tmp_path)
    writer.add("a", id="a")
    writer.close()

    def always_fail(self):
        raise RuntimeError("persistent corruption")

    monkeypatch.setattr(LodeEngine, "_load_persisted_indexes", always_fail)
    with pytest.raises(RuntimeError, match="persistent corruption"):
        LodeDB(path=tmp_path, model="minilm", read_only=True, _embedding_backend=_be())
