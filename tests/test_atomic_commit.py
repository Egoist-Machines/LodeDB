"""Stage 2: atomic multi-file commits over generation-addressed artifacts.

Each index commit writes its durable artifacts under ``<key>.gen/`` keyed by
base epoch and is sealed by an atomic swap of the ``<key>.commit.json`` root
manifest. These tests assert the two guarantees that buys: a crashed commit
recovers to the last good generation (not corrupted, not stuck), and a lock-free
reader always loads one consistent generation. They also cover migration from
the pre-commit-manifest layout and base-epoch GC.
"""

from __future__ import annotations

import shutil

import pytest

import lodedb.engine.core as core
from lodedb.engine._commit_manifest import (
    COMMIT_MANIFEST_SUFFIX,
    DEFAULT_EPOCHS_RETAINED,
    commit_manifest_path,
    generation_dir,
    list_base_epochs,
    read_commit_manifest,
)
from lodedb.engine.core import audit_persisted_index_snapshots
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB


def _be() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=384)


def _writer(path) -> LodeDB:
    return LodeDB(path=path, model="minilm", _embedding_backend=_be())


def _reader(path) -> LodeDB:
    return LodeDB.open_readonly(path, model="minilm", _embedding_backend=_be())


def _index_key(path) -> str:
    commits = list(path.glob(f"*{COMMIT_MANIFEST_SUFFIX}"))
    assert commits, "expected a root commit manifest"
    return commits[0].name[: -len(COMMIT_MANIFEST_SUFFIX)]


def _boom(*_args, **_kwargs):
    raise RuntimeError("injected crash at the commit point")


def test_commit_layout_and_round_trip(tmp_path):
    """A committed index uses the <key>.commit.json + <key>.gen/ layout and reloads."""

    writer = _writer(tmp_path)
    writer.add_many([{"text": f"doc number {i}", "id": f"d{i}"} for i in range(12)])
    writer.add("a late delta", id="late")  # delta append onto the live epoch
    assert writer.count() == 13
    writer.close()

    key = _index_key(tmp_path)
    body = read_commit_manifest(commit_manifest_path(tmp_path, key))
    assert body is not None and body["chunk_count"] >= 13
    assert generation_dir(tmp_path, key).is_dir()
    # No legacy top-level base leaks once committed under the new layout.
    assert not (tmp_path / f"{key}.json").exists()
    assert not (tmp_path / f"{key}.tvim").exists()

    reopened = _writer(tmp_path)
    try:
        assert reopened.count() == 13
        assert reopened.get("d3") == "doc number 3"
        assert reopened.get("late") == "a late delta"
    finally:
        reopened.close()


def test_delta_commit_crash_recovers_to_last_good(tmp_path, monkeypatch):
    """A crash at a delta commit's root swap rolls back to the last committed generation."""

    writer = _writer(tmp_path)
    writer.add_many([{"text": f"doc {i}", "id": f"d{i}"} for i in range(10)])
    assert writer.count() == 10
    # Crash exactly at the commit point of the next (delta) commit: the .jsd/.tvd
    # segments get written, but the root manifest swap never happens.
    monkeypatch.setattr(core, "write_commit_manifest", _boom)
    with pytest.raises(RuntimeError):
        writer.add("ghost", id="ghost")
    writer.close()  # release the writer lock, as a process exit would
    monkeypatch.undo()  # "restart" the process with a working commit path

    recovered = _writer(tmp_path)
    try:
        # The uncommitted delta is gone; the last good generation is intact.
        assert recovered.count() == 10
        assert recovered.get("ghost") is None
        assert recovered.get("d7") == "doc 7"
        audit_persisted_index_snapshots(tmp_path)  # no corruption / orphan leak
        # The store is healthy: a fresh commit lands and persists.
        recovered.add("real", id="real")
        assert recovered.count() == 11
    finally:
        recovered.close()

    again = _writer(tmp_path)
    try:
        assert again.count() == 11 and again.get("real") == "real"
    finally:
        again.close()


def test_base_rewrite_crash_recovers_to_last_good(tmp_path, monkeypatch):
    """A crash at a base-rewrite (compaction) commit also recovers to the last good gen.

    Generation-addressed bases mean the new epoch's files are written under new
    names, so the previous epoch stays fully intact for recovery.
    """

    writer = _writer(tmp_path)
    writer.add_many([{"text": f"doc {i}", "id": f"d{i}"} for i in range(10)])
    assert writer.count() == 10
    writer.close()

    writer = _writer(tmp_path)
    # Force the next commit to be a base rewrite (as compaction would), then crash
    # its root swap after the new-epoch base files are written.
    monkeypatch.setattr(
        "lodedb.engine.state_journal_store.StateJournalStore.should_compact",
        lambda self, **kwargs: True,
    )
    monkeypatch.setattr(
        "lodedb.engine.turbovec_delta_store.TvimDeltaStore.should_compact",
        lambda self, **kwargs: True,
    )
    monkeypatch.setattr(core, "write_commit_manifest", _boom)
    with pytest.raises(RuntimeError):
        writer.add("ghost", id="ghost")
    writer.close()
    monkeypatch.undo()

    recovered = _writer(tmp_path)
    try:
        assert recovered.count() == 10
        assert recovered.get("ghost") is None
        assert recovered.get("d4") == "doc 4"
        audit_persisted_index_snapshots(tmp_path)
    finally:
        recovered.close()


def test_reader_loads_consistent_generation_while_writer_open(tmp_path):
    """A lock-free reader loads one consistent committed generation."""

    writer = _writer(tmp_path)
    writer.add_many([{"text": f"doc {i}", "id": f"d{i}"} for i in range(8)])
    try:
        # Reader opens while the writer still holds the path and sees the
        # committed generation (8 docs).
        reader = _reader(tmp_path)
        try:
            assert reader.count() == 8
            assert reader.get("d2") == "doc 2"
        finally:
            reader.close()

        # A later committed delta becomes visible to a freshly opened reader.
        writer.add("nine", id="d8")
        later = _reader(tmp_path)
        try:
            assert later.count() == 9
            assert later.get("d8") == "nine"
        finally:
            later.close()
    finally:
        writer.close()


def test_migrates_from_legacy_layout(tmp_path):
    """A pre-commit-manifest store loads via fallback and migrates on its next write."""

    # Build a Stage-2 store with a single cold-build commit (base only, no deltas),
    # then rewrite it into the legacy top-level layout (no commit manifest).
    seed = _writer(tmp_path)
    seed.add_many([{"text": f"legacy doc {i}", "id": f"d{i}"} for i in range(6)])
    seed.close()
    key = _index_key(tmp_path)
    body = read_commit_manifest(commit_manifest_path(tmp_path, key))
    epoch = body["base_epoch"]
    gen = generation_dir(tmp_path, key)
    shutil.copy(gen / f"g{epoch}.json", tmp_path / f"{key}.json")
    shutil.copy(gen / f"g{epoch}.tvim", tmp_path / f"{key}.tvim")
    commit_manifest_path(tmp_path, key).unlink()
    shutil.rmtree(gen)
    assert not commit_manifest_path(tmp_path, key).exists()

    # Legacy fallback load works.
    migrated = _writer(tmp_path)
    try:
        assert migrated.count() == 6
        assert migrated.get("d1") == "legacy doc 1"
        # The next write migrates to the commit-manifest layout and removes the
        # legacy top-level base files.
        migrated.add("post-migration", id="new")
        assert migrated.count() == 7
    finally:
        migrated.close()

    assert commit_manifest_path(tmp_path, key).exists()
    assert generation_dir(tmp_path, key).is_dir()
    assert not (tmp_path / f"{key}.json").exists()
    assert not (tmp_path / f"{key}.tvim").exists()

    final = _writer(tmp_path)
    try:
        assert final.count() == 7 and final.get("new") == "post-migration"
    finally:
        final.close()


def test_gc_bounds_base_epochs(tmp_path, monkeypatch):
    """Repeated base rewrites GC superseded epochs, keeping only the most recent few."""

    writer = _writer(tmp_path)
    writer.add_many([{"text": f"doc {i}", "id": f"d{i}"} for i in range(5)])
    # Force every subsequent commit to be a base rewrite at a new epoch.
    monkeypatch.setattr(
        "lodedb.engine.state_journal_store.StateJournalStore.should_compact",
        lambda self, **kwargs: True,
    )
    monkeypatch.setattr(
        "lodedb.engine.turbovec_delta_store.TvimDeltaStore.should_compact",
        lambda self, **kwargs: True,
    )
    for i in range(6):
        writer.add(f"extra {i}", id=f"x{i}")
    writer.close()

    key = _index_key(tmp_path)
    epochs = list_base_epochs(tmp_path, key)
    assert 0 < len(epochs) <= DEFAULT_EPOCHS_RETAINED  # old epochs were collected

    reopened = _writer(tmp_path)
    try:
        assert reopened.count() == 11  # all data survives the GC
    finally:
        reopened.close()
