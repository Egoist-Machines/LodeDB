"""Stage 2: atomic multi-file commits over generation-addressed artifacts.

Each index commit writes its durable artifacts under ``<key>.gen/`` keyed by
base epoch and is sealed by an atomic swap of the ``<key>.commit.json`` root
manifest. These tests assert the committed layout round-trips, a lock-free reader
always loads one consistent generation, and base-epoch GC bounds the retained
epochs.
"""

from __future__ import annotations

import gc

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
    # This suite exercises the generation commit subsystem (per-mutation atomic
    # generation publish, O(changed) deltas, MVCC reader snapshots), so it pins the
    # opt-out generation mode rather than the WAL default.
    return LodeDB(path=path, model="minilm", commit_mode="generation", _embedding_backend=_be())


def _reader(path) -> LodeDB:
    return LodeDB.open_readonly(path, model="minilm", _embedding_backend=_be())


def _index_key(path) -> str:
    commits = list(path.glob(f"*{COMMIT_MANIFEST_SUFFIX}"))
    assert commits, "expected a root commit manifest"
    return commits[0].name[: -len(COMMIT_MANIFEST_SUFFIX)]


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


def test_generation_commits_survive_unclean_shutdown(tmp_path):
    """Each generation-mode commit is durable on its own across an unclean shutdown.

    In generation mode every mutation publishes a crash-atomic generation, so a
    handle dropped without ``close()`` (no checkpoint, as a crash would leave it)
    reopens on the last committed generation with every acknowledged write intact,
    and the auditor reports that same committed count with no orphan leak.
    """

    writer = _writer(tmp_path)
    writer.add_many([{"text": f"doc {i}", "id": f"d{i}"} for i in range(6)])
    writer.add("a late delta", id="late")
    assert writer.count() == 7
    # Simulate a crash: drop the handle WITHOUT close(), leaving no checkpoint
    # behind as a real crash would. The native engine (and the single-writer lock
    # it holds) is released when its weakref finalizer drops it on its worker, so
    # the reopen below can reacquire the lock. Each add already committed its own
    # generation, so nothing is lost.
    del writer
    gc.collect()

    report = audit_persisted_index_snapshots(tmp_path)  # no corruption / orphan leak
    assert report["snapshot_count"] == 1
    assert report["snapshot_files"][0]["document_count"] == 7

    recovered = _writer(tmp_path)
    try:
        assert recovered.count() == 7
        assert recovered.get("d3") == "doc 3"
        assert recovered.get("late") == "a late delta"
    finally:
        recovered.close()


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
