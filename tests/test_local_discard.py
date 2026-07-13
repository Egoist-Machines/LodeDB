"""``LodeDB.discard()``: close-without-persist, the fold abort path.

The contrast with ``close()``: a graceful close persists a writable handle's
un-persisted in-memory state; ``discard()`` drops that state, leaves the store
at its last committed generation, and releases the writer lock immediately.
The partial-fold abort itself is covered in ``test_segments.py``
(``test_discard_abandons_a_partially_applied_fold``); these tests pin the
general handle semantics.
"""

from __future__ import annotations

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB

DIM = 384


def _be() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=DIM)


def _open(path, **kwargs) -> LodeDB:
    """Opens a writable store with the deterministic hash embedding backend."""

    return LodeDB(path=path, model="minilm", _embedding_backend=_be(), **kwargs)


def test_discard_keeps_wal_logged_writes(tmp_path):
    """WAL-mode writes are durable at write time, so discard() loses nothing:
    the un-checkpointed WAL tail replays on the next writable open."""

    db = _open(tmp_path, commit_mode="wal")
    db.add("wal logged before discard", id="a")
    db.discard()

    reopened = _open(tmp_path, commit_mode="wal")
    try:
        assert reopened.count() == 1
        assert reopened.get("a") == "wal logged before discard"
    finally:
        reopened.close()


def test_discard_is_idempotent_and_a_later_close_is_safe(tmp_path):
    """Double discard and close-after-discard are no-ops, and the writer lock
    is released by the first discard (the reopen would fail otherwise)."""

    db = _open(tmp_path, commit_mode="generation")
    db.add("committed by the generation-mode write", id="a")
    db.discard()
    db.discard()
    db.close()

    reopened = _open(tmp_path, commit_mode="generation")
    try:
        # Generation mode commits each mutation durably at write time; discard
        # only drops *un-persisted* state (the fold case in test_segments.py).
        assert reopened.count() == 1
    finally:
        reopened.close()


def test_discard_read_only_handle(tmp_path):
    """Read-only handles hold no lock and never persist: discard == close."""

    _open(tmp_path, commit_mode="generation").close()
    reader = LodeDB.open_readonly(tmp_path, model="minilm", _embedding_backend=_be())
    reader.discard()
