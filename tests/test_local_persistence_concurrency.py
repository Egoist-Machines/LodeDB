"""Single-writer concurrency safety for LodeDB persistence.

LodeDB is single-writer per path: a handle holds an exclusive lock from open to
close. Concurrent opens serialize (each waits for the previous to close, then
loads the accumulated state and composes); an open fails fast with
``ConcurrentWriterError`` once the timeout elapses. The store is never corrupted.
"""

from __future__ import annotations

import multiprocessing as mp
import os

from lodedb.engine._filelock import ConcurrentWriterError
from lodedb.engine.core import audit_persisted_index_snapshots
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB

_N_WRITERS = 5
_DOCS_EACH = 12


def _open(path):
    return LodeDB(
        path=path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )


def test_sequential_writers_compose_and_lock_releases(tmp_path):
    """close() releases the lock; a later handle loads the accumulated state."""

    first = _open(tmp_path)
    first.add("first doc", id="a-0")
    first.close()  # releases the writer lock

    second = _open(tmp_path)  # loads the first writer's state
    second.add("second doc", id="b-0")
    assert second.count() == 2
    second.close()

    reopened = _open(tmp_path)
    try:
        assert reopened.count() == 2
    finally:
        reopened.close()


def _serial_writer(path_str, idx, count):
    # Blocking acquire: waits for any prior writer to close, then composes.
    db = _open(path_str)
    for i in range(count):
        db.add(f"doc {idx} number {i}", id=f"{idx}-{i}")
    db.close()


def test_concurrent_opens_serialize_and_compose(tmp_path):
    """Many processes opening one path serialize on the lock and all writes land."""

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_serial_writer, args=(str(tmp_path), k, _DOCS_EACH))
        for k in range(_N_WRITERS)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(120)

    assert [proc.exitcode for proc in procs] == [0] * _N_WRITERS
    audit_persisted_index_snapshots(tmp_path)  # no corruption / orphaned segments

    db = _open(tmp_path)
    try:
        assert db.count() == _N_WRITERS * _DOCS_EACH
    finally:
        db.close()


def _fail_fast_writer(path_str, result):
    os.environ["LODEDB_PERSIST_LOCK_TIMEOUT"] = "0.3"
    try:
        db = _open(path_str)
        db.close()
        result.value = 0  # unexpectedly acquired the lock
    except ConcurrentWriterError:
        result.value = 42  # expected fail-fast


def test_second_writer_fails_fast_while_first_is_open(tmp_path):
    """While one handle is open, another process fails fast after its timeout."""

    holder = _open(tmp_path)  # holds the lock for its lifetime
    try:
        ctx = mp.get_context("spawn")
        result = ctx.Value("i", -1)
        proc = ctx.Process(target=_fail_fast_writer, args=(str(tmp_path), result))
        proc.start()
        proc.join(30)
        assert proc.exitcode == 0
        assert result.value == 42  # the child raised ConcurrentWriterError
    finally:
        holder.close()
