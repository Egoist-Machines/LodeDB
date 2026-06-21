"""In-process thread safety for a shared LodeDB handle.

``lodedb serve`` runs a ThreadingHTTPServer over one shared handle, so request
threads call add/search/remove on the same engine concurrently. The engine
serializes its public operations under a reentrant lock; without it those
threads race on shared dicts and the cached columnar index a query reads while a
mutation rebuilds it. These tests exercise that contention directly.
"""

from __future__ import annotations

import threading

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB


def _open(path) -> LodeDB:
    return LodeDB(
        path=path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )


def test_concurrent_threads_writing_one_handle_stay_consistent(tmp_path):
    """Many threads adding/searching one shared handle: no errors, exact count."""

    db = _open(tmp_path)
    n_threads, per_thread = 6, 15
    errors: list[BaseException] = []
    barrier = threading.Barrier(n_threads)

    def worker(worker_id: int) -> None:
        barrier.wait()  # maximize overlap
        try:
            for i in range(per_thread):
                db.add(f"worker {worker_id} item {i}", id=f"{worker_id}-{i}")
                db.search("item", k=3)  # interleave reads with the writes
        except BaseException as exc:  # noqa: BLE001 - capture for the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)

    assert not errors, errors
    assert db.count() == n_threads * per_thread
    db.close()

    # The persisted state survives the concurrent writes intact on reopen.
    reopened = _open(tmp_path)
    try:
        assert reopened.count() == n_threads * per_thread
    finally:
        reopened.close()


def test_concurrent_add_and_remove_threads_do_not_corrupt(tmp_path):
    """Interleaved adders and removers leave a consistent, reloadable store."""

    db = _open(tmp_path)
    seed = 40
    db.add_many([{"text": f"seed {i}", "id": f"s-{i}"} for i in range(seed)])
    errors: list[BaseException] = []

    def adder() -> None:
        try:
            for i in range(20):
                db.add(f"added {i}", id=f"a-{i}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def remover() -> None:
        try:
            for i in range(seed):
                db.remove(f"s-{i}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def searcher() -> None:
        try:
            for _ in range(40):
                db.search("seed", k=5)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=adder),
        threading.Thread(target=remover),
        threading.Thread(target=searcher),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)

    assert not errors, errors
    final = db.count()
    db.close()

    reopened = _open(tmp_path)
    try:
        assert reopened.count() == final  # persisted count matches the live count
    finally:
        reopened.close()
