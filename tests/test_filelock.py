"""Tests for the single-writer file lock (lodedb.engine._filelock)."""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest

from lodedb.engine._filelock import (
    DEFAULT_LOCK_TIMEOUT_S,
    ConcurrentWriterError,
    WriterLock,
    _os_try_lock,
    _os_unlock,
    lock_path_for,
    lodedb_lock_timeout_from_env,
)


def test_lock_path_for(tmp_path):
    assert lock_path_for(tmp_path) == tmp_path / ".lodedb.lock"


def test_timeout_from_env():
    assert lodedb_lock_timeout_from_env({}) == DEFAULT_LOCK_TIMEOUT_S
    assert lodedb_lock_timeout_from_env({"LODEDB_PERSIST_LOCK_TIMEOUT": "1.5"}) == 1.5
    with pytest.raises(ValueError):
        lodedb_lock_timeout_from_env({"LODEDB_PERSIST_LOCK_TIMEOUT": "nope"})
    with pytest.raises(ValueError):
        lodedb_lock_timeout_from_env({"LODEDB_PERSIST_LOCK_TIMEOUT": "0"})


def test_acquire_release_reacquire(tmp_path):
    lock = WriterLock(tmp_path)
    assert not lock.held
    lock.acquire(2)
    assert lock.held
    lock.release()
    assert not lock.held
    lock.release()  # idempotent
    lock.acquire(2)  # re-acquirable after release
    lock.release()


def test_os_primitive_exclusion(tmp_path):
    """Two independent open descriptions cannot both hold the exclusive lock."""

    lock_file = lock_path_for(tmp_path)
    fd1 = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o644)
    fd2 = os.open(lock_file, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        assert _os_try_lock(fd1) is True
        assert _os_try_lock(fd2) is False
        _os_unlock(fd1)
        assert _os_try_lock(fd2) is True
        _os_unlock(fd2)
    finally:
        os.close(fd1)
        os.close(fd2)


def test_second_handle_same_path_in_process_fails_fast(tmp_path):
    """A second writer lock on the same path (even in-process) is excluded."""

    first = WriterLock(tmp_path)
    first.acquire(2)
    second = WriterLock(tmp_path)
    try:
        with pytest.raises(ConcurrentWriterError):
            second.acquire(0.3)
        first.release()
        second.acquire(2)  # free now
        second.release()
    finally:
        first.release()


def _hold_writer_lock(dir_str, acquired, release):
    lock = WriterLock(dir_str)
    lock.acquire(10)
    acquired.set()
    release.wait(10)
    lock.release()


def test_cross_process_exclusion_and_release(tmp_path):
    """A lock held by another process blocks acquisition until it is released."""

    ctx = mp.get_context("spawn")
    acquired = ctx.Event()
    release = ctx.Event()
    proc = ctx.Process(target=_hold_writer_lock, args=(str(tmp_path), acquired, release))
    proc.start()
    try:
        assert acquired.wait(10), "child failed to acquire the lock"
        blocked = WriterLock(tmp_path)
        with pytest.raises(ConcurrentWriterError):
            blocked.acquire(0.3)
        release.set()
        proc.join(10)
        assert proc.exitcode == 0
        freed = WriterLock(tmp_path)
        freed.acquire(5)  # released by the child
        freed.release()
    finally:
        release.set()
        if proc.is_alive():
            proc.terminate()
        proc.join(5)
