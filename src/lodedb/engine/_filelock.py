"""Single-writer file lock for LodeDB persistence (stdlib only).

A LodeDB handle loads on-disk state at open and persists its own in-memory view,
so two *independent* writers on one directory cannot safely compose — interleaved
deltas leave the journal un-replayable. LodeDB is therefore **single-writer per
path**: a handle takes an exclusive OS advisory lock when it opens and holds it
until it closes. A second open blocks until the first closes (so it then loads
the accumulated state and composes cleanly) and fails fast with
:class:`ConcurrentWriterError` once the timeout elapses — the model SQLite uses
with a busy-timeout. The kernel releases the lock automatically on process exit,
so a crashed or forgotten handle never wedges the path.

POSIX uses ``fcntl.flock``; Windows uses ``msvcrt.locking``. The lock is taken on
a dedicated sentinel file ``<dir>/.lodedb.lock``, never the data files (which are
replaced via ``os.replace`` — a new inode that would drop a lock held on the old
one). Advisory locks are unreliable on network filesystems (NFS/SMB); LodeDB
targets local disk.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path

_LOCK_FILE_NAME = ".lodedb.lock"
DEFAULT_LOCK_TIMEOUT_S = 30.0
_POLL_INTERVAL_S = 0.05

_IS_WINDOWS = sys.platform == "win32"

if _IS_WINDOWS:  # pragma: no cover - exercised only on Windows CI
    import msvcrt
else:
    import fcntl


class ConcurrentWriterError(RuntimeError):
    """Raised when another process already holds the LodeDB writer lock."""


def lock_path_for(persistence_dir: str | Path) -> Path:
    """Returns the sentinel lock-file path for a persistence directory."""

    return Path(persistence_dir) / _LOCK_FILE_NAME


def lodedb_lock_timeout_from_env(env: Mapping[str, str] | None = None) -> float:
    """Returns the writer-lock acquire timeout (seconds) from the environment.

    ``LODEDB_PERSIST_LOCK_TIMEOUT`` overrides the default; unset uses
    :data:`DEFAULT_LOCK_TIMEOUT_S`. A non-positive or non-numeric value raises.
    """

    source = os.environ if env is None else env
    raw = source.get("LODEDB_PERSIST_LOCK_TIMEOUT")
    if raw is None or raw == "":
        return DEFAULT_LOCK_TIMEOUT_S
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("LODEDB_PERSIST_LOCK_TIMEOUT must be a number") from exc
    if value <= 0:
        raise ValueError("LODEDB_PERSIST_LOCK_TIMEOUT must be positive")
    return value


class WriterLock:
    """An exclusive, held-for-the-handle's-lifetime advisory lock on a directory.

    Not reentrant: a second :meth:`acquire` on the same path (this process or
    another) contends with the first and fails after the timeout — that is the
    single-writer guarantee. Acquire blocks (polling) until the lock is free or
    the timeout elapses; :meth:`release` is idempotent.
    """

    __slots__ = ("_lock_path", "_fd")

    def __init__(self, persistence_dir: str | Path) -> None:
        self._lock_path = lock_path_for(persistence_dir)
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def acquire(self, timeout: float = DEFAULT_LOCK_TIMEOUT_S) -> None:
        if self._fd is not None:
            return
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(os.fspath(self._lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + timeout
        while True:
            if _os_try_lock(fd):
                self._fd = fd
                return
            if time.monotonic() >= deadline:
                os.close(fd)
                raise ConcurrentWriterError(
                    f"LodeDB at {self._lock_path.parent} is already open by another "
                    "process (LodeDB is single-writer per path); close the other "
                    "handle, or raise LODEDB_PERSIST_LOCK_TIMEOUT to wait longer."
                )
            time.sleep(_POLL_INTERVAL_S)

    def release(self) -> None:
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            _os_unlock(fd)
        finally:
            os.close(fd)


def _os_try_lock(fd: int) -> bool:
    """Attempts a non-blocking exclusive OS lock; True on success, False if held."""

    if _IS_WINDOWS:  # pragma: no cover - Windows only
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _os_unlock(fd: int) -> None:
    if _IS_WINDOWS:  # pragma: no cover - Windows only
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    fcntl.flock(fd, fcntl.LOCK_UN)
