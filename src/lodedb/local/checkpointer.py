"""Public running-checkpointer API for LodeDB.

`Checkpointer` folds the write-ahead log that concurrent :class:`Appender` processes
log into fresh committed generations, continuously, without an application ever
re-opening a writable :class:`LodeDB`. One process holds a crash-reclaimable lease
and drives :meth:`checkpoint` on a loop or timer; appended records then become
durable and queryable (through a read-only handle's ``refresh``) shortly after they
are logged. It is the counterpart to the exclusive writer that used to be the only
thing that could fold the WAL. It requires WAL commit mode (the default).
"""

from __future__ import annotations

from os import PathLike

from lodedb.engine._atomic_io import normalize_durability
from lodedb.engine._filelock import ConcurrentWriterError
from lodedb.engine.native_adapter import NativeCoreAdapter, NativeCoreCheckpointerHandle
from lodedb.local.db import _is_writer_lock_contention

__all__ = ["Checkpointer"]

# The core's lease-contention message (distinct from the writer-lock message, since
# the lease is a separate sentinel): a second checkpointer on the same store.
_CHECKPOINTER_LEASE_CONTENTION_MARKER = "lodedb checkpointer lease"


class Checkpointer:
    """A running single-checkpointer over a persisted store's single index.

    Open one with :meth:`open`; each :meth:`checkpoint` folds the WAL that concurrent
    appenders logged into a fresh committed generation and returns how many records it
    folded (``0`` when nothing new was appended). Unlike a writable ``LodeDB``, it does
    not hold the writer lock for its lifetime: it holds only a lease and takes the
    exclusive writer lock for the brief window of each fold, so appenders keep logging
    between folds. Drive :meth:`checkpoint` on a loop or timer to keep a store
    continuously current. A single instance is not thread-safe (it is thread-confined
    to the thread that opened it); serialize calls to it. Use it as a context manager,
    or call :meth:`close`, to release the lease promptly.

    One process at a time holds the lease: a second :meth:`open` waits for the first to
    close (up to ``LODEDB_PERSIST_LOCK_TIMEOUT``), then raises
    :class:`ConcurrentWriterError`. A dead holder's lease is reclaimable (the OS
    releases it on death), so a fresh checkpointer can take over after a crash.
    """

    def __init__(
        self,
        handle: NativeCoreCheckpointerHandle,
        path: str | PathLike[str] = "",
    ) -> None:
        self._handle: NativeCoreCheckpointerHandle | None = handle
        self._path = path

    @classmethod
    def open(
        cls,
        path: str | PathLike[str],
        *,
        durability: str = "fast",
        store_text: bool = False,
        index_text: bool = False,
        chunk_character_limit: int = 900,
    ) -> Checkpointer:
        """Opens a checkpointer over the store at ``path``, acquiring the lease.

        ``durability`` is ``"fast"`` (default; each folded generation is atomic but not
        fsynced) or ``"fsync"`` (each fold fsynced before returning); any other value
        raises, matching :class:`LodeDB`. ``store_text``/``index_text``/
        ``chunk_character_limit`` mirror the store's writer exactly as for an
        :class:`Appender`: the fold retains a document's text only under ``store_text``
        and its lexical tokens only under ``index_text``, and re-tokenizes at
        ``chunk_character_limit``. Open the checkpointer with the same retention the
        store's writer uses, or the fold rewrites the store to the checkpointer's
        policy (dropping retained text/tokens on a mismatch). The store must hold
        exactly one index and be in WAL commit mode.
        """

        # Validate/normalize up front (raises on an unknown mode), matching LodeDB.
        native_durability = "fsync" if normalize_durability(durability) else "relaxed"
        adapter = NativeCoreAdapter()
        if not adapter.available:
            raise RuntimeError("native core extension is not available")
        try:
            handle = adapter.open_checkpointer(
                path=path,
                durability=native_durability,
                store_text=store_text,
                index_text=index_text,
                chunk_character_limit=chunk_character_limit,
            )
        except ValueError as exc:
            # `open` acquires the lease AND briefly takes the writer lock (to fold on
            # open); either can contend. Surface both as the SDK's stable contention
            # error so one retry loop covers a rival checkpointer and an active writer.
            if _is_checkpointer_contention(exc):
                raise ConcurrentWriterError(
                    f"checkpointer at {path} could not start (another checkpointer holds "
                    "the lease, or a writer holds the lodedb lock); close the other "
                    "handle, or raise LODEDB_PERSIST_LOCK_TIMEOUT to wait longer."
                ) from exc
            raise
        return cls(handle, path)

    def checkpoint(self) -> int:
        """Folds the appended WAL tail into a fresh committed generation.

        Returns the number of records folded (``0`` when nothing new was appended).
        Takes the exclusive writer lock only for this fold. If a concurrent writer or
        another checkpointer holds it past ``LODEDB_PERSIST_LOCK_TIMEOUT``, raises
        :class:`ConcurrentWriterError`; retry the fold.
        """

        try:
            return self._require_open().checkpoint()
        except ValueError as exc:
            if _is_writer_lock_contention(exc):
                raise ConcurrentWriterError(
                    f"checkpointer at {self._path} could not take the lodedb lock to fold "
                    "(held by an exclusive writer); retry, or raise "
                    "LODEDB_PERSIST_LOCK_TIMEOUT to wait longer."
                ) from exc
            raise

    def close(self) -> None:
        """Releases the lease by dropping the native handle. Idempotent."""

        if self._handle is not None:
            self._handle.close()
        self._handle = None

    def __enter__(self) -> Checkpointer:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_open(self) -> NativeCoreCheckpointerHandle:
        if self._handle is None:
            raise RuntimeError("checkpointer is closed")
        return self._handle


def _is_checkpointer_contention(error: BaseException) -> bool:
    """Whether an open failed on the checkpointer lease or the writer lock."""

    return (
        _CHECKPOINTER_LEASE_CONTENTION_MARKER in str(error)
        or _is_writer_lock_contention(error)
    )
