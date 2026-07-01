"""Public concurrent-append API for LodeDB.

`Appender` lets many processes durably log pre-embedded (vector-in) records to one
on-disk store's write-ahead log at once, folded into the index by the next writable
open. It is the multi-writer ingest counterpart to :meth:`LodeDB.add_vectors`: no
embedding, no chunking, vector plus metadata only. It requires WAL commit mode (the
default) and is exact about ids, since auto-generated ids would collide across
concurrent processes.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import Any

from lodedb.engine._atomic_io import normalize_durability
from lodedb.engine.core import EngineVectorDocument
from lodedb.engine.native_adapter import NativeCoreAdapter, NativeCoreAppenderHandle
from lodedb.local.db import _coerce_metadata, _coerce_optional_text

__all__ = ["Appender"]


class Appender:
    """A shared-lock appender over a persisted store's single index.

    Open one per process with :meth:`open`; each :meth:`append` / :meth:`append_many`
    / :meth:`delete` durably logs one WAL record and returns the log sequence number
    (LSN) assigned to it. Records become queryable only after the next *writable*
    ``LodeDB`` open folds the WAL in (a read-only handle still loads the last
    checkpointed generation). A single instance is not thread-safe; serialize calls
    to it. Use it as a context manager, or call :meth:`close`, to release the shared
    lock promptly.
    """

    def __init__(self, handle: NativeCoreAppenderHandle) -> None:
        self._handle: NativeCoreAppenderHandle | None = handle

    @classmethod
    def open(
        cls,
        path: str | PathLike[str],
        *,
        durability: str = "fast",
        store_text: bool = False,
        index_text: bool = False,
        acquire_writer_lock: bool = True,
    ) -> Appender:
        """Opens an appender over the store at ``path``.

        ``durability`` is ``"fast"`` (default, atomic but not fsynced per append) or
        ``"fsync"`` (each append fsynced before returning); any other value raises,
        matching :class:`LodeDB`. ``store_text``/``index_text`` control whether an
        appended document's ``text`` is retained and its lexical tokens logged. Both
        default **off** for privacy: no raw text reaches ``<key>.wal`` unless you opt
        in. Enable ``store_text`` only for a store whose writer also retains text
        (i.e. was opened ``store_text=True``), or the writer drops the caption at
        checkpoint. ``acquire_writer_lock`` takes the shared ``<dir>/.lodedb.lock`` so
        appenders exclude an exclusive writer; pass ``False`` only when an outer
        caller owns exclusion. The store must hold exactly one index and be in WAL
        commit mode.
        """

        # Validate/normalize up front (raises on an unknown mode) so a typo cannot
        # silently degrade to buffered instead of the requested fsync.
        native_durability = "fsync" if normalize_durability(durability) else "relaxed"
        adapter = NativeCoreAdapter()
        if not adapter.available:
            raise RuntimeError("native core extension is not available")
        handle = adapter.open_appender(
            path=path,
            durability=native_durability,
            store_text=store_text,
            index_text=index_text,
            acquire_writer_lock=acquire_writer_lock,
        )
        return cls(handle)

    def append(
        self,
        vector: Sequence[float],
        *,
        id: str,
        metadata: Mapping[str, Any] | None = None,
        text: str | None = None,
        normalize: bool = True,
    ) -> int:
        """Logs one vector-in record and returns its LSN.

        The vector is L2-normalized by default (matching :meth:`LodeDB.add_vectors`,
        so cosine scores stay comparable); pass ``normalize=False`` for a unit-norm
        vector. ``id`` is required: auto-ids would collide across concurrent writers.
        ``text`` is the optional caption (e.g. for an image), retained only when the
        appender was opened with ``store_text``; it is never embedded or chunked.
        """

        return self.append_many(
            [{"vector": vector, "id": id, "metadata": metadata, "text": text}],
            normalize=normalize,
        )

    def append_many(
        self,
        documents: Iterable[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> int:
        """Logs a batch of ``{"vector", "id", "metadata"?, "text"?}`` records as one
        WAL record and returns its LSN. Each ``id`` must be present and non-empty."""

        prepared: list[EngineVectorDocument] = []
        for item in documents:
            vector = item.get("vector")
            if vector is None:
                raise ValueError("each document needs a 'vector'")
            document_id = item.get("id")
            if document_id is None or not str(document_id).strip():
                raise ValueError("each document needs a non-empty 'id'")
            prepared.append(
                EngineVectorDocument(
                    document_id=str(document_id),
                    vector=_normalize_vector(vector, normalize=normalize),
                    metadata=_coerce_metadata(item.get("metadata")),
                    text=_coerce_optional_text(item.get("text")),
                )
            )
        if not prepared:
            raise ValueError("append_many requires at least one document")
        return self._require_open().append_vectors(prepared)

    def delete(self, ids: str | Sequence[str]) -> int:
        """Logs a delete as one WAL record and returns its LSN.

        Accepts a single id or a sequence of ids. A bare string is treated as one
        id (not iterated into characters).
        """

        document_ids = [ids] if isinstance(ids, str) else [str(document_id) for document_id in ids]
        if not document_ids:
            raise ValueError("delete requires at least one id")
        return self._require_open().append_deletes(document_ids)

    def close(self) -> None:
        """Releases the shared lock by dropping the native handle. Idempotent."""

        self._handle = None

    def __enter__(self) -> Appender:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _require_open(self) -> NativeCoreAppenderHandle:
        if self._handle is None:
            raise RuntimeError("appender is closed")
        return self._handle


def _normalize_vector(vector: Sequence[float], *, normalize: bool) -> list[float]:
    # Mirror LodeDB.add_vectors' _prepare_vector: compute in Python float (float64)
    # and validate finiteness, so a large-but-finite vector cannot overflow to inf
    # under a float32 norm and be stored as an all-zero (direction-losing) record.
    try:
        values = [float(component) for component in vector]
    except (TypeError, ValueError) as error:
        raise ValueError("vector must be a sequence of numbers") from error
    if not values:
        raise ValueError("vector must be non-empty")
    if not all(math.isfinite(component) for component in values):
        raise ValueError("vector must contain only finite values")
    if normalize:
        # math.hypot scales internally, so a large-but-finite component (whose square
        # would overflow float64) does not push the norm to inf and zero the vector.
        norm = math.hypot(*values)
        if norm == 0.0:
            raise ValueError(
                "cannot normalize a zero vector; pass normalize=False to store it as-is"
            )
        if not math.isfinite(norm):
            raise ValueError("vector norm overflows; scale the vector down before appending")
        values = [component / norm for component in values]
    return values
