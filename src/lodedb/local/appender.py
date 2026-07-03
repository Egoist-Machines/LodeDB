"""Public concurrent-append API for LodeDB.

`Appender` lets many processes durably log records to one on-disk store's
write-ahead log at once, folded into the index by the next writable open. The
vector-in :meth:`append` / :meth:`append_many` are the multi-writer counterpart to
:meth:`LodeDB.add_vectors` (no embedding, no chunking, vector plus metadata only);
with an ``embedder`` configured at :meth:`open`, :meth:`append_text` /
:meth:`append_text_many` are the counterpart to :meth:`LodeDB.add` (chunk, embed,
then log a post-embedding record). It requires WAL commit mode (the default) and is
exact about ids, since auto-generated ids would collide across concurrent processes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import Any

from lodedb.engine._atomic_io import normalize_durability
from lodedb.engine._filelock import ConcurrentWriterError
from lodedb.engine.core import EngineDocument, EngineVectorDocument
from lodedb.engine.embedding_backends import EngineEmbeddingBackend
from lodedb.engine.native_adapter import NativeCoreAdapter, NativeCoreAppenderHandle
from lodedb.local.db import (
    _coerce_metadata,
    _coerce_optional_text,
    _finite_float_vector,
    _is_writer_lock_contention,
)

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

    On Windows the shared lock degrades to an exclusive hold, so appenders exclude
    each other there: a second concurrent :meth:`open` waits for the first appender
    to close (up to ``LODEDB_PERSIST_LOCK_TIMEOUT``), then raises
    :class:`ConcurrentWriterError`. On Unix appenders coexist freely.
    """

    def __init__(
        self,
        handle: NativeCoreAppenderHandle,
        embedder: EngineEmbeddingBackend | None = None,
    ) -> None:
        self._handle: NativeCoreAppenderHandle | None = handle
        self._embedder = embedder

    @classmethod
    def open(
        cls,
        path: str | PathLike[str],
        *,
        durability: str = "fast",
        store_text: bool = False,
        index_text: bool = False,
        acquire_writer_lock: bool = True,
        embedder: EngineEmbeddingBackend | None = None,
        chunk_character_limit: int = 900,
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

        ``embedder`` is an optional :class:`~lodedb.engine.embedding_backends.
        EngineEmbeddingBackend`; supply it to enable :meth:`append_text` /
        :meth:`append_text_many` (which chunk and embed like :meth:`LodeDB.add`).
        The vector-in :meth:`append` / :meth:`append_many` never embed and do not
        need it. ``chunk_character_limit`` must match the store writer's (LodeDB's
        default is 900) so appended text chunks identically; it is used only by the
        text-append path.
        """

        # Validate/normalize up front (raises on an unknown mode) so a typo cannot
        # silently degrade to buffered instead of the requested fsync.
        native_durability = "fsync" if normalize_durability(durability) else "relaxed"
        adapter = NativeCoreAdapter()
        if not adapter.available:
            raise RuntimeError("native core extension is not available")
        try:
            handle = adapter.open_appender(
                path=path,
                durability=native_durability,
                store_text=store_text,
                index_text=index_text,
                acquire_writer_lock=acquire_writer_lock,
                chunk_character_limit=chunk_character_limit,
            )
        except ValueError as exc:
            if _is_writer_lock_contention(exc):
                # Same lock and timeout as a writable LodeDB open; surface the
                # SDK's stable contention error so one retry loop covers both.
                raise ConcurrentWriterError(
                    f"appender at {path} could not take the lodedb lock (held by an "
                    "exclusive writer, or by another appender on Windows); close the "
                    "other handle, or raise LODEDB_PERSIST_LOCK_TIMEOUT to wait longer."
                ) from exc
            raise
        return cls(handle, embedder)

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
                    vector=_finite_float_vector(vector, normalize=normalize),
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

    def append_text(
        self,
        text: str,
        *,
        id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        """Chunks and embeds ``text``, logs one text record, and returns its LSN.

        Requires an ``embedder`` (see :meth:`open`). The document is chunked by the
        native core exactly as :meth:`LodeDB.add` chunks it, each new chunk is
        embedded by the configured backend, and the post-embedding record is logged
        to the WAL for the next writable open to fold. ``id`` is required (auto-ids
        would collide across concurrent writers). Whether the raw text and lexical
        tokens are retained follows the ``store_text``/``index_text`` the appender was
        opened with (match the store's writer).
        """

        return self.append_text_many([{"text": text, "id": id, "metadata": metadata}])

    def append_text_many(self, documents: Iterable[Mapping[str, Any]]) -> int:
        """Chunks and embeds a batch of ``{"text", "id", "metadata"?}`` documents,
        logs them as one text record, and returns its LSN. Each ``id`` and ``text``
        must be present and non-empty. Requires an ``embedder`` (see :meth:`open`)."""

        embedder = self._require_embedder()
        handle = self._require_open()
        prepared: list[EngineDocument] = []
        for item in documents:
            text = item.get("text")
            if text is None or not str(text).strip():
                raise ValueError("each document needs a non-empty 'text'")
            document_id = item.get("id")
            if document_id is None or not str(document_id).strip():
                raise ValueError("each document needs a non-empty 'id'")
            prepared.append(
                EngineDocument(
                    document_id=str(document_id),
                    text=str(text),
                    metadata=_coerce_metadata(item.get("metadata")),
                )
            )
        if not prepared:
            raise ValueError("append_text_many requires at least one document")
        # The native core chunks (embeddings stay here); embed the chunks it asks for,
        # then log the post-embedding record for the next writable open to fold.
        plan = handle.prepare_documents(prepared)
        chunks_to_embed = tuple(
            str(chunk.get("text", "")) for chunk in plan.get("chunks_to_embed", [])
        )
        embeddings = embedder.embed_documents(chunks_to_embed) if chunks_to_embed else ()
        return handle.append_embedded_documents(plan, embeddings)

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

    def _require_embedder(self) -> EngineEmbeddingBackend:
        if self._embedder is None:
            raise RuntimeError(
                "text append requires opening the appender with an embedder= backend"
            )
        return self._embedder
