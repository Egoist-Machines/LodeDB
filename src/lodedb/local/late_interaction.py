"""Late-interaction (multi-vector / MaxSim) retrieval for visual-document RAG (issue #25).

Each document is a *set* of patch vectors (e.g. one per image patch from a
ColPali/ColBERT encoder). A multi-vector query is scored against a document with
**MaxSim**: the sum over query tokens of the maximum dot product against that
document's patches. Retrieval is exact MaxSim over the candidate set.

Storage and scoring run in the native Rust core: each document is one row in an
embedded :class:`~lodedb.local.db.LodeDB` vector store (its id is the document id,
its row vector is the mean-pooled patch vector) and its full patch matrix is kept
in the native multi-vector store (``g<epoch>.tvmv`` + delta journal) at the
configured precision. The native engine decodes those matrices and scores them
with the ``turbovec`` MaxSim kernel; Python only embeds (optionally, via a
bring-your-own encoder) and orchestrates. Data stays on local disk.

The native engine is thread-confined, so every native operation for an index runs
on a single dedicated worker thread; a shared handle can therefore be used from
many threads (the writers serialize onto that one thread).
"""

from __future__ import annotations

import json
import threading
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from lodedb.engine._atomic_io import (
    durability_from_env,
    durable_replace,
    normalize_durability,
)
from lodedb.local.db import (
    _LOCAL_INDEX_ID,
    LodeDB,
    ReadOnlyError,
    _coerce_metadata,
    _normalize_filter,
)

# Reserved per-row metadata keys (bookkeeping, stripped from returned metadata).
_PATCH_COUNT_KEY = "patch_count"  # number of patch vectors in the document
_DTYPE_KEY = "li_dtype"  # storage precision the patch matrix was written at
_RESERVED_METADATA_KEYS = frozenset({_PATCH_COUNT_KEY, _DTYPE_KEY})

# Patch-matrix storage precision (persisted with the index).
_STORAGE_CHOICES = ("float16", "float32", "int8")
_DEFAULT_STORAGE = "float32"

# Per-index config sidecar (records the storage precision).
_CONFIG_FILENAME = "lodedb_late_interaction.meta"
_CONFIG_VERSION = 1


class LodeLateInteractionHit:
    """One MaxSim search hit: score, document id, redacted metadata, patch count."""

    def __init__(
        self,
        *,
        score: float,
        id: str,
        metadata: Mapping[str, Any] | None = None,
        patch_count: int = 0,
    ) -> None:
        """Stores the MaxSim score, document id, redacted metadata, and patch count."""

        self.score = float(score)
        self.id = str(id)
        self.metadata = dict(metadata or {})
        self.patch_count = int(patch_count)

    def __iter__(self):
        """Yields ``(score, id, metadata)`` so hits unpack like the spec tuple."""

        yield self.score
        yield self.id
        yield self.metadata

    def __repr__(self) -> str:
        """Returns a compact, payload-free representation of the hit."""

        return (
            f"LodeLateInteractionHit(score={self.score:.4f}, id={self.id!r}, "
            f"metadata={self.metadata!r}, patch_count={self.patch_count})"
        )

    def __eq__(self, other: object) -> bool:
        """Compares hits structurally (and to a plain ``(score, id, metadata)``)."""

        if isinstance(other, LodeLateInteractionHit):
            return (self.score, self.id, self.metadata, self.patch_count) == (
                other.score,
                other.id,
                other.metadata,
                other.patch_count,
            )
        if isinstance(other, tuple) and len(other) == 3:
            return (self.score, self.id, self.metadata) == other
        return NotImplemented


class LodeLateInteractionIndex:
    """Multi-vector (MaxSim) retrieval over a native bring-your-own-vectors index.

    Example::

        idx = LodeLateInteractionIndex("./pages", dim=128)
        # `page_patches` is a (num_patches, 128) matrix from your ColPali encoder.
        idx.add_document("report-p1", page_patches, metadata={"file": "report.pdf"})
        idx.persist()

        # `query_tokens` is a (num_query_tokens, 128) matrix for the query.
        for score, doc_id, meta in idx.search(query_tokens, k=5):
            ...

    Each document is one row in the embedded :class:`~lodedb.local.db.LodeDB`: its
    id is the document id, its row vector is the mean-pooled patch vector, and its
    full patch matrix is stored (at ``storage`` precision) in the native
    multi-vector store so MaxSim is computed exactly, natively, at query time.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dim: int,
        encoder: Any | None = None,
        bit_width: int = 4,
        storage: str | None = None,
        candidate_depth: int = 16,
        scoring: str = "numpy",
        resident: bool | str = "auto",
        resident_max_bytes: int = 512 * 1024 * 1024,
        read_only: bool = False,
        **lodedb_kwargs: Any,
    ) -> None:
        """Opens (or creates) a late-interaction index at ``path``.

        ``dim`` is the per-patch vector dimension (e.g. 128 for ColPali) and must
        be a positive multiple of 8. ``encoder`` is an optional bring-your-own
        encoder exposing ``encode_documents`` / ``encode_queries`` (only used by
        :meth:`add_texts` / :meth:`search_text`). ``storage`` is the patch-matrix
        precision persisted with the index: ``"float32"`` (default), ``"float16"``,
        or ``"int8"``; passing a value that disagrees with the stored one raises.

        ``scoring``, ``resident``, ``resident_max_bytes``, and ``candidate_depth``
        are accepted for backward compatibility but are now no-ops: storage and
        exact MaxSim run in the native core. ``read_only=True`` opens a non-mutating
        reader. ``bit_width`` and extra ``lodedb_kwargs`` are forwarded to the
        underlying vector store (``commit_mode`` is forced to ``"generation"`` so
        the patch matrices are durably journaled).
        """

        if int(dim) <= 0 or int(dim) % 8 != 0:
            raise ValueError("dim must be a positive multiple of 8")
        if storage is not None and storage not in _STORAGE_CHOICES:
            raise ValueError(f"storage must be one of {_STORAGE_CHOICES}")
        if scoring not in ("numpy", "native"):
            raise ValueError("scoring must be 'numpy' or 'native'")
        if resident not in (True, False, "auto"):
            raise ValueError("resident must be True, False, or 'auto'")
        if lodedb_kwargs.get("store_text") is False:
            raise ValueError(
                "LodeLateInteractionIndex retains per-document state; store_text must remain True"
            )
        self.dim = int(dim)
        # Retained for back-compat; storage + scoring are native now.
        self.scoring = scoring
        self.resident = resident
        self.resident_max_bytes = int(resident_max_bytes)
        self.candidate_depth = int(candidate_depth)
        self.encoder = encoder
        self.read_only = bool(read_only)
        durability = lodedb_kwargs.get("durability")
        self._fsync = (
            durability_from_env() if durability is None else normalize_durability(durability)
        )
        # _write_lock orders each mutation as a unit across concurrent callers.
        self._write_lock = threading.Lock()
        self._closed = False
        # The native CoreEngine is unsendable / thread-confined: run every native
        # op (including opening the LodeDB, which creates the engine) on one worker
        # thread, so a shared handle works across caller threads.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lodedb-li")
        self._worker_id: int | None = None

        lodedb_kwargs.pop("store_text", None)
        # Generation mode: the WAL replay path does not carry multi-vector ops, so
        # the patch matrix is made durable via the per-mutation generation commit.
        lodedb_kwargs["commit_mode"] = "generation"

        def _open() -> LodeDB:
            self._worker_id = threading.get_ident()
            return LodeDB.open_vector_store(
                path,
                vector_dim=self.dim,
                bit_width=int(bit_width),
                store_text=True,
                read_only=self.read_only,
                **lodedb_kwargs,
            )

        self._db = self._executor.submit(_open).result()
        # The native engine was created on the worker thread and must be dropped
        # there too (it is unsendable); a finalizer guarantees that even if the
        # caller never calls close() and the handle is garbage-collected.
        self._finalizer = weakref.finalize(
            self, _shutdown_native, self._executor, self._db, self._worker_id
        )
        if not self.read_only and self._db._native_vector_engine is None:
            self.close()
            raise RuntimeError(
                "late interaction requires the native core (LODEDB_NATIVE_CORE_WRITE) engine"
            )
        # Resolve the storage precision against any value persisted with the index.
        self.storage = self._resolve_storage(storage)

    def _call_on_native_thread(self, fn):
        """Runs ``fn`` on the engine's home thread (inline if already there)."""

        if self._closed:
            raise RuntimeError("late-interaction index is closed")
        if threading.get_ident() == self._worker_id:
            return fn()
        return self._executor.submit(fn).result()

    def _native(self):
        """Returns the native engine handle, raising if native is unavailable."""

        handle = self._db._native_vector_engine
        if handle is None:
            raise RuntimeError("late interaction requires the native core engine")
        return handle

    def _require_writable(self) -> None:
        """Raises :class:`ReadOnlyError` if this handle was opened read-only."""

        if self.read_only:
            raise ReadOnlyError("this late-interaction index is read-only")

    def _resolve_storage(self, requested: str | None) -> str:
        """Reconciles a requested precision with the one persisted in the index."""

        config_path = Path(self._db.path) / _CONFIG_FILENAME
        stored = _read_li_config(config_path)
        if requested is not None and stored is not None and requested != stored:
            raise ValueError(
                f"this index was created with storage={stored!r}; reopen with "
                f"storage={stored!r} or omit it (got storage={requested!r})"
            )
        resolved = requested or stored or _DEFAULT_STORAGE
        if not self.read_only and stored != resolved:
            _write_li_config(config_path, resolved, fsync=self._fsync)
        return resolved

    # -- write path ---------------------------------------------------------

    def add_document(
        self,
        id: str,
        patches: Any,
        *,
        metadata: Mapping[str, Any] | None = None,
        normalize: bool = True,
    ) -> str:
        """Stores one document from its set of patch vectors; returns its id.

        ``patches`` is a 2-D ``(num_patches, dim)`` matrix, L2-normalized by
        default so MaxSim dot-products are cosine similarities. The document is one
        row keyed ``id``; re-adding an id replaces it. Commits atomically.
        """

        self._require_writable()
        document, document_id = self._prepare_write(id, patches, metadata, normalize)
        with self._write_lock:
            self._call_on_native_thread(lambda: self._upsert_and_persist([document]))
        return document_id

    def add_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> list[str]:
        """Adds a batch of ``{"id", "patches", "metadata"?}`` documents; returns ids."""

        self._require_writable()
        prepared: list[dict[str, Any]] = []
        ids: list[str] = []
        for document in documents:
            if not isinstance(document, Mapping):
                raise ValueError("each document must be a mapping")
            patches = document.get("patches")
            if patches is None:
                raise ValueError("each document needs a 'patches' matrix")
            row, document_id = self._prepare_write(
                document.get("id"), patches, document.get("metadata"), normalize
            )
            prepared.append(row)
            ids.append(document_id)
        if prepared:
            with self._write_lock:
                self._call_on_native_thread(lambda: self._upsert_and_persist(prepared))
        return ids

    def add_texts(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> list[str]:
        """Encodes documents with the bring-your-own ``encoder`` and stores them."""

        self._require_writable()
        encoder = self._require_encoder()
        items = list(documents)
        contents = [item.get("content") for item in items]
        if any(content is None for content in contents):
            raise ValueError("each document needs 'content' to encode")
        matrices = list(encoder.encode_documents(contents))
        if len(matrices) != len(items):
            raise RuntimeError("encoder returned the wrong number of patch matrices")
        prepared = [
            {"id": item.get("id"), "patches": matrix, "metadata": item.get("metadata")}
            for item, matrix in zip(items, matrices, strict=True)
        ]
        return self.add_documents(prepared, normalize=normalize)

    def remove(self, id: str) -> bool:
        """Removes a document; returns True if it existed."""

        self._require_writable()
        document_id = _require_doc_id(id)
        with self._write_lock:
            return self._call_on_native_thread(lambda: self._delete_and_persist(document_id))

    def _prepare_write(
        self,
        id: Any,
        patches: Any,
        metadata: Mapping[str, Any] | None,
        normalize: bool,
    ) -> tuple[dict[str, Any], str]:
        """Builds one native multi-vector upsert document: pooled vector + matrix."""

        document_id = _require_doc_id(id)
        matrix = _as_matrix(patches, self.dim, normalize=normalize)
        row_meta = dict(_coerce_user_metadata(metadata))
        row_meta[_PATCH_COUNT_KEY] = str(matrix.shape[0])
        row_meta[_DTYPE_KEY] = self.storage
        document = {
            "document_id": document_id,
            "vector": _pool(matrix).tolist(),
            "metadata": row_meta,
            "dtype": self.storage,
            "patch_count": int(matrix.shape[0]),
            "patch_bytes": _encode_matrix(matrix, self.storage),
        }
        return document, document_id

    def _upsert_and_persist(self, documents: list[dict[str, Any]]) -> None:
        """Runs on the worker thread: native upsert then a durable generation commit."""

        native = self._native()
        native.upsert_multivector(_LOCAL_INDEX_ID, documents)
        native.persist()

    def _delete_and_persist(self, document_id: str) -> bool:
        """Runs on the worker thread: native delete then a durable generation commit."""

        native = self._native()
        response = native.delete_documents(_LOCAL_INDEX_ID, (document_id,))
        native.persist()
        return int(response.get("documents_deleted", 0) or 0) > 0

    # -- read path ----------------------------------------------------------

    def search(
        self,
        query: Any,
        *,
        k: int = 10,
        candidate_depth: int | None = None,
        filter: Mapping[str, Any] | None = None,
        normalize: bool = True,
    ) -> list[LodeLateInteractionHit]:
        """Returns the top-``k`` documents by exact MaxSim for a multi-vector query.

        ``query`` is a 2-D ``(num_query_tokens, dim)`` matrix. ``filter`` takes the
        same grammar as :meth:`LodeDB.search`; query tokens are L2-normalized by
        default. ``candidate_depth`` is accepted for compatibility but unused.
        """

        if int(k) <= 0:
            raise ValueError("k must be positive")
        query_matrix = _as_matrix(query, self.dim, normalize=normalize)
        normalized_filter = _normalize_filter(filter)
        response = self._call_on_native_thread(
            lambda: self._native().query_multivector(
                _LOCAL_INDEX_ID, query_matrix, top_k=int(k), filter=normalized_filter
            )
        )
        return [_hit_from_native(hit) for hit in response.get("hits", [])]

    def search_text(
        self,
        query: Any,
        *,
        k: int = 10,
        candidate_depth: int | None = None,
        filter: Mapping[str, Any] | None = None,
        normalize: bool = True,
    ) -> list[LodeLateInteractionHit]:
        """Encodes ``query`` with the bring-your-own ``encoder`` then searches."""

        encoder = self._require_encoder()
        matrices = list(encoder.encode_queries([query]))
        if len(matrices) != 1:
            raise RuntimeError("encoder must return exactly one query matrix")
        return self.search(
            matrices[0],
            k=k,
            candidate_depth=candidate_depth,
            filter=filter,
            normalize=normalize,
        )

    def search_many(
        self,
        queries: Sequence[Any],
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        normalize: bool = True,
    ) -> list[list[LodeLateInteractionHit]]:
        """Scores several multi-vector queries; returns top-``k`` per query, in order."""

        if int(k) <= 0:
            raise ValueError("k must be positive")
        return [
            self.search(query, k=k, filter=filter, normalize=normalize) for query in queries
        ]

    def count(self) -> int:
        """Returns the number of documents stored."""

        stats = self._call_on_native_thread(lambda: self._native().stats(_LOCAL_INDEX_ID))
        return int(stats.get("document_count", 0) or 0)

    def patch_count(self) -> int:
        """Returns the total number of patch vectors across all documents."""

        records = self._call_on_native_thread(
            lambda: self._native().list_documents(_LOCAL_INDEX_ID)
        )
        return sum(_patch_count_from_metadata(record.get("metadata", {})) for record in records)

    def list_documents(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns one redacted ``{"id", "metadata", "patch_count"}`` per document."""

        normalized = _normalize_filter(filter)
        records = self._call_on_native_thread(
            lambda: self._native().list_documents(_LOCAL_INDEX_ID, filter=normalized)
        )
        out = [
            {
                "id": _record_id(record),
                "metadata": _strip_internal_metadata(record.get("metadata", {})),
                "patch_count": _patch_count_from_metadata(record.get("metadata", {})),
            }
            for record in records
        ]
        out.sort(key=lambda item: item["id"])
        return out

    def persist(self) -> dict[str, Any]:
        """Flushes durable state and returns the underlying redacted storage stats."""

        def _do() -> dict[str, Any]:
            native = self._native()
            native.persist()
            return native.stats(_LOCAL_INDEX_ID)

        return self._call_on_native_thread(_do)

    def close(self) -> None:
        """Closes the underlying engine on its home thread; state stays on disk."""

        if self._closed:
            return
        self._closed = True
        # Runs _shutdown_native exactly once (the GC finalizer then no-ops).
        self._finalizer()

    def __enter__(self) -> LodeLateInteractionIndex:
        """Enters a context manager (state is already loaded on open)."""

        return self

    def __exit__(self, *exc: object) -> None:
        """Exits the context manager (state is durable on disk already)."""

        self.close()

    # -- internals ----------------------------------------------------------

    def _require_encoder(self) -> Any:
        """Returns the bring-your-own encoder or raises a clear error."""

        encoder = self.encoder
        if encoder is None or not (
            callable(getattr(encoder, "encode_documents", None))
            and callable(getattr(encoder, "encode_queries", None))
        ):
            raise RuntimeError(
                "this index has no encoder; pass encoder= exposing encode_documents / "
                "encode_queries, or use add_document / search with precomputed matrices"
            )
        return encoder


# -- module helpers ---------------------------------------------------------


def _shutdown_native(executor: ThreadPoolExecutor, db: LodeDB, worker_id: int | None) -> None:
    """Closes the engine on its home thread, then shuts the worker down.

    Used by both :meth:`LodeLateInteractionIndex.close` and the GC finalizer so the
    unsendable native engine is always dropped on the thread that created it.
    """

    on_worker = worker_id is not None and threading.get_ident() == worker_id
    try:
        if on_worker:
            db.close()
        elif worker_id is not None:
            executor.submit(db.close).result()
    except Exception:  # noqa: BLE001 - best-effort teardown (also runs at GC / exit)
        pass
    # Never wait when running on the worker itself (that would join this thread).
    executor.shutdown(wait=not on_worker)


def _hit_from_native(hit: Mapping[str, Any]) -> LodeLateInteractionHit:
    """Maps a native MaxSim hit to a public, redacted hit object."""

    metadata = hit.get("metadata", {})
    return LodeLateInteractionHit(
        score=float(hit.get("score", 0.0)),
        id=str(hit.get("document_id") or hit.get("id")),
        metadata=_strip_internal_metadata(metadata),
        patch_count=_patch_count_from_metadata(metadata),
    )


def _record_id(record: Mapping[str, Any]) -> str:
    """Returns the document id from a native list_documents record."""

    return str(record.get("document_id") or record.get("id") or "")


def _encode_matrix(matrix: np.ndarray, storage: str) -> bytes:
    """Serializes a ``(num_patches, dim)`` matrix to raw bytes at ``storage``.

    The native multi-vector store keeps these raw bytes verbatim and decodes them
    for MaxSim, so the layout matches the Rust ``MultiVecRecord::decode``: raw
    little-endian f4/f2 for float32/float16, or per-vector symmetric int8 (the f32
    scales followed by the i8 codes).
    """

    if storage == "float32":
        return np.ascontiguousarray(matrix, dtype="<f4").tobytes()
    if storage == "float16":
        return np.ascontiguousarray(matrix, dtype="<f2").tobytes()
    if storage == "int8":
        scales = np.abs(matrix).max(axis=1).astype("<f4")
        safe = np.where(scales == 0.0, 1.0, scales).astype(np.float32)
        codes = np.clip(np.round(matrix / safe[:, None] * 127.0), -127, 127).astype(np.int8)
        return scales.tobytes() + codes.tobytes()
    raise ValueError(f"unknown storage {storage!r}")


def _require_doc_id(value: Any) -> str:
    """Validates and returns a non-empty document id."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("document id must be a non-empty string")
    return value


def _coerce_user_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validates user metadata and reserves the internal bookkeeping keys."""

    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    collisions = sorted(str(key) for key in metadata if str(key) in _RESERVED_METADATA_KEYS)
    if collisions:
        raise ValueError(
            f"metadata uses reserved late-interaction keys: {', '.join(collisions)}"
        )
    # Coerce values to the engine's string metadata model (bool -> "true"/"false",
    # numbers -> str) so live, filtered, and reopened reads agree byte-for-byte.
    return _coerce_metadata(dict(metadata))


def _strip_internal_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Drops the internal bookkeeping keys from row metadata."""

    return {
        key: value
        for key, value in metadata.items()
        if key not in _RESERVED_METADATA_KEYS
    }


def _patch_count_from_metadata(metadata: Mapping[str, Any]) -> int:
    """Reads the stamped patch count from a row's metadata, or 0 if absent."""

    try:
        return int(metadata.get(_PATCH_COUNT_KEY, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _read_li_config(config_path: Path) -> str | None:
    """Returns the index's persisted storage precision, or ``None`` if unset."""

    try:
        raw = config_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        data = json.loads(raw)
        storage = data["storage"] if isinstance(data, Mapping) else None
    except (ValueError, TypeError, KeyError):
        storage = None
    if storage not in _STORAGE_CHOICES:
        raise ValueError(
            f"corrupt late-interaction config at {config_path}: storage={storage!r}"
        )
    return storage


def _write_li_config(config_path: Path, storage: str, *, fsync: bool) -> None:
    """Atomically persists the index's storage precision to its config sidecar."""

    tmp = config_path.with_name(config_path.name + ".tmp")
    tmp.write_text(
        json.dumps({"version": _CONFIG_VERSION, "storage": storage}),
        encoding="utf-8",
    )
    durable_replace(tmp, config_path, fsync=fsync)


def _as_matrix(matrix: Any, dim: int, *, normalize: bool) -> np.ndarray:
    """Coerces a patch/token matrix to a finite float32 ``(rows, dim)`` array."""

    array = np.asarray(matrix, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError("patches/query must be a 2-D (rows, dim) matrix")
    if array.shape[0] == 0:
        raise ValueError("matrix must have at least one row")
    if array.shape[1] != dim:
        raise ValueError(f"each vector must have dimension {dim}, got {array.shape[1]}")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix must contain only finite values")
    if normalize:
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if np.any(norms == 0.0):
            raise ValueError(
                "cannot normalize a zero-vector row; pass non-zero rows or normalize=False"
            )
        array = array / norms
    return np.ascontiguousarray(array, dtype=np.float32)


def _pool(matrix: np.ndarray) -> np.ndarray:
    """Returns the document's mean-pooled unit vector (the stored row vector)."""

    pooled = matrix.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm == 0.0:
        return np.ascontiguousarray(matrix[0], dtype=np.float32)
    return np.ascontiguousarray(pooled / norm, dtype=np.float32)
