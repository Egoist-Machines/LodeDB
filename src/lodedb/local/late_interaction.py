"""Late-interaction (multi-vector / MaxSim) retrieval for visual-document RAG (issue #25).

ColBERT-style late interaction, and its visual-document descendants (ColPali /
ColQwen), represent each document -- or each page rendered as an image -- as a
*set* of token/patch vectors rather than a single pooled vector, and score a
query against a document with **MaxSim**::

    score(query, doc) = sum over query tokens t of  max over doc patches p of  <q_t, d_p>

This is built on top of a bring-your-own-vectors :class:`LodeDB`
(:meth:`LodeDB.open_vector_store`), with no engine change. Each document is stored
as **one row**: the row's id is the document id, its vector is the document's
mean-pooled patch vector, and its full patch matrix is kept in the per-row text
sidecar -- a compact multi-vector layout that holds a page's ~1000 patches in a
single row instead of ~1000 rows, which keeps both ingest and on-disk footprint
low. The patch matrix is stored at a configurable precision (``storage=``):
``float32`` (default, fastest query and bit exact), ``float16`` (~exact at half
the size), or ``int8`` (a per-vector-scaled quantization, ~4x smaller).

Retrieval is exact MaxSim over the documents, by one of three paths
(:meth:`search` picks one automatically; all return the true top-``k``):

1. *Resident* -- the default for an unfiltered query whose corpus fits
   ``resident_max_bytes``: every patch is held in one in-memory matrix and scored
   in a single GEMM plus a segmented max, at a few milliseconds on thousands of
   pages.
2. *Filtered* -- a query with a ``filter`` scores the matching subset
   exhaustively (the filter is resolved engine-side, then those documents are
   scored), so a metadata filter both narrows and stays exact.
3. *Streaming* -- a corpus over the resident budget (or ``resident=False``) is
   scored by reading the documents back from disk in bounded chunks. Slower
   (disk-bound) but exact and constant-memory, so the exact path is never capped
   by RAM.

The exact MaxSim itself defaults to numpy (a ``query @ patches.T`` BLAS GEMM); a
native TurboVec ``maxsim_scores`` Rust kernel is also available
(``scoring="native"``) for builds without a fast BLAS. Both return identical
scores.

The page/token encoder is **bring-your-own** (ColPali / ColQwen weights are
multi-GB and out of scope to bundle): pass precomputed patch matrices to
:meth:`add_document` / :meth:`search`, or an optional ``encoder`` exposing
``encode_documents`` / ``encode_queries``.
"""

from __future__ import annotations

import base64
import importlib
import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lodedb.engine._atomic_io import (
    durability_from_env,
    durable_replace,
    normalize_durability,
)
from lodedb.local.db import LodeDB, _coerce_metadata

# The bundled TurboVec extension exposes a native MaxSim kernel (`maxsim_scores`)
# that scores documents in parallel Rust. It is resolved once and cached; a build
# that predates the kernel (or a stock standalone `turbovec`) falls back to the
# numpy path, so the SDK never hard-depends on the kernel being present.
_TURBOVEC_PACKAGE_NAMES = ("lodedb._turbovec", "turbovec")
_UNRESOLVED = object()  # distinct from None, which means "looked up, not present"
_native_maxsim: Callable[..., Any] | None | object = _UNRESOLVED

# Metadata keys LodeLateInteractionIndex reserves on each document row; a
# caller-supplied metadata mapping may not set them, and they are stripped from
# the metadata returned to callers.
_PATCH_COUNT_KEY = "patch_count"  # number of patch vectors in the document
_DTYPE_KEY = "li_dtype"  # storage precision the patch matrix was written at
_RESERVED_METADATA_KEYS = frozenset({_PATCH_COUNT_KEY, _DTYPE_KEY})

# Supported patch-matrix storage precisions.
_STORAGE_CHOICES = ("float16", "float32", "int8")
# Default precision for a brand-new index when the caller does not choose one.
# float32 favors query speed and bit-exactness; float16 / int8 trade a little for
# a smaller footprint.
_DEFAULT_STORAGE = "float32"
# Index-level config sidecar (precision the index was created with), so the
# choice persists and is reused on reopen without re-passing ``storage=``. The
# extension deliberately avoids ``.json`` so the engine's ``*.json`` index scan
# never mistakes it for persisted engine state; the contents are still JSON.
_CONFIG_FILENAME = "lodedb_late_interaction.meta"
_CONFIG_VERSION = 1

# Sentinel stored in place of the resident cache once it is known to exceed the
# budget, so an over-budget "auto" index streams without rebuilding (and
# re-enumerating) the cache on every query. A remove resets it so the index can
# become resident again as it shrinks.
_RESIDENT_OVER_BUDGET = object()

# Cap the transient float32 work buffer when scoring a resident or streamed scan,
# so peak memory stays bounded regardless of corpus size.
_SCORE_CHUNK_BYTES = 64 * 1024 * 1024
# Resident-cache compaction triggers: when stale (pending + tombstoned) patches
# reach max(this floor, half the base), or the pending delta reaches this many
# documents. The floor keeps small indexes from compacting over trivial churn.
_COMPACT_MIN_STALE_PATCHES = 50_000
_COMPACT_MAX_PENDING_DOCS = 2_000


def _resolve_native_maxsim() -> Callable[..., Any] | None:
    """Returns the native ``maxsim_scores`` kernel, or ``None`` if unavailable.

    The lookup is performed once and memoized. The bundled ``lodedb._turbovec``
    is tried before a standalone ``turbovec`` so a stray stock package cannot
    shadow the patched core, mirroring the engine's TurboVec loader.
    """

    global _native_maxsim
    if _native_maxsim is not _UNRESOLVED:  # already resolved (callable or None)
        return _native_maxsim  # type: ignore[return-value]
    resolved: Callable[..., Any] | None = None
    for name in _TURBOVEC_PACKAGE_NAMES:
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        candidate = getattr(module, "maxsim_scores", None)
        if callable(candidate):
            resolved = candidate
            break
    _native_maxsim = resolved
    return resolved


class LodeLateInteractionHit:
    """One late-interaction result: ``(score, id, metadata)`` for a *document*.

    ``score`` is the MaxSim score (sum over query tokens of the max patch
    similarity), ``id`` is the document id, and ``metadata`` is the user metadata
    supplied to :meth:`LodeLateInteractionIndex.add_document` (the internal
    bookkeeping keys are stripped). Unpacks like a
    :class:`~lodedb.local.db.LodeSearchHit`.
    """

    __slots__ = ("score", "id", "metadata", "patch_count")

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
    """Multi-vector (MaxSim) retrieval over a bring-your-own-vectors LodeDB index.

    Example::

        idx = LodeLateInteractionIndex("./pages", dim=128)
        # `page_patches` is a (num_patches, 128) matrix from your ColPali encoder.
        idx.add_document("report-p1", page_patches, metadata={"file": "report.pdf"})
        idx.persist()

        # `query_tokens` is a (num_query_tokens, 128) matrix for the query.
        for score, doc_id, meta in idx.search(query_tokens, k=5):
            ...

    Each document is one row in the embedded :class:`~lodedb.local.db.LodeDB`: its
    id is the document id, its vector is the mean-pooled patch vector, and its full
    patch matrix is kept (at ``storage`` precision, base64, in the per-row text
    sidecar) so MaxSim is computed exactly at query time. Data stays on local disk;
    nothing is sent anywhere.
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
        be a positive multiple of 8 (the TurboVec store's requirement). All
        document patches and query tokens must share it. ``encoder`` is an optional
        bring-your-own page/token encoder exposing
        ``encode_documents(list[...]) -> list[2-D matrix]`` and
        ``encode_queries(list[...]) -> list[2-D matrix]``; it is only used by
        :meth:`add_texts` / :meth:`search_text` and is never required for the
        precomputed-matrix API.

        ``storage`` is the patch-matrix precision and is **persisted with the
        index**: ``"float32"`` (default for a new index, fastest query and bit
        exact), ``"float16"`` (near-exact at half the size), or ``"int8"``
        (per-vector-scaled, ~4x smaller, a small recall cost). Leave it ``None`` to
        adopt the precision the index was created with (or float32 for a new
        index); passing a value that disagrees with the stored one raises
        :class:`ValueError`, so an index keeps a single precision. ``resident``
        controls the default
        fast path: ``"auto"`` (default) holds the corpus in memory and scans it
        exactly for unfiltered queries when it fits ``resident_max_bytes`` (default
        512 MB), and streams from disk otherwise; ``True`` always builds the
        resident matrix; ``False`` always streams. ``scoring`` selects the
        exact-MaxSim backend: ``"numpy"`` (default BLAS GEMM) or ``"native"`` (the
        TurboVec Rust kernel; falls back to numpy if absent). ``candidate_depth`` is
        accepted for backward compatibility but unused -- all paths are exhaustive
        and exact. ``read_only=True`` opens a non-mutating reader and requires the
        path to exist. ``bit_width`` and any extra ``lodedb_kwargs`` (e.g.
        ``durability=``, ``commit_mode=``) are forwarded to the underlying
        vector-only :class:`LodeDB`. The patch text sidecar is always retained, so
        ``store_text`` may not be set ``False``.
        """

        if int(dim) <= 0 or int(dim) % 8 != 0:
            # The TurboVec store requires a positive multiple of 8; fail fast here
            # with a clear message instead of a cryptic engine error on first add.
            raise ValueError("dim must be a positive multiple of 8")
        if storage is not None and storage not in _STORAGE_CHOICES:
            raise ValueError(f"storage must be one of {_STORAGE_CHOICES}")
        if scoring not in ("numpy", "native"):
            raise ValueError("scoring must be 'numpy' or 'native'")
        if resident not in (True, False, "auto"):
            raise ValueError("resident must be True, False, or 'auto'")
        if lodedb_kwargs.get("store_text") is False:
            raise ValueError(
                "LodeLateInteractionIndex stores patch matrices in the text sidecar; "
                "store_text must remain True"
            )
        self.dim = int(dim)
        self.scoring = scoring
        self.resident = resident
        self.resident_max_bytes = int(resident_max_bytes)
        self.candidate_depth = int(candidate_depth)
        self.encoder = encoder
        self.read_only = bool(read_only)
        # Match the underlying DB's durability for the config sidecar write: fsync
        # the precision marker when the DB fsyncs its own commits.
        durability = lodedb_kwargs.get("durability")
        self._fsync = (
            durability_from_env() if durability is None else normalize_durability(durability)
        )
        # In-memory serving cache (all patches as one compact matrix); built lazily
        # on the first eligible search and maintained in place on writes. Guarded by
        # a lock so a query that reads it can run concurrently with a mutation on a
        # shared handle (the engine offers the same in-process guarantee), and set
        # to _RESIDENT_OVER_BUDGET once it is known not to fit.
        self._resident_cache: dict[str, Any] | object | None = None
        # _write_lock orders each mutation's (DB commit + cache note) as one unit so
        # concurrent writers cannot reorder cache notes vs commits; _cache_lock
        # guards the cache structure so a reader's snapshot is consistent.
        self._write_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        lodedb_kwargs.pop("store_text", None)
        self._db = LodeDB.open_vector_store(
            path,
            vector_dim=self.dim,
            bit_width=int(bit_width),
            store_text=True,
            read_only=self.read_only,
            **lodedb_kwargs,
        )
        # Late interaction is a multi-vector store (the patch matrix rides the
        # per-row text sidecar). The native core has no late-interaction concept
        # and its sole-writer model is thread-confined, so keep the Python engine
        # the durable writer here: it serves multi-vector reads/removes correctly
        # and supports the shared-handle concurrent writers `lodedb serve` needs.
        # No write has happened yet, so this is a clean handoff (a no-op on disk).
        if not self.read_only:
            self._db._disable_native_write_through()
        # Resolve the storage precision against any value persisted with the index,
        # so the choice survives reopen and stays consistent across writes.
        self.storage = self._resolve_storage(storage)

    def _resolve_storage(self, requested: str | None) -> str:
        """Reconciles a requested precision with the one persisted in the index.

        Reads the index's config sidecar; a requested value that disagrees with a
        stored one is rejected so the index keeps a single precision. The resolved
        precision is written back (on a writable handle) so an index created
        empty, or with an explicit choice, remembers it on reopen.
        """

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

        ``patches`` is a 2-D ``(num_patches, dim)`` matrix (a numpy array or a
        sequence of equal-length rows), L2-normalized by default so MaxSim
        dot-products are cosine similarities. The document is one row keyed ``id``;
        re-adding an existing id replaces it. Commits atomically.
        """

        row, document_id = self._prepare_write(id, patches, metadata, normalize)
        # Serialize the commit and its cache note as one ordered mutation, so
        # concurrent writers can never apply the cache notes in a different order
        # than they committed to disk (which would leave the cache stale).
        with self._write_lock:
            self._db.add_vectors_many([row])
            self._cache_note_writes([row])
        return document_id

    def add_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> list[str]:
        """Adds a batch of ``{"id", "patches", "metadata"?}`` documents in one commit.

        Returns the ids in input order.
        """

        rows: list[dict[str, Any]] = []
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
            rows.append(row)
            ids.append(document_id)
        if rows:
            with self._write_lock:
                self._db.add_vectors_many(rows, normalize=False)
                self._cache_note_writes(rows)
        return ids

    def add_texts(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> list[str]:
        """Encodes documents with the bring-your-own ``encoder`` and stores them.

        Each item is ``{"id", "content", "metadata"?}`` and
        ``encoder.encode_documents([content, ...])`` must return one 2-D patch
        matrix per item. Raises :class:`RuntimeError` if no encoder is set.
        """

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

        document_id = _require_doc_id(id)
        # Same ordered-mutation discipline as add: commit and cache note together.
        with self._write_lock:
            removed = self._db.remove(document_id)
            if removed:
                self._cache_note_remove(document_id)
        return removed

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

        ``query`` is a 2-D ``(num_query_tokens, dim)`` matrix. One of three exact
        paths is chosen automatically: a filtered query scores the matching subset
        exhaustively; an unfiltered query uses the in-memory resident scan when the
        corpus fits ``resident_max_bytes`` and otherwise streams from disk. All
        return the true top-``k``. ``filter`` takes the same grammar as
        :meth:`LodeDB.search`; query tokens are L2-normalized by default.
        ``candidate_depth`` is accepted for compatibility but unused.
        """

        if int(k) <= 0:
            raise ValueError("k must be positive")
        query_matrix = _as_matrix(query, self.dim, normalize=normalize)
        prefer_native = self.scoring == "native"
        budget = self._chunk_patch_budget(query_matrix.shape[0])

        if filter is not None:
            matching = self._filtered_documents_with_counts(filter)
            return self._topk_from_chunks(
                query_matrix, self._disk_chunks(matching, budget), int(k), prefer_native
            )

        snapshot = self._resident_snapshot()
        if snapshot is not None:
            return self._topk_from_chunks(
                query_matrix,
                self._resident_chunks(snapshot, budget),
                int(k),
                prefer_native,
            )

        # Over the resident budget (or resident=False): stream from disk, exact.
        return self._topk_from_chunks(
            query_matrix,
            self._disk_chunks(self._all_documents_with_counts(), budget),
            int(k),
            prefer_native,
        )

    def search_text(
        self,
        query: Any,
        *,
        k: int = 10,
        candidate_depth: int | None = None,
        filter: Mapping[str, Any] | None = None,
        normalize: bool = True,
    ) -> list[LodeLateInteractionHit]:
        """Encodes ``query`` with the bring-your-own ``encoder`` then searches.

        ``encoder.encode_queries([query])`` must return one 2-D token matrix.
        Raises :class:`RuntimeError` with no encoder.
        """

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
        """Scores several multi-vector queries together; returns top-``k`` per query.

        Each item of ``queries`` is a 2-D ``(num_query_tokens, dim)`` matrix. The
        whole batch is scored with one GEMM per document chunk (all queries'
        tokens stacked) instead of one GEMM per query, so this is the throughput
        path for evaluating or answering many queries at once; single-query latency
        is unchanged from :meth:`search`. The same ``filter`` applies to every
        query. Results preserve query order. (This path always uses the numpy BLAS
        scorer; ``scoring="native"`` applies to single-query :meth:`search`.)
        """

        if int(k) <= 0:
            raise ValueError("k must be positive")
        matrices = [_as_matrix(q, self.dim, normalize=normalize) for q in queries]
        if not matrices:
            return []
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for matrix in matrices:
            offsets.append((cursor, cursor + matrix.shape[0]))
            cursor += matrix.shape[0]
        queries_concat = np.ascontiguousarray(np.vstack(matrices), dtype=np.float32)
        budget = self._chunk_patch_budget(queries_concat.shape[0])

        if filter is not None:
            matching = self._filtered_documents_with_counts(filter)
            return self._topk_from_chunks_multi(
                queries_concat, offsets, self._disk_chunks(matching, budget), int(k)
            )

        snapshot = self._resident_snapshot()
        if snapshot is not None:
            return self._topk_from_chunks_multi(
                queries_concat,
                offsets,
                self._resident_chunks(snapshot, budget),
                int(k),
            )

        return self._topk_from_chunks_multi(
            queries_concat,
            offsets,
            self._disk_chunks(self._all_documents_with_counts(), budget),
            int(k),
        )

    def count(self) -> int:
        """Returns the number of documents stored."""

        return self._db.count()

    def patch_count(self) -> int:
        """Returns the total number of patch vectors across all documents."""

        return sum(
            _patch_count_from_metadata(record.get("metadata", {}))
            for record in self._db.list_documents()
        )

    def list_documents(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns one redacted ``{"id", "metadata", "patch_count"}`` per document.

        ``filter`` (the same grammar as :meth:`LodeDB.list_documents`) is matched
        against your document metadata.
        """

        records = self._db.list_documents(filter=dict(filter) if filter else None)
        out = [
            {
                "id": record["id"],
                "metadata": _strip_internal_metadata(record.get("metadata", {})),
                "patch_count": _patch_count_from_metadata(record.get("metadata", {})),
            }
            for record in records
        ]
        out.sort(key=lambda item: item["id"])
        return out

    def persist(self) -> dict[str, Any]:
        """Flushes durable state and returns the underlying redacted storage stats."""

        return self._db.persist()

    def close(self) -> None:
        """Releases the writer lock and underlying engine; state stays on disk."""

        self._db.close()

    def __enter__(self) -> LodeLateInteractionIndex:
        """Enters a context manager (state is already loaded on open)."""

        return self

    def __exit__(self, *exc: object) -> None:
        """Exits the context manager (state is durable on disk already)."""

        self.close()

    # -- internals ----------------------------------------------------------

    def _prepare_write(
        self,
        id: Any,
        patches: Any,
        metadata: Mapping[str, Any] | None,
        normalize: bool,
    ) -> tuple[dict[str, Any], str]:
        """Prepares one document write: returns ``(engine_row, document_id)``.

        Only the engine row is produced here (pooled vector + encoded matrix +
        stamped metadata). The resident-cache representation is derived later, in
        :meth:`_cache_note_writes`, and only when a cache actually exists -- so a
        cold bulk ingest does not decode/retain a second copy of every matrix.
        """

        document_id = _require_doc_id(id)
        matrix = _as_matrix(patches, self.dim, normalize=normalize)
        row_meta = dict(_coerce_user_metadata(metadata))
        row_meta[_PATCH_COUNT_KEY] = str(matrix.shape[0])
        row_meta[_DTYPE_KEY] = self.storage
        row = {
            "vector": _pool(matrix).tolist(),
            "id": document_id,
            "metadata": row_meta,
            "text": _encode_matrix(matrix, self.storage),
        }
        return row, document_id

    # -- resident cache maintenance -----------------------------------------

    def _cache_note_writes(self, rows: list[dict[str, Any]]) -> None:
        """Folds just-committed rows into the live resident cache, if it is built.

        Called while holding ``_write_lock`` (so ``self._resident_cache`` is stable
        and no build can race). When no cache exists -- e.g. a cold bulk ingest
        before the first query, or an over-budget index -- this returns immediately
        without decoding anything, so ingest does not pay for cache work it will not
        use. New/updated documents land in a pending delta that is compacted
        periodically; the cached representation is decoded from the stored row so it
        matches a fresh reopen (int8/float16 precision and string metadata).
        """

        if not isinstance(self._resident_cache, dict):
            return
        resident_dtype = self._resident_dtype()
        notes: list[tuple[str, np.ndarray, dict[str, Any], int]] = []
        for row in rows:
            row_meta = row["metadata"]
            matrix = _decode_matrix(row["text"], self.storage, self.dim).astype(
                resident_dtype, copy=False
            )
            notes.append(
                (
                    row["id"],
                    matrix,
                    _strip_internal_metadata(_coerce_metadata(row_meta)),
                    _patch_count_from_metadata(row_meta),
                )
            )
        with self._cache_lock:
            cache = self._resident_cache
            if not isinstance(cache, dict):
                return
            for document_id, matrix, user_meta, patch_count in notes:
                self._cache_replace(cache, document_id, matrix, user_meta, patch_count)
            self._cache_maybe_compact_or_evict(cache)

    def _cache_note_remove(self, document_id: str) -> None:
        """Reflects a removed document in the live resident cache, if built."""

        with self._cache_lock:
            cache = self._resident_cache
            if cache is _RESIDENT_OVER_BUDGET:
                # A delete may bring the corpus back under budget: drop the
                # over-budget marker so the next query re-evaluates residency.
                self._resident_cache = None
                return
            if not isinstance(cache, dict):
                return
            if document_id in cache["pending_ids"]:
                self._cache_drop_pending(cache, document_id)
            self._tombstone_base(cache, document_id)
            self._cache_maybe_compact_or_evict(cache)

    def _cache_replace(
        self,
        cache: dict[str, Any],
        document_id: str,
        matrix: np.ndarray,
        user_meta: dict[str, Any],
        patch_count: int,
    ) -> None:
        """Upserts one document into the pending delta (masking any prior copy)."""

        if document_id in cache["pending_ids"]:
            self._cache_drop_pending(cache, document_id)
        self._tombstone_base(cache, document_id)
        cache["pending"].append((matrix, document_id, user_meta, int(patch_count)))
        cache["pending_ids"].add(document_id)
        cache["pending_patches"] += int(matrix.shape[0])

    @staticmethod
    def _tombstone_base(cache: dict[str, Any], document_id: str) -> None:
        """Masks a base copy of the document and accounts its patch volume once.

        Tracking removed patch volume lets a delete-heavy workload trigger
        compaction, instead of scoring dead base rows on every query forever.
        """

        if document_id in cache["base_ids_set"] and document_id not in cache["removed"]:
            cache["removed"].add(document_id)
            cache["removed_patches"] += cache["base_pc_by_id"].get(document_id, 0)

    @staticmethod
    def _cache_drop_pending(cache: dict[str, Any], document_id: str) -> None:
        """Removes a document's entry from the pending delta."""

        for index, entry in enumerate(cache["pending"]):
            if entry[1] == document_id:
                cache["pending_patches"] -= int(entry[0].shape[0])
                del cache["pending"][index]
                break
        cache["pending_ids"].discard(document_id)

    def _cache_maybe_compact_or_evict(self, cache: dict[str, Any]) -> None:
        """Evicts if the live corpus can't fit the budget, else compacts if stale.

        The eviction check uses the *post-compaction* live size (base minus
        tombstones plus pending), so a growth-heavy index that cannot fit is
        marked over budget directly -- without first paying a full base+pending
        ``vstack`` only to discard it. When it can fit, stale (pending +
        tombstoned) volume is compacted so reclaiming it keeps a delete-heavy index
        resident. Called while holding the lock.
        """

        base_patches = int(cache["counts"].sum()) if cache["counts"].size else 0
        live_patches = base_patches - cache["removed_patches"] + cache["pending_patches"]
        itemsize = np.dtype(self._resident_dtype()).itemsize
        if (
            self.resident == "auto"
            and live_patches * self.dim * itemsize > self.resident_max_bytes
        ):
            self._resident_cache = _RESIDENT_OVER_BUDGET
            return
        # Stale = queued adds + tombstoned base patches. Compact when it reaches
        # ~half the base (amortized O(N)), past a pending-doc cap (bounds the
        # per-query pending vstack), or once the base is entirely tombstoned.
        stale = cache["pending_patches"] + cache["removed_patches"]
        all_base_removed = bool(cache["ids"]) and len(cache["removed"]) == len(cache["ids"])
        if (
            stale >= max(_COMPACT_MIN_STALE_PATCHES, base_patches // 2)
            or len(cache["pending"]) >= _COMPACT_MAX_PENDING_DOCS
            or all_base_removed
        ):
            self._cache_compact(cache)

    def _cache_compact(self, cache: dict[str, Any]) -> None:
        """Folds the pending delta and tombstones into a fresh contiguous base."""

        resident_dtype = self._resident_dtype()
        counts = cache["counts"]
        offsets = np.zeros(len(counts), dtype=np.intp)
        if len(counts) > 1:
            np.cumsum(counts[:-1], out=offsets[1:])
        flat = cache["flat"]
        new_ids: list[str] = []
        new_mats: list[np.ndarray] = []
        new_metas: list[dict[str, Any]] = []
        new_patch_counts: list[int] = []
        removed = cache["removed"]
        for index, document_id in enumerate(cache["ids"]):
            if document_id in removed:
                continue
            start = int(offsets[index])
            count = int(counts[index])
            new_ids.append(document_id)
            new_mats.append(flat[start : start + count])
            new_metas.append(cache["metas"][index])
            new_patch_counts.append(cache["patch_counts"][index])
        for matrix, document_id, user_meta, patch_count in cache["pending"]:
            new_ids.append(document_id)
            new_mats.append(matrix)
            new_metas.append(user_meta)
            new_patch_counts.append(patch_count)
        if new_mats:
            cache["flat"] = np.ascontiguousarray(np.vstack(new_mats), dtype=resident_dtype)
            cache["counts"] = np.fromiter(
                (m.shape[0] for m in new_mats), dtype=np.int64, count=len(new_mats)
            )
        else:
            cache["flat"] = np.zeros((0, self.dim), dtype=resident_dtype)
            cache["counts"] = np.zeros(0, dtype=np.int64)
        cache["ids"] = new_ids
        cache["base_ids_set"] = set(new_ids)
        cache["metas"] = new_metas
        cache["patch_counts"] = new_patch_counts
        cache["base_pc_by_id"] = dict(zip(new_ids, new_patch_counts, strict=True))
        cache["pending"] = []
        cache["pending_ids"] = set()
        cache["pending_patches"] = 0
        cache["removed"] = set()
        cache["removed_patches"] = 0

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

    def _resident_dtype(self) -> type[np.floating]:
        """The in-memory dtype for the resident matrix (compact unless float32)."""

        return np.float32 if self.storage == "float32" else np.float16

    def _all_documents_with_counts(self) -> list[tuple[str, int]]:
        """Returns every ``(document_id, patch_count)`` (sorted by id).

        ``LodeDB.list_documents`` already returns ``[]`` for an empty / not-yet-
        written store; any other engine error is a real failure and propagates, so
        a corrupt store fails closed rather than silently scoring nothing.
        """

        out = [
            (record["id"], _patch_count_from_metadata(record.get("metadata", {})))
            for record in self._db.list_documents()
        ]
        out.sort(key=lambda item: item[0])
        return out

    def _filtered_documents_with_counts(
        self, filter: Mapping[str, Any]
    ) -> list[tuple[str, int]]:
        """Returns ``(id, patch_count)`` for the filter matches (``[]`` if none)."""

        return [
            (record["id"], _patch_count_from_metadata(record.get("metadata", {})))
            for record in self._db.list_documents(filter=dict(filter))
        ]

    def _load_documents(
        self, document_ids: list[str]
    ) -> dict[str, tuple[np.ndarray, dict[str, Any], int]]:
        """Batch-loads ``{id: (patch_matrix_f32, user_metadata, patch_count)}``.

        One enumeration plus one text read cover the whole batch. Each document is
        decoded using the precision it was written at.
        """

        if not document_ids:
            return {}
        records = self._db.list_documents(filter={"document_ids": list(document_ids)})
        meta_by_id = {record["id"]: record.get("metadata", {}) for record in records}
        texts = self._db.get_texts(list(document_ids))
        out: dict[str, tuple[np.ndarray, dict[str, Any], int]] = {}
        for doc_id in document_ids:
            blob = texts.get(doc_id)
            if blob is None:
                continue
            meta = meta_by_id.get(doc_id, {})
            row_dtype = str(meta.get(_DTYPE_KEY, "float32"))
            # Cast to the resident serving dtype so every path (resident, streaming,
            # filtered) scores int8/float16 at the same precision and returns the
            # same top-k; otherwise streaming would score int8 at float32 while the
            # resident cache scores it at float16.
            matrix = _decode_matrix(blob, row_dtype, self.dim).astype(
                self._resident_dtype(), copy=False
            )
            patch_count = _patch_count_from_metadata(meta) or matrix.shape[0]
            out[doc_id] = (matrix, _strip_internal_metadata(meta), patch_count)
        return out

    def _chunk_patch_budget(self, total_query_tokens: int) -> int:
        """Patches per scoring chunk that keeps the work buffers under the cap.

        Bounded so neither the float32 upcast (``patches x dim``) nor the score
        matrix (``total_query_tokens x patches``) exceeds ``_SCORE_CHUNK_BYTES``.
        """

        width = max(self.dim, int(total_query_tokens), 1)
        return max(1, _SCORE_CHUNK_BYTES // (width * 4))

    def _disk_chunks(
        self, documents: list[tuple[str, int]], max_patches: int
    ) -> Iterator[tuple]:
        """Yields bounded scoring chunks read from disk, batched by patch budget.

        ``documents`` is ``(id, patch_count)`` pairs. Documents are accumulated into
        a load batch until adding the next would exceed ``max_patches`` (a single
        document larger than the budget loads alone), so the decoded/vstacked
        working set stays bounded -- the streaming path is constant-memory in the
        patch budget, not in a fixed document count.
        """

        batch_ids: list[str] = []
        batch_patches = 0
        for doc_id, patch_count in documents:
            estimate = max(int(patch_count), 1)
            if batch_ids and batch_patches + estimate > max_patches:
                yield from self._load_and_chunk(batch_ids, max_patches)
                batch_ids = []
                batch_patches = 0
            batch_ids.append(doc_id)
            batch_patches += estimate
        if batch_ids:
            yield from self._load_and_chunk(batch_ids, max_patches)

    def _load_and_chunk(self, batch_ids: list[str], max_patches: int) -> Iterator[tuple]:
        """Loads one document batch from disk and yields bounded scoring chunks."""

        loaded = self._load_documents(batch_ids)
        mats: list[np.ndarray] = []
        ids: list[str] = []
        metas: list[dict[str, Any]] = []
        patch_counts: list[int] = []
        for doc_id in batch_ids:
            entry = loaded.get(doc_id)
            if entry is None or entry[0].shape[0] == 0:
                continue
            mats.append(entry[0])
            ids.append(doc_id)
            metas.append(entry[1])
            patch_counts.append(entry[2])
        if not mats:
            return
        flat = np.ascontiguousarray(np.vstack(mats), dtype=np.float32)
        counts = np.fromiter((m.shape[0] for m in mats), dtype=np.int64, count=len(mats))
        yield from _subchunk(flat, counts, ids, metas, patch_counts, max_patches, None)

    def _resident_chunks(self, cache: dict[str, Any], max_patches: int) -> Iterator[tuple]:
        """Yields bounded scoring chunks over the resident base plus pending delta.

        The tombstone set masks only the **base** chunks (a replaced/removed base
        document); the pending delta carries the live replacements and is never
        masked, so an updated document stays visible before compaction.
        """

        removed = cache["removed"]
        yield from _subchunk(
            cache["flat"],
            cache["counts"],
            cache["ids"],
            cache["metas"],
            cache["patch_counts"],
            max_patches,
            removed if removed else None,
        )
        pending = cache["pending"]
        if pending:
            pending_flat = np.vstack([entry[0] for entry in pending])
            yield from _subchunk(
                pending_flat,
                np.fromiter(
                    (entry[0].shape[0] for entry in pending),
                    dtype=np.int64,
                    count=len(pending),
                ),
                [entry[1] for entry in pending],
                [entry[2] for entry in pending],
                [entry[3] for entry in pending],
                max_patches,
                None,
            )

    def _topk_from_chunks(
        self,
        query_matrix: np.ndarray,
        chunks: Iterator[tuple],
        k: int,
        prefer_native: bool,
    ) -> list[LodeLateInteractionHit]:
        """Returns the global top-``k`` (score desc, id asc) across the chunks.

        The running top-``k`` is merged chunk by chunk and each chunk's score
        block is discarded after merging, so retained memory is O(k) regardless of
        corpus size. Each chunk carries its own ``skip`` set (resident base
        tombstones); pending and disk chunks carry none.
        """

        best: list[tuple[float, str, dict[str, Any], int]] = []
        for flat, counts, ids, metas, patch_counts, skip in chunks:
            if len(ids) == 0:
                continue
            scores = _maxsim_scores_flat(
                query_matrix, flat, counts, prefer_native=prefer_native
            )
            best = _merge_topk(best, scores, ids, metas, patch_counts, k, skip)
        return [
            LodeLateInteractionHit(score=s, id=i, metadata=m, patch_count=p)
            for s, i, m, p in best
        ]

    def _topk_from_chunks_multi(
        self,
        queries_concat: np.ndarray,
        query_offsets: list[tuple[int, int]],
        chunks: Iterator[tuple],
        k: int,
    ) -> list[list[LodeLateInteractionHit]]:
        """Batched counterpart of :meth:`_topk_from_chunks` -- top-``k`` per query.

        Each chunk is scored once for the whole batch with one GEMM, then merged
        into per-query running top-``k`` lists; the ``(n_queries, n_docs)`` score
        block is discarded after the chunk, so retained memory is O(n_queries * k)
        rather than O(n_queries * total_docs). Each chunk's own ``skip`` set masks
        only its rows (resident base tombstones), not pending/disk rows.
        """

        best: list[list[tuple[float, str, dict[str, Any], int]]] = [
            [] for _ in query_offsets
        ]
        for flat, counts, ids, metas, patch_counts, skip in chunks:
            if len(ids) == 0:
                continue
            block = _maxsim_scores_flat_multi(queries_concat, query_offsets, flat, counts)
            for query_index in range(len(query_offsets)):
                best[query_index] = _merge_topk(
                    best[query_index], block[query_index], ids, metas, patch_counts, k, skip
                )
        return [
            [
                LodeLateInteractionHit(score=s, id=i, metadata=m, patch_count=p)
                for s, i, m, p in bucket
            ]
            for bucket in best
        ]

    def _resident_cache_get(self) -> dict[str, Any] | None:
        """Returns the resident cache, building it if needed; ``None`` if over budget.

        Must be called with ``self._cache_lock`` held (callers do). An index known
        to be over budget returns ``None`` without re-enumerating/rebuilding.
        """

        cache = self._resident_cache
        if isinstance(cache, dict):
            return cache
        if cache is _RESIDENT_OVER_BUDGET:
            return None
        built = self._build_resident_cache()
        self._resident_cache = built if built is not None else _RESIDENT_OVER_BUDGET
        return built

    def _resident_snapshot(self) -> dict[str, Any] | None:
        """Returns a consistent, scan-safe snapshot of the resident cache, or ``None``.

        Snapshots are consistent with concurrent mutations: the large arrays/lists
        are only ever replaced (never mutated in place), so capturing their
        references under the lock is safe, while the in-place-mutated ``pending``
        and ``removed`` are copied. The caller then scores the snapshot without the
        lock, so a query runs concurrently with adds/removes on a shared handle.

        The fast path (cache already built) takes only ``_cache_lock``. Building
        takes ``_write_lock`` first so the from-disk build is ordered against
        writers -- a row committed before the build is captured by the build, one
        committed after is folded by that writer's note, never both. Returns
        ``None`` when there is no resident cache to scan (``resident=False`` or
        over budget).
        """

        if self.resident is False:
            return None
        if isinstance(self._resident_cache, dict):
            with self._cache_lock:
                cache = self._resident_cache
                if isinstance(cache, dict):
                    return _snapshot_cache(cache)
        if self._resident_cache is _RESIDENT_OVER_BUDGET:
            return None
        with self._write_lock, self._cache_lock:
            cache = self._resident_cache_get()
            return None if cache is None else _snapshot_cache(cache)

    def _build_resident_cache(self) -> dict[str, Any] | None:
        """Builds the in-memory matrix from the stored documents.

        Returns ``None`` when the corpus exceeds ``resident_max_bytes`` under
        ``resident="auto"`` (so the caller streams instead), or an empty cache for
        a no-document index.
        """

        resident_dtype = self._resident_dtype()
        records = self._db.list_documents()  # [] for an empty store; real errors raise
        if not records:
            return _empty_resident_cache(self.dim, resident_dtype)
        records.sort(key=lambda record: record["id"])
        total_patches = sum(
            _patch_count_from_metadata(record.get("metadata", {})) for record in records
        )
        itemsize = np.dtype(resident_dtype).itemsize
        if (
            self.resident == "auto"
            and total_patches * self.dim * itemsize > self.resident_max_bytes
        ):
            return None

        ids = [record["id"] for record in records]
        texts = self._db.get_texts(ids)
        kept_ids: list[str] = []
        mats: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        patch_counts: list[int] = []
        for record in records:
            blob = texts.get(record["id"])
            if blob is None:
                continue
            meta = record.get("metadata", {})
            row_dtype = str(meta.get(_DTYPE_KEY, "float32"))
            matrix = _decode_matrix(blob, row_dtype, self.dim)
            kept_ids.append(record["id"])
            mats.append(matrix.astype(resident_dtype, copy=False))
            metas.append(_strip_internal_metadata(meta))
            patch_counts.append(_patch_count_from_metadata(meta) or matrix.shape[0])
        if not mats:
            return _empty_resident_cache(self.dim, resident_dtype)
        return {
            "ids": kept_ids,
            "flat": np.ascontiguousarray(np.vstack(mats), dtype=resident_dtype),
            "counts": np.fromiter(
                (m.shape[0] for m in mats), dtype=np.int64, count=len(mats)
            ),
            "metas": metas,
            "patch_counts": patch_counts,
            "base_ids_set": set(kept_ids),
            "base_pc_by_id": dict(zip(kept_ids, patch_counts, strict=True)),
            "pending": [],
            "pending_ids": set(),
            "pending_patches": 0,
            "removed": set(),
            "removed_patches": 0,
        }


# -- module helpers ---------------------------------------------------------


def _read_li_config(config_path: Path) -> str | None:
    """Returns the index's persisted storage precision, or ``None`` if unset.

    A missing sidecar means "no stored precision" (a brand-new index, or one
    written before this field existed) and returns ``None``. A sidecar that is
    *present* but unparseable or carries an unknown precision is treated as
    corruption and raises, rather than silently defaulting the index to a
    different precision -- the file is written atomically, so a torn write never
    produces a present-but-partial file in normal operation.
    """

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
    """Atomically persists the index's storage precision to its config sidecar.

    Writes a sibling temp file and ``durable_replace``s it into place (the same
    durable-write primitive the engine uses), so a crash mid-write leaves either
    the old config or the new one, never a torn file. ``fsync`` matches the DB's
    durability mode so the marker is power-loss durable when the DB's files are.
    """

    tmp = config_path.with_name(config_path.name + ".tmp")
    tmp.write_text(
        json.dumps({"version": _CONFIG_VERSION, "storage": storage}),
        encoding="utf-8",
    )
    durable_replace(tmp, config_path, fsync=fsync)


def _snapshot_cache(cache: dict[str, Any]) -> dict[str, Any]:
    """Captures a scan-safe snapshot of a resident cache (caller holds the lock).

    Base arrays/lists are referenced (they are only ever replaced, not mutated in
    place); the in-place-mutated pending delta and tombstone set are copied.
    """

    return {
        "flat": cache["flat"],
        "counts": cache["counts"],
        "ids": cache["ids"],
        "metas": cache["metas"],
        "patch_counts": cache["patch_counts"],
        "pending": list(cache["pending"]),
        "removed": frozenset(cache["removed"]),
    }


def _empty_resident_cache(dim: int, dtype: type[np.floating]) -> dict[str, Any]:
    """An empty resident cache (a no-document index), so search returns ``[]``."""

    return {
        "ids": [],
        "flat": np.zeros((0, dim), dtype=dtype),
        "counts": np.zeros(0, dtype=np.int64),
        "metas": [],
        "patch_counts": [],
        "base_ids_set": set(),
        "base_pc_by_id": {},
        "pending": [],
        "pending_ids": set(),
        "pending_patches": 0,
        "removed": set(),
        "removed_patches": 0,
    }


def _maxsim(query: np.ndarray, document: np.ndarray) -> float:
    """Computes the MaxSim score of a query matrix against a document matrix.

    ``sum over query tokens of max over doc patches of <q_t, d_p>``. With
    unit-norm rows each dot-product is a cosine similarity. Shapes are
    ``query=(Nq, dim)`` and ``document=(Nd, dim)``.
    """

    sims = query @ document.T
    return float(sims.max(axis=1).sum())


def _maxsim_batch(
    query: np.ndarray,
    documents: list[np.ndarray],
    *,
    prefer_native: bool = False,
) -> np.ndarray:
    """Returns the MaxSim score of ``query`` against each document matrix.

    A convenience over :func:`_maxsim_scores_flat` that packs a list of
    per-document matrices into the flat-plus-counts layout the scorer expects.
    """

    if not documents:
        return np.empty(0, dtype=np.float32)
    flat = np.ascontiguousarray(np.vstack(documents), dtype=np.float32)
    counts = np.fromiter(
        (doc.shape[0] for doc in documents), dtype=np.int64, count=len(documents)
    )
    return _maxsim_scores_flat(query, flat, counts, prefer_native=prefer_native)


def _maxsim_scores_flat(
    query: np.ndarray,
    flat: np.ndarray,
    counts: np.ndarray,
    *,
    prefer_native: bool = False,
) -> np.ndarray:
    """MaxSim of ``query`` against documents packed as one ``(total_patches, dim)``
    float32 matrix partitioned by ``counts`` (patches per document, in order).

    The native path hands the buffers to the Rust kernel. The numpy path is a
    single ``query @ flat.T`` GEMM followed by a segmented max
    (``np.maximum.reduceat``) summed over query tokens -- vectorized across all
    documents. Both return identical scores. Documents are assumed non-empty
    (``counts >= 1``), which the callers guarantee.
    """

    n_docs = int(counts.shape[0])
    if n_docs == 0:
        return np.empty(0, dtype=np.float32)
    native = _resolve_native_maxsim() if prefer_native else None
    if native is not None:
        query_c = np.ascontiguousarray(query, dtype=np.float32)
        flat_c = np.ascontiguousarray(flat, dtype=np.float32)
        return np.asarray(native(query_c, flat_c, counts), dtype=np.float32)
    sims = np.ascontiguousarray(query, dtype=np.float32) @ flat.T
    starts = np.empty(n_docs, dtype=np.intp)
    starts[0] = 0
    if n_docs > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    segment_max = np.maximum.reduceat(sims, starts, axis=1)
    return segment_max.sum(axis=0).astype(np.float32)


def _merge_topk(
    current: list[tuple[float, str, dict[str, Any], int]],
    scores_row: np.ndarray,
    ids: list[str],
    metas: list[dict[str, Any]],
    patch_counts: list[int],
    k: int,
    skip: set[str] | None,
) -> list[tuple[float, str, dict[str, Any], int]]:
    """Merges one chunk's scores into a running top-``k`` (score desc, id asc).

    ``current`` is the running top-``k`` (already sorted, length <= k). Rather than
    building a Python tuple per document and sorting all of them -- O(n_docs) tuples
    and O(n_docs log n_docs) per chunk -- this selects the chunk's top-``k`` with
    ``np.partition`` and materializes a tuple only for the few candidates at or
    above the k-th score. Selecting *all* rows tied at the k-th score keeps the
    ``(-score, id)`` tie-break exact and chunk-order independent. The chunk's score
    row is not retained, so a caller that iterates many chunks keeps only O(k)
    state.
    """

    row = scores_row
    index_map = None  # maps positions in `row` back to original chunk indices
    if skip:
        keep = [i for i in range(len(ids)) if ids[i] not in skip]
        if not keep:
            return current
        row = scores_row[keep]
        index_map = keep
    n = row.shape[0]
    if n == 0:
        return current
    depth = min(k, n)
    if depth < n:
        # k-th largest score; take every row >= it (ties included) as candidates.
        threshold = np.partition(row, n - depth)[n - depth]
        selected = np.flatnonzero(row >= threshold)
    else:
        selected = np.arange(n)
    candidates = list(current)
    for position in selected.tolist():
        index = index_map[position] if index_map is not None else position
        candidates.append(
            (float(scores_row[index]), ids[index], metas[index], int(patch_counts[index]))
        )
    candidates.sort(key=lambda item: (-item[0], item[1]))
    del candidates[k:]
    return candidates


def _subchunk(
    flat: np.ndarray,
    counts: np.ndarray,
    ids: list[str],
    metas: list[dict[str, Any]],
    patch_counts: list[int],
    max_patches: int,
    skip: frozenset[str] | None,
) -> Iterator[tuple]:
    """Splits one (flat, counts, ids, ...) group into chunks of <= max_patches.

    Each emitted chunk is ``(flat, counts, ids, metas, patch_counts, skip)``; its
    patch rows are upcast to float32 for the GEMM, so the transient score buffer
    (``query_tokens x chunk_patches``) and the upcast buffer both stay bounded
    regardless of corpus size or query-batch width. ``skip`` travels with the chunk
    so a tombstone set masks only the rows it belongs to (resident base rows), not
    pending replacements or disk rows.
    """

    n_docs = len(ids)
    doc = 0
    row = 0
    while doc < n_docs:
        start = doc
        patches = 0
        while doc < n_docs and (patches == 0 or patches + int(counts[doc]) <= max_patches):
            patches += int(counts[doc])
            doc += 1
        yield (
            np.ascontiguousarray(flat[row : row + patches], dtype=np.float32),
            counts[start:doc],
            ids[start:doc],
            metas[start:doc],
            patch_counts[start:doc],
            skip,
        )
        row += patches


def _maxsim_scores_flat_multi(
    queries: np.ndarray,
    query_offsets: list[tuple[int, int]],
    flat: np.ndarray,
    counts: np.ndarray,
) -> np.ndarray:
    """Batched MaxSim: scores several queries against the same packed documents.

    ``queries`` is every query's tokens concatenated into one ``(total_tokens,
    dim)`` matrix, with ``query_offsets[i] = (start, end)`` the row range of query
    ``i``. Returns an ``(n_queries, n_docs)`` score matrix from a single
    ``queries @ flat.T`` GEMM plus one segmented max -- so a batch of queries costs
    one GEMM instead of one per query.
    """

    n_docs = int(counts.shape[0])
    if n_docs == 0:
        return np.zeros((len(query_offsets), 0), dtype=np.float32)
    sims = np.ascontiguousarray(queries, dtype=np.float32) @ flat.T
    starts = np.empty(n_docs, dtype=np.intp)
    starts[0] = 0
    if n_docs > 1:
        np.cumsum(counts[:-1], out=starts[1:])
    segment_max = np.maximum.reduceat(sims, starts, axis=1)  # (total_tokens, n_docs)
    out = np.empty((len(query_offsets), n_docs), dtype=np.float32)
    for index, (start, end) in enumerate(query_offsets):
        out[index] = segment_max[start:end].sum(axis=0)
    return out


def _as_matrix(matrix: Any, dim: int, *, normalize: bool) -> np.ndarray:
    """Coerces a patch/token matrix to a finite float32 ``(rows, dim)`` array.

    Optionally L2-normalizes each row so dot-products are cosine similarities.
    Zero rows are rejected when normalizing (they have no direction).
    """

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
    """Returns the document's mean-pooled unit vector (the stored row vector).

    Used only as the row's index vector; retrieval scores the full patch matrix.
    A zero mean (degenerate) falls back to the first patch so the vector is valid.
    """

    pooled = matrix.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm == 0.0:
        return np.ascontiguousarray(matrix[0], dtype=np.float32)
    return np.ascontiguousarray(pooled / norm, dtype=np.float32)


def _encode_matrix(matrix: np.ndarray, storage: str) -> str:
    """Serializes a ``(num_patches, dim)`` matrix to a base64 blob at ``storage``."""

    if storage == "float32":
        buffer = np.ascontiguousarray(matrix, dtype="<f4").tobytes()
    elif storage == "float16":
        buffer = np.ascontiguousarray(matrix, dtype="<f2").tobytes()
    elif storage == "int8":
        # Per-vector symmetric quantization: scale each patch by its max-abs so the
        # int8 range is used fully, then store the f32 scales followed by the codes.
        scales = np.abs(matrix).max(axis=1).astype("<f4")
        safe = np.where(scales == 0.0, 1.0, scales).astype(np.float32)
        codes = np.clip(np.round(matrix / safe[:, None] * 127.0), -127, 127).astype(np.int8)
        buffer = scales.tobytes() + codes.tobytes()
    else:  # pragma: no cover - guarded at construction
        raise ValueError(f"unknown storage {storage!r}")
    return base64.b64encode(buffer).decode("ascii")


def _decode_matrix(blob: str, storage: str, dim: int) -> np.ndarray:
    """Decodes a base64 blob back to a float32 ``(num_patches, dim)`` matrix."""

    raw = base64.b64decode(blob.encode("ascii"))
    if storage == "float32":
        return np.array(np.frombuffer(raw, dtype="<f4"), dtype=np.float32).reshape(-1, dim)
    if storage == "float16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(-1, dim)
    if storage == "int8":
        n = len(raw) // (4 + dim)
        scales = np.frombuffer(raw[: n * 4], dtype="<f4").astype(np.float32)
        codes = np.frombuffer(raw[n * 4 :], dtype=np.int8).reshape(n, dim).astype(np.float32)
        return codes * (scales[:, None] / 127.0)
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
    return dict(metadata)


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
