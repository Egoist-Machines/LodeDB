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
``float16`` (default, ~exact and half the size of float32), ``float32`` (bit
exact), or ``int8`` (a per-vector-scaled quantization, ~4x smaller).

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
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lodedb.engine.index import EngineError
from lodedb.local.db import LodeDB

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

# Cap the transient float32 work buffer when scoring a resident or streamed scan,
# so peak memory stays bounded regardless of corpus size.
_SCORE_CHUNK_BYTES = 64 * 1024 * 1024
# Documents read per batch from disk on the streaming / filtered paths.
_LOAD_BATCH_DOCS = 128


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
        storage: str = "float16",
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

        ``storage`` is the patch-matrix precision: ``"float16"`` (default,
        near-exact at half the size of float32), ``"float32"`` (bit exact), or
        ``"int8"`` (per-vector-scaled, ~4x smaller, a small recall cost). Each
        document records the precision it was written at, so reopening decodes
        correctly regardless of this argument. ``resident`` controls the default
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

        if int(dim) <= 0:
            raise ValueError("dim must be a positive integer")
        if storage not in _STORAGE_CHOICES:
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
        self.storage = storage
        self.scoring = scoring
        self.resident = resident
        self.resident_max_bytes = int(resident_max_bytes)
        self.candidate_depth = int(candidate_depth)
        self.encoder = encoder
        self.read_only = bool(read_only)
        # In-memory serving cache (all patches as one compact matrix); built lazily
        # on the first eligible search and invalidated on every write.
        self._resident_cache: dict[str, Any] | None = None
        lodedb_kwargs.pop("store_text", None)
        self._db = LodeDB.open_vector_store(
            path,
            vector_dim=self.dim,
            bit_width=int(bit_width),
            store_text=True,
            read_only=self.read_only,
            **lodedb_kwargs,
        )

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

        self._db.add_vectors_many([self._build_row(id, patches, metadata, normalize)])
        self._resident_cache = None  # serving cache is now stale
        return _require_doc_id(id)

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
            row = self._build_row(
                document.get("id"), patches, document.get("metadata"), normalize
            )
            rows.append(row)
            ids.append(row["id"])
        if rows:
            self._db.add_vectors_many(rows, normalize=False)
            self._resident_cache = None
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

        removed = self._db.remove(_require_doc_id(id))
        if removed:
            self._resident_cache = None
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

        if filter is not None:
            try:
                doc_ids = [
                    record["id"] for record in self._db.list_documents(filter=dict(filter))
                ]
            except EngineError:
                return []
            return self._topk_from_chunks(
                query_matrix, self._disk_chunks(doc_ids), int(k), prefer_native
            )

        if self.resident is not False:
            cache = self._resident_cache_get()
            if cache is not None:
                return self._topk_from_chunks(
                    query_matrix, self._resident_chunks(cache), int(k), prefer_native
                )

        # Over the resident budget (or resident=False): stream from disk, exact.
        return self._topk_from_chunks(
            query_matrix, self._disk_chunks(self._all_document_ids()), int(k), prefer_native
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

    def _build_row(
        self,
        id: Any,
        patches: Any,
        metadata: Mapping[str, Any] | None,
        normalize: bool,
    ) -> dict[str, Any]:
        """Builds the single engine row (pooled vector + encoded matrix) for a doc."""

        document_id = _require_doc_id(id)
        matrix = _as_matrix(patches, self.dim, normalize=normalize)
        row_meta = _coerce_user_metadata(metadata)
        row_meta[_PATCH_COUNT_KEY] = str(matrix.shape[0])
        row_meta[_DTYPE_KEY] = self.storage
        return {
            "vector": _pool(matrix).tolist(),
            "id": document_id,
            "metadata": row_meta,
            "text": _encode_matrix(matrix, self.storage),
        }

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

    def _all_document_ids(self) -> list[str]:
        """Returns every document id (sorted for determinism)."""

        try:
            records = self._db.list_documents()
        except EngineError:
            return []
        return sorted(record["id"] for record in records)

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
            matrix = _decode_matrix(blob, row_dtype, self.dim)
            patch_count = _patch_count_from_metadata(meta) or matrix.shape[0]
            out[doc_id] = (matrix, _strip_internal_metadata(meta), patch_count)
        return out

    def _disk_chunks(self, document_ids: list[str]) -> Iterator[tuple]:
        """Yields scoring chunks read from disk in bounded batches of documents."""

        for start in range(0, len(document_ids), _LOAD_BATCH_DOCS):
            batch = document_ids[start : start + _LOAD_BATCH_DOCS]
            loaded = self._load_documents(batch)
            mats: list[np.ndarray] = []
            ids: list[str] = []
            metas: list[dict[str, Any]] = []
            patch_counts: list[int] = []
            for doc_id in batch:
                entry = loaded.get(doc_id)
                if entry is None or entry[0].shape[0] == 0:
                    continue
                mats.append(entry[0])
                ids.append(doc_id)
                metas.append(entry[1])
                patch_counts.append(entry[2])
            if not mats:
                continue
            flat = np.ascontiguousarray(np.vstack(mats), dtype=np.float32)
            counts = np.fromiter((m.shape[0] for m in mats), dtype=np.int64, count=len(mats))
            yield flat, counts, ids, metas, patch_counts

    def _resident_chunks(self, cache: dict[str, Any]) -> Iterator[tuple]:
        """Yields scoring chunks by slicing the resident matrix to a patch budget.

        Each chunk's patch rows are upcast to float32 just for the GEMM, so the
        transient float32 buffer stays bounded even when the resident matrix is a
        compact dtype and the corpus is large.
        """

        flat = cache["flat"]
        counts = cache["counts"]
        ids = cache["ids"]
        metas = cache["metas"]
        patch_counts = cache["patch_counts"]
        budget = max(1, _SCORE_CHUNK_BYTES // (self.dim * 4))
        n_docs = len(ids)
        doc = 0
        row = 0
        while doc < n_docs:
            start_doc = doc
            patches = 0
            while doc < n_docs and (patches == 0 or patches + int(counts[doc]) <= budget):
                patches += int(counts[doc])
                doc += 1
            sub = np.ascontiguousarray(flat[row : row + patches], dtype=np.float32)
            yield (
                sub,
                counts[start_doc:doc],
                ids[start_doc:doc],
                metas[start_doc:doc],
                patch_counts[start_doc:doc],
            )
            row += patches

    def _topk_from_chunks(
        self,
        query_matrix: np.ndarray,
        chunks: Iterator[tuple],
        k: int,
        prefer_native: bool,
    ) -> list[LodeLateInteractionHit]:
        """Scores every chunk and returns the global top-``k`` (score desc, id asc)."""

        collected: list[tuple[float, str, dict[str, Any], int]] = []
        for flat, counts, ids, metas, patch_counts in chunks:
            if len(ids) == 0:
                continue
            scores = _maxsim_scores_flat(
                query_matrix, flat, counts, prefer_native=prefer_native
            )
            for i in range(len(ids)):
                collected.append((float(scores[i]), ids[i], metas[i], int(patch_counts[i])))
        collected.sort(key=lambda item: (-item[0], item[1]))
        return [
            LodeLateInteractionHit(score=s, id=i, metadata=m, patch_count=p)
            for s, i, m, p in collected[:k]
        ]

    def _resident_cache_get(self) -> dict[str, Any] | None:
        """Returns the resident cache, building it if needed; ``None`` if over budget."""

        if self._resident_cache is not None:
            return self._resident_cache
        cache = self._build_resident_cache()
        self._resident_cache = cache
        return cache

    def _build_resident_cache(self) -> dict[str, Any] | None:
        """Builds the in-memory matrix from the stored documents.

        Returns ``None`` when the corpus exceeds ``resident_max_bytes`` under
        ``resident="auto"`` (so the caller streams instead), or an empty cache for
        a no-document index.
        """

        resident_dtype = self._resident_dtype()
        try:
            records = self._db.list_documents()
        except EngineError:
            return _empty_resident_cache(self.dim, resident_dtype)
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
        }


# -- module helpers ---------------------------------------------------------


def _empty_resident_cache(dim: int, dtype: type[np.floating]) -> dict[str, Any]:
    """An empty resident cache (a no-document index), so search returns ``[]``."""

    return {
        "ids": [],
        "flat": np.zeros((0, dim), dtype=dtype),
        "counts": np.zeros(0, dtype=np.int64),
        "metas": [],
        "patch_counts": [],
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
