"""Late-interaction (multi-vector / MaxSim) retrieval -- the issue #25 stage-1 prototype.

ColBERT-style late interaction, and its visual-document descendants (ColPali /
ColQwen), represent each document -- or each page rendered as an image -- as a
*set* of token/patch vectors rather than a single pooled vector, and score a
query against a document with **MaxSim**::

    score(query, doc) = sum over query tokens t of  max over doc patches p of  <q_t, d_p>

This does not fit LodeDB's one-vector-per-id TurboVec core, so it is built here
as a pure-SDK prototype on top of a bring-your-own-vectors index
(:meth:`LodeDB.open_vector_store`), with **no engine change** -- exactly the
staged plan in issue #25. Each document's patches are stored as ordinary vector
rows keyed ``<doc_id>#<NNNNN>`` carrying a ``parent_id`` in metadata, and the
full-precision patch vectors are kept verbatim (float32, base64, in the per-row
text sidecar) so MaxSim is always computed at **exact** full precision, never from
the quantized codes. There are two retrieval paths (:meth:`search` picks one;
both return identical scores):

1. *Resident* (default for an unfiltered query within ``resident_max_bytes``) --
   every patch is held in one in-memory matrix and the whole corpus is scored in a
   single GEMM plus a segmented max. No candidate-generation scan and no
   per-candidate read-back, so it returns the true top-``k`` (no recall loss) at a
   few milliseconds on thousands of pages.
2. *Indexed* (a filtered query, a corpus over the resident budget, or
   ``resident=False``) -- the two-stage approach the issue prescribes: a batched
   any-patch scan to depth ``candidate_depth`` gathers candidate documents (the
   filter is pushed engine-side), then exact MaxSim rescores them.

The exact MaxSim itself defaults to numpy (a ``query @ patches.T`` BLAS GEMM); a
native TurboVec ``maxsim_scores`` Rust kernel is also available
(``scoring="native"``) for builds without a fast BLAS. Both return identical
scores.

The page/token encoder is **bring-your-own** (ColPali / ColQwen weights are
multi-GB and out of scope to bundle): pass precomputed patch matrices to
:meth:`add_document` / :meth:`search`, or an optional ``encoder`` exposing
``encode_documents`` / ``encode_queries``.

Footprint note: a document contributes one stored row per patch, so a page with
~1000 patches is ~1000 rows (plus the float32 patch sidecar the exact rescore
reads back). That is the known cost of late interaction; native quantized
multi-vector *storage* in the TurboVec core (to shrink that footprint) remains the
deferred half of the stage-3 track (issue #25), while the MaxSim scoring kernel is
already native here.
"""

from __future__ import annotations

import base64
import importlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lodedb.engine.index import EngineError
from lodedb.local.db import LodeDB

# The bundled TurboVec extension exposes a native MaxSim kernel (`maxsim_scores`)
# that scores the candidate set in parallel Rust, replacing the per-candidate
# Python loop. It is resolved once and cached; a build that predates the kernel
# (or a stock standalone `turbovec`) simply falls back to the numpy path, so the
# SDK never hard-depends on the kernel being present.
_TURBOVEC_PACKAGE_NAMES = ("lodedb._turbovec", "turbovec")
_UNRESOLVED = object()  # distinct from None, which means "looked up, not present"
_native_maxsim: Callable[..., Any] | None | object = _UNRESOLVED


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

# Separator between a document id and its per-patch suffix in a stored row id.
# Mirrors the ``<doc_id>#patch_NNNN`` shape suggested in issue #25, kept compact.
_PATCH_SEP = "#"
# Zero-padding width for the patch index in a row id, so row ids sort naturally;
# five figures comfortably covers a rendered page's ~1000 patches with headroom.
_PATCH_ID_WIDTH = 5
# Metadata key holding the owning document id on every patch row. Reserved: a
# caller-supplied metadata mapping may not set it.
_PARENT_KEY = "parent_id"
# Metadata key holding the document's patch count, stamped on patch 0 so a
# document can be counted without scanning all of its rows. Also reserved.
_PATCH_COUNT_KEY = "patch_count"

_RESERVED_METADATA_KEYS = frozenset({_PARENT_KEY, _PATCH_COUNT_KEY})


class LodeLateInteractionHit:
    """One late-interaction result: ``(score, id, metadata)`` for a *document*.

    ``score`` is the MaxSim score (sum over query tokens of the max patch
    similarity), ``id`` is the document (parent) id, and ``metadata`` is the user
    metadata supplied to :meth:`LodeLateInteractionIndex.add_document` (the
    internal ``parent_id`` / ``patch_count`` keys are stripped). Unpacks like a
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

    All storage and scan reuse the embedded :class:`~lodedb.local.db.LodeDB`
    vector-only index unchanged: each patch is one row keyed ``<id>#NNNNN`` with a
    ``parent_id`` in metadata, and the float32 patch vectors are retained (base64,
    in the per-row text sidecar) so the exact MaxSim score can be recomputed at
    query time over the candidate documents. Data stays on local disk; nothing is
    sent anywhere.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        dim: int,
        encoder: Any | None = None,
        bit_width: int = 4,
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
        document patches and query tokens must share it. ``encoder`` is an
        optional bring-your-own page/token encoder exposing
        ``encode_documents(list[...]) -> list[2-D matrix]`` and
        ``encode_queries(list[...]) -> list[2-D matrix]``; it is only used by
        :meth:`add_texts` / :meth:`search_text` and is never required for the
        precomputed-matrix API. ``candidate_depth`` is the per-query-token
        any-patch search depth used to gather candidates on the *indexed* path only
        (higher = better recall there, more rescoring work; the resident path is
        exhaustive and always exact). ``resident`` controls the default fast path:
        ``"auto"`` (default) uses the in-memory exact scan for unfiltered queries
        when the patch corpus fits ``resident_max_bytes`` (default 512 MB) and
        falls back to the indexed path otherwise; ``True`` always uses it; ``False``
        always uses the indexed path. ``scoring`` selects the exact-MaxSim backend:
        ``"numpy"`` (default, BLAS GEMM -- fastest on builds with an optimized BLAS,
        always available) or ``"native"`` (the TurboVec ``maxsim_scores`` Rust
        kernel, for builds without a fast BLAS; falls back to numpy if absent). Both
        return identical scores. ``read_only=True`` opens a non-mutating reader and
        requires the path to exist. ``bit_width`` and any extra ``lodedb_kwargs``
        (e.g. ``durability=``, ``commit_mode=``) are forwarded to the underlying
        vector-only :class:`LodeDB`. The patch text sidecar that holds the vectors
        is always retained, so ``store_text`` may not be set ``False``.
        """

        if int(dim) <= 0:
            raise ValueError("dim must be a positive integer")
        if int(candidate_depth) <= 0:
            raise ValueError("candidate_depth must be a positive integer")
        if scoring not in ("numpy", "native"):
            raise ValueError("scoring must be 'numpy' or 'native'")
        if resident not in (True, False, "auto"):
            raise ValueError("resident must be True, False, or 'auto'")
        if lodedb_kwargs.get("store_text") is False:
            raise ValueError(
                "LodeLateInteractionIndex stores patch vectors in the text sidecar; "
                "store_text must remain True"
            )
        self.dim = int(dim)
        self.scoring = scoring
        self.resident = resident
        self.resident_max_bytes = int(resident_max_bytes)
        self.encoder = encoder
        self.candidate_depth = int(candidate_depth)
        self.read_only = bool(read_only)
        # In-memory exact-MaxSim serving cache (all patches as one float32 matrix);
        # built lazily on first eligible search and invalidated on every write.
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
        """Stores one document as its set of patch vectors; returns its id.

        ``patches`` is a 2-D ``(num_patches, dim)`` matrix (a numpy array or a
        sequence of equal-length rows). Each patch becomes one row keyed
        ``<id>#NNNNN`` carrying ``parent_id=<id>`` (plus your ``metadata``) and is
        L2-normalized by default so MaxSim dot-products are cosine similarities.
        Re-adding an existing id first removes its old patches, so a document can
        be replaced even if its patch count changed. The whole document commits in
        one atomic batch.
        """

        document_id = _require_doc_id(id)
        matrix = _as_matrix(patches, self.dim, normalize=normalize)
        user_meta = _coerce_user_metadata(metadata)
        # Replace cleanly: drop any prior patches for this id so a shorter re-add
        # cannot leave stale tail rows behind. (Raises ReadOnlyError on a reader.)
        self._remove_patches(document_id)
        patch_count = matrix.shape[0]
        rows: list[dict[str, Any]] = []
        for index in range(patch_count):
            row_meta = dict(user_meta)
            row_meta[_PARENT_KEY] = document_id
            if index == 0:
                # Stamp the count on patch 0 only, so count() can tally documents
                # by counting parent-marker rows without enumerating every patch.
                row_meta[_PATCH_COUNT_KEY] = str(patch_count)
            rows.append(
                {
                    "vector": matrix[index].tolist(),
                    "id": _patch_id(document_id, index),
                    "metadata": row_meta,
                    # Patch vectors are unit-norm above; persist verbatim so the
                    # exact MaxSim recompute reads back what was scored against.
                    "text": _encode_vector(matrix[index]),
                }
            )
        # Vectors are pre-normalized above; do not normalize twice.
        self._db.add_vectors_many(rows, normalize=False)
        self._resident_cache = None  # serving cache is now stale
        return document_id

    def add_documents(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> list[str]:
        """Adds a batch of ``{"id", "patches", "metadata"?}`` documents.

        Each document is expanded to its patch rows and committed. Returns the ids
        in input order.
        """

        ids: list[str] = []
        for document in documents:
            if not isinstance(document, Mapping):
                raise ValueError("each document must be a mapping")
            patches = document.get("patches")
            if patches is None:
                raise ValueError("each document needs a 'patches' matrix")
            ids.append(
                self.add_document(
                    document.get("id"),
                    patches,
                    metadata=document.get("metadata"),
                    normalize=normalize,
                )
            )
        return ids

    def add_texts(
        self,
        documents: Sequence[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> list[str]:
        """Encodes documents with the bring-your-own ``encoder`` and stores them.

        Convenience over :meth:`add_documents` for when an ``encoder`` was
        supplied: each item is ``{"id", "content", "metadata"?}`` and
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
        """Removes a document and all its patches; True if any patch existed."""

        return self._remove_patches(_require_doc_id(id)) > 0

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
        """Returns the top-``k`` documents by MaxSim for a multi-vector query.

        ``query`` is a 2-D ``(num_query_tokens, dim)`` matrix. There are two
        retrieval paths, chosen automatically:

        - **Resident** (default for an unfiltered query when the patch corpus fits
          ``resident_max_bytes``): the full exact MaxSim is computed against every
          document in one pass over an in-memory patch matrix. This skips both the
          per-query-token quantized scan and the per-candidate read-back -- the two
          costs that dominate query time -- and returns the true top-``k`` (no
          candidate-recall loss).
        - **Indexed** (used when a ``filter`` is given, or the corpus exceeds the
          resident budget, or ``resident=False``): the two-stage path -- a batched
          any-patch search to depth ``candidate_depth`` gathers candidate documents
          (the ``filter`` is pushed engine-side), then exact MaxSim rescores the
          candidates.

        ``candidate_depth`` overrides the index default for the indexed path.
        ``filter`` takes the same exact-match-or-predicate grammar as
        :meth:`LodeDB.search`. Query tokens are L2-normalized by default to match
        stored patches. Both paths return identical scores.
        """

        if int(k) <= 0:
            raise ValueError("k must be positive")
        depth = self.candidate_depth if candidate_depth is None else int(candidate_depth)
        if depth <= 0:
            raise ValueError("candidate_depth must be positive")
        query_matrix = _as_matrix(query, self.dim, normalize=normalize)

        # Prefer the resident exact scan for unfiltered queries (a filter is pushed
        # engine-side on the indexed path instead). Falls through to indexed when
        # the corpus is over the resident budget. An empty index yields an empty
        # resident cache, so this also handles the no-documents case without the
        # (surprisingly costly) stats-based count() on the hot path.
        if filter is None and self.resident is not False:
            cache = self._resident_cache_get()
            if cache is not None:
                return self._search_resident(query_matrix, cache, int(k))

        return self._search_indexed(query_matrix, int(k), depth, filter)

    def _search_resident(
        self,
        query_matrix: np.ndarray,
        cache: dict[str, Any],
        k: int,
    ) -> list[LodeLateInteractionHit]:
        """Exact MaxSim over the whole resident patch matrix; returns top-``k``."""

        scores = _maxsim_scores_flat(
            query_matrix,
            cache["flat"],
            cache["counts"],
            prefer_native=self.scoring == "native",
        )
        ids = cache["ids"]
        metas = cache["metas"]
        patch_counts = cache["patch_counts"]
        scored = [
            LodeLateInteractionHit(
                score=float(scores[i]),
                id=ids[i],
                metadata=metas[i],
                patch_count=patch_counts[i],
            )
            for i in range(len(ids))
        ]
        # Deterministic order: score desc, then id asc to break ties stably.
        scored.sort(key=lambda hit: (-hit.score, hit.id))
        return scored[:k]

    def _search_indexed(
        self,
        query_matrix: np.ndarray,
        k: int,
        depth: int,
        filter: Mapping[str, Any] | None,
    ) -> list[LodeLateInteractionHit]:
        """Two-stage retrieval: candidate generation then exact MaxSim rescore."""

        # Stage 1: candidate generation. One any-patch sub-query per query token;
        # union the owning documents of the hits. Pass the user filter as-is -- in a
        # dedicated late-interaction index every row is a patch row, so a synthetic
        # "parent_id exists" filter would only add allowlist overhead for no effect.
        token_queries = [query_matrix[i].tolist() for i in range(query_matrix.shape[0])]
        try:
            per_token_hits = self._db.search_many_by_vector(
                token_queries,
                k=depth,
                filter=dict(filter) if filter else None,
                normalize=False,
            )
        except EngineError:
            # An index with no committed patches has no serving snapshot to scan;
            # treat that as no results rather than an error.
            return []
        candidates: list[str] = []
        seen: set[str] = set()
        for hits in per_token_hits:
            for hit in hits:
                parent = hit.metadata.get(_PARENT_KEY)
                if parent and parent not in seen:
                    seen.add(parent)
                    candidates.append(parent)
        if not candidates:
            return []

        # Stage 2: exact MaxSim rescoring over the candidates' stored patches.
        # One batched read of all candidates' patches, then score in one shot.
        loaded_docs = self._load_documents(candidates)
        loaded: list[np.ndarray] = []
        meta_rows: list[tuple[str, dict[str, Any], int]] = []
        for parent in candidates:
            entry = loaded_docs.get(parent)
            if entry is None or entry[0].shape[0] == 0:
                continue
            loaded.append(entry[0])
            meta_rows.append((parent, entry[1], entry[2]))
        if not loaded:
            return []
        scores = _maxsim_batch(
            query_matrix, loaded, prefer_native=self.scoring == "native"
        )
        scored = [
            LodeLateInteractionHit(
                score=float(score), id=parent, metadata=user_meta, patch_count=patch_count
            )
            for score, (parent, user_meta, patch_count) in zip(
                scores, meta_rows, strict=True
            )
        ]
        # Deterministic order: score desc, then id asc to break ties stably.
        scored.sort(key=lambda hit: (-hit.score, hit.id))
        return scored[:k]

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

        Convenience over :meth:`search`: ``encoder.encode_queries([query])`` must
        return one 2-D token matrix. Raises :class:`RuntimeError` with no encoder.
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
        """Returns the number of distinct documents (parents) stored."""

        # Patch 0 of every document carries the patch_count marker, so counting
        # those rows counts documents without enumerating every patch.
        return self._db.count(filter={_PATCH_COUNT_KEY: {"$exists": True}})

    def patch_count(self) -> int:
        """Returns the total number of stored patch rows across all documents."""

        return self._db.count()

    def list_documents(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns one redacted ``{"id", "metadata", "patch_count"}`` per document.

        ``filter`` (the same grammar as :meth:`LodeDB.list_documents`) is matched
        against your document metadata.
        """

        records = self._db.list_documents(filter=_patch_scan_filter(filter))
        parents: dict[str, dict[str, Any]] = {}
        for record in records:
            metadata = record.get("metadata", {})
            if not isinstance(metadata, Mapping):
                continue
            parent = metadata.get(_PARENT_KEY)
            if not isinstance(parent, str) or not parent or parent in parents:
                continue
            parents[parent] = {
                "id": parent,
                "metadata": _strip_internal_metadata(metadata),
                "patch_count": _patch_count_from_metadata(metadata),
            }
        return [parents[parent] for parent in sorted(parents)]

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

    def _remove_patches(self, document_id: str) -> int:
        """Removes every patch row owned by ``document_id``; returns count removed.

        Uses the public per-id ``remove`` (the same pattern the graph layer uses
        for multi-row deletes), so it goes through the engine's read-only guard and
        atomic commit path rather than the private index.
        """

        records = self._db.list_documents(filter={_PARENT_KEY: document_id})
        removed = 0
        for record in records:
            if self._db.remove(record["id"]):
                removed += 1
        if removed:
            self._resident_cache = None  # serving cache is now stale
        return removed

    def _load_document(
        self, document_id: str
    ) -> tuple[np.ndarray | None, dict[str, Any], int]:
        """Loads a document's ``(patch_matrix, user_metadata, patch_count)``.

        Reads the patch rows back via metadata enumeration and decodes each stored
        patch vector from its text-sidecar blob.
        """

        records = self._db.list_documents(filter={_PARENT_KEY: document_id})
        if not records:
            return None, {}, 0
        records.sort(key=lambda record: record["id"])
        texts = self._db.get_texts([record["id"] for record in records])
        vectors: list[np.ndarray] = []
        for record in records:
            blob = texts.get(record["id"])
            if blob is None:
                continue
            vectors.append(_decode_vector(blob, self.dim))
        user_meta = _strip_internal_metadata(records[0].get("metadata", {}))
        patch_count = max(
            (_patch_count_from_metadata(record.get("metadata", {})) for record in records),
            default=len(records),
        )
        if not vectors:
            return None, user_meta, patch_count
        return np.vstack(vectors), user_meta, patch_count

    def _load_documents(
        self, document_ids: list[str]
    ) -> dict[str, tuple[np.ndarray, dict[str, Any], int]]:
        """Batch-loads several documents' ``(patches, metadata, patch_count)``.

        One enumeration and one text read cover every requested document, instead
        of two engine round-trips per document.
        """

        if not document_ids:
            return {}
        records = self._db.list_documents(filter={_PARENT_KEY: {"$in": list(document_ids)}})
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records:
            parent = record.get("metadata", {}).get(_PARENT_KEY)
            if isinstance(parent, str) and parent:
                grouped.setdefault(parent, []).append(record)
        all_ids = [record["id"] for rows in grouped.values() for record in rows]
        texts = self._db.get_texts(all_ids)
        out: dict[str, tuple[np.ndarray, dict[str, Any], int]] = {}
        for parent, rows in grouped.items():
            rows.sort(key=lambda record: record["id"])
            vectors = [
                _decode_vector(texts[record["id"]], self.dim)
                for record in rows
                if texts.get(record["id"]) is not None
            ]
            if not vectors:
                continue
            user_meta = _strip_internal_metadata(rows[0].get("metadata", {}))
            patch_count = max(
                (_patch_count_from_metadata(r.get("metadata", {})) for r in rows),
                default=len(rows),
            )
            out[parent] = (np.vstack(vectors), user_meta, patch_count)
        return out

    def _resident_cache_get(self) -> dict[str, Any] | None:
        """Returns the resident exact-MaxSim cache, building it if needed.

        Returns ``None`` when the corpus exceeds ``resident_max_bytes`` under the
        ``resident="auto"`` policy (so the caller falls back to the indexed path),
        or when the index is empty.
        """

        if self._resident_cache is not None:
            return self._resident_cache
        cache = self._build_resident_cache()
        self._resident_cache = cache
        return cache

    def _build_resident_cache(self) -> dict[str, Any] | None:
        """Builds the in-memory exact-MaxSim cache from the stored patch sidecar.

        Lays every document's patches out as one contiguous float32 matrix with a
        per-document patch-count partition, so a query is one GEMM plus a segmented
        reduction. Honors the ``resident_max_bytes`` budget under ``"auto"``.
        """

        records = self._db.list_documents(filter={_PARENT_KEY: {"$exists": True}})
        if not records:
            return _empty_resident_cache(self.dim)
        order: list[str] = []
        groups: dict[str, dict[str, Any]] = {}
        for record in records:
            metadata = record.get("metadata", {})
            parent = metadata.get(_PARENT_KEY)
            if not isinstance(parent, str) or not parent:
                continue
            group = groups.get(parent)
            if group is None:
                group = {"ids": [], "meta": metadata}
                groups[parent] = group
                order.append(parent)
            group["ids"].append(record["id"])
        total_patches = sum(len(groups[parent]["ids"]) for parent in order)
        if (
            self.resident == "auto"
            and total_patches * self.dim * 4 > self.resident_max_bytes
        ):
            return None  # over budget: caller uses the indexed path instead

        all_ids = [pid for parent in order for pid in groups[parent]["ids"]]
        texts = self._db.get_texts(all_ids)
        ids: list[str] = []
        mats: list[np.ndarray] = []
        metas: list[dict[str, Any]] = []
        patch_counts: list[int] = []
        for parent in order:
            group = groups[parent]
            vectors = [
                _decode_vector(texts[pid], self.dim)
                for pid in sorted(group["ids"])
                if texts.get(pid) is not None
            ]
            if not vectors:
                continue
            ids.append(parent)
            mats.append(np.vstack(vectors).astype(np.float32, copy=False))
            metas.append(_strip_internal_metadata(group["meta"]))
            patch_counts.append(max(_patch_count_from_metadata(group["meta"]), len(vectors)))
        if not mats:
            return _empty_resident_cache(self.dim)
        return {
            "ids": ids,
            "flat": np.ascontiguousarray(np.vstack(mats), dtype=np.float32),
            "counts": np.fromiter(
                (m.shape[0] for m in mats), dtype=np.int64, count=len(mats)
            ),
            "metas": metas,
            "patch_counts": patch_counts,
        }


# -- module helpers ---------------------------------------------------------


def _empty_resident_cache(dim: int) -> dict[str, Any]:
    """An empty resident cache (a no-document index), so search returns ``[]``."""

    return {
        "ids": [],
        "flat": np.zeros((0, dim), dtype=np.float32),
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

    # (Nq, Nd) similarity matrix, then max over doc patches, then sum over tokens.
    sims = query @ document.T
    return float(sims.max(axis=1).sum())


def _maxsim_batch(
    query: np.ndarray,
    documents: list[np.ndarray],
    *,
    prefer_native: bool = False,
) -> np.ndarray:
    """Returns the MaxSim score of ``query`` against each document matrix.

    Two paths return identical scores (parity holds to f32 rounding):

    - ``numpy`` (default): per-document ``query @ doc.T`` via numpy, which on a
      build with an optimized BLAS (e.g. Apple Accelerate, OpenBLAS) is the
      fastest scoring path and needs no compiled kernel.
    - ``native`` (``prefer_native=True``): the TurboVec ``maxsim_scores`` Rust
      kernel (per-document faer GEMM, parallel across documents, GIL released),
      used when the compiled extension provides it.

    numpy is the default because it is consistently fastest for this step on
    common platforms and is always available; the native kernel is exposed for
    builds without a fast BLAS and as the basis for a future native
    multi-vector storage path. Either way, MaxSim scoring is a small fraction of
    query time (candidate generation and patch loading dominate), so the choice
    is not a latency lever today. ``prefer_native`` silently falls back to numpy
    if the kernel is unavailable.
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
    matrix partitioned by ``counts`` (patches per document, in order).

    The native path hands the buffers straight to the Rust kernel. The numpy path
    is a single ``query @ flat.T`` GEMM followed by a segmented max
    (``np.maximum.reduceat``) summed over query tokens -- fully vectorized across
    all documents, no per-document Python loop. Both return identical scores. All
    documents are assumed non-empty (``counts >= 1``), which the callers guarantee.
    """

    n_docs = int(counts.shape[0])
    if n_docs == 0:
        return np.empty(0, dtype=np.float32)
    native = _resolve_native_maxsim() if prefer_native else None
    if native is not None:
        query_c = np.ascontiguousarray(query, dtype=np.float32)
        return np.asarray(native(query_c, flat, counts), dtype=np.float32)
    sims = np.ascontiguousarray(query, dtype=np.float32) @ flat.T
    # Segment boundaries (start column of each document) for reduceat.
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


def _encode_vector(vector: np.ndarray) -> str:
    """Serializes one float32 patch vector to a compact base64 text blob."""

    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def _decode_vector(blob: str, dim: int) -> np.ndarray:
    """Decodes a base64 text blob back to a float32 ``(dim,)`` vector."""

    raw = base64.b64decode(blob.encode("ascii"))
    array = np.frombuffer(raw, dtype="<f4")
    if array.shape[0] != dim:
        raise ValueError(
            f"stored patch vector has dimension {array.shape[0]}, expected {dim}"
        )
    return np.array(array, dtype=np.float32)


def _patch_id(document_id: str, index: int) -> str:
    """Builds the deterministic row id for a document's ``index``-th patch."""

    return f"{document_id}{_PATCH_SEP}{index:0{_PATCH_ID_WIDTH}d}"


def _require_doc_id(value: Any) -> str:
    """Validates and returns a non-empty document id without the patch separator."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("document id must be a non-empty string")
    if _PATCH_SEP in value:
        raise ValueError(f"document id may not contain {_PATCH_SEP!r}")
    return value


def _coerce_user_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validates user metadata and reserves the internal patch keys."""

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
    """Drops the internal patch bookkeeping keys from row metadata."""

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


def _patch_scan_filter(metadata_filter: Mapping[str, Any] | None) -> dict[str, Any]:
    """Builds a candidate-scan filter restricted to patch rows.

    Always constrains the scan to rows that carry a ``parent_id`` (every patch
    row does), and AND-composes any caller metadata filter on top.
    """

    base: dict[str, Any] = {_PARENT_KEY: {"$exists": True}}
    if metadata_filter is None:
        return base
    if not isinstance(metadata_filter, Mapping):
        raise ValueError("filter must be a mapping")
    if not metadata_filter:
        return base
    return {"$and": [base, dict(metadata_filter)]}
