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
rows keyed ``<doc_id>#<NNNNN>`` carrying a ``parent_id`` in metadata, and
retrieval is the two-stage approach the issue prescribes:

1. *Candidate generation* -- a batched any-patch nearest-neighbour search (one
   sub-query per query token) over the existing TurboVec scan, whose hit parents
   form a small candidate set.
2. *MaxSim rescoring* -- the full MaxSim score is recomputed in Python (numpy)
   over the candidate documents' patches, and the top ``k`` parents are returned.

The rescore is **exact**: TurboVec's quantized codes are used only to surface
candidates cheaply, while the patch vectors themselves are kept verbatim (float32,
base64, in the per-row text sidecar) so MaxSim is computed at full precision over
the candidate set rather than from the quantized candidate-generation scores.

The page/token encoder is **bring-your-own** (ColPali / ColQwen weights are
multi-GB and out of scope to bundle): pass precomputed patch matrices to
:meth:`add_document` / :meth:`search`, or an optional ``encoder`` exposing
``encode_documents`` / ``encode_queries``.

Footprint note: a document contributes one stored row per patch, so a page with
~1000 patches is ~1000 rows (plus the float32 patch sidecar the exact rescore
reads back). That is the known cost of late interaction and the reason native
multi-vector storage plus a MaxSim kernel in the TurboVec core is a separate,
benchmark-gated track (issue #25, stage 3); this prototype is for validating
retrieval quality and the API shape first.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from lodedb.local.db import LodeDB

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
        any-patch search depth used to gather rescoring candidates (higher =
        better recall, more rescoring work). ``read_only=True`` opens a
        non-mutating reader and requires the path to exist. ``bit_width`` and any
        extra ``lodedb_kwargs`` (e.g. ``durability=``, ``commit_mode=``) are
        forwarded to the underlying vector-only :class:`LodeDB`. The patch text
        sidecar that holds the vectors is always retained, so ``store_text`` may
        not be set ``False``.
        """

        if int(dim) <= 0:
            raise ValueError("dim must be a positive integer")
        if int(candidate_depth) <= 0:
            raise ValueError("candidate_depth must be a positive integer")
        if lodedb_kwargs.get("store_text") is False:
            raise ValueError(
                "LodeLateInteractionIndex stores patch vectors in the text sidecar; "
                "store_text must remain True"
            )
        self.dim = int(dim)
        self.encoder = encoder
        self.candidate_depth = int(candidate_depth)
        self.read_only = bool(read_only)
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

        ``query`` is a 2-D ``(num_query_tokens, dim)`` matrix. Retrieval is two
        stage: a batched any-patch search (one sub-query per query token, each to
        depth ``candidate_depth``) gathers a candidate set of documents, then the
        exact MaxSim score is computed over each candidate's stored patches and
        the top ``k`` are returned. ``candidate_depth`` overrides the index
        default for this call. ``filter`` (the same exact-match-or-predicate
        grammar as :meth:`LodeDB.search`) narrows the candidate scan by your
        document metadata. Query tokens are L2-normalized by default to match
        stored patches.
        """

        if int(k) <= 0:
            raise ValueError("k must be positive")
        depth = self.candidate_depth if candidate_depth is None else int(candidate_depth)
        if depth <= 0:
            raise ValueError("candidate_depth must be positive")
        query_matrix = _as_matrix(query, self.dim, normalize=normalize)

        # An index with no patches has no serving snapshot to scan; short-circuit
        # rather than letting the empty-store scan raise.
        if self._db.count() == 0:
            return []

        # Stage 1: candidate generation. One any-patch sub-query per query token;
        # union the owning documents of the hits. The quantized scan only needs to
        # surface the right parents -- exact ranking happens in stage 2.
        token_queries = [query_matrix[i].tolist() for i in range(query_matrix.shape[0])]
        per_token_hits = self._db.search_many_by_vector(
            token_queries,
            k=depth,
            filter=_patch_scan_filter(filter),
            normalize=False,
        )
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
        scored: list[LodeLateInteractionHit] = []
        for parent in candidates:
            doc_patches, user_meta, patch_count = self._load_document(parent)
            if doc_patches is None or doc_patches.shape[0] == 0:
                continue
            score = _maxsim(query_matrix, doc_patches)
            scored.append(
                LodeLateInteractionHit(
                    score=score,
                    id=parent,
                    metadata=user_meta,
                    patch_count=patch_count,
                )
            )
        # Deterministic order: score desc, then id asc to break ties stably.
        scored.sort(key=lambda hit: (-hit.score, hit.id))
        return scored[: int(k)]

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


# -- module helpers ---------------------------------------------------------


def _maxsim(query: np.ndarray, document: np.ndarray) -> float:
    """Computes the MaxSim score of a query matrix against a document matrix.

    ``sum over query tokens of max over doc patches of <q_t, d_p>``. With
    unit-norm rows each dot-product is a cosine similarity. Shapes are
    ``query=(Nq, dim)`` and ``document=(Nd, dim)``.
    """

    # (Nq, Nd) similarity matrix, then max over doc patches, then sum over tokens.
    sims = query @ document.T
    return float(sims.max(axis=1).sum())


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
