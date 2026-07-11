"""kotaemon vector-store adapter for LodeDB (no extra install needed).

``LodeDBVectorStore`` implements kotaemon's ``BaseVectorStore`` contract
(``add`` / ``delete`` / ``query`` / ``drop``, plus the conventional ``count`` and
``__persist_flow__``) against LodeDB's vector-in API: kotaemon owns the
embeddings, LodeDB stores and searches those vectors locally and persists
changed rows incrementally. Chunk text and full documents stay in kotaemon's
docstore (retrieval re-reads them by id), so this adapter keeps only what the
vector path needs: the vectors, the row ids, and the scalar metadata fields
kotaemon filters on (for example ``file_id``).

The adapter is deliberately dependency-free: it duck-types kotaemon's
``DocumentWithEmbedding`` (``.embedding`` / ``.metadata`` / ``.doc_id``) and
LlamaIndex's ``MetadataFilters`` instead of importing either package, so it
imports cleanly in any environment and needs no ``lodedb[...]`` extra. Select it
from kotaemon with only a settings change (no kotaemon fork)::

    # flowsettings.py
    KH_VECTORSTORE = {
        "__type__": "lodedb.local.integrations.kotaemon.LodeDBVectorStore",
        "path": str(KH_USER_DATA_DIR / "vectorstore"),
    }

kotaemon does not configure an embedding dimension anywhere — the store meets
the dimension of whatever embedding model the user selected at runtime — so the
LodeDB index is created lazily on the first ``add`` and its shape is recorded in
a small ``kotaemon_store.json`` sidecar for reopens. LodeDB indexes require a
dimension that is a multiple of 8, so vectors at any other dimension are
zero-padded up to the next multiple of 8; zero padding changes neither vector
norms nor dot products, so cosine scores are exactly what the unpadded vectors
would produce.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durability_from_env, durable_replace
from lodedb.local.db import LodeDB

# One row's kotaemon id is mirrored into this reserved metadata key so kotaemon's
# id allowlists (``query(ids=...)`` and the retrieval pipeline's ``doc_ids`` chunk
# scope) push down into LodeDB's metadata planner as an exact ``$in`` filter. The
# ``::`` prefix keeps it from colliding with real kotaemon metadata keys.
_ID_KEY = "kh::id"

# Shape sidecar written next to the LodeDB artifacts inside the collection
# directory. It records the logical embedding dimension (kotaemon never
# configures one) so a later open can reopen the index without an ``add``.
_META_FILENAME = "kotaemon_store.json"

_SCALAR_TYPES = (str, int, float, bool)

# VectorStoreQuery ranking hints that kotaemon may forward (e.g. ``mode="mmr"``,
# ``mmr_threshold=0.5``) but that kotaemon's default Chroma backend also ignores.
# They are accepted and ignored for drop-in parity; unknown kwargs still raise.
_IGNORED_QUERY_KWARGS = {
    "mode",
    "mmr_threshold",
    "alpha",
    "query_str",
    "sparse_top_k",
    "hybrid_top_k",
}

# LlamaIndex FilterOperator enum values -> LodeDB predicate operators.
_LI_OPERATOR_MAP = {
    "==": "$eq",
    "!=": "$ne",
    ">": "$gt",
    ">=": "$gte",
    "<": "$lt",
    "<=": "$lte",
    "in": "$in",
    "nin": "$nin",
}


class LodeDBVectorStore:
    """kotaemon ``BaseVectorStore`` backed by a vector-only :class:`LodeDB`.

    ``path`` is required: it is the base directory kotaemon configures in
    ``KH_VECTORSTORE`` and each collection persists to ``<path>/<collection_name>``.
    There is no default, so index data is never written somewhere ephemeral by
    accident. ``collection_name`` is injected per index by kotaemon's
    ``get_vectorstore``.

    ``store_text`` (default False) controls whether chunk text carried on
    ``DocumentWithEmbedding`` inputs is retained in LodeDB's raw-text sidecar.
    kotaemon's docstore is the authority for text (retrieval re-reads documents
    from it by id), so the vector store keeps no text by default; turn it on only
    for debugging or standalone use of the adapter.

    Scores returned by :meth:`query` are cosine similarities (higher is better,
    1.0 for an exact match). kotaemon's default Chroma collection ranks by an
    L2-derived score instead; for the normalized embeddings kotaemon produces the
    two orderings are identical, but absolute score values differ between
    backends.
    """

    def __init__(
        self,
        path: str | None = None,
        collection_name: str = "default",
        *,
        store_text: bool = False,
        bit_width: int = 4,
        **kwargs: Any,
    ) -> None:
        """Prepares a lazily-opened collection under ``<path>/<collection_name>``.

        The LodeDB index itself is opened on first use: immediately when the
        collection already exists on disk (its recorded shape is reused), else on
        the first :meth:`add`, which fixes the embedding dimension.
        """

        if kwargs:
            raise TypeError(
                "LodeDBVectorStore got unexpected config keys: "
                + ", ".join(sorted(kwargs))
                + ". Supported keys: path, collection_name, store_text, bit_width."
            )
        if path is None:
            raise ValueError(
                "LodeDBVectorStore requires an explicit 'path' for durable storage "
                "(set 'path' in KH_VECTORSTORE or pass path=...)"
            )
        self._base_path = Path(path)
        self._collection_name = _validated_collection_name(collection_name)
        self._collection_path = self._base_path / self._collection_name
        self._store_text = bool(store_text)
        self._bit_width = int(bit_width)
        self._db: LodeDB | None = None
        self._vector_dim: int | None = None
        self._padded_dim: int | None = None
        # Guards lazy open: kotaemon's hybrid retrieval queries from worker
        # threads, and two racing first-writes must not both create the index.
        self._open_lock = threading.Lock()
        if self._meta_path().exists():
            self._open_existing()

    # ------------------------------------------------------------------ #
    # kotaemon BaseVectorStore contract
    # ------------------------------------------------------------------ #

    def add(
        self,
        embeddings: Sequence[Sequence[float]] | Sequence[Any],
        metadatas: list[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Adds (or upserts) embeddings and returns their ids.

        ``embeddings`` is either a list of raw vectors or a list of kotaemon
        ``DocumentWithEmbedding``-like objects (anything with an ``.embedding``
        attribute; ``.metadata`` and ``.doc_id`` are honored when present).
        Explicit ``metadatas`` / ``ids`` win over document attributes, matching
        kotaemon's ``LlamaIndexVectorStore`` behavior. Missing ids are generated.
        Only scalar metadata values are kept (they back kotaemon's ``filters``
        push-down); nested values are dropped because kotaemon re-reads full
        documents from its docstore, never from the vector store.
        """

        if not embeddings:
            return []
        vectors: list[list[float]] = []
        doc_metadatas: list[dict[str, Any]] = []
        doc_ids: list[str | None] = []
        texts: list[str | None] = []
        for item in embeddings:
            embedding = getattr(item, "embedding", item)
            vectors.append([float(value) for value in embedding])
            metadata = getattr(item, "metadata", None)
            doc_metadatas.append(dict(metadata) if isinstance(metadata, Mapping) else {})
            item_id = getattr(item, "doc_id", None) or getattr(item, "id_", None)
            doc_ids.append(str(item_id) if item_id is not None else None)
            text = getattr(item, "text", None)
            texts.append(text if isinstance(text, str) and text.strip() else None)
        if metadatas is not None:
            if len(metadatas) != len(vectors):
                raise ValueError("metadatas must have the same length as embeddings")
            doc_metadatas = [dict(metadata or {}) for metadata in metadatas]
        if ids is not None:
            if len(ids) != len(vectors):
                raise ValueError("ids must have the same length as embeddings")
            doc_ids = [str(value) for value in ids]
        resolved_ids = [value if value is not None else str(uuid.uuid4()) for value in doc_ids]

        db = self._ensure_open(vector_dim=len(vectors[0]))
        documents = []
        for vector, doc_id, metadata, text in zip(
            vectors, resolved_ids, doc_metadatas, texts, strict=True
        ):
            documents.append(
                {
                    "vector": self._pad(vector),
                    "id": doc_id,
                    "metadata": _scalar_metadata(metadata, doc_id),
                    "text": text if self._store_text else None,
                }
            )
        db.add_vectors_many(documents)
        return resolved_ids

    def delete(self, ids: list[str], **kwargs: Any) -> None:
        """Deletes vectors by id; ids that are absent are ignored.

        ``kwargs`` is accepted for interface parity and ignored (kotaemon's other
        stores take backend-specific options there).
        """

        db = self._get_db()
        if db is None:
            return
        for doc_id in ids:
            db.remove(str(doc_id))

    def query(
        self,
        embedding: Sequence[float],
        top_k: int = 1,
        ids: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[float], list[str]]:
        """Returns ``(embeddings, similarities, ids)`` for the top-``top_k`` hits.

        ``ids`` and the ``doc_ids`` keyword (kotaemon's retrieval passes its
        chunk-id ``scope`` as ``doc_ids``) are id allowlists; both given means
        their intersection. A ``filters`` keyword accepts a LlamaIndex
        ``MetadataFilters``-like object (duck-typed) or a plain LodeDB filter
        dict, and pushes down into LodeDB's metadata planner. Ranking hints that
        kotaemon's default Chroma backend also ignores (``mode``,
        ``mmr_threshold``, ...) are accepted and ignored; any other keyword
        raises. Similarities are exact cosine scores, higher is better. The first
        tuple element is ``[None] * n``: LodeDB does not expose stored vectors,
        and kotaemon's pipelines discard this element.
        """

        li_filters = kwargs.pop("filters", None)
        allowlist = _combine_allowlists(ids, kwargs.pop("doc_ids", None))
        unexpected = set(kwargs) - _IGNORED_QUERY_KWARGS
        if unexpected:
            raise TypeError(
                "LodeDBVectorStore.query got unexpected keyword(s): "
                + ", ".join(sorted(unexpected))
            )
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        db = self._get_db()
        if db is None or allowlist == []:
            # No index yet (nothing was ever added) or an explicitly empty scope.
            return [], [], []

        clauses: list[dict[str, Any]] = []
        if allowlist is not None:
            clauses.append({_ID_KEY: {"$in": allowlist}})
        translated = _translate_li_filters(li_filters)
        if translated:
            clauses.append(translated)
        lode_filter = clauses[0] if len(clauses) == 1 else ({"$and": clauses} if clauses else None)

        hits = db.search_by_vector(self._pad(embedding), k=int(top_k), filter=lode_filter)
        # TurboVec's quantized cosine can exceed 1.0 by a small artifact on a
        # near-exact match; clamp so downstream "score <= 1" assumptions hold.
        return (
            [None] * len(hits),
            [min(float(hit.score), 1.0) for hit in hits],
            [hit.id for hit in hits],
        )

    def drop(self) -> None:
        """Deletes the entire collection from disk.

        Removes ``<path>/<collection_name>`` — but only when this adapter's shape
        sidecar is present, so a misconfigured path can never delete a directory
        the adapter did not create. A collection that was never created is a
        no-op.
        """

        with self._open_lock:
            if self._db is not None:
                self._db.close()
                self._db = None
            if self._meta_path().exists():
                shutil.rmtree(self._collection_path)
            self._vector_dim = None
            self._padded_dim = None

    # ------------------------------------------------------------------ #
    # conventional extras (kotaemon stores expose these too)
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        """Returns the number of stored vectors (0 before the first add)."""

        db = self._get_db()
        return 0 if db is None else db.count()

    def close(self) -> None:
        """Closes the underlying LodeDB handle (reopened lazily on next use)."""

        with self._open_lock:
            if self._db is not None:
                self._db.close()
                self._db = None

    def __persist_flow__(self) -> dict[str, Any]:
        """Returns the constructor kwargs theflow needs to re-create this store."""

        return {
            "path": str(self._base_path),
            "collection_name": self._collection_name,
            "store_text": self._store_text,
            "bit_width": self._bit_width,
        }

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _meta_path(self) -> Path:
        """Returns the path of the collection's shape sidecar."""

        return self._collection_path / _META_FILENAME

    def _get_db(self) -> LodeDB | None:
        """Returns the handle, lazily reopening a closed-but-existing collection.

        ``None`` means the collection was never created (nothing added yet). A
        handle that was ``close()``d but whose shape sidecar is on disk reopens
        here, so a closed adapter never masquerades as an empty collection.
        """

        if self._db is None and self._meta_path().exists():
            self._open_existing()
        return self._db

    def _open_existing(self) -> None:
        """Reopens a collection whose shape sidecar is already on disk."""

        with self._open_lock:
            self._open_existing_locked()

    def _open_existing_locked(self) -> None:
        """Reopen body; the caller must hold ``_open_lock``."""

        if self._db is not None:
            return
        meta = json.loads(self._meta_path().read_text(encoding="utf-8"))
        self._vector_dim = int(meta["vector_dim"])
        self._padded_dim = int(meta["padded_dim"])
        # A LodeDB store must be reopened with the store_text value it was
        # written with; the recorded value wins over a changed config.
        self._store_text = bool(meta["store_text"])
        self._db = LodeDB.open_vector_store(
            self._collection_path,
            vector_dim=self._padded_dim,
            bit_width=self._bit_width,
            store_text=self._store_text,
        )

    def _ensure_open(self, *, vector_dim: int) -> LodeDB:
        """Opens (creating if needed) the index and pins the embedding dimension.

        The first ``add`` fixes the collection's dimension; later adds at a
        different dimension raise rather than silently mixing incomparable
        vectors (kotaemon hits this when the user switches embedding models on an
        existing index).
        """

        with self._open_lock:
            if self._db is None and self._meta_path().exists():
                # A close()d handle on an existing collection: reopen the
                # recorded shape rather than re-deriving it from this add.
                self._open_existing_locked()
            if self._db is not None:
                if vector_dim != self._vector_dim:
                    raise ValueError(
                        f"collection {self._collection_name!r} stores "
                        f"{self._vector_dim}-dim embeddings, got {vector_dim}-dim; "
                        "drop the collection before switching embedding models"
                    )
                return self._db
            if vector_dim <= 0:
                raise ValueError("embeddings must be non-empty vectors")
            self._vector_dim = int(vector_dim)
            # LodeDB indexes require a multiple-of-8 dimension; zero padding is
            # exactly score-preserving for cosine similarity.
            self._padded_dim = ((self._vector_dim + 7) // 8) * 8
            self._collection_path.mkdir(parents=True, exist_ok=True)
            self._db = LodeDB.open_vector_store(
                self._collection_path,
                vector_dim=self._padded_dim,
                bit_width=self._bit_width,
                store_text=self._store_text,
            )
            self._write_meta()
            return self._db

    def _write_meta(self) -> None:
        """Persists the collection shape sidecar atomically.

        A later open depends on this sidecar to know the collection's shape, so
        it follows the same durability mode as the LodeDB store it describes
        (the adapter opens the store without an explicit ``durability=``, which
        resolves from ``LODEDB_DURABILITY`` — mirror that here).
        """

        payload = json.dumps(
            {
                "vector_dim": self._vector_dim,
                "padded_dim": self._padded_dim,
                "store_text": self._store_text,
            },
            sort_keys=True,
        )
        fd, tmp = tempfile.mkstemp(
            dir=self._collection_path, prefix=_META_FILENAME, suffix=".tmp"
        )
        with open(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
        durable_replace(tmp, self._meta_path(), fsync=durability_from_env())

    def _pad(self, vector: Sequence[float]) -> list[float]:
        """Validates a vector's dimension and zero-pads it to the index shape."""

        values = [float(value) for value in vector]
        if self._vector_dim is None or self._padded_dim is None:
            raise RuntimeError("collection has no recorded dimension (nothing added yet)")
        if len(values) != self._vector_dim:
            raise ValueError(
                f"expected a {self._vector_dim}-dim embedding, got {len(values)}-dim"
            )
        return values + [0.0] * (self._padded_dim - len(values))


def _validated_collection_name(collection_name: str) -> str:
    """Returns the collection name, rejecting anything but a single path component.

    The collection directory is joined under the configured base ``path`` and
    :meth:`LodeDBVectorStore.drop` deletes it recursively, so a name carrying
    path separators, ``..``, or an absolute prefix must never reach that join —
    it would let a config value write to (and drop) a directory outside the
    vector-store root.
    """

    name = str(collection_name)
    if (
        not name
        or name in (".", "..")
        or "/" in name
        or "\\" in name
        or name != Path(name).name
    ):
        raise ValueError(
            f"collection_name must be a single path component, got {collection_name!r}"
        )
    return name


def _scalar_metadata(metadata: Mapping[str, Any], doc_id: str) -> dict[str, Any]:
    """Returns the scalar, filterable subset of kotaemon metadata plus the id mirror.

    Non-scalar values (lists, dicts) and ``None`` are dropped: LodeDB metadata is
    scalar-only, kotaemon's docstore is the authority for full documents, and an
    absent key is what ``$exists`` filters should see for a null field.
    """

    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, _SCALAR_TYPES):
            out[str(key)] = value
    out[_ID_KEY] = doc_id
    return out


def _combine_allowlists(
    ids: Sequence[str] | None, doc_ids: Sequence[str] | None
) -> list[str] | None:
    """Merges the ``ids`` and ``doc_ids`` allowlists (intersection when both given).

    Returns ``None`` when neither is given; an empty list is a real, empty scope
    (the caller must return no hits, not all hits).
    """

    if ids is None and doc_ids is None:
        return None
    if ids is None:
        return [str(value) for value in doc_ids or []]
    if doc_ids is None:
        return [str(value) for value in ids]
    scope = {str(value) for value in doc_ids}
    return [str(value) for value in ids if str(value) in scope]


def _translate_li_filters(filters: Any) -> dict[str, Any] | None:
    """Translates a LlamaIndex ``MetadataFilters``-like object to LodeDB's grammar.

    Duck-typed on purpose (``.filters`` / ``.condition`` / per-filter ``.key`` /
    ``.value`` / ``.operator``) so this module never imports llama-index. A plain
    mapping is passed through as an already-native LodeDB filter. Operators
    outside LodeDB's grammar (``text_match``, ``contains``, ...) raise rather
    than silently degrade.
    """

    if filters is None:
        return None
    if isinstance(filters, Mapping):
        return dict(filters)
    subfilters = getattr(filters, "filters", None)
    if subfilters is None:
        raise TypeError(
            "filters must be a LodeDB filter dict or a MetadataFilters-like object "
            f"with a .filters list, got {type(filters).__name__}"
        )
    parts: list[dict[str, Any]] = []
    for item in subfilters:
        if getattr(item, "filters", None) is not None:
            nested = _translate_li_filters(item)
            if nested:
                parts.append(nested)
            continue
        operator = getattr(item, "operator", "==")
        operator_value = str(getattr(operator, "value", operator))
        lode_op = _LI_OPERATOR_MAP.get(operator_value)
        if lode_op is None:
            raise NotImplementedError(
                f"metadata filter operator {operator_value!r} is not supported by LodeDB"
            )
        parts.append({str(item.key): {lode_op: item.value}})
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    condition = getattr(filters, "condition", "and")
    condition_value = str(getattr(condition, "value", condition)).lower()
    if condition_value not in ("and", "or"):
        raise NotImplementedError(f"filter condition {condition_value!r} is not supported")
    return {f"${condition_value}": parts}


__all__ = ["LodeDBVectorStore"]
