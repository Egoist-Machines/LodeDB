"""LlamaIndex ``VectorStore`` adapter for LodeDB (optional ``lodedb[llama-index]``).

Wraps the LodeDB SDK as a ``llama_index.core.vector_stores.types.BasePydanticVectorStore``
so LlamaIndex RAG apps can drop in the local-first store, alongside the LangChain adapter.

**Text-path, not vector-path.** LodeDB embeds text internally (with the model chosen at
``LodeDB(model=...)``), so this adapter is *text-path*: it sets ``is_embedding_query = False``
and feeds LlamaIndex node text (``node.get_content``) and the raw ``query.query_str`` to
LodeDB, which does the embedding. **LlamaIndex's own ``embed_model`` is therefore ignored for
indexing and querying.** If you build the index through :class:`VectorStoreIndex`, set a cheap
``embed_model`` (e.g. ``MockEmbedding``) so LlamaIndex does not try to embed nodes with a
remote model. The vectors it computes are discarded; LodeDB re-embeds the text. (The SDK does
have a vector-in path, but that is a separate *vector-only* index that owns no embedder and is
out of scope for this text-path adapter; mixing externally embedded vectors with LodeDB's own
embeddings in one index makes similarity scores meaningless.)

**Search modes.** :meth:`query` maps :class:`VectorStoreQueryMode` onto LodeDB's retrieval
modes: ``DEFAULT`` -> vector cosine search, ``HYBRID``/``SEMANTIC_HYBRID`` -> LodeDB's BM25 +
Reciprocal Rank Fusion hybrid, and ``SPARSE``/``TEXT_SEARCH`` -> lexical (BM25-only) search.
Hybrid and lexical need a lexical source, so the wrapped DB must be opened with
``store_text=True`` (the default) or ``index_text=True``; otherwise LodeDB raises. RRF has no
tunable weight, so ``query.alpha`` is ignored. ``MMR`` and the learned modes
(``SVM``/``LOGISTIC_REGRESSION``/``LINEAR_REGRESSION``) need full-precision or per-document
vector access LodeDB does not expose, and raise :class:`NotImplementedError`.

**Metadata filters.** :class:`MetadataFilters` translate into LodeDB's predicate grammar:
operators ``EQ`` ``NE`` ``GT`` ``GTE`` ``LT`` ``LTE`` ``IN`` ``NIN`` and ``IS_EMPTY`` (field
missing or empty), composed with ``AND`` / ``OR`` / ``NOT`` (``NOT`` meaning "none of these
match", as in LlamaIndex's own evaluator) and nestable. The substring/list operators
(``CONTAINS`` ``TEXT_MATCH`` ``TEXT_MATCH_INSENSITIVE`` ``ANY`` ``ALL``) have no LodeDB
metadata equivalent and raise :class:`NotImplementedError`. LodeDB stores metadata as a
string->string map, so the ordered comparisons (``GT``/``GTE``/``LT``/``LTE``) are numeric
only when both the stored value and the operand parse as numbers (otherwise lexicographic).

**Durability.** LodeDB owns persistence at its constructor ``path`` and commits every mutation
atomically; you do not use ``StorageContext.persist``/``from_persist_dir`` for durability here.
To reopen, construct a new :class:`LodeDB` at the same ``path`` and wrap it again.
:meth:`persist` just forces a durability checkpoint on the wrapped DB. A node's source document
(``ref_doc_id``) is recorded in a reserved metadata key, so :meth:`delete` by ``ref_doc_id`` and
``doc_ids`` query scoping resolve through metadata enumeration and survive a reopen (they are
**not** session-local).

**Fidelity caveats.** Retrieved hits and :meth:`get_nodes` reconstruct :class:`TextNode` from
the id, the durable ``.tvtext`` text (when ``store_text=True``), and the stored metadata. The
``SOURCE`` relationship is rebuilt from the reserved ref-doc key; other node relationships and
non-metadata node fields are not round-tripped, and because LodeDB stores metadata as strings,
numeric/bool metadata values come back as strings. :meth:`get` (read a stored embedding back as
floats) stays unsupported: LodeDB keeps vectors in its compact TurboVec format and exposes no
float-vector read.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lodedb.local.db import LodeDB

try:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.schema import (
        BaseNode,
        MetadataMode,
        NodeRelationship,
        RelatedNodeInfo,
        TextNode,
    )
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
        VectorStoreQueryMode,
        VectorStoreQueryResult,
    )
except ImportError as exc:  # pragma: no cover - clear install hint
    raise ImportError(
        "the LodeDB LlamaIndex adapter needs llama-index-core: pip install 'lodedb[llama-index]'"
    ) from exc

# Reserved metadata key holding a node's source-document id (``ref_doc_id``). Stored on add so
# delete-by-ref-doc and ``doc_ids`` scoping resolve durably via metadata enumeration; stripped
# from the metadata handed back on reconstructed nodes.
_REF_DOC_KEY = "_lodedb_ref_doc_id"

# LlamaIndex query mode -> LodeDB ``search(mode=...)``. Modes absent here are unsupported.
_MODE_TO_LODE: dict[VectorStoreQueryMode, str] = {
    VectorStoreQueryMode.DEFAULT: "vector",
    VectorStoreQueryMode.HYBRID: "hybrid",
    VectorStoreQueryMode.SEMANTIC_HYBRID: "hybrid",
    VectorStoreQueryMode.SPARSE: "lexical",
    VectorStoreQueryMode.TEXT_SEARCH: "lexical",
}

# LlamaIndex comparison operator -> LodeDB predicate operator. ``EQ`` is handled as a bare
# scalar (LodeDB's exact-match sugar); ``IS_EMPTY`` and the unsupported operators are handled
# explicitly in :meth:`_translate_leaf`.
_OP_TO_LODE: dict[FilterOperator, str] = {
    FilterOperator.NE: "$ne",
    FilterOperator.GT: "$gt",
    FilterOperator.GTE: "$gte",
    FilterOperator.LT: "$lt",
    FilterOperator.LTE: "$lte",
    FilterOperator.IN: "$in",
    FilterOperator.NIN: "$nin",
}


class LodeDBVectorStore(BasePydanticVectorStore):
    """A LlamaIndex ``BasePydanticVectorStore`` backed by a local :class:`LodeDB`."""

    stores_text: bool = True
    # False => LlamaIndex passes us query.query_str (not an embedding); LodeDB embeds it.
    is_embedding_query: bool = False
    flat_metadata: bool = False

    _db: LodeDB = PrivateAttr()

    def __init__(self, db: LodeDB, **kwargs: Any) -> None:
        """Wraps an already-open LodeDB."""

        super().__init__(**kwargs)
        self._db = db

    @classmethod
    def class_name(cls) -> str:
        """Stable type name used by LlamaIndex's (de)serialization registry."""

        return "LodeDBVectorStore"

    @classmethod
    def from_path(cls, path: str, *, model: str = "minilm", device: str = "auto", **kwargs: Any):
        """Opens a LodeDB at ``path`` and wraps it (convenience for examples/tests)."""

        return cls(LodeDB(path=path, model=model, device=device), **kwargs)

    @property
    def client(self) -> LodeDB:
        """Returns the underlying :class:`LodeDB` handle."""

        return self._db

    def add(self, nodes: Sequence[BaseNode], **_: Any) -> list[str]:
        """Indexes ``nodes`` by their text (LodeDB embeds it) and returns their ids.

        The node's own embedding is ignored (text-path). Empty/non-text nodes are
        rejected because LodeDB cannot embed them. A node's ``ref_doc_id`` is stored in a
        reserved metadata key so delete-by-ref-doc and ``doc_ids`` scoping stay durable.
        """

        if not nodes:
            return []
        items: list[dict[str, Any]] = []
        ids: list[str] = []
        for node in nodes:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            if not text or not text.strip():
                raise ValueError(
                    f"node {node.node_id!r} has no text content; the LodeDB LlamaIndex "
                    "adapter is text-path (LodeDB embeds text internally) and cannot index "
                    "empty or non-text nodes"
                )
            metadata = dict(node.metadata or {})
            ref = node.ref_doc_id
            if ref is not None:
                metadata[_REF_DOC_KEY] = ref
            items.append({"text": text, "id": node.node_id, "metadata": metadata})
            ids.append(node.node_id)
        self._db.add_many(items)
        return ids

    def delete(self, ref_doc_id: str, **_: Any) -> None:
        """Deletes every node whose source document is ``ref_doc_id`` (durable).

        Resolves the member ids through metadata enumeration on the reserved ref-doc key,
        so it removes nodes added in any session, not just the current one.
        """

        for record in self._db.list_documents(filter={_REF_DOC_KEY: ref_doc_id}):
            self._db.remove(record["id"])

    def delete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **_: Any,
    ) -> None:
        """Deletes documents by explicit ``node_ids`` and/or a metadata ``filters`` match.

        When both are given the intersection is removed (matching LlamaIndex's reference
        store). With neither this is a no-op rather than a full wipe, so an empty call never
        clears the store by accident.
        """

        if node_ids is None and filters is None:
            return
        if node_ids is not None and not node_ids and filters is None:
            return
        if filters is None:
            targets: list[str] = list(node_ids or [])
        else:
            lode_filter, empty = self._enumeration_filter(node_ids, filters)
            if empty:
                return
            targets = [record["id"] for record in self._db.list_documents(filter=lode_filter)]
        for nid in targets:
            self._db.remove(nid)

    def get_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        """Returns the stored nodes matching ``node_ids`` and/or ``filters`` (enumeration).

        Backed by LodeDB metadata enumeration: each node is reconstructed from its id, the
        durable text (when ``store_text=True``), and the stored metadata, with the ``SOURCE``
        relationship rebuilt from the reserved ref-doc key. With neither argument it returns
        every stored node. When ``node_ids`` is given, results follow that order.
        """

        lode_filter, empty = self._enumeration_filter(node_ids, filters)
        if empty:
            return []
        records = self._db.list_documents(filter=lode_filter)
        nodes = self._nodes_from_records(records)
        if node_ids:
            by_id = {node.node_id: node for node in nodes}
            return [by_id[nid] for nid in node_ids if nid in by_id]
        return nodes

    def get(self, text_id: str) -> list[float]:
        """Unsupported: LodeDB stores compact vectors and exposes no float-vector read."""

        raise NotImplementedError(
            "LodeDBVectorStore.get(text_id) cannot return a full-precision embedding; "
            "LodeDB stores vectors in its compact TurboVec format and does not expose them. "
            "Use get_nodes() for text/metadata, or keep a parallel store for raw embeddings."
        )

    def query(self, query: VectorStoreQuery, **_: Any) -> VectorStoreQueryResult:
        """Runs a top-``k`` search for ``query.query_str`` in the requested mode.

        ``DEFAULT`` is vector search; ``HYBRID``/``SEMANTIC_HYBRID`` run LodeDB's BM25 + RRF
        hybrid; ``SPARSE``/``TEXT_SEARCH`` run lexical (BM25) search. ``MMR`` and learned
        modes raise (LodeDB exposes no full-precision vectors). Hybrid/lexical require the
        wrapped DB to retain a lexical source (``store_text=True`` default, or ``index_text``).
        """

        lode_mode = self._resolve_mode(query.mode)
        query_str = query.query_str
        if not query_str or not query_str.strip():
            raise ValueError(
                "LodeDBVectorStore is text-path (is_embedding_query=False) and needs "
                "query.query_str; an embedding-only query was supplied."
            )

        lode_filter, empty = self._build_query_filter(query)
        if empty:
            # An id/doc filter was requested but resolved to no candidates.
            return VectorStoreQueryResult(nodes=[], similarities=[], ids=[])

        hits = self._db.search(
            query_str, k=self._top_k(query, lode_mode), filter=lode_filter, mode=lode_mode
        )
        retains_text = getattr(self._db, "store_text", False)
        nodes: list[TextNode] = []
        similarities: list[float] = []
        ids: list[str] = []
        for hit in hits:
            text = self._db.get_text(hit.id) if retains_text else None
            nodes.append(self._build_node(hit.id, hit.metadata, text))
            similarities.append(hit.score)
            ids.append(hit.id)
        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)

    def persist(self, persist_path: str | None = None, fs: Any = None) -> None:
        """Forces a durability checkpoint on the wrapped LodeDB.

        ``persist_path``/``fs`` are accepted for signature compatibility but ignored:
        LodeDB persists to its own constructor ``path`` (see the module docstring).
        """

        self._db.persist()

    # -- internals ----------------------------------------------------------

    def _build_node(self, node_id: str, metadata: dict[str, Any], text: str | None) -> TextNode:
        """Rebuilds a :class:`TextNode`, restoring the ``SOURCE`` ref-doc relationship."""

        meta = dict(metadata)
        ref = meta.pop(_REF_DOC_KEY, None)
        node = TextNode(id_=node_id, text=text or "", metadata=meta)
        if ref is not None:
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=ref)
        return node

    def _nodes_from_records(self, records: list[dict[str, Any]]) -> list[TextNode]:
        """Reconstructs nodes from enumeration records, batching the durable text read."""

        ids = [record["id"] for record in records]
        texts = self._db.get_texts(ids) if (ids and getattr(self._db, "store_text", False)) else {}
        return [
            self._build_node(record["id"], record.get("metadata", {}), texts.get(record["id"]))
            for record in records
        ]

    @staticmethod
    def _top_k(query: VectorStoreQuery, lode_mode: str) -> int:
        """Picks the mode-appropriate top-k, falling back to ``similarity_top_k``."""

        if lode_mode == "hybrid" and query.hybrid_top_k:
            return int(query.hybrid_top_k)
        if lode_mode == "lexical" and query.sparse_top_k:
            return int(query.sparse_top_k)
        return int(query.similarity_top_k or 1)

    @staticmethod
    def _resolve_mode(mode: VectorStoreQueryMode) -> str:
        """Maps a LlamaIndex query mode to a LodeDB search mode, or raises if unsupported."""

        lode_mode = _MODE_TO_LODE.get(mode)
        if lode_mode is None:
            raise NotImplementedError(
                f"LodeDBVectorStore does not support query mode {mode!r}; supported modes are "
                "DEFAULT (vector), HYBRID/SEMANTIC_HYBRID (BM25 + RRF), and SPARSE/TEXT_SEARCH "
                "(lexical). MMR and learned modes need full-precision vectors LodeDB does not "
                "expose."
            )
        return lode_mode

    def _build_query_filter(self, query: VectorStoreQuery) -> tuple[dict[str, Any] | None, bool]:
        """Builds the LodeDB filter for a :meth:`query`; returns ``(filter, is_empty)``.

        Combines translated metadata filters with engine-side id scoping: ``node_ids`` become a
        ``document_ids`` allowlist and ``doc_ids`` become a predicate on the reserved ref-doc
        key, so the engine intersects them with the metadata match. ``is_empty`` is True when an
        id/doc constraint was requested but is empty (caller short-circuits to no results).
        """

        if (query.node_ids is not None and not query.node_ids) or (
            query.doc_ids is not None and not query.doc_ids
        ):
            return None, True

        metadata = self._translate_filters(query.filters) if query.filters is not None else {}
        if query.doc_ids:
            ref_clause = {_REF_DOC_KEY: {"$in": list(query.doc_ids)}}
            metadata = {"$and": [metadata, ref_clause]} if metadata else ref_clause

        out: dict[str, Any] = {}
        if metadata:
            out["metadata"] = metadata
        if query.node_ids:
            out["document_ids"] = list(query.node_ids)
        return (out or None), False

    def _enumeration_filter(
        self,
        node_ids: list[str] | None,
        filters: MetadataFilters | None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """Builds the LodeDB filter for :meth:`get_nodes`/:meth:`delete_nodes`.

        Returns ``(filter, is_empty)``; ``is_empty`` is True when ``node_ids`` is an empty
        list (so the caller returns nothing without enumerating the corpus).
        """

        if node_ids is not None and not node_ids:
            return None, True
        metadata = self._translate_filters(filters) if filters is not None else {}
        out: dict[str, Any] = {}
        if metadata:
            out["metadata"] = metadata
        if node_ids:
            out["document_ids"] = list(node_ids)
        return (out or None), False

    def _translate_filters(self, filters: MetadataFilters) -> dict[str, Any]:
        """Translates :class:`MetadataFilters` into a LodeDB predicate node (recursively).

        ``AND``/``OR`` map to ``$and``/``$or``; ``NOT`` means "none of these match"
        (``not any`` in LlamaIndex's own evaluator), i.e. ``$not`` over the ``$or`` of the
        parts. Nested :class:`MetadataFilters` recurse.
        """

        parts: list[dict[str, Any]] = []
        for sub in filters.filters:
            if isinstance(sub, MetadataFilters):
                translated = self._translate_filters(sub)
            else:
                translated = self._translate_leaf(sub)
            if translated:
                parts.append(translated)
        if not parts:
            return {}

        condition = filters.condition or FilterCondition.AND
        if condition == FilterCondition.AND:
            return parts[0] if len(parts) == 1 else {"$and": parts}
        if condition == FilterCondition.OR:
            return parts[0] if len(parts) == 1 else {"$or": parts}
        if condition == FilterCondition.NOT:
            inner = parts[0] if len(parts) == 1 else {"$or": parts}
            return {"$not": inner}
        raise NotImplementedError(
            f"LodeDBVectorStore does not support filter condition {condition!r}."
        )

    @staticmethod
    def _translate_leaf(f: MetadataFilter) -> dict[str, Any]:
        """Translates one :class:`MetadataFilter` into a LodeDB predicate node."""

        op = f.operator
        if op == FilterOperator.EQ:
            return {f.key: f.value}  # bare scalar is LodeDB's $eq sugar
        if op == FilterOperator.IS_EMPTY:
            # "Empty" in LlamaIndex is missing/None/"" ; over LodeDB's string metadata that is
            # field-missing OR the empty string.
            return {"$or": [{f.key: {"$exists": False}}, {f.key: ""}]}
        lode_op = _OP_TO_LODE.get(op)
        if lode_op is None:
            raise NotImplementedError(
                f"LodeDBVectorStore does not support filter operator {op!r}; LodeDB metadata "
                "predicates have no substring/list match (CONTAINS / TEXT_MATCH / ANY / ALL)."
            )
        return {f.key: {lode_op: f.value}}

    # -- async overrides ----------------------------------------------------

    async def async_add(self, nodes: Sequence[BaseNode], **kwargs: Any) -> list[str]:
        """Async shim over :meth:`add` (LodeDB is synchronous, in-process)."""

        return self.add(nodes, **kwargs)

    async def adelete(self, ref_doc_id: str, **kwargs: Any) -> None:
        """Async shim over :meth:`delete`."""

        self.delete(ref_doc_id, **kwargs)

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
        **kwargs: Any,
    ) -> None:
        """Async shim over :meth:`delete_nodes`."""

        self.delete_nodes(node_ids=node_ids, filters=filters, **kwargs)

    async def aget_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: MetadataFilters | None = None,
    ) -> list[BaseNode]:
        """Async shim over :meth:`get_nodes`."""

        return self.get_nodes(node_ids=node_ids, filters=filters)

    async def aquery(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Async shim over :meth:`query`."""

        return self.query(query, **kwargs)


__all__ = ["LodeDBVectorStore"]
