"""LlamaIndex ``PropertyGraphStore`` adapter for LodeDB (optional ``lodedb[llama-index]``).

Wraps :class:`lodedb.graph.KnowledgeGraph` as a
``llama_index.core.graph_stores.types.PropertyGraphStore`` so LlamaIndex's
``PropertyGraphIndex`` can use LodeDB's hybrid graph layer (SQLite topology + LodeDB
semantic index) as its graph store — the graph counterpart to the
:class:`~lodedb.local.integrations.llama_index.LodeDBVectorStore` vector-store adapter.

**Node mapping.** A LlamaIndex :class:`EntityNode` maps to a graph node typed by its
``label`` (e.g. ``"PERSON"``) and embedded by its ``name``; a :class:`ChunkNode` maps to a
node typed ``"text_chunk"`` and embedded by its ``text``. The original kind is recorded in a
reserved node property so :meth:`get` reconstructs the right subclass; node ``properties`` are
stored as JSON in the topology store, so unlike the vector-store adapter's string-only
metadata they round-trip with their original types. A :class:`Relation` maps to a directed,
typed edge ``(source_id) -[label]-> (target_id)`` whose id is derived from that triple (so
re-upserting the same relation upserts).

**Text-path.** Like the vector-store adapter, node text is embedded by LodeDB's own model
(`KnowledgeGraph(model=...)`); LlamaIndex's ``embed_model`` is not used for storage, so a node's
precomputed ``embedding`` is ignored and the node is (re)embedded from its name/text.

**Semantic queries.** ``supports_vector_queries`` is True. :meth:`vector_query` prefers
``query.query_str`` (the fully text-path route: LodeDB embeds the query with its own model) and
maps ``VectorStoreQueryMode`` the same way the vector-store adapter does (``DEFAULT`` ->
vector, ``HYBRID``/``SEMANTIC_HYBRID`` -> BM25 + RRF, ``SPARSE``/``TEXT_SEARCH`` -> lexical).
When only ``query.query_embedding`` is supplied — which is what LlamaIndex's high-level
``VectorContextRetriever`` does — it is used directly against the index; that is meaningful
only when LlamaIndex's ``embed_model`` matches the KG's model and dimension (minilm -> 384,
bge -> 768), because LodeDB compares it against node-text embeddings it produced with the KG
model. ``supports_structured_queries`` is False: :meth:`structured_query` (Cypher) raises.

**Traversal.** :meth:`get_rel_map` expands the graph from seed nodes over the SQLite adjacency,
and :meth:`get_triplets` / :meth:`get` enumerate the topology, so these are deterministic
complete-set reads, not similarity rankings.
"""

from __future__ import annotations

from typing import Any

from lodedb.graph import Edge, KnowledgeGraph, Node

try:
    from llama_index.core.graph_stores.types import (
        ChunkNode,
        EntityNode,
        LabelledNode,
        PropertyGraphStore,
        Relation,
        Triplet,
    )
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode
except ImportError as exc:  # pragma: no cover - clear install hint
    raise ImportError(
        "the LodeDB LlamaIndex graph adapter needs llama-index-core: "
        "pip install 'lodedb[llama-index]'"
    ) from exc

# Reserved node-property key recording the LlamaIndex node kind ("entity" / "chunk") so
# :meth:`get` rebuilds the right subclass; stripped from the properties handed back.
_KIND_KEY = "_lodedb_pg_kind"
_ENTITY = "entity"
_CHUNK = "chunk"

# LlamaIndex query mode -> LodeDB ``semantic_nodes(mode=...)``. Modes absent here are unsupported.
_MODE_TO_LODE: dict[VectorStoreQueryMode, str] = {
    VectorStoreQueryMode.DEFAULT: "vector",
    VectorStoreQueryMode.HYBRID: "hybrid",
    VectorStoreQueryMode.SEMANTIC_HYBRID: "hybrid",
    VectorStoreQueryMode.SPARSE: "lexical",
    VectorStoreQueryMode.TEXT_SEARCH: "lexical",
}


class LodeDBPropertyGraphStore(PropertyGraphStore):
    """A LlamaIndex ``PropertyGraphStore`` backed by a local :class:`KnowledgeGraph`."""

    supports_structured_queries: bool = False
    supports_vector_queries: bool = True

    def __init__(self, kg: KnowledgeGraph) -> None:
        """Wraps an already-open :class:`KnowledgeGraph`."""

        self._kg = kg

    @classmethod
    def from_path(cls, path: str, *, model: str = "minilm", device: str = "auto", **kwargs: Any):
        """Opens a :class:`KnowledgeGraph` at ``path`` and wraps it (examples/tests)."""

        return cls(KnowledgeGraph(path=path, model=model, device=device, **kwargs))

    @property
    def client(self) -> KnowledgeGraph:
        """Returns the underlying :class:`KnowledgeGraph` handle."""

        return self._kg

    # -- writes -------------------------------------------------------------

    def upsert_nodes(self, nodes: list[LabelledNode]) -> None:
        """Adds or replaces nodes (text-path: LodeDB embeds each node's name/text)."""

        items = [self._node_to_kg(node) for node in nodes]
        if items:
            self._kg.add_nodes(items)

    def upsert_relations(self, relations: list[Relation]) -> None:
        """Adds or replaces directed, typed edges ``(source) -[label]-> (target)``."""

        items = [
            {
                "src": relation.source_id,
                "relation": relation.label,
                "dst": relation.target_id,
                "properties": dict(relation.properties or {}),
            }
            for relation in relations
        ]
        if items:
            self._kg.add_edges(items)

    def delete(
        self,
        entity_names: list[str] | None = None,
        relation_names: list[str] | None = None,
        properties: dict | None = None,
        ids: list[str] | None = None,
    ) -> None:
        """Deletes matching relations and (by ``ids``/``properties``) nodes.

        Relations are resolved via :meth:`get_triplets` and removed; nodes are removed only
        when ``ids`` or ``properties`` is given, so a fully unfiltered call never wipes the
        graph by accident (removing a node also drops its incident edges).
        """

        for _src, relation, _dst in self.get_triplets(
            entity_names=entity_names,
            relation_names=relation_names,
            properties=properties,
            ids=ids,
        ):
            self._kg.remove_edge(f"{relation.source_id}:{relation.label}:{relation.target_id}")
        if ids or properties:
            for node in self.get(properties=properties, ids=ids):
                self._kg.remove_node(node.id)

    # -- reads --------------------------------------------------------------

    def get(
        self,
        properties: dict | None = None,
        ids: list[str] | None = None,
    ) -> list[LabelledNode]:
        """Returns nodes by ``ids`` and/or matching ``properties`` (any key matches)."""

        if ids and not properties:
            out: list[LabelledNode] = []
            for node_id in ids:
                node = self._kg.get_node(node_id)
                if node is not None:
                    out.append(self._node_from_kg(node))
            return out

        nodes = self._kg.list_nodes()
        if properties:
            nodes = [
                node
                for node in nodes
                if any(node.properties.get(k) == v for k, v in properties.items())
            ]
        if ids:
            id_set = set(ids)
            nodes = [node for node in nodes if node.id in id_set]
        return [self._node_from_kg(node) for node in nodes]

    def get_triplets(
        self,
        entity_names: list[str] | None = None,
        relation_names: list[str] | None = None,
        properties: dict | None = None,
        ids: list[str] | None = None,
    ) -> list[Triplet]:
        """Returns ``(source, relation, target)`` triplets matching the given filters.

        With no filters this returns nothing (matching LlamaIndex's reference store). Filters
        compose as in that store: ``entity_names``/``ids`` match a triplet's source *or* target
        id; ``relation_names`` match the relation label; ``properties`` match any of the three
        elements' properties.
        """

        if not entity_names and not relation_names and not properties and not ids:
            return []

        seed_ids = set(entity_names or []) | set(ids or [])
        if seed_ids:
            edges: dict[str, Edge] = {}
            for node_id in seed_ids:
                for edge in self._kg.neighbors(node_id, direction="both"):
                    edges[edge.id] = edge
            edge_list = list(edges.values())
        else:
            edge_list = self._kg.list_edges()

        cache: dict[str, LabelledNode | None] = {}

        def resolve(node_id: str) -> LabelledNode | None:
            if node_id not in cache:
                node = self._kg.get_node(node_id)
                cache[node_id] = self._node_from_kg(node) if node is not None else None
            return cache[node_id]

        triplets: list[Triplet] = []
        for edge in edge_list:
            source = resolve(edge.src)
            target = resolve(edge.dst)
            if source is None or target is None:
                continue  # dangling edge endpoint
            relation = Relation(
                label=edge.relation,
                source_id=edge.src,
                target_id=edge.dst,
                properties=dict(edge.properties or {}),
            )
            triplets.append((source, relation, target))

        if entity_names:
            names = set(entity_names)
            triplets = [t for t in triplets if t[0].id in names or t[2].id in names]
        if relation_names:
            rels = set(relation_names)
            triplets = [t for t in triplets if t[1].id in rels]
        if properties:
            triplets = [
                t
                for t in triplets
                if any(
                    t[0].properties.get(k) == v
                    or t[1].properties.get(k) == v
                    or t[2].properties.get(k) == v
                    for k, v in properties.items()
                )
            ]
        if ids:
            id_set = set(ids)
            triplets = [t for t in triplets if t[0].id in id_set or t[2].id in id_set]
        return triplets

    def get_rel_map(
        self,
        graph_nodes: list[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: list[str] | None = None,
    ) -> list[Triplet]:
        """Returns the depth-bounded triplet neighbourhood around ``graph_nodes``.

        Breadth-first over the topology from the seed nodes, following the target side at each
        hop, deduping triplets, dropping ``ignore_rels``, and capping at ``limit`` — the same
        expansion shape as LlamaIndex's reference store.
        """

        triplets: list[Triplet] = []
        seen: set[str] = set()
        frontier = self.get_triplets(ids=[node.id for node in graph_nodes])
        current_depth = 0
        while frontier and current_depth < depth:
            triplets.extend(frontier)
            frontier = self.get_triplets(entity_names=[t[2].id for t in frontier])
            frontier = [t for t in frontier if str(t) not in seen]
            seen.update(str(t) for t in frontier)
            current_depth += 1

        ignore = set(ignore_rels or [])
        triplets = [t for t in triplets if t[1].id not in ignore]
        return triplets[:limit]

    def vector_query(
        self,
        query: VectorStoreQuery,
        **_: Any,
    ) -> tuple[list[LabelledNode], list[float]]:
        """Returns the top-``k`` nodes for ``query``, with their scores.

        Prefers ``query.query_str`` (text-path; LodeDB embeds the query and honors the mapped
        ``query.mode``); falls back to ``query.query_embedding`` for a pure vector lookup when
        no query string is given (see the module docstring on matching the embedding model).
        """

        lode_mode = self._resolve_mode(query.mode)
        k = int(query.similarity_top_k or 1)
        if query.query_str and query.query_str.strip():
            results = self._kg.semantic_nodes(query.query_str, k=k, mode=lode_mode)
        elif query.query_embedding is not None:
            if lode_mode != "vector":
                raise NotImplementedError(
                    "LodeDBPropertyGraphStore: hybrid/lexical modes need query.query_str; an "
                    "embedding-only query can only run a pure vector search."
                )
            results = self._kg.semantic_nodes(embedding=query.query_embedding, k=k)
        else:
            raise ValueError(
                "LodeDBPropertyGraphStore.vector_query needs query.query_str or "
                "query.query_embedding."
            )
        nodes = [self._node_from_kg(node) for _score, node in results]
        scores = [score for score, _node in results]
        return nodes, scores

    def structured_query(self, query: str, param_map: dict[str, Any] | None = None) -> Any:
        """Unsupported: LodeDB's graph layer has no structured (e.g. Cypher) query engine."""

        raise NotImplementedError(
            "LodeDBPropertyGraphStore does not support structured_query (no Cypher engine); "
            "use get / get_triplets / get_rel_map / vector_query."
        )

    def persist(self, persist_path: str, fs: Any = None) -> None:
        """Forces a durability checkpoint on the wrapped graph.

        ``persist_path``/``fs`` are accepted for signature compatibility but ignored: the
        :class:`KnowledgeGraph` persists to its own constructor ``path``.
        """

        self._kg.persist()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _node_to_kg(node: LabelledNode) -> dict[str, Any]:
        """Builds :meth:`KnowledgeGraph.add_nodes` args for one LlamaIndex node (text-path)."""

        properties = dict(node.properties or {})
        if isinstance(node, ChunkNode):
            properties[_KIND_KEY] = _CHUNK
            return {
                "id": node.id,
                "type": node.label,
                "label": node.text,
                "properties": properties,
            }
        # EntityNode (or a bare LabelledNode): embed by the entity name.
        properties[_KIND_KEY] = _ENTITY
        name = getattr(node, "name", node.id)
        return {
            "id": node.id,
            "type": node.label,
            "label": name,
            "properties": properties,
        }

    @staticmethod
    def _node_from_kg(node: Node) -> LabelledNode:
        """Reconstructs a LlamaIndex node from a graph node (reserved kind stripped)."""

        properties = dict(node.properties)
        kind = properties.pop(_KIND_KEY, None)
        if kind == _CHUNK:
            return ChunkNode(text=node.label, id_=node.id, properties=properties)
        return EntityNode(name=node.id, label=node.type or "entity", properties=properties)

    @staticmethod
    def _resolve_mode(mode: VectorStoreQueryMode) -> str:
        """Maps a LlamaIndex query mode to a LodeDB search mode, or raises if unsupported."""

        lode_mode = _MODE_TO_LODE.get(mode)
        if lode_mode is None:
            raise NotImplementedError(
                f"LodeDBPropertyGraphStore does not support query mode {mode!r}; supported "
                "modes are DEFAULT (vector), HYBRID/SEMANTIC_HYBRID (BM25 + RRF), and "
                "SPARSE/TEXT_SEARCH (lexical)."
            )
        return lode_mode


__all__ = ["LodeDBPropertyGraphStore"]
