"""KnowledgeGraph is a hybrid graph/knowledge-graph layer over LodeDB.

The architecture is the one every comparable agent-memory system uses (Zep/
Graphiti, cognee): keep the **topology** in a store built for traversal and use
a vector index for **semantic entry points**, then expand the graph from there.
Here the topology is an embedded SQLite sidecar (:mod:`lodedb.graph._store`) and
the semantic index is LodeDB.

Two design choices make this robust rather than a leaky two-database hack:

- **SQLite is the source of truth; LodeDB is a rebuildable index.** Writes hit
  SQLite first, then the index. If an index write is lost (crash between the
  two), the topology is still correct and :meth:`reindex` rebuilds the index
  from it, so there is no cross-store atomicity problem to get wrong.
- **Retrieval is hybrid.** :meth:`search_subgraph` runs a LodeDB similarity
  search to find entry-point nodes, then does deterministic k-hop expansion over
  the SQLite adjacency, the complete-set traversal LodeDB's top-k search can't
  express on its own.

Nodes are embedded by their ``label`` text (LodeDB embeds internally) or by a
caller-supplied vector (the vector-in path), so an external embedder can be
reused. Edges are topology by default; pass ``index_edges=True`` to also index
edge "facts" for semantic edge search.
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lodedb.local.db import LodeDB, ReadOnlyError

from ._store import Edge, Node, TopologyStore

_NODE_PREFIX = "n:"
_EDGE_PREFIX = "e:"

logger = logging.getLogger("lodedb.graph")


class _IngestBuffer:
    """Collects add_node/add_edge calls inside a :meth:`KnowledgeGraph.ingest`
    block so they can flush as one batched write per kind (one index commit each)."""

    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []

    def add_node(
        self,
        *,
        id: str | None = None,
        type: str = "",
        label: str = "",
        properties: Mapping[str, Any] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> str:
        """Buffers a node; returns its id (auto-assigned when not given)."""

        node_id = str(id) if id is not None else f"node-{secrets.token_hex(8)}"
        self.nodes.append(
            {
                "id": node_id,
                "type": type,
                "label": label,
                "properties": properties,
                "embedding": embedding,
            }
        )
        return node_id

    def add_edge(
        self,
        src: str,
        relation: str,
        dst: str,
        *,
        id: str | None = None,
        properties: Mapping[str, Any] | None = None,
        fact: str | None = None,
        embedding: Sequence[float] | None = None,
    ) -> str:
        """Buffers an edge; returns its id (derived from the triple when not given)."""

        edge_id = str(id) if id is not None else f"{src}:{relation}:{dst}"
        self.edges.append(
            {
                "src": src,
                "relation": relation,
                "dst": dst,
                "id": edge_id,
                "properties": properties,
                "fact": fact,
                "embedding": embedding,
            }
        )
        return edge_id


@dataclass
class Subgraph:
    """A retrieved neighbourhood: nodes by id, the edges among them, and the
    semantic seed nodes (with scores) the expansion started from."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    seeds: list[tuple[str, float]] = field(default_factory=list)

    def __len__(self) -> int:
        """Returns the number of nodes in the subgraph."""

        return len(self.nodes)


class KnowledgeGraph:
    """A knowledge graph backed by SQLite (topology) + LodeDB (semantic index).

    Example::

        kg = KnowledgeGraph(path="./kg")
        kg.add_node(id="alice", type="Person", label="Alice, software engineer")
        kg.add_node(id="acme",  type="Org",    label="Acme Corp")
        kg.add_edge("alice", "works_at", "acme")

        sub = kg.search_subgraph("who works in engineering?", k=3, hops=1)
        for node_id, score in sub.seeds:
            ...                      # semantic entry points
        for edge in sub.edges:
            ...                      # their 1-hop neighbourhood
    """

    def __init__(
        self,
        path: str | Path,
        *,
        model: str = "minilm",
        device: str = "auto",
        vector_dim: int | None = None,
        bit_width: int = 4,
        index_edges: bool = False,
        store_text: bool = True,
        read_only: bool = False,
        retain_vectors: bool = False,
        _embedding_backend: Any | None = None,
        **lodedb_kwargs: Any,
    ) -> None:
        """Opens (or creates) a knowledge graph rooted at ``path``.

        Lays down ``path/topology.sqlite3`` (the source-of-truth adjacency) and
        ``path/index`` (the LodeDB semantic index). ``model``/``device`` and any
        extra ``lodedb_kwargs`` are forwarded to the underlying :class:`LodeDB`.
        ``index_edges=True`` also indexes edge facts for semantic edge search.

        ``vector_dim`` opens the semantic index as a *vector-only* index at that
        dimension (any value your own embedder produces, e.g. 384, 768, 1536) with
        **no internal embedding model**: nodes (and edges) are indexed only by a
        caller-supplied ``embedding``, label/fact text becomes topology-only, and
        semantic retrieval takes a precomputed query vector. This is the
        bring-your-own-embeddings path for a framework that owns its embedder, so
        ``model``/``device`` are ignored. ``bit_width`` sets the TurboVec
        quantization width for that index.

        ``retain_vectors=True`` keeps a copy of each node's precomputed embedding
        in the topology store (the source of truth), so :meth:`reindex` can
        faithfully rebuild the semantic index for vector-in nodes too, not just
        labelled ones. It trades roughly 4 bytes per dimension per node of extra
        on-disk space for that rebuildability; leave it off for label-only graphs.
        """

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.index_edges = bool(index_edges)
        self.read_only = bool(read_only)
        self.retain_vectors = bool(retain_vectors)
        self.vector_only = vector_dim is not None
        self._store = TopologyStore(self.path / "topology.sqlite3")
        if self.vector_only:
            # Bring-your-own-vectors: the semantic index has no embedder, so nodes and
            # edges are indexed by a caller-supplied embedding at an arbitrary dimension
            # and label/fact text is topology-only (kept in SQLite, not embedded).
            self._db = LodeDB(
                self.path / "index",
                vector_dim=int(vector_dim),
                bit_width=int(bit_width),
                store_text=False,
                read_only=read_only,
                **lodedb_kwargs,
            )
        else:
            self._db = LodeDB(
                self.path / "index",
                model=model,
                device=device,
                store_text=store_text,
                read_only=read_only,
                _embedding_backend=_embedding_backend,
                **lodedb_kwargs,
            )

    # -- mutation -----------------------------------------------------------

    def add_node(
        self,
        *,
        id: str | None = None,
        type: str = "",
        label: str = "",
        properties: Mapping[str, Any] | None = None,
        embedding: Sequence[float] | None = None,
    ) -> str:
        """Adds or replaces a node and (re)indexes it for semantic retrieval.

        The node is embedded by ``embedding`` if given (vector-in), else by its
        ``label`` text; a node with neither is stored in the topology but is not
        semantically searchable. Returns the node id.
        """

        self._require_writable()
        node_id = str(id) if id is not None else f"node-{secrets.token_hex(8)}"
        node = Node(
            id=node_id,
            type=str(type),
            label=str(label),
            properties=dict(properties or {}),
        )
        self._store.upsert_node(node)
        self._index_node(node, embedding=embedding)
        return node_id

    def add_edge(
        self,
        src: str,
        relation: str,
        dst: str,
        *,
        id: str | None = None,
        properties: Mapping[str, Any] | None = None,
        fact: str | None = None,
        embedding: Sequence[float] | None = None,
    ) -> str:
        """Adds or replaces a directed, typed edge ``(src) -[relation]-> (dst)``.

        A missing ``id`` derives a stable id from the triple, so re-adding the
        same triple upserts. When ``index_edges`` is enabled and a ``fact`` text
        (or ``embedding``) is supplied, the edge is also indexed for semantic
        edge search. Returns the edge id.
        """

        self._require_writable()
        edge_id = str(id) if id is not None else f"{src}:{relation}:{dst}"
        edge = Edge(
            id=edge_id,
            src=str(src),
            relation=str(relation),
            dst=str(dst),
            properties=dict(properties or {}),
        )
        self._store.upsert_edge(edge)
        if self.index_edges:
            self._index_edge(edge, fact=fact, embedding=embedding)
        return edge_id

    def add_nodes(self, nodes: list[Mapping[str, Any]]) -> list[str]:
        """Adds or replaces many nodes in one batch.

        One SQLite transaction plus a single index commit per kind (text label vs
        precomputed embedding), versus one index commit per node with
        :meth:`add_node`. Each item is ``{"id"?, "type"?, "label"?, "properties"?,
        "embedding"?}``. Returns the ids in input order.
        """

        self._require_writable()
        node_objs: list[Node] = []
        ids: list[str] = []
        text_docs: list[dict[str, Any]] = []
        vector_docs: list[dict[str, Any]] = []
        stale_doc_ids: list[str] = []
        retained_vectors: list[tuple[str, Sequence[float]]] = []
        stale_retained_vector_ids: list[str] = []
        for item in nodes:
            raw_id = item.get("id")
            node_id = str(raw_id) if raw_id is not None else f"node-{secrets.token_hex(8)}"
            node = Node(
                id=node_id,
                type=str(item.get("type", "")),
                label=str(item.get("label", "")),
                properties=dict(item.get("properties") or {}),
            )
            node_objs.append(node)
            ids.append(node_id)
            metadata = {"kind": "node", "type": node.type, "node_id": node.id}
            embedding = item.get("embedding")
            if embedding is not None:
                vector_docs.append(
                    {"vector": embedding, "id": _NODE_PREFIX + node.id, "metadata": metadata}
                )
                if self.retain_vectors:
                    retained_vectors.append((node.id, embedding))
            elif not self.vector_only and node.label.strip():
                text_docs.append(
                    {"text": node.label, "id": _NODE_PREFIX + node.id, "metadata": metadata}
                )
                if self.retain_vectors:
                    stale_retained_vector_ids.append(node.id)
            else:
                stale_doc_ids.append(_NODE_PREFIX + node.id)
                if self.retain_vectors:
                    stale_retained_vector_ids.append(node.id)
        self._store.upsert_nodes(node_objs)
        for node_id in stale_retained_vector_ids:
            self._store.remove_node_vector(node_id)
        if self.retain_vectors and retained_vectors:
            self._store.set_node_vectors(retained_vectors)
        for doc_id in stale_doc_ids:
            self._db.remove(doc_id)
        if text_docs:
            self._db.add_many(text_docs)
        if vector_docs:
            self._db.add_vectors_many(vector_docs)
        return ids

    def add_edges(self, edges: list[Mapping[str, Any]]) -> list[str]:
        """Adds or replaces many edges in one batch.

        One SQLite transaction plus, when ``index_edges`` is on, one index commit
        per kind. Each item is ``{"src", "relation", "dst", "id"?, "properties"?,
        "fact"?, "embedding"?}``. Returns the ids in input order.
        """

        self._require_writable()
        edge_objs: list[Edge] = []
        ids: list[str] = []
        text_docs: list[dict[str, Any]] = []
        vector_docs: list[dict[str, Any]] = []
        stale_doc_ids: list[str] = []
        for item in edges:
            src = str(item["src"])
            relation = str(item["relation"])
            dst = str(item["dst"])
            raw_id = item.get("id")
            edge_id = str(raw_id) if raw_id is not None else f"{src}:{relation}:{dst}"
            edge_objs.append(
                Edge(
                    id=edge_id,
                    src=src,
                    relation=relation,
                    dst=dst,
                    properties=dict(item.get("properties") or {}),
                )
            )
            ids.append(edge_id)
            if self.index_edges:
                metadata = {
                    "kind": "edge",
                    "relation": relation,
                    "src": src,
                    "dst": dst,
                    "edge_id": edge_id,
                }
                embedding = item.get("embedding")
                fact = item.get("fact")
                if embedding is not None:
                    vector_docs.append(
                        {"vector": embedding, "id": _EDGE_PREFIX + edge_id, "metadata": metadata}
                    )
                elif not self.vector_only and fact and str(fact).strip():
                    text_docs.append(
                        {"text": fact, "id": _EDGE_PREFIX + edge_id, "metadata": metadata}
                    )
                else:
                    stale_doc_ids.append(_EDGE_PREFIX + edge_id)
        self._store.upsert_edges(edge_objs)
        for doc_id in stale_doc_ids:
            self._db.remove(doc_id)
        if text_docs:
            self._db.add_many(text_docs)
        if vector_docs:
            self._db.add_vectors_many(vector_docs)
        return ids

    @contextmanager
    def ingest(self) -> Iterator[_IngestBuffer]:
        """Buffers add_node/add_edge calls and flushes them as batched writes on exit.

        Inside the block, use the yielded buffer's ``add_node``/``add_edge`` (same
        arguments as the graph's own). On exit they are applied via
        :meth:`add_nodes` / :meth:`add_edges`, so a bulk load pays one SQLite
        transaction and one index commit per kind instead of one commit per entity::

            with kg.ingest() as batch:
                for row in rows:
                    batch.add_node(id=row.id, label=row.text)
        """

        self._require_writable()
        buffer = _IngestBuffer()
        yield buffer
        if buffer.nodes:
            self.add_nodes(buffer.nodes)
        if buffer.edges:
            self.add_edges(buffer.edges)

    def remove_node(self, node_id: str) -> bool:
        """Removes a node and its incident edges from both stores.

        Returns True if the node existed.
        """

        self._require_writable()
        existed, removed_edge_ids = self._store.remove_node(str(node_id))
        self._db.remove(_NODE_PREFIX + str(node_id))
        if self.index_edges:
            for edge_id in removed_edge_ids:
                self._db.remove(_EDGE_PREFIX + edge_id)
        return existed

    def remove_edge(self, edge_id: str) -> bool:
        """Removes one edge from both stores. Returns True if it existed."""

        self._require_writable()
        existed = self._store.remove_edge(str(edge_id))
        if self.index_edges:
            self._db.remove(_EDGE_PREFIX + str(edge_id))
        return existed

    # -- topology reads / traversal -----------------------------------------

    def get_node(self, node_id: str) -> Node | None:
        """Returns the node with ``node_id`` or ``None``."""

        return self._store.get_node(str(node_id))

    def get_edge(self, edge_id: str) -> Edge | None:
        """Returns the edge with ``edge_id`` or ``None``."""

        return self._store.get_edge(str(edge_id))

    def neighbors(
        self,
        node_id: str,
        *,
        direction: str = "out",
        relation: str | None = None,
    ) -> list[Edge]:
        """Returns edges incident to ``node_id`` (out/in/both, optional relation)."""

        return self._store.neighbors(str(node_id), direction=direction, relation=relation)

    def list_nodes(self) -> list[Node]:
        """Returns every node in the graph (complete-set enumeration).

        The deterministic counterpart to :meth:`semantic_nodes`: no ranking, no
        ``k`` cap, just the full node set from the source-of-truth topology store.
        This is the primitive a property-graph view needs ("every node", or every
        node whose property matches) that a top-``k`` similarity search cannot
        express on its own.
        """

        return list(self._store.iter_nodes())

    def list_edges(self) -> list[Edge]:
        """Returns every edge in the graph (complete-set enumeration).

        Like :meth:`list_nodes`, full enumeration over the topology store, the
        primitive behind an "all triplets" / "all edges of relation R" view.
        """

        return list(self._store.iter_edges())

    def k_hop(
        self,
        seeds: str | Sequence[str],
        *,
        k: int = 1,
        direction: str = "both",
        relation: str | None = None,
        max_nodes: int | None = None,
    ) -> Subgraph:
        """Expands the ``k``-hop neighbourhood around ``seeds`` over the topology.

        This is the deterministic, complete-set traversal (BFS) that a top-``k``
        similarity search cannot express: every node within ``k`` hops and every
        edge traversed is returned. ``max_nodes`` optionally caps the visited set
        (a budget for dense graphs); when it truncates, a warning is logged so the
        cap is never silent and the result is a partial subgraph.
        """

        seed_ids = [str(seeds)] if isinstance(seeds, str) else [str(value) for value in seeds]
        visited: set[str] = set(seed_ids)
        frontier: set[str] = set(seed_ids)
        edges_seen: dict[str, Edge] = {}
        truncated = False
        for _hop in range(max(0, int(k))):
            if not frontier or truncated:
                break
            next_frontier: set[str] = set()
            for edge in self._store.edges_for(frontier, direction=direction, relation=relation):
                edges_seen[edge.id] = edge
                for endpoint in (edge.src, edge.dst):
                    if endpoint in visited:
                        continue
                    if max_nodes is not None and len(visited) >= max_nodes:
                        truncated = True
                        continue
                    visited.add(endpoint)
                    next_frontier.add(endpoint)
            frontier = next_frontier
        if truncated:
            logger.warning(
                "k_hop truncated at max_nodes=%s (visited %d nodes); subgraph is partial",
                max_nodes,
                len(visited),
            )
        # Batched read: one query per _IN_CHUNK ids, not one round-trip per node.
        nodes = {node.id: node for node in self._store.get_nodes(visited)}
        return Subgraph(nodes=nodes, edges=list(edges_seen.values()))

    # -- semantic retrieval -------------------------------------------------

    def semantic_nodes(
        self,
        query: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 10,
        node_type: str | None = None,
        filter: Mapping[str, Any] | None = None,
        mode: str | None = None,
    ) -> list[tuple[float, Node]]:
        """Returns the top-``k`` nodes most relevant to ``query`` (or ``embedding``).

        This is the semantic entry-point step: a LodeDB similarity search scoped
        to node documents, optionally narrowed by ``node_type`` or an arbitrary
        metadata ``filter`` (same predicate grammar as :meth:`LodeDB.search`).
        ``mode`` matches :meth:`LodeDB.search`: left unset it defaults to
        ``"hybrid"`` when a lexical source is available (``store_text=True`` or
        ``index_text=True``, both on by default) and to ``"vector"`` otherwise, so
        exact tokens in node labels (error codes, serials, dates) the embedding
        misses are matched by default. Lexical ranking uses the ``query`` text, so
        an explicit ``"hybrid"``/``"lexical"`` mode cannot be combined with a
        precomputed ``embedding`` (a pure vector query).
        """

        hits = self._search_index(
            query,
            embedding=embedding,
            k=k,
            kind="node",
            entity_type=node_type,
            extra_filter=filter,
            mode=mode,
        )
        hit_node_ids = [
            hit.metadata.get("node_id") or _strip(hit.id, _NODE_PREFIX) for hit in hits
        ]
        nodes_by_id = {node.id: node for node in self._store.get_nodes(hit_node_ids)}
        out: list[tuple[float, Node]] = []
        for hit, node_id in zip(hits, hit_node_ids, strict=True):
            node = nodes_by_id.get(node_id)
            if node is not None:
                out.append((hit.score, node))
        return out

    def semantic_edges(
        self,
        query: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 10,
        relation: str | None = None,
        mode: str | None = None,
    ) -> list[tuple[float, Edge]]:
        """Returns the top-``k`` indexed edges most relevant to ``query``.

        Requires ``index_edges=True`` and that the edges were added with a
        ``fact`` (or ``embedding``); otherwise the result is empty. ``mode``
        matches :meth:`semantic_nodes` (unset defaults to ``"hybrid"`` when a
        lexical source is available, else ``"vector"``, over the edge ``fact``
        text).
        """

        if not self.index_edges:
            return []
        extra = {"relation": relation} if relation is not None else None
        hits = self._search_index(
            query,
            embedding=embedding,
            k=k,
            kind="edge",
            entity_type=None,
            extra_filter=extra,
            mode=mode,
        )
        out: list[tuple[float, Edge]] = []
        for hit in hits:
            edge_id = hit.metadata.get("edge_id") or _strip(hit.id, _EDGE_PREFIX)
            edge = self._store.get_edge(edge_id)
            if edge is not None:
                out.append((hit.score, edge))
        return out

    def search_subgraph(
        self,
        query: str | None = None,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
        hops: int = 1,
        direction: str = "both",
        relation: str | None = None,
        node_type: str | None = None,
        filter: Mapping[str, Any] | None = None,
        mode: str | None = None,
    ) -> Subgraph:
        """Semantic entry points + k-hop graph expansion.

        Finds the ``k`` most relevant nodes for ``query``/``embedding`` (the
        semantic step), then expands ``hops`` hops around them over the topology
        (the structural step), returning the combined subgraph with the seed
        nodes and their scores recorded on :attr:`Subgraph.seeds`. ``mode`` is
        forwarded to the seed search (:meth:`semantic_nodes`), which by default
        fuses a lexical BM25 ranker with the vector seeds (hybrid) so exact tokens
        in node labels are not missed; pass ``mode="vector"`` for vector-only
        seeds. This is orthogonal to the structural expansion, which composes
        either way.
        """

        seeds = self.semantic_nodes(
            query,
            embedding=embedding,
            k=k,
            node_type=node_type,
            filter=filter,
            mode=mode,
        )
        subgraph = self.k_hop(
            [node.id for _score, node in seeds],
            k=hops,
            direction=direction,
            relation=relation,
        )
        subgraph.seeds = [(node.id, score) for score, node in seeds]
        # Ensure seed nodes are present even if they had no edges to expand into.
        for _score, node in seeds:
            subgraph.nodes.setdefault(node.id, node)
        return subgraph

    # -- maintenance --------------------------------------------------------

    def reindex(self) -> dict[str, int]:
        """Rebuilds the LodeDB semantic index from the SQLite source of truth.

        Drops index documents for entities no longer in the topology (orphans from
        a crash between the two stores, or out-of-band edits), then re-creates each
        node's index entry: from its retained vector when ``retain_vectors`` kept
        one (so vector-in nodes rebuild faithfully), else from its ``label``. This
        is what makes the index a derived, throwaway artifact.

        Nodes with neither a retained vector nor a label cannot be rebuilt; they
        are counted in ``unrebuildable`` and a warning is logged, never a silent
        corruption. Open the graph with ``retain_vectors=True`` to make vector-in
        nodes rebuildable. Returns counts: ``reindexed_nodes`` (labels + vectors),
        ``reindexed_labels``, ``reindexed_vectors``, ``unrebuildable``,
        ``removed_orphans``.
        """

        self._require_writable()
        node_ids = set(self._store.all_node_ids())
        wanted_node_docs = {_NODE_PREFIX + node_id for node_id in node_ids}
        removed = 0
        for record in self._db.list_documents(filter={"kind": "node"}):
            if record["id"] not in wanted_node_docs:
                self._db.remove(record["id"])
                removed += 1
        retained = dict(self._store.iter_node_vectors()) if self.retain_vectors else {}
        label_docs: list[dict[str, Any]] = []
        vector_docs: list[dict[str, Any]] = []
        unrebuildable = 0
        for node in self._store.iter_nodes():
            metadata = {"kind": "node", "type": node.type, "node_id": node.id}
            vector = retained.get(node.id)
            if vector is not None:
                vector_docs.append(
                    {"vector": vector, "id": _NODE_PREFIX + node.id, "metadata": metadata}
                )
            elif not self.vector_only and node.label.strip():
                label_docs.append(
                    {"text": node.label, "id": _NODE_PREFIX + node.id, "metadata": metadata}
                )
            else:
                unrebuildable += 1
        if label_docs:
            self._db.add_many(label_docs)
        if vector_docs:
            self._db.add_vectors_many(vector_docs)
        if self.index_edges:
            wanted_edge_docs = {_EDGE_PREFIX + edge.id for edge in self._store.iter_edges()}
            for record in self._db.list_documents(filter={"kind": "edge"}):
                if record["id"] not in wanted_edge_docs:
                    self._db.remove(record["id"])
                    removed += 1
        if unrebuildable:
            logger.warning(
                "reindex: %d node(s) have neither a retained vector nor a label and "
                "were not rebuilt; open with retain_vectors=True to make vector-in "
                "nodes rebuildable",
                unrebuildable,
            )
        return {
            "reindexed_nodes": len(label_docs) + len(vector_docs),
            "reindexed_labels": len(label_docs),
            "reindexed_vectors": len(vector_docs),
            "unrebuildable": unrebuildable,
            "removed_orphans": removed,
        }

    def stats(self) -> dict[str, Any]:
        """Returns node/edge counts and the underlying index document count."""

        return {
            "nodes": self._store.node_count(),
            "edges": self._store.edge_count(),
            "indexed_documents": self._db.count(),
            "index_edges": self.index_edges,
        }

    def persist(self) -> None:
        """Checkpoints the LodeDB index (SQLite autocommits each write)."""

        self._db.persist()

    def close(self) -> None:
        """Closes both stores; state stays durable on disk."""

        self._db.close()
        self._store.close()

    def __enter__(self) -> KnowledgeGraph:
        """Enters a context manager."""

        return self

    def __exit__(self, *exc: object) -> None:
        """Closes both stores on context exit."""

        self.close()

    # -- internals ----------------------------------------------------------

    def _require_writable(self) -> None:
        if self.read_only:
            raise ReadOnlyError("this KnowledgeGraph is read-only; reopen without read_only=True")

    def _index_node(self, node: Node, *, embedding: Sequence[float] | None = None) -> None:
        """Indexes (or removes) a node's semantic document to match its label/embedding."""

        doc_id = _NODE_PREFIX + node.id
        metadata = {"kind": "node", "type": node.type, "node_id": node.id}
        if embedding is not None:
            self._db.add_vectors(embedding, id=doc_id, metadata=metadata)
            if self.retain_vectors:
                self._store.set_node_vectors([(node.id, embedding)])
        elif not self.vector_only and node.label.strip():
            self._db.add(node.label, id=doc_id, metadata=metadata)
            if self.retain_vectors:
                self._store.remove_node_vector(node.id)
        else:
            # No semantic content: make sure any stale index doc is cleared.
            self._db.remove(doc_id)
            if self.retain_vectors:
                self._store.remove_node_vector(node.id)

    def _index_edge(
        self,
        edge: Edge,
        *,
        fact: str | None = None,
        embedding: Sequence[float] | None = None,
    ) -> None:
        """Indexes an edge's fact text/embedding for semantic edge search."""

        doc_id = _EDGE_PREFIX + edge.id
        metadata = {
            "kind": "edge",
            "relation": edge.relation,
            "src": edge.src,
            "dst": edge.dst,
            "edge_id": edge.id,
        }
        if embedding is not None:
            self._db.add_vectors(embedding, id=doc_id, metadata=metadata)
        elif not self.vector_only and fact and fact.strip():
            self._db.add(fact, id=doc_id, metadata=metadata)
        else:
            self._db.remove(doc_id)

    def _search_index(
        self,
        query: str | None,
        *,
        embedding: Sequence[float] | None,
        k: int,
        kind: str,
        entity_type: str | None,
        extra_filter: Mapping[str, Any] | None,
        mode: str | None = None,
    ):
        """Runs the scoped LodeDB search shared by semantic_nodes/_edges.

        ``mode`` is forwarded to :meth:`LodeDB.search` on the text-query path
        (unset resolves to ``"hybrid"`` when a lexical source is available, else
        ``"vector"``). A precomputed ``embedding`` is a pure vector query, so an
        explicit lexical/hybrid ``mode`` paired with an ``embedding`` is rejected
        (lexical ranking needs the query string); an unset ``mode`` with an
        ``embedding`` runs the vector query.
        """

        base: dict[str, Any] = {"kind": kind}
        if entity_type is not None:
            base["type"] = entity_type
        if extra_filter:
            search_filter: dict[str, Any] = {"$and": [base, dict(extra_filter)]}
        else:
            search_filter = base
        if embedding is not None:
            if mode not in (None, "vector"):
                raise ValueError(
                    "mode must be 'vector' when searching by embedding; "
                    "'hybrid'/'lexical' rank the query text, not a precomputed vector"
                )
            return self._db.search_by_vector(embedding, k=k, filter=search_filter)
        if not query:
            raise ValueError("provide a query string or an embedding")
        return self._db.search(query, k=k, filter=search_filter, mode=mode)


def _strip(value: str, prefix: str) -> str:
    """Strips a known LodeDB doc-id prefix to recover the entity id."""

    return value[len(prefix):] if value.startswith(prefix) else value
