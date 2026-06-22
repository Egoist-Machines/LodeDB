"""Tests for the lodedb.graph knowledge-graph layer.

Topology operations (CRUD, traversal, cascade removal, reindex, persistence) are
asserted directly. Semantic retrieval is exercised through the vector-in path
with orthogonal one-hot vectors so the nearest-neighbour result is deterministic
(``HashEmbeddingBackend`` is a non-semantic hash, unsuitable for ranking asserts).
"""

from __future__ import annotations

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.graph import KnowledgeGraph

DIM = 384


def _kg(path, **kwargs) -> KnowledgeGraph:
    return KnowledgeGraph(
        path=path,
        model="minilm",
        _embedding_backend=HashEmbeddingBackend(native_dim=DIM),
        **kwargs,
    )


def _onehot(i: int, dim: int = DIM) -> list[float]:
    vector = [0.0] * dim
    vector[i] = 1.0
    return vector


def test_add_and_get_node_and_edge(tmp_path):
    kg = _kg(tmp_path)
    kg.add_node(id="alice", type="Person", label="Alice", properties={"age": 30})
    kg.add_node(id="acme", type="Org", label="Acme")
    kg.add_edge("alice", "works_at", "acme", properties={"since": 2021})

    alice = kg.get_node("alice")
    assert alice is not None
    assert alice.type == "Person"
    assert alice.properties == {"age": 30}
    edge = kg.get_edge("alice:works_at:acme")
    assert edge is not None
    assert (edge.src, edge.relation, edge.dst) == ("alice", "works_at", "acme")
    assert edge.properties == {"since": 2021}
    assert kg.get_node("nobody") is None


def test_neighbors_directions_and_relation_filter(tmp_path):
    kg = _kg(tmp_path)
    for n in ("a", "b", "c"):
        kg.add_node(id=n, label=n)
    kg.add_edge("a", "knows", "b")
    kg.add_edge("a", "likes", "c")
    kg.add_edge("c", "knows", "a")

    out = {(e.relation, e.dst) for e in kg.neighbors("a", direction="out")}
    assert out == {("knows", "b"), ("likes", "c")}
    incoming = {(e.src, e.relation) for e in kg.neighbors("a", direction="in")}
    assert incoming == {("c", "knows")}
    both = kg.neighbors("a", direction="both")
    assert len(both) == 3
    knows_only = {e.dst for e in kg.neighbors("a", direction="out", relation="knows")}
    assert knows_only == {"b"}


def test_k_hop_expansion(tmp_path):
    kg = _kg(tmp_path)
    # chain a -> b -> c -> d
    for n in ("a", "b", "c", "d"):
        kg.add_node(id=n, label=n)
    kg.add_edge("a", "next", "b")
    kg.add_edge("b", "next", "c")
    kg.add_edge("c", "next", "d")

    assert set(kg.k_hop("a", k=1, direction="out").nodes) == {"a", "b"}
    assert set(kg.k_hop("a", k=2, direction="out").nodes) == {"a", "b", "c"}
    assert set(kg.k_hop("a", k=3, direction="out").nodes) == {"a", "b", "c", "d"}
    # direction matters: nothing flows into 'a'
    assert set(kg.k_hop("a", k=3, direction="in").nodes) == {"a"}


def test_semantic_nodes_with_vectors(tmp_path):
    kg = _kg(tmp_path)
    kg.add_node(id="p0", type="Person", embedding=_onehot(0))
    kg.add_node(id="p1", type="Person", embedding=_onehot(40))
    kg.add_node(id="o0", type="Org", embedding=_onehot(80))

    top = kg.semantic_nodes(embedding=_onehot(40), k=3)
    assert top[0][1].id == "p1"

    # node_type narrows the semantic search to Person docs only
    people = kg.semantic_nodes(embedding=_onehot(80), k=5, node_type="Person")
    assert {node.id for _score, node in people} == {"p0", "p1"}


def test_search_subgraph_hybrid(tmp_path):
    kg = _kg(tmp_path)
    kg.add_node(id="alice", type="Person", embedding=_onehot(0))
    kg.add_node(id="acme", type="Org", embedding=_onehot(40))
    kg.add_node(id="nyc", type="Place", embedding=_onehot(80))
    kg.add_edge("alice", "works_at", "acme")
    kg.add_edge("acme", "hq_in", "nyc")

    sub = kg.search_subgraph(embedding=_onehot(0), k=1, hops=1)
    assert sub.seeds and sub.seeds[0][0] == "alice"
    # 1-hop around alice reaches acme
    assert {"alice", "acme"} <= set(sub.nodes)
    assert any(e.id == "alice:works_at:acme" for e in sub.edges)


def test_remove_node_cascades_edges_and_index(tmp_path):
    kg = _kg(tmp_path, index_edges=True)
    kg.add_node(id="a", label="a")
    kg.add_node(id="b", label="b")
    kg.add_edge("a", "knows", "b", fact="a knows b")
    assert kg.stats()["edges"] == 1

    assert kg.remove_node("a") is True
    assert kg.get_node("a") is None
    assert kg.stats()["edges"] == 0  # incident edge removed
    assert kg.neighbors("b", direction="in") == []
    # index docs for the node and its edge are gone (only node 'b' remains)
    assert kg.stats()["indexed_documents"] == 1


def test_remove_edge(tmp_path):
    kg = _kg(tmp_path)
    kg.add_node(id="a", label="a")
    kg.add_node(id="b", label="b")
    eid = kg.add_edge("a", "knows", "b")
    assert kg.remove_edge(eid) is True
    assert kg.get_edge(eid) is None
    assert kg.remove_edge(eid) is False


def test_index_edges_disabled_by_default(tmp_path):
    kg = _kg(tmp_path)  # index_edges=False
    kg.add_node(id="a", label="a")
    kg.add_node(id="b", label="b")
    kg.add_edge("a", "knows", "b", fact="a knows b")
    # edge not indexed -> no semantic edge hits, and only 2 node docs indexed
    assert kg.semantic_edges("knows") == []
    assert kg.stats()["indexed_documents"] == 2


def test_reindex_rebuilds_and_drops_orphans(tmp_path):
    kg = _kg(tmp_path)
    kg.add_node(id="a", label="alpha")
    kg.add_node(id="b", label="beta")
    assert kg.stats()["indexed_documents"] == 2

    # Simulate a lost index write (topology has the node, index doesn't).
    kg._db.remove("n:a")
    assert kg.stats()["indexed_documents"] == 1
    # Simulate an orphan index doc (node deleted out-of-band from topology only).
    kg._store.remove_node("b")

    report = kg.reindex()
    assert report["removed_orphans"] == 1  # the dangling 'n:b' doc
    assert report["reindexed_nodes"] == 1  # 'a' re-embedded
    assert kg.stats()["indexed_documents"] == 1
    assert {r["id"] for r in kg._db.list_documents()} == {"n:a"}


def test_k_hop_large_frontier_chunks_in_clause(tmp_path):
    # A frontier larger than the IN-clause chunk size must still traverse
    # correctly (the edges_for query is chunked to stay under SQLite's
    # bound-parameter limit). Hub connected to 500 leaves -> 2-hop both reaches all.
    kg = _kg(tmp_path)
    kg.add_node(id="hub", label="hub")
    leaves = 500
    for i in range(leaves):
        kg.add_node(id=f"leaf{i}", label=f"leaf{i}")
        kg.add_edge("hub", "has", f"leaf{i}")
    sub = kg.k_hop("hub", k=2, direction="both")
    assert len(sub.nodes) == leaves + 1
    assert len(sub.edges) == leaves


def test_persistence_roundtrip(tmp_path):
    kg = _kg(tmp_path, index_edges=True)
    kg.add_node(id="alice", type="Person", embedding=_onehot(0))
    kg.add_node(id="acme", type="Org", embedding=_onehot(40))
    kg.add_edge("alice", "works_at", "acme", fact="alice works at acme")
    kg.persist()
    kg.close()

    reopened = _kg(tmp_path, index_edges=True)
    assert reopened.stats()["nodes"] == 2
    assert reopened.stats()["edges"] == 1
    assert set(reopened.k_hop("alice", k=1).nodes) == {"alice", "acme"}
    assert reopened.semantic_nodes(embedding=_onehot(0), k=1)[0][1].id == "alice"


def test_read_only_graph_blocks_writes(tmp_path):
    writer = _kg(tmp_path)
    writer.add_node(id="a", label="a")
    writer.persist()
    writer.close()

    reader = _kg(tmp_path, read_only=True)
    # reads work
    assert reader.get_node("a") is not None
    # writes through the index are blocked by the read-only LodeDB handle
    from lodedb.local.db import ReadOnlyError

    with pytest.raises(ReadOnlyError):
        reader.add_node(id="b", label="b")
