"""Graph bulk-load and batched topology reads.

`add_nodes`/`add_edges` (and the `ingest()` buffer) write a batch in one SQLite
transaction plus one index commit per kind, instead of one commit per entity.
`k_hop`/`semantic_nodes` materialize the visited set with one batched `get_nodes`
read, and `k_hop(max_nodes=)` caps a dense frontier with a logged truncation.
"""

from __future__ import annotations

import logging

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


def _onehot(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def test_add_nodes_and_edges_batch(tmp_path):
    kg = _kg(tmp_path)
    ids = kg.add_nodes(
        [
            {"id": "a", "type": "Person", "embedding": _onehot(0)},
            {"id": "b", "type": "Org", "embedding": _onehot(40)},
            {"id": "c", "type": "Place", "label": "labelled"},  # mixed: label-indexed
        ]
    )
    assert ids == ["a", "b", "c"]
    edge_ids = kg.add_edges(
        [{"src": "a", "relation": "at", "dst": "b"}, {"src": "b", "relation": "in", "dst": "c"}]
    )
    assert kg.stats()["nodes"] == 3
    assert kg.stats()["edges"] == 2
    assert edge_ids == ["a:at:b", "b:in:c"]
    # traversal + semantic both work over the batched graph
    assert set(kg.k_hop("a", k=2).nodes) == {"a", "b", "c"}
    assert kg.semantic_nodes(embedding=_onehot(40), k=1)[0][1].id == "b"


def test_bulk_node_replacement_clears_stale_semantic_doc(tmp_path):
    """Replacing a node with no semantic content removes its derived index doc."""

    kg = _kg(tmp_path)
    kg.add_nodes([{"id": "a", "label": "alpha E1234"}])
    assert [node.id for _score, node in kg.semantic_nodes("E1234", mode="lexical")] == ["a"]

    kg.add_nodes([{"id": "a", "label": ""}])

    assert kg.get_node("a") is not None
    assert kg.semantic_nodes("E1234", mode="lexical") == []


def test_bulk_edge_replacement_clears_stale_semantic_doc(tmp_path):
    """Replacing an indexed edge without fact/embedding removes its derived index doc."""

    kg = _kg(tmp_path, index_edges=True)
    kg.add_nodes([{"id": "a", "label": "a"}, {"id": "b", "label": "b"}])
    kg.add_edges([{"id": "e", "src": "a", "relation": "rel", "dst": "b", "fact": "edge E1234"}])
    assert [edge.id for _score, edge in kg.semantic_edges("E1234", mode="lexical")] == ["e"]

    kg.add_edges([{"id": "e", "src": "a", "relation": "rel", "dst": "b"}])

    assert kg.get_edge("e") is not None
    assert kg.semantic_edges("E1234", mode="lexical") == []


def test_add_nodes_equivalent_to_per_node(tmp_path):
    one = _kg(tmp_path / "one")
    many = _kg(tmp_path / "many")
    spec = [{"id": f"n{i}", "type": "T", "embedding": _onehot(i * 7)} for i in range(8)]
    for item in spec:
        one.add_node(id=item["id"], type=item["type"], embedding=item["embedding"])
    many.add_nodes(spec)
    edges = [{"src": f"n{i}", "relation": "next", "dst": f"n{i + 1}"} for i in range(7)]
    for e in edges:
        one.add_edge(e["src"], e["relation"], e["dst"])
    many.add_edges(edges)

    assert one.stats() == many.stats()
    assert set(one.k_hop("n0", k=7, direction="out").nodes) == set(
        many.k_hop("n0", k=7, direction="out").nodes
    )
    assert (
        one.semantic_nodes(embedding=_onehot(21), k=1)[0][1].id
        == many.semantic_nodes(embedding=_onehot(21), k=1)[0][1].id
    )


def test_ingest_context_manager(tmp_path):
    kg = _kg(tmp_path)
    with kg.ingest() as batch:
        a = batch.add_node(id="a", embedding=_onehot(0))
        b = batch.add_node(id="b", embedding=_onehot(40))
        batch.add_edge(a, "knows", b)
    assert kg.stats()["nodes"] == 2
    assert kg.stats()["edges"] == 1
    assert set(kg.k_hop("a", k=1).nodes) == {"a", "b"}


def test_k_hop_max_nodes_truncates_with_log(tmp_path, caplog):
    kg = _kg(tmp_path)
    nodes = [{"id": "hub", "label": "hub"}]
    nodes += [{"id": f"leaf{i}", "label": f"l{i}"} for i in range(200)]
    kg.add_nodes(nodes)
    kg.add_edges([{"src": "hub", "relation": "has", "dst": f"leaf{i}"} for i in range(200)])

    with caplog.at_level(logging.WARNING, logger="lodedb.graph"):
        sub = kg.k_hop("hub", k=2, direction="both", max_nodes=50)
    assert len(sub.nodes) == 50  # hub + 49 leaves, then capped
    assert any("truncated" in record.message for record in caplog.records)

    # without the cap, the full neighbourhood is returned
    full = kg.k_hop("hub", k=1, direction="both")
    assert len(full.nodes) == 201
