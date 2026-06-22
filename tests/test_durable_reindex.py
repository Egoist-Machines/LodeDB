"""Finding 04: a vector-in knowledge graph is faithfully rebuildable.

With ``retain_vectors=True`` the topology store keeps each node's raw vector, so
``reindex()`` reconstructs the semantic index for vector-in nodes (not just
labelled ones) after the index is lost. Without it, such nodes are reported as
``unrebuildable`` and a warning is logged, never silently corrupted.
"""

from __future__ import annotations

import logging
import shutil

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


def test_reindex_rebuilds_vector_in_nodes_with_retain(tmp_path):
    kg = _kg(tmp_path, retain_vectors=True)
    for i in range(5):
        kg.add_node(id=f"n{i}", type="T", embedding=_onehot(i * 20))
    before = {i: kg.semantic_nodes(embedding=_onehot(i * 20), k=1)[0][1].id for i in range(5)}
    assert before == {i: f"n{i}" for i in range(5)}
    kg.persist()
    kg.close()

    # Simulate a lost semantic index (it is a derived, throwaway artifact).
    shutil.rmtree(tmp_path / "index")
    kg2 = _kg(tmp_path, retain_vectors=True)
    assert kg2.stats()["indexed_documents"] == 0  # index is gone

    report = kg2.reindex()
    assert report["reindexed_vectors"] == 5
    assert report["unrebuildable"] == 0
    assert kg2.stats()["indexed_documents"] == 5
    # retrieval is byte-identical to before the index was dropped
    after = {i: kg2.semantic_nodes(embedding=_onehot(i * 20), k=1)[0][1].id for i in range(5)}
    assert after == before


def test_reindex_reports_unrebuildable_without_retain(tmp_path, caplog):
    kg = _kg(tmp_path)  # retain_vectors=False
    for i in range(4):
        kg.add_node(id=f"n{i}", type="T", embedding=_onehot(i * 20))
    kg.persist()
    kg.close()

    shutil.rmtree(tmp_path / "index")
    kg2 = _kg(tmp_path)
    with caplog.at_level(logging.WARNING, logger="lodedb.graph"):
        report = kg2.reindex()
    assert report["unrebuildable"] == 4  # vectors were not retained
    assert report["reindexed_vectors"] == 0
    assert any("retain_vectors=True" in record.message for record in caplog.records)


def test_reindex_label_nodes_rebuild(tmp_path):
    kg = _kg(tmp_path, retain_vectors=True)
    kg.add_node(id="a", label="alpha")
    kg.add_node(id="b", label="beta")
    kg.persist()
    kg.close()

    shutil.rmtree(tmp_path / "index")
    kg2 = _kg(tmp_path, retain_vectors=True)
    report = kg2.reindex()
    assert report["reindexed_labels"] == 2
    assert report["reindexed_vectors"] == 0
    assert report["unrebuildable"] == 0
    assert {r["id"] for r in kg2._db.list_documents()} == {"n:a", "n:b"}


def test_retained_vector_survives_node_property_update(tmp_path):
    # Re-adding a vector-in node (e.g. to change properties) keeps it rebuildable.
    kg = _kg(tmp_path, retain_vectors=True)
    kg.add_node(id="a", type="T", embedding=_onehot(7), properties={"v": 1})
    kg.add_node(id="a", type="T", embedding=_onehot(7), properties={"v": 2})
    assert kg.get_node("a").properties == {"v": 2}
    kg.persist()
    kg.close()
    shutil.rmtree(tmp_path / "index")
    kg2 = _kg(tmp_path, retain_vectors=True)
    assert kg2.reindex()["reindexed_vectors"] == 1
    assert kg2.semantic_nodes(embedding=_onehot(7), k=1)[0][1].id == "a"
