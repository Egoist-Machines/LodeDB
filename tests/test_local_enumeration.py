"""Tests for the public enumeration surface.

``LodeDB.list_documents()`` and ``get_document()`` expose the engine's
payload-free document records on the SDK, with a ``filter=`` complete-set
enumeration (no ``k`` cap, no query vector), the primitive a graph /
knowledge-graph layer needs for deterministic traversal.
"""

from __future__ import annotations

import pytest

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend


def _db(path, **kwargs) -> LodeDB:
    return LodeDB(
        path=path,
        model="minilm",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
        **kwargs,
    )


def test_list_documents_returns_payload_free_records(tmp_path):
    db = _db(tmp_path)
    db.add("the quick brown fox", id="a", metadata={"topic": "animals"})
    db.add("a treatise on tax law", id="b", metadata={"topic": "law"})

    records = db.list_documents()
    assert len(records) == 2
    by_id = {record["id"]: record for record in records}
    assert set(by_id) == {"a", "b"}
    for record in records:
        assert set(record) == {"id", "metadata", "chunk_count", "content_hash"}
        # Payload-free: never leak text or vectors through enumeration.
        assert "text" not in record
        assert "embedding" not in record
        assert record["chunk_count"] >= 1
    assert by_id["a"]["metadata"] == {"topic": "animals"}


def test_get_document_by_id_and_missing(tmp_path):
    db = _db(tmp_path)
    db.add("hello world", id="doc-1", metadata={"k": "v", "n": 7})

    record = db.get_document("doc-1")
    assert record is not None
    assert record["id"] == "doc-1"
    assert record["metadata"] == {"k": "v", "n": "7"}  # metadata is stringified

    assert db.get_document("does-not-exist") is None
    with pytest.raises(ValueError):
        db.get_document("   ")


def test_list_documents_exact_and_predicate_filters(tmp_path):
    db = _db(tmp_path)
    db.add_many(
        [
            {"text": "paper one", "id": "p1", "metadata": {"topic": "ml", "year": 2019}},
            {"text": "paper two", "id": "p2", "metadata": {"topic": "ml", "year": 2023}},
            {"text": "paper three", "id": "p3", "metadata": {"topic": "bio", "year": 2024}},
        ]
    )

    exact = {record["id"] for record in db.list_documents(filter={"topic": "ml"})}
    assert exact == {"p1", "p2"}

    recent = {record["id"] for record in db.list_documents(filter={"year": {"$gte": 2023}})}
    assert recent == {"p2", "p3"}

    either = {
        record["id"]
        for record in db.list_documents(
            filter={"$or": [{"topic": "bio"}, {"year": {"$lte": 2019}}]}
        )
    }
    assert either == {"p1", "p3"}

    in_set = {
        record["id"]
        for record in db.list_documents(filter={"topic": {"$in": ["bio", "ml"]}})
    }
    assert in_set == {"p1", "p2", "p3"}


def test_list_documents_document_ids_allowlist(tmp_path):
    db = _db(tmp_path)
    for i in range(5):
        db.add(f"text {i}", id=f"d{i}", metadata={"group": "g"})
    allow = {
        record["id"]
        for record in db.list_documents(
            filter={"document_ids": ["d0", "d3"], "metadata": {"group": "g"}}
        )
    }
    assert allow == {"d0", "d3"}


def test_list_documents_is_complete_not_topk(tmp_path):
    # Enumeration must return the COMPLETE matching set, unlike search which
    # caps at k. 25 docs share metadata; list_documents returns all 25.
    db = _db(tmp_path)
    db.add_many(
        [{"text": f"node {i}", "id": f"n{i}", "metadata": {"kind": "node"}} for i in range(25)]
    )
    nodes = db.list_documents(filter={"kind": "node"})
    assert len(nodes) == 25
    # A ranked search with default k would only surface 10.
    assert len(db.search("node", k=10, filter={"kind": "node"})) == 10


def test_enumeration_on_readonly_handle(tmp_path):
    writer = _db(tmp_path)
    writer.add("persisted", id="keep", metadata={"kind": "node"})
    writer.persist()
    writer.close()

    reader = LodeDB.open_readonly(
        tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    assert {record["id"] for record in reader.list_documents()} == {"keep"}
    assert reader.get_document("keep") is not None
    assert {r["id"] for r in reader.list_documents(filter={"kind": "node"})} == {"keep"}


def test_list_documents_empty_store(tmp_path):
    db = _db(tmp_path)
    assert db.list_documents() == []
    assert db.list_documents(filter={"kind": "node"}) == []
