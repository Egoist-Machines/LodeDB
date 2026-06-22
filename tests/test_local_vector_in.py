"""Tests for the vector-in API (Phase 2).

``add_vectors`` / ``search_by_vector`` (and their batch forms) let callers store
and query precomputed embeddings, bypassing the internal embedder while reusing
the same atomic-commit + TurboVec scan path as the text API. No raw text is
retained for a vector-in document.
"""

from __future__ import annotations

import pytest

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import ReadOnlyError

DIM = 384


def _db(path, **kwargs) -> LodeDB:
    return LodeDB(
        path=path,
        model="minilm",
        _embedding_backend=HashEmbeddingBackend(native_dim=DIM),
        **kwargs,
    )


def _onehot(i: int, *, scale: float = 1.0, dim: int = DIM) -> list[float]:
    vector = [0.0] * dim
    vector[i] = scale
    return vector


def test_add_vectors_and_search_roundtrip(tmp_path):
    db = _db(tmp_path)
    db.add_vectors(_onehot(0), id="a", metadata={"label": "first"})
    db.add_vectors(_onehot(40), id="b", metadata={"label": "second"})
    db.add_vectors(_onehot(80), id="c", metadata={"label": "third"})
    assert db.count() == 3

    hits = db.search_by_vector(_onehot(40), k=3)
    assert hits[0].id == "b"
    assert hits[0].score > 0.9
    assert hits[0].metadata == {"label": "second"}


def test_add_vectors_many(tmp_path):
    db = _db(tmp_path)
    ids = db.add_vectors_many(
        [
            {"vector": _onehot(0), "id": "x", "metadata": {"g": "1"}},
            {"vector": _onehot(50), "id": "y", "metadata": {"g": "2"}},
            {"vector": _onehot(100)},  # auto id
        ]
    )
    assert ids[0] == "x" and ids[1] == "y"
    assert ids[2].startswith("doc-")
    assert db.count() == 3
    assert db.search_by_vector(_onehot(0), k=1)[0].id == "x"


def test_vector_in_documents_have_no_text(tmp_path):
    db = _db(tmp_path)
    db.add_vectors(_onehot(0), id="a")
    assert db.get("a") is None
    assert db.get_text("a") is None
    # but the record is enumerable / has metadata
    assert db.get_document("a")["chunk_count"] == 1


def test_normalize_dedup_is_noop(tmp_path):
    db = _db(tmp_path)
    db.add_vectors(_onehot(0, scale=1.0), id="a")
    db.add_vectors(_onehot(0, scale=7.0), id="a")  # same direction -> same stored vector
    assert db.count() == 1


def test_dim_mismatch_raises(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(ValueError, match="dimension"):
        db.add_vectors([0.1, 0.2, 0.3], id="bad")
    with pytest.raises(ValueError, match="dimension"):
        db.search_by_vector([0.1, 0.2, 0.3], k=1)


def test_zero_vector_raises_when_normalizing(tmp_path):
    db = _db(tmp_path)
    with pytest.raises(ValueError, match="zero vector"):
        db.add_vectors([0.0] * DIM, id="z")
    # but allowed when normalization is disabled
    db.add_vectors([0.0] * DIM, id="z", normalize=False)
    assert db.count() == 1


def test_non_finite_vector_raises(tmp_path):
    db = _db(tmp_path)
    bad = _onehot(0)
    bad[1] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        db.add_vectors(bad, id="bad")


def test_filtered_vector_search(tmp_path):
    db = _db(tmp_path)
    db.add_vectors(_onehot(0), id="n1", metadata={"kind": "node"})
    db.add_vectors(_onehot(10), id="e1", metadata={"kind": "edge"})
    db.add_vectors(_onehot(20), id="e2", metadata={"kind": "edge"})
    hits = db.search_by_vector(_onehot(0), k=5, filter={"kind": "edge"})
    assert {h.id for h in hits} == {"e1", "e2"}  # n1 excluded despite being closest


def test_search_many_by_vector_preserves_order(tmp_path):
    db = _db(tmp_path)
    db.add_vectors(_onehot(0), id="a")
    db.add_vectors(_onehot(60), id="b")
    batches = db.search_many_by_vector([_onehot(60), _onehot(0)], k=1)
    assert [batch[0].id for batch in batches] == ["b", "a"]


def test_upsert_vector_replaces(tmp_path):
    db = _db(tmp_path)
    db.add_vectors(_onehot(0), id="x")
    db.add_vectors(_onehot(120), id="x")  # replace with a different direction
    assert db.count() == 1
    assert db.search_by_vector(_onehot(120), k=1)[0].id == "x"


def test_mixed_text_and_vector_in_same_index(tmp_path):
    db = _db(tmp_path)
    db.add("the quick brown fox", id="t1", metadata={"kind": "text"})
    db.add_vectors(_onehot(0), id="v1", metadata={"kind": "vector"})
    assert db.count() == 2
    assert {r["id"] for r in db.list_documents()} == {"t1", "v1"}
    # Text search still works on the text doc. (HashEmbeddingBackend is a
    # non-semantic hash, so restrict to the text doc by filter to make ranking
    # deterministic — the point here is coexistence, not retrieval quality.)
    assert db.search("the quick brown fox", k=5, filter={"kind": "text"})[0].id == "t1"
    # vector search finds the vector doc
    assert db.search_by_vector(_onehot(0), k=1)[0].id == "v1"


def test_vector_in_persistence_roundtrip(tmp_path):
    db = _db(tmp_path)
    db.add_vectors(_onehot(0), id="a", metadata={"label": "kept"})
    db.add_vectors(_onehot(80), id="b")
    db.persist()
    db.close()

    reopened = _db(tmp_path)
    assert reopened.count() == 2
    hit = reopened.search_by_vector(_onehot(0), k=1)[0]
    assert hit.id == "a"
    assert reopened.get_document("a")["metadata"] == {"label": "kept"}


def test_read_only_blocks_add_vectors(tmp_path):
    writer = _db(tmp_path)
    writer.add_vectors(_onehot(0), id="a")
    writer.persist()
    writer.close()

    reader = LodeDB.open_readonly(
        tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=DIM)
    )
    with pytest.raises(ReadOnlyError):
        reader.add_vectors(_onehot(1), id="b")
    # reads still work
    assert reader.search_by_vector(_onehot(0), k=1)[0].id == "a"
