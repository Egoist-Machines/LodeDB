"""A bring-your-own-vectors index at an arbitrary dimension.

`LodeDB.open_vector_store(path, vector_dim=N)` (or `LodeDB(path, vector_dim=N)`)
creates an index with no internal embedding model, pinned to a caller-chosen dim
(any value an external embedder produces, e.g. 256/1536/3072). Only the vector-in
verbs work; text-in raises. The dim and a redacted identity persist and re-enforce
on reopen. Uses the real TurboVec (no embedding backend injected).
"""

from __future__ import annotations

import pytest

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import VectorOnlyIndexError

# A dimension that matches no preset (presets are 384 and 768).
DIM = 256


def _onehot(i: int, dim: int = DIM) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def test_open_vector_store_roundtrip(tmp_path):
    db = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    assert db.vector_only is True
    assert db._vector_dim == DIM
    db.add_vectors(_onehot(0), id="a", metadata={"kind": "node"})
    db.add_vectors_many([{"vector": _onehot(40), "id": "b"}, {"vector": _onehot(80), "id": "c"}])
    assert db.count() == 3

    hits = db.search_by_vector(_onehot(40), k=3)
    assert hits[0].id == "b"
    assert hits[0].score > 0.9
    # metadata inlines correctly for vector-only too
    assert db.search_by_vector(_onehot(0), k=1)[0].metadata == {"kind": "node"}


def test_vector_dim_via_init_is_equivalent(tmp_path):
    db = LodeDB(tmp_path, vector_dim=DIM)
    assert db.vector_only is True
    db.add_vectors(_onehot(0), id="a")
    assert db.search_by_vector(_onehot(0), k=1)[0].id == "a"


def test_text_in_methods_raise(tmp_path):
    db = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    with pytest.raises(VectorOnlyIndexError):
        db.add("hello", id="t")
    with pytest.raises(VectorOnlyIndexError):
        db.add_many([{"text": "hello", "id": "t"}])
    with pytest.raises(VectorOnlyIndexError):
        db.search("hello")
    with pytest.raises(VectorOnlyIndexError):
        db.search_many(["hello"])


def test_dim_validation(tmp_path):
    db = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    with pytest.raises(ValueError, match="dimension"):
        db.add_vectors([0.1, 0.2, 0.3], id="bad")
    with pytest.raises(ValueError, match="dimension"):
        db.search_by_vector([0.1, 0.2], k=1)


def test_vector_dim_out_of_range(tmp_path):
    with pytest.raises(ValueError, match="between 1 and 65536"):
        LodeDB.open_vector_store(tmp_path / "a", vector_dim=0)
    with pytest.raises(ValueError, match="between 1 and 65536"):
        LodeDB.open_vector_store(tmp_path / "b", vector_dim=100000)


def test_vector_dim_and_embedding_backend_mutually_exclusive(tmp_path):
    from lodedb.engine.embedding_backends import HashEmbeddingBackend

    with pytest.raises(ValueError, match="mutually exclusive"):
        LodeDB(tmp_path, vector_dim=DIM, _embedding_backend=HashEmbeddingBackend(native_dim=DIM))


def test_vector_only_has_no_text_and_enumerates(tmp_path):
    db = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    db.add_vectors(_onehot(0), id="a", metadata={"t": "1"})
    assert db.get("a") is None  # no text stored
    assert {r["id"] for r in db.list_documents()} == {"a"}
    assert db.get_document("a")["metadata"] == {"t": "1"}


def test_persist_and_reopen(tmp_path):
    db = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    db.add_vectors(_onehot(0), id="a", metadata={"label": "kept"})
    db.add_vectors(_onehot(80), id="b")
    db.persist()
    db.close()

    reopened = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    assert reopened.count() == 2
    assert reopened.search_by_vector(_onehot(0), k=1)[0].id == "a"
    assert reopened.get_document("a")["metadata"] == {"label": "kept"}


def test_reopen_at_wrong_dim_is_rejected(tmp_path):
    writer = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    writer.add_vectors(_onehot(0), id="a")
    writer.persist()
    writer.close()

    # The persisted index is DIM-dimensional; reopening "as" a different dim is
    # rejected at open by the engine's identity enforcement (fail fast, before any
    # mismatched ingest).
    with pytest.raises(RuntimeError, match="does not match"):
        LodeDB.open_vector_store(tmp_path, vector_dim=128)


def test_vector_only_store_cannot_reopen_as_custom_embedder(tmp_path):
    # A vector-only store pins model="external"; a custom embedder can collide on
    # that identity, but the persisted task ("vector-only") differs from the custom
    # route ("custom-embedder"), so reopening must reject the route collision rather
    # than serve text queries against vectors from an unknown external space.
    writer = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    writer.add_vectors(_onehot(0), id="a")
    writer.persist()
    writer.close()

    class _ExternalIdBackend(HashEmbeddingBackend):
        def __init__(self) -> None:
            super().__init__(native_dim=DIM)
            self.required_model_name = "external"  # collides with the vector-only model id

    with pytest.raises(RuntimeError, match="does not match"):
        LodeDB(tmp_path, embedder=_ExternalIdBackend())


def test_reopen_at_wrong_bit_width_is_rejected(tmp_path):
    writer = LodeDB.open_vector_store(tmp_path, vector_dim=DIM, bit_width=4)
    writer.add_vectors(_onehot(0), id="a")
    writer.persist()
    writer.close()
    # Reopening at a different (valid) width must not silently keep the stored width.
    with pytest.raises(RuntimeError, match="bit_width"):
        LodeDB.open_vector_store(tmp_path, vector_dim=DIM, bit_width=2)


def test_invalid_bit_width_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="bit_width must be 2 or 4"):
        LodeDB.open_vector_store(tmp_path, vector_dim=DIM, bit_width=8)
