"""Tests for the public ``embedder=`` argument.

``LodeDB(embedder=...)`` drives a text-capable index with a caller-supplied
:class:`~lodedb.engine.embedding_backends.EngineEmbeddingBackend` at the
backend's own dimension, instead of a built-in preset. The vector-in verbs work
too (at the same dimension), and the dimension/identity round-trips on reopen.
"""

from __future__ import annotations

import pytest

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend

# Deliberately not a preset dimension (minilm=384, bge=768) so the test proves
# the shape is taken from the embedder, not a preset.
DIM = 512


def _embedder() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=DIM)


def test_custom_embedder_text_roundtrip(tmp_path):
    db = LodeDB(path=tmp_path, embedder=_embedder())
    a = db.add("alpha document", metadata={"k": "1"})
    db.add("beta document", metadata={"k": "2"})
    assert db.count() == 2
    # The hash embedder maps identical text to the same vector, so the matching
    # query is the top hit.
    hits = db.search("alpha document", k=2)
    assert hits[0].id == a
    assert hits[0].score > 0.9


def test_custom_embedder_dimension_is_derived(tmp_path):
    db = LodeDB(path=tmp_path, embedder=_embedder())
    assert db._vector_dim == DIM
    assert not db.vector_only
    # vector-in verbs work at the embedder's dimension
    vector = [0.0] * DIM
    vector[3] = 1.0
    db.add_vectors(vector, id="v1")
    assert db.search_by_vector(vector, k=1)[0].id == "v1"


def test_custom_embedder_persistence_roundtrip(tmp_path):
    db = LodeDB(path=tmp_path, embedder=_embedder())
    db.add("kept document", id="x", metadata={"label": "kept"})
    db.persist()
    db.close()

    reopened = LodeDB(path=tmp_path, embedder=_embedder())
    assert reopened.count() == 1
    assert reopened.search("kept document", k=1)[0].id == "x"
    assert reopened.get_document("x")["metadata"] == {"label": "kept"}


def test_embedder_and_vector_dim_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="mutually exclusive"):
        LodeDB(path=tmp_path, vector_dim=DIM, embedder=_embedder())


class _IdentBackend(HashEmbeddingBackend):
    """A hash backend that declares a model identity, for reopen-identity tests."""

    def __init__(self, model_name: str, dim: int = DIM) -> None:
        super().__init__(native_dim=dim)
        self.required_model_name = model_name
        self.name = "ident"


def test_reopen_with_wrong_embedder_identity_is_rejected(tmp_path):
    db = LodeDB(path=tmp_path, embedder=_IdentBackend("model-A"))
    db.add("hello world", id="a")
    db.persist()
    db.close()

    # Same dimension, different declared model identity: rejected at open rather
    # than silently serving meaningless scores.
    with pytest.raises(RuntimeError, match="does not match"):
        LodeDB(path=tmp_path, embedder=_IdentBackend("model-B"))

    # The matching identity still reopens cleanly.
    reopened = LodeDB(path=tmp_path, embedder=_IdentBackend("model-A"))
    assert reopened.count() == 1


class _NoDimBackend:
    """A backend that fails to declare a usable dimension."""

    name = "nodim"
    native_dim = 0
    required_model_name = None

    def embed_documents(self, texts):
        return ()

    def embed_query(self, text):
        return ()


def test_embedder_rejects_non_positive_native_dim(tmp_path):
    with pytest.raises(ValueError, match="native_dim"):
        LodeDB(path=tmp_path, embedder=_NoDimBackend())
