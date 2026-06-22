"""Tests for the LodeDB framework adapters (gated on the optional framework deps)."""

from __future__ import annotations

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB


def test_langchain_vectorstore_roundtrip(tmp_path):
    """The LangChain adapter adds/searches/deletes and round-trips page_content."""

    pytest.importorskip("langchain_core")  # needs lodedb[langchain]
    from langchain_core.documents import Document

    from lodedb.local.integrations.langchain import LodeDBVectorStore

    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    store = LodeDBVectorStore(db)
    ids = store.add_texts(
        ["alpha document", "beta document"],
        metadatas=[{"k": "a"}, {"k": "b"}],
        ids=["a", "b"],
    )
    assert ids == ["a", "b"]

    docs = store.similarity_search("alpha", k=2)
    assert docs and all(isinstance(d, Document) for d in docs)
    # page_content round-trips via the stored `text` metadata key.
    assert "alpha document" in {d.page_content for d in docs}
    assert all("id" in d.metadata for d in docs)

    scored = store.similarity_search_with_score("alpha", k=2)
    assert all(isinstance(s, float) for _, s in scored)

    assert store.delete(["a"]) is True
    assert store.delete([]) is False
    db.close()


def test_langchain_vectorstore_predicate_filter(tmp_path):
    """The LangChain adapter passes predicate filters straight through to LodeDB."""

    pytest.importorskip("langchain_core")  # needs lodedb[langchain]

    from lodedb.local.integrations.langchain import LodeDBVectorStore

    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    store = LodeDBVectorStore(db)
    store.add_texts(
        ["alpha document", "beta document", "gamma document"],
        metadatas=[{"year": 2019}, {"year": 2021}, {"year": 2023}],
        ids=["a", "b", "c"],
    )
    docs = store.similarity_search("document", k=10, filter={"year": {"$gte": 2021}})
    assert {d.metadata["id"] for d in docs} == {"b", "c"}
    db.close()
