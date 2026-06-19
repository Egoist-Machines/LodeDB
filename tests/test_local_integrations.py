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
