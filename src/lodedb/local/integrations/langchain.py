"""LangChain ``VectorStore`` adapter for LodeDB (optional ``lodedb[langchain]``).

Wraps the LodeDB SDK as a ``langchain_core.vectorstores.VectorStore`` so RAG apps
can drop in the local-first store. LodeDB embeds internally (via its preset), so
no LangChain ``Embeddings`` object is needed.

**Design note: page_content.** LodeDB's *redacted* artifacts (snapshot, journal,
telemetry, audit) never carry raw text, but the original text is retained by default in
a separate ``.tvtext`` sidecar (``store_text=True``). This adapter fills LangChain
``Document.page_content`` from a **session-local** ``id -> text`` cache for documents
added in the current process, and falls back to ``db.get_text(id)`` for documents from a
prior session whenever the underlying LodeDB retains text. If the store was opened with
``store_text=False``, no text is kept and ``page_content`` is empty across a reopen; keep
your own text store keyed by the returned id/metadata in that case. Retrieval (ids,
scores, metadata) is always durable.
"""

from __future__ import annotations

from typing import Any

from lodedb.local.db import LodeDB

try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
except ImportError as exc:  # pragma: no cover - clear install hint
    raise ImportError(
        "the LodeDB LangChain adapter needs langchain-core: pip install 'lodedb[langchain]'"
    ) from exc


class LodeDBVectorStore(VectorStore):
    """A LangChain ``VectorStore`` backed by a local :class:`LodeDB` instance."""

    def __init__(self, db: LodeDB) -> None:
        """Wraps an open LodeDB and starts an empty session-local text cache."""

        self._db = db
        self._texts: dict[str, str] = {}

    @property
    def embeddings(self):
        """LodeDB embeds internally, so there is no external Embeddings object."""

        return None

    def add_texts(self, texts, metadatas=None, **kwargs):
        """Adds texts (with optional metadatas/ids) and returns the assigned ids.

        Text is sent to LodeDB for embedding (and retained in its ``.tvtext`` sidecar
        unless the store was opened with ``store_text=False``); a session-local cache
        also keeps it for immediate ``page_content`` reconstruction.
        """

        texts = list(texts)
        metadatas = list(metadatas) if metadatas is not None else [None] * len(texts)
        ids = kwargs.get("ids")
        items: list[dict[str, Any]] = []
        for i, text in enumerate(texts):
            items.append(
                {
                    "text": text,
                    "id": (ids[i] if ids else None),
                    "metadata": dict(metadatas[i] or {}),
                }
            )
        assigned = self._db.add_many(items)
        for doc_id, text in zip(assigned, texts, strict=True):
            self._texts[doc_id] = text
        return assigned

    def similarity_search_with_score(self, query, k=4, **kwargs):
        """Returns ``[(Document, score)]`` for the top-k hits.

        ``page_content`` is filled from the session-local cache, falling back to the
        durable ``.tvtext`` sidecar when the store retains text (empty otherwise);
        ``metadata`` carries the durable user metadata + id.
        """

        filter = kwargs.get("filter")
        results: list[tuple[Document, float]] = []
        for hit in self._db.search(query, k=k, filter=filter):
            meta = dict(hit.metadata)
            meta["id"] = hit.id
            page = self._texts.get(hit.id)
            if page is None and getattr(self._db, "store_text", False):
                # Durable fallback: recover text for prior-session docs from the
                # .tvtext sidecar (only when this store retains text).
                page = self._db.get_text(hit.id)
            results.append((Document(page_content=page or "", metadata=meta), hit.score))
        return results

    def similarity_search(self, query, k=4, **kwargs):
        """Returns the top-k ``Document`` results (scores dropped)."""

        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    def delete(self, ids=None, **kwargs):
        """Deletes documents by id; returns True only if all existed."""

        if not ids:
            return False
        results = [self._db.remove(doc_id) for doc_id in ids]
        for doc_id in ids:
            self._texts.pop(doc_id, None)
        return all(results)

    @classmethod
    def from_texts(
        cls,
        texts,
        embedding=None,
        metadatas=None,
        *,
        path,
        ids=None,
        model: str = "minilm",
        device: str = "auto",
        **kwargs,
    ):
        """Opens a LodeDB at ``path`` and indexes ``texts`` (``embedding`` is ignored)."""

        db = LodeDB(path=path, model=model, device=device)
        store = cls(db)
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store
