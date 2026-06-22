"""Canonical Python facade for the LodeDB engine."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from lodedb.engine.core import (
    EngineDocument,
    EngineQuery,
    EngineRequestContext,
    EngineVectorDocument,
    LodeEngine,
    _validate_metadata,
    normalize_index_id,
)


@dataclass(frozen=True)
class EngineSearchResult:
    """Represents one redacted direct-Python engine search hit."""

    chunk_id: str
    document_id: str
    score: float


class EngineError(RuntimeError):
    """Signals a direct-Python engine validation or response error."""

    def __init__(self, message: str, *, status_code: int, response: Mapping[str, Any]) -> None:
        """Stores the public error message, status code, and redacted response body."""

        super().__init__(message)
        self.status_code = int(status_code)
        self.response = dict(response)


class LodeIndex:
    """Canonical direct-Python interface over one client engine index."""

    def __init__(
        self,
        engine: LodeEngine,
        *,
        client_id: str,
        index_id: str | None = None,
    ) -> None:
        """Binds a local engine instance to one client context."""

        self.engine = engine
        self.client_id = _required_text(client_id, "client_id")
        self.index_id = normalize_index_id(index_id) if index_id is not None else None

    def create(
        self,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Creates this client's index using the engine runtime route profile."""

        response = self._unwrap(
            self.engine.create_index(
                context=self._context(),
                index_id=self.index_id,
                name=name,
                metadata=metadata,
            )
        )
        self.index_id = str(response.get("index_id", self.index_id))
        return response

    def list_indexes(self) -> list[dict[str, Any]]:
        """Lists redacted index resources for this authenticated client."""

        response = self._unwrap(self.engine.list_indexes(context=self._context()))
        indexes = response.get("indexes", [])
        if not isinstance(indexes, list):
            raise EngineError(
                "invalid engine response: indexes must be a list",
                status_code=500,
                response=response,
            )
        return [dict(item) for item in indexes]

    def get_index(self, index_id: str | None = None) -> dict[str, Any]:
        """Returns redacted metadata for the selected or explicitly supplied index."""

        return self._unwrap(
            self.engine.get_index(
                context=self._context(),
                index_id=self.index_id if index_id is None else index_id,
            )
        )

    def update_index(
        self,
        *,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Updates redacted metadata for this index resource."""

        return self._unwrap(
            self.engine.update_index(
                context=self._context(),
                index_id=self.index_id,
                name=name,
                metadata=metadata,
            )
        )

    def delete_index(self) -> dict[str, Any]:
        """Deletes this index resource and its redacted persisted state."""

        return self._unwrap(
            self.engine.delete_index(context=self._context(), index_id=self.index_id)
        )

    def upsert_batch(
        self,
        documents: Iterable[Mapping[str, Any] | EngineDocument],
        *,
        embed_batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Upserts a document batch using request-level chunk embedding batches."""

        payload = tuple(_document_from_item(item) for item in documents)
        return self._unwrap(
            self.engine.upsert_documents(
                context=self._context(),
                documents=payload,
                index_id=self.index_id,
                embed_batch_size=embed_batch_size,
            )
        )

    def upsert(self, documents: Iterable[Mapping[str, Any] | EngineDocument]) -> dict[str, Any]:
        """Compatibility wrapper for upserting documents through this index."""

        return self.upsert_batch(documents)

    def build_batch(
        self,
        documents: Iterable[Mapping[str, Any] | EngineDocument],
        *,
        embed_batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Builds the initial cold corpus using request-level embedding batches."""

        payload = tuple(_document_from_item(item) for item in documents)
        return self._unwrap(
            self.engine.build_documents(
                context=self._context(),
                documents=payload,
                index_id=self.index_id,
                embed_batch_size=embed_batch_size,
            )
        )

    def build(self, documents: Iterable[Mapping[str, Any] | EngineDocument]) -> dict[str, Any]:
        """Compatibility wrapper for initial corpus build."""

        return self.build_batch(documents)

    def query(
        self,
        text: str,
        *,
        top_k: int = 10,
        filter: Mapping[str, Any] | None = None,
        include: Iterable[str] = (),
        route_drifted: bool = False,
        route_failed: bool = False,
        high_risk: bool = False,
    ) -> dict[str, Any]:
        """Queries this index and returns the redacted engine response payload."""

        return self._unwrap(
            self.engine.query(
                context=self._context(),
                index_id=self.index_id,
                query=EngineQuery(
                    text=_required_text(text, "text"),
                    top_k=top_k,
                    filter=dict(filter) if filter is not None else None,
                    include=tuple(include),
                    route_drifted=_required_bool(route_drifted, "route_drifted"),
                    route_failed=_required_bool(route_failed, "route_failed"),
                    high_risk=_required_bool(high_risk, "high_risk"),
                ),
            )
        )

    def query_batch(
        self,
        queries: Iterable[Mapping[str, Any] | EngineQuery],
    ) -> dict[str, Any]:
        """Queries this index with a public batch request and preserves request order."""

        payload = tuple(_query_from_item(item) for item in queries)
        return self._unwrap(
            self.engine.query_batch(
                context=self._context(),
                index_id=self.index_id,
                queries=payload,
            )
        )

    def upsert_vectors_batch(
        self,
        vectors: Iterable[Mapping[str, Any] | EngineVectorDocument],
    ) -> dict[str, Any]:
        """Upserts a batch of pre-embedded (vector-in) documents through this index."""

        payload = tuple(_vector_document_from_item(item) for item in vectors)
        return self._unwrap(
            self.engine.upsert_vectors(
                context=self._context(),
                vectors=payload,
                index_id=self.index_id,
            )
        )

    def query_vector(
        self,
        vector: Iterable[float],
        *,
        top_k: int = 10,
        filter: Mapping[str, Any] | None = None,
        include: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Queries this index with a precomputed embedding vector (vector-in)."""

        return self._unwrap(
            self.engine.query(
                context=self._context(),
                index_id=self.index_id,
                query=EngineQuery(
                    text="",
                    top_k=top_k,
                    filter=dict(filter) if filter is not None else None,
                    include=tuple(include),
                    embedding=tuple(float(value) for value in vector),
                ),
            )
        )

    def query_vectors_batch(
        self,
        items: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Queries this index with a batch of precomputed embedding vectors."""

        payload = tuple(_query_vector_from_item(item) for item in items)
        return self._unwrap(
            self.engine.query_batch(
                context=self._context(),
                index_id=self.index_id,
                queries=payload,
            )
        )

    def search(
        self,
        text: str,
        *,
        top_k: int = 10,
        filter: Mapping[str, Any] | None = None,
    ) -> list[EngineSearchResult]:
        """Queries this index and returns typed redacted search results."""

        response = self.query(text, top_k=top_k, filter=filter)
        results = response.get("results", [])
        if not isinstance(results, list):
            raise EngineError(
                "invalid engine response: results must be a list",
                status_code=500,
                response=response,
            )
        return [_search_result_from_payload(item, response=response) for item in results]

    def search_batch(
        self,
        queries: Iterable[Mapping[str, Any] | EngineQuery],
    ) -> list[list[EngineSearchResult]]:
        """Queries a batch and returns typed redacted results for each request."""

        response = self.query_batch(queries)
        items = response.get("queries", [])
        if not isinstance(items, list):
            raise EngineError(
                "invalid engine response: queries must be a list",
                status_code=500,
                response=response,
            )
        batches: list[list[EngineSearchResult]] = []
        for item in items:
            if not isinstance(item, Mapping):
                raise EngineError(
                    "invalid engine response: query item must be an object",
                    status_code=500,
                    response=response,
                )
            results = item.get("results", [])
            if not isinstance(results, list):
                raise EngineError(
                    "invalid engine response: results must be a list",
                    status_code=500,
                    response=response,
                )
            batches.append(
                [_search_result_from_payload(result, response=response) for result in results]
            )
        return batches


    def delete_batch(self, document_ids: Iterable[str]) -> dict[str, Any]:
        """Deletes stable document IDs and returns redacted mutation counts."""

        stable_ids = tuple(
            _required_text(document_id, "document_id") for document_id in document_ids
        )
        return self._unwrap(
            self.engine.delete_documents(
                context=self._context(),
                document_ids=stable_ids,
                index_id=self.index_id,
            )
        )

    def delete(self, document_ids: Iterable[str]) -> dict[str, Any]:
        """Compatibility wrapper for deleting stable document IDs."""

        return self.delete_batch(document_ids)

    def list_documents(self) -> list[dict[str, Any]]:
        """Lists redacted document records for this index."""

        response = self._unwrap(
            self.engine.list_documents(context=self._context(), index_id=self.index_id)
        )
        documents = response.get("documents", [])
        if not isinstance(documents, list):
            raise EngineError(
                "invalid engine response: documents must be a list",
                status_code=500,
                response=response,
            )
        return [dict(item) for item in documents]

    def get_document(self, document_id: str) -> dict[str, Any]:
        """Returns one redacted document record for this index."""

        return self._unwrap(
            self.engine.get_document(
                context=self._context(),
                index_id=self.index_id,
                document_id=_required_text(document_id, "document_id"),
            )
        )

    def get_document_text(self, document_id: str) -> str:
        """Returns one stored document's raw text (opt-in raw-text storage only).

        Raises :class:`EngineError` when raw-text storage is disabled, the
        document is unknown, or its text was not stored.
        """

        response = self._unwrap(
            self.engine.get_document_text(
                context=self._context(),
                index_id=self.index_id,
                document_id=_required_text(document_id, "document_id"),
            )
        )
        return str(response.get("text", ""))

    def get_document_texts(self, document_ids: Iterable[str]) -> dict[str, str]:
        """Returns stored raw text for several document ids (opt-in storage only).

        Unknown or not-stored ids are omitted from the returned mapping. Raises
        :class:`EngineError` only when raw-text storage is disabled.
        """

        stable_ids = tuple(
            _required_text(document_id, "document_id") for document_id in document_ids
        )
        response = self._unwrap(
            self.engine.get_document_texts(
                context=self._context(),
                index_id=self.index_id,
                document_ids=stable_ids,
            )
        )
        documents = response.get("documents", {})
        if not isinstance(documents, Mapping):
            raise EngineError(
                "invalid engine response: documents must be an object",
                status_code=500,
                response=response,
            )
        return {str(key): str(value) for key, value in documents.items()}

    def stats(self) -> dict[str, Any]:
        """Returns redacted stats metadata for this client index."""

        return self._unwrap(self.engine.stats(context=self._context(), index_id=self.index_id))

    def audit(self) -> dict[str, Any]:
        """Returns redacted audit metadata for this client index."""

        return self._unwrap(
            self.engine.audit(context=self._context(), index_id=self.index_id)
        )

    def _context(self) -> EngineRequestContext:
        """Builds a fresh request context for one direct-Python operation."""

        return EngineRequestContext(
            client_id=self.client_id,
            now=datetime.now(tz=UTC),
        )

    def _unwrap(self, response: Any) -> dict[str, Any]:
        """Returns the response body or raises a direct-Python engine error."""

        body = dict(response.body)
        if int(response.status_code) >= 400:
            raise EngineError(
                str(body.get("error") or "engine request failed"),
                status_code=int(response.status_code),
                response=body,
            )
        return body


def _document_from_item(item: Mapping[str, Any] | EngineDocument) -> EngineDocument:
    """Normalizes one direct-Python document mapping into an engine document."""

    if isinstance(item, EngineDocument):
        return item
    if not isinstance(item, Mapping):
        raise EngineError(
            "document must be a mapping or EngineDocument",
            status_code=400,
            response={"status": "error", "error": "invalid document"},
        )
    metadata = item.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise EngineError(
            "document metadata must be a mapping",
            status_code=400,
            response={"status": "error", "error": "invalid metadata"},
        )
    try:
        safe_metadata = _validate_metadata(metadata)
    except ValueError as exc:
        raise EngineError(
            str(exc),
            status_code=400,
            response={"status": "error", "error": str(exc)},
        ) from exc
    return EngineDocument(
        document_id=_required_text(item.get("document_id"), "document_id"),
        text=_required_text(item.get("text"), "text"),
        metadata=safe_metadata,
    )


def _vector_document_from_item(
    item: Mapping[str, Any] | EngineVectorDocument,
) -> EngineVectorDocument:
    """Normalizes one direct-Python vector-document mapping into an engine vector doc."""

    if isinstance(item, EngineVectorDocument):
        return item
    if not isinstance(item, Mapping):
        raise EngineError(
            "vector document must be a mapping or EngineVectorDocument",
            status_code=400,
            response={"status": "error", "error": "invalid vector document"},
        )
    vector = item.get("vector")
    if vector is None:
        raise EngineError(
            "vector document requires a 'vector'",
            status_code=400,
            response={"status": "error", "error": "vector is required"},
        )
    metadata = item.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise EngineError(
            "document metadata must be a mapping",
            status_code=400,
            response={"status": "error", "error": "invalid metadata"},
        )
    try:
        safe_metadata = _validate_metadata(metadata)
    except ValueError as exc:
        raise EngineError(
            str(exc),
            status_code=400,
            response={"status": "error", "error": str(exc)},
        ) from exc
    return EngineVectorDocument(
        document_id=_required_text(item.get("document_id"), "document_id"),
        vector=tuple(float(value) for value in vector),
        metadata=safe_metadata,
    )


def _query_vector_from_item(item: Mapping[str, Any]) -> EngineQuery:
    """Normalizes one direct-Python vector-query mapping into an engine query."""

    if not isinstance(item, Mapping):
        raise EngineError(
            "query must be a mapping",
            status_code=400,
            response={"status": "error", "error": "invalid query"},
        )
    vector = item.get("vector")
    if vector is None:
        raise EngineError(
            "vector query requires a 'vector'",
            status_code=400,
            response={"status": "error", "error": "vector is required"},
        )
    top_k = item.get("top_k", 10)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise EngineError(
            "top_k must be an integer",
            status_code=400,
            response={"status": "error", "error": "top_k must be an integer"},
        )
    include = item.get("include", ())
    return EngineQuery(
        text="",
        top_k=top_k,
        filter=dict(item["filter"]) if "filter" in item and item["filter"] is not None else None,
        include=tuple(str(value) for value in include),
        embedding=tuple(float(value) for value in vector),
    )


def _query_from_item(item: Mapping[str, Any] | EngineQuery) -> EngineQuery:
    """Normalizes one direct-Python batch query mapping into an engine query."""

    if isinstance(item, EngineQuery):
        return item
    if not isinstance(item, Mapping):
        raise EngineError(
            "query must be a mapping or EngineQuery",
            status_code=400,
            response={"status": "error", "error": "invalid query"},
        )
    include = item.get("include", ())
    if not isinstance(include, Iterable) or isinstance(include, str):
        raise EngineError(
            "include must be a list",
            status_code=400,
            response={"status": "error", "error": "include must be a list"},
        )
    top_k = item.get("top_k", 10)
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise EngineError(
            "top_k must be an integer",
            status_code=400,
            response={"status": "error", "error": "top_k must be an integer"},
        )
    return EngineQuery(
        text=_required_text(item.get("query"), "query"),
        top_k=top_k,
        filter=dict(item["filter"]) if "filter" in item and item["filter"] is not None else None,
        include=tuple(str(value) for value in include),
        route_drifted=_mapping_bool(item, "route_drifted"),
        route_failed=_mapping_bool(item, "route_failed"),
        high_risk=_mapping_bool(item, "high_risk"),
    )


def _search_result_from_payload(
    payload: Any,
    *,
    response: Mapping[str, Any],
) -> EngineSearchResult:
    """Builds a typed search result from one redacted engine result payload."""

    if not isinstance(payload, Mapping):
        raise EngineError(
            "invalid engine response: result row must be an object",
            status_code=500,
            response=response,
        )
    forbidden = {"text", "chunk_text", "document_text", "embedding", "raw_payload"}
    if forbidden.intersection(payload):
        raise EngineError(
            "invalid engine response: raw payload fields are not allowed",
            status_code=500,
            response=response,
        )
    return EngineSearchResult(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        score=float(payload["score"]),
    )


def _required_text(value: Any, name: str) -> str:
    """Returns a stripped nonblank string for direct-Python public inputs."""

    if not isinstance(value, str):
        raise EngineError(
            f"{name} must be a string",
            status_code=400,
            response={"status": "error", "error": f"{name} must be a string"},
        )
    stripped = value.strip()
    if not stripped:
        raise EngineError(
            f"{name} is required",
            status_code=400,
            response={"status": "error", "error": f"{name} is required"},
        )
    return stripped


def _required_bool(value: Any, name: str) -> bool:
    """Returns a JSON-style boolean value while rejecting strings and integers."""

    if not isinstance(value, bool):
        raise EngineError(
            f"{name} must be a boolean",
            status_code=400,
            response={"status": "error", "error": f"{name} must be a boolean"},
        )
    return value


def _mapping_bool(item: Mapping[str, Any], name: str) -> bool:
    """Returns an optional batch-query boolean, rejecting stringly booleans."""

    if name not in item:
        return False
    return _required_bool(item[name], name)
