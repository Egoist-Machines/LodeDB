"""Read-only exporter for a mem0 Qdrant collection (vector-preserve).

mem0's default vector store is Qdrant, commonly run on-disk via a local ``path``.
mem0 owns the embeddings, so this importer copies ids, vectors, and full payloads
verbatim into vector-preserve :class:`ExportedRow` rows: LodeDB stores the vector
as-is and keeps the full payload JSON in its raw-text sidecar, while scalar filter
keys (``user_id`` / ``agent_id`` / ``run_id`` / …) are carried into LodeDB metadata
by the mem0 adapter so filtered reads stay exact.

The Qdrant store is opened read-only through ``qdrant_client`` (the same client
mem0 uses). Scrolling the collection never writes to it. A non-local Qdrant
(``url``/``host``) is allowed only when the caller passes ``allow_remote=True``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from lodedb.local.migrate.sources.base import (
    MODE_VECTOR_PRESERVE,
    ExportedRow,
    SourceExport,
    SourceExportError,
)

_SCROLL_BATCH = 256


class Mem0QdrantExport(SourceExport):
    """Streams a mem0 Qdrant collection as vector-preserve rows."""

    def __init__(
        self,
        *,
        collection_name: str,
        path: str | None = None,
        url: str | None = None,
        host: str | None = None,
        port: int | None = None,
        api_key: str | None = None,
        embedding_model_dims: int | None = None,
        allow_remote: bool = False,
    ) -> None:
        """Opens the Qdrant collection read-only and reads its vector dimension."""

        is_local = path is not None
        if not is_local and not allow_remote:
            raise SourceExportError(
                "refusing to connect to a non-local Qdrant without an explicit remote override; "
                "pass a local 'path', or re-run with --allow-remote-source"
            )
        client = _open_qdrant(
            path=path, url=url, host=host, port=port, api_key=api_key
        )
        dim, count = _collection_shape(client, collection_name, embedding_model_dims)
        location = path if path is not None else (url or f"{host}:{port}")
        super().__init__(
            framework="mem0",
            provider="qdrant",
            mode=MODE_VECTOR_PRESERVE,
            location=str(location),
            vector_dim=dim,
            count=count,
        )
        self._client = client
        self._collection = collection_name

    def iter_rows(self) -> Iterator[ExportedRow]:
        """Scrolls the collection in stable batches, yielding vector-preserve rows."""

        offset: Any = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection,
                limit=_SCROLL_BATCH,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                vector = _coerce_vector(getattr(point, "vector", None))
                payload = dict(getattr(point, "payload", None) or {})
                yield ExportedRow(
                    id=str(getattr(point, "id", "")),
                    text=None,
                    metadata={},  # the mem0 adapter derives filterable metadata from the payload
                    vector=vector,
                    raw_payload=payload,
                )
            if offset is None:
                break

    def close(self) -> None:
        """Closes the Qdrant client handle."""

        close = getattr(self._client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - best-effort close of a read-only client
                pass


def _open_qdrant(
    *,
    path: str | None,
    url: str | None,
    host: str | None,
    port: int | None,
    api_key: str | None,
) -> Any:
    """Opens a ``QdrantClient`` for the given local path or remote target (read-only use)."""

    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise SourceExportError(
            "exporting a mem0 Qdrant store needs qdrant-client (it ships with mem0ai's qdrant "
            "extra); install it in the source project's environment"
        ) from exc

    if path is not None:
        return QdrantClient(path=path)
    if url is not None:
        return QdrantClient(url=url, api_key=api_key)
    if host is not None:
        return QdrantClient(host=host, port=port or 6333, api_key=api_key)
    raise SourceExportError("a mem0 Qdrant export needs one of: path, url, or host")


def _collection_shape(
    client: Any, collection_name: str, fallback_dim: int | None
) -> tuple[int, int | None]:
    """Returns ``(vector_dim, count)`` for a collection, read from Qdrant metadata."""

    try:
        info = client.get_collection(collection_name=collection_name)
    except Exception as exc:  # noqa: BLE001 - missing collection is a clean export error
        raise SourceExportError(
            f"Qdrant collection {collection_name!r} could not be read: {exc}"
        ) from exc
    dim = _vector_dim_from_info(info)
    if dim is None:
        dim = fallback_dim
    if dim is None:
        raise SourceExportError(
            f"could not determine the vector dimension of Qdrant collection {collection_name!r}; "
            "pass --vector-dim / --embedding-dim"
        )
    count = None
    try:
        count = int(client.count(collection_name=collection_name, exact=True).count)
    except Exception:  # noqa: BLE001 - count is best-effort; streaming still works without it
        count = getattr(info, "points_count", None)
        count = int(count) if isinstance(count, int) else None
    return int(dim), count


def _vector_dim_from_info(info: Any) -> int | None:
    """Extracts the vector size from a Qdrant ``get_collection`` response."""

    try:
        params = info.config.params.vectors
    except AttributeError:
        return None
    size = getattr(params, "size", None)
    if isinstance(size, int):
        return size
    # Named-vector configs map name -> VectorParams; mem0 uses the default unnamed vector.
    if isinstance(params, dict):
        for value in params.values():
            inner = getattr(value, "size", None)
            if isinstance(inner, int):
                return inner
    return None


def _coerce_vector(vector: Any) -> list[float] | None:
    """Normalizes a Qdrant point vector (list or named-vector dict) to a float list."""

    if vector is None:
        return None
    if isinstance(vector, dict):
        # Named vectors: take the single/default entry.
        if not vector:
            return None
        vector = next(iter(vector.values()))
    try:
        return [float(component) for component in vector]
    except (TypeError, ValueError):
        return None
