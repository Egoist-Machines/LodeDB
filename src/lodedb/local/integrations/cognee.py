"""cognee vector-database adapter for LodeDB (optional ``lodedb[cognee]``).

``CogneeLodeDBAdapter`` implements cognee's ``VectorDBInterface`` against LodeDB's
vector-in API, so cognee can use a local-first, on-disk, no-server vector store as
its ``vector_db_provider``. cognee owns the embeddings (via its configured
``EmbeddingEngine``); LodeDB stores and searches those vectors and persists changed
rows incrementally.

Each cognee *collection* is one LodeDB vector-only index under a shared base
directory (``url``). A cognee ``DataPoint`` becomes one LodeDB document whose vector
is ``embedding_engine.embed_text(DataPoint.get_embeddable_data(dp))``. The full
serialized DataPoint payload is kept in LodeDB's dedicated raw-text sidecar (so
``retrieve`` and ``include_payload`` searches can return it), never in redacted
metadata. LodeDB metadata carries only ``belongs_to_set`` membership, encoded as
scalar presence keys, so cognee's ``node_name`` (NodeSet) filtering pushes into the
engine's metadata planner instead of post-filtering in Python.

cognee ranks by cosine *distance* (lower is better); LodeDB scores by cosine
*similarity* (higher is better), so search results report ``1 - similarity``, the
same convention as the built-in LanceDB adapter's ``.distance_type("cosine")``.

This targets ``cognee>=1.1.0,<2``. It is registered with cognee via
``register_cognee_adapter()`` (which calls cognee's ``use_vector_adapter``), or from
the ``cognee-community-vector-adapter-lodedb`` package's ``register`` module.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from lodedb.local.db import LodeDB

try:
    from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
    from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
    from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
    from cognee.infrastructure.engine import DataPoint
except ImportError as exc:  # pragma: no cover - clear install hint
    raise ImportError(
        "the LodeDB cognee adapter needs cognee: pip install 'lodedb[cognee]'"
    ) from exc


# LodeDB metadata is scalar-only, so ``belongs_to_set`` membership is stored as one
# presence key per tag. The prefix keeps these keys from ever colliding with a real
# field: LodeDB metadata for a cognee document holds *only* these keys (the full
# payload, including the canonical belongs_to_set list, lives in the text sidecar).
_BELONGS_TO_SET_PREFIX = "belongs_to_set::"


class IndexSchema(DataPoint):
    """Data point stored by ``index_data_points`` for a text vector index.

    Mirrors the built-in adapters' ``IndexSchema``: an ``id`` and the indexed
    ``text``, with ``belongs_to_set`` carried through for NodeSet filtering.
    """

    id: str
    text: str

    metadata: dict = {"index_fields": ["text"]}
    belongs_to_set: list[str] = []


class CogneeLodeDBAdapter:
    """cognee ``VectorDBInterface`` backed by local, on-disk :class:`LodeDB` indexes.

    Instantiated by cognee's ``create_vector_engine`` with keyword arguments
    ``url`` / ``api_key`` / ``embedding_engine`` / ``database_name`` once registered
    via :func:`register_cognee_adapter`. ``url`` is the base directory that holds one
    LodeDB index per collection; ``api_key`` is unused (LodeDB is local and needs no
    credential) and accepted only for signature compatibility.
    """

    name = "LodeDB"

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        embedding_engine: EmbeddingEngine = None,
        database_name: str = "cognee",
        *,
        normalize: bool = True,
        bit_width: int = 4,
        **kwargs: Any,
    ) -> None:
        """Prepares the base directory and holds the cognee embedding engine.

        No LodeDB index is opened here; each collection is opened lazily on first
        use and cached, so constructing the adapter never touches disk.
        """

        if not url:
            raise ValueError(
                "CogneeLodeDBAdapter requires a non-empty 'url' (the local directory "
                "for LodeDB-backed cognee collections); set vector_db_url in cognee config"
            )
        if embedding_engine is None:
            raise ValueError("CogneeLodeDBAdapter requires an embedding_engine")
        self.url = str(url)
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.database_name = str(database_name)
        self.normalize = bool(normalize)
        self.bit_width = int(bit_width)
        self._base_dir = Path(self.url)
        # Lazily opened LodeDB handles keyed by collection name; the async lock
        # serializes opens/creates so two coroutines never race to open one path.
        self._collections: dict[str, LodeDB] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ helpers
    def _vector_size(self) -> int:
        """Returns the embedding dimension, validated for a LodeDB vector index."""

        dim = int(self.embedding_engine.get_vector_size())
        if dim <= 0 or dim % 8 != 0:
            raise ValueError(
                f"embedding dimension {dim} is not a positive multiple of 8, which "
                "LodeDB vector indexes require; use an embedding model whose dimension "
                "is a multiple of 8 (e.g. 384, 768, 1024, 1536, 3072)"
            )
        return dim

    def _collection_key(self, collection_name: str) -> str:
        """Sanitizes a cognee collection name into its on-disk directory name.

        cognee collection names are ``Type_field`` identifiers and pass through
        unchanged; path separators are still folded so an unexpected name can never
        escape the base directory. This key is used for both the on-disk directory
        and the open-handle cache, so lookups and enumeration always agree.
        """

        safe = str(collection_name).replace(os.sep, "_")
        if os.altsep:
            safe = safe.replace(os.altsep, "_")
        safe = safe.strip().lstrip(".")
        if not safe or safe in {".", ".."}:
            raise ValueError(f"invalid collection name: {collection_name!r}")
        return safe

    def _collection_path(self, collection_name: str) -> Path:
        """Maps a cognee collection name to its on-disk LodeDB index directory."""

        return self._base_dir / self._collection_key(collection_name)

    def _open_collection(self, collection_name: str, *, create: bool) -> LodeDB | None:
        """Opens (and caches) the LodeDB index for a collection.

        Returns ``None`` without creating anything when ``create`` is False and the
        collection does not exist on disk, so read paths (search / retrieve / delete)
        stay side-effect free on a missing collection.
        """

        key = self._collection_key(collection_name)
        cached = self._collections.get(key)
        if cached is not None:
            return cached
        path = self._base_dir / key
        if not create and not path.exists():
            return None
        self._base_dir.mkdir(parents=True, exist_ok=True)
        db = LodeDB.open_vector_store(
            path,
            vector_dim=self._vector_size(),
            bit_width=self.bit_width,
            store_text=True,
        )
        self._collections[key] = db
        return db

    def _iter_existing_collection_names(self) -> list[str]:
        """Lists collection (directory) names that currently exist under the base dir.

        Both the open-handle cache keys and the on-disk directory names are already
        sanitized keys, so they can be unioned directly.
        """

        names = set(self._collections)
        if self._base_dir.exists():
            for child in self._base_dir.iterdir():
                if child.is_dir():
                    names.add(child.name)
        return sorted(names)

    # -------------------------------------------------------------- embed / info
    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """Embeds text via cognee's configured embedding engine."""

        return await self.embedding_engine.embed_text(list(data))

    async def has_collection(self, collection_name: str) -> bool:
        """Returns whether the collection's LodeDB index exists."""

        if self._collection_key(collection_name) in self._collections:
            return True
        return self._collection_path(collection_name).exists()

    async def create_collection(
        self,
        collection_name: str,
        payload_schema: Any | None = None,
    ) -> None:
        """Creates the collection's LodeDB index if absent (idempotent).

        ``payload_schema`` is accepted for interface compatibility but unused:
        LodeDB stores the full serialized payload as JSON in its text sidecar, so
        there is no fixed column schema to declare.
        """

        async with self._lock:
            self._open_collection(collection_name, create=True)

    # --------------------------------------------------------------- data points
    async def create_data_points(
        self,
        collection_name: str,
        data_points: list[DataPoint],
    ) -> None:
        """Embeds and upserts cognee DataPoints, unioning ``belongs_to_set`` tags.

        Tags are merged with any prior on-disk row for the same id and with
        duplicate ids within the batch, so re-adding a data point under a new
        NodeSet never drops the sets it already belonged to.
        """

        if not data_points:
            return
        # Embed outside the lock (the slow, store-independent step). The lock is then
        # held across open + prior-tag read + write so the belongs_to_set union is
        # atomic and a concurrent prune() cannot close the handle mid-write.
        vectors = await self.embed_data(
            [_embeddable_text(data_point) for data_point in data_points]
        )
        ids = [str(dp.id) for dp in data_points]
        async with self._lock:
            db = self._open_collection(collection_name, create=True)
            prior_tags = _prior_belongs_to_set(db, ids)

            merged: dict[str, dict[str, Any]] = {}
            for data_point, vector, doc_id in zip(data_points, vectors, ids, strict=True):
                tags = _tag_names(getattr(data_point, "belongs_to_set", None))
                payload = _serialize_data(data_point.model_dump())
                existing = merged.get(doc_id)
                base_tags = existing["tags"] if existing else prior_tags.get(doc_id, [])
                all_tags = _dedupe(base_tags + tags)
                payload["belongs_to_set"] = all_tags
                merged[doc_id] = {
                    "id": doc_id,
                    "vector": vector,
                    "tags": all_tags,
                    "payload": payload,
                }

            documents = [
                {
                    "id": item["id"],
                    "vector": item["vector"],
                    "metadata": _belongs_to_set_metadata(item["tags"]),
                    "text": _encode_payload(item["payload"]),
                }
                for item in merged.values()
            ]
            db.add_vectors_many(documents, normalize=self.normalize)

    async def upsert_raw_vectors(
        self,
        collection_name: str,
        points: list[dict],
        payload_schema: Any | None = None,
    ) -> None:
        """Upserts caller-supplied ``{id, vector, payload}`` rows without embedding.

        Used for small system-owned vector state (e.g. truth-subspace centroids)
        where re-embedding from text would be wrong.
        """

        if not points:
            return
        documents = []
        for point in points:
            point_id = point.get("id")
            vector = point.get("vector")
            if point_id is None:
                raise ValueError("raw vector point is missing 'id'")
            if vector is None:
                raise ValueError("raw vector point is missing 'vector'")
            payload = _serialize_data(dict(point.get("payload") or {}))
            tags = _tag_names(payload.get("belongs_to_set"))
            if tags:
                payload["belongs_to_set"] = tags
            documents.append(
                {
                    "id": str(point_id),
                    "vector": list(vector),
                    "metadata": _belongs_to_set_metadata(tags),
                    "text": _encode_payload(payload),
                }
            )
        # Hold the lock across open + write so a concurrent prune() cannot close the
        # handle between them (upsert replaces rows; no prior-tag union, matching the
        # built-in adapters' raw-vector upsert semantics).
        async with self._lock:
            db = self._open_collection(collection_name, create=True)
            db.add_vectors_many(documents, normalize=self.normalize)

    async def retrieve(self, collection_name: str, data_point_ids: list[str]):
        """Returns the stored payloads for the given ids (score 0, unranked)."""

        db = self._open_collection(collection_name, create=False)
        if db is None or not data_point_ids:
            return []
        ids = [str(value) for value in data_point_ids]
        payloads = _payloads_for_ids(db, ids)
        return [
            ScoredResult(id=parse_id(doc_id), payload=payloads[doc_id], score=0)
            for doc_id in ids
            if doc_id in payloads
        ]

    # -------------------------------------------------------------------- search
    async def search(
        self,
        collection_name: str,
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        limit: int | None = 15,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
        node_name_filter_operator: str = "OR",
    ):
        """Vector search returning ``ScoredResult`` rows ranked by cosine distance.

        A missing collection returns ``[]`` (cognee's brute-force fallback queries
        collections that may not exist). ``limit=None`` searches the whole
        collection. ``with_vector`` is accepted but has no effect: ``ScoredResult``
        carries no vector field and LodeDB does not read stored vectors back.
        """

        db = self._open_collection(collection_name, create=False)
        if db is None:
            return []
        if query_vector is None:
            if query_text is None:
                raise MissingQueryParameterError()
            query_vector = (await self.embed_data([query_text]))[0]

        if limit is None:
            limit = db.count()
        if limit <= 0:
            return []

        filter = _node_name_filter(node_name, node_name_filter_operator)
        hits = db.search_by_vector(
            list(query_vector), k=int(limit), filter=filter, normalize=self.normalize
        )
        payloads = (
            _payloads_for_ids(db, [hit.id for hit in hits]) if include_payload and hits else {}
        )
        return [
            ScoredResult(
                id=parse_id(hit.id),
                payload=payloads.get(hit.id) if include_payload else None,
                score=1.0 - float(hit.score),
            )
            for hit in hits
        ]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int | None = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
    ):
        """Runs one search per query text, embedding the batch in a single call."""

        if not query_texts:
            return []
        query_vectors = await self.embed_data(query_texts)
        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                    include_payload=include_payload,
                    node_name=node_name,
                )
                for query_vector in query_vectors
            ]
        )

    # -------------------------------------------------------------------- delete
    async def delete_data_points(self, collection_name: str, data_point_ids: list) -> None:
        """Deletes data points by id (no-op for a missing collection)."""

        db = self._open_collection(collection_name, create=False)
        if db is None:
            return
        for data_point_id in data_point_ids:
            db.remove(str(data_point_id))

    async def remove_belongs_to_set_tags(
        self,
        tags: list[str],
        node_ids: list[str] | None = None,
    ) -> None:
        """Strips NodeSet tags from every collection, deleting rows left tagless.

        Called when a dataset / NodeSet is removed: every row that referenced one of
        ``tags`` has those tags dropped from both its payload and its metadata
        presence keys, and any row whose ``belongs_to_set`` becomes empty is deleted.
        With ``node_ids`` the rewrite is scoped to those ids (shared rows that lose
        one dataset's anchor while others still own them).
        """

        if not tags:
            return
        if node_ids is not None and not node_ids:
            return
        tag_set = set(_tag_names(tags))
        id_set = {str(node_id) for node_id in node_ids} if node_ids is not None else None

        # Hold the lock across this multi-collection read-modify-write so tag cleanup
        # is serialized against concurrent writers and prune().
        async with self._lock:
            for collection_name in self._iter_existing_collection_names():
                db = self._open_collection(collection_name, create=False)
                if db is None:
                    continue
                match_filter = _node_name_filter(sorted(tag_set), "OR")
                records = db.list_documents(filter=match_filter)
                candidate_ids = [
                    record["id"]
                    for record in records
                    if id_set is None or record["id"] in id_set
                ]
                if not candidate_ids:
                    continue
                payloads = _payloads_for_ids(db, candidate_ids)
                for doc_id in candidate_ids:
                    payload = payloads.get(doc_id)
                    if payload is None:
                        continue
                    current = _tag_names(payload.get("belongs_to_set"))
                    remaining = [tag for tag in current if tag not in tag_set]
                    if remaining == current:
                        continue
                    if remaining:
                        payload["belongs_to_set"] = remaining
                        db._update_document_payload(
                            doc_id,
                            metadata=_belongs_to_set_metadata(remaining),
                            text=_encode_payload(payload),
                            clear_text=False,
                        )
                    else:
                        db.remove(doc_id)

    # --------------------------------------------------------------- index verbs
    async def create_vector_index(self, index_name: str, index_property_name: str) -> None:
        """Creates the index collection ``f"{index_name}_{index_property_name}"``."""

        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self,
        index_name: str,
        index_property_name: str,
        data_points: list[DataPoint],
    ) -> None:
        """Indexes the ``index_property_name`` field of each data point as text."""

        points = [
            IndexSchema(
                id=str(data_point.id),
                text=str(_embeddable_text(data_point)),
                belongs_to_set=_tag_names(getattr(data_point, "belongs_to_set", None)),
            )
            for data_point in data_points
        ]
        await self.create_data_points(f"{index_name}_{index_property_name}", points)

    # ------------------------------------------------------------------- lifecycle
    async def prune(self) -> None:
        """Deletes every collection and its on-disk data."""

        async with self._lock:
            for db in self._collections.values():
                try:
                    db.close()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass
            self._collections.clear()
            if self._base_dir.exists():
                shutil.rmtree(self._base_dir, ignore_errors=True)

    def close(self) -> None:
        """Closes every open LodeDB handle; on-disk data stays put. Idempotent."""

        for db in self._collections.values():
            try:
                db.close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        self._collections.clear()

    # ---------------------------------------------------- optional / no-op hooks
    async def get_connection(self):
        """LodeDB is embedded and has no connection object; returns ``None``."""

        return None

    async def get_collection(self, collection_name: str):
        """Returns the underlying :class:`LodeDB` handle for a collection, or ``None``."""

        return self._open_collection(collection_name, create=False)

    async def run_migrations(self):
        """No stored-vector migrations for LodeDB; a no-op."""

        return None

    def get_data_point_schema(self, model_type: Any) -> Any:
        """Returns ``model_type`` unchanged (LodeDB stores payloads as JSON)."""

        return model_type


def register_cognee_adapter(provider_name: str = "lodedb") -> None:
    """Registers this adapter with cognee's vector-adapter registry.

    Call once before configuring cognee, then select LodeDB with
    ``cognee.config.set_vector_db_config({"vector_db_provider": "lodedb",
    "vector_db_url": "<dir>"})``.
    """

    from cognee.infrastructure.databases.vector import use_vector_adapter

    use_vector_adapter(str(provider_name), CogneeLodeDBAdapter)


# --------------------------------------------------------------------- utilities
def _embeddable_text(data_point: DataPoint) -> str:
    """Returns the text cognee would embed for a data point (empty string if none)."""

    value = DataPoint.get_embeddable_data(data_point)
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _tag_names(belongs_to_set: Any) -> list[str]:
    """Normalizes a ``belongs_to_set`` value to a de-duplicated list of tag names."""

    if not belongs_to_set:
        return []
    names: list[str] = []
    for item in belongs_to_set:
        if isinstance(item, str):
            names.append(item)
            continue
        name = getattr(item, "name", None)
        names.append(str(name) if name is not None else str(item))
    return _dedupe(names)


def _dedupe(values: list[str]) -> list[str]:
    """Order-preserving de-duplication."""

    return list(dict.fromkeys(values))


def _belongs_to_set_metadata(tags: list[str]) -> dict[str, Any]:
    """Encodes tag membership as scalar presence keys for LodeDB metadata."""

    return {f"{_BELONGS_TO_SET_PREFIX}{tag}": True for tag in tags}


def _node_name_filter(
    node_name: list[str] | None,
    operator: str,
) -> dict[str, Any] | None:
    """Builds a LodeDB metadata filter for cognee ``node_name`` (belongs_to_set)."""

    tags = _tag_names(node_name) if node_name else []
    if not tags:
        return None
    clauses = [{f"{_BELONGS_TO_SET_PREFIX}{tag}": {"$exists": True}} for tag in tags]
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses} if operator == "AND" else {"$or": clauses}


def _prior_belongs_to_set(db: LodeDB, ids: list[str]) -> dict[str, list[str]]:
    """Reads existing ``belongs_to_set`` tags for ids already present in the index."""

    payloads = _payloads_for_ids(db, ids)
    return {
        doc_id: _tag_names(payload.get("belongs_to_set"))
        for doc_id, payload in payloads.items()
    }


def _payloads_for_ids(db: LodeDB, ids: list[str]) -> dict[str, dict[str, Any]]:
    """Returns ``{id: payload}`` for the ids whose payload JSON is stored."""

    if not ids:
        return {}
    try:
        texts = db.get_texts(ids)
    except ValueError:
        return {}
    return {doc_id: _decode_payload(raw) for doc_id, raw in texts.items()}


def _encode_payload(payload: dict[str, Any]) -> str:
    """Serializes a payload dict to compact JSON for the LodeDB text sidecar."""

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _decode_payload(raw: str) -> dict[str, Any]:
    """Parses stored payload JSON back to a dict (empty dict on bad JSON)."""

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _serialize_data(data: Any) -> Any:
    """Recursively converts UUID / datetime leaves to strings (JSON-safe).

    A local copy of cognee's ``serialize_data`` so the adapter does not import
    cognee's pgvector subpackage (which pulls SQLAlchemy) just for this helper.
    Non-string dict keys are stringified as well, so a payload with UUID / int /
    tuple keys serializes cleanly (JSON keys are strings regardless) instead of
    raising ``TypeError`` and failing the whole batch.
    """

    if isinstance(data, dict):
        return {
            (key if isinstance(key, str) else str(key)): _serialize_data(value)
            for key, value in data.items()
        }
    if isinstance(data, list):
        return [_serialize_data(item) for item in data]
    if isinstance(data, datetime):
        return data.isoformat()
    if isinstance(data, UUID):
        return str(data)
    return data


def parse_id(value: Any) -> Any:
    """Converts a string id to :class:`UUID` when possible, else returns it as-is.

    Matches cognee's ``parse_id`` semantics so ``ScoredResult.id`` is a ``UUID`` for
    cognee's canonical UUID data-point ids, without importing cognee's engine utils.
    """

    if isinstance(value, str):
        try:
            return UUID(value)
        except ValueError:
            return value
    return value


__all__ = [
    "CogneeLodeDBAdapter",
    "IndexSchema",
    "register_cognee_adapter",
]
