"""mem0 vector-store adapter for LodeDB (optional ``lodedb[mem0]``).

``LodeDBVectorStore`` implements mem0's ``VectorStoreBase`` against LodeDB's
vector-in API: mem0 owns the embeddings, while LodeDB stores and searches those
vectors locally and persists changed rows incrementally. mem0 payloads can carry
raw memory text and list-valued fields (such as ``linked_memory_ids``), so the
full payload JSON is serialized into LodeDB's dedicated raw-text sidecar, never
into the redacted metadata. LodeDB metadata keeps only scalar filter fields such
as ``user_id`` / ``agent_id`` / ``run_id`` so mem0's filtered reads stay exact.

This targets ``mem0ai>=2.0.0``. mem0's v2 line dropped the optional OSS graph
memory layer (there is no ``graph_store`` config and no ``GraphStoreBase``), so
this adapter is vector-store only. For a LodeDB-backed graph, use
:class:`lodedb.graph.KnowledgeGraph` directly or the LlamaIndex
``LodeDBPropertyGraphStore``.
"""

from __future__ import annotations

import json
import sys
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lodedb.engine._lexical import Bm25Index
from lodedb.local.db import LodeDB

try:
    from mem0.vector_stores.base import VectorStoreBase
    from pydantic import BaseModel, ConfigDict, Field, model_validator
except ImportError as exc:  # pragma: no cover - clear install hint
    raise ImportError("the LodeDB mem0 adapter needs mem0ai: pip install 'lodedb[mem0]'") from exc


_FILTERABLE_SCALARS = (str, int, float, bool)
_RAW_PAYLOAD_KEYS = {
    "data",
    "memory",
    "text",
    "text_lemmatized",
    "document",
    "raw_payload",
}
_OPERATOR_MAP = {
    "eq": "$eq",
    "ne": "$ne",
    "gt": "$gt",
    "gte": "$gte",
    "lt": "$lt",
    "lte": "$lte",
    "in": "$in",
    "nin": "$nin",
}
_UNSUPPORTED_FILTER_OPERATORS = {"contains", "icontains"}


@dataclass
class LodeDBMem0Result:
    """mem0-compatible result object with id, score, and payload attributes."""

    id: str
    score: float | None = None
    payload: dict[str, Any] | None = None


class LodeDBConfig(BaseModel):
    """Pydantic config used when registering LodeDB with mem0's factory."""

    collection_name: str = Field("mem0", description="Logical mem0 collection name")
    path: str | None = Field(None, description="Directory for LodeDB-backed mem0 data (required)")
    embedding_model_dims: int = Field(1536, description="Dimension of mem0 embeddings")
    distance_strategy: str = Field("cosine", description="Only cosine is supported by LodeDB")
    normalize: bool = Field(True, description="Normalize vectors on insert/search for cosine")
    bit_width: int = Field(4, description="TurboVec quantization bit width")

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        allowed = set(cls.model_fields)
        extra = set(values) - allowed
        if extra:
            raise ValueError(
                "Extra fields not allowed: "
                + ", ".join(sorted(extra))
                + ". Please input only the following fields: "
                + ", ".join(sorted(allowed))
            )
        dims = int(
            values.get("embedding_model_dims", cls.model_fields["embedding_model_dims"].default)
        )
        if dims <= 0 or dims % 8 != 0:
            raise ValueError(
                "embedding_model_dims must be a positive multiple of 8 for LodeDB vector indexes"
            )
        return values

    model_config = ConfigDict(arbitrary_types_allowed=True)


class LodeDBVectorStore(VectorStoreBase):
    """mem0 ``VectorStoreBase`` backed by a vector-only :class:`LodeDB`.

    ``path`` is required: it is the directory the collection persists to (passed
    through from the mem0 ``vector_store`` config). There is no default, so memory
    is never written somewhere ephemeral by accident.

    ``store_payloads`` (default True) keeps the full mem0 payload JSON in LodeDB's
    raw-text sidecar. Leave it on for a live mem0 backend; setting it False makes
    the store filter-only (reads lose ``data`` / ``linked_memory_ids``). It is a
    direct-constructor option only, not exposed via :class:`LodeDBConfig`, so the
    mem0 factory path always retains payloads.
    """

    def __init__(
        self,
        collection_name: str = "mem0",
        path: str | None = None,
        embedding_model_dims: int = 1536,
        distance_strategy: str = "cosine",
        normalize: bool = True,
        bit_width: int = 4,
        store_payloads: bool = True,
        db: LodeDB | None = None,
    ) -> None:
        """Opens a vector-only LodeDB collection for mem0-owned embeddings."""

        if distance_strategy.lower() not in {"cosine", "inner_product"}:
            raise ValueError("LodeDBVectorStore supports cosine-style similarity only")
        self.collection_name = str(collection_name)
        self.embedding_model_dims = int(embedding_model_dims)
        if self.embedding_model_dims <= 0 or self.embedding_model_dims % 8 != 0:
            raise ValueError(
                "embedding_model_dims must be a positive multiple of 8 for LodeDB vector indexes"
            )
        self.distance_strategy = distance_strategy
        self.normalize = bool(normalize)
        self.store_payloads = bool(store_payloads)
        self.path = _collection_path(path, self.collection_name)
        self._db = db or LodeDB.open_vector_store(
            self.path,
            vector_dim=self.embedding_model_dims,
            bit_width=int(bit_width),
            store_text=self.store_payloads,
        )

    @property
    def client(self) -> LodeDB:
        """Returns the underlying LodeDB handle."""

        return self._db

    def create_col(self, name, vector_size=None, distance=None):
        """Creates or switches the logical collection name.

        The LodeDB directory is opened during ``__init__``. mem0 calls this on
        some providers as an idempotent collection initializer, so returning
        ``self`` is the useful behavior here.
        """

        self.collection_name = str(name)
        if vector_size is not None and int(vector_size) != self.embedding_model_dims:
            raise ValueError(
                f"collection dimension is {self.embedding_model_dims}, got {vector_size}"
            )
        if distance and str(distance).lower() not in {"cosine", "inner_product"}:
            raise ValueError("LodeDBVectorStore supports cosine-style similarity only")
        return self

    def insert(self, vectors, payloads=None, ids=None):
        """Inserts or upserts vectors, retaining full mem0 payload JSON durably."""

        vectors = [] if vectors is None else list(vectors)
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        else:
            ids = [str(value) for value in ids]
        if payloads is None:
            payloads = [{} for _ in vectors]
        else:
            payloads = [dict(payload or {}) for payload in payloads]
        if len(vectors) != len(ids) or len(vectors) != len(payloads):
            raise ValueError("vectors, payloads, and ids must have the same length")
        documents = []
        for vector, vector_id, payload in zip(vectors, ids, payloads, strict=True):
            documents.append(
                {
                    "vector": vector,
                    "id": vector_id,
                    "metadata": _filterable_metadata(payload),
                    "text": _encode_payload(payload) if self.store_payloads else None,
                }
            )
        if documents:
            self._db.add_vectors_many(documents, normalize=self.normalize)

    def search(self, query, vectors, top_k=5, filters=None):
        """Searches by mem0's precomputed query vector."""

        lode_filter = _translate_mem0_filters(filters)
        hits = self._db.search_by_vector(
            _one_query_vector(vectors),
            k=int(top_k),
            filter=lode_filter,
            normalize=self.normalize,
        )
        return self._results_from_hits(hits)

    def search_batch(self, queries: list, vectors_list: list, top_k: int = 1, filters=None):
        """Batch vector search using LodeDB's vector-in batch query path."""

        lode_filter = _translate_mem0_filters(filters)
        batches = self._db.search_many_by_vector(
            [_one_query_vector(vector) for vector in vectors_list],
            k=int(top_k),
            filter=lode_filter,
            normalize=self.normalize,
        )
        return [self._results_from_hits(hits) for hits in batches]

    def keyword_search(self, query: str, top_k: int = 5, filters=None):
        """BM25 keyword search over retained mem0 payload text.

        Returns ``[]`` when payload retention is off (``store_payloads=False``),
        since there is then no payload text to rank.
        """

        if not self.store_payloads:
            return []
        records = self._db.list_documents(filter=_translate_mem0_filters(filters))
        if not records:
            return []
        payloads = self._payloads_for_ids([record["id"] for record in records], records=records)
        ids: list[str] = []
        texts: list[str] = []
        for record in records:
            payload = payloads.get(record["id"], {})
            text = _payload_keyword_text(payload)
            if text:
                ids.append(record["id"])
                texts.append(text)
        if not ids:
            return []
        ranked = Bm25Index(ids, texts).rank(str(query), limit=int(top_k))
        return [
            LodeDBMem0Result(id=doc_id, score=float(score), payload=payloads.get(doc_id, {}))
            for doc_id, score in ranked
        ]

    def delete(self, vector_id):
        """Deletes one vector by id."""

        self._db.remove(str(vector_id))

    def update(self, vector_id, vector=None, payload=None):
        """Updates a vector and/or its mem0 payload."""

        vector_id = str(vector_id)
        if payload is None:
            current = self.get(vector_id)
            payload = dict(current.payload or {}) if current else {}
        else:
            payload = dict(payload)
        text = _encode_payload(payload) if self.store_payloads else None
        metadata = _filterable_metadata(payload)
        if vector is not None:
            self._db.add_vectors(
                vector,
                id=vector_id,
                metadata=metadata,
                text=text,
                normalize=self.normalize,
            )
            return
        self._db._update_document_payload(
            vector_id,
            metadata=metadata,
            text=text,
            clear_text=False,
        )

    def get(self, vector_id):
        """Returns a mem0 result object for one id, or ``None`` if absent."""

        vector_id = str(vector_id)
        record = self._db.get_document(vector_id)
        if record is None:
            return None
        payload = self._payloads_for_ids([vector_id], records=[record]).get(vector_id, {})
        return LodeDBMem0Result(id=vector_id, score=None, payload=payload)

    def list_cols(self):
        """Lists this adapter's logical collection."""

        return [self.collection_name] if self.path.exists() else []

    def delete_col(self):
        """Clears every document in the collection."""

        for record in list(self._db.list_documents()):
            self._db.remove(record["id"])

    def col_info(self):
        """Returns collection stats in mem0's provider style."""

        stats = self._db.stats()
        return {
            "name": self.collection_name,
            "count": int(stats.get("document_count", 0) or 0),
            "dimension": self.embedding_model_dims,
            "distance": "cosine",
            "path": str(self.path),
        }

    def list(self, filters=None, top_k=None):
        """Lists payloads matching filters; mem0 expects a list containing rows."""

        records = self._db.list_documents(filter=_translate_mem0_filters(filters))
        if top_k is not None:
            records = records[: int(top_k)]
        payloads = self._payloads_for_ids([record["id"] for record in records], records=records)
        return [
            [
                LodeDBMem0Result(
                    id=record["id"],
                    score=None,
                    payload=payloads.get(record["id"], {}),
                )
                for record in records
            ]
        ]

    def reset(self):
        """Clears the collection in place."""

        self.delete_col()

    def close(self) -> None:
        """Closes the underlying LodeDB handle."""

        self._db.close()

    def _results_from_hits(self, hits) -> list[LodeDBMem0Result]:
        ids = [hit.id for hit in hits]
        payloads = self._payloads_for_ids(ids)
        return [
            LodeDBMem0Result(id=hit.id, score=float(hit.score), payload=payloads.get(hit.id, {}))
            for hit in hits
        ]

    def _payloads_for_ids(
        self,
        ids: list[str],
        *,
        records: list[dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}
        by_id = {record["id"]: record for record in records or []}
        texts: dict[str, str] = {}
        if self.store_payloads:
            try:
                texts = self._db.get_texts(ids)
            except ValueError:
                texts = {}
        payloads: dict[str, dict[str, Any]] = {}
        for doc_id in ids:
            if doc_id in texts:
                payloads[doc_id] = _decode_payload(texts[doc_id])
                continue
            record = by_id.get(doc_id) or self._db.get_document(doc_id)
            payloads[doc_id] = dict(record.get("metadata", {})) if record else {}
        return payloads


def register_mem0_provider(provider_name: str = "lodedb") -> None:
    """Registers this adapter with mem0's runtime config and vector-store factory.

    mem0 keeps provider registries inside the mem0 package. Calling this once
    before ``Memory.from_config(...)`` enables:

    ``{"vector_store": {"provider": "lodedb", "config": {"path": "./mem0"}}}``
    """

    from mem0.utils.factory import VectorStoreFactory
    from mem0.vector_stores.configs import VectorStoreConfig

    provider_name = str(provider_name)
    module = types.ModuleType(f"mem0.configs.vector_stores.{provider_name}")
    module.LodeDBConfig = LodeDBConfig
    sys.modules[module.__name__] = module

    class_path = "lodedb.local.integrations.mem0.LodeDBVectorStore"
    VectorStoreFactory.provider_to_class[provider_name] = class_path

    private_attrs = getattr(VectorStoreConfig, "__private_attributes__", {})
    provider_attr = (
        private_attrs.get("_provider_configs") if isinstance(private_attrs, dict) else None
    )
    if provider_attr is not None and isinstance(getattr(provider_attr, "default", None), dict):
        provider_attr.default[provider_name] = "LodeDBConfig"
    provider_configs = getattr(VectorStoreConfig, "_provider_configs", None)
    if isinstance(provider_configs, dict):
        provider_configs[provider_name] = "LodeDBConfig"


def _collection_path(path: str | None, collection_name: str) -> Path:
    if path is None:
        raise ValueError(
            "LodeDBVectorStore requires an explicit 'path' for durable storage "
            "(pass path=... or set 'path' in the mem0 vector_store config)"
        )
    return Path(path) / collection_name


def _filterable_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    """Returns the scalar, filterable subset of a mem0 payload for LodeDB metadata.

    Raw payload text and list fields stay in the sidecar (see ``_RAW_PAYLOAD_KEYS``
    and ``linked_memory_ids``). ``None`` values are dropped rather than stored:
    LodeDB metadata is scalar-only, and an absent key is the correct semantics for
    mem0's ``"*"`` (field-present) filter.
    """

    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _RAW_PAYLOAD_KEYS or key == "linked_memory_ids":
            continue
        if isinstance(value, _FILTERABLE_SCALARS):
            metadata[str(key)] = value
    return metadata


def _encode_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _decode_payload(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _one_query_vector(vector: Any) -> list[float]:
    if isinstance(vector, tuple):
        vector = list(vector)
    if not isinstance(vector, list) or not vector:
        raise ValueError("query vector must be a non-empty list")
    first = vector[0]
    if isinstance(first, (list, tuple)):
        return [float(value) for value in first]
    return [float(value) for value in vector]


def _payload_keyword_text(payload: dict[str, Any]) -> str:
    for key in ("text_lemmatized", "data", "memory", "text"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _translate_mem0_filters(filters: dict | None) -> dict[str, Any] | None:
    if not filters:
        return None
    return _translate_filter_node(filters)


def _translate_filter_node(filters: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in filters.items():
        if key in _RAW_PAYLOAD_KEYS:
            raise NotImplementedError(
                f"mem0 filter field {key!r} is stored as retained payload text, not LodeDB metadata"
            )
        if key in ("AND", "$and"):
            out["$and"] = [_translate_filter_node(item) for item in _filter_list("AND", value)]
            continue
        if key in ("OR", "$or"):
            out["$or"] = [_translate_filter_node(item) for item in _filter_list("OR", value)]
            continue
        if key in ("NOT", "$not"):
            parts = [_translate_filter_node(item) for item in _filter_list("NOT", value)]
            out["$not"] = parts[0] if len(parts) == 1 else {"$or": parts}
            continue
        out[str(key)] = _translate_filter_value(str(key), value)
    return out


def _filter_list(name: str, value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} filter value must be a non-empty list of dicts")
    if not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{name} filter value must contain only dicts")
    return value


def _translate_filter_value(key: str, value: Any) -> Any:
    if value == "*":
        return {"$exists": True}
    if isinstance(value, list):
        return {"$in": value}
    if not isinstance(value, dict):
        return value
    translated: dict[str, Any] = {}
    for op, operand in value.items():
        if op in _UNSUPPORTED_FILTER_OPERATORS:
            raise NotImplementedError(
                f"mem0 filter operator {op!r} for field {key!r} is not supported by LodeDB"
            )
        lode_op = _OPERATOR_MAP.get(op)
        if lode_op is None:
            raise ValueError(f"unsupported mem0 filter operator {op!r} for field {key!r}")
        translated[lode_op] = operand
    return translated


__all__ = [
    "LodeDBConfig",
    "LodeDBMem0Result",
    "LodeDBVectorStore",
    "register_mem0_provider",
]
