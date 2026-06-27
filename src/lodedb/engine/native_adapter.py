"""Private Python compatibility adapter for the native core.

This module is intentionally not part of the public API. It keeps the current
Python engine dataclasses as the oracle-facing shape and translates them to the
native-core JSON contracts used by the hidden ``_native_core`` extension. The
extension is imported lazily so ``import lodedb`` remains dependency-light and a
missing native module is a rollout decision, not an import-time failure.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, Protocol

from lodedb.engine.core import EngineDocument, EngineQuery, EngineResponse, EngineVectorDocument


class NativeCorePayload(dict[str, Any]):
    """Native JSON-backed dict for private adapter round-trips."""

    def __init__(self, payload: Mapping[str, Any], *, native_json: str) -> None:
        super().__init__(payload)
        self.native_json = native_json


class NativeCoreModule(Protocol):
    """Subset of the hidden native module used by the adapter."""

    def CoreEngine(self) -> Any: ...
    def native_core_abi_version(self) -> int: ...
    def native_core_version(self) -> str: ...
    def round_trip_core_json(self, type_name: str, json_payload: str) -> str: ...


@dataclass(frozen=True)
class NativeCoreError:
    """Stable native error mapped into the endpoint-shaped Python response."""

    code: str
    message: str


class NativeCoreAdapter:
    """Maps Python engine dataclasses to hidden native-core JSON contracts."""

    def __init__(self, native_module: NativeCoreModule | None = None) -> None:
        self._native_module = native_module

    @property
    def available(self) -> bool:
        return self._module_or_none() is not None

    @property
    def version(self) -> str:
        module = self._module_or_none()
        if module is None:
            return ""
        return str(module.native_core_version())

    @property
    def abi_version(self) -> int:
        module = self._module_or_none()
        if module is None:
            return 0
        return int(module.native_core_abi_version())

    def document_json(self, document: EngineDocument) -> str:
        return json.dumps(self.document_payload(document), sort_keys=True, separators=(",", ":"))

    def vector_document_json(self, document: EngineVectorDocument) -> str:
        return json.dumps(
            self.vector_document_payload(document), sort_keys=True, separators=(",", ":")
        )

    def query_json(self, query: EngineQuery) -> str:
        return json.dumps(self.query_payload(query), sort_keys=True, separators=(",", ":"))

    def round_trip(self, type_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        module = self._require_module()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return json.loads(module.round_trip_core_json(type_name, encoded))

    def new_engine(self) -> NativeCoreEngineHandle:
        """Creates an in-memory native engine handle through the hidden extension."""

        module = self._require_module()
        return NativeCoreEngineHandle(module.CoreEngine())

    def open_engine(
        self,
        *,
        path: str | PathLike[str],
        read_only: bool,
        durability: str,
        commit_mode: str,
        store_text: bool,
        index_text: bool,
    ) -> NativeCoreEngineHandle:
        """Opens a persistent native engine handle through the hidden extension."""

        module = self._require_module()
        options = self.open_options_payload(
            path=path,
            read_only=read_only,
            durability=durability,
            commit_mode=commit_mode,
            store_text=store_text,
            index_text=index_text,
        )
        return NativeCoreEngineHandle(module.CoreEngine.open(self._dumps(options)))

    def open_readonly_engine(
        self,
        path: str | PathLike[str],
        *,
        durability: str,
        commit_mode: str,
        store_text: bool,
        index_text: bool,
    ) -> NativeCoreEngineHandle:
        """Opens a lock-free read-only native engine snapshot."""

        module = self._require_module()
        options = self.open_options_payload(
            path=path,
            read_only=True,
            durability=durability,
            commit_mode=commit_mode,
            store_text=store_text,
            index_text=index_text,
        )
        return NativeCoreEngineHandle(
            module.CoreEngine.open_readonly(str(path), self._dumps(options))
        )

    @staticmethod
    def document_payload(document: EngineDocument) -> dict[str, Any]:
        return {
            "document_id": str(document.document_id),
            "text": str(document.text),
            "metadata": {str(key): str(value) for key, value in document.metadata.items()},
        }

    @staticmethod
    def vector_document_payload(document: EngineVectorDocument) -> dict[str, Any]:
        return {
            "document_id": str(document.document_id),
            "vector": [float(value) for value in document.vector],
            "metadata": {str(key): str(value) for key, value in document.metadata.items()},
            "text": None if document.text is None else str(document.text),
        }

    @staticmethod
    def query_payload(query: EngineQuery) -> dict[str, Any]:
        return {
            "text": str(query.text),
            "top_k": int(query.top_k),
            "filter": query.filter,
            "include": [str(value) for value in query.include],
            "mode": str(query.mode),
            "embedding": (
                None if query.embedding is None else [float(value) for value in query.embedding]
            ),
        }

    @staticmethod
    def open_options_payload(
        *,
        path: str | PathLike[str],
        read_only: bool,
        durability: str,
        commit_mode: str,
        store_text: bool,
        index_text: bool,
    ) -> dict[str, Any]:
        return {
            "path": str(path),
            "read_only": bool(read_only),
            "durability": str(durability),
            "commit_mode": str(commit_mode),
            "store_text": bool(store_text),
            "index_text": bool(index_text),
        }

    @staticmethod
    def index_create_options_payload(
        *,
        index_id: str,
        index_key: str,
        client_id_hash: str,
        name: str,
        model: str,
        provider: str,
        task: str,
        route_profile: str,
        storage_profile: str,
        vector_dim: int,
        bit_width: int,
    ) -> dict[str, Any]:
        return {
            "index_id": str(index_id),
            "index_key": str(index_key),
            "client_id_hash": str(client_id_hash),
            "name": str(name),
            "model": str(model),
            "provider": str(provider),
            "task": str(task),
            "route_profile": str(route_profile),
            "storage_profile": str(storage_profile),
            "vector_dim": int(vector_dim),
            "bit_width": int(bit_width),
        }

    @staticmethod
    def response_from_native(status_code: int, payload: dict[str, Any]) -> EngineResponse:
        return EngineResponse(int(status_code), dict(payload))

    @staticmethod
    def response_from_error(error: NativeCoreError) -> EngineResponse:
        status_code = 404 if error.code == "NOT_FOUND" else 400
        if error.code in {"CORRUPT_STORE", "INTERNAL"}:
            status_code = 500
        return EngineResponse(
            status_code,
            {
                "status": "error",
                "error": error.message,
                "native_core_error": error.code,
            },
        )

    @staticmethod
    def _dumps(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _require_module(self) -> NativeCoreModule:
        module = self._module_or_none()
        if module is None:
            raise RuntimeError("native core extension is not available")
        return module

    def _module_or_none(self) -> NativeCoreModule | None:
        if self._native_module is not None:
            return self._native_module
        try:
            self._native_module = importlib.import_module("lodedb._native_core")
        except ImportError:
            return None
        return self._native_module


class NativeCoreEngineHandle:
    """Small JSON-backed wrapper over ``lodedb._native_core.CoreEngine``."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def create_index(self, index_id: str, *, vector_dim: int, bit_width: int = 4) -> None:
        self._engine.create_index(str(index_id), int(vector_dim), int(bit_width))

    def create_index_with_options(self, options: Mapping[str, Any]) -> None:
        self._engine.create_index_with_options(self._dumps(dict(options)))

    def upsert_vectors(
        self,
        index_id: str,
        documents: Iterable[EngineVectorDocument],
    ) -> dict[str, Any]:
        payload = [NativeCoreAdapter.vector_document_payload(document) for document in documents]
        return self._loads(
            self._engine.upsert_vectors(
                str(index_id),
                self._dumps(payload),
            )
        )

    def delete_documents(self, index_id: str, document_ids: Iterable[str]) -> dict[str, Any]:
        return self._loads(
            self._engine.delete_documents(
                str(index_id),
                self._dumps([str(document_id) for document_id in document_ids]),
            )
        )

    def query_vector(
        self,
        index_id: str,
        vector: Iterable[float],
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._loads(
            self._engine.query_vector(
                str(index_id),
                self._dumps([float(value) for value in vector]),
                int(top_k),
                None if filter is None else self._dumps(dict(filter)),
            )
        )

    def query_vectors_batch(
        self,
        index_id: str,
        vectors: Iterable[Iterable[float]],
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        value = json.loads(
            self._engine.query_vectors_batch(
                str(index_id),
                self._dumps([[float(value) for value in vector] for vector in vectors]),
                int(top_k),
                None if filter is None else self._dumps(dict(filter)),
            )
        )
        if not isinstance(value, list):
            raise RuntimeError("native core returned a non-list JSON payload")
        return [dict(item) for item in value]

    def prepare_text_upsert(
        self,
        index_id: str,
        documents: Iterable[EngineDocument],
        *,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: int,
    ) -> dict[str, Any]:
        payload = [NativeCoreAdapter.document_payload(document) for document in documents]
        plan_json = self._engine.prepare_text_upsert(
            str(index_id),
            self._dumps(payload),
            bool(store_text),
            bool(index_text),
            int(chunk_character_limit),
        )
        return self._loads_native_payload(plan_json)

    def apply_text_upsert(
        self,
        plan: Mapping[str, Any],
        embeddings: Iterable[Iterable[float]],
        *,
        embedding_time_ms: float,
    ) -> dict[str, Any]:
        embedding_payload = [[float(value) for value in row] for row in embeddings]
        plan_json = (
            plan.native_json
            if isinstance(plan, NativeCorePayload)
            else self._dumps(dict(plan))
        )
        return self._loads(
            self._engine.apply_text_upsert(
                plan_json,
                self._dumps(embedding_payload),
                float(embedding_time_ms),
            )
        )

    def prepare_query_text(self, query: str, mode: str) -> dict[str, Any]:
        return self._loads_native_payload(self._engine.prepare_query_text(str(query), str(mode)))

    def search_embedded_text(
        self,
        index_id: str,
        query_plan: Mapping[str, Any],
        query_embedding: Iterable[float] | None,
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_plan_json = (
            query_plan.native_json
            if isinstance(query_plan, NativeCorePayload)
            else self._dumps(dict(query_plan))
        )
        return self._loads(
            self._engine.search_embedded_text(
                str(index_id),
                query_plan_json,
                None
                if query_embedding is None
                else self._dumps([float(value) for value in query_embedding]),
                int(top_k),
                None if filter is None else self._dumps(dict(filter)),
            )
        )

    def stats(self, index_id: str) -> dict[str, Any]:
        return self._loads(self._engine.stats(str(index_id)))

    def persist(self) -> None:
        self._engine.persist()

    def close(self) -> None:
        self._engine.close()

    @staticmethod
    def _dumps(payload: Any) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _loads(payload: str) -> dict[str, Any]:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise RuntimeError("native core returned a non-object JSON payload")
        return value

    @staticmethod
    def _loads_native_payload(payload: str) -> NativeCorePayload:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise RuntimeError("native core returned a non-object JSON payload")
        return NativeCorePayload(value, native_json=payload)
