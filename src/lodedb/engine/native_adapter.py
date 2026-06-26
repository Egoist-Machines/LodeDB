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
from dataclasses import dataclass
from typing import Any, Protocol

from lodedb.engine.core import EngineDocument, EngineQuery, EngineResponse, EngineVectorDocument


class NativeCoreModule(Protocol):
    """Subset of the hidden native module used by the adapter."""

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
