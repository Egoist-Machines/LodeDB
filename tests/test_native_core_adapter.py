from __future__ import annotations

import json

import pytest

from lodedb.engine.core import EngineDocument, EngineQuery, EngineVectorDocument
from lodedb.engine.native_adapter import NativeCoreAdapter, NativeCoreError


class FakeNativeModule:
    def round_trip_core_json(self, type_name: str, json_payload: str) -> str:
        assert type_name in {"CoreDocument", "CoreVectorDocument", "CoreQuery"}
        return json_payload


def test_adapter_maps_engine_document_to_native_payload() -> None:
    adapter = NativeCoreAdapter(FakeNativeModule())
    payload = adapter.round_trip(
        "CoreDocument",
        adapter.document_payload(
            EngineDocument("doc-1", "hello", metadata={"rank": 1, "topic": "ops"})  # type: ignore[dict-item]
        ),
    )
    assert payload == {
        "document_id": "doc-1",
        "text": "hello",
        "metadata": {"rank": "1", "topic": "ops"},
    }
    assert json.loads(adapter.document_json(EngineDocument("d", "t"))) == {
        "document_id": "d",
        "text": "t",
        "metadata": {},
    }


def test_adapter_maps_vector_document_and_query() -> None:
    adapter = NativeCoreAdapter(FakeNativeModule())
    vector_payload = adapter.round_trip(
        "CoreVectorDocument",
        adapter.vector_document_payload(
            EngineVectorDocument("vec", (1, 0), metadata={"kind": "unit"}, text=None)
        ),
    )
    assert vector_payload == {
        "document_id": "vec",
        "vector": [1.0, 0.0],
        "metadata": {"kind": "unit"},
        "text": None,
    }

    query_payload = adapter.round_trip(
        "CoreQuery",
        adapter.query_payload(
            EngineQuery(
                "needle",
                top_k=3,
                filter={"topic": "ops"},
                include=("metadata",),
                mode="vector",
                embedding=(0.5, 0.25),
            )
        ),
    )
    assert query_payload["embedding"] == [0.5, 0.25]
    assert query_payload["include"] == ["metadata"]


def test_adapter_maps_native_errors_to_engine_response() -> None:
    response = NativeCoreAdapter.response_from_error(
        NativeCoreError("CORRUPT_STORE", "manifest failed checksum")
    )
    assert response.status_code == 500
    assert response.body == {
        "status": "error",
        "error": "manifest failed checksum",
        "native_core_error": "CORRUPT_STORE",
    }


def test_adapter_lazily_reports_missing_native_module(monkeypatch) -> None:
    def missing_module(name: str):
        assert name == "lodedb._native_core"
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", missing_module)
    adapter = NativeCoreAdapter()
    assert adapter.available is False
    with pytest.raises(RuntimeError, match="native core extension"):
        adapter.round_trip("CoreDocument", {})
