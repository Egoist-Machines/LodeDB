from __future__ import annotations

import importlib
import json

import pytest


native_core = pytest.importorskip("lodedb._native_core")


def _onehot(axis: int, dim: int = 8) -> list[float]:
    vector = [0.0] * dim
    vector[axis] = 1.0
    return vector


def _loads(payload: str) -> dict:
    return json.loads(payload)


def test_native_core_extension_executes_vector_store_flow() -> None:
    engine = native_core.CoreEngine()
    engine.create_index("default", 8, 4)
    mutation = _loads(
        engine.upsert_vectors(
            "default",
            json.dumps(
                [
                    {
                        "document_id": "a",
                        "vector": _onehot(0),
                        "metadata": {"topic": "ops"},
                        "text": None,
                    },
                    {
                        "document_id": "b",
                        "vector": _onehot(1),
                        "metadata": {"topic": "ml"},
                        "text": None,
                    },
                ]
            ),
        )
    )
    assert mutation["documents_upserted"] == 2

    hits = _loads(
        engine.query_vector(
            "default",
            json.dumps(_onehot(1)),
            2,
            json.dumps({"metadata": {"topic": "ml"}}),
        )
    )
    assert hits["hits"][0]["document_id"] == "b"
    assert hits["hits"][0]["metadata"] == {"topic": "ml"}

    stats = _loads(engine.stats("default"))
    assert stats["document_count"] == 2
    assert stats["raw_payload_text_present"] is False


def test_native_core_extension_executes_text_prepare_apply_flow() -> None:
    engine = native_core.CoreEngine()
    engine.create_index("text", 8, 4)
    plan = _loads(
        engine.prepare_text_upsert(
            "text",
            json.dumps(
                [
                    {
                        "document_id": "doc-alpha",
                        "text": "Alpha launch notes mention error code E-1001.",
                        "metadata": {"topic": "ops"},
                    }
                ]
            ),
            True,
            True,
            900,
        )
    )
    assert plan["chunks_to_embed"][0]["document_id"] == "doc-alpha"
    assert plan["chunks_to_embed"][0]["text"] == "Alpha launch notes mention error code E-1001."

    applied = _loads(engine.apply_text_upsert(json.dumps(plan), json.dumps([_onehot(3)]), 1.25))
    assert applied["embedded_chunks"] == 1
    assert applied["embedding_time_ms"] == 1.25

    query_plan = _loads(engine.prepare_query_text("E-1001", "vector"))
    assert query_plan["requires_embedding"] is True
    hits = _loads(
        engine.search_embedded_text(
            "text",
            json.dumps(query_plan),
            json.dumps(_onehot(3)),
            1,
            json.dumps({"metadata": {"topic": "ops"}}),
        )
    )
    assert hits["hits"][0]["document_id"] == "doc-alpha"


def test_native_core_adapter_can_discover_extension_when_installed(monkeypatch) -> None:
    module = importlib.import_module("lodedb._native_core")
    monkeypatch.setattr("importlib.import_module", lambda name: module)

    from lodedb.engine.native_adapter import NativeCoreAdapter

    assert NativeCoreAdapter().available is True
