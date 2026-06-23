"""Tests for the optional mem0 vector-store adapter."""

from __future__ import annotations

import pytest

DIM = 8


def _onehot(i: int) -> list[float]:
    vector = [0.0] * DIM
    vector[i] = 1.0
    return vector


def _store(tmp_path, **kwargs):
    pytest.importorskip("mem0")
    from lodedb.local.integrations.mem0 import LodeDBVectorStore

    return LodeDBVectorStore(
        path=str(tmp_path),
        collection_name="memories",
        embedding_model_dims=DIM,
        **kwargs,
    )


def test_mem0_vector_store_roundtrip_and_payload_boundary(tmp_path):
    store = _store(tmp_path)
    store.insert(
        vectors=[_onehot(0), _onehot(1)],
        ids=["m1", "m2"],
        payloads=[
            {
                "data": "Alice likes espresso",
                "text_lemmatized": "alice likes espresso",
                "user_id": "u1",
                "linked_memory_ids": ["root"],
            },
            {
                "data": "Bob likes tea",
                "text_lemmatized": "bob likes tea",
                "user_id": "u2",
            },
        ],
    )

    hits = store.search("alice", _onehot(0), top_k=2)
    assert hits[0].id == "m1"
    assert hits[0].payload["data"] == "Alice likes espresso"
    assert hits[0].payload["linked_memory_ids"] == ["root"]

    # Raw mem0 payload text is retained in the raw-text sidecar, not redacted metadata.
    assert store.client.get_document("m1")["metadata"] == {"user_id": "u1"}
    store.close()

    reopened = _store(tmp_path)
    assert reopened.get("m1").payload["data"] == "Alice likes espresso"
    reopened.close()


def test_mem0_vector_store_filters_batch_keyword_and_update(tmp_path):
    store = _store(tmp_path)
    store.insert(
        vectors=[_onehot(0), _onehot(1), _onehot(2)],
        ids=["a", "b", "c"],
        payloads=[
            {
                "data": "alpha E1234",
                "text_lemmatized": "alpha e1234",
                "user_id": "u1",
                "year": 2021,
            },
            {"data": "beta", "text_lemmatized": "beta", "user_id": "u1", "year": 2023},
            {"data": "gamma", "text_lemmatized": "gamma", "user_id": "u2", "year": 2024},
        ],
    )

    filtered = store.search(
        "beta",
        _onehot(1),
        top_k=10,
        filters={"AND": [{"user_id": "u1"}, {"year": {"gte": 2022}}]},
    )
    assert [hit.id for hit in filtered] == ["b"]

    batches = store.search_batch(["a", "c"], [_onehot(0), _onehot(2)], top_k=1)
    assert [[hit.id for hit in batch] for batch in batches] == [["a"], ["c"]]

    keyword = store.keyword_search("E1234", top_k=5, filters={"user_id": "u1"})
    assert keyword and keyword[0].id == "a"
    assert keyword[0].payload["data"] == "alpha E1234"

    store.update(
        "a",
        payload={
            "data": "alpha E1234",
            "user_id": "u1",
            "linked_memory_ids": ["m1", "m2"],
        },
    )
    assert store.get("a").payload["linked_memory_ids"] == ["m1", "m2"]
    assert store.search("alpha", _onehot(0), top_k=1)[0].id == "a"
    store.close()


def test_mem0_vector_store_updates_metadata_without_payload_retention(tmp_path):
    store = _store(tmp_path, store_payloads=False)
    store.insert(
        vectors=[_onehot(0)],
        ids=["a"],
        payloads=[{"data": "alpha", "user_id": "u1"}],
    )

    assert store.get("a").payload == {"user_id": "u1"}

    store.update("a", payload={"data": "alpha", "user_id": "u2", "tag": "updated"})
    assert store.get("a").payload == {"tag": "updated", "user_id": "u2"}
    assert store.search("alpha", _onehot(0), filters={"user_id": "u2"})[0].id == "a"
    store.close()


def test_mem0_provider_registration(tmp_path):
    pytest.importorskip("mem0")
    from mem0.utils.factory import VectorStoreFactory
    from mem0.vector_stores.configs import VectorStoreConfig

    from lodedb.local.integrations.mem0 import (
        LodeDBConfig,
        LodeDBVectorStore,
        register_mem0_provider,
    )

    register_mem0_provider()
    config = VectorStoreConfig(
        provider="lodedb",
        config={
            "path": str(tmp_path),
            "collection_name": "registered",
            "embedding_model_dims": DIM,
        },
    )
    assert isinstance(config.config, LodeDBConfig)
    created = VectorStoreFactory.create("lodedb", config.config)
    assert isinstance(created, LodeDBVectorStore)
    created.close()


def test_mem0_config_rejects_invalid_vector_dimensions(tmp_path):
    pytest.importorskip("mem0")
    from pydantic import ValidationError

    from lodedb.local.integrations.mem0 import LodeDBConfig, LodeDBVectorStore

    with pytest.raises(ValidationError, match="positive multiple of 8"):
        LodeDBConfig(embedding_model_dims=4)
    with pytest.raises(ValueError, match="positive multiple of 8"):
        LodeDBVectorStore(path=str(tmp_path), embedding_model_dims=4)


def test_mem0_requires_explicit_path():
    pytest.importorskip("mem0")
    from lodedb.local.integrations.mem0 import LodeDBVectorStore

    # No /tmp fallback: a path-less store must fail loudly rather than persist
    # memories somewhere ephemeral.
    with pytest.raises(ValueError, match="requires an explicit"):
        LodeDBVectorStore(embedding_model_dims=DIM)


def test_mem0_config_does_not_expose_store_payloads(tmp_path):
    pytest.importorskip("mem0")
    from lodedb.local.integrations.mem0 import LodeDBConfig

    # store_payloads is a direct-constructor option only; the mem0 factory path must
    # not be able to select the payload-less mode that mem0's reads cannot use.
    assert "store_payloads" not in LodeDBConfig.model_fields
    with pytest.raises(ValueError, match="Extra fields not allowed"):
        LodeDBConfig(path=str(tmp_path), store_payloads=False)


def test_mem0_memory_config_accepts_lodedb_provider(tmp_path):
    pytest.importorskip("mem0")
    from mem0.configs.base import MemoryConfig

    from lodedb.local.integrations.mem0 import LodeDBConfig, register_mem0_provider

    register_mem0_provider()
    config = MemoryConfig(
        **{
            "vector_store": {
                "provider": "lodedb",
                "config": {
                    "path": str(tmp_path),
                    "collection_name": "memory_from_config",
                    "embedding_model_dims": DIM,
                },
            },
            "history_db_path": str(tmp_path / "history.db"),
        }
    )

    assert isinstance(config.vector_store.config, LodeDBConfig)
    assert config.vector_store.config.collection_name == "memory_from_config"


def test_mem0_payload_with_none_scalar_inserts_and_filters(tmp_path):
    # mem0 routinely sets unused scope keys to None (e.g. agent_id/run_id). Those must
    # not crash insert, must drop out of scalar filter metadata, and must still round-trip
    # in the retained payload.
    store = _store(tmp_path)
    store.insert(
        vectors=[_onehot(0)],
        ids=["m1"],
        payloads=[{"data": "hi", "user_id": "u1", "agent_id": None, "run_id": None}],
    )

    assert store.client.get_document("m1")["metadata"] == {"user_id": "u1"}
    assert store.get("m1").payload["agent_id"] is None
    # "*" (field-present) must not match a dropped None key.
    assert store.search("hi", _onehot(0), filters={"agent_id": "*"}) == []
    assert [h.id for h in store.search("hi", _onehot(0), filters={"user_id": "u1"})] == ["m1"]
    store.close()


def test_mem0_metadata_only_update_survives_reopen(tmp_path):
    store = _store(tmp_path)
    store.insert(vectors=[_onehot(0)], ids=["m1"], payloads=[{"data": "x", "user_id": "u1"}])
    store.close()

    # Reopen, then a payload-only update before any search (cold serving index).
    reopened = _store(tmp_path)
    reopened.update("m1", payload={"data": "y", "user_id": "u2"})
    assert reopened.get("m1").payload == {"data": "y", "user_id": "u2"}
    assert [h.id for h in reopened.search("y", _onehot(0), filters={"user_id": "u2"})] == ["m1"]
    reopened.close()

    again = _store(tmp_path)
    assert again.get("m1").payload == {"data": "y", "user_id": "u2"}
    again.close()


def test_mem0_keyword_search_without_payload_retention(tmp_path):
    store = _store(tmp_path, store_payloads=False)
    store.insert(vectors=[_onehot(0)], ids=["a"], payloads=[{"data": "alpha", "user_id": "u1"}])
    assert store.keyword_search("alpha") == []
    store.close()


def test_mem0_collection_management(tmp_path):
    store = _store(tmp_path)
    store.insert(
        vectors=[_onehot(0), _onehot(1)],
        ids=["a", "b"],
        payloads=[{"data": "a", "user_id": "u1"}, {"data": "b", "user_id": "u1"}],
    )
    assert store.list_cols() == ["memories"]
    info = store.col_info()
    assert info["count"] == 2
    assert info["dimension"] == DIM
    assert info["distance"] == "cosine"
    # mem0's list() returns a one-element list whose [0] is the rows.
    rows = store.list()[0]
    assert {row.id for row in rows} == {"a", "b"}

    store.reset()
    assert store.col_info()["count"] == 0
    assert store.list()[0] == []
    store.close()


def test_mem0_filters_reject_raw_payload_fields(tmp_path):
    store = _store(tmp_path)
    store.insert(vectors=[_onehot(0)], ids=["a"], payloads=[{"data": "secret", "user_id": "u1"}])

    with pytest.raises(NotImplementedError, match="retained payload text"):
        store.search("secret", _onehot(0), filters={"data": "secret"})
    with pytest.raises(NotImplementedError, match="not supported"):
        store.search("secret", _onehot(0), filters={"user_id": {"contains": "u"}})
    store.close()
