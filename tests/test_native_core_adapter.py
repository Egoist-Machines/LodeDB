from __future__ import annotations

import json

import pytest

from lodedb.engine.core import EngineDocument, EngineQuery, EngineVectorDocument
from lodedb.engine.native_adapter import NativeCoreAdapter, NativeCoreError


class FakeNativeModule:
    @property
    def CoreEngine(self) -> type[FakeCoreEngine]:
        return FakeCoreEngine

    def round_trip_core_json(self, type_name: str, json_payload: str) -> str:
        assert type_name in {"CoreDocument", "CoreVectorDocument", "CoreQuery"}
        return json_payload

    def native_core_version(self) -> str:
        return "test-native-core"

    def native_core_abi_version(self) -> int:
        return 1


class FakeCoreEngine:
    def __init__(self) -> None:
        self.index: tuple[str, int, int] | None = None
        self.documents: list[dict] = []
        self.open_options: dict | None = None
        self.last_plan_json: str | None = None
        self.applied_plan_json: str | None = None

    @staticmethod
    def open(options_json: str) -> FakeCoreEngine:
        engine = FakeCoreEngine()
        engine.open_options = json.loads(options_json)
        return engine

    @staticmethod
    def open_readonly(path: str, options_json: str) -> FakeCoreEngine:
        engine = FakeCoreEngine()
        engine.open_options = json.loads(options_json)
        engine.open_options["path_arg"] = path
        return engine

    def create_index(self, index_id: str, vector_dim: int, bit_width: int) -> None:
        self.index = (index_id, vector_dim, bit_width)

    def create_index_with_options(self, options_json: str) -> None:
        options = json.loads(options_json)
        self.index = (options["index_id"], options["vector_dim"], options["bit_width"])

    def upsert_vectors(self, index_id: str, documents_json: str) -> str:
        self.documents = json.loads(documents_json)
        return json.dumps(
            {
                "documents_upserted": len(self.documents),
                "documents_deleted": 0,
                "chunks_upserted": len(self.documents),
                "chunks_deleted": 0,
                "generation": 1,
            }
        )

    def delete_documents(self, index_id: str, document_ids_json: str) -> str:
        ids = set(json.loads(document_ids_json))
        self.documents = [
            document for document in self.documents if document["document_id"] not in ids
        ]
        return json.dumps(
            {
                "documents_upserted": 0,
                "documents_deleted": len(ids),
                "chunks_upserted": 0,
                "chunks_deleted": len(ids),
                "generation": 2,
            }
        )

    def query_vector(
        self,
        index_id: str,
        query_vector_json: str,
        top_k: int,
        filter_json: str | None,
    ) -> str:
        query = json.loads(query_vector_json)
        metadata_filter = json.loads(filter_json or "{}").get("metadata", {})
        rows = [
            document
            for document in self.documents
            if all(document["metadata"].get(key) == value for key, value in metadata_filter.items())
        ]
        hits = [
            {
                "document_id": document["document_id"],
                "chunk_id": document["document_id"],
                "score": sum(
                    left * right for left, right in zip(query, document["vector"], strict=True)
                ),
                "metadata": document["metadata"],
            }
            for document in rows
        ]
        hits.sort(key=lambda hit: (-hit["score"], hit["document_id"]))
        return json.dumps({"hits": hits[:top_k], "total_considered": len(rows)})

    def prepare_text_upsert(
        self,
        index_id: str,
        documents_json: str,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: int,
    ) -> str:
        documents = json.loads(documents_json)
        document = documents[0]
        self.last_plan_json = json.dumps(
            {
                "plan_id": 0,
                "index_id": index_id,
                "base_generation": 0,
                "documents": [
                    {
                        "document_id": document["document_id"],
                        "metadata": document["metadata"],
                        "text": document["text"] if store_text else None,
                        "chunks": [
                            {
                                "chunk_id": f"{document['document_id']}:chunk:0000",
                                "text": document["text"],
                                "tokens": ["alpha"],
                                "needs_embedding": True,
                            }
                        ],
                    }
                ],
                "chunks_to_embed": [
                    {
                        "document_id": document["document_id"],
                        "chunk_id": f"{document['document_id']}:chunk:0000",
                        "text": document["text"],
                    }
                ],
                "store_text": store_text,
                "index_text": index_text,
            }
        )
        return self.last_plan_json

    def apply_text_upsert(
        self,
        plan_json: str,
        embeddings_json: str,
        embedding_time_ms: float,
    ) -> str:
        self.applied_plan_json = plan_json
        plan = json.loads(plan_json)
        embeddings = json.loads(embeddings_json)
        return json.dumps(
            {
                "mutation": {
                    "documents_upserted": len(plan["documents"]),
                    "documents_deleted": 0,
                    "chunks_upserted": len(embeddings),
                    "chunks_deleted": 0,
                    "generation": 1,
                },
                "embedded_chunks": len(embeddings),
                "reused_chunks": 0,
                "embedding_time_ms": embedding_time_ms,
            }
        )

    def prepare_query_text(self, query: str, mode: str) -> str:
        return json.dumps(
            {
                "query": query,
                "mode": mode,
                "query_tokens": [query.lower()],
                "requires_embedding": mode in {"vector", "hybrid"},
            }
        )

    def search_embedded_text(
        self,
        index_id: str,
        query_plan_json: str,
        query_embedding_json: str | None,
        top_k: int,
        filter_json: str | None,
    ) -> str:
        plan = json.loads(query_plan_json)
        return json.dumps(
            {
                "hits": [
                    {
                        "document_id": "doc-alpha",
                        "chunk_id": "doc-alpha:chunk:0000",
                        "score": 1.0,
                        "metadata": {"query": plan["query"]},
                    }
                ],
                "total_considered": 1,
            }
        )

    def stats(self, index_id: str) -> str:
        return json.dumps({"document_count": len(self.documents), "native_core_enabled": True})


def test_adapter_maps_engine_document_to_native_payload() -> None:
    adapter = NativeCoreAdapter(FakeNativeModule())
    assert adapter.version == "test-native-core"
    assert adapter.abi_version == 1
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


def test_adapter_wraps_native_engine_vector_flow() -> None:
    engine = NativeCoreAdapter(FakeNativeModule()).new_engine()
    engine.create_index_with_options(
        NativeCoreAdapter.index_create_options_payload(
            index_id="default",
            index_key="storage-key",
            client_id_hash="client-hash",
            name="lodedb-local",
            model="external",
            provider="external",
            task="vector-only",
            route_profile="vector-only",
            storage_profile="turbovec_direct",
            vector_dim=2,
            bit_width=4,
        )
    )
    mutation = engine.upsert_vectors(
        "default",
        (
            EngineVectorDocument("a", (1, 0), metadata={"topic": "ops"}, text=None),
            EngineVectorDocument("b", (0, 1), metadata={"topic": "ml"}, text=None),
        ),
    )
    assert mutation["documents_upserted"] == 2

    hits = engine.query_vector(
        "default",
        (0, 1),
        top_k=1,
        filter={"metadata": {"topic": "ml"}},
    )
    assert hits["hits"][0]["document_id"] == "b"
    assert engine.stats("default")["document_count"] == 2


def test_adapter_wraps_persistent_native_engine_open(tmp_path) -> None:
    engine = NativeCoreAdapter(FakeNativeModule()).open_engine(
        path=tmp_path,
        read_only=False,
        durability="relaxed",
        commit_mode="generation",
        store_text=False,
        index_text=False,
    )

    assert engine._engine.open_options == {
        "path": str(tmp_path),
        "read_only": False,
        "durability": "relaxed",
        "commit_mode": "generation",
        "store_text": False,
        "index_text": False,
    }


def test_adapter_wraps_readonly_native_engine_open(tmp_path) -> None:
    engine = NativeCoreAdapter(FakeNativeModule()).open_readonly_engine(
        tmp_path,
        durability="fsync",
        commit_mode="generation",
        store_text=True,
        index_text=False,
    )

    assert engine._engine.open_options == {
        "path": str(tmp_path),
        "path_arg": str(tmp_path),
        "read_only": True,
        "durability": "fsync",
        "commit_mode": "generation",
        "store_text": True,
        "index_text": False,
    }


def test_adapter_wraps_native_engine_text_prepare_apply_flow() -> None:
    engine = NativeCoreAdapter(FakeNativeModule()).new_engine()
    plan = engine.prepare_text_upsert(
        "text",
        (EngineDocument("doc-alpha", "Alpha body", metadata={"topic": "ops"}),),
        store_text=True,
        index_text=True,
        chunk_character_limit=900,
    )
    assert plan["chunks_to_embed"][0]["text"] == "Alpha body"

    applied = engine.apply_text_upsert(plan, ([1.0, 0.0],), embedding_time_ms=2.5)
    assert applied["embedded_chunks"] == 1
    assert applied["embedding_time_ms"] == 2.5
    assert engine._engine.applied_plan_json == engine._engine.last_plan_json

    query_plan = engine.prepare_query_text("Alpha", "vector")
    hits = engine.search_embedded_text("text", query_plan, (1.0, 0.0), top_k=1)
    assert hits["hits"][0]["document_id"] == "doc-alpha"


def test_adapter_lazily_reports_missing_native_module(monkeypatch) -> None:
    def missing_module(name: str):
        assert name == "lodedb._native_core"
        raise ImportError(name)

    monkeypatch.setattr("importlib.import_module", missing_module)
    adapter = NativeCoreAdapter()
    assert adapter.available is False
    with pytest.raises(RuntimeError, match="native core extension"):
        adapter.round_trip("CoreDocument", {})
