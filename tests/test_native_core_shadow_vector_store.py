from __future__ import annotations

import pytest

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend


class FakeNativeAdapter:
    def __init__(self, handle: FakeNativeVectorEngine, *, available: bool = True) -> None:
        self._handle = handle
        self._available = available
        self.opened_readonly = False
        self.opened_writable = False

    @property
    def available(self) -> bool:
        return self._available

    @property
    def version(self) -> str:
        return "test-native-core"

    @property
    def abi_version(self) -> int:
        return 1

    def new_engine(self) -> FakeNativeVectorEngine:
        return self._handle

    def open_readonly_engine(self, *args, **kwargs) -> FakeNativeVectorEngine:
        self.opened_readonly = True
        return self._handle

    def open_engine(self, *args, **kwargs) -> FakeNativeVectorEngine:
        self.opened_writable = True
        return self._handle


class FakeNativeVectorEngine:
    def __init__(self, *, score_offset: float = 0.0) -> None:
        self.score_offset = score_offset
        self.created: tuple[str, int, int] | None = None
        self.documents: dict[str, dict] = {}
        self.query_calls = 0
        self.batch_query_calls = 0
        self.text_prepare_calls = 0
        self.text_apply_calls = 0
        self.text_query_calls = 0
        self.persist_calls = 0
        self.close_calls = 0

    def create_index(self, index_id: str, *, vector_dim: int, bit_width: int = 4) -> None:
        self.created = (index_id, vector_dim, bit_width)

    def create_index_with_options(self, options: dict) -> None:
        self.created = (options["index_id"], options["vector_dim"], options["bit_width"])

    def upsert_vectors(self, index_id: str, documents) -> dict:
        for document in documents:
            self.documents[document.document_id] = {
                "vector": list(document.vector),
                "metadata": dict(document.metadata),
            }
        return {
            "documents_upserted": len(tuple(documents)),
            "documents_deleted": 0,
            "chunks_upserted": len(tuple(documents)),
            "chunks_deleted": 0,
            "generation": 1,
        }

    def delete_documents(self, index_id: str, document_ids) -> dict:
        deleted = 0
        for document_id in document_ids:
            if self.documents.pop(str(document_id), None) is not None:
                deleted += 1
        return {
            "documents_upserted": 0,
            "documents_deleted": deleted,
            "chunks_upserted": 0,
            "chunks_deleted": deleted,
            "generation": 2,
        }

    def query_vector(self, index_id: str, vector, *, top_k: int, filter=None) -> dict:
        self.query_calls += 1
        metadata_filter = (filter or {}).get("metadata", {})
        hits = []
        for document_id, document in self.documents.items():
            if any(
                document["metadata"].get(key) != value
                for key, value in metadata_filter.items()
            ):
                continue
            score = sum(
                left * right for left, right in zip(vector, document["vector"], strict=True)
            )
            hits.append(
                {
                    "document_id": document_id,
                    "chunk_id": document_id,
                    "score": score + self.score_offset,
                    "metadata": document["metadata"],
                }
            )
        hits.sort(key=lambda hit: (-hit["score"], hit["document_id"]))
        return {"hits": hits[:top_k], "total_considered": len(hits)}

    def query_vectors_batch(self, index_id: str, vectors, *, top_k: int, filter=None) -> list[dict]:
        self.batch_query_calls += 1
        return [
            self.query_vector(index_id, vector, top_k=top_k, filter=filter)
            for vector in vectors
        ]

    def prepare_text_upsert(
        self,
        index_id: str,
        documents,
        *,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: int,
    ) -> dict:
        self.text_prepare_calls += 1
        plan_documents = []
        chunks_to_embed = []
        for document in documents:
            chunk_id = f"{document.document_id}:native:0000"
            chunk = {
                "chunk_id": chunk_id,
                "text": document.text,
                "tokens": [token.lower() for token in document.text.split()],
                "needs_embedding": True,
            }
            plan_documents.append(
                {
                    "document_id": document.document_id,
                    "metadata": dict(document.metadata),
                    "text": document.text if store_text else None,
                    "chunks": [chunk],
                }
            )
            chunks_to_embed.append(
                {
                    "document_id": document.document_id,
                    "chunk_id": chunk_id,
                    "text": document.text,
                }
            )
        return {
            "plan_id": self.text_prepare_calls,
            "index_id": index_id,
            "base_generation": 0,
            "documents": plan_documents,
            "chunks_to_embed": chunks_to_embed,
            "store_text": store_text,
            "index_text": index_text,
        }

    def apply_text_upsert(self, plan: dict, embeddings, *, embedding_time_ms: float) -> dict:
        self.text_apply_calls += 1
        embeddings = list(embeddings)
        for document, embedding in zip(plan["documents"], embeddings, strict=True):
            self.documents[document["document_id"]] = {
                "vector": list(embedding),
                "metadata": dict(document["metadata"]),
                "tokens": document["chunks"][0]["tokens"],
            }
        return {
            "embedded_chunks": len(embeddings),
            "reused_chunks": 0,
            "embedding_time_ms": embedding_time_ms,
        }

    def prepare_query_text(self, query: str, mode: str) -> dict:
        return {
            "query": query,
            "mode": mode,
            "query_tokens": [token.lower() for token in query.split()],
            "requires_embedding": mode in {"vector", "hybrid"},
        }

    def search_embedded_text(
        self,
        index_id: str,
        query_plan: dict,
        query_embedding,
        *,
        top_k: int,
        filter=None,
    ) -> dict:
        self.text_query_calls += 1
        if query_embedding is not None:
            return self.query_vector(index_id, query_embedding, top_k=top_k, filter=filter)
        query_terms = set(query_plan["query_tokens"])
        metadata_filter = (filter or {}).get("metadata", {})
        hits = []
        for document_id, document in self.documents.items():
            if any(
                document["metadata"].get(key) != value
                for key, value in metadata_filter.items()
            ):
                continue
            score = len(query_terms.intersection(document.get("tokens", ())))
            if score:
                hits.append(
                    {
                        "document_id": document_id,
                        "chunk_id": document_id,
                        "score": float(score),
                        "metadata": document["metadata"],
                    }
                )
        hits.sort(key=lambda hit: (-hit["score"], hit["document_id"]))
        return {"hits": hits[:top_k], "total_considered": len(self.documents)}

    def stats(self, index_id: str) -> dict:
        return {
            "document_count": len(self.documents),
            "chunk_count": len(self.documents),
            "native_core_enabled": True,
        }

    def persist(self) -> None:
        self.persist_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def _onehot(i: int, dim: int = 8) -> list[float]:
    vector = [0.0] * dim
    vector[i] = 1.0
    return vector


def test_open_vector_store_shadow_mode_matches_python_path(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine(score_offset=100)
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: FakeNativeAdapter(native),
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "shadow")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    db.add_vectors(_onehot(0), id="a", metadata={"topic": "ops"})
    db.add_vectors(_onehot(1), id="b", metadata={"topic": "ml"})

    hits = db.search_by_vector(_onehot(1), k=2)
    assert [hit.id for hit in hits] == ["b", "a"]
    assert hits[0].score < 10.0  # shadow mode keeps Python authoritative
    assert native.query_calls == 1
    assert [hit.id for hit in db.search_by_vector(_onehot(0), k=2, filter={"topic": "ops"})] == [
        "a"
    ]
    assert db.remove("a") is True
    assert db.stats()["document_count"] == 1
    assert db.stats()["native_core"]["covered"] is True


def test_text_shadow_mode_mirrors_prepare_apply_and_query(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine()
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: FakeNativeAdapter(native),
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "shadow")
    db = LodeDB(
        tmp_path,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )

    db.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    hits = db.search("Alpha", k=1, mode="lexical")

    assert [hit.id for hit in hits] == ["doc-alpha"]
    assert native.text_prepare_calls == 1
    assert native.text_apply_calls == 1
    assert native.text_query_calls == 1
    assert db.stats()["native_core"]["mode"] == "shadow"
    assert db.stats()["native_core"]["covered"] is True


def test_text_shadow_parity_mismatch_disables_native_text(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine()
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: FakeNativeAdapter(native),
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "shadow")
    db = LodeDB(
        tmp_path,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )

    db.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    native.documents["doc-other"] = {
        "vector": native.documents["doc-alpha"]["vector"],
        "metadata": {},
        "tokens": ["alpha"],
    }
    native.documents.pop("doc-alpha")

    hits = db.search("Alpha", k=1, mode="lexical")

    assert [hit.id for hit in hits] == ["doc-alpha"]
    assert db.stats()["native_core"]["covered"] is False
    assert db.stats()["native_core"]["fallback_reason"] == "native_core_text_parity_mismatch"


def test_text_shadow_write_persists_to_temp_native_handle(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine()
    adapter = FakeNativeAdapter(native)
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: adapter,
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "shadow")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "shadow")
    db = LodeDB(
        tmp_path,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )

    db.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    db.persist()
    stats = db.stats()["native_core"]

    assert adapter.opened_writable is True
    assert native.text_prepare_calls == 1
    assert native.text_apply_calls == 1
    assert native.persist_calls == 1
    assert stats["write_mode"] == "shadow"
    assert stats["covered"] is True
    assert stats["shadow_persist_count"] == 1
    assert stats["shadow_persist_verified"] is True


def test_open_vector_store_native_on_uses_native_vector_results(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine(score_offset=100)
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: FakeNativeAdapter(native),
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    db.add_vectors(_onehot(0), id="a", metadata={"topic": "ops"})
    db.add_vectors(_onehot(1), id="b", metadata={"topic": "ml"})

    hits = db.search_by_vector(_onehot(1), k=2)
    assert [hit.id for hit in hits] == ["b", "a"]
    assert hits[0].score > 100.0

    batches = db.search_many_by_vector([_onehot(0), _onehot(1)], k=1)
    assert [[hit.id for hit in batch] for batch in batches] == [["a"], ["b"]]
    assert batches[0][0].score > 100.0
    assert db.count() == 2
    assert db.stats()["native_core"]["enabled"] is True
    assert db.stats()["native_core"]["version"] == "test-native-core"
    assert db.stats()["native_core"]["abi_version"] == 1


def test_native_on_requires_available_extension(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine()
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: FakeNativeAdapter(native, available=False),
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")

    try:
        LodeDB.open_vector_store(tmp_path, vector_dim=8)
    except RuntimeError as exc:
        assert "lodedb._native_core" in str(exc)
    else:
        raise AssertionError("native-on open unexpectedly succeeded")


def test_default_native_on_falls_back_when_extension_unavailable(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine()
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: FakeNativeAdapter(native, available=False),
    )
    monkeypatch.delenv("LODEDB_NATIVE_CORE", raising=False)
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    db.add_vectors(_onehot(0), id="a")

    assert [hit.id for hit in db.search_by_vector(_onehot(0), k=1)] == ["a"]
    assert db.stats()["native_core"]["mode"] == "on"
    assert db.stats()["native_core"]["version"] == ""
    assert db.stats()["native_core"]["abi_version"] == 0
    assert db.stats()["native_core"]["enabled"] is False
    assert db.stats()["native_core"]["fallback_reason"] == "native_core_extension_unavailable"


def test_native_on_readonly_existing_store_uses_persistent_seed(tmp_path, monkeypatch) -> None:
    writer = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    writer.add_vectors(_onehot(0), id="a", metadata={"topic": "ops"})
    writer.close()

    native = FakeNativeVectorEngine(score_offset=100)
    native.documents["a"] = {"vector": _onehot(0), "metadata": {"topic": "ops"}}
    adapter = FakeNativeAdapter(native)
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: adapter,
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8, read_only=True)

    hits = db.search_by_vector(_onehot(0), k=1)
    assert [hit.id for hit in hits] == ["a"]
    assert hits[0].score > 100.0
    assert adapter.opened_readonly is True
    assert db.stats()["native_core"]["covered"] is True
    assert db.stats()["native_core"]["fallback_reason"] == ""


def test_native_write_on_fails_closed_until_cutover(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine()
    adapter = FakeNativeAdapter(native)
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: adapter,
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "shadow")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")

    with pytest.raises(RuntimeError, match="LODEDB_NATIVE_CORE_WRITE=on"):
        LodeDB.open_vector_store(tmp_path, vector_dim=8)

    assert adapter.opened_writable is False


def test_native_write_shadow_persists_to_temp_native_handle(tmp_path, monkeypatch) -> None:
    native = FakeNativeVectorEngine(score_offset=100)
    adapter = FakeNativeAdapter(native)
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: adapter,
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "shadow")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "shadow")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    db.add_vectors(_onehot(0), id="a", metadata={"topic": "ops"})

    db.persist()
    stats = db.stats()["native_core"]

    assert adapter.opened_writable is True
    assert native.persist_calls == 1
    assert stats["write_mode"] == "shadow"
    assert stats["covered"] is True
    assert stats["shadow_persist_count"] == 1
    assert stats["shadow_persist_verified"] is True
    assert [hit.id for hit in db.search_by_vector(_onehot(0), k=1)] == ["a"]


def test_native_on_existing_store_falls_back_to_python_until_seeded(tmp_path, monkeypatch) -> None:
    writer = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    writer.add_vectors(_onehot(0), id="a")
    writer.close()

    native = FakeNativeVectorEngine(score_offset=100)
    monkeypatch.setattr(
        "lodedb.local.db.NativeCoreAdapter",
        lambda: FakeNativeAdapter(native),
    )
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)

    hits = db.search_by_vector(_onehot(0), k=1)
    assert [hit.id for hit in hits] == ["a"]
    assert hits[0].score < 10.0
    assert db.stats()["native_core"]["covered"] is False
    assert db.stats()["native_core"]["fallback_reason"] == (
        "native_core_existing_store_seed_unavailable"
    )
