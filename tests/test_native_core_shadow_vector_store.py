from __future__ import annotations

from lodedb import LodeDB


class FakeNativeAdapter:
    def __init__(self, handle: FakeNativeVectorEngine, *, available: bool = True) -> None:
        self._handle = handle
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def new_engine(self) -> FakeNativeVectorEngine:
        return self._handle


class FakeNativeVectorEngine:
    def __init__(self, *, score_offset: float = 0.0) -> None:
        self.score_offset = score_offset
        self.created: tuple[str, int, int] | None = None
        self.documents: dict[str, dict] = {}
        self.query_calls = 0
        self.batch_query_calls = 0

    def create_index(self, index_id: str, *, vector_dim: int, bit_width: int = 4) -> None:
        self.created = (index_id, vector_dim, bit_width)

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

    def stats(self, index_id: str) -> dict:
        return {
            "document_count": len(self.documents),
            "native_core_enabled": True,
        }


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
    assert db.stats()["native_core"]["enabled"] is False
    assert db.stats()["native_core"]["fallback_reason"] == "native_core_extension_unavailable"


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
