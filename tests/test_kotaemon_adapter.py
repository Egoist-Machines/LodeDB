"""Tests for the kotaemon vector-store adapter.

The adapter is duck-typed and dependency-free (it imports neither kotaemon nor
llama-index), so these tests run in the base suite with no skip guard. They
mirror the shapes of kotaemon's own ``test_vectorstore.py`` (add / add-from-docs
/ delete / query / persist-reopen / drop) — including its 3-dimensional toy
vectors, which exercise the zero-padding path — plus the retrieval-pipeline
behaviors kotaemon relies on at runtime: the ``doc_ids`` chunk scope and
duck-typed LlamaIndex ``MetadataFilters`` push-down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lodedb.local.integrations.kotaemon import LodeDBVectorStore

EMBEDDINGS = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
METADATAS = [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]
IDS = ["a", "b", "c"]


@dataclass
class _DocWithEmbedding:
    """Duck-typed stand-in for kotaemon's DocumentWithEmbedding."""

    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str | None = None
    text: str = ""


class _LIFilter:
    """Duck-typed stand-in for llama_index's MetadataFilter."""

    def __init__(self, key: str, value: Any, operator: str = "==") -> None:
        self.key = key
        self.value = value
        self.operator = operator


class _LIFilters:
    """Duck-typed stand-in for llama_index's MetadataFilters."""

    def __init__(self, filters: list[Any], condition: str = "and") -> None:
        self.filters = filters
        self.condition = condition


def _store(tmp_path, **kwargs) -> LodeDBVectorStore:
    """Opens a test collection under the pytest tmp dir."""

    return LodeDBVectorStore(path=str(tmp_path), collection_name="test", **kwargs)


def test_add_returns_ids_and_counts(tmp_path):
    db = _store(tmp_path)
    assert db.count() == 0
    output = db.add(embeddings=EMBEDDINGS[:2], metadatas=METADATAS[:2], ids=IDS[:2])
    assert output == IDS[:2]
    assert db.count() == 2


def test_add_from_docs_generates_ids(tmp_path):
    db = _store(tmp_path)
    documents = [
        _DocWithEmbedding(embedding=embedding, metadata=metadata)
        for embedding, metadata in zip(EMBEDDINGS[:2], METADATAS[:2], strict=True)
    ]
    output = db.add(documents)
    assert len(output) == 2
    assert all(isinstance(value, str) and value for value in output)
    assert db.count() == 2


def test_add_from_docs_uses_doc_ids(tmp_path):
    db = _store(tmp_path)
    documents = [
        _DocWithEmbedding(embedding=embedding, doc_id=doc_id)
        for embedding, doc_id in zip(EMBEDDINGS, IDS, strict=True)
    ]
    assert db.add(documents) == IDS


def test_add_empty_is_noop(tmp_path):
    db = _store(tmp_path)
    assert db.add([]) == []
    assert db.count() == 0


def test_add_upserts_on_same_id(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, ids=IDS)
    db.add(embeddings=[EMBEDDINGS[0]], ids=["c"])
    assert db.count() == 3
    _, _, out_ids = db.query(embedding=EMBEDDINGS[0], top_k=2)
    assert set(out_ids) == {"a", "c"}


def test_delete(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, metadatas=METADATAS, ids=IDS)
    assert db.count() == 3
    db.delete(ids=["a", "b"])
    assert db.count() == 1
    db.delete(ids=["c"])
    assert db.count() == 0
    db.delete(ids=["missing"])  # absent ids are ignored, matching Chroma


def test_query_ranks_by_cosine(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, metadatas=METADATAS, ids=IDS)

    embeddings_out, sim, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=1)
    assert out_ids == ["a"]
    # 4-bit quantized cosine wobbles slightly on toy vectors; the adapter clamps
    # the >1.0 artifact so kotaemon's own `sim - 1.0 < 1e-6` assertion holds.
    assert sim[0] == pytest.approx(1.0, abs=0.05)
    assert sim[0] <= 1.0
    assert embeddings_out == [None]

    # kotaemon's Chroma test expects "b" here because Chroma defaults to L2
    # distance; by cosine the true ranking is c > b > a (angle, not proximity),
    # and LodeDB returns exactly that. The two metrics agree on the normalized
    # embeddings real models produce.
    _, sim, out_ids = db.query(embedding=[0.42, 0.52, 0.53], top_k=3)
    assert out_ids == ["c", "b", "a"]
    assert sim == sorted(sim, reverse=True)


def test_query_empty_store_returns_nothing(tmp_path):
    db = _store(tmp_path)
    assert db.query(embedding=[0.1, 0.2, 0.3], top_k=5) == ([], [], [])


def test_query_scope_allowlists(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, metadatas=METADATAS, ids=IDS)

    # doc_ids is how kotaemon's retrieval passes its chunk-id scope.
    _, _, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=3, doc_ids=["b", "c"])
    assert "a" not in out_ids and set(out_ids) == {"b", "c"}

    # ids is the node-id allowlist from the BaseVectorStore signature.
    _, _, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=3, ids=["a"])
    assert out_ids == ["a"]

    # both given -> intersection; empty scope -> no hits, not all hits.
    _, _, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=3, ids=["a"], doc_ids=["a", "b"])
    assert out_ids == ["a"]
    assert db.query(embedding=[0.1, 0.2, 0.3], top_k=3, doc_ids=[]) == ([], [], [])


def test_query_translates_metadata_filters(tmp_path):
    db = _store(tmp_path)
    db.add(
        embeddings=EMBEDDINGS,
        metadatas=[{"file_id": "f1"}, {"file_id": "f1"}, {"file_id": "f2"}],
        ids=IDS,
    )

    # The shape kotaemon's DocumentRetrievalPipeline sends: file_id IN [...].
    filters = _LIFilters([_LIFilter("file_id", ["f2"], operator="in")], condition="or")
    _, _, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=3, filters=filters)
    assert out_ids == ["c"]

    # A plain dict passes through as a native LodeDB filter.
    _, _, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=3, filters={"file_id": "f1"})
    assert set(out_ids) == {"a", "b"}

    # Filters compose with the chunk scope.
    _, _, out_ids = db.query(
        embedding=[0.1, 0.2, 0.3], top_k=3, doc_ids=["a", "c"], filters={"file_id": "f1"}
    )
    assert out_ids == ["a"]


def test_query_filter_operator_translation(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, metadatas=METADATAS, ids=IDS)

    filters = _LIFilters(
        [_LIFilter("a", 1, operator=">"), _LIFilter("a", 5, operator="<")],
        condition="and",
    )
    _, _, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=3, filters=filters)
    assert out_ids == ["b"]

    filters = _LIFilters(
        [_LIFilter("a", 1, operator="=="), _LIFilter("a", 5, operator="==")],
        condition="or",
    )
    _, _, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=3, filters=filters)
    assert set(out_ids) == {"a", "c"}

    unsupported = _LIFilters([_LIFilter("a", "x", operator="text_match")])
    with pytest.raises(NotImplementedError):
        db.query(embedding=[0.1, 0.2, 0.3], top_k=1, filters=unsupported)


def test_query_ignores_chroma_parity_hints_but_rejects_unknown(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, ids=IDS)

    # kotaemon forwards these when MMR is enabled; Chroma ignores them too.
    _, _, out_ids = db.query(embedding=[0.1, 0.2, 0.3], top_k=1, mode="mmr", mmr_threshold=0.5)
    assert out_ids == ["a"]

    with pytest.raises(TypeError, match="unexpected keyword"):
        db.query(embedding=[0.1, 0.2, 0.3], top_k=1, no_such_option=True)


def test_persist_and_reopen(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, metadatas=METADATAS, ids=IDS)
    db.close()

    db2 = _store(tmp_path)
    assert db2.count() == 3
    _, sim, out_ids = db2.query(embedding=[0.1, 0.2, 0.3], top_k=1)
    assert out_ids == ["a"]
    assert sim[0] == pytest.approx(1.0, abs=0.05)
    db2.close()


def test_drop_removes_collection(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, ids=IDS)
    collection_dir = tmp_path / "test"
    assert collection_dir.exists()
    db.drop()
    assert not collection_dir.exists()

    # Re-instantiating after drop starts an empty collection (Chroma parity).
    db2 = _store(tmp_path)
    assert db2.count() == 0
    # And the collection is usable again, at a fresh dimension if need be.
    db2.add(embeddings=[[1.0] * 8], ids=["x"])
    assert db2.count() == 1
    db2.close()


def test_drop_without_data_is_noop(tmp_path):
    db = _store(tmp_path)
    db.drop()
    assert db.count() == 0


def test_drop_never_deletes_foreign_directories(tmp_path):
    foreign = tmp_path / "test"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("precious", encoding="utf-8")
    db = _store(tmp_path)
    db.drop()
    assert (foreign / "keep.txt").read_text(encoding="utf-8") == "precious"


def test_dimension_mismatch_raises(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, ids=IDS)
    with pytest.raises(ValueError, match="3-dim"):
        db.add(embeddings=[[0.1] * 8], ids=["z"])
    with pytest.raises(ValueError, match="3-dim"):
        db.query(embedding=[0.1] * 8, top_k=1)


def test_multiple_of_8_dims_are_not_padded(tmp_path):
    db = _store(tmp_path)
    vectors = [[float(i) for i in range(8)], [float(8 - i) for i in range(8)]]
    db.add(embeddings=vectors, ids=["a", "b"])
    _, sim, out_ids = db.query(embedding=[float(i) for i in range(8)], top_k=1)
    assert out_ids == ["a"]
    assert sim[0] == pytest.approx(1.0, abs=0.05)
    assert sim[0] <= 1.0


def test_collections_are_isolated(tmp_path):
    db1 = LodeDBVectorStore(path=str(tmp_path), collection_name="one")
    db2 = LodeDBVectorStore(path=str(tmp_path), collection_name="two")
    db1.add(embeddings=[EMBEDDINGS[0]], ids=["a"])
    db2.add(embeddings=[[1.0] * 16], ids=["b"])  # different dim per collection
    assert db1.count() == 1
    assert db2.count() == 1
    _, _, out_ids = db2.query(embedding=[1.0] * 16, top_k=5)
    assert out_ids == ["b"]


def test_requires_explicit_path():
    with pytest.raises(ValueError, match="path"):
        LodeDBVectorStore()


def test_rejects_unknown_config_keys(tmp_path):
    with pytest.raises(TypeError, match="unexpected config"):
        LodeDBVectorStore(path=str(tmp_path), no_such_key=1)


def test_persist_flow_roundtrip(tmp_path):
    db = _store(tmp_path)
    db.add(embeddings=EMBEDDINGS, ids=IDS)
    db.close()
    # theflow re-creates the store from __persist_flow__ kwargs.
    db2 = LodeDBVectorStore(**db.__persist_flow__())
    assert db2.count() == 3
    db2.close()


def test_mismatched_lengths_raise(tmp_path):
    db = _store(tmp_path)
    with pytest.raises(ValueError, match="ids"):
        db.add(embeddings=EMBEDDINGS, ids=["only-one"])
    with pytest.raises(ValueError, match="metadatas"):
        db.add(embeddings=EMBEDDINGS, metadatas=[{}])


def test_store_text_retains_doc_text(tmp_path):
    db = _store(tmp_path, store_text=True)
    documents = [
        _DocWithEmbedding(embedding=EMBEDDINGS[0], doc_id="a", text="hello chunk"),
    ]
    db.add(documents)
    assert db._db is not None
    assert db._db.get("a") == "hello chunk"


def test_default_store_keeps_no_text(tmp_path):
    db = _store(tmp_path)
    documents = [
        _DocWithEmbedding(embedding=EMBEDDINGS[0], doc_id="a", text="hello chunk"),
    ]
    db.add(documents)
    assert db._db is not None
    # store_text=False keeps no text on disk at all; LodeDB refuses the read.
    with pytest.raises(ValueError, match="store_text"):
        db._db.get("a")
