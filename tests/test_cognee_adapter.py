"""Tests for the optional cognee vector-database adapter.

Skipped entirely unless cognee is installed (``pip install 'lodedb[cognee]'``). The
tests use a deterministic stub ``EmbeddingEngine`` that maps each distinct text to a
one-hot vector, so exact-match ranking is orthogonal and robust to TurboVec's
quantization; no LLM or network is involved.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

pytest.importorskip("cognee")

from cognee.infrastructure.databases.exceptions import MissingQueryParameterError  # noqa: E402
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult  # noqa: E402
from cognee.infrastructure.engine import DataPoint  # noqa: E402

from lodedb.local.integrations.cognee import (  # noqa: E402
    CogneeLodeDBAdapter,
    register_cognee_adapter,
)

DIM = 32


class _StubEmbeddingEngine:
    """Deterministic text -> one-hot embedding (orthogonal, quantization-robust)."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self._index: dict[str, int] = {}

    async def embed_text(self, text: list[str]) -> list[list[float]]:
        return [self._vector(value) for value in text]

    def _vector(self, text: str) -> list[float]:
        if text not in self._index:
            slot = len(self._index)
            assert slot < self.dim, "stub embedding ran out of dimensions; raise DIM"
            self._index[text] = slot
        vector = [0.0] * self.dim
        vector[self._index[text]] = 1.0
        return vector

    def get_vector_size(self) -> int:
        return self.dim

    def get_batch_size(self) -> int:
        return 32


class _Doc(DataPoint):
    """A minimal cognee DataPoint whose ``text`` field is the embeddable field."""

    text: str
    metadata: dict = {"index_fields": ["text"]}


def _adapter(tmp_path, engine: _StubEmbeddingEngine | None = None) -> CogneeLodeDBAdapter:
    return CogneeLodeDBAdapter(
        url=str(tmp_path),
        embedding_engine=engine or _StubEmbeddingEngine(),
    )


def test_create_search_retrieve_roundtrip_and_reopen(tmp_path):
    async def body():
        engine = _StubEmbeddingEngine()
        adapter = _adapter(tmp_path, engine)

        docs = [
            _Doc(text="alice likes espresso"),
            _Doc(text="bob likes tea"),
            _Doc(text="carol likes matcha"),
        ]
        await adapter.create_collection("Doc_text")
        assert await adapter.has_collection("Doc_text") is True
        await adapter.create_data_points("Doc_text", docs)

        hits = await adapter.search("Doc_text", query_text="alice likes espresso", limit=3)
        assert [type(hit) for hit in hits] == [ScoredResult, ScoredResult, ScoredResult]
        # Exact-text match ranks first, at (near-)zero cosine distance.
        assert hits[0].id == docs[0].id
        assert isinstance(hits[0].id, UUID)
        assert hits[0].score < 1e-3
        # Scores are cosine distances (lower is better), ascending.
        assert hits[0].score <= hits[1].score <= hits[2].score
        # Without include_payload, no payload is materialized.
        assert hits[0].payload is None

        got = await adapter.retrieve("Doc_text", [str(docs[0].id), str(docs[1].id)])
        assert {result.id for result in got} == {docs[0].id, docs[1].id}
        by_id = {result.id: result for result in got}
        assert by_id[docs[0].id].payload["text"] == "alice likes espresso"
        assert by_id[docs[0].id].score == 0
        adapter.close()

        # A fresh adapter opens the existing on-disk collection and still reads it.
        reopened = _adapter(tmp_path, _StubEmbeddingEngine())
        assert await reopened.has_collection("Doc_text") is True
        again = await reopened.retrieve("Doc_text", [str(docs[2].id)])
        assert again[0].payload["text"] == "carol likes matcha"
        reopened.close()

    asyncio.run(body())


def test_search_include_payload_and_missing_collection(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        # brute-force fallback queries collections that may not exist -> [].
        assert await adapter.search("NoSuch_text", query_text="x", limit=5) == []
        assert await adapter.has_collection("NoSuch_text") is False

        docs = [_Doc(text="quantum computing"), _Doc(text="garden tools")]
        await adapter.create_data_points("Doc_text", docs)

        hits = await adapter.search(
            "Doc_text", query_text="quantum computing", limit=5, include_payload=True
        )
        assert hits[0].id == docs[0].id
        assert hits[0].payload["text"] == "quantum computing"
        # A missing query (no text, no vector) is an error.
        with pytest.raises(MissingQueryParameterError):
            await adapter.search("Doc_text", limit=5)
        adapter.close()

    asyncio.run(body())


def test_node_name_belongs_to_set_filtering(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        a = _Doc(text="red big apple", belongs_to_set=["red", "big"])
        b = _Doc(text="red small cherry", belongs_to_set=["red"])
        c = _Doc(text="blue whale", belongs_to_set=["blue"])
        await adapter.create_data_points("Doc_text", [a, b, c])

        async def ids(node_name, operator):
            hits = await adapter.search(
                "Doc_text",
                query_text="red big apple",
                limit=10,
                node_name=node_name,
                node_name_filter_operator=operator,
            )
            return {hit.id for hit in hits}

        assert await ids(["red"], "OR") == {a.id, b.id}
        assert await ids(["red", "big"], "AND") == {a.id}
        assert await ids(["red", "blue"], "OR") == {a.id, b.id, c.id}
        assert await ids(["red", "blue"], "AND") == set()
        # No node_name -> unfiltered.
        assert await ids(None, "OR") == {a.id, b.id, c.id}
        adapter.close()

    asyncio.run(body())


def test_belongs_to_set_union_on_reupsert(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        first = _Doc(text="shared node", belongs_to_set=["s1"])
        doc_id = first.id
        await adapter.create_data_points("Doc_text", [first])
        # Re-add the same id under a new set; the prior set must be preserved.
        second = _Doc(id=doc_id, text="shared node", belongs_to_set=["s2"])
        await adapter.create_data_points("Doc_text", [second])

        got = await adapter.retrieve("Doc_text", [str(doc_id)])
        assert sorted(got[0].payload["belongs_to_set"]) == ["s1", "s2"]
        # Filterable under either set.
        for tag in ("s1", "s2"):
            hits = await adapter.search(
                "Doc_text", query_text="shared node", limit=5, node_name=[tag]
            )
            assert {hit.id for hit in hits} == {doc_id}
        adapter.close()

    asyncio.run(body())


def test_delete_data_points(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        docs = [_Doc(text="keep me"), _Doc(text="delete me")]
        await adapter.create_data_points("Doc_text", docs)
        await adapter.delete_data_points("Doc_text", [docs[1].id])

        remaining = await adapter.retrieve(
            "Doc_text", [str(docs[0].id), str(docs[1].id)]
        )
        assert {result.id for result in remaining} == {docs[0].id}
        # Deleting from a missing collection is a no-op, not an error.
        await adapter.delete_data_points("Ghost_text", [docs[0].id])
        adapter.close()

    asyncio.run(body())


def test_index_verbs_and_batch_search(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        await adapter.create_vector_index("Entity", "name")
        assert await adapter.has_collection("Entity_name") is True

        docs = [_Doc(text="apple"), _Doc(text="banana"), _Doc(text="cherry")]
        await adapter.index_data_points("Entity", "name", docs)

        batched = await adapter.batch_search(
            "Entity_name", query_texts=["banana", "cherry"], limit=1
        )
        assert len(batched) == 2
        assert batched[0][0].id == docs[1].id
        assert batched[1][0].id == docs[2].id
        assert await adapter.batch_search("Entity_name", query_texts=[], limit=1) == []
        adapter.close()

    asyncio.run(body())


def test_remove_belongs_to_set_tags(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        a = _Doc(text="node a", belongs_to_set=["s1", "s2"])
        b = _Doc(text="node b", belongs_to_set=["s1"])
        c = _Doc(text="node c", belongs_to_set=["s3"])
        await adapter.create_data_points("Doc_text", [a, b, c])

        await adapter.remove_belongs_to_set_tags(["s1"])

        got = {r.id: r for r in await adapter.retrieve(
            "Doc_text", [str(a.id), str(b.id), str(c.id)]
        )}
        # a keeps s2; b loses its only tag and is deleted; c untouched.
        assert got[a.id].payload["belongs_to_set"] == ["s2"]
        assert b.id not in got
        assert got[c.id].payload["belongs_to_set"] == ["s3"]
        # s1 is now unqueryable; s2 still resolves to a.
        assert {h.id for h in await adapter.search(
            "Doc_text", query_text="node a", limit=5, node_name=["s1"]
        )} == set()
        assert {h.id for h in await adapter.search(
            "Doc_text", query_text="node a", limit=5, node_name=["s2"]
        )} == {a.id}
        adapter.close()

    asyncio.run(body())


def test_remove_belongs_to_set_tags_scoped_by_node_ids(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        a = _Doc(text="node a", belongs_to_set=["s1"])
        b = _Doc(text="node b", belongs_to_set=["s1"])
        await adapter.create_data_points("Doc_text", [a, b])

        # Scoped: only a loses s1 (and is deleted, tagless); b keeps it.
        await adapter.remove_belongs_to_set_tags(["s1"], node_ids=[str(a.id)])
        got = {r.id for r in await adapter.retrieve("Doc_text", [str(a.id), str(b.id)])}
        assert got == {b.id}
        adapter.close()

    asyncio.run(body())


def test_upsert_raw_vectors(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        vec = [0.0] * DIM
        vec[3] = 1.0
        rid = "11111111-1111-1111-1111-111111111111"
        await adapter.upsert_raw_vectors(
            "Centroid_text",
            [{"id": rid, "vector": vec, "payload": {"text": "centroid", "belongs_to_set": ["k"]}}],
        )
        hits = await adapter.search(
            "Centroid_text", query_vector=vec, limit=1, include_payload=True
        )
        assert hits[0].id == UUID(rid)
        assert hits[0].payload["text"] == "centroid"
        assert {h.id for h in await adapter.search(
            "Centroid_text", query_vector=vec, limit=5, node_name=["k"]
        )} == {UUID(rid)}
        adapter.close()

    asyncio.run(body())


def test_prune_removes_all_collections(tmp_path):
    async def body():
        adapter = _adapter(tmp_path)
        await adapter.create_data_points("A_text", [_Doc(text="a")])
        await adapter.create_data_points("B_text", [_Doc(text="b")])
        assert await adapter.has_collection("A_text") is True

        await adapter.prune()
        assert await adapter.has_collection("A_text") is False
        assert await adapter.has_collection("B_text") is False
        # Usable again after prune.
        await adapter.create_data_points("A_text", [_Doc(text="a2")])
        assert await adapter.has_collection("A_text") is True
        adapter.close()

    asyncio.run(body())


def test_rejects_non_multiple_of_8_dimension(tmp_path):
    async def body():
        adapter = _adapter(tmp_path, _StubEmbeddingEngine(dim=10))
        with pytest.raises(ValueError, match="multiple of 8"):
            await adapter.create_data_points("Doc_text", [_Doc(text="x")])

    asyncio.run(body())


def test_payload_with_non_str_dict_keys_round_trips(tmp_path):
    # A DataPoint whose payload has non-string dict keys must not crash json encoding
    # (json keys are strings anyway); keys are stringified on the way in.
    class _DocWithMap(DataPoint):
        text: str
        weights: dict = {}
        metadata: dict = {"index_fields": ["text"]}

    async def body():
        adapter = _adapter(tmp_path)
        doc = _DocWithMap(text="mixed keys", weights={1: "a", "b": 2})
        await adapter.create_data_points("Doc_text", [doc])
        got = await adapter.retrieve("Doc_text", [str(doc.id)])
        assert got[0].payload["weights"] == {"1": "a", "b": 2}
        adapter.close()

    asyncio.run(body())


def test_register_cognee_adapter():
    from cognee.infrastructure.databases.vector.supported_databases import supported_databases

    register_cognee_adapter("lodedb")
    assert supported_databases.get("lodedb") is CogneeLodeDBAdapter
