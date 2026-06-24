"""Tests for the LodeDB framework adapters (gated on the optional framework deps)."""

from __future__ import annotations

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB


def test_langchain_vectorstore_roundtrip(tmp_path):
    """The LangChain adapter adds/searches/deletes and round-trips page_content."""

    pytest.importorskip("langchain_core")  # needs lodedb[langchain]
    from langchain_core.documents import Document

    from lodedb.local.integrations.langchain import LodeDBVectorStore

    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    store = LodeDBVectorStore(db)
    ids = store.add_texts(
        ["alpha document", "beta document"],
        metadatas=[{"k": "a"}, {"k": "b"}],
        ids=["a", "b"],
    )
    assert ids == ["a", "b"]

    docs = store.similarity_search("alpha", k=2)
    assert docs and all(isinstance(d, Document) for d in docs)
    # page_content round-trips via the stored `text` metadata key.
    assert "alpha document" in {d.page_content for d in docs}
    assert all("id" in d.metadata for d in docs)

    scored = store.similarity_search_with_score("alpha", k=2)
    assert all(isinstance(s, float) for _, s in scored)

    assert store.delete(["a"]) is True
    assert store.delete([]) is False
    db.close()


def test_langchain_vectorstore_predicate_filter(tmp_path):
    """The LangChain adapter passes predicate filters straight through to LodeDB."""

    pytest.importorskip("langchain_core")  # needs lodedb[langchain]

    from lodedb.local.integrations.langchain import LodeDBVectorStore

    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    store = LodeDBVectorStore(db)
    store.add_texts(
        ["alpha document", "beta document", "gamma document"],
        metadatas=[{"year": 2019}, {"year": 2021}, {"year": 2023}],
        ids=["a", "b", "c"],
    )
    docs = store.similarity_search("document", k=10, filter={"year": {"$gte": 2021}})
    assert {d.metadata["id"] for d in docs} == {"b", "c"}
    db.close()


# -- LlamaIndex adapter -----------------------------------------------------


def _llama_db(tmp_path):
    """Opens a LodeDB with a deterministic hash backend (no model download)."""

    return LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )


def _llama_nodes():
    """Four TextNodes: two share source docA, one has docB, one carries an exact token."""

    from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

    n1 = TextNode(id_="n1", text="the quick brown fox", metadata={"topic": "animals", "year": 2019})
    n1.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="docA")
    n2 = TextNode(id_="n2", text="a lazy dog sleeps", metadata={"topic": "animals", "year": 2021})
    n2.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="docA")
    n3 = TextNode(
        id_="n3", text="quantum field theory", metadata={"topic": "physics", "year": 2023}
    )
    n3.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id="docB")
    n4 = TextNode(id_="n4", text="request failed with error E1234", metadata={"topic": "logs"})
    return n1, n2, n3, n4


def test_llama_index_vectorstore_roundtrip(tmp_path):
    """add / query / get_nodes round-trip text, metadata, and the SOURCE relationship."""

    pytest.importorskip("llama_index.core")  # needs lodedb[llama-index]
    from llama_index.core.vector_stores.types import VectorStoreQuery

    from lodedb.local.integrations.llama_index import LodeDBVectorStore

    db = _llama_db(tmp_path)
    store = LodeDBVectorStore(db)
    assert store.is_embedding_query is False  # text-path: LodeDB embeds query_str

    n1, n2, n3, n4 = _llama_nodes()
    assert store.add([n1, n2, n3, n4]) == ["n1", "n2", "n3", "n4"]

    res = store.query(VectorStoreQuery(query_str="fox", similarity_top_k=10))
    assert set(res.ids) == {"n1", "n2", "n3", "n4"}
    assert all(isinstance(s, float) for s in res.similarities)

    # get_nodes reconstructs text + metadata + SOURCE, and preserves requested order.
    nodes = store.get_nodes(node_ids=["n3", "n1"])
    assert [n.node_id for n in nodes] == ["n3", "n1"]
    assert nodes[1].get_content() == "the quick brown fox"
    assert nodes[1].metadata["topic"] == "animals"
    assert nodes[1].ref_doc_id == "docA"
    # The reserved ref-doc key is not leaked into user-facing metadata.
    assert all(k.startswith("_lodedb_") is False for k in nodes[1].metadata)

    # doc_ids scoping resolves through stored metadata (docA -> n1, n2).
    by_doc = store.query(
        VectorStoreQuery(query_str="anything", similarity_top_k=10, doc_ids=["docA"])
    )
    assert set(by_doc.ids) == {"n1", "n2"}

    db.close()


def test_llama_index_vectorstore_delete_is_durable(tmp_path):
    """delete(ref_doc_id) resolves through durable metadata, so it survives a reopen."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.vector_stores.types import VectorStoreQuery

    from lodedb.local.integrations.llama_index import LodeDBVectorStore

    db = _llama_db(tmp_path)
    store = LodeDBVectorStore(db)
    store.add(list(_llama_nodes()))
    db.close()

    # Reopen as a fresh handle (new session, empty in-memory state) and delete by ref doc.
    db2 = _llama_db(tmp_path)
    store2 = LodeDBVectorStore(db2)
    store2.delete("docA")
    remaining = store2.query(VectorStoreQuery(query_str="anything", similarity_top_k=10))
    assert set(remaining.ids) == {"n3", "n4"}
    db2.close()


def test_llama_index_vectorstore_filters(tmp_path):
    """MetadataFilters translate across operators and AND/OR/NOT (incl. nested)."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.vector_stores.types import (
        FilterCondition,
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
    )

    from lodedb.local.integrations.llama_index import LodeDBVectorStore

    db = _llama_db(tmp_path)
    store = LodeDBVectorStore(db)
    store.add(list(_llama_nodes()))

    def ids(filters):
        return {n.node_id for n in store.get_nodes(filters=filters)}

    # Ordered comparison (numeric over string metadata).
    gte = MetadataFilters(
        filters=[MetadataFilter(key="year", value=2021, operator=FilterOperator.GTE)]
    )
    assert ids(gte) == {"n2", "n3"}

    # IN membership.
    in_topic = MetadataFilters(
        filters=[MetadataFilter(key="topic", value=["physics", "logs"], operator=FilterOperator.IN)]
    )
    assert ids(in_topic) == {"n3", "n4"}

    # OR composition.
    or_filter = MetadataFilters(
        filters=[
            MetadataFilter(key="topic", value="logs", operator=FilterOperator.EQ),
            MetadataFilter(key="year", value=2023, operator=FilterOperator.GTE),
        ],
        condition=FilterCondition.OR,
    )
    assert ids(or_filter) == {"n3", "n4"}

    # NOT means "none of these match".
    not_physics = MetadataFilters(
        filters=[MetadataFilter(key="topic", value="physics", operator=FilterOperator.EQ)],
        condition=FilterCondition.NOT,
    )
    assert ids(not_physics) == {"n1", "n2", "n4"}

    # IS_EMPTY: n4 has no "year" metadata.
    empty_year = MetadataFilters(
        filters=[MetadataFilter(key="year", value=None, operator=FilterOperator.IS_EMPTY)]
    )
    assert ids(empty_year) == {"n4"}

    # Nested: animals AND (year < 2020 OR year >= 2023) -> just n1.
    nested = MetadataFilters(
        filters=[
            MetadataFilter(key="topic", value="animals", operator=FilterOperator.EQ),
            MetadataFilters(
                filters=[
                    MetadataFilter(key="year", value=2020, operator=FilterOperator.LT),
                    MetadataFilter(key="year", value=2023, operator=FilterOperator.GTE),
                ],
                condition=FilterCondition.OR,
            ),
        ],
        condition=FilterCondition.AND,
    )
    assert ids(nested) == {"n1"}

    # Filters also flow through the query() path (set is filter-determined for a wide top-k).
    physics = MetadataFilters(
        filters=[MetadataFilter(key="topic", value="physics", operator=FilterOperator.EQ)]
    )
    res = store.query(VectorStoreQuery(query_str="science", similarity_top_k=10, filters=physics))
    assert set(res.ids) == {"n3"}

    db.close()


def test_llama_index_vectorstore_hybrid_and_lexical(tmp_path):
    """Hybrid/lexical modes recover an exact token the embedding would miss."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    from lodedb.local.integrations.llama_index import LodeDBVectorStore

    db = _llama_db(tmp_path)  # store_text=True by default -> lexical source available
    store = LodeDBVectorStore(db)
    store.add(list(_llama_nodes()))

    lexical = store.query(
        VectorStoreQuery(
            query_str="E1234", similarity_top_k=3, mode=VectorStoreQueryMode.TEXT_SEARCH
        )
    )
    assert lexical.ids and lexical.ids[0] == "n4"

    hybrid = store.query(
        VectorStoreQuery(query_str="E1234", similarity_top_k=4, mode=VectorStoreQueryMode.HYBRID)
    )
    assert "n4" in hybrid.ids

    db.close()


def test_llama_index_vectorstore_delete_nodes(tmp_path):
    """delete_nodes removes by explicit ids or by filter; an empty call is a no-op."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

    from lodedb.local.integrations.llama_index import LodeDBVectorStore

    db = _llama_db(tmp_path)
    store = LodeDBVectorStore(db)
    store.add(list(_llama_nodes()))

    # No-op safety: neither argument must not wipe the store.
    store.delete_nodes()
    assert {n.node_id for n in store.get_nodes()} == {"n1", "n2", "n3", "n4"}

    # By explicit ids.
    store.delete_nodes(node_ids=["n1"])
    assert {n.node_id for n in store.get_nodes()} == {"n2", "n3", "n4"}

    # By filter.
    logs = MetadataFilters(
        filters=[MetadataFilter(key="topic", value="logs", operator=FilterOperator.EQ)]
    )
    store.delete_nodes(filters=logs)
    assert {n.node_id for n in store.get_nodes()} == {"n2", "n3"}

    db.close()


def test_llama_index_vectorstore_unsupported(tmp_path):
    """The adapter rejects the operations LodeDB cannot honor, loudly."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
        VectorStoreQueryMode,
    )

    from lodedb.local.integrations.llama_index import LodeDBVectorStore

    db = _llama_db(tmp_path)
    store = LodeDBVectorStore(db)

    # Empty/non-text node cannot be embedded.
    with pytest.raises(ValueError):
        store.add([TextNode(id_="empty", text="   ")])

    # MMR / learned modes need full-precision vectors LodeDB does not expose.
    with pytest.raises(NotImplementedError):
        store.query(VectorStoreQuery(query_str="x", mode=VectorStoreQueryMode.MMR))

    # Text-path needs a query string, not an embedding-only query.
    with pytest.raises(ValueError):
        store.query(VectorStoreQuery(query_embedding=[0.0] * 384, similarity_top_k=1))

    # Substring/list filter operators have no LodeDB metadata equivalent.
    contains = MetadataFilters(
        filters=[MetadataFilter(key="topic", value="ani", operator=FilterOperator.CONTAINS)]
    )
    with pytest.raises(NotImplementedError):
        store.get_nodes(filters=contains)

    # No full-precision vector read.
    with pytest.raises(NotImplementedError):
        store.get("n1")

    db.close()


# -- LlamaIndex PropertyGraphStore adapter ----------------------------------


def _llama_kg(tmp_path):
    """Opens a KnowledgeGraph with a deterministic hash backend (no model download)."""

    from lodedb.graph import KnowledgeGraph

    return KnowledgeGraph(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )


def test_llama_index_property_graph_roundtrip(tmp_path):
    """upsert / get / get_triplets / get_rel_map / vector_query / delete round-trip."""

    pytest.importorskip("llama_index.core")  # needs lodedb[llama-index]
    from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation
    from llama_index.core.vector_stores.types import VectorStoreQuery

    from lodedb.local.integrations.llama_index_graph import LodeDBPropertyGraphStore

    kg = _llama_kg(tmp_path)
    store = LodeDBPropertyGraphStore(kg)
    assert store.supports_vector_queries is True
    assert store.supports_structured_queries is False

    alice = EntityNode(name="alice", label="PERSON", properties={"role": "engineer", "level": 3})
    acme = EntityNode(name="acme", label="ORG")
    chunk = ChunkNode(text="Alice works at Acme on the search team.", id_="c1")
    store.upsert_nodes([alice, acme, chunk])
    store.upsert_relations(
        [
            Relation(
                label="WORKS_AT", source_id="alice", target_id="acme", properties={"since": 2020}
            ),
            Relation(label="MENTIONS", source_id="c1", target_id="alice"),
        ]
    )

    # get by id reconstructs the right subclass; JSON properties keep their types.
    got = store.get(ids=["alice"])
    assert len(got) == 1 and isinstance(got[0], EntityNode)
    assert got[0].name == "alice" and got[0].label == "PERSON"
    assert got[0].properties == {"role": "engineer", "level": 3}  # int kept, reserved key stripped
    chunks = store.get(ids=["c1"])
    assert isinstance(chunks[0], ChunkNode) and chunks[0].text.startswith("Alice works at Acme")

    # get by property (any key matches).
    assert {n.id for n in store.get(properties={"role": "engineer"})} == {"alice"}

    # triplets touching alice (source or target).
    touching = {(t[0].id, t[1].id, t[2].id) for t in store.get_triplets(entity_names=["alice"])}
    assert ("alice", "WORKS_AT", "acme") in touching
    assert ("c1", "MENTIONS", "alice") in touching

    # relation-name filter enumerates all edges of that type.
    works = store.get_triplets(relation_names=["WORKS_AT"])
    assert {(t[0].id, t[2].id) for t in works} == {("alice", "acme")}

    # rel map expands from a seed node over the topology.
    rel_map = store.get_rel_map([EntityNode(name="alice", label="PERSON")], depth=2)
    assert any(t[2].id == "acme" for t in rel_map)

    # vector_query, text-path: top-k over the indexed node text.
    nodes, scores = store.vector_query(
        VectorStoreQuery(query_str="who is the engineer?", similarity_top_k=3)
    )
    assert "alice" in {n.id for n in nodes}
    assert all(isinstance(s, float) for s in scores)

    # delete by id removes the node and its incident edges.
    store.delete(ids=["acme"])
    assert store.get(ids=["acme"]) == []
    assert store.get_triplets(relation_names=["WORKS_AT"]) == []

    kg.close()


def test_llama_index_property_graph_hybrid_and_embedding(tmp_path):
    """vector_query honors hybrid mode (query_str) and a pure embedding query."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.graph_stores.types import ChunkNode
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    from lodedb.local.integrations.llama_index_graph import LodeDBPropertyGraphStore

    kg = _llama_kg(tmp_path)  # store_text=True by default -> lexical source available
    store = LodeDBPropertyGraphStore(kg)
    store.upsert_nodes(
        [
            ChunkNode(text="request failed with error E1234", id_="c1"),
            ChunkNode(text="the cat sat on the mat", id_="c2"),
        ]
    )

    # Hybrid recovers the exact token via the query string.
    nodes, _ = store.vector_query(
        VectorStoreQuery(query_str="E1234", similarity_top_k=2, mode=VectorStoreQueryMode.HYBRID)
    )
    assert "c1" in {n.id for n in nodes}

    # Pure embedding query (what the high-level VectorContextRetriever passes).
    by_vec, _ = store.vector_query(
        VectorStoreQuery(query_embedding=[1.0] * 384, similarity_top_k=2)
    )
    assert {n.id for n in by_vec} <= {"c1", "c2"}

    kg.close()


def test_llama_index_property_graph_unsupported(tmp_path):
    """Structured queries and unsupported vector modes raise; empty get_triplets is []."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.vector_stores.types import VectorStoreQuery, VectorStoreQueryMode

    from lodedb.local.integrations.llama_index_graph import LodeDBPropertyGraphStore

    kg = _llama_kg(tmp_path)
    store = LodeDBPropertyGraphStore(kg)

    with pytest.raises(NotImplementedError):
        store.structured_query("MATCH (n) RETURN n")
    with pytest.raises(NotImplementedError):
        store.vector_query(VectorStoreQuery(query_str="x", mode=VectorStoreQueryMode.MMR))
    # No filters -> no triplets (matching LlamaIndex's reference store).
    assert store.get_triplets() == []

    kg.close()


def test_llama_index_property_graph_vector_only(tmp_path):
    """A vector-only graph stores LlamaIndex node embeddings and queries by embedding.

    This is the path that makes the high-level PropertyGraphIndex work with any embed_model:
    nodes carry their own embeddings (here dim 8) and the query is a precomputed vector.
    """

    pytest.importorskip("llama_index.core")
    from llama_index.core.graph_stores.types import EntityNode
    from llama_index.core.vector_stores.types import VectorStoreQuery

    from lodedb.local.integrations.llama_index_graph import LodeDBPropertyGraphStore

    def onehot(i: int) -> list[float]:
        vector = [0.0] * 8
        vector[i] = 1.0
        return vector

    store = LodeDBPropertyGraphStore.from_path(str(tmp_path / "kg"), vector_dim=8)
    assert store.client.vector_only is True

    store.upsert_nodes(
        [
            EntityNode(name="a", label="X", embedding=onehot(0)),
            EntityNode(name="b", label="X", embedding=onehot(3)),
        ]
    )

    # Query by a precomputed embedding (what VectorContextRetriever passes); deterministic
    # via one-hot vectors.
    nodes, scores = store.vector_query(
        VectorStoreQuery(query_embedding=onehot(3), similarity_top_k=2)
    )
    assert nodes[0].id == "b"
    assert all(isinstance(s, float) for s in scores)

    # Topology / get still work; reconstructed nodes are EntityNodes.
    assert {n.id for n in store.get(ids=["a", "b"])} == {"a", "b"}

    # A text query has nothing to embed in a vector-only graph.
    with pytest.raises(ValueError):
        store.vector_query(VectorStoreQuery(query_str="hello", similarity_top_k=1))

    store.client.close()


def test_llama_index_property_graph_high_level_retriever(tmp_path):
    """End-to-end: LlamaIndex's VectorContextRetriever drives a vector-only graph.

    Proves the high-level PropertyGraphIndex retrieval path "just works" with any embed_model:
    the retriever embeds the query, vector_query finds the seed, and get_rel_map returns its
    relational context.
    """

    pytest.importorskip("llama_index.core")
    vector_mod = pytest.importorskip(
        "llama_index.core.indices.property_graph.sub_retrievers.vector"
    )
    from llama_index.core.embeddings import MockEmbedding
    from llama_index.core.graph_stores.types import EntityNode, Relation

    from lodedb.local.integrations.llama_index_graph import LodeDBPropertyGraphStore

    emb = MockEmbedding(embed_dim=8)  # arbitrary dim; the graph is opened to match
    store = LodeDBPropertyGraphStore.from_path(str(tmp_path / "kg"), vector_dim=8)
    alice = EntityNode(name="alice", label="X")
    alice.embedding = emb.get_text_embedding("alice")
    acme = EntityNode(name="acme", label="X")
    acme.embedding = emb.get_text_embedding("acme")
    store.upsert_nodes([alice, acme])
    store.upsert_relations([Relation(label="WORKS_AT", source_id="alice", target_id="acme")])

    retriever = vector_mod.VectorContextRetriever(
        graph_store=store, embed_model=emb, similarity_top_k=2, include_text=False
    )
    results = retriever.retrieve("who is alice?")
    assert results  # the seed's relational context (alice -[WORKS_AT]-> acme)

    store.client.close()


# -- PrivateGPT vector-store provider ---------------------------------------
#
# PrivateGPT is an application, not a pip-installable library, so it cannot be a dependency or an
# importorskip target. Its vector-store contract is, however, tiny and stable: a VectorStoreFactory
# ABC with vector_store(collection) -> BasePydanticVectorStore, plus a register_vector_store() that
# fills a process-local _PROVIDERS dict. These tests stand up that exact contract as a local stub
# and exercise the real LodeDB provider code (register_lodedb_provider / LodeDBVectorStoreFactory)
# against it, with the LlamaIndex extra supplying BasePydanticVectorStore.


def _install_private_gpt_factory_stub(monkeypatch):
    """Builds a stand-in for PrivateGPT's factory module and binds the provider shim to it.

    Returns ``(VectorStoreFactory, providers)`` where ``providers`` is the stub registry the
    provider registers into. Mirrors ``private_gpt.components.vector_store.factory``.
    """

    from abc import ABC, abstractmethod

    from llama_index.core.vector_stores.types import BasePydanticVectorStore

    providers: dict[str, object] = {}

    class VectorStoreFactory(ABC):
        def __init__(self, settings, embed_dim=None):
            self.settings = settings
            self.embed_dim = embed_dim

        @abstractmethod
        def vector_store(self, collection: str) -> BasePydanticVectorStore: ...

    def register_vector_store(database: str, provider) -> None:
        providers[database] = provider

    # The provider shim imports PrivateGPT lazily through this single helper, so patching it is
    # enough to run the real registration/factory code without a PrivateGPT checkout.
    from lodedb.local.integrations import privategpt as pgpt_mod

    monkeypatch.setattr(
        pgpt_mod,
        "_load_private_gpt_factory_base",
        lambda: (VectorStoreFactory, register_vector_store),
    )

    # Build the provider's per-collection indexes with the deterministic hash embedding backend
    # the other adapter tests use, so these tests never download or load a real
    # SentenceTransformer model. That keeps them offline and fast and avoids loading a torch
    # model onto a GPU/MPS device on CI runners. Device is dropped since the hash backend needs
    # none.
    def _hash_backed_lodedb(*args, **kwargs):
        kwargs.pop("device", None)
        kwargs.setdefault("_embedding_backend", HashEmbeddingBackend(native_dim=384))
        return LodeDB(*args, **kwargs)

    monkeypatch.setattr(pgpt_mod, "LodeDB", _hash_backed_lodedb)
    return VectorStoreFactory, providers


class _StubVectorstoreSettings:
    """Stand-in for PrivateGPT's ``settings.vectorstore`` (only the fields the shim reads)."""

    def __init__(self, embed_dim=384):
        self.embed_dim = embed_dim


class _StubSettings:
    """Stand-in for PrivateGPT's ``Settings`` carrying an optional ``lodedb`` block."""

    def __init__(self, lodedb=None, embed_dim=384):
        self.vectorstore = _StubVectorstoreSettings(embed_dim=embed_dim)
        self.lodedb = lodedb


def test_privategpt_provider_registers_and_builds_store(tmp_path, monkeypatch):
    """register_lodedb_provider registers under 'lodedb'; the factory builds a working store."""

    pytest.importorskip("llama_index.core")  # needs lodedb[llama-index]
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryMode,
    )

    from lodedb.local.integrations.privategpt import register_lodedb_provider

    _factory_base, providers = _install_private_gpt_factory_stub(monkeypatch)

    factory_cls = register_lodedb_provider()
    # The provider landed in PrivateGPT's registry under the default name.
    assert providers["lodedb"] is factory_cls

    settings = _StubSettings(lodedb={"path": str(tmp_path / "pgpt"), "model": "minilm"})
    factory = factory_cls(settings, embed_dim=384)
    store = factory.vector_store("docs")
    assert isinstance(store, BasePydanticVectorStore)
    assert store.is_embedding_query is False  # text-path, like the LlamaIndex adapter

    store.add(
        [
            TextNode(id_="d1", text="LodeDB keeps data local", metadata={"src": "a"}),
            TextNode(id_="d2", text="request failed with error E1234", metadata={"src": "log"}),
        ]
    )
    res = store.query(VectorStoreQuery(query_str="local data", similarity_top_k=2))
    assert set(res.ids) == {"d1", "d2"}
    # Hybrid still works through the provider-built store (store_text on by default).
    hybrid = store.query(
        VectorStoreQuery(query_str="E1234", similarity_top_k=2, mode=VectorStoreQueryMode.HYBRID)
    )
    assert "d2" in hybrid.ids

    factory.close()


def test_privategpt_provider_collections_are_isolated(tmp_path, monkeypatch):
    """Each PrivateGPT collection maps to its own LodeDB index; same name is cached."""

    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import TextNode

    from lodedb.local.integrations.privategpt import register_lodedb_provider

    _factory_base, _providers = _install_private_gpt_factory_stub(monkeypatch)
    factory_cls = register_lodedb_provider()

    factory = factory_cls(_StubSettings(lodedb={"path": str(tmp_path / "pgpt")}), embed_dim=384)
    alpha = factory.vector_store("alpha")
    beta = factory.vector_store("beta")
    # Requesting the same collection returns the cached store (one handle per collection).
    assert factory.vector_store("alpha") is alpha

    alpha.add([TextNode(id_="a1", text="only in alpha", metadata={})])
    beta.add([TextNode(id_="b1", text="only in beta", metadata={})])

    a_ids = {n.node_id for n in alpha.get_nodes()}
    b_ids = {n.node_id for n in beta.get_nodes()}
    assert a_ids == {"a1"} and b_ids == {"b1"}  # collections are separate indexes

    # Each collection persisted to its own subdirectory under the configured path.
    assert (tmp_path / "pgpt" / "alpha").exists()
    assert (tmp_path / "pgpt" / "beta").exists()

    factory.close()


def test_privategpt_provider_settings_defaults_and_override(tmp_path, monkeypatch):
    """An absent lodedb block falls back to defaults; a provided block overrides them."""

    pytest.importorskip("llama_index.core")

    from lodedb.local.integrations.privategpt import _lodedb_settings, register_lodedb_provider

    # Defaults when no lodedb block is configured.
    defaults = _lodedb_settings(_StubSettings(lodedb=None))
    assert defaults["model"] == "minilm"
    assert defaults["store_text"] is True
    assert defaults["index_text"] is False
    assert defaults["path"] == "local_data/lodedb"

    # A partial block overrides only the keys it sets.
    merged = _lodedb_settings(_StubSettings(lodedb={"model": "bge", "index_text": True}))
    assert merged["model"] == "bge"
    assert merged["index_text"] is True
    assert merged["store_text"] is True  # untouched default preserved

    # The override actually reaches the constructed LodeDB handle.
    _factory_base, _providers = _install_private_gpt_factory_stub(monkeypatch)
    factory_cls = register_lodedb_provider()
    factory = factory_cls(
        _StubSettings(lodedb={"path": str(tmp_path / "pgpt"), "index_text": True}), embed_dim=384
    )
    store = factory.vector_store("docs")
    assert store.client.index_text is True
    factory.close()


def test_privategpt_provider_requires_private_gpt():
    """Outside a PrivateGPT environment, registration raises a clear, actionable error."""

    pytest.importorskip("llama_index.core")

    from lodedb.local.integrations.privategpt import register_lodedb_provider

    # No stub installed and PrivateGPT is genuinely not importable in this repo's env.
    with pytest.raises(ImportError, match="PrivateGPT"):
        register_lodedb_provider()
