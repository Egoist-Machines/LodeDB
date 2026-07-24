"""End-to-end tests for the native bi-temporal ``TemporalKnowledgeGraph`` (the
``lodedb-graph`` Rust crate via the ``lodedb.graph`` Python wrapper).

Mirrors the crate's Rust integration suite through the Python surface: the
bi-temporal invariants (invalidation preserves history; as-of/now/history resolve
correctly), deterministic traversal, hybrid semantic retrieval, reindex, and episode
provenance. Uses a tiny deterministic embedder so the suite is offline.
"""

from __future__ import annotations

import math
import time

import pytest

from lodedb.graph import (
    TemporalKnowledgeGraph,
    episode_mentions_reranker,
    maximal_marginal_relevance,
    node_distance_reranker,
    rrf,
)


class HashEmbedder:
    """Deterministic dim-8 embedder: bucket bytes into 8 bins, L2-normalize.

    Similar text -> similar vectors, which is all these membership assertions need.
    """

    dimension = 8

    def embed(self, texts, role):  # role in {"document","query"}; symmetric here
        out = []
        for t in texts:
            v = [0.0] * 8
            for b in t.lower().encode("utf-8"):
                v[b % 8] += 1.0
            norm = math.sqrt(sum(x * x for x in v))
            out.append([x / norm for x in v] if norm else [1.0, 0, 0, 0, 0, 0, 0, 0])
        return out


class FixedQueryEmbedder:
    """Return one fixed query vector so vector-in ranking is fully controlled."""

    dimension = 8

    def embed(self, texts, role):
        return [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for _text in texts]


def graph() -> TemporalKnowledgeGraph:
    return TemporalKnowledgeGraph(embedder=HashEmbedder())


def test_embedder_dimension_mismatch_is_rejected_before_creating_path(tmp_path):
    path = tmp_path / "graph"
    with pytest.raises(ValueError, match="embedder dimension"):
        TemporalKnowledgeGraph(path=str(path), embedder=HashEmbedder(), vector_dim=16)
    assert not path.exists()


def test_invalidation_preserves_history_and_as_of():
    g = graph()
    g.upsert_entity("alice", "Person", "Alice, engineer")
    g.upsert_entity("acme", "Org", "Acme Corp")
    g.upsert_entity("globex", "Org", "Globex Corp")

    f_acme = g.add_fact("alice", "works_at", "acme", "Alice works at Acme", valid_at=1000)
    g.add_fact("alice", "works_at", "globex", "Alice works at Globex",
               valid_at=2000, invalidates=[f_acme])

    now = g.neighbors("alice", relation="works_at")
    assert [f["dst"] for f in now] == ["globex"]

    then = g.neighbors("alice", relation="works_at", as_of=1500)
    assert [f["dst"] for f in then] == ["acme"]

    later = g.neighbors("alice", relation="works_at", as_of=2500)
    assert [f["dst"] for f in later] == ["globex"]

    hist = g.history("alice")
    assert len(hist) == 2  # both assertions preserved
    acme_fact = next(f for f in hist if f["id"] == f_acme)
    assert acme_fact["invalid_at"] == 2000  # closed at the new fact's valid_at
    assert acme_fact["expired_at"] is not None  # closed on the transaction axis


def test_k_hop_traversal():
    g = graph()
    for eid, label in [("a", "node a"), ("b", "node b"), ("c", "node c"), ("d", "node d")]:
        g.upsert_entity(eid, "Thing", label)
    g.add_fact("a", "rel", "b", "a rel b", valid_at=1)
    g.add_fact("b", "rel", "c", "b rel c", valid_at=1)
    g.add_fact("c", "rel", "d", "c rel d", valid_at=1)

    one = g.k_hop("a", k=1, direction="out")
    ids = {e["id"] for e in one["entities"]}
    assert {"a", "b"} <= ids and "c" not in ids

    two = g.k_hop("a", k=2, direction="out")
    ids2 = {e["id"] for e in two["entities"]}
    assert "c" in ids2 and "d" not in ids2


def test_enumerate_and_search():
    g = graph()
    g.upsert_entity("alice", "Person", "Alice builds robots")
    g.upsert_entity("acme", "Org", "Acme robotics company")
    g.upsert_entity("nyc", "Place", "New York City")

    people = g.entities("Person")
    assert [e["id"] for e in people] == ["alice"]
    assert len(g.entities()) == 3

    hits = g.semantic_entities("robots", k=5)
    assert hits  # returns (score, entity) pairs
    assert g.stats()["entities"] == 3


def test_reindex_rebuilds_from_truth():
    g = graph()
    g.upsert_entity("x", "Thing", "widget")
    g.upsert_entity("y", "Thing", "gadget")
    g.add_fact("x", "is", "y", "x is y", valid_at=1)
    out = g.reindex()
    assert out["reindexed_entities"] == 2
    assert out["reindexed_facts"] == 1


def test_episode_reference_time():
    g = graph()
    g.upsert_entity("p", "Person", "Pat")
    g.upsert_entity("q", "Org", "QCo")
    ep = g.add_episode("note", "Pat joined QCo", 4242, mentions=["p"])
    fid = g.add_fact("p", "works_at", "q", "Pat works at QCo", episodes=[ep], valid_at=4242)
    assert g.get_fact(fid)["reference_time"] == 4242


def test_properties_roundtrip_as_dict():
    g = graph()
    g.upsert_entity("z", "Thing", "zed", properties={"color": "red", "n": 3})
    got = g.get_entity("z")
    assert got["properties"] == {"color": "red", "n": 3}


def test_open_start_as_of_consistency():
    g = graph()
    g.upsert_entity("s", "Thing", "source thing")
    g.upsert_entity("t", "Thing", "target thing")
    # valid_at=None -> "always started"; must be included by both topology and index.
    g.add_fact("s", "linked", "t", "s linked to t forever")

    nbrs = g.neighbors("s", relation="linked", as_of=500)
    assert [f["dst"] for f in nbrs] == ["t"]

    hits = g.semantic_facts("linked forever", k=5, as_of=500)
    assert hits  # index agrees the open-start fact is valid at any T


def test_vector_in_graph_reindexes_and_preserves_history():
    """Retained caller vectors rebuild the index and keep invalidated history searchable."""
    g = TemporalKnowledgeGraph(vector_dim=8)
    v = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    g.upsert_entity_vec("a", "Thing", "thing a", v)
    g.upsert_entity_vec("b", "Thing", "thing b", v)
    old = g.add_fact_vec("a", "rel", "b", "old fact", v, valid_at=10)
    g.add_fact_vec("a", "rel", "b", "new fact", v, valid_at=20, invalidates=[old])

    hits = g.semantic_facts(None, k=5, embedding=v, as_of=15)
    assert [f["id"] for _s, f in hits] == [old]

    result = g.reindex()
    assert result["reindexed_entities"] == 2
    assert result["reindexed_facts"] == 2
    assert [f["id"] for _s, f in g.semantic_facts(None, k=5, embedding=v, as_of=15)] == [
        old
    ]


def test_bool_as_of_is_rejected():
    g = graph()
    g.upsert_entity("a", "Thing", "thing a")
    with pytest.raises(TypeError):
        g.entities(as_of=True)


def test_invalidates_unknown_id_fails_atomically():
    g = graph()
    g.upsert_entity("a", "Thing", "thing a")
    g.upsert_entity("b", "Thing", "thing b")
    with pytest.raises(Exception, match="f-nope"):
        g.add_fact("a", "rel", "b", "a rel b", valid_at=10, invalidates=["f-nope"])
    assert g.stats()["facts"] == 0, "nothing persisted from the failed call"


def test_dangling_endpoints_and_episodes_are_refused():
    g = graph()
    g.upsert_entity("a", "Thing", "thing a")
    with pytest.raises(Exception, match="ghost"):
        g.add_fact("a", "rel", "ghost", "a rel ghost")
    with pytest.raises(Exception, match="ghost"):
        g.add_episode("note", "text", 100, mentions=["ghost"])


def test_index_text_false_opens_and_searches_by_vector():
    g = TemporalKnowledgeGraph(vector_dim=8, index_text=False)
    v = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    g.upsert_entity_vec("a", "Thing", "thing a", v)
    hits = g.semantic_entities(None, k=5, embedding=v)
    assert [e["id"] for _s, e in hits] == ["a"]


def test_strict_now_rejects_future_dated_facts():
    g = graph()
    g.upsert_entity("a", "Thing", "future source")
    g.upsert_entity("b", "Thing", "future target")
    now = int(time.time() * 1000)
    future = now + 7 * 24 * 60 * 60 * 1000
    g.add_fact("a", "rel", "b", "future source relation", valid_at=future)

    assert g.neighbors("a"), "the compatibility current view is unchanged"
    assert g.neighbors("a", as_of="now_valid") == []
    assert g.semantic_facts("future source relation", as_of="strict") == []
    assert g.search_subgraph(
        "future source", hops=1, as_of="now_valid"
    )["facts"] == []


def test_bitemporal_frame_selects_event_and_knowledge_time():
    g = graph()
    for id in ("person", "old", "new"):
        g.upsert_entity(id, "Thing", id)
    old = g.add_fact("person", "works_at", "old", "old employer", valid_at=1000)
    time.sleep(0.003)
    new = g.add_fact(
        "person",
        "works_at",
        "new",
        "new employer",
        valid_at=2000,
        invalidates=[old],
    )
    old_row = g.get_fact(old)
    new_row = g.get_fact(new)
    learned_at = old_row["expired_at"]

    before = g.neighbors("person", as_of=(2500, learned_at - 1))
    assert [fact["id"] for fact in before] == [old]
    before_semantic = g.semantic_facts(
        "employer", relation="works_at", as_of=(2500, learned_at - 1)
    )
    assert [fact["id"] for _score, fact in before_semantic] == [old]
    after = g.neighbors("person", as_of=(2500, new_row["created_at"]))
    assert [fact["id"] for fact in after] == [new]
    after_semantic = g.semantic_facts(
        "employer", relation="works_at", as_of=(2500, new_row["created_at"])
    )
    assert [fact["id"] for _score, fact in after_semantic] == [new]


def test_stable_ids_episode_enumeration_and_rollback():
    g = graph()
    g.upsert_entity("a", "Thing", "a")
    g.upsert_entity("b", "Thing", "b")
    ep1 = g.add_episode("note", "one", 1, id="episode-one")
    assert g.add_episode("note", "one", 1, id="episode-one") == ep1
    ep2 = g.add_episode("note", "two", 2, id="episode-two")
    retained = g.add_fact(
        "a",
        "rel",
        "b",
        "supported twice",
        episodes=[ep1, ep2],
        id="stable-fact",
    )
    assert (
        g.add_fact(
            "a",
            "rel",
            "b",
            "supported twice",
            episodes=[ep1, ep2],
            id="stable-fact",
        )
        == retained
    )
    removed = g.add_fact("a", "rel2", "b", "episode two", episodes=[ep2])

    assert {episode["id"] for episode in g.episodes()} == {ep1, ep2}
    assert {fact["id"] for fact in g.facts_by_episode(ep2)} == {retained, removed}
    assert g.remove_episode(ep2)
    assert g.get_fact(removed) is None
    assert g.get_fact(retained)["episodes"] == [ep1]


def test_entity_property_lineage_tracks_versions_and_sources():
    g = graph()
    g.upsert_entity(
        "device",
        "Device",
        "device",
        properties={"status": "new", "region": "us"},
        valid_at=100,
    )
    episode = g.add_episode("event", "activated", 200, id="activation")
    g.upsert_entity(
        "device",
        "Device",
        "device",
        properties={"status": "active", "region": "us"},
        valid_at=200,
        property_sources={"status": episode},
    )

    status = g.entity_property_history("device", "status")
    assert [item["value"] for item in status] == ["new", "active"]
    assert status[0]["expired_at"] is not None
    assert status[1]["episode_id"] == episode
    assert len(g.entity_property_history("device", "region")) == 1


def test_authorization_predicate_crowd_out_preserves_allowed_top_k():
    g = TemporalKnowledgeGraph(embedder=FixedQueryEmbedder())
    query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    allowed = [0.8, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    other = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    predicate = {"$and": [{"owner": "u1"}, {"allowed": True}]}

    for entity_id, label, properties, embedding in (
        (
            "forbidden-top",
            "crowd-out query",
            {"owner": "u2", "allowed": False},
            query,
        ),
        (
            "allowed-best",
            "allowed fallback",
            {"owner": "u1", "allowed": True},
            allowed,
        ),
        (
            "allowed-other",
            "other fallback",
            {"owner": "u1", "allowed": True},
            other,
        ),
    ):
        g.upsert_entity_vec(
            entity_id,
            "Thing",
            label,
            embedding,
            properties=properties,
        )

    forbidden_fact = g.add_fact_vec(
        "allowed-best",
        "rel",
        "allowed-other",
        "forbidden top fact",
        query,
        properties={"owner": "u2", "allowed": False},
    )
    allowed_fact = g.add_fact_vec(
        "allowed-best",
        "rel",
        "allowed-other",
        "allowed fallback fact",
        allowed,
        properties={"owner": "u1", "allowed": True},
    )

    assert g.semantic_entities(None, embedding=query, k=1)[0][1]["id"] == "forbidden-top"
    assert g.semantic_facts(None, embedding=query, k=1, relation="rel")[0][1]["id"] == (
        forbidden_fact
    )
    assert g.resolve_entity("crowd-out query", k=1)[0][1]["id"] == "forbidden-top"

    entities = g.semantic_entities(None, embedding=query, k=1, predicate=predicate)
    assert [entity["id"] for _score, entity in entities] == ["allowed-best"]

    facts = g.semantic_facts(
        None,
        embedding=query,
        k=1,
        relation="rel",
        predicate=predicate,
    )
    assert [fact["id"] for _score, fact in facts] == [allowed_fact]

    resolved = g.resolve_entity("crowd-out query", k=1, predicate=predicate)
    assert [entity["id"] for _score, entity in resolved] == ["allowed-best"]

    unfiltered_subgraph = g.search_subgraph(
        None,
        embedding=query,
        k=1,
        hops=0,
        relation="rel",
        seed_kind="fact",
    )
    assert [fact["id"] for fact in unfiltered_subgraph["facts"]] == [forbidden_fact]

    filtered_subgraph = g.search_subgraph(
        None,
        embedding=query,
        k=1,
        hops=0,
        relation="rel",
        seed_kind="fact",
        predicate=predicate,
    )
    assert [fact["id"] for fact in filtered_subgraph["facts"]] == [allowed_fact]


def test_authorization_predicate_blocks_forbidden_bridge_expansion():
    g = TemporalKnowledgeGraph(embedder=FixedQueryEmbedder())
    query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    other = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    allowed_properties = {"owner": "u1", "allowed": True}
    predicate = {"$and": [{"owner": "u1"}, {"allowed": True}]}

    for entity_id, properties, embedding in (
        ("allowed-a", allowed_properties, query),
        ("forbidden-m", {"owner": "u2", "allowed": False}, other),
        ("allowed-c", allowed_properties, other),
    ):
        g.upsert_entity_vec(
            entity_id,
            "Thing",
            entity_id,
            embedding,
            properties=properties,
        )
    g.add_fact_vec(
        "allowed-a",
        "rel",
        "forbidden-m",
        "allowed a to forbidden m",
        other,
        properties=allowed_properties,
    )
    g.add_fact_vec(
        "forbidden-m",
        "rel",
        "allowed-c",
        "forbidden m to allowed c",
        other,
        properties=allowed_properties,
    )

    unfiltered = g.k_hop("allowed-a", k=2, direction="out")
    assert "allowed-c" in {entity["id"] for entity in unfiltered["entities"]}
    assert len(unfiltered["facts"]) == 2

    filtered = g.k_hop(
        "allowed-a",
        k=2,
        direction="out",
        predicate=predicate,
    )
    filtered_ids = {entity["id"] for entity in filtered["entities"]}
    assert "allowed-a" in filtered_ids
    assert "forbidden-m" not in filtered_ids
    assert "allowed-c" not in filtered_ids
    assert all(
        fact["src"] != "forbidden-m" and fact["dst"] != "forbidden-m"
        for fact in filtered["facts"]
    )

    unfiltered_subgraph = g.search_subgraph(
        None,
        embedding=query,
        k=1,
        hops=2,
        direction="out",
        relation="rel",
        seed_kind="entity",
    )
    assert "allowed-c" in {
        entity["id"] for entity in unfiltered_subgraph["entities"]
    }

    filtered_subgraph = g.search_subgraph(
        None,
        embedding=query,
        k=1,
        hops=2,
        direction="out",
        relation="rel",
        seed_kind="entity",
        predicate=predicate,
    )
    filtered_subgraph_ids = {
        entity["id"] for entity in filtered_subgraph["entities"]
    }
    assert "allowed-a" in filtered_subgraph_ids
    assert "forbidden-m" not in filtered_subgraph_ids
    assert "allowed-c" not in filtered_subgraph_ids
    assert all(
        fact["src"] != "forbidden-m" and fact["dst"] != "forbidden-m"
        for fact in filtered_subgraph["facts"]
    )


def test_native_rerankers_are_exposed():
    fused = rrf([["a", "b"], ["a", "c"]])
    assert fused[0][0] == "a"
    assert maximal_marginal_relevance(
        [1.0, 0.0],
        [("near", [1.0, 0.0]), ("diverse", [0.0, 1.0])],
        lambda_=1.0,
    ) == ["near", "diverse"]
    assert node_distance_reranker(["far", "near"], {"far": 2, "near": 0}) == [
        "near",
        "far",
    ]
    assert episode_mentions_reranker(["low", "high"], {"low": 1, "high": 3}) == [
        "high",
        "low",
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
