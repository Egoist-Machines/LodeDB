"""End-to-end tests for the native bi-temporal ``TemporalKnowledgeGraph`` (the
``lodedb-graph`` Rust crate via the ``lodedb.graph`` Python wrapper).

Mirrors the crate's Rust integration suite through the Python surface: the
bi-temporal invariants (invalidation preserves history; as-of/now/history resolve
correctly), deterministic traversal, hybrid semantic retrieval, reindex, and episode
provenance. Uses a tiny deterministic embedder so the suite is offline.
"""

from __future__ import annotations

import math

import pytest

from lodedb.graph import TemporalKnowledgeGraph


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
