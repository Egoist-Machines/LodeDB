"""Hybrid (BM25 + RRF) lexical retrieval through the knowledge-graph search API.

The graph's text-query search forwards ``mode`` to LodeDB, so an exact token in a
node label or edge fact (an error code, serial, or date) that a content-blind
embedding ranks low is recovered by ``mode="hybrid"`` / ``"lexical"``. A
deterministic backend embeds the carrier orthogonally to the query so pure vector
misses it, isolating the lexical contribution (the same trick the SDK hybrid
tests use, since a hash embedding cannot rank a literal token).
"""

from __future__ import annotations

import pytest

from lodedb.graph import KnowledgeGraph

# Matches the default "minilm" preset dim the graph opens LodeDB with; the engine
# rejects a backend whose native_dim disagrees with the configured model.
DIM = 384


class _ExactMissBackend:
    """Embeds ``anchor`` text and every query to one axis, other text to another.

    A pure-vector query then ranks the ``anchor`` distractors first and the
    carrier (whose label only adds an exact code) orthogonally, so vector search
    misses it; the lexical ranker matches the code in the label and recovers it.
    """

    name = "exact_miss"
    required_model_name = None
    native_dim = DIM

    def _axis(self, i: int) -> tuple[float, ...]:
        vector = [0.0] * DIM
        vector[i] = 1.0
        return tuple(vector)

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple(self._axis(0) if "anchor" in text else self._axis(1) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self._axis(0)


def _kg(path, **kwargs) -> KnowledgeGraph:
    return KnowledgeGraph(path=path, _embedding_backend=_ExactMissBackend(), **kwargs)


def _seed_carrier_and_distractors(kg: KnowledgeGraph) -> str:
    """Adds one carrier node whose label carries an exact code, plus distractors."""

    kg.add_node(id="carrier", type="Incident", label="maintenance log fault code E1234 overnight")
    for i in range(6):
        kg.add_node(id=f"d{i}", type="Note", label=f"anchor distractor note number {i}")
    return "carrier"


@pytest.mark.parametrize(
    "open_kwargs",
    [{"store_text": True}, {"store_text": False, "index_text": True}],
    ids=["store_text", "index_text"],
)
def test_semantic_nodes_hybrid_surfaces_exact_label_token(tmp_path, open_kwargs):
    """A label-only exact token vector misses is surfaced by hybrid/lexical search."""

    kg = _kg(tmp_path, **open_kwargs)
    carrier = _seed_carrier_and_distractors(kg)

    vector_ids = [node.id for _score, node in kg.semantic_nodes("E1234", k=3)]
    hybrid_ids = [node.id for _score, node in kg.semantic_nodes("E1234", k=3, mode="hybrid")]
    lexical_ids = [node.id for _score, node in kg.semantic_nodes("E1234", k=3, mode="lexical")]

    assert carrier not in vector_ids  # the content-blind embedding misses the code
    assert hybrid_ids[0] == carrier  # BM25 over the label recovers it, RRF lifts it
    assert lexical_ids == [carrier]  # only the carrier contains the token
    kg.close()


def test_search_subgraph_hybrid_seeds_on_exact_token(tmp_path):
    """search_subgraph forwards mode, so a lexical seed drives the k-hop expansion."""

    kg = _kg(tmp_path)
    carrier = _seed_carrier_and_distractors(kg)
    kg.add_node(id="unit3", type="Unit", label="anchor turbine unit three")
    kg.add_edge(carrier, "affects", "unit3")

    sub = kg.search_subgraph("E1234", k=1, hops=1, mode="hybrid")

    assert carrier in {node_id for node_id, _score in sub.seeds}
    assert "unit3" in sub.nodes  # one hop out from the lexically seeded node
    kg.close()


def test_semantic_edges_hybrid_over_fact_text(tmp_path):
    """Edge facts are text-indexed, so hybrid edge search matches exact tokens too."""

    kg = _kg(tmp_path, index_edges=True)
    kg.add_node(id="a", label="anchor alpha")
    kg.add_node(id="b", label="anchor beta")
    kg.add_edge("a", "logged", "b", fact="error E1234 raised during the overnight sync")
    kg.add_edge("a", "notes", "b", fact="anchor routine maintenance note, nothing unusual")

    hits = kg.semantic_edges("E1234", k=2, mode="hybrid")

    assert hits and hits[0][1].relation == "logged"
    kg.close()


def test_graph_embedding_with_nonvector_mode_raises(tmp_path):
    """A precomputed embedding is a pure vector query; a lexical mode is rejected."""

    kg = _kg(tmp_path)
    _seed_carrier_and_distractors(kg)
    with pytest.raises(ValueError, match="mode must be 'vector' when searching by embedding"):
        kg.semantic_nodes(embedding=[0.0] * DIM, mode="hybrid")
    kg.close()


def test_graph_default_mode_is_vector(tmp_path):
    """Omitting mode equals mode='vector': the default graph behavior is unchanged."""

    kg = _kg(tmp_path)
    carrier = _seed_carrier_and_distractors(kg)
    default_ids = [node.id for _score, node in kg.semantic_nodes("E1234", k=3)]
    vector_ids = [node.id for _score, node in kg.semantic_nodes("E1234", k=3, mode="vector")]
    assert default_ids == vector_ids
    assert carrier not in default_ids
    kg.close()
