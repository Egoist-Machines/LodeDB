"""Unit tests for the stdlib-only lexical ranker and RRF fusion.

These exercise ``lodedb.engine._lexical`` in isolation: the tokenizer's handling
of code/serial/date shapes, BM25 ranking and corpus statistics, and the RRF
fusion order. No engine, embedding backend, or TurboVec dependency is involved.
"""

from __future__ import annotations

from lodedb.engine._lexical import (
    Bm25Index,
    fuse_unit_rankings,
    reciprocal_rank_fusion,
    tokenize,
)

# -- tokenizer --------------------------------------------------------------


def test_tokenize_lowercases_and_splits_on_whitespace():
    """Plain prose lowercases and splits into word tokens."""

    assert tokenize("The Quick Brown Fox") == ["the", "quick", "brown", "fox"]


def test_tokenize_preserves_error_code():
    """An alphanumeric error code survives as one token, lowercased."""

    assert tokenize("fault code E1234 reported") == ["fault", "code", "e1234", "reported"]


def test_tokenize_strips_trailing_sentence_punctuation_from_code():
    """Trailing punctuation is a boundary, so a code at end of sentence stays whole."""

    assert tokenize("the failure was E1234.") == ["the", "failure", "was", "e1234"]
    assert tokenize("(E1234)") == ["e1234"]


def test_tokenize_preserves_hyphenated_serial():
    """A hyphenated serial keeps its interior hyphens as a single token."""

    assert tokenize("serial ABC-123 shipped") == ["serial", "abc-123", "shipped"]
    assert tokenize("part ABC-123-X") == ["part", "abc-123-x"]


def test_tokenize_preserves_iso_date():
    """An ISO date keeps its hyphens and is a single findable token."""

    assert tokenize("logged on 2024-01-15 today") == ["logged", "on", "2024-01-15", "today"]


def test_tokenize_preserves_dotted_and_slashed_codes():
    """Interior dots and slashes inside a run are preserved (versions, paths)."""

    assert tokenize("upgrade to v1.2.3") == ["upgrade", "to", "v1.2.3"]
    assert tokenize("ratio a/b here") == ["ratio", "a/b", "here"]


def test_tokenize_doubled_separators_are_boundaries():
    """Doubled or leading/trailing separators split rather than glue."""

    assert tokenize("a--b") == ["a", "b"]
    assert tokenize("-E1234-") == ["e1234"]
    assert tokenize("x... y") == ["x", "y"]


def test_tokenize_empty_and_punctuation_only():
    """Empty or punctuation-only input yields no tokens."""

    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("!!! ??? ...") == []


# -- BM25 -------------------------------------------------------------------


def test_bm25_ranks_unit_with_query_term_first():
    """A unit containing the query term outranks one that does not."""

    index = Bm25Index(
        ["u1", "u2", "u3"],
        [
            "the turbine reported fault code e1234 overnight",
            "quick brown foxes and lazy dogs",
            "quarterly revenue and operating costs",
        ],
    )
    ranked = index.rank("e1234")
    assert ranked, "expected at least one lexical match"
    assert ranked[0][0] == "u1"
    # Units that share no term with the query are never ranked.
    assert {unit_id for unit_id, _ in ranked} == {"u1"}


def test_bm25_only_returns_positive_scores():
    """A zero-overlap query returns nothing even on a tiny corpus."""

    index = Bm25Index(["u1", "u2"], ["alpha beta", "gamma delta"])
    assert index.rank("epsilon") == []


def test_bm25_rare_term_outweighs_common_term():
    """A rarer term contributes more (higher IDF) than a corpus-wide one."""

    texts = ["common token", "common token", "common rare"]
    index = Bm25Index(["a", "b", "c"], texts)
    # Unit c has both 'common' (df=3) and 'rare' (df=1); a/b have only 'common'.
    ranked = dict(index.rank("common rare"))
    assert ranked["c"] > ranked["a"]


def test_bm25_respects_allowed_indices():
    """An allowlist of positions restricts which units BM25 may score."""

    index = Bm25Index(
        ["u0", "u1", "u2"],
        ["match here", "match again", "no overlap"],
    )
    # Restrict to position 1 only; u0 must not appear even though it matches.
    ranked = index.rank("match", allowed_indices={1})
    assert [unit_id for unit_id, _ in ranked] == ["u1"]


def test_bm25_limit_caps_results():
    """``limit`` caps the number of ranked units returned."""

    index = Bm25Index(
        ["u0", "u1", "u2"],
        ["match one", "match two", "match three"],
    )
    assert len(index.rank("match", limit=2)) == 2


def test_bm25_empty_index_is_safe():
    """An empty corpus ranks to an empty list without error."""

    index = Bm25Index([], [])
    assert len(index) == 0
    assert index.rank("anything") == []


def test_bm25_deterministic_tie_break_on_unit_id():
    """Equal-scoring units break ties on unit id for a stable order."""

    index = Bm25Index(["b", "a"], ["solo", "solo"])
    ranked = index.rank("solo")
    assert [unit_id for unit_id, _ in ranked] == ["a", "b"]


# -- RRF --------------------------------------------------------------------


def test_rrf_known_rankings_produce_known_order():
    """A worked RRF example fuses to the hand-computed order (c=60)."""

    vector = ["A", "B", "C"]
    lexical = ["B", "C", "D"]
    fused = reciprocal_rank_fusion((vector, lexical), c=60)
    # Scores: B = 1/62 + 1/61; C = 1/63 + 1/62; A = 1/61; D = 1/63.
    order = [unit_id for unit_id, _ in fused]
    assert order[0] == "B"
    assert order[1] == "C"
    # A (1/61 ~= 0.016393) edges out D (1/63 ~= 0.015873).
    assert order[2] == "A"
    assert order[3] == "D"


def test_rrf_scores_match_closed_form():
    """RRF scores equal Σ 1/(c+rank) over the rankers that returned the id."""

    fused = dict(reciprocal_rank_fusion((["A", "B"], ["B", "A"]), c=60))
    expected_a = 1 / 61 + 1 / 62
    expected_b = 1 / 62 + 1 / 61
    assert abs(fused["A"] - expected_a) < 1e-12
    assert abs(fused["B"] - expected_b) < 1e-12


def test_rrf_single_ranker_preserves_order():
    """A single ranker fuses to its own order (monotonic in rank)."""

    fused = [unit_id for unit_id, _ in reciprocal_rank_fusion((["x", "y", "z"],))]
    assert fused == ["x", "y", "z"]


def test_rrf_dedupes_within_a_ranker_keeping_best_rank():
    """A repeated id within one ranker keeps its first (best) rank only."""

    fused = dict(reciprocal_rank_fusion((["A", "A", "B"],), c=60))
    assert abs(fused["A"] - 1 / 61) < 1e-12  # rank 1, not 1/61 + 1/62


def test_rrf_tie_breaks_on_id():
    """Equal fused scores break ties on the id string."""

    fused = [unit_id for unit_id, _ in reciprocal_rank_fusion((["b"], ["a"]), c=60)]
    assert fused == ["a", "b"]


def test_rrf_weights_bias_a_ranker():
    """A heavier weight lifts the ranker's top id above the other's."""

    # Equal weights tie A and B at rank 1; weighting the second ranker breaks it.
    fused = dict(
        reciprocal_rank_fusion((["A"], ["B"]), c=60, weights=(1.0, 5.0))
    )
    assert fused["B"] > fused["A"]


def test_fuse_unit_rankings_returns_ids_only():
    """The two-ranker convenience returns just the fused id order."""

    assert fuse_unit_rankings(["A", "B"], ["B", "C"]) == ["B", "A", "C"]
