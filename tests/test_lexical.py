"""Unit tests for the stdlib-only lexical ranker and RRF fusion.

These exercise ``lodedb.engine._lexical`` in isolation: the tokenizer's handling
of code/serial/date shapes, BM25 ranking and corpus statistics, and the RRF
fusion order. No engine, embedding backend, or TurboVec dependency is involved.
"""

from __future__ import annotations

import pytest

from lodedb.engine._lexical import (
    Bm25Index,
    build_chunk_token_lists,
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


# -- from_token_lists -------------------------------------------------------


def test_from_token_lists_matches_text_constructor():
    """Building from pre-tokenized lists equals tokenizing the same texts."""

    texts = [
        "the turbine reported fault code e1234 overnight",
        "quick brown foxes and lazy dogs",
        "quarterly revenue and operating costs",
    ]
    unit_ids = ["u1", "u2", "u3"]
    from_text = Bm25Index(unit_ids, texts)
    from_tokens = Bm25Index.from_token_lists(unit_ids, [tokenize(t) for t in texts])
    # Identical rankings and scores for the same query: the two paths share the
    # postings-building logic, so a persisted index serves exactly the live one.
    assert from_tokens.rank("e1234 revenue") == from_text.rank("e1234 revenue")
    assert len(from_tokens) == len(from_text) == 3


def test_from_token_lists_length_mismatch_raises():
    """A unit_ids/token_lists length mismatch is rejected like the text path."""

    with pytest.raises(ValueError, match="same length"):
        Bm25Index.from_token_lists(["u1", "u2"], [["only", "one"]])


def test_from_token_lists_empty_is_safe():
    """An empty pre-tokenized corpus builds an empty, safe index."""

    index = Bm25Index.from_token_lists([], [])
    assert len(index) == 0
    assert index.rank("anything") == []


# -- incremental add/remove -------------------------------------------------


def _ranks_equal(left: Bm25Index, right: Bm25Index, queries) -> bool:
    """Returns True iff two indexes rank every query to identical ids and scores."""

    for query in queries:
        left_ranked = [(uid, round(score, 12)) for uid, score in left.rank(query)]
        right_ranked = [(uid, round(score, 12)) for uid, score in right.rank(query)]
        if left_ranked != right_ranked:
            return False
    return True


def test_incremental_add_matches_bulk_build():
    """Adding units one by one equals a bulk build over the same final set."""

    units = {
        "u1": "the turbine reported fault code e1234 overnight",
        "u2": "quick brown foxes and lazy dogs at noon",
        "u3": "quarterly revenue and operating costs with e1234",
        "u4": "common token common token rare token",
        "u5": "another e1234 mention for good measure here",
    }
    bulk = Bm25Index(list(units), list(units.values()))
    incremental = Bm25Index([], [])
    # Insert out of order to prove ranking does not depend on insertion order.
    for unit_id in ["u3", "u1", "u5", "u2", "u4"]:
        incremental.add_unit(unit_id, tokenize(units[unit_id]))
    assert len(incremental) == len(bulk) == 5
    assert _ranks_equal(
        bulk, incremental,
        ["e1234", "e1234 revenue", "common rare", "foxes", "turbine", "zzz"],
    )


def test_incremental_add_remove_reupsert_matches_bulk_build():
    """A churn of add/remove/no-op/re-upsert lands on the same index as a fresh build."""

    final = {
        "u1": "the turbine reported fault code e1234 overnight",
        "u2": "quick brown foxes and lazy dogs at noon",
        "u4": "common token common token rare token",
        "u5": "another e1234 mention for good measure here",
    }
    bulk = Bm25Index(list(final), list(final.values()))

    incremental = Bm25Index([], [])
    incremental.add_unit("u3", tokenize("quarterly revenue and operating costs e1234"))
    incremental.add_unit("u1", tokenize(final["u1"]))
    incremental.add_unit("u2", tokenize(final["u2"]))
    incremental.remove_unit("u3")  # remove a previously added unit
    incremental.add_unit("u4", tokenize(final["u4"]))
    incremental.add_unit("u5", tokenize("WRONG tokens that will be overwritten"))
    incremental.remove_unit("does-not-exist")  # no-op
    incremental.add_unit("u5", tokenize(final["u5"]))  # re-upsert replaces tokens

    assert len(incremental) == len(bulk) == 4
    assert incremental.unit_ids == bulk.unit_ids == frozenset(final)
    assert _ranks_equal(
        bulk, incremental,
        ["e1234", "e1234 revenue", "common rare", "foxes", "turbine", "measure", "zzz"],
    )


def test_incremental_remove_of_absent_id_is_noop():
    """Removing an id that was never added changes nothing."""

    index = Bm25Index(["u1", "u2"], ["alpha beta", "gamma delta"])
    before = index.rank("alpha gamma")
    index.remove_unit("never-added")
    assert len(index) == 2
    assert index.rank("alpha gamma") == before


def test_incremental_reupsert_replaces_not_accumulates():
    """Re-adding an id replaces its tokens rather than double-counting them."""

    replaced = Bm25Index([], [])
    replaced.add_unit("u1", tokenize("first version with alpha"))
    replaced.add_unit("u1", tokenize("second version with beta only"))
    # The unit no longer matches the old token and now matches the new one,
    # and its length/postings equal a fresh single-unit build of the new text.
    fresh = Bm25Index(["u1"], ["second version with beta only"])
    assert len(replaced) == 1
    assert _ranks_equal(fresh, replaced, ["alpha", "beta", "second", "version"])


def test_incremental_remove_drops_term_from_vocabulary():
    """Removing the last document carrying a term makes that term vanish (df=0)."""

    index = Bm25Index(["u1", "u2"], ["unique-zztoken here", "common words only"])
    assert [uid for uid, _ in index.rank("unique-zztoken")] == ["u1"]
    index.remove_unit("u1")
    # No document carries the term now, so it ranks to nothing exactly as a build
    # without u1 would (the term left the vocabulary, not merely emptied posting).
    rebuilt = Bm25Index(["u2"], ["common words only"])
    assert index.rank("unique-zztoken") == rebuilt.rank("unique-zztoken") == []


def test_incremental_position_is_stable_for_present_units():
    """A unit's stable position does not move when other units are added/removed."""

    index = Bm25Index(["keep"], ["alpha beta keep"])
    pos = index.position_of("keep")
    index.add_unit("other", tokenize("gamma delta"))
    index.remove_unit("other")
    index.add_unit("third", tokenize("epsilon"))
    assert index.position_of("keep") == pos  # unchanged across the churn
    assert index.position_of("missing") is None


# -- document groups (replace_group / remove_group) -------------------------


_GROUP_QUERIES = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "e1234",
    "abc-123",
    "alpha beta gamma",
    "zzz-absent",
]


def _bulk(units):
    """Builds a grouped bulk index from ``{group: {chunk_id: text}}``."""

    unit_ids: list[str] = []
    texts: list[str] = []
    group_ids: list[str] = []
    for group_id, chunks in units.items():
        for chunk_id, text in chunks.items():
            unit_ids.append(chunk_id)
            texts.append(text)
            group_ids.append(group_id)
    return Bm25Index(unit_ids, texts, group_ids=group_ids)


def test_default_group_is_unit_id():
    """A unit added with no explicit group is its own group, removable by id."""

    index = Bm25Index([], [])
    index.add_unit("u1", tokenize("alpha beta"))
    index.add_unit("u2", tokenize("gamma delta"))
    # Each unit is its own group, so removing the group named by its id drops it.
    index.remove_group("u1")
    assert index.unit_ids == frozenset({"u2"})
    assert [uid for uid, _ in index.rank("alpha")] == []


def test_replace_then_remove_group_ranks_like_fresh_bulk_build():
    """Bulk-build, then replace/remove whole document groups; rank equals a fresh build.

    The core invariant: an index built bulk with ``group_ids`` and then mutated
    with ``replace_group``/``remove_group`` ranks identically (ids and scores) to
    a fresh bulk build over the final unit set, regardless of how it got there.
    """

    base = {
        "docA": {"a#0": "alpha beta gamma", "a#1": "alpha alpha delta"},
        "docB": {"b#0": "gamma e1234 note", "b#1": "delta delta beta"},
        "docC": {"c#0": "epsilon only here", "c#1": "abc-123 serial line"},
    }
    index = _bulk(base)

    # Replace docA's two chunks with three differently-tokened, differently-id'd
    # chunks (a re-upsert whose chunk ids changed), drop docB entirely, and leave
    # docC untouched.
    index.replace_group(
        "docA",
        [
            ("a#7", tokenize("alpha rewritten")),
            ("a#8", tokenize("beta gamma e1234")),
            ("a#9", tokenize("delta epsilon")),
        ],
    )
    index.remove_group("docB")

    final = {
        "docA": {
            "a#7": "alpha rewritten",
            "a#8": "beta gamma e1234",
            "a#9": "delta epsilon",
        },
        "docC": {"c#0": "epsilon only here", "c#1": "abc-123 serial line"},
    }
    fresh = _bulk(final)

    assert index.unit_ids == fresh.unit_ids
    assert _ranks_equal(index, fresh, _GROUP_QUERIES)


def test_remove_group_is_noop_when_absent():
    """Removing a group that was never present changes nothing."""

    index = _bulk({"docA": {"a#0": "alpha beta"}})
    before = index.rank("alpha beta")
    index.remove_group("does-not-exist")
    assert len(index) == 1
    assert index.rank("alpha beta") == before


def test_replace_group_with_empty_units_drops_the_group():
    """``replace_group`` with no units removes the document's chunks."""

    index = _bulk({"docA": {"a#0": "alpha"}, "docB": {"b#0": "beta"}})
    index.replace_group("docA", [])
    assert index.unit_ids == frozenset({"b#0"})
    assert _ranks_equal(index, _bulk({"docB": {"b#0": "beta"}}), _GROUP_QUERIES)


def test_replace_group_on_incremental_index_matches_bulk():
    """Groups assigned via add_unit then replaced rank like a fresh grouped build."""

    index = Bm25Index([], [])
    index.add_unit("a#0", tokenize("alpha beta"), "docA")
    index.add_unit("a#1", tokenize("gamma"), "docA")
    index.add_unit("b#0", tokenize("delta epsilon"), "docB")
    index.replace_group("docA", [("a#2", tokenize("alpha gamma e1234"))])

    fresh = _bulk(
        {"docA": {"a#2": "alpha gamma e1234"}, "docB": {"b#0": "delta epsilon"}}
    )
    assert index.unit_ids == fresh.unit_ids
    assert _ranks_equal(index, fresh, _GROUP_QUERIES)


def test_group_ids_length_mismatch_raises():
    """A group_ids sequence that does not align with unit_ids is rejected."""

    with pytest.raises(ValueError, match="group_ids must align"):
        Bm25Index(["u1", "u2"], ["a", "b"], group_ids=["only-one"])


# -- build_chunk_token_lists ------------------------------------------------


def test_build_chunk_token_lists_zips_ids_with_tokens():
    """Per-document token lists flatten to chunk-aligned (ids, token_lists)."""

    document_tokens = {
        "doc-a": [["alpha", "beta"], ["gamma"]],
        "doc-b": [["delta"]],
    }
    document_chunk_ids = {
        "doc-a": ["a#0", "a#1"],
        "doc-b": ["b#0"],
    }
    chunk_ids, token_lists, group_ids = build_chunk_token_lists(
        document_tokens, document_chunk_ids
    )
    assert chunk_ids == ["a#0", "a#1", "b#0"]
    assert token_lists == [["alpha", "beta"], ["gamma"], ["delta"]]
    # The owning document id rides along per chunk so the index can group by it.
    assert group_ids == ["doc-a", "doc-a", "doc-b"]


def test_build_chunk_token_lists_skips_documents_without_tokens():
    """A document with no stored tokens contributes no chunks (defensive)."""

    document_tokens = {"doc-a": [["alpha"]]}
    document_chunk_ids = {"doc-a": ["a#0"], "doc-missing": ["m#0"]}
    chunk_ids, token_lists, group_ids = build_chunk_token_lists(
        document_tokens, document_chunk_ids
    )
    assert chunk_ids == ["a#0"]
    assert token_lists == [["alpha"]]
    assert group_ids == ["doc-a"]


def test_build_chunk_token_lists_aligns_on_shorter_on_mismatch():
    """A count mismatch aligns on the shorter side rather than mislabeling."""

    document_tokens = {"doc-a": [["alpha"], ["beta"], ["gamma"]]}
    document_chunk_ids = {"doc-a": ["a#0", "a#1"]}  # one fewer id than token lists
    chunk_ids, token_lists, group_ids = build_chunk_token_lists(
        document_tokens, document_chunk_ids
    )
    assert chunk_ids == ["a#0", "a#1"]
    assert token_lists == [["alpha"], ["beta"]]
    assert group_ids == ["doc-a", "doc-a"]


def test_build_chunk_token_lists_feeds_bm25():
    """The flattened output builds a working BM25 index over the chunk id space."""

    document_tokens = {"doc": [tokenize("fault code e1234 logged")]}
    document_chunk_ids = {"doc": ["doc#0"]}
    chunk_ids, token_lists, group_ids = build_chunk_token_lists(
        document_tokens, document_chunk_ids
    )
    index = Bm25Index.from_token_lists(chunk_ids, token_lists, group_ids=group_ids)
    assert [unit_id for unit_id, _ in index.rank("e1234")] == ["doc#0"]


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
