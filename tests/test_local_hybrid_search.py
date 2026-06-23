"""Tests for hybrid (BM25 + RRF) retrieval over the local LodeDB SDK.

These use the deterministic hash embedding backend, which embeds the whole
chunk text and therefore cannot "see" an exact token buried in unrelated prose.
That is exactly the motivating case: a pure-vector query for an error code,
serial, or date ranks the carrying document low or misses it, while hybrid
surfaces it in the top-k via the lexical ranker.
"""

from __future__ import annotations

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local import LodeDB


def _open(tmp_path, *, store_text: bool = True, dim: int = 384) -> LodeDB:
    """Opens a LodeDB with an injected deterministic hash backend."""

    return LodeDB(
        path=tmp_path,
        store_text=store_text,
        _embedding_backend=HashEmbeddingBackend(native_dim=dim),
    )


def _seed_with_exact_token(db: LodeDB, *, token: str, topic: str = "ops") -> str:
    """Adds one document carrying ``token`` in its body plus noisy distractors.

    Returns the id of the carrying document. The distractors share no token with
    the code/serial/date, so a lexical ranker isolates the carrier cleanly while
    the hash embedding spreads everything roughly uniformly.
    """

    carrier = db.add(
        "The overnight maintenance log records that the auxiliary turbine tripped "
        f"and the controller reported {token} before the unit recovered.",
        id="carrier",
        metadata={"topic": topic},
    )
    db.add(
        "Quick brown foxes and lazy dogs wander the meadow at noon under a warm sky.",
        id="distractor-animals",
        metadata={"topic": "animals"},
    )
    db.add(
        "Quarterly revenue grew while operating costs declined across every region.",
        id="distractor-finance",
        metadata={"topic": "finance"},
    )
    for i in range(12):
        db.add(
            f"General notes number {i} covering miscellaneous unrelated topics and asides.",
            id=f"filler-{i}",
            metadata={"topic": "misc"},
        )
    return carrier


# -- motivating regression cases -------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["E1234", "ABC-123", "2024-01-15"],
    ids=["error-code", "hyphenated-serial", "iso-date"],
)
def test_hybrid_surfaces_exact_token_vector_misses(tmp_path, token):
    """Pure vector misses/under-ranks an exact token; hybrid surfaces it top-k.

    Covers the error-code, hyphenated-serial, and date acceptance cases in one
    parametrized regression.
    """

    db = _open(tmp_path)
    carrier = _seed_with_exact_token(db, token=token)

    vector_ids = [hit.id for hit in db.search(token, k=3)]
    hybrid_ids = [hit.id for hit in db.search(token, k=3, mode="hybrid")]

    # The hash embedding does not encode the literal token, so pure vector does
    # not put the carrier first (it is essentially random over the corpus).
    assert vector_ids[0] != carrier or carrier not in vector_ids[:1]
    # Hybrid surfaces the carrier in the top-k, and as the top hit (it is the
    # only document containing the token, so BM25 ranks it first).
    assert carrier in hybrid_ids
    assert hybrid_ids[0] == carrier
    db.close()


def test_lexical_only_isolates_the_carrier(tmp_path):
    """mode='lexical' returns only documents that actually contain the token."""

    db = _open(tmp_path)
    carrier = _seed_with_exact_token(db, token="E1234")
    hits = db.search("E1234", k=5, mode="lexical")
    assert [hit.id for hit in hits] == [carrier]
    db.close()


# -- filter interaction -----------------------------------------------------


def test_hybrid_honors_metadata_allowlist(tmp_path):
    """A filtered hybrid query equals the hybrid of the filtered subset.

    The carrier is the only doc containing the token, so filtering to a topic it
    is not in must exclude it from both rankers.
    """

    db = _open(tmp_path)
    carrier = _seed_with_exact_token(db, token="E1234", topic="ops")

    # Filter to the topic the carrier carries: it survives and ranks first.
    in_topic = [hit.id for hit in db.search("E1234", k=5, mode="hybrid", filter={"topic": "ops"})]
    assert carrier in in_topic
    assert all(hit_meta == "ops" for hit_meta in _topics(db, in_topic))

    # Filter to a topic the carrier is NOT in: it must be excluded from both
    # the lexical and the vector ranker, so it never appears.
    out_topic = [
        hit.id for hit in db.search("E1234", k=5, mode="hybrid", filter={"topic": "animals"})
    ]
    assert carrier not in out_topic
    assert all(hit_meta == "animals" for hit_meta in _topics(db, out_topic))
    db.close()


def test_hybrid_honors_predicate_allowlist(tmp_path):
    """A predicate filter (compiled-scan allowlist path) constrains hybrid too."""

    db = LodeDB(path=tmp_path, _embedding_backend=HashEmbeddingBackend(native_dim=384))
    db.add("incident report references fault E1234", id="c2020", metadata={"year": 2020})
    db.add("a second incident with the same E1234 fault", id="c2023", metadata={"year": 2023})
    db.add("unrelated summary about logistics", id="other", metadata={"year": 2019})

    hits = db.search("E1234", k=10, mode="hybrid", filter={"year": {"$gte": 2021}})
    ids = {hit.id for hit in hits}
    assert ids == {"c2023"}  # only the >=2021 doc that also matches lexically
    db.close()


def test_hybrid_filter_no_matches_returns_empty(tmp_path):
    """A filter that matches no documents yields no hybrid hits."""

    db = _open(tmp_path)
    _seed_with_exact_token(db, token="E1234")
    hits = db.search("E1234", k=5, mode="hybrid", filter={"topic": "nonexistent"})
    assert hits == []
    db.close()


# -- batch parity -----------------------------------------------------------


def test_search_many_hybrid_matches_repeated_single(tmp_path):
    """search_many(mode='hybrid') equals repeated single search(mode='hybrid')."""

    db = _open(tmp_path)
    _seed_with_exact_token(db, token="E1234")
    db.add("the second device used serial ABC-123 in its label", id="serial-doc")

    queries = ["E1234", "ABC-123", "foxes"]
    batched = db.search_many(queries, k=3, mode="hybrid")
    singles = [db.search(query, k=3, mode="hybrid") for query in queries]

    assert [[hit.id for hit in row] for row in batched] == [
        [hit.id for hit in row] for row in singles
    ]
    # Scores match too (RRF is deterministic), confirming true parity.
    for batch_row, single_row in zip(batched, singles, strict=True):
        assert [round(hit.score, 9) for hit in batch_row] == [
            round(hit.score, 9) for hit in single_row
        ]
    db.close()


def test_search_many_hybrid_with_filter_matches_single(tmp_path):
    """Filtered hybrid batch equals filtered repeated single search."""

    db = _open(tmp_path)
    _seed_with_exact_token(db, token="E1234", topic="ops")
    queries = ["E1234", "turbine"]
    batched = db.search_many(queries, k=5, mode="hybrid", filter={"topic": "ops"})
    singles = [db.search(query, k=5, mode="hybrid", filter={"topic": "ops"}) for query in queries]
    assert [[hit.id for hit in row] for row in batched] == [
        [hit.id for hit in row] for row in singles
    ]
    db.close()


def test_search_many_hybrid_large_batch_matches_single_and_batches(tmp_path):
    """A larger hybrid batch equals repeated single search (ids + scores) and batches.

    Exercises the batched-vector hybrid path: several hybrid queries share the
    (empty) filter, so their vector component runs through one grouped scan and
    each fuses on the CPU. The result must still be byte-for-byte the repeated
    single-query result, and the batched group path must actually have run.
    """

    db = _open(tmp_path)
    _seed_with_exact_token(db, token="E1234")
    db.add("the second device used serial ABC-123 in its label", id="serial-doc")
    db.add("log mentions both E1234 and ABC-123 on 2024-01-15", id="multi")

    engine = db._engine  # noqa: SLF001 - test introspection of the batched path
    calls = {"batched_group": 0}
    original = engine._run_batched_hybrid_group  # noqa: SLF001

    def _spy(*args, **kwargs):
        calls["batched_group"] += 1
        return original(*args, **kwargs)

    engine._run_batched_hybrid_group = _spy  # noqa: SLF001
    try:
        queries = ["E1234", "ABC-123", "foxes", "2024-01-15", "turbine recovered"]
        batched = db.search_many(queries, k=4, mode="hybrid")
    finally:
        engine._run_batched_hybrid_group = original  # noqa: SLF001
    singles = [db.search(query, k=4, mode="hybrid") for query in queries]

    assert [[hit.id for hit in row] for row in batched] == [
        [hit.id for hit in row] for row in singles
    ]
    for batch_row, single_row in zip(batched, singles, strict=True):
        assert [round(hit.score, 9) for hit in batch_row] == [
            round(hit.score, 9) for hit in single_row
        ]
    # The grouped batched-vector hybrid path actually ran (not the per-query path).
    assert calls["batched_group"] == 1
    db.close()


def test_search_many_mixed_modes_matches_single(tmp_path):
    """A batch mixing vector, hybrid, and lexical queries matches repeated singles."""

    db = _open(tmp_path)
    _seed_with_exact_token(db, token="E1234")
    db.add("the second device used serial ABC-123 in its label", id="serial-doc")

    specs = [
        ("E1234", "hybrid"),
        ("turbine recovered", "vector"),
        ("ABC-123", "hybrid"),
        ("E1234", "lexical"),
        ("foxes", "vector"),
    ]
    queries = [text for text, _ in specs]
    # search_many takes one mode for the batch, so compare per-mode batches to the
    # single calls of the same mode at the same positions.
    for mode in ("hybrid", "vector", "lexical"):
        batched = db.search_many(queries, k=4, mode=mode)
        singles = [db.search(query, k=4, mode=mode) for query in queries]
        assert [[hit.id for hit in row] for row in batched] == [
            [hit.id for hit in row] for row in singles
        ], mode
    db.close()


# -- store_text=False -------------------------------------------------------


@pytest.mark.parametrize("mode", ["hybrid", "lexical"])
def test_hybrid_requires_store_text(tmp_path, mode):
    """A lexical/hybrid query on a store_text=False handle raises a clear error."""

    db = _open(tmp_path, store_text=False)
    db.add("a document that mentions E1234 somewhere in its text")
    with pytest.raises(ValueError, match="store_text=True"):
        db.search("E1234", k=5, mode=mode)
    with pytest.raises(ValueError, match="store_text=True"):
        db.search_many(["E1234"], k=5, mode=mode)
    db.close()


def test_invalid_mode_raises(tmp_path):
    """An unknown mode is rejected with a clear message listing the valid modes."""

    db = _open(tmp_path)
    db.add("hello world")
    with pytest.raises(ValueError, match="mode must be one of"):
        db.search("hello", mode="nonsense")
    db.close()


# -- default mode is unchanged ----------------------------------------------


def test_default_mode_is_vector_and_unchanged(tmp_path):
    """Omitting mode and passing mode='vector' produce identical vector results."""

    db = _open(tmp_path)
    _seed_with_exact_token(db, token="E1234")
    default_hits = db.search("turbine recovered", k=5)
    explicit_hits = db.search("turbine recovered", k=5, mode="vector")
    assert [hit.id for hit in default_hits] == [hit.id for hit in explicit_hits]
    assert [hit.score for hit in default_hits] == [hit.score for hit in explicit_hits]
    db.close()


def test_hybrid_index_rebuilds_after_mutation(tmp_path):
    """The lexical index is generation-keyed: a new doc is searchable next query."""

    db = _open(tmp_path)
    _seed_with_exact_token(db, token="E1234")
    assert db.search("ABC-123", k=5, mode="lexical") == []  # not yet present
    db.add("replacement part labeled ABC-123 installed", id="late", metadata={"topic": "ops"})
    late_hits = db.search("ABC-123", k=5, mode="lexical")
    assert [hit.id for hit in late_hits] == ["late"]
    db.close()


def test_hybrid_top_k_not_capped_by_vector_pool(tmp_path):
    """Fused top-k can include lexical-only hits beyond a narrow vector top-k."""

    db = _open(tmp_path)
    # Two docs carry the token; many distractors do not. A naive k=2 vector-only
    # pool could miss them, but the widened pool + lexical ranker recovers both.
    db.add("first log entry citing fault E1234 on the line", id="c1")
    db.add("second unrelated-looking note that also mentions E1234", id="c2")
    for i in range(20):
        db.add(f"noise document {i} with no codes at all", id=f"n{i}", metadata={"topic": "misc"})
    ids = {hit.id for hit in db.search("E1234", k=5, mode="hybrid")}
    assert {"c1", "c2"} <= ids
    db.close()


# -- incremental in-memory BM25 index ---------------------------------------


def _open_lexical(tmp_path, *, source: str):
    """Opens a DB whose lexical source is the raw-text path or the token path."""

    if source == "store_text":
        return LodeDB(
            path=tmp_path, store_text=True,
            _embedding_backend=HashEmbeddingBackend(native_dim=384),
        )
    return LodeDB(
        path=tmp_path, index_text=True, store_text=False,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )


@pytest.mark.parametrize("source", ["store_text", "index_text"])
def test_lexical_index_incremental_add_is_searchable_without_full_rebuild(tmp_path, source):
    """A small add after a warm lexical index folds in incrementally and is found."""

    db = _open_lexical(tmp_path, source=source)
    for i in range(20):
        db.add(f"base body number {i} with token{i}", id=f"b{i}", metadata={"topic": "base"})
    # Warm the lexical index (full build on this first query).
    assert db.search("token1", k=3, mode="lexical")

    import lodedb.engine._lexical as lx

    builds = {"count": 0}
    original_build = lx.Bm25Index._build

    def _spy_build(self, *args, **kwargs):
        builds["count"] += 1
        return original_build(self, *args, **kwargs)

    lx.Bm25Index._build = _spy_build
    try:
        db.add("a freshly added doc carrying ABC-123 serial", id="late", metadata={"topic": "ops"})
        late = db.search("ABC-123", k=5, mode="lexical")
    finally:
        lx.Bm25Index._build = original_build

    assert [hit.id for hit in late] == ["late"]
    # The small add did NOT trigger a full O(total tokens) rebuild.
    assert builds["count"] == 0
    db.close()


@pytest.mark.parametrize("source", ["store_text", "index_text"])
def test_lexical_index_incremental_delete_drops_doc(tmp_path, source):
    """Deleting a doc removes it from the in-memory index on the next query."""

    db = _open_lexical(tmp_path, source=source)
    for i in range(20):
        db.add(f"base body number {i} with token{i}", id=f"b{i}")
    db.add("removable entry citing E1234 fault", id="gone")
    assert [hit.id for hit in db.search("E1234", k=5, mode="lexical")] == ["gone"]
    assert db.remove("gone") is True
    assert db.search("E1234", k=5, mode="lexical") == []
    db.close()


@pytest.mark.parametrize("source", ["store_text", "index_text"])
def test_lexical_incremental_served_equals_full_rebuild(tmp_path, source):
    """The incrementally-served ranking equals what a forced full rebuild gives.

    Build a base, warm the index, then add and remove docs (incremental folds),
    capture the served hybrid ranking, force a full rebuild by clearing the
    in-memory index cache, and assert the rebuilt ranking matches ids and scores.
    """

    db = _open_lexical(tmp_path, source=source)
    for i in range(16):
        db.add(f"base body number {i} with token{i}", id=f"b{i}")
    db.search("token0", k=3, mode="lexical")  # warm

    db.add("first carrier citing fault E1234 here", id="c1")
    db.add("second note also mentioning ABC-123 serial", id="c2")
    db.remove("b0")
    db.add("third upserted body for token3 again", id="b3")  # re-upsert existing id

    query = "E1234 ABC-123 token3 token5"
    served = [(hit.id, round(hit.score, 9)) for hit in db.search(query, k=10, mode="hybrid")]

    # Force a full rebuild: drop the cached (incrementally updated) index so the
    # next query builds it from scratch over the same final corpus.
    db._engine._lexical_indexes.clear()  # noqa: SLF001 - force a full rebuild
    rebuilt = [(hit.id, round(hit.score, 9)) for hit in db.search(query, k=10, mode="hybrid")]

    assert served == rebuilt
    db.close()


def test_lexical_incremental_large_delta_full_rebuilds(tmp_path):
    """A delta above the incremental fraction falls back to a full rebuild."""

    db = _open_lexical(tmp_path, source="store_text")
    for i in range(8):
        db.add(f"doc {i} alpha token{i}", id=f"d{i}")
    db.search("alpha", k=3, mode="lexical")  # warm

    import lodedb.engine._lexical as lx

    builds = {"count": 0}
    original_build = lx.Bm25Index._build

    def _spy_build(self, *args, **kwargs):
        builds["count"] += 1
        return original_build(self, *args, **kwargs)

    lx.Bm25Index._build = _spy_build
    try:
        # Add 5 docs to a corpus of 8 -> 5/13 > 0.25 -> a full rebuild is taken.
        for i in range(5):
            db.add(f"new doc {i} beta token{i}", id=f"e{i}")
        db.search("beta", k=3, mode="lexical")
    finally:
        lx.Bm25Index._build = original_build

    assert builds["count"] >= 1  # a full rebuild ran for the large delta
    db.close()


# -- helpers ----------------------------------------------------------------


def _topics(db: LodeDB, ids: list[str]) -> list[str]:
    """Returns the stored ``topic`` metadata for a list of hit ids."""

    out: list[str] = []
    for hit_id in ids:
        record = db._index.get_document(hit_id)  # noqa: SLF001 - test introspection
        out.append(record.get("metadata", {}).get("topic", ""))
    return out
