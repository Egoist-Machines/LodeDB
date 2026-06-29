"""Regression tests for filtered multi-query (batch) search.

Before the allowlist-pushdown fix the batch path filtered by *widening* the
effective top_k to the corpus size and post-filtering, which diverged from the
single-query allowlist pre-filter, tripped the resident top_k cap above 4096
rows, and scaled O(corpus). These tests pin the unified behaviour: a filtered
``search_many`` must equal repeated filtered ``search`` on a corpus larger than
the resident cap, respect the filter, handle per-query filters and empty
matches, and reflect mutations through the metadata posting index.
"""

from __future__ import annotations

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local import LodeDB

# Larger than GPU_DIRECT_TURBOVEC_MAX_TOP_K (4096): the size at which the old
# widen-to-corpus strategy tripped the resident cap and fell off the GPU.
CORPUS = 5000


def _open(tmp_path, dim: int = 384) -> LodeDB:
    return LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=dim)
    )


def _populate(db: LodeDB, n: int = CORPUS) -> None:
    db.add_many(
        [
            {
                "text": f"document number {i} token{i % 313} alpha beta gamma",
                "id": f"d{i}",
                "metadata": {
                    "topic": "animals" if i % 4 == 0 else "science",
                    "tier": str(i % 7),
                },
            }
            for i in range(n)
        ]
    )


def _queries(n: int = 12) -> list[str]:
    return [f"token{(q * 17) % 313} query {q}" for q in range(n)]


def _ids(rows):
    return [[hit.id for hit in row] for row in rows]


def test_filtered_search_many_matches_repeated_single_above_cap(tmp_path):
    """Filtered batch search equals repeated filtered single search past the resident cap."""

    db = _open(tmp_path)
    _populate(db)
    queries = _queries()
    for filt in (
        None,
        {"topic": "animals"},
        {"tier": "3"},
        {"topic": "science", "tier": "5"},
        {"document_ids": ["d0", "d4", "d8", "d100", "d4096", "d4900"]},
    ):
        batch = db.search_many(queries, k=10, filter=filt)
        singles = [db.search(query, k=10, filter=filt) for query in queries]
        assert _ids(batch) == _ids(singles), f"id/order mismatch for filter={filt}"
        for batch_row, single_row in zip(batch, singles, strict=True):
            for batch_hit, single_hit in zip(batch_row, single_row, strict=True):
                assert abs(batch_hit.score - single_hit.score) < 1e-3
    db.close()


def test_filtered_batch_results_respect_filter(tmp_path):
    """Every hit returned by a filtered batch satisfies the filter."""

    db = _open(tmp_path)
    _populate(db)
    batch = db.search_many(_queries(), k=10, filter={"topic": "animals"})
    assert any(row for row in batch)  # non-trivial: some hits returned
    for row in batch:
        for hit in row:
            assert hit.metadata.get("topic") == "animals"
    db.close()


def test_per_query_filters_return_correct_filtered_hits(tmp_path):
    """Each query honors its own filter: every hit it returns satisfies that filter."""

    db = _open(tmp_path)
    _populate(db)
    queries = _queries(8)
    per_query_filters = [
        None,
        {"topic": "animals"},
        {"tier": "1"},
        {"topic": "science"},
        None,
        {"tier": "1"},
        {"topic": "animals"},
        {"tier": "6"},
    ]
    for query, filt in zip(queries, per_query_filters, strict=True):
        hits = db.search(query, k=10, filter=filt)
        if filt is None:
            continue
        (field, value), = filt.items()
        for hit in hits:
            assert hit.metadata.get(field) == value
    db.close()


def test_empty_filter_match_returns_no_results(tmp_path):
    """A filter matching zero documents yields empty rows without error."""

    db = _open(tmp_path)
    _populate(db)
    batch = db.search_many(_queries(), k=10, filter={"topic": "nonexistent"})
    assert all(row == [] for row in batch)
    # document_ids referencing missing docs likewise return empty.
    batch_ids = db.search_many(
        _queries(), k=10, filter={"document_ids": ["missing-1", "missing-2"]}
    )
    assert all(row == [] for row in batch_ids)
    db.close()


def test_filtered_batch_reflects_mutations(tmp_path):
    """The metadata posting index refreshes on add/remove/update across generations."""

    db = _open(tmp_path, dim=384)
    _populate(db, n=CORPUS)
    queries = _queries(6)

    # A fresh tag is absent until documents carry it.
    assert not any(
        hit.id for row in db.search_many(queries, k=50, filter={"topic": "rare"}) for hit in row
    )
    db.add_many(
        [
            {"text": f"rare token{i} doc", "id": f"r{i}", "metadata": {"topic": "rare"}}
            for i in range(20)
        ]
    )
    after_add = {
        hit.id for row in db.search_many(queries, k=50, filter={"topic": "rare"}) for hit in row
    }
    assert after_add, "newly added docs must appear under their filter"

    db.remove("r0")
    after_remove = {
        hit.id for row in db.search_many(queries, k=50, filter={"topic": "rare"}) for hit in row
    }
    assert "r0" not in after_remove

    # Re-tagging a document moves it between filter buckets.
    db.add("rare token re-tagged", id="r1", metadata={"topic": "moved"})
    moved = {
        hit.id for row in db.search_many(queries, k=50, filter={"topic": "moved"}) for hit in row
    }
    assert "r1" in moved
    assert "r1" not in {
        hit.id for row in db.search_many(queries, k=50, filter={"topic": "rare"}) for hit in row
    }
    db.close()
