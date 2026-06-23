"""The filter planner resolves a validated filter to the same set as
the per-document compiled matcher (the _predicate oracle), for every operator.

`_filter_plan.resolve` replaces the O(corpus) compiled-matcher scan on the search
allowlist path with set operations + bisect. This asserts exact parity against
`_predicate.compile_metadata_filter` (the source of truth for comparison
semantics) across the full grammar, so the planner can be trusted as the
single resolver for filtered search and enumeration.
"""

from __future__ import annotations

from lodedb.engine._filter_plan import build_field_indexes, resolve
from lodedb.engine._predicate import coerce_sdk_filter, compile_metadata_filter


def _corpus() -> dict[str, dict[str, str]]:
    """A small, varied corpus (values stored as strings, as the engine does).

    Exercises: missing fields, numeric and non-numeric values in the same field,
    booleans, and a range of years.
    """

    docs: dict[str, dict[str, str]] = {}
    topics = ["a", "b", "c"]
    for i in range(40):
        meta: dict[str, str] = {"year": str(2000 + i % 26)}
        if i % 4 != 0:  # some docs have no topic
            meta["topic"] = topics[i % 3]
        if i % 5 == 0:
            meta["flag"] = "true" if i % 2 == 0 else "false"
        # price: mostly numeric, a couple non-numeric, some missing
        if i % 3 == 0:
            meta["price"] = str(round(5 + (i % 7) * 1.5, 1))
        elif i % 7 == 0:
            meta["price"] = "free"
        docs[f"d{i}"] = meta
    return docs


# A broad set of filters (SDK form; coerced before use), spanning every operator,
# logical composition, missing fields, and the numeric/lexicographic ordered edges.
_FILTERS = [
    {"topic": "a"},
    {"topic": "missing-value"},
    {"topic": {"$eq": "b"}},
    {"topic": {"$ne": "a"}},  # includes docs missing 'topic'
    {"topic": {"$in": ["a", "c"]}},
    {"topic": {"$nin": ["a"]}},  # includes missing
    {"topic": {"$exists": True}},
    {"topic": {"$exists": False}},
    {"year": {"$gte": 2010}},
    {"year": {"$gt": 2010, "$lt": 2015}},
    {"year": {"$lte": 2005}},
    {"year": {"$ne": 2000}},
    {"flag": True},
    {"flag": {"$exists": False}},
    {"price": {"$gte": 9.9}},  # numeric operand vs mixed numeric/"free"
    {"price": {"$lt": 8}},
    {"price": {"$exists": True}},
    {"price": "free"},
    {"missingfield": {"$exists": False}},  # field absent from whole corpus
    {"missingfield": {"$ne": "x"}},  # absent field, $ne matches all
    {"missingfield": "x"},  # absent field, $eq matches none
    {"$and": [{"topic": "a"}, {"year": {"$gte": 2010}}]},
    {"$or": [{"topic": "b"}, {"year": {"$lt": 2003}}]},
    {"$not": {"topic": "a"}},
    {"$or": [{"$and": [{"topic": "c"}, {"flag": True}]}, {"year": {"$gte": 2024}}]},
    {"topic": "a", "year": {"$gte": 2005}},  # node-level AND of two fields
    {"$and": [{"$not": {"topic": "a"}}, {"price": {"$exists": True}}]},
]


def test_planner_matches_compiled_matcher_for_all_operators():
    docs = _corpus()
    fields, all_docs = build_field_indexes(docs)
    assert all_docs == set(docs)

    for raw in _FILTERS:
        coerced = coerce_sdk_filter(raw)
        planned = resolve(coerced, fields, all_docs)
        matcher = compile_metadata_filter(coerced)
        expected = {doc_id for doc_id, meta in docs.items() if matcher(meta)}
        assert planned == expected, f"planner != matcher for filter {raw!r}"


def test_planner_on_empty_corpus():
    fields, all_docs = build_field_indexes({})
    assert resolve(coerce_sdk_filter({"topic": "a"}), fields, all_docs) == set()
    # $ne / $exists False over an empty corpus is still empty (no docs at all)
    assert resolve(coerce_sdk_filter({"topic": {"$ne": "a"}}), fields, all_docs) == set()
