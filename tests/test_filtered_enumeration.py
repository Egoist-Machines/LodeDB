"""Engine-side filtered enumeration and count (O(matches), not O(corpus)).

`list_documents(filter=)` and the new `count(filter=)` resolve the matching set
through the per-field planner instead of materializing every record and filtering
in Python. The engine also supports an `after`/`limit` keyset cursor for streaming.
"""

from __future__ import annotations

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend


def _db(path) -> LodeDB:
    return LodeDB(
        path=path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )


def _seed(db: LodeDB, n: int = 30) -> None:
    db.add_many(
        [
            {
                "text": f"doc {i}",
                "id": f"d{i:03d}",
                "metadata": {"topic": ["a", "b", "c"][i % 3], "year": 2000 + i},
            }
            for i in range(n)
        ]
    )


def test_count_total_and_filtered(tmp_path):
    db = _db(tmp_path)
    _seed(db, 30)
    assert db.count() == 30
    assert db.count(filter={"topic": "a"}) == 10
    assert db.count(filter={"year": {"$gte": 2020}}) == 10  # 2020..2029
    assert db.count(filter={"$or": [{"topic": "a"}, {"topic": "b"}]}) == 20
    assert db.count(filter={"topic": "nope"}) == 0


def test_count_matches_list_length(tmp_path):
    db = _db(tmp_path)
    _seed(db, 30)
    for flt in (
        {"topic": "b"},
        {"year": {"$gte": 2010, "$lt": 2015}},
        {"topic": {"$in": ["a", "c"]}},
        {"$not": {"topic": "a"}},
        {"document_ids": ["d000", "d001", "d002"], "metadata": {"topic": "a"}},
    ):
        assert db.count(filter=flt) == len(db.list_documents(filter=flt)), flt


def test_filtered_enumeration_is_complete(tmp_path):
    # The match set is returned in full regardless of corpus size (no k cap).
    db = _db(tmp_path)
    _seed(db, 30)
    a_ids = {r["id"] for r in db.list_documents(filter={"topic": "a"})}
    assert a_ids == {f"d{i:03d}" for i in range(30) if i % 3 == 0}
    assert len(a_ids) == 10  # a ranked search with default k would cap at 10 here too


def test_keyset_cursor_pages(tmp_path):
    db = _db(tmp_path)
    _seed(db, 10)
    # The public cursor pages by stable id order.
    page1 = db.list_documents(after=None, limit=4)
    ids1 = [r["id"] for r in page1]
    assert ids1 == ["d000", "d001", "d002", "d003"]
    page2 = db.list_documents(after=ids1[-1], limit=4)
    ids2 = [r["id"] for r in page2]
    assert ids2 == ["d004", "d005", "d006", "d007"]


def test_cursor_with_filter(tmp_path):
    db = _db(tmp_path)
    _seed(db, 30)
    flt = {"topic": "a"}  # d000, d003, d006, ...
    page = db.list_documents(filter=flt, after="d003", limit=2)
    assert [r["id"] for r in page] == ["d006", "d009"]
    # cursor is consistent with the full filtered set
    assert db.count(filter=flt) == 10


def test_count_filter_on_empty_store(tmp_path):
    db = _db(tmp_path)
    assert db.count() == 0
    assert db.count(filter={"topic": "a"}) == 0
