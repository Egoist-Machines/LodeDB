"""Search hydrates metadata inline, with no per-hit get_document N+1.

The engine inlines redacted metadata in result rows when the query opts in via
include=("metadata",), which all search verbs now do. So `_hits_from_result_rows`
reads row["metadata"] and never issues a per-hit get_document, while still
returning byte-identical metadata.
"""

from __future__ import annotations

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend

DIM = 384


def _db(path) -> LodeDB:
    return LodeDB(
        path=path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=DIM)
    )


def _onehot(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


class _CountingIndex:
    """Wraps a LodeIndex, counting get_document calls (to detect the N+1)."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.get_document_calls = 0

    def get_document(self, document_id):
        self.get_document_calls += 1
        return self._inner.get_document(document_id)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_search_inlines_metadata_without_get_document(tmp_path):
    db = _db(tmp_path)
    db.add_many(
        [
            {"text": f"doc {i}", "id": f"d{i}", "metadata": {"topic": f"t{i % 3}", "n": i}}
            for i in range(20)
        ]
    )
    counter = _CountingIndex(db._index)
    db._index = counter

    hits = db.search("doc", k=10)
    assert len(hits) == 10
    assert counter.get_document_calls == 0  # no N+1
    # metadata is present and correct (byte-identical to what was stored)
    by_id = {h.id: h.metadata for h in hits}
    for hit_id, meta in by_id.items():
        i = int(hit_id[1:])
        assert meta == {"topic": f"t{i % 3}", "n": str(i)}


def test_search_many_inlines_metadata(tmp_path):
    db = _db(tmp_path)
    db.add_many([{"text": f"doc {i}", "id": f"d{i}", "metadata": {"k": "v"}} for i in range(10)])
    counter = _CountingIndex(db._index)
    db._index = counter

    batches = db.search_many(["doc", "doc"], k=5)
    assert [len(b) for b in batches] == [5, 5]
    assert counter.get_document_calls == 0
    for batch in batches:
        for hit in batch:
            assert hit.metadata == {"k": "v"}


def test_search_by_vector_inlines_metadata(tmp_path):
    db = _db(tmp_path)
    for i in range(0, 60, 10):
        db.add_vectors(_onehot(i), id=f"v{i}", metadata={"slot": str(i)})
    counter = _CountingIndex(db._index)
    db._index = counter

    hits = db.search_by_vector(_onehot(20), k=3)
    assert hits[0].id == "v20"
    assert hits[0].metadata == {"slot": "20"}
    assert counter.get_document_calls == 0

    bm = db.search_many_by_vector([_onehot(0), _onehot(10)], k=1)
    assert [b[0].id for b in bm] == ["v0", "v10"]
    assert counter.get_document_calls == 0


def test_hits_fall_back_to_get_document_when_metadata_absent(tmp_path):
    # Safety net: a result row without inlined metadata still resolves it by id.
    db = _db(tmp_path)
    db.add("hello", id="a", metadata={"topic": "x"})
    rows = [{"document_id": "a", "score": 0.9}]  # no "metadata" key
    hits = db._hits_from_result_rows(rows)
    assert hits[0].id == "a"
    assert hits[0].metadata == {"topic": "x"}  # fetched via get_document fallback
