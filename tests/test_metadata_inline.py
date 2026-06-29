"""Search returns redacted metadata inline on every hit.

The native core returns each hit's redacted metadata in the result row, so the
public search verbs surface byte-identical metadata without any per-hit by-id
lookup. These tests assert that metadata is present and correct across the text,
batch, and vector-in search paths.
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


def test_search_returns_inline_metadata(tmp_path):
    db = _db(tmp_path)
    db.add_many(
        [
            {"text": f"doc {i}", "id": f"d{i}", "metadata": {"topic": f"t{i % 3}", "n": i}}
            for i in range(20)
        ]
    )
    try:
        hits = db.search("doc", k=10)
        assert len(hits) == 10
        # metadata is present and correct (byte-identical to what was stored)
        by_id = {h.id: h.metadata for h in hits}
        for hit_id, meta in by_id.items():
            i = int(hit_id[1:])
            assert meta == {"topic": f"t{i % 3}", "n": str(i)}
    finally:
        db.close()


def test_search_many_returns_inline_metadata(tmp_path):
    db = _db(tmp_path)
    db.add_many([{"text": f"doc {i}", "id": f"d{i}", "metadata": {"k": "v"}} for i in range(10)])
    try:
        batches = db.search_many(["doc", "doc"], k=5)
        assert [len(b) for b in batches] == [5, 5]
        for batch in batches:
            for hit in batch:
                assert hit.metadata == {"k": "v"}
    finally:
        db.close()


def test_search_by_vector_returns_inline_metadata(tmp_path):
    db = _db(tmp_path)
    for i in range(0, 60, 10):
        db.add_vectors(_onehot(i), id=f"v{i}", metadata={"slot": str(i)})
    try:
        hits = db.search_by_vector(_onehot(20), k=3)
        assert hits[0].id == "v20"
        assert hits[0].metadata == {"slot": "20"}

        bm = db.search_many_by_vector([_onehot(0), _onehot(10)], k=1)
        assert [b[0].id for b in bm] == ["v0", "v10"]
        assert bm[0][0].metadata == {"slot": "0"}
    finally:
        db.close()
