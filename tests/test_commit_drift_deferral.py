"""Commit path stays O(changed): no per-commit search/repack; drift deferred to query.

The TurboVec SIMD "blocked" layout is rebuilt lazily by the first search after a
mutation. The engine therefore must not touch it on the commit path: it does not
eagerly ``prepare()``, and it defers the quantization-drift self-score sample
(which needs a search) to the next query that warms the layout. The transient
full embeddings are dropped O(changed). These tests pin that behavior.
"""

from __future__ import annotations

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.turbovec_index import TurboVecServingIndex
from lodedb.local.db import LodeDB


def _be() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=384)


def _open(path) -> LodeDB:
    return LodeDB(path=path, model="minilm", _embedding_backend=_be())


def test_commit_does_no_search_and_defers_drift_to_query(tmp_path, monkeypatch):
    """A commit performs no search; drift is buffered and sampled on the next query."""

    db = _open(tmp_path)
    db.add_many([{"text": f"base doc {i}", "id": f"b{i}"} for i in range(50)])
    db.search("base", k=3)  # warm the layout
    engine = db._engine
    key = next(iter(engine._index_generations))

    calls = {"n": 0}
    original = TurboVecServingIndex.search

    def counting_search(self, *args, **kwargs):
        calls["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TurboVecServingIndex, "search", counting_search)

    # A single-doc commit must not search (no per-commit drift sample / repack).
    calls["n"] = 0
    db.add("a brand new memory row", id="new")
    assert calls["n"] == 0, "commit path must not search"
    assert engine._pending_drift_samples.get(key), "new row should be buffered for drift"

    # The next query warms the layout and consumes the buffered drift sample.
    assert db.search("memory", k=3) is not None
    assert calls["n"] > 0
    assert not engine._pending_drift_samples.get(key), "drift buffer should be consumed"
    assert key in engine._turbovec_drift_telemetry, "drift sampled at query time"
    db.close()


def test_search_correct_after_deferred_commit(tmp_path):
    """The lazily-rebuilt layout returns correct results after a commit."""

    db = _open(tmp_path)
    db.add("the quick brown fox", id="fox")
    db.add("a lazy sleeping dog", id="dog")  # layout invalidated, not rebuilt on commit
    hits = db.search("fox", k=2)  # forces the lazy rebuild
    assert {hit.id for hit in hits} == {"fox", "dog"}
    db.close()


def test_incremental_commits_persist_intact(tmp_path):
    """O(changed) transient discard keeps every row served and durable on reopen."""

    db = _open(tmp_path)
    db.add_many([{"text": f"doc {i}", "id": f"d{i}"} for i in range(20)])
    for i in range(20, 45):
        db.add(f"doc {i}", id=f"d{i}")  # incremental commits
    assert db.count() == 45
    db.close()

    reopened = _open(tmp_path)
    try:
        assert reopened.count() == 45
        assert reopened.search("doc", k=5)  # serves correctly after reopen
    finally:
        reopened.close()
