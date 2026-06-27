"""Tests for late-interaction (multi-vector / MaxSim) retrieval -- issue #25.

These run fully offline: documents and queries are supplied as precomputed patch
matrices (a deterministic fake encoder stands in for ColPali on the
encoder-convenience path), so no model is downloaded. They exercise the
one-row-per-document storage layout, the float16/float32/int8 precisions, the
exact-MaxSim retrieval paths (resident, filtered, streaming), durability across
reopen, metadata filtering, and the read-only / validation contracts.

The patch dimension is a multiple of 8 because the TurboVec store requires it.
"""

from __future__ import annotations

import numpy as np
import pytest

import lodedb.local.late_interaction as li_module
from lodedb import LodeLateInteractionHit, LodeLateInteractionIndex, ReadOnlyError
from lodedb.local.late_interaction import (
    _maxsim,
    _maxsim_batch,
    _resolve_native_maxsim,
)

DIM = 16


def _onehot_matrix(indices: list[int], *, dim: int = DIM) -> list[list[float]]:
    rows = []
    for i in indices:
        row = [0.0] * dim
        row[i] = 1.0
        rows.append(row)
    return rows


def _unit_rows(rng, rows: int) -> np.ndarray:
    m = rng.standard_normal((rows, DIM)).astype(np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def test_count_and_patch_count(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("doc-a", _onehot_matrix([0, 1, 2]), metadata={"file": "a.pdf"})
    idx.add_document("doc-b", _onehot_matrix([5, 6]))
    assert idx.count() == 2  # two documents (one row each)
    assert idx.patch_count() == 5  # total patches across documents


def test_maxsim_ranks_best_overlap_first(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    # doc-a patches cover dims {0,1,2}; doc-b covers {5,6,7}.
    idx.add_document("doc-a", _onehot_matrix([0, 1, 2]))
    idx.add_document("doc-b", _onehot_matrix([5, 6, 7]))

    # Query tokens land on dims 0 and 1 -> should match doc-a.
    hits = idx.search(_onehot_matrix([0, 1]), k=2)
    assert hits[0].id == "doc-a"
    assert hits[0].score > hits[1].score
    # Two query tokens each perfectly matching a doc-a patch -> exact MaxSim == 2.0
    # (the exact rescore reads back the float32 patches, so quantization noise in
    # the candidate scan does not leak into the final score).
    assert hits[0].score == pytest.approx(2.0, abs=1e-5)


def test_search_returns_hit_tuple_and_attributes(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("doc-a", _onehot_matrix([0, 1]), metadata={"k": "v"})
    [hit] = idx.search(_onehot_matrix([0]), k=1)
    assert isinstance(hit, LodeLateInteractionHit)
    score, doc_id, meta = hit
    assert doc_id == "doc-a"
    assert meta == {"k": "v"}  # internal parent_id / patch_count stripped
    assert hit.patch_count == 2
    assert score == pytest.approx(1.0, abs=1e-5)


def test_persist_and_reopen_roundtrip(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("doc-a", _onehot_matrix([0, 1, 2]), metadata={"file": "a.pdf"})
    idx.persist()
    idx.close()

    reopened = LodeLateInteractionIndex(tmp_path, dim=DIM)
    assert reopened.count() == 1
    hits = reopened.search(_onehot_matrix([2]), k=1)
    assert hits[0].id == "doc-a"
    assert hits[0].metadata == {"file": "a.pdf"}
    assert reopened.list_documents() == [
        {"id": "doc-a", "metadata": {"file": "a.pdf"}, "patch_count": 3}
    ]


def test_readd_replaces_patches_even_when_shorter(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("doc-a", _onehot_matrix([0, 1, 2, 3]))
    assert idx.patch_count() == 4
    idx.add_document("doc-a", _onehot_matrix([5]))  # shorter re-add
    assert idx.count() == 1
    assert idx.patch_count() == 1  # no stale tail rows
    hits = idx.search(_onehot_matrix([5]), k=1)
    assert hits[0].id == "doc-a"
    assert hits[0].patch_count == 1


def test_remove_drops_all_patches(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("doc-a", _onehot_matrix([0, 1]))
    idx.add_document("doc-b", _onehot_matrix([5, 6]))
    assert idx.remove("doc-a") is True
    assert idx.count() == 1
    assert idx.patch_count() == 2
    assert idx.remove("doc-a") is False  # already gone


def test_resident_and_streaming_paths_agree(tmp_path):
    # The resident exact scan (default) and the disk-streaming path must return the
    # same ranking and scores (both exhaustive, same stored precision).
    rng = np.random.default_rng(7)
    docs = []
    for i in range(40):
        m = rng.standard_normal((6, DIM)).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        docs.append((f"d{i:03d}", m))
    res_idx = LodeLateInteractionIndex(tmp_path / "res", dim=DIM, resident=True)
    stream_idx = LodeLateInteractionIndex(tmp_path / "stream", dim=DIM, resident=False)
    for store in (res_idx, stream_idx):
        for doc_id, m in docs:
            store.add_document(doc_id, m, normalize=False)
    q = rng.standard_normal((5, DIM)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    res_hits = res_idx.search(q, k=10, normalize=False)
    stream_hits = stream_idx.search(q, k=10, normalize=False)
    assert [h.id for h in res_hits] == [h.id for h in stream_hits]
    for a, b in zip(res_hits, stream_hits, strict=True):
        assert a.score == pytest.approx(b.score, abs=1e-4)


def test_resident_budget_falls_back_to_streaming(tmp_path):
    # A tiny resident budget under "auto" forces the streaming path, still correct.
    idx = LodeLateInteractionIndex(
        tmp_path, dim=DIM, resident="auto", resident_max_bytes=1
    )
    idx.add_document("doc-a", _onehot_matrix([0, 1]))
    idx.add_document("doc-b", _onehot_matrix([5, 6]))
    assert idx._resident_cache_get() is None  # over budget
    hits = idx.search(_onehot_matrix([0]), k=1)
    assert hits[0].id == "doc-a"


def test_resident_cache_reflects_incremental_writes(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("doc-a", _onehot_matrix([0, 1]))
    assert idx.search(_onehot_matrix([0]), k=5)[0].id == "doc-a"  # builds cache
    idx.add_document("doc-b", _onehot_matrix([5, 6]))  # folds into the live cache
    hits = idx.search(_onehot_matrix([5]), k=5)
    assert hits[0].id == "doc-b"
    assert {h.id for h in idx.search(_onehot_matrix([0]), k=5)} == {"doc-a", "doc-b"}
    idx.remove("doc-a")
    assert {h.id for h in idx.search(_onehot_matrix([0]), k=5)} == {"doc-b"}
    # Re-adding an id updates the live cache to the new content (doc-b -> dim 9).
    idx.add_document("doc-b", _onehot_matrix([9]))
    hit = idx.search(_onehot_matrix([9]), k=1)[0]
    assert hit.id == "doc-b" and hit.patch_count == 1


def test_incremental_cache_matches_fresh_rebuild(tmp_path):
    # An interleaved add/remove/re-add sequence against the live cache must give
    # the same results as a fresh handle that rebuilds the cache from disk.
    rng = np.random.default_rng(5)

    def mat():
        m = rng.standard_normal((6, DIM)).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("seed", mat(), normalize=False)
    idx.search(_onehot_matrix([0]), k=1)  # build the live cache
    for i in range(30):
        idx.add_document(f"d{i:02d}", mat(), normalize=False)
        if i % 5 == 0:
            idx.add_document("d00", mat(), normalize=False)  # re-add (replace)
        if i % 7 == 0 and i:
            idx.remove(f"d{i - 1:02d}")
    idx.persist()

    fresh = LodeLateInteractionIndex(tmp_path, dim=DIM, read_only=True)
    for _ in range(10):
        q = mat()
        live_hits = idx.search(q, k=8, normalize=False)
        fresh_hits = fresh.search(q, k=8, normalize=False)
        assert [h.id for h in live_hits] == [h.id for h in fresh_hits]
        for a, b in zip(live_hits, fresh_hits, strict=True):
            assert a.score == pytest.approx(b.score, abs=1e-4)


def test_incremental_compaction_folds_pending(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("a", _onehot_matrix([0, 1]))
    idx.search(_onehot_matrix([0]), k=1)  # build cache
    idx.add_document("b", _onehot_matrix([5]))  # goes to pending
    idx.add_document("a", _onehot_matrix([2]))  # replace -> tombstone + pending
    cache = idx._resident_cache
    assert cache["pending"]  # delta present before compaction
    idx._cache_compact(cache)
    assert cache["pending"] == [] and cache["removed"] == set()
    assert set(cache["ids"]) == {"a", "b"}
    # Results still correct after compaction: "a" now matches dim 2, not 0/1.
    assert idx.search(_onehot_matrix([2]), k=1)[0].id == "a"
    assert idx.search(_onehot_matrix([5]), k=1)[0].id == "b"
    assert {h.id for h in idx.search(_onehot_matrix([0]), k=5)} == {"a", "b"}


def test_incremental_growth_evicts_over_budget(tmp_path):
    # Start within budget so the cache builds, then grow past it incrementally:
    # the cache evicts and queries fall back to the (exact) streaming path.
    idx = LodeLateInteractionIndex(
        tmp_path, dim=DIM, resident="auto", resident_max_bytes=DIM * 4 * 3
    )
    idx.add_document("a", _onehot_matrix([0]))
    assert idx._resident_cache_get() is not None  # fits
    for i in range(10):
        idx.add_document(f"x{i}", _onehot_matrix([i % DIM]))
    assert idx._resident_cache is None  # evicted on growth
    assert idx.search(_onehot_matrix([0]), k=1)[0].id == "a"  # streaming still exact


def test_add_documents_batch(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    ids = idx.add_documents(
        [
            {"id": "x", "patches": _onehot_matrix([0])},
            {"id": "y", "patches": _onehot_matrix([5])},
        ]
    )
    assert ids == ["x", "y"]
    assert idx.count() == 2


def test_search_many_matches_single_search(tmp_path):
    rng = np.random.default_rng(13)

    def mat(rows):
        m = rng.standard_normal((rows, DIM)).astype(np.float32)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    for i in range(50):
        idx.add_document(f"d{i:02d}", mat(6), normalize=False)
    queries = [mat(rows) for rows in (5, 3, 8, 4)]  # ragged query lengths
    batched = idx.search_many(queries, k=7, normalize=False)
    assert len(batched) == len(queries)
    for q, batch_hits in zip(queries, batched, strict=True):
        single = idx.search(q, k=7, normalize=False)
        assert [h.id for h in batch_hits] == [h.id for h in single]
        for a, b in zip(batch_hits, single, strict=True):
            assert a.score == pytest.approx(b.score, abs=1e-4)


def test_search_many_multichunk_matches_single(tmp_path, monkeypatch):
    # Force many small scoring chunks; the incremental per-query top-k merge must
    # still equal looped single-query search (covers bounded-memory search_many).
    monkeypatch.setattr(li_module, "_SCORE_CHUNK_BYTES", 4096)
    rng = np.random.default_rng(21)
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    for i in range(60):
        idx.add_document(f"d{i:02d}", _unit_rows(rng, 6), normalize=False)
    queries = [_unit_rows(rng, 5) for _ in range(4)]
    batched = idx.search_many(queries, k=7, normalize=False)
    for q, batch_hits in zip(queries, batched, strict=True):
        single = idx.search(q, k=7, normalize=False)
        assert [h.id for h in batch_hits] == [h.id for h in single]
        for a, b in zip(batch_hits, single, strict=True):
            assert a.score == pytest.approx(b.score, abs=1e-4)


def test_streaming_chunked_matches_resident(tmp_path, monkeypatch):
    # A tiny chunk budget forces the streaming path to chunk during load; results
    # must match the resident scan over the same documents.
    monkeypatch.setattr(li_module, "_SCORE_CHUNK_BYTES", 8192)
    rng = np.random.default_rng(22)
    docs = [(f"d{i:02d}", _unit_rows(rng, 40)) for i in range(30)]
    stream = LodeLateInteractionIndex(tmp_path / "s", dim=DIM, resident=False)
    resident = LodeLateInteractionIndex(tmp_path / "r", dim=DIM, resident=True)
    for doc_id, m in docs:
        stream.add_document(doc_id, m, normalize=False)
        resident.add_document(doc_id, m, normalize=False)
    q = _unit_rows(rng, 8)
    s_hits = stream.search(q, k=5, normalize=False)
    r_hits = resident.search(q, k=5, normalize=False)
    assert [h.id for h in s_hits] == [h.id for h in r_hits]
    for a, b in zip(s_hits, r_hits, strict=True):
        assert a.score == pytest.approx(b.score, abs=1e-4)


def test_single_document_larger_than_chunk_budget(tmp_path, monkeypatch):
    # A document with more patches than the chunk budget is scored as one unit.
    monkeypatch.setattr(li_module, "_SCORE_CHUNK_BYTES", 2048)
    rng = np.random.default_rng(23)
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM, resident=False)
    big = _unit_rows(rng, 300)
    idx.add_document("big", big, normalize=False)
    idx.add_document("small", _onehot_matrix([0]))
    hits = idx.search(big[:3], k=2, normalize=False)
    assert hits[0].id == "big"


def test_dim_must_be_multiple_of_8(tmp_path):
    with pytest.raises(ValueError, match="multiple of 8"):
        LodeLateInteractionIndex(tmp_path, dim=12)


def test_corrupt_storage_config_rejected(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM, storage="int8")
    idx.add_document("a", _onehot_matrix([0]))
    idx.persist()
    idx.close()
    # A present-but-unparseable sidecar is corruption, not "no config": reopening
    # must raise rather than silently switch the index to the float32 default.
    (tmp_path / li_module._CONFIG_FILENAME).write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        LodeLateInteractionIndex(tmp_path, dim=DIM)


def test_search_many_edges(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    assert idx.search_many([], k=5) == []  # no queries
    assert idx.search_many([_onehot_matrix([0])], k=5) == [[]]  # empty index
    idx.add_document("a", _onehot_matrix([0, 1]), metadata={"t": "x"})
    idx.add_document("b", _onehot_matrix([5]), metadata={"t": "y"})
    # Filter applies to every query in the batch.
    out = idx.search_many([_onehot_matrix([0]), _onehot_matrix([5])], k=5, filter={"t": "y"})
    assert [[h.id for h in hits] for hits in out] == [["b"], ["b"]]


def test_filter_narrows_candidates_to_matching_documents(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("public", _onehot_matrix([0, 1]), metadata={"tenant": "a"})
    idx.add_document("private", _onehot_matrix([0, 2]), metadata={"tenant": "b"})

    hits = idx.search(_onehot_matrix([0]), k=5, filter={"tenant": "b"})
    assert [hit.id for hit in hits] == ["private"]


def test_list_documents_filter(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("p1", _onehot_matrix([0]), metadata={"book": "x"})
    idx.add_document("p2", _onehot_matrix([1]), metadata={"book": "y"})
    listed = idx.list_documents(filter={"book": "y"})
    assert listed == [{"id": "p2", "metadata": {"book": "y"}, "patch_count": 1}]


def test_numpy_array_input_is_accepted(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    patches = np.asarray(_onehot_matrix([0, 1]), dtype=np.float32)
    idx.add_document("doc-a", patches)
    hits = idx.search(np.asarray(_onehot_matrix([1]), dtype=np.float32), k=1)
    assert hits[0].id == "doc-a"


def test_dimension_mismatch_is_rejected(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    with pytest.raises(ValueError):
        idx.add_document("doc-a", [[0.0] * (DIM + 1)])
    with pytest.raises(ValueError):
        idx.search([[0.0] * (DIM - 1)])


def test_reserved_metadata_key_rejected(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    with pytest.raises(ValueError, match="reserved"):
        idx.add_document("doc-a", _onehot_matrix([0]), metadata={"patch_count": "x"})


def test_empty_doc_id_rejected(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    with pytest.raises(ValueError):
        idx.add_document("  ", _onehot_matrix([0]))


def test_read_only_rejects_writes(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    idx.add_document("doc-a", _onehot_matrix([0]))
    idx.persist()
    idx.close()

    ro = LodeLateInteractionIndex(tmp_path, dim=DIM, read_only=True)
    with pytest.raises(ReadOnlyError):
        ro.add_document("doc-b", _onehot_matrix([1]))
    # reads still work
    assert ro.count() == 1
    assert ro.search(_onehot_matrix([0]), k=1)[0].id == "doc-a"


def test_store_text_false_rejected(tmp_path):
    with pytest.raises(ValueError):
        LodeLateInteractionIndex(tmp_path, dim=DIM, store_text=False)


def test_empty_doc_rejected(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    with pytest.raises(ValueError):
        idx.add_document("doc-a", np.zeros((0, DIM), dtype=np.float32))


def test_search_empty_index_returns_nothing(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    assert idx.search(_onehot_matrix([0]), k=5) == []


def test_hit_equality():
    hit = LodeLateInteractionHit(score=1.5, id="x", metadata={"a": "b"}, patch_count=3)
    assert hit == (1.5, "x", {"a": "b"})
    assert hit == LodeLateInteractionHit(
        score=1.5, id="x", metadata={"a": "b"}, patch_count=3
    )


def test_maxsim_kernel_matches_reference():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((4, DIM)).astype(np.float32)
    d = rng.standard_normal((7, DIM)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    expected = sum(max(float(qt @ dp) for dp in d) for qt in q)
    assert _maxsim(q, d) == pytest.approx(expected, abs=1e-5)


def test_native_kernel_is_available_and_matches_numpy():
    # Stage 3: the bundled extension exposes the native MaxSim kernel. (If a build
    # predates it, the SDK falls back to numpy; this asserts the bundled build.)
    native = _resolve_native_maxsim()
    assert native is not None, "native maxsim_scores kernel not found in lodedb._turbovec"
    rng = np.random.default_rng(3)
    q = rng.standard_normal((6, DIM)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    docs = []
    for n in (5, 8, 3, 11):
        d = rng.standard_normal((n, DIM)).astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        docs.append(d)
    reference = np.array([_maxsim(q, d) for d in docs], dtype=np.float32)
    native_scores = _maxsim_batch(q, docs, prefer_native=True)
    numpy_scores = _maxsim_batch(q, docs, prefer_native=False)
    assert np.allclose(native_scores, reference, atol=1e-4)
    assert np.allclose(native_scores, numpy_scores, atol=1e-4)


def test_maxsim_batch_empty_returns_empty():
    assert _maxsim_batch(np.zeros((2, DIM), dtype=np.float32), []).shape == (0,)


def test_native_scoring_matches_numpy_end_to_end(tmp_path):
    # The same corpus scored through the native kernel and the numpy path returns
    # the same ranking and scores.
    docs = [("a", [0, 1, 2]), ("b", [5, 6]), ("c", [2, 7, 9])]
    np_idx = LodeLateInteractionIndex(tmp_path / "np", dim=DIM, scoring="numpy")
    nat_idx = LodeLateInteractionIndex(tmp_path / "nat", dim=DIM, scoring="native")
    for idx in (np_idx, nat_idx):
        for doc_id, dims in docs:
            idx.add_document(doc_id, _onehot_matrix(dims))
    query = _onehot_matrix([0, 2])
    np_hits = np_idx.search(query, k=3)
    nat_hits = nat_idx.search(query, k=3)
    assert [h.id for h in np_hits] == [h.id for h in nat_hits]
    for a, b in zip(np_hits, nat_hits, strict=True):
        assert a.score == pytest.approx(b.score, abs=1e-4)


def test_invalid_scoring_rejected(tmp_path):
    with pytest.raises(ValueError):
        LodeLateInteractionIndex(tmp_path, dim=DIM, scoring="bogus")


def test_invalid_storage_rejected(tmp_path):
    with pytest.raises(ValueError):
        LodeLateInteractionIndex(tmp_path, dim=DIM, storage="bogus")


@pytest.mark.parametrize("storage", ["float32", "float16", "int8"])
def test_storage_roundtrip_and_reopen(tmp_path, storage):
    # Each precision stores, retrieves, and reopens correctly. One-hot vectors are
    # represented exactly by every precision, so ranking is unambiguous.
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM, storage=storage)
    idx.add_document("doc-a", _onehot_matrix([0, 1, 2]), metadata={"file": "a.pdf"})
    idx.add_document("doc-b", _onehot_matrix([5, 6]))
    assert idx.count() == 2
    assert idx.patch_count() == 5
    hits = idx.search(_onehot_matrix([0, 1]), k=2)
    assert hits[0].id == "doc-a"
    assert hits[0].score == pytest.approx(2.0, abs=1e-2)
    idx.persist()
    idx.close()

    # Reopen without specifying storage: each document records its own precision.
    reopened = LodeLateInteractionIndex(tmp_path, dim=DIM)
    assert reopened.count() == 2
    assert reopened.search(_onehot_matrix([5]), k=1)[0].id == "doc-b"
    assert reopened.list_documents()[0] == {
        "id": "doc-a",
        "metadata": {"file": "a.pdf"},
        "patch_count": 3,
    }


def test_storage_persists_across_reopen(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM, storage="int8")
    assert idx.storage == "int8"
    idx.add_document("doc-a", _onehot_matrix([0, 1]))
    idx.persist()
    idx.close()

    # Reopen without specifying storage: it adopts the persisted precision.
    reopened = LodeLateInteractionIndex(tmp_path, dim=DIM)
    assert reopened.storage == "int8"


def test_new_index_defaults_to_float32(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)
    assert idx.storage == "float32"
    idx.close()
    assert LodeLateInteractionIndex(tmp_path, dim=DIM).storage == "float32"


def test_conflicting_storage_on_reopen_rejected(tmp_path):
    LodeLateInteractionIndex(tmp_path, dim=DIM, storage="float32").close()
    with pytest.raises(ValueError, match="created with storage"):
        LodeLateInteractionIndex(tmp_path, dim=DIM, storage="int8")
    # Matching the stored precision is fine.
    assert LodeLateInteractionIndex(tmp_path, dim=DIM, storage="float32").storage == "float32"


def test_read_only_reopen_adopts_persisted_storage(tmp_path):
    writer = LodeLateInteractionIndex(tmp_path, dim=DIM, storage="int8")
    writer.add_document("doc-a", _onehot_matrix([0, 1]))
    writer.persist()
    writer.close()
    reader = LodeLateInteractionIndex(tmp_path, dim=DIM, read_only=True)
    assert reader.storage == "int8"
    assert reader.search(_onehot_matrix([0]), k=1)[0].id == "doc-a"


def test_storage_precision_recall_on_random_vectors(tmp_path):
    # float16 should match the float32 ranking; int8 should be close.
    rng = np.random.default_rng(11)
    docs = []
    for i in range(60):
        m = rng.standard_normal((8, DIM)).astype(np.float32)
        m /= np.linalg.norm(m, axis=1, keepdims=True)
        docs.append((f"d{i:03d}", m))
    queries = []
    for _ in range(30):
        q = rng.standard_normal((5, DIM)).astype(np.float32)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        queries.append(q)

    def ranking(storage):
        idx = LodeLateInteractionIndex(tmp_path / storage, dim=DIM, storage=storage)
        for doc_id, m in docs:
            idx.add_document(doc_id, m, normalize=False)
        return [[h.id for h in idx.search(q, k=5, normalize=False)] for q in queries]

    truth = ranking("float32")

    def recall_at_5(pred):
        hits = sum(
            len(set(p) & set(t)) / len(t) for p, t in zip(pred, truth, strict=True)
        )
        return hits / len(truth)

    assert recall_at_5(ranking("float16")) == pytest.approx(1.0, abs=1e-6)
    assert recall_at_5(ranking("int8")) >= 0.9


class _FakeColPali:
    """Deterministic stand-in for a ColPali-style encoder (no download).

    Each input string is hashed token-by-token into unit one-hot patch vectors,
    so the encoder path can be exercised offline and deterministically.
    """

    def encode_documents(self, contents):
        return [self._matrix(content) for content in contents]

    def encode_queries(self, queries):
        return [self._matrix(query) for query in queries]

    @staticmethod
    def _matrix(text):
        rows = []
        for token in str(text).split():
            dim_index = (sum(ord(ch) for ch in token)) % DIM
            row = [0.0] * DIM
            row[dim_index] = 1.0
            rows.append(row)
        return rows or [[1.0] + [0.0] * (DIM - 1)]


def test_encoder_convenience_path(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM, encoder=_FakeColPali())
    idx.add_texts(
        [
            {"id": "p1", "content": "alpha beta gamma", "metadata": {"f": "1.pdf"}},
            {"id": "p2", "content": "delta epsilon"},
        ]
    )
    assert idx.count() == 2
    hits = idx.search_text("alpha beta", k=2)
    assert hits[0].id == "p1"


def test_encoder_required_for_text_path(tmp_path):
    idx = LodeLateInteractionIndex(tmp_path, dim=DIM)  # no encoder
    with pytest.raises(RuntimeError):
        idx.search_text("anything")
    with pytest.raises(RuntimeError):
        idx.add_texts([{"content": "x"}])
