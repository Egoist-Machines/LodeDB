"""Policy + engine-dispatch tests for the opt-in MPS direct TurboVec route.

The policy tests run everywhere; the engine-dispatch tests require a usable MPS
device (there is no fake-device seam — MPS *is* torch — so they skip on CI).
"""

from __future__ import annotations

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.mps_turbovec import (
    estimate_mps_direct_turbovec_memory,
    mps_exact_scan_available,
)
from lodedb.engine.runtime_policy import (
    MpsDirectTurboVecPolicy,
    mps_direct_turbovec_policy_from_env,
    mps_direct_turbovec_should_use,
)
from lodedb.local.db import LodeDB

_MPS_OK, _MPS_REASON = mps_exact_scan_available()
_mps_only = pytest.mark.skipif(not _MPS_OK, reason=f"MPS unavailable: {_MPS_REASON}")


# --- policy: always-run, no device required --------------------------------


def test_mps_policy_from_env_defaults_off():
    assert mps_direct_turbovec_policy_from_env({}) == MpsDirectTurboVecPolicy.OFF
    assert (
        mps_direct_turbovec_policy_from_env({"LODEDB_MPS_DIRECT_TURBOVEC": "auto"})
        == MpsDirectTurboVecPolicy.AUTO
    )
    assert (
        mps_direct_turbovec_policy_from_env({"LODEDB_MPS_DIRECT_TURBOVEC": "required"})
        == MpsDirectTurboVecPolicy.REQUIRED
    )
    with pytest.raises(ValueError):
        mps_direct_turbovec_policy_from_env({"LODEDB_MPS_DIRECT_TURBOVEC": "bogus"})


def test_mps_should_use_rules():
    required = MpsDirectTurboVecPolicy.REQUIRED
    auto = MpsDirectTurboVecPolicy.AUTO
    off = MpsDirectTurboVecPolicy.OFF

    def use(policy, available, batch, **kwargs):
        return mps_direct_turbovec_should_use(
            policy=policy, dependency_available=available, query_batch_size=batch, **kwargs
        )

    assert use(required, True, 1) is False  # single query bypasses, no raise
    assert use(off, True, 8) is False
    assert use(auto, True, 8) is True
    assert use(auto, True, 8, maximum_batch_size=4) is False  # auto window flip
    with pytest.raises(RuntimeError):  # required fails closed when unavailable
        use(required, False, 8)


def test_mps_memory_estimate_is_corpus_bounded_not_tile_ceiling():
    """A small index's estimate reflects the real corpus, not the 64+128 MiB tile cap.

    The earlier estimate always charged the full per-tile budget ceiling, so any
    sub-ceiling ``LODEDB_MPS_MEMORY_BUDGET_BYTES`` would reject even a tiny index.
    """

    est = estimate_mps_direct_turbovec_memory(row_count=600, dim=384, max_batch_size=64)
    # 600 rows is a few MB, far under the old ~192 MiB tile-budget ceiling.
    assert est.estimated_bytes < 16 * 1024 * 1024
    # A 64 MiB budget (which the ceiling estimate always rejected) now admits it.
    admitted = estimate_mps_direct_turbovec_memory(
        row_count=600, dim=384, max_batch_size=64, memory_budget_bytes=64 * 1024 * 1024
    )
    assert admitted.admitted is True
    # A budget below the genuine resident+transient cost still rejects (fail-closed).
    rejected = estimate_mps_direct_turbovec_memory(
        row_count=600, dim=384, max_batch_size=64, memory_budget_bytes=64 * 1024
    )
    assert rejected.admitted is False
    assert "exceeds MPS budget" in rejected.reason


# --- engine dispatch: MPS device required ----------------------------------


def _open(tmp_path):
    return LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )


@_mps_only
def test_mps_required_dispatch_builds_and_patches_in_place(tmp_path, monkeypatch):
    """A batched search builds the MPS session; mutations patch it in place."""

    monkeypatch.setenv("LODEDB_MPS_DIRECT_TURBOVEC", "required")
    db = _open(tmp_path)
    try:
        db.add("first doc", id="doc-a")
        db.add("second doc", id="doc-b")
        db.search_many(["first", "second"], k=1)  # batch >= 2 builds the MPS session

        sessions = db._engine._mps_direct_turbovec_sessions
        assert len(sessions) == 1
        key = next(iter(sessions))
        before = sessions[key]
        assert before.row_count == 2

        db.add("third doc", id="doc-c")  # upsert -> patched in place (no search between)
        after_add = sessions[key]
        assert after_add is before
        assert after_add.row_count == 3
        serving = db._engine._turbovec_indexes[key]
        resident = {int(x) for x in after_add.stable_ids[: after_add.row_count]}
        assert resident == set(serving.chunk_ids_by_stable_id)

        db.remove("doc-a")  # removal -> patched in place
        after_remove = sessions[key]
        assert after_remove is before
        assert after_remove.row_count == 2

        assert len(db.search_many(["second", "third"], k=1)) == 2
    finally:
        db.close()


@_mps_only
def test_mps_required_batch_returns_correct_top1(tmp_path, monkeypatch):
    """With MPS required, a batched search is served on MPS with correct results."""

    monkeypatch.setenv("LODEDB_MPS_DIRECT_TURBOVEC", "required")
    db = _open(tmp_path)
    try:
        texts = {f"d{i}": f"unique marker {i} alpha beta gamma" for i in range(40)}
        for cid, text in texts.items():
            db.add(text, id=cid)
        batched = db.search_many(
            ["unique marker 3 alpha beta gamma", "unique marker 17 alpha beta gamma"], k=5
        )
        assert db._engine._mps_direct_turbovec_sessions  # MPS actually served it
        assert batched[0][0].id == "d3"  # an exact-text query -> that doc is top-1
        assert batched[1][0].id == "d17"
        assert all(hit.id in texts for row in batched for hit in row)
    finally:
        db.close()


@_mps_only
def test_mps_off_by_default_keeps_batch_on_cpu(tmp_path):
    """The default policy is OFF, so a batched search never builds an MPS session."""

    db = _open(tmp_path)
    try:
        for i in range(20):
            db.add(f"doc {i}", id=f"d{i}")
        db.search_many(["doc 1", "doc 2"], k=3)
        assert not db._engine._mps_direct_turbovec_sessions
    finally:
        db.close()


@_mps_only
def test_mps_auto_bypass_reason_is_visible_on_cpu_fallback_rows(tmp_path, monkeypatch):
    """When MPS auto bypasses a batch, the NEON rows still carry the MPS bypass reason.

    Regression test: the CPU-fallback rows previously attached only the (empty) GPU
    telemetry and dropped the MPS status/reason, hiding why the batch fell back. With
    CUDA forced off, the MPS reason is the only resident-scan telemetry, so it must
    survive onto the emitted ``query_batch_completed`` event.
    """

    # Force CUDA off so MPS is the only resident path contributing telemetry.
    monkeypatch.setenv("LODEDB_GPU_DIRECT_TURBOVEC", "off")
    monkeypatch.setenv("LODEDB_MPS_DIRECT_TURBOVEC", "auto")
    # max-batch 1 means any batch >= 2 is above the auto window -> bypassed.
    monkeypatch.setenv("LODEDB_MPS_DIRECT_MAX_BATCH", "1")
    db = _open(tmp_path)
    try:
        for i in range(8):
            db.add(f"doc {i}", id=f"d{i}")
        db.search_many(["doc 1", "doc 2"], k=2)  # batch of 2 -> bypassed -> NEON
        assert not db._engine._mps_direct_turbovec_sessions  # bypassed before any build
        batch_events = [
            event
            for event in db._engine.audit_events
            if event["event"] == "query_batch_completed"
        ]
        assert batch_events
        assert batch_events[-1]["gpu_stage_one_status"] == "bypassed"
        assert batch_events[-1]["gpu_fallback_reason"] == "mps_direct_batch_above_window"
    finally:
        db.close()
