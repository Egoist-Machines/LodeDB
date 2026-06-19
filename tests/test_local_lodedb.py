"""Tests for the local-first LodeDB SDK over the engine engine.

These exercise the additive local layer (no auth, on-disk, reusing the
engine + TurboVec storage + .tvim/.tvd/.jsd persistence) with a deterministic
hash embedding backend, so they neither download models nor import torch into
the test process (keeping them safely separable from faiss tests on macOS).
"""

from __future__ import annotations

import pytest

from lodedb.engine.core import (
    DIRECT_TURBOVEC_STORAGE_PROFILE,
    storage_profile_for_route_policy,
)
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.route_profiles import (
    DEFAULT_ROUTE_PROFILE,
    client_route_policy_manifest,
    route_policy_for_profile,
)
from lodedb.local import LodeDB, LodeSearchHit


def _open(tmp_path, dim: int = 384, model: str = "minilm") -> LodeDB:
    """Opens a LodeDB with an injected deterministic hash backend."""

    return LodeDB(
        path=tmp_path, model=model, _embedding_backend=HashEmbeddingBackend(native_dim=dim)
    )


def test_route_policy_exposes_index_backend_not_stale_serving_backend():
    """Route policy metadata names the direct index backend and defaults to direct storage."""

    manifest = client_route_policy_manifest()
    assert manifest
    assert all(row["index_backend"] == DIRECT_TURBOVEC_STORAGE_PROFILE for row in manifest)
    assert all("serving_backend" not in row for row in manifest)

    policy = route_policy_for_profile(DEFAULT_ROUTE_PROFILE)
    assert policy.index_backend == DIRECT_TURBOVEC_STORAGE_PROFILE
    assert storage_profile_for_route_policy(policy) == DIRECT_TURBOVEC_STORAGE_PROFILE
    assert storage_profile_for_route_policy(None) == DIRECT_TURBOVEC_STORAGE_PROFILE


def test_add_search_returns_tuple_shaped_hits(tmp_path):
    """add/search returns redacted (score, id, metadata) rows that unpack."""

    db = _open(tmp_path)
    a = db.add("the quick brown fox", metadata={"topic": "animals"})
    db.add("a slow green turtle", id="turtle", metadata={"topic": "animals"})
    assert isinstance(a, str) and a
    hits = db.search("fox", k=5)
    assert hits and all(isinstance(h, LodeSearchHit) for h in hits)
    score, hit_id, metadata = hits[0]  # tuple unpacking per the documented API
    assert isinstance(score, float)
    assert isinstance(hit_id, str)
    assert isinstance(metadata, dict)
    # metadata is hydrated from the stored document, and is string-valued.
    ids = {h.id for h in hits}
    assert "turtle" in ids
    db.close()


def test_auto_id_is_generated_and_unique(tmp_path):
    """A missing id is auto-generated and ids do not collide."""

    db = _open(tmp_path)
    ids = [db.add(f"document number {i}") for i in range(20)]
    assert len(set(ids)) == 20
    assert db.count() == 20
    db.close()


def test_metadata_filter_flat_and_structured(tmp_path):
    """search supports a flat metadata filter and the structured filter form."""

    db = _open(tmp_path)
    db.add("alpha content here", id="a", metadata={"topic": "x"})
    db.add("beta content here", id="b", metadata={"topic": "y"})
    db.add("gamma content here", id="c", metadata={"topic": "x"})

    flat = {h.id for h in db.search("content", k=10, filter={"topic": "x"})}
    assert flat == {"a", "c"}

    structured = {h.id for h in db.search("content", k=10, filter={"metadata": {"topic": "y"}})}
    assert structured == {"b"}

    by_id = {h.id for h in db.search("content", k=10, filter={"document_ids": ["a"]})}
    assert by_id == {"a"}
    db.close()


def test_search_many_preserves_order_and_hit_shape(tmp_path):
    """search_many batches queries while preserving query order and hit shape."""

    db = _open(tmp_path)
    db.add("alpha fox dossier", id="fox", metadata={"topic": "animals"})
    db.add("beta turtle file", id="turtle", metadata={"topic": "animals"})
    db.add("gamma physics lecture", id="physics", metadata={"topic": "science"})

    batches = db.search_many(["fox", "physics"], k=2)
    assert len(batches) == 2
    assert all(isinstance(hit, LodeSearchHit) for row in batches for hit in row)
    assert batches[0]
    score, hit_id, metadata = batches[0][0]
    assert isinstance(score, float)
    assert isinstance(hit_id, str)
    assert isinstance(metadata, dict)
    singles = [db.search("fox", k=2), db.search("physics", k=2)]
    assert [[hit.id for hit in row] for row in batches] == [
        [hit.id for hit in row] for row in singles
    ]
    db.close()


def test_search_many_supports_filters(tmp_path):
    """search_many applies the same metadata/document filter to every query."""

    db = _open(tmp_path)
    db.add("alpha fox dossier", id="fox", metadata={"topic": "animals"})
    db.add("beta turtle file", id="turtle", metadata={"topic": "animals"})
    db.add("gamma physics lecture", id="physics", metadata={"topic": "science"})

    batches = db.search_many(["fox", "lecture"], k=10, filter={"topic": "animals"})
    assert len(batches) == 2
    assert all({hit.id for hit in row} <= {"fox", "turtle"} for row in batches)
    db.close()


def test_numeric_and_bool_metadata_is_coerced_and_matchable(tmp_path):
    """Numeric/bool metadata is stringified consistently on add and on filter."""

    db = _open(tmp_path)
    db.add("doc one", id="one", metadata={"year": 2020, "fresh": True})
    db.add("doc two", id="two", metadata={"year": 2021, "fresh": False})
    hits = db.search("doc", k=10, filter={"year": 2020})
    assert {h.id for h in hits} == {"one"}
    # stored metadata round-trips as strings
    only = db.search("doc", k=10, filter={"document_ids": ["one"]})[0]
    assert only.metadata["year"] == "2020"
    assert only.metadata["fresh"] == "true"
    db.close()


def test_remove_returns_true_only_when_document_existed(tmp_path):
    """remove deletes by id and reports whether a document was actually removed."""

    db = _open(tmp_path)
    db.add("removable doc", id="gone")
    assert db.count() == 1
    assert db.remove("gone") is True
    assert db.count() == 0
    assert db.remove("never-existed") is False
    db.close()


def test_persist_and_reopen_round_trip(tmp_path):
    """State persists on disk and reloads on reopen (fail-closed sidecar replay)."""

    db = _open(tmp_path)
    db.add("the quick brown fox jumps", id="fox", metadata={"topic": "animals"})
    db.add("quantum chromodynamics lecture", id="phys", metadata={"topic": "physics"})
    db.remove("phys")
    db.persist()
    before = {h.id for h in db.search("fox", k=10)}
    db.close()

    reopened = _open(tmp_path)
    assert reopened.count() == 1
    after = {h.id for h in reopened.search("fox", k=10)}
    assert after == before == {"fox"}
    # The removed document stays removed across the restart.
    assert reopened.search("fox", k=10, filter={"document_ids": ["phys"]}) == []
    reopened.close()


def test_reopen_uses_stable_index_id(tmp_path):
    """Reopening the same path binds to the same stable (default) index id."""

    db = _open(tmp_path)
    db.add("persisted document", id="p1")
    first_id = db._index.index_id
    db.close()
    db2 = _open(tmp_path)
    assert db2._index.index_id == first_id == "default"
    assert db2.count() == 1
    db2.close()


def test_add_many_batches_and_counts(tmp_path):
    """add_many indexes a batch and returns ids preserving caller-supplied ids."""

    db = _open(tmp_path)
    ids = db.add_many(
        [
            {"text": "first doc", "id": "f"},
            {"text": "second doc", "metadata": {"k": "v"}},
            {"text": "third doc", "id": "t"},
        ]
    )
    assert ids[0] == "f" and ids[2] == "t"
    assert db.count() == 3
    db.close()


def test_stats_is_metrics_only_no_raw_text(tmp_path):
    """stats exposes counts/storage but never raw document/query payloads."""

    db = _open(tmp_path)
    db.add("sensitive document text that must never appear in telemetry", id="s")
    stats = db.stats()
    assert stats["document_count"] == 1
    assert stats["raw_payload_text_present"] is False
    blob = repr(stats)
    assert "sensitive document text" not in blob
    db.close()


def test_no_auth_required_and_loopback_only(tmp_path):
    """Local mode requires no credentials and binds the engine to loopback."""

    db = _open(tmp_path)
    # The SDK never asks the caller for a credential — the engine carries no
    # bearer/license/mTLS auth, only a loopback binding and metrics-only telemetry.
    assert not hasattr(db._engine.security, "mtls_required")
    assert not hasattr(db._engine.security, "bearer_token_sha256")
    assert db._engine.security.bind_host in {"127.0.0.1", "localhost"}
    assert db._engine.security.telemetry_mode == "metrics_only"
    db.close()


def test_invalid_inputs_raise_clear_errors(tmp_path):
    """Empty text/query and bad k raise clear ValueErrors before hitting the engine."""

    db = _open(tmp_path)
    with pytest.raises(ValueError):
        db.add("   ")
    with pytest.raises(ValueError):
        db.search("", k=5)
    with pytest.raises(ValueError):
        db.search_many([], k=5)
    with pytest.raises(ValueError):
        db.search_many(["ok", ""], k=5)
    db.add("ok doc", id="ok")
    with pytest.raises(ValueError):
        db.search("ok", k=0)
    with pytest.raises(ValueError):
        db.search_many(["ok"], k=0)
    db.close()


def test_unknown_model_preset_raises(tmp_path):
    """An unknown model preset fails clearly at open."""

    with pytest.raises(ValueError, match="unknown local model preset"):
        LodeDB(path=tmp_path, model="not-a-real-preset")


def test_bge_preset_uses_768_dim_and_query_prefix(tmp_path):
    """The bge preset maps to the 768-dim route profile with its query prefix."""

    from lodedb.local.presets import resolve_preset

    preset = resolve_preset("bge")
    assert preset.native_dim == 768
    assert preset.model_name == "BAAI/bge-base-en-v1.5"
    assert preset.query_prefix.startswith("Represent this sentence")
    # Construct end-to-end with a matching-dim hash backend.
    db = _open(tmp_path, dim=768, model="bge")
    db.add("a document for the quality preset", id="q")
    assert db.count() == 1
    assert db.search("document", k=1)[0].id == "q"
    db.close()


def test_context_manager_closes(tmp_path):
    """LodeDB works as a context manager and persists across the boundary."""

    with _open(tmp_path) as db:
        db.add("ctx doc", id="ctx")
    reopened = _open(tmp_path)
    assert reopened.count() == 1
    reopened.close()
