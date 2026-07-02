"""The public ANN options: local validation and the native create-payload plumbing."""

import pytest

import lodedb
from lodedb import AnnOptions
from lodedb.local.db import _native_vector_index_options, _resolve_ann_options


def test_exact_default_resolves_to_none() -> None:
    assert _resolve_ann_options(None, None, None) is None


def test_ann_options_is_publicly_exported() -> None:
    assert lodedb.AnnOptions is AnnOptions
    assert AnnOptions().algorithm == "cluster"


def test_structured_options_resolve_to_the_core_dict() -> None:
    assert _resolve_ann_options(AnnOptions(), None, None) == {"algorithm": "cluster"}
    assert _resolve_ann_options(AnnOptions(clusters=256, nprobe=16), None, None) == {
        "algorithm": "cluster",
        "clusters": 256,
        "nprobe": 16,
    }
    assert AnnOptions(clusters=4, nprobe=2).to_core_dict() == {
        "algorithm": "cluster",
        "clusters": 4,
        "nprobe": 2,
    }


def test_structured_options_defer_value_checks_to_the_core() -> None:
    # The structured form carries the shape without re-validating knob values; the
    # native core is the single authority and rejects out-of-range values on create.
    assert _resolve_ann_options(AnnOptions(clusters=0), None, None) == {
        "algorithm": "cluster",
        "clusters": 0,
    }


def test_structured_options_reject_mixed_loose_tuning() -> None:
    with pytest.raises(ValueError, match="not both"):
        _resolve_ann_options(AnnOptions(clusters=4), None, 2)


def test_tuning_without_ann_raises() -> None:
    with pytest.raises(ValueError, match="require ann="):
        _resolve_ann_options(None, 4, None)
    with pytest.raises(ValueError, match="require ann="):
        _resolve_ann_options(None, None, 2)


def test_non_positive_tuning_raises() -> None:
    with pytest.raises(ValueError, match="ann_clusters must be a positive integer"):
        _resolve_ann_options("cluster", 0, None)
    with pytest.raises(ValueError, match="ann_nprobe must be a positive integer"):
        _resolve_ann_options("cluster", 4, -1)


def test_resolved_options_carry_only_set_tuning() -> None:
    assert _resolve_ann_options("cluster", None, None) == {"algorithm": "cluster"}
    assert _resolve_ann_options("cluster", 4, 2) == {
        "algorithm": "cluster",
        "clusters": 4,
        "nprobe": 2,
    }


def test_create_payload_omits_ann_when_exact_and_carries_it_when_set() -> None:
    # The persisted state header must stay byte-for-byte unchanged for exact
    # indexes, so the exact payload has no ann key at all.
    base = dict(
        index_id="default",
        index_key="k",
        client_id_hash="k",
        name="lodedb-local",
        model="external",
        provider="external",
        task="vector-only",
        route_profile="vector-only",
        storage_profile="turbovec_direct",
        vector_dim=8,
        bit_width=4,
    )
    exact = _native_vector_index_options(**base, ann=None)
    assert "ann" not in exact
    ann = _native_vector_index_options(
        **base, ann=_resolve_ann_options("cluster", 4, 2)
    )
    assert ann["ann"] == {"algorithm": "cluster", "clusters": 4, "nprobe": 2}
