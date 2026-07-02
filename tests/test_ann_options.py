"""The public ANN options: local validation and the native create-payload plumbing."""

import pytest

from lodedb.local.db import _native_vector_index_options, _resolve_ann_options


def test_exact_default_resolves_to_none() -> None:
    assert _resolve_ann_options(None, None, None) is None


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
