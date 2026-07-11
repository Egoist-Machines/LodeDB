"""Public rescore options, durable identity checks, and session query overrides."""

from __future__ import annotations

import math

import pytest

import lodedb
from lodedb import AnnOptions, RescoreOptions
from lodedb.local.db import _native_vector_index_options, _resolve_rescore_options


def _boundary_vectors() -> tuple[list[tuple[str, list[float]]], list[float]]:
    """Returns the core's deterministic four-bit boundary-ranking fixture."""

    query = [0.43, -0.37, 0.29, -0.23, 0.19, -0.17, 0.13, -0.11]
    query_norm_sq = sum(value * value for value in query)
    documents: list[tuple[str, list[float]]] = []
    for index in range(16):
        vector = [((index * 17 + dimension * 11 + 3) % 29 - 14) * 0.13 for dimension in range(8)]
        projection = (
            sum(left * right for left, right in zip(query, vector, strict=True)) / query_norm_sq
        )
        vector = [
            value - projection * query_value
            for value, query_value in zip(vector, query, strict=True)
        ]
        score = 0.7 + index * 0.0001
        vector = [
            value + score * query_value / query_norm_sq
            for value, query_value in zip(vector, query, strict=True)
        ]
        documents.append((f"boundary-{index:02}", vector))
    return documents, query


def _write_boundary_store(path, **kwargs):
    db = lodedb.LodeDB.open_vector_store(path, vector_dim=8, **kwargs)
    documents, query = _boundary_vectors()
    db.add_vectors_many(
        [{"id": document_id, "vector": vector} for document_id, vector in documents],
        normalize=False,
    )
    return db, documents, query


def _base_native_options() -> dict[str, object]:
    return {
        "index_id": "default",
        "index_key": "k",
        "client_id_hash": "k",
        "name": "lodedb-local",
        "model": "external",
        "provider": "external",
        "task": "vector-only",
        "route_profile": "vector-only",
        "storage_profile": "turbovec_direct",
        "vector_dim": 8,
        "bit_width": 4,
    }


def test_rescore_options_are_public_and_resolve_to_core_payloads() -> None:
    assert lodedb.RescoreOptions is RescoreOptions
    assert _resolve_rescore_options(None, None, None) is None
    assert _resolve_rescore_options(RescoreOptions(), None, None) == {"mode": "original"}
    assert _resolve_rescore_options(
        RescoreOptions(dtype="float32", oversample=3), None, None
    ) == {"mode": "original", "dtype": "float32", "oversample": 3.0}
    assert _resolve_rescore_options("original", "int8", 2) == {
        "mode": "original",
        "dtype": "int8",
        "oversample": 2.0,
    }


def test_rescore_resolution_rejects_ambiguous_and_invalid_loose_forms() -> None:
    with pytest.raises(ValueError, match="not both"):
        _resolve_rescore_options(RescoreOptions(), "float32", None)
    with pytest.raises(ValueError, match="require rescore="):
        _resolve_rescore_options(None, "float32", None)
    with pytest.raises(ValueError, match="require rescore="):
        _resolve_rescore_options(None, None, 2)
    with pytest.raises(ValueError, match="finite and at least 1.0"):
        _resolve_rescore_options("original", None, 0)
    with pytest.raises(ValueError, match="finite and at least 1.0"):
        _resolve_rescore_options("original", None, math.inf)


def test_rescore_dtype_is_an_exact_core_validated_string() -> None:
    # Python deliberately does not translate fp16/fp32 aliases. The native core
    # owns acceptance of the three exact dtype strings.
    assert _resolve_rescore_options("original", "fp16", None) == {
        "mode": "original",
        "dtype": "fp16",
    }


def test_create_payload_omits_or_carries_rescore() -> None:
    exact = _native_vector_index_options(**_base_native_options())
    assert "rescore" not in exact
    rescored = _native_vector_index_options(
        **_base_native_options(),
        rescore=_resolve_rescore_options("original", "float32", 4),
    )
    assert rescored["rescore"] == {"mode": "original", "dtype": "float32", "oversample": 4.0}


def test_float32_rescore_restores_the_original_fp32_order(tmp_path) -> None:
    plain, documents, query = _write_boundary_store(tmp_path / "plain")
    rescored, _, _ = _write_boundary_store(
        tmp_path / "rescored",
        rescore="original",
        rescore_dtype="float32",
        rescore_oversample=32,
    )
    try:
        expected_ids = [
            document_id
            for document_id, _ in sorted(
                documents,
                key=lambda row: (
                    -sum(left * right for left, right in zip(query, row[1], strict=True)),
                    row[0],
                ),
            )[:3]
        ]
        plain_ids = [hit.id for hit in plain.search_by_vector(query, k=3, normalize=False)]
        hits = rescored.search_by_vector(query, k=3, normalize=False)
        assert [hit.id for hit in hits] == expected_ids
        assert plain_ids != expected_ids
        expected_scores = [
            sum(
                left * right
                for left, right in zip(query, dict(documents)[document_id], strict=True)
            )
            for document_id in expected_ids
        ]
        assert [hit.score for hit in hits] == pytest.approx(expected_scores, rel=1e-6)
    finally:
        plain.close()
        rescored.close()


def test_rescore_reopen_checks_identity_and_accepts_oversample_override(tmp_path) -> None:
    store = tmp_path / "rescored"
    db = lodedb.LodeDB.open_vector_store(
        store,
        vector_dim=8,
        rescore="original",
        rescore_dtype="float32",
    )
    db.add_vectors([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], id="one")
    db.close()

    reopened = lodedb.LodeDB.open_vector_store(
        store,
        vector_dim=8,
        rescore="original",
        rescore_dtype="float32",
        rescore_oversample=2,
    )
    try:
        assert [hit.id for hit in reopened.search_by_vector([1.0] + [0.0] * 7)] == ["one"]
    finally:
        reopened.close()

    with pytest.raises(ValueError, match="rescore_dtype does not match"):
        lodedb.LodeDB.open_vector_store(store, vector_dim=8, rescore_dtype="float16")

    plain = tmp_path / "plain"
    unrescored = lodedb.LodeDB.open_vector_store(plain, vector_dim=8)
    unrescored.add_vectors([1.0] + [0.0] * 7, id="one")
    unrescored.close()
    with pytest.raises(ValueError, match="cannot be retro-enabled"):
        lodedb.LodeDB.open_vector_store(plain, vector_dim=8, rescore="original")


def test_reopen_ann_nprobe_override_can_probe_all_clusters(tmp_path) -> None:
    exact, documents, query = _write_boundary_store(tmp_path / "exact")
    ann, _, _ = _write_boundary_store(
        tmp_path / "ann",
        ann="cluster",
        ann_clusters=4,
        ann_nprobe=1,
    )
    try:
        expected_ids = [hit.id for hit in exact.search_by_vector(query, k=5, normalize=False)]
    finally:
        exact.close()
        ann.close()

    # No ann= is needed on reopen: nprobe is now a session override while the
    # persisted algorithm and cluster count remain authoritative.
    reopened = lodedb.LodeDB.open_vector_store(
        tmp_path / "ann", vector_dim=8, ann_nprobe=4
    )
    try:
        actual_ids = [hit.id for hit in reopened.search_by_vector(query, k=5, normalize=False)]
        assert actual_ids == expected_ids
    finally:
        reopened.close()

    structured = lodedb.LodeDB.open_vector_store(
        tmp_path / "ann", vector_dim=8, ann=AnnOptions(nprobe=4)
    )
    try:
        actual_ids = [hit.id for hit in structured.search_by_vector(query, k=5, normalize=False)]
        assert actual_ids == expected_ids
    finally:
        structured.close()
