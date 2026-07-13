"""Public rescore options, durable identity checks, and session query overrides."""

from __future__ import annotations

import json
import math

import pytest

import lodedb
from lodedb import AnnOptions, RescoreOptions
from lodedb.local.db import _native_vector_index_options, _resolve_rescore_options
from lodedb.local.doctor import native_store_findings


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


def test_compact_and_stats_expose_ann_rescore_and_mask_skips(tmp_path) -> None:
    """The deployment compact operator persists warmed ANN and rescore state."""

    from lodedb import _turbovec

    store = tmp_path / "compact"
    db = lodedb.LodeDB.open_vector_store(
        store,
        vector_dim=8,
        ann="cluster",
        ann_clusters=4,
        ann_nprobe=1,
        rescore="original",
        rescore_dtype="float32",
        rescore_oversample=4,
    )
    vectors: list[dict[str, object]] = []
    for cluster, axis in enumerate((0, 2, 4, 6)):
        for row in range(32):
            vector = [0.0] * 8
            vector[axis] = 1.0
            vector[(axis + 1) % 8] = (row + 1) * 0.001
            vectors.append({"id": f"c{cluster}-{row}", "vector": vector})
    db.add_vectors_many(vectors, normalize=False)
    try:
        before = db.search_by_vector([1.0] + [0.0] * 7, k=5, normalize=False)
        outcome = db.compact()
        assert outcome == {"ann_warmed": True, "base_rewritten": True}

        stats = db.stats()
        assert stats["rescore"]["dtype"] == "float32"
        assert stats["rescore"]["sidecar_rows"] == len(vectors)
        assert stats["rescore"]["pending_rows"] == 0
        assert stats["ann"] == {
            "clusters": 4,
            "nprobe_effective": 1,
            "cluster_resident": True,
        }

        _turbovec.reset_blocks_skipped_by_mask()
        skips_before = _turbovec.blocks_skipped_by_mask()
        after = db.search_by_vector([1.0] + [0.0] * 7, k=5, normalize=False)
        skips_after = _turbovec.blocks_skipped_by_mask()
        assert skips_after >= skips_before
        assert skips_after > 0
        assert [(hit.id, hit.score) for hit in after] == [
            (hit.id, hit.score) for hit in before
        ]
    finally:
        db.close()


def test_doctor_reports_a_corrupt_tvvf_without_removing_it(tmp_path) -> None:
    """TVVF corruption is a doctor finding and leaves the base vector store openable."""

    store = tmp_path / "doctor"
    db = lodedb.LodeDB.open_vector_store(
        store, vector_dim=8, rescore="original", rescore_dtype="float32"
    )
    db.add_vectors([1.0] + [0.0] * 7, id="one", normalize=False)
    db.persist()
    db.close()

    commit_path = next(store.glob("*.commit.json"))
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    body = commit["body"]
    index_key = body["index_key"]
    vf_epoch = body["tvvf"]["vf_epoch"]
    sidecar = store / f"{index_key}.gen" / f"vf{vf_epoch}.tvvf"
    corrupted = bytearray(sidecar.read_bytes())
    corrupted[-1] ^= 0xFF
    sidecar.write_bytes(corrupted)

    findings = native_store_findings(store)
    finding = next(item for item in findings if item["name"] == "tvvf_sidecar")
    assert finding["status"] == "fail"
    assert "never removes files" in finding["message"]
    assert sidecar.read_bytes() == corrupted

    reopened = lodedb.LodeDB.open_vector_store(store, vector_dim=8)
    try:
        assert reopened.count() == 1
    finally:
        reopened.close()


def test_doctor_validates_a_healthy_int8_sidecar(tmp_path) -> None:
    """The int8 row stride includes the per-row scale, so a healthy store is ok."""

    store = tmp_path / "doctor_int8"
    db = lodedb.LodeDB.open_vector_store(
        store, vector_dim=8, rescore="original", rescore_dtype="int8"
    )
    db.add_vectors([1.0] + [0.0] * 7, id="one", normalize=False)
    db.add_vectors([0.0, 1.0] + [0.0] * 6, id="two", normalize=False)
    db.persist()
    db.close()

    findings = native_store_findings(store)
    finding = next(item for item in findings if item["name"] == "tvvf_sidecar")
    assert finding["status"] == "ok", finding["message"]


def test_doctor_fails_on_a_missing_or_non_directory_store_path(tmp_path) -> None:
    """A mistyped path is a failure finding, not a clean empty report."""

    missing = native_store_findings(tmp_path / "nowhere")
    assert missing and missing[0]["status"] == "fail"
    assert "not an existing directory" in missing[0]["message"]

    regular_file = tmp_path / "file.txt"
    regular_file.write_text("not a store")
    as_file = native_store_findings(regular_file)
    assert as_file and as_file[0]["status"] == "fail"


def test_doctor_rejects_a_tampered_commit_manifest_body(tmp_path) -> None:
    """A body edit that breaks the root checksum cannot yield a healthy TVVF report."""

    store = tmp_path / "doctor_root"
    db = lodedb.LodeDB.open_vector_store(
        store, vector_dim=8, rescore="original", rescore_dtype="float32"
    )
    db.add_vectors([1.0] + [0.0] * 7, id="one", normalize=False)
    db.persist()
    db.close()

    commit_path = next(store.glob("*.commit.json"))
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    commit["body"]["document_count"] = 999
    commit_path.write_text(json.dumps(commit, sort_keys=True), encoding="utf-8")

    findings = native_store_findings(store)
    finding = next(item for item in findings if item["name"] == "tvvf_sidecar")
    assert finding["status"] == "fail"
    assert "checksum" in finding["message"]
