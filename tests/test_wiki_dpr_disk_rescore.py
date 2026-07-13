"""Correctness tests for benchmark provenance, resume guards, and artifact commits."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "wiki_dpr_disk_rescore"


@pytest.fixture(scope="module")
def modules():
    sys.path.insert(0, str(BENCH_DIR))
    try:
        common = importlib.import_module("common")
        bench = importlib.import_module("lodedb_bench")
        report = importlib.import_module("report")
        yield common, bench, report
    finally:
        sys.path.remove(str(BENCH_DIR))


def test_synthetic_manifest_commits_content_identities(tmp_path, modules) -> None:
    common, _, _ = modules
    manifest = common.make_synthetic_dataset(tmp_path, rows=200, dim=8, n_queries=4, seed=7)

    assert manifest["version"] == 2
    assert len(manifest["corpus_id"]) == 64
    assert len(manifest["evaluation_id"]) == 64
    assert all(manifest["sha256"].values())
    assert manifest["files"]["base"] != common.BASE_NAME
    assert manifest["files"]["queries"] != common.QUERIES_NAME
    assert common.validate_dataset(tmp_path, verify_base_hash=True) == manifest


def test_data_prep_rejects_nonfinite_vectors(modules) -> None:
    sys.path.insert(0, str(BENCH_DIR))
    try:
        data_prep = importlib.import_module("data_prep")
    finally:
        sys.path.remove(str(BENCH_DIR))
    with pytest.raises(ValueError, match="NaN or infinite"):
        data_prep._normalize_rows(np.array([[1.0, np.nan]], dtype=np.float32))


def test_store_and_result_resume_require_exact_provenance(tmp_path, modules) -> None:
    common, bench, _ = modules
    first_data = tmp_path / "first-data"
    second_data = tmp_path / "second-data"
    store = tmp_path / "store"
    common.make_synthetic_dataset(first_data, rows=200, dim=8, n_queries=4, seed=7)
    common.make_synthetic_dataset(second_data, rows=200, dim=8, n_queries=4, seed=8)

    result = bench.run_benchmark(
        data_dir=first_data,
        store_dir=store,
        label="exact",
        loop_seconds=0.05,
        builder_git_sha="a" * 40,
    )
    output = tmp_path / "exact.json"
    bench.write_result_atomic(output, result)
    loaded = bench.load_result_for_resume(
        output,
        label="exact",
        evaluation_id=result["dataset"]["evaluation_id"],
        store_id=result["store"]["provenance"]["store_id"],
        measurement=result["measurement"],
        serve_overrides=result["store"]["serve_overrides"],
    )
    assert loaded["run_id"] == result["run_id"]

    with pytest.raises(ValueError, match="belongs to corpus"):
        bench.run_benchmark(
            data_dir=second_data,
            store_dir=store,
            label="wrong-corpus",
            build=False,
            loop_seconds=0.05,
        )
    with pytest.raises(ValueError, match="requested run"):
        bench.load_result_for_resume(
            output,
            label="exact",
            evaluation_id=result["dataset"]["evaluation_id"],
            store_id=result["store"]["provenance"]["store_id"],
            measurement={**result["measurement"], "loop_seconds_requested": 9.0},
            serve_overrides=result["store"]["serve_overrides"],
        )


def test_store_config_tampering_and_historical_build_fail_closed(tmp_path, modules) -> None:
    common, bench, _ = modules
    data = tmp_path / "data"
    store = tmp_path / "store"
    common.make_synthetic_dataset(data, rows=200, dim=8, n_queries=4, seed=7)
    bench.run_benchmark(
        data_dir=data,
        store_dir=store,
        label="build",
        build=True,
        serve=False,
        builder_git_sha="b" * 40,
    )
    config_path = store / "benchmark_store_config.json"
    config = json.loads(config_path.read_text())
    config["builder"]["layout_id"] = "forged-layout"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config identity is invalid"):
        bench.run_benchmark(
            data_dir=data,
            store_dir=store,
            label="serve",
            build=False,
            loop_seconds=0.05,
        )

    with pytest.raises(ValueError, match="historical artifact"):
        bench.run_benchmark(
            data_dir=data,
            store_dir=tmp_path / "historical",
            label="prechange",
            build=True,
            serve=False,
            buildable=False,
            builder_git_sha="c" * 40,
        )


def test_build_only_result_does_not_require_ground_truth(tmp_path, modules) -> None:
    common, bench, _ = modules
    data = tmp_path / "data"
    store = tmp_path / "store"
    manifest = common.make_synthetic_dataset(data, rows=200, dim=8, n_queries=4, seed=7)
    manifest["files"]["gt_indices"] = None
    manifest["files"]["gt_scores"] = None
    manifest["sha256"]["gt_indices"] = None
    manifest["sha256"]["gt_scores"] = None
    common.write_manifest(data, manifest)

    result = bench.run_benchmark(
        data_dir=data,
        store_dir=store,
        label="build-without-gt",
        build=True,
        serve=False,
        builder_git_sha="d" * 40,
    )

    assert result["dataset"]["gt"] is None
    assert result["dataset"]["evaluation_id"] is None
    bench.validate_result_schema(result)


def test_report_refuses_legacy_unverified_json(tmp_path, modules) -> None:
    _, _, report = modules
    (tmp_path / "legacy.json").write_text(json.dumps({"label": "legacy", "store": {}}))
    with pytest.raises(ValueError, match="legacy-unverified"):
        report.render_report(tmp_path)


def test_sweep_validates_store_and_result_before_resume(tmp_path, modules) -> None:
    common, _, report = modules
    sys.path.insert(0, str(BENCH_DIR))
    try:
        sweep = importlib.import_module("sweep")
    finally:
        sys.path.remove(str(BENCH_DIR))
    data = tmp_path / "data"
    work = tmp_path / "work"
    common.make_synthetic_dataset(data, rows=200, dim=8, n_queries=4, seed=9)
    first = sweep.run_sweep(
        data_dir=data,
        work_dir=work,
        only="exact_bw4",
        loop_seconds=0.05,
        ingest_batch=64,
    )
    second = sweep.run_sweep(
        data_dir=data,
        work_dir=work,
        only="exact_bw4",
        loop_seconds=0.05,
        ingest_batch=64,
    )
    assert first["completed"] == second["completed"] == ["exact_bw4"]
    rendered = report.render_report(work / "results")
    assert "eval id" in rendered
    assert "External published systems are intentionally omitted" in rendered
    assert "Qdrant" not in rendered
