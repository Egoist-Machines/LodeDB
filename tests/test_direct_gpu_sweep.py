"""Tests for the LodeDB direct GPU sweep benchmark harness."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from lodedb.engine.gpu_turbovec import (
    GpuDirectTurboVecDependencies,
    gpu_direct_turbovec_dependencies,
)

_BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "direct_gpu_sweep"
sys.path.insert(0, str(_BENCH_DIR))

from direct_gpu_sweep import run_direct_gpu_sweep  # noqa: E402

# The direct-GPU-sweep harness reads the Python engine's resident TurboVec serving
# index (LodeEngine._turbovec_index_for_state). With the native core now the sole
# reader/writer, that Python index is never populated, so the benchmark needs
# porting to the native GPU-resident scan before this can run again.
pytestmark = pytest.mark.skip(
    reason="direct GPU sweep benchmark drives the Python TurboVec serving index, "
    "which the native-core-sole path no longer populates; pending a native GPU port"
)


class _FakeRuntime:
    """Tiny CUDA runtime facade for memory admission and sync."""

    @staticmethod
    def memGetInfo() -> tuple[int, int]:
        return (1 << 40, 1 << 40)

    @staticmethod
    def deviceSynchronize() -> None:
        return None


class _FakeCuda:
    runtime = _FakeRuntime()


class _FakeCupy:
    """Numpy-backed subset of CuPy used by the GPU direct path."""

    cuda = _FakeCuda()
    float32 = np.float32
    float16 = np.float16

    @staticmethod
    def asarray(value):
        return np.asarray(value)

    @staticmethod
    def empty(shape, dtype):
        return np.empty(shape, dtype=dtype)

    @staticmethod
    def argpartition(value, kth, axis):
        return np.argpartition(value, kth, axis=axis)

    @staticmethod
    def take_along_axis(arr, indices, axis):
        return np.take_along_axis(arr, indices, axis=axis)

    @staticmethod
    def asnumpy(value):
        return np.asarray(value)


@pytest.fixture
def fake_gpu_dependencies(monkeypatch):
    """Installs fake CuPy and forces the CuPy fallback top-k path."""

    from lodedb.engine import gpu_turbovec

    monkeypatch.setattr(gpu_turbovec, "_maybe_import_torch", lambda: None)
    gpu_direct_turbovec_dependencies._override = GpuDirectTurboVecDependencies(
        available=True,
        cupy=_FakeCupy(),
        unavailable_reason="",
    )
    try:
        yield
    finally:
        del gpu_direct_turbovec_dependencies._override


def _dataset(count: int = 8) -> dict:
    """Builds a deterministic one-chunk-per-document test dataset."""

    documents = []
    queries = []
    for index in range(count):
        doc_id = f"doc-{index:03d}"
        documents.append(
            {
                "id": doc_id,
                "text": f"topic {index} retrieval evidence detail section {index}",
                "metadata": {"group": "synthetic"},
            }
        )
        queries.append(
            {
                "id": f"query-{index:03d}",
                "text": f"topic {index} evidence",
                "document_id": doc_id,
            }
        )
    return {"name": "synthetic", "documents": documents, "queries": queries}


def test_direct_gpu_sweep_offline_fake_gpu(tmp_path: Path, fake_gpu_dependencies) -> None:
    """The sweep pairs CPU/GPU rows, writes redacted JSON, and checks memory policy."""

    summary = run_direct_gpu_sweep(
        output_dir=tmp_path,
        dataset_override=_dataset(),
        persistence_root=tmp_path / "state",
        model="minilm",
        top_k=8,
        query_count=4,
        batch_sizes="1,4",
        query_repeats=1,
        use_hash_backend=True,
    )
    serialized = (tmp_path / "summary.json").read_text(encoding="utf-8")

    assert summary["artifact_type"] == "lodedb_direct_turbovec_gpu_sweep"
    assert summary["raw_payload_text_present"] is False
    assert "retrieval evidence detail" not in serialized
    assert summary["audit_status"] == "passed"

    rows = {row["row"]: row for row in summary["rows"]}
    assert rows["gpu_direct_batch_1"]["gpu_stage_one_status"] in {
        "",
        "bypassed",
        "not_applicable",
    }
    gpu_row = rows["gpu_direct_batch_4"]
    assert gpu_row["gpu_stage_one_status"] == "used"
    assert gpu_row["stage_one_backend"] == "gpu_cupy_exact_direct"
    assert gpu_row["gpu_vs_cpu_top_k_overlap"] >= 0.9
    assert gpu_row["document_recall_gap_vs_cpu"] <= 0.002

    assert rows["gpu_direct_auto_memory_rejected"]["gpu_stage_one_status"] == (
        "memory_rejected"
    )
    assert rows["gpu_direct_required_memory_rejected"]["failed_closed"] is True
    assert (tmp_path / "rows").is_dir()
