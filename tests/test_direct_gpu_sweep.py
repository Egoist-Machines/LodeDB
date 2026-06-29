"""Tests for the LodeDB native GPU-resident sweep benchmark harness.

The sweep drives the native CUDA GPU-resident scan, which runs in the Rust core
(cudarc) and cannot be faked from Python, so this is a CUDA-host integration test:
it gates on a CUDA-capable runtime and skips cleanly on CI / macOS. The bundled
native extension already exercises the GPU patch + scan directly under
``benchmarks/gpu_patch/modal_rust_gpu_test.py`` on real hardware.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lodedb.engine.gpu_turbovec import gpu_direct_turbovec_dependencies

_BENCH_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "direct_gpu_sweep"
sys.path.insert(0, str(_BENCH_DIR))

from direct_gpu_sweep import run_direct_gpu_sweep  # noqa: E402

# The native GPU scan needs a CUDA GPU; gate on the CuPy probe as a CUDA-presence
# proxy and skip cleanly elsewhere rather than fabricating a device.
_GPU = gpu_direct_turbovec_dependencies()
pytestmark = pytest.mark.skipif(
    not _GPU.available,
    reason=f"native GPU sweep needs a CUDA host: {_GPU.unavailable_reason or 'no CUDA GPU'}",
)


def _dataset(count: int = 320) -> dict:
    """Builds a deterministic one-chunk-per-document dataset above the GPU corpus floor.

    The native GPU-resident scan only engages once the corpus clears the engine's
    minimum (256 rows), so the synthetic set is sized above it.
    """

    documents = []
    queries = []
    for index in range(count):
        doc_id = f"doc-{index:04d}"
        documents.append(
            {
                "id": doc_id,
                "text": f"topic {index} retrieval evidence detail section {index}",
                "metadata": {"group": "synthetic"},
            }
        )
        queries.append(
            {
                "id": f"query-{index:04d}",
                "text": f"topic {index} evidence",
                "document_id": doc_id,
            }
        )
    return {"name": "synthetic", "documents": documents, "queries": queries}


def test_direct_gpu_sweep_cpu_gpu_parity(tmp_path: Path) -> None:
    """Pairs CPU/GPU rows, writes redacted JSON, and the GPU scan matches the CPU scan."""

    summary = run_direct_gpu_sweep(
        output_dir=tmp_path,
        dataset_override=_dataset(),
        persistence_root=tmp_path / "state",
        model="minilm",
        top_k=8,
        query_count=16,
        batch_sizes="1,8",
        query_repeats=1,
        use_hash_backend=True,
    )
    serialized = (tmp_path / "summary.json").read_text(encoding="utf-8")

    assert summary["artifact_type"] == "lodedb_direct_turbovec_gpu_sweep"
    assert summary["raw_payload_text_present"] is False
    assert "retrieval evidence detail" not in serialized  # no raw payload leaks
    assert summary["audit_status"] == "passed"

    rows = {row["row"]: row for row in summary["rows"]}
    # A batch above the GPU minimum (8 >= 4) is GPU-eligible; the GPU scan must rank
    # like the CPU scan (it reads back the same reconstructed rows).
    gpu_row = rows["gpu_batch_8"]
    assert gpu_row["gpu_enabled"] is True
    assert gpu_row["gpu_vs_cpu_top_k_overlap"] >= 0.9
    assert gpu_row["document_recall_gap_vs_cpu"] <= 0.002
    assert (tmp_path / "rows").is_dir()
