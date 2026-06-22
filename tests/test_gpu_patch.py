from __future__ import annotations

import numpy as np
import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.gpu_turbovec import (
    GpuDirectTurboVecDependencies,
    gpu_direct_turbovec_dependencies,
)
from lodedb.local.db import LodeDB


class _FakeRuntime:
    @staticmethod
    def memGetInfo() -> tuple[int, int]:
        return (1 << 40, 1 << 40)

    @staticmethod
    def deviceSynchronize() -> None:
        return None


class _FakeCuda:
    runtime = _FakeRuntime()


class _FakeCupy:
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


def test_gpu_incremental_patch_in_place(tmp_path, fake_gpu_dependencies, monkeypatch) -> None:
    # Force the GPU direct serving policy to be REQUIRED so LodeEngine executes the GPU direct path
    monkeypatch.setenv("LODEDB_GPU_DIRECT_TURBOVEC", "required")

    # Open database using the dummy hash embedding backend
    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )

    # 1. Add initial documents
    doc_id_a = db.add("first doc", id="doc-a")
    _doc_id_b = db.add("second doc", id="doc-b")
    assert db.count() == 2

    # 2. Force initialization of GPU session by running a batched search
    # Batch size >= 2 is required by the direct GPU policy
    results = db.search_many(["first", "second"], k=1)
    assert len(results) == 2

    # Verify that the GPU session was built
    sessions = db._engine._gpu_direct_turbovec_sessions
    assert len(sessions) == 1
    state_key = list(sessions.keys())[0]
    session_before = sessions[state_key]
    assert session_before is not None
    assert session_before.row_count == 2
    session_id_before = id(session_before)

    # 3. Add a third document (an upsert)
    _doc_id_c = db.add("third doc", id="doc-c")
    assert db.count() == 3

    # Check that the GPU session was updated and is the exact same object (patched in-place)
    session_after_add = sessions[state_key]
    assert id(session_after_add) == session_id_before
    assert session_after_add.row_count == 3
    # The session's resident stable IDs must exactly match the CPU index's IDs
    # (extension-agnostic: no reliance on a mock-only `.vectors` attribute).
    serving = db._engine._turbovec_indexes[state_key]
    assert session_after_add.generation == serving.generation
    expected_ids = set(serving.chunk_ids_by_stable_id.keys())
    resident_ids = {int(x) for x in session_after_add.stable_ids[: session_after_add.row_count]}
    assert resident_ids == expected_ids

    # 4. Remove a document
    db.remove(doc_id_a)
    assert db.count() == 2

    # Check that the GPU session is still the same object and was patched in-place for removal
    session_after_remove = sessions[state_key]
    assert id(session_after_remove) == session_id_before
    assert session_after_remove.row_count == 2
    serving = db._engine._turbovec_indexes[state_key]
    assert session_after_remove.generation == serving.generation

    # Run another batched search to confirm everything still queries correctly
    results_after = db.search_many(["second", "third"], k=1)
    assert len(results_after) == 2
    assert sessions[state_key] is session_before
