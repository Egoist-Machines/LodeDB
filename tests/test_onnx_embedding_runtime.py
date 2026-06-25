"""Tests for the ONNX embedding runtime and runtime selection.

Three layers:

- Unit tests for :class:`ONNXRuntimeEmbeddingBackend` with a fake ONNX session +
  tokenizer, so pooling/normalization/validation are covered without installing
  onnxruntime or downloading a model (these run everywhere, including CI).
- Runtime-selection tests for :func:`build_local_embedding_backend`, with the
  ONNX path monkeypatched, covering auto-prefers-ONNX, torch fallback, and the
  forced runtimes.
- Real-model parity + end-to-end tests against the actual MiniLM ONNX export.
  These download a model, so they are opt-in via ``LODEDB_RUN_ONNX_MODEL_TESTS=1``
  and skipped otherwise.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from lodedb.engine.embedding_backends import (
    ONNXRuntimeEmbeddingBackend,
    _pool_onnx_output,
)
from lodedb.local import backends as backends_mod
from lodedb.local.backends import build_local_embedding_backend
from lodedb.local.onnx_artifacts import OnnxArtifact, OnnxMaterializationError
from lodedb.local.presets import resolve_preset

_RUN_MODEL_TESTS = os.environ.get("LODEDB_RUN_ONNX_MODEL_TESTS") == "1"
_model_only = pytest.mark.skipif(
    not _RUN_MODEL_TESTS,
    reason="set LODEDB_RUN_ONNX_MODEL_TESTS=1 to run tests that download a real ONNX model",
)


class _FakeONNXInput:
    """A fake ONNX graph input/output descriptor exposing a ``name``."""

    def __init__(self, name: str) -> None:
        self.name = name


class _FakeONNXSession:
    """Returns deterministic token embeddings without importing ONNX Runtime."""

    def get_inputs(self) -> list[_FakeONNXInput]:
        return [_FakeONNXInput("input_ids"), _FakeONNXInput("attention_mask")]

    def get_outputs(self) -> list[_FakeONNXInput]:
        return [_FakeONNXInput("last_hidden_state")]

    def run(self, output_names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch, seq = inputs["input_ids"].shape
        values = np.arange(batch * seq * 4, dtype=np.float32).reshape(batch, seq, 4)
        return [values]


class _FakeTokenizer:
    """Builds deterministic tokenizer arrays for backend unit tests."""

    def __call__(
        self,
        texts: list[str],
        *,
        padding: bool,
        truncation: bool,
        max_length: int,
        return_tensors: str,
    ) -> dict[str, np.ndarray]:
        assert padding is True
        assert truncation is True
        assert return_tensors == "np"
        width = 3
        return {
            "input_ids": np.ones((len(texts), width), dtype=np.int64),
            "attention_mask": np.ones((len(texts), width), dtype=np.int64),
        }


def _fake_backend(**overrides) -> ONNXRuntimeEmbeddingBackend:
    """Builds an ONNX backend wired to the fake session + tokenizer."""

    params = {
        "model_name": "fake-model",
        "native_dim": 4,
        "onnx_model_path": "fake.onnx",
        "tokenizer_name_or_path": "fake-tokenizer",
        "batch_size": 2,
        "max_seq_length": 8,
        "pooling": "cls",
    }
    params.update(overrides)
    backend = ONNXRuntimeEmbeddingBackend(**params)
    backend._session = _FakeONNXSession()
    backend._tokenizer = _FakeTokenizer()
    return backend


def test_onnx_backend_pools_normalizes_and_validates_shape() -> None:
    """The ONNX backend applies prefixes, pools, normalizes, and returns the native dim."""

    backend = _fake_backend(query_prefix="query: ", document_prefix="doc: ")

    documents = backend.embed_documents(("alpha", "beta"))
    query = backend.embed_query("alpha")

    assert len(documents) == 2
    assert all(len(row) == 4 for row in documents)
    assert len(query) == 4
    assert np.linalg.norm(np.asarray(query, dtype=np.float32)) == pytest.approx(1.0)


def test_onnx_backend_validates_native_dim() -> None:
    """A native_dim that disagrees with the model output is a deterministic error."""

    backend = _fake_backend(native_dim=8)  # fake session emits dim 4
    with pytest.raises(ValueError, match="expected 8"):
        backend.embed_documents(("alpha",))


def test_onnx_mean_pooling_uses_attention_mask() -> None:
    """Mean pooling ignores masked token positions in ONNX token outputs."""

    outputs = {
        "last_hidden_state": np.asarray(
            [[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]],
            dtype=np.float32,
        )
    }
    mask = np.asarray([[1, 1, 0]], dtype=np.int64)

    pooled = _pool_onnx_output(outputs, attention_mask=mask, pooling="mean", output_name=None)

    assert pooled.tolist() == [[2.0, 2.0]]


def test_onnx_backend_rejects_unknown_pooling() -> None:
    """Pooling must be one of the supported strategies."""

    with pytest.raises(ValueError, match="pooling must be"):
        ONNXRuntimeEmbeddingBackend(
            model_name="m",
            native_dim=4,
            onnx_model_path="m.onnx",
            tokenizer_name_or_path="t",
            pooling="max",
        )


# -- runtime selection ------------------------------------------------------


def _patch_onnx(monkeypatch, *, available: bool, materialize=None) -> None:
    """Stubs the ONNX availability + materialization + provider probe in backends."""

    monkeypatch.setattr(backends_mod, "onnxruntime_available", lambda: available)
    monkeypatch.setattr(
        backends_mod, "_resolve_onnx_providers", lambda device: ("CPUExecutionProvider",)
    )
    if materialize is not None:
        monkeypatch.setattr(backends_mod, "materialize_onnx_model", materialize)


def test_auto_prefers_onnx_when_available(monkeypatch) -> None:
    """``embedding_runtime='auto'`` builds the ONNX backend when onnxruntime is present."""

    preset = resolve_preset("minilm")

    def fake_materialize(model_name: str) -> OnnxArtifact:
        return OnnxArtifact(model_name, "model.onnx", "tok", source="cached")

    _patch_onnx(monkeypatch, available=True, materialize=fake_materialize)
    backend, resolution = build_local_embedding_backend(preset, device="cpu")

    assert backend.name == "onnx_runtime"
    assert resolution.backend_name == "onnx_runtime"
    assert resolution.fallback_used is False


def test_auto_falls_back_to_torch_when_onnx_unmaterializable(monkeypatch) -> None:
    """If the ONNX artifact cannot be obtained, ``auto`` falls back to sentence-transformers."""

    preset = resolve_preset("minilm")

    def boom(model_name: str):
        raise OnnxMaterializationError("no artifact")

    _patch_onnx(monkeypatch, available=True, materialize=boom)
    backend, resolution = build_local_embedding_backend(preset, device="cpu")

    assert backend.name == "sentence_transformers"
    assert resolution.fallback_used is True
    assert "torch" in resolution.fallback_reason


def test_auto_uses_torch_when_onnxruntime_missing(monkeypatch) -> None:
    """Without onnxruntime installed, ``auto`` uses the torch backend and says why."""

    preset = resolve_preset("minilm")
    _patch_onnx(monkeypatch, available=False)
    backend, resolution = build_local_embedding_backend(preset, device="cpu")

    assert backend.name == "sentence_transformers"
    assert resolution.fallback_used is True
    assert "onnxruntime not installed" in resolution.fallback_reason


def test_forced_onnx_raises_when_unmaterializable(monkeypatch) -> None:
    """``embedding_runtime='onnx'`` surfaces the error instead of falling back."""

    preset = resolve_preset("minilm")

    def boom(model_name: str):
        raise OnnxMaterializationError("no artifact")

    _patch_onnx(monkeypatch, available=True, materialize=boom)
    with pytest.raises(OnnxMaterializationError):
        build_local_embedding_backend(preset, device="cpu", embedding_runtime="onnx")


def test_forced_torch_skips_onnx(monkeypatch) -> None:
    """``embedding_runtime='torch'`` builds sentence-transformers even if ONNX is available."""

    preset = resolve_preset("minilm")

    def fail(model_name: str):  # must not be called
        raise AssertionError("ONNX must not be attempted for runtime='torch'")

    _patch_onnx(monkeypatch, available=True, materialize=fail)
    backend, resolution = build_local_embedding_backend(
        preset, device="cpu", embedding_runtime="torch"
    )

    assert backend.name == "sentence_transformers"
    assert resolution.fallback_used is False


def test_unknown_runtime_is_rejected() -> None:
    """An unknown runtime name is a clear error."""

    preset = resolve_preset("minilm")
    with pytest.raises(ValueError, match="unknown embedding_runtime"):
        build_local_embedding_backend(preset, embedding_runtime="tensorflow")


def test_doctor_reports_embedding_runtime() -> None:
    """The doctor report carries the preferred embedding runtime, fallback note, and providers."""

    from lodedb.local.doctor import format_capability_report, local_capability_report

    report = local_capability_report(device="cpu")
    runtime = report["embedding"]["runtime"]
    assert runtime["preferred"] in {"onnx", "torch"}
    assert isinstance(runtime["onnx_providers"], list)
    # The report states a preference with an explicit fallback, not a guaranteed resolution.
    assert runtime["note"]
    assert "runtime (auto prefers)" in format_capability_report(report)


def test_onnx_providers_default_to_cpu_without_coreml(monkeypatch) -> None:
    """Core ML is off by default: cpu and mps both prefer the CPU provider; CUDA prefers CUDA."""

    from lodedb.local.backends import _preferred_onnx_providers

    monkeypatch.delenv("LODEDB_ONNX_COREML", raising=False)
    assert _preferred_onnx_providers("cpu") == ("CPUExecutionProvider",)
    assert _preferred_onnx_providers("mps") == ("CPUExecutionProvider",)
    assert _preferred_onnx_providers("cuda") == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_onnx_coreml_provider_is_opt_in(monkeypatch) -> None:
    """LODEDB_ONNX_COREML=1 puts the Core ML provider first for an Apple (mps) device only."""

    from lodedb.local.backends import _preferred_onnx_providers

    monkeypatch.setenv("LODEDB_ONNX_COREML", "1")
    assert _preferred_onnx_providers("mps") == ("CoreMLExecutionProvider", "CPUExecutionProvider")
    assert _preferred_onnx_providers("cpu") == ("CPUExecutionProvider",)


# -- real model (opt-in) ----------------------------------------------------


@_model_only
def test_onnx_matches_sentence_transformers_on_minilm() -> None:
    """ONNX MiniLM embeddings match the sentence-transformers reference (cosine ~1.0)."""

    pytest.importorskip("onnxruntime")
    from lodedb.engine.embedding_backends import SentenceTransformerEmbeddingBackend
    from lodedb.local.onnx_artifacts import materialize_onnx_model

    preset = resolve_preset("minilm")
    artifact = materialize_onnx_model(preset.model_name)
    onnx_backend = ONNXRuntimeEmbeddingBackend(
        model_name=preset.model_name,
        native_dim=preset.native_dim,
        onnx_model_path=artifact.model_path,
        tokenizer_name_or_path=str(artifact.tokenizer_dir),
        pooling=preset.pooling,
        max_seq_length=256,
    )
    torch_backend = SentenceTransformerEmbeddingBackend(
        model_name=preset.model_name,
        native_dim=preset.native_dim,
        device="cpu",
        max_seq_length=256,
    )

    texts = ("the quick brown fox", "a slow green turtle", "vector databases are fast")
    onnx_vecs = np.asarray(onnx_backend.embed_documents(texts), dtype=np.float32)
    torch_vecs = np.asarray(torch_backend.embed_documents(texts), dtype=np.float32)

    cosines = np.sum(onnx_vecs * torch_vecs, axis=1)  # both are L2-normalized
    assert float(cosines.min()) > 0.99


@_model_only
def test_lodedb_end_to_end_with_onnx_runtime(tmp_path) -> None:
    """A LodeDB opened with the ONNX runtime indexes and searches end to end."""

    pytest.importorskip("onnxruntime")
    from lodedb.local.db import LodeDB

    db = LodeDB(path=tmp_path / "store", model="minilm", device="cpu", embedding_runtime="onnx")
    assert db.embedding_resolution.backend_name == "onnx_runtime"
    db.add("the quick brown fox", metadata={"topic": "animals"})
    db.add("structured query languages", metadata={"topic": "databases"})
    hits = db.search("fox", k=2)
    assert hits[0].metadata["topic"] == "animals"
    db.close()
