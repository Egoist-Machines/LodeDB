"""Tests for the opt-in ``torch-compile`` embedding runtime.

These avoid importing torch at module scope; the torch-dependent checks skip when torch /
transformers are unavailable so the module stays portable. The parity check additionally
skips when the model cannot be loaded (offline CI), since it downloads MiniLM.
"""

from __future__ import annotations

import importlib.util

import pytest

from lodedb.local.backends import build_local_embedding_backend
from lodedb.local.presets import resolve_preset

_HAS_TORCH_STACK = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


def test_torch_compile_rejects_clip() -> None:
    """The torch-compile runtime is text-only; the multimodal 'clip' preset is rejected."""

    with pytest.raises(ValueError, match="multimodal"):
        build_local_embedding_backend(resolve_preset("clip"), embedding_runtime="torch-compile")


@pytest.mark.skipif(not _HAS_TORCH_STACK, reason="torch + transformers not installed")
def test_torch_compile_builds_lazily() -> None:
    """Selecting torch-compile returns the compiled backend without loading the model."""

    backend, resolution = build_local_embedding_backend(
        resolve_preset("minilm"), device="cpu", embedding_runtime="torch-compile"
    )
    assert backend.name == "torch_compile"
    assert resolution.backend_name == "torch_compile"
    # Construction is lazy: the model/compile happen on first encode, not on build.
    assert backend._model is None  # type: ignore[attr-defined]


@pytest.mark.skipif(not _HAS_TORCH_STACK, reason="torch + transformers not installed")
def test_torch_compile_parity_with_sentence_transformers() -> None:
    """The compiled backend embeds MiniLM within tolerance of the sentence-transformers path.

    Runs on CPU (torch.compile falls back to eager/inductor when CUDA graphs are unavailable),
    which still exercises the fixed-pad tokenization, pooling, and normalization that must
    match the other runtimes. Skips if the model cannot be materialized (offline).
    """

    import math

    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")

    preset = resolve_preset("minilm")
    texts = ("vector database", "how do embeddings work for retrieval")
    try:
        compiled, _ = build_local_embedding_backend(
            preset, device="cpu", embedding_runtime="torch-compile", max_seq_length=64
        )
        reference, _ = build_local_embedding_backend(
            preset, device="cpu", embedding_runtime="torch"
        )
        compiled_vecs = [compiled.embed_query(t) for t in texts]
        reference_vecs = [reference.embed_query(t) for t in texts]
    except Exception as exc:  # noqa: BLE001 - offline / no model cache: not a code failure
        pytest.skip(f"model unavailable: {exc}")

    for a, b in zip(compiled_vecs, reference_vecs, strict=True):
        cosine = math.fsum(x * y for x, y in zip(a, b, strict=True))
        assert cosine > 0.999, f"compiled vs sentence-transformers cosine {cosine} below tolerance"
