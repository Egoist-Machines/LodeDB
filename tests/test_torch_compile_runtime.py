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


# --- pure helpers (no torch): pad buckets + batch planning ---------------------------------


def test_resolve_pad_buckets_default_and_cap() -> None:
    """Default buckets are (32,64,128,max); buckets >= max collapse to the max cap."""

    from lodedb.engine.embedding_backends import _resolve_pad_buckets

    assert _resolve_pad_buckets(256, None) == (32, 64, 128, 256)
    assert _resolve_pad_buckets(64, None) == (32, 64)  # 128 dropped (>= max), max kept
    assert _resolve_pad_buckets(200, (96, 48, 48)) == (48, 96, 200)  # dedup, sort, append cap
    assert _resolve_pad_buckets(32, None) == (32,)  # everything collapses to the cap


def test_select_bucket() -> None:
    """Selects the smallest bucket that fits; overflow falls to the last (cap) bucket."""

    from lodedb.engine.embedding_backends import _select_bucket

    buckets = (32, 64, 128, 256)
    assert _select_bucket(1, buckets) == 32
    assert _select_bucket(32, buckets) == 32
    assert _select_bucket(33, buckets) == 64
    assert _select_bucket(500, buckets) == 256


def test_plan_batches_groups_by_bucket_and_preserves_indices() -> None:
    """Batches never mix buckets, chunk by batch_size, and cover every index exactly once."""

    from lodedb.engine.embedding_backends import _plan_batches

    lengths = [5, 200, 6, 60, 7]  # buckets: 32, 256, 32, 64, 32
    plan = _plan_batches(lengths, batch_size=2, buckets=(32, 64, 128, 256))
    # bucket 32 has indices [0,2,4] -> chunks (0,2),(4,); bucket 64 -> (3,); bucket 256 -> (1,)
    assert [(b.indices, b.pad_len) for b in plan] == [
        ((0, 2), 32),
        ((4,), 32),
        ((3,), 64),
        ((1,), 256),
    ]
    covered = sorted(i for b in plan for i in b.indices)
    assert covered == [0, 1, 2, 3, 4]


def test_plan_batches_degenerate_same_length() -> None:
    """All-same-length inputs degenerate to plain batch_size chunking in one bucket."""

    from lodedb.engine.embedding_backends import _plan_batches

    plan = _plan_batches([10, 10, 10], batch_size=2, buckets=(32, 256))
    assert [b.indices for b in plan] == [(0, 1), (2,)]
    assert {b.pad_len for b in plan} == {32}


# --- dtype validation + coercion -----------------------------------------------------------


@pytest.mark.parametrize("runtime", ["onnx", "torch", "auto"])
def test_dtype_rejected_for_non_compile_runtimes(runtime: str) -> None:
    """embedding_dtype other than float32 is only valid with the torch-compile runtime."""

    with pytest.raises(ValueError, match="torch-compile"):
        build_local_embedding_backend(
            resolve_preset("minilm"), embedding_runtime=runtime, embedding_dtype="float16"
        )


def test_unknown_dtype_rejected() -> None:
    """A bogus embedding_dtype is a clear error."""

    with pytest.raises(ValueError, match="embedding_dtype"):
        build_local_embedding_backend(
            resolve_preset("minilm"), embedding_runtime="torch-compile", embedding_dtype="int4"
        )


@pytest.mark.skipif(not _HAS_TORCH_STACK, reason="torch + transformers not installed")
def test_fp16_coerced_to_fp32_off_cuda(caplog) -> None:
    """fp16 is honored only on CUDA; on CPU it coerces to float32 and warns (no model load)."""

    import logging

    with caplog.at_level(logging.WARNING, logger="lodedb.engine.embedding"):
        backend, _ = build_local_embedding_backend(
            resolve_preset("minilm"), device="cpu",
            embedding_runtime="torch-compile", embedding_dtype="float16",
        )
    assert backend.dtype == "float16"  # type: ignore[attr-defined]
    assert backend.effective_dtype == "float32"  # type: ignore[attr-defined]


@pytest.mark.skipif(not _HAS_TORCH_STACK, reason="torch + transformers not installed")
def test_bfloat16_cpu_parity() -> None:
    """bf16 on CPU (the CPU-testable half-precision proxy) stays within tolerance of fp32.

    bf16 has ~3 decimal digits of mantissa, so the bar is 0.99, not 0.999.
    """

    import math

    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")
    preset = resolve_preset("minilm")
    texts = ("vector database", "nearest neighbor search over embeddings")
    try:
        half, _ = build_local_embedding_backend(
            preset, device="cpu", embedding_runtime="torch-compile",
            embedding_dtype="bfloat16", max_seq_length=64,
        )
        reference, _ = build_local_embedding_backend(
            preset, device="cpu", embedding_runtime="torch"
        )
        half_vecs = [half.embed_query(t) for t in texts]
        ref_vecs = [reference.embed_query(t) for t in texts]
    except Exception as exc:  # noqa: BLE001 - offline / no model cache
        pytest.skip(f"model unavailable: {exc}")
    for a, b in zip(half_vecs, ref_vecs, strict=True):
        cosine = math.fsum(x * y for x, y in zip(a, b, strict=True))
        assert cosine > 0.99, f"bf16 vs fp32 cosine {cosine} below tolerance"


@pytest.mark.skipif(not _HAS_TORCH_STACK, reason="torch + transformers not installed")
def test_bulk_embed_documents_matches_per_query_across_buckets() -> None:
    """Mixed-length embed_documents (bucketed, scattered) equals per-text embed_query, in order.

    Exercises bucket routing, order preservation, and the batch loop on a real model.
    """

    import math

    preset = resolve_preset("minilm")
    # Lengths span several buckets; batch_size 2 forces multi-batch within a bucket.
    docs = (
        "vector database",
        "a considerably longer passage about retrieval augmented generation and "
        "approximate nearest neighbor search over dense embedding vectors at scale",
        "short",
        "a medium length sentence about compilers and fused kernels",
        "tiny",
    )
    try:
        backend, _ = build_local_embedding_backend(
            preset, device="cpu", embedding_runtime="torch-compile",
            batch_size=2, max_seq_length=64,
        )
        bulk = backend.embed_documents(docs)
        per_text = [backend.embed_query(d) for d in docs]
    except Exception as exc:  # noqa: BLE001 - offline / no model cache
        pytest.skip(f"model unavailable: {exc}")
    assert len(bulk) == len(docs)
    for i, (a, b) in enumerate(zip(bulk, per_text, strict=True)):
        cosine = math.fsum(x * y for x, y in zip(a, b, strict=True))
        assert cosine > 0.9999, f"row {i} bulk vs per-text cosine {cosine}"


@pytest.mark.skipif(
    not _HAS_TORCH_STACK, reason="torch + transformers not installed"
)
def test_fp16_cuda_parity() -> None:
    """fp16 on CUDA stays within tolerance of fp32. Skipped without a CUDA device (CI).

    The binding fp16 parity assertion lives in the Modal spike artifact; this documents intent.
    """

    import math

    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    if importlib.util.find_spec("sentence_transformers") is None:
        pytest.skip("sentence-transformers not installed")
    preset = resolve_preset("minilm")
    texts = ("vector database", "how do embeddings work for retrieval")
    half, _ = build_local_embedding_backend(
        preset, device="cuda", embedding_runtime="torch-compile",
        embedding_dtype="float16", max_seq_length=64,
    )
    reference, _ = build_local_embedding_backend(
        preset, device="cuda", embedding_runtime="torch"
    )
    for t in texts:
        a, b = half.embed_query(t), reference.embed_query(t)
        cosine = math.fsum(x * y for x, y in zip(a, b, strict=True))
        assert cosine > 0.999, f"fp16 vs fp32 cosine {cosine}"
