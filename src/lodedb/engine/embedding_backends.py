"""Embedding backends for the LodeDB engine."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

logger = logging.getLogger("lodedb.engine.embedding")


class EngineEmbeddingBackend(Protocol):
    """Defines the document/query embedding API used by the local engine."""

    name: str
    native_dim: int
    required_model_name: str | None

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Embeds document chunks into normalized vectors for local indexing."""

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embeds one retrieval query into a normalized vector."""


class HashEmbeddingBackend:
    """Builds deterministic fixture embeddings for tests and offline validation."""

    name = "hash_fixture"
    required_model_name = None

    def __init__(self, *, native_dim: int) -> None:
        """Initializes the fixture backend with the required embedding dimension."""

        if native_dim <= 0:
            raise ValueError("native_dim must be positive")
        self.native_dim = native_dim

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Embeds document chunks with deterministic normalized hash vectors."""

        return tuple(hash_embedding(text, self.native_dim) for text in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embeds one query with the same deterministic fixture transform."""

        return hash_embedding(text, self.native_dim)


class SentenceTransformerEmbeddingBackend:
    """Embeds text with a locally hosted SentenceTransformers model."""

    name = "sentence_transformers"

    def __init__(
        self,
        *,
        model_name: str,
        native_dim: int,
        device: str = "cuda",
        batch_size: int = 16,
        max_seq_length: int | None = 512,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        """Initializes a lazy local embedding backend for GPU engine deployment."""

        if not model_name:
            raise ValueError("model_name is required")
        if native_dim <= 0:
            raise ValueError("native_dim must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.required_model_name = model_name
        self.native_dim = native_dim
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model: object | None = None

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Embeds document chunks with the configured local model and document prefix."""

        return self._encode(tuple(f"{self.document_prefix}{text}" for text in texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embeds one query with the configured local model and query prefix."""

        return self._encode((f"{self.query_prefix}{text}",))[0]

    def _encode(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Encodes and normalizes a batch of texts, returning immutable vectors."""

        if not texts:
            return ()
        model = self._load_model()
        embeddings = model.encode(  # type: ignore[attr-defined]
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.native_dim:
            returned_dim = array.shape[1] if array.ndim == 2 else "unknown"
            raise ValueError(
                f"{self.model_name} returned dim {returned_dim}; expected {self.native_dim}"
            )
        return tuple(tuple(float(value) for value in row) for row in array)

    def _load_model(self) -> object:
        """Loads the SentenceTransformers model lazily to keep tests dependency-light."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name, device=self.device)
            if self.max_seq_length is not None:
                model.max_seq_length = int(self.max_seq_length)
            self._model = model
        return self._model


_EMBEDDING_DTYPES = ("float32", "float16", "bfloat16")


@dataclass(frozen=True)
class _PlannedBatch:
    """One embedding batch: original indices to fill, and the pad length they share."""

    indices: tuple[int, ...]
    pad_len: int


def _resolve_pad_buckets(
    max_seq_length: int, pad_buckets: tuple[int, ...] | None
) -> tuple[int, ...]:
    """Returns the ascending pad-length buckets, always capped by (and ending at) max_seq_length.

    ``None`` derives ``(32, 64, 128, max_seq_length)``. Buckets above ``max_seq_length`` are
    dropped and ``max_seq_length`` is always included as the final (truncation) bucket, so a
    text longer than every smaller bucket pads to the cap rather than overflowing it.
    """

    candidates = (32, 64, 128) if pad_buckets is None else tuple(pad_buckets)
    buckets = sorted({int(b) for b in candidates if 0 < int(b) < max_seq_length})
    return (*buckets, max_seq_length)


def _select_bucket(longest_len: int, buckets: tuple[int, ...]) -> int:
    """Returns the smallest bucket that fits ``longest_len`` (the last bucket is the cap)."""

    for bucket in buckets:
        if longest_len <= bucket:
            return bucket
    return buckets[-1]


def _plan_batches(
    lengths: list[int], batch_size: int, buckets: tuple[int, ...]
) -> list[_PlannedBatch]:
    """Groups texts by pad bucket, then chunks each group into ``batch_size`` batches.

    Grouping by bucket means a batch never mixes pad lengths (short texts stop paying the
    longest text's pad), and every batch carries its original indices so the caller can scatter
    results back into input order. Relative order within a bucket is preserved (stable).
    """

    by_bucket: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        by_bucket.setdefault(_select_bucket(length, buckets), []).append(index)
    batches: list[_PlannedBatch] = []
    for bucket in sorted(by_bucket):
        indices = by_bucket[bucket]
        for start in range(0, len(indices), batch_size):
            batches.append(_PlannedBatch(tuple(indices[start : start + batch_size]), bucket))
    return batches


def _resolve_embedding_dtype(requested: str, device: str) -> tuple[str, str]:
    """Resolves a requested embedding dtype to the one usable on ``device``.

    Returns ``(effective_dtype, warning_or_empty)``. CUDA honors fp16/bf16; CPU honors bf16 but
    coerces fp16 to fp32 (fp16 CPU inference is slow/patchy); MPS coerces both to fp32
    (conservative). The message is returned rather than logged so this stays torch-free/testable.
    """

    if requested == "float32":
        return "float32", ""
    if device == "cuda":
        return requested, ""
    if device == "cpu" and requested == "bfloat16":
        return "bfloat16", ""
    reason = "fp16 CPU inference is slow/patchy" if device == "cpu" else "MPS half is untested here"
    return "float32", (
        f"embedding_dtype={requested} is not used on device={device}; running float32 ({reason})."
    )


class CompiledTorchEmbeddingBackend:
    """Embeds text with a ``torch.compile``d Hugging Face encoder for low single-query latency.

    An opt-in fast path for GPU serving. It loads the raw HF encoder (``AutoModel``), wraps
    encoder + pooling + L2 normalization in one module, and runs that through ``torch.compile``
    so a stock compiler fuses the whole forward and (on CUDA, with ``mode="reduce-overhead"``)
    replays it as a CUDA graph, removing per-op launch overhead. The embedding spike measured
    the forward at roughly 4x and end-to-end ``embed_query`` at ~1.3x (A10) / ~1.4x (L40S) vs
    the ONNX-CUDA default; tokenization and the device->host copy do not compile away, which is
    why the end-to-end gain is smaller than the forward gain.

    Optionally runs the model in half precision (``dtype="float16"``/``"bfloat16"``, CUDA), which
    halves the weight bytes streamed per forward. Half-precision embeddings are not bit-identical
    to fp32 (measured cosine ~0.999 on MiniLM); pooling output is cast to fp32 before
    normalization and the backend always returns fp32, so a store built at one dtype stays
    searchable from another.

    Correctness is otherwise preserved: it reproduces the same computation as the ONNX /
    sentence-transformers backends (encoder, then mean pooling over the attention mask or CLS
    pooling, then L2 normalization). CUDA graphs require a static input shape, so inputs are
    padded to one of a few length buckets (mean/CLS pooling ignores the padding via the
    attention mask, so parity is unaffected). ``torch`` and ``transformers`` import lazily, so a
    plain ``import lodedb`` never pays for them. This is the ``lodedb[torch]`` tier.
    """

    name = "torch_compile"

    def __init__(
        self,
        *,
        model_name: str,
        native_dim: int,
        device: str = "cuda",
        batch_size: int = 32,
        max_seq_length: int = 256,
        query_prefix: str = "",
        document_prefix: str = "",
        pooling: str = "mean",
        normalize: bool = True,
        compile_mode: str = "reduce-overhead",
        dtype: str = "float32",
        pad_buckets: tuple[int, ...] | None = None,
    ) -> None:
        """Stores config while keeping the torch import and model load/compile lazy."""

        if not model_name:
            raise ValueError("model_name is required")
        if native_dim <= 0:
            raise ValueError("native_dim must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be cls or mean")
        if dtype not in _EMBEDDING_DTYPES:
            raise ValueError(f"dtype must be one of {_EMBEDDING_DTYPES}")
        self.model_name = model_name
        self.required_model_name = model_name
        self.native_dim = native_dim
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.pooling = pooling
        self.normalize = normalize
        self.compile_mode = compile_mode
        self.dtype = dtype
        self.effective_dtype, self._dtype_warning = _resolve_embedding_dtype(dtype, device)
        self.pad_buckets = _resolve_pad_buckets(max_seq_length, pad_buckets)
        self._model: object | None = None
        self._tokenizer: object | None = None
        self._compiled = False

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Embeds document chunks with the configured document prefix."""

        return self._encode(tuple(f"{self.document_prefix}{text}" for text in texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embeds one retrieval query with the configured query prefix."""

        return self._encode((f"{self.query_prefix}{text}",))[0]

    def _encode(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Tokenizes, buckets, runs the fused compiled encoder, and scatters back into order."""

        if not texts:
            return ()
        import torch

        model = self._load_model()
        tokenizer = self._load_tokenizer()
        encoded = tokenizer(
            list(texts), padding=False, truncation=True, max_length=self.max_seq_length
        )
        lengths = [len(ids) for ids in encoded["input_ids"]]
        keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in encoded]
        plan = _plan_batches(lengths, self.batch_size, self.pad_buckets)
        # Pad the batch dimension to batch_size only for genuinely bulk CUDA calls with a live
        # CUDA graph: it collapses the per-shape captures to {1, batch_size} x buckets. A small
        # single-batch call (e.g. add() with a few chunks) is left at its own size so it does
        # not do batch_size-worth of work.
        pad_to_full = self._compiled and self.device == "cuda" and len(texts) > self.batch_size

        out = np.empty((len(texts), self.native_dim), dtype=np.float32)
        for batch in plan:
            slice_encoded = {k: [encoded[k][i] for i in batch.indices] for k in keys}
            tokens = tokenizer.pad(
                slice_encoded,
                padding="max_length",
                max_length=batch.pad_len,
                return_tensors="pt",
            )
            real = len(batch.indices)
            feed = {k: tokens[k].to(self.device) for k in keys}
            if pad_to_full and real < self.batch_size:
                pad_rows = self.batch_size - real
                feed = {
                    k: torch.cat([v, v[:1].expand(pad_rows, -1)], dim=0) for k, v in feed.items()
                }
            with torch.no_grad():
                pooled = model(**feed)
            # reduce-overhead reuses the CUDA-graph output buffer on the next replay, so the
            # result must leave the device before the next batch runs: .cpu() copies it out.
            pooled_cpu = pooled[:real].detach().to("cpu").numpy().astype(np.float32, copy=False)
            for row, index in enumerate(batch.indices):
                out[index] = pooled_cpu[row]

        _validate_embedding_array(out, native_dim=self.native_dim, model_name=self.model_name)
        return tuple(tuple(float(value) for value in row) for row in out)

    def _load_model(self) -> object:
        """Loads, fuses (encoder + pool + norm), and compiles the encoder lazily."""

        if self._model is not None:
            return self._model
        try:
            import torch
            from torch import nn
            from torch.nn import functional as functional_nn
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError(
                "the torch-compile embedding runtime needs torch + transformers "
                "(install lodedb[embeddings,torch])."
            ) from exc

        if self._dtype_warning:
            logger.warning("%s", self._dtype_warning)
        # bf16 needs an Ampere-or-newer GPU (sm_80+); older CUDA cards (e.g. T4/V100) would
        # accept the .to(bfloat16) cast but fail in the kernels. Fall back to fp16 there, which
        # every CUDA GPU LodeDB targets supports. (Resolved here, not in the torch-free helper.)
        if (
            self.effective_dtype == "bfloat16"
            and self.device == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            logger.warning(
                "bfloat16 is not supported on this CUDA device; using float16 instead."
            )
            self.effective_dtype = "float16"
        torch_dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.effective_dtype]

        pooling, normalize = self.pooling, self.normalize

        class _FusedSentenceEncoder(nn.Module):
            """Encoder + pooling + L2 normalize in one module, so the CUDA graph covers it all."""

            def __init__(self, encoder: object) -> None:
                super().__init__()
                self.encoder = encoder

            def forward(self, input_ids, attention_mask, token_type_ids=None):  # type: ignore[no-untyped-def]
                kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
                if token_type_ids is not None:
                    kwargs["token_type_ids"] = token_type_ids
                tokens = self.encoder(**kwargs).last_hidden_state
                if pooling == "cls":
                    pooled = tokens[:, 0, :]
                else:
                    mask = attention_mask.unsqueeze(-1).to(tokens.dtype)
                    summed = (tokens * mask).sum(dim=1)
                    counts = torch.clamp(mask.sum(dim=1), min=1.0)
                    pooled = summed / counts
                # Cast to fp32 before normalizing so half-precision norms stay stable, and so the
                # backend always returns fp32. eps=1e-12 matches _l2_normalize_rows on zero rows.
                pooled = pooled.float()
                if normalize:
                    pooled = functional_nn.normalize(pooled, p=2, dim=1, eps=1e-12)
                return pooled

        encoder = AutoModel.from_pretrained(self.model_name)
        encoder = encoder.to(device=self.device, dtype=torch_dtype)
        encoder.eval()
        fused = _FusedSentenceEncoder(encoder).eval()
        # dynamic=False specializes one graph per (batch, bucket) shape, so the encoder
        # legitimately needs more than dynamo's default 8 cached graphs (a few buckets x a few
        # batch dims, across every compiled backend built in the process, all sharing this one
        # forward code object). Raise the limit so later shapes stay compiled instead of silently
        # falling back to eager once the cache fills.
        try:
            import torch._dynamo as _dynamo

            _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
        except Exception:  # noqa: BLE001 - config knob is best-effort
            pass
        # reduce-overhead uses CUDA graphs (CUDA only); off CUDA fall back to the default inductor
        # mode so the path still works (smaller gain) on CPU/MPS.
        mode = self.compile_mode
        if mode == "reduce-overhead" and self.device != "cuda":
            mode = "default"
        try:
            self._model = torch.compile(fused, mode=mode, dynamic=False)
            self._compiled = True
        except Exception:  # noqa: BLE001 - no compiler backend on this platform: run eager
            self._model = fused
            self._compiled = False
        return self._model

    def _load_tokenizer(self) -> object:
        """Loads the Hugging Face tokenizer lazily to avoid import cost on a plain import."""

        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional runtime
                raise RuntimeError(
                    "transformers is required to tokenize for the torch-compile runtime."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer


class ClipEmbeddingBackend:
    """Embeds text and images into one shared CLIP space for cross-modal search.

    Backs the local ``"clip"`` preset. It wraps a sentence-transformers CLIP model
    (e.g. ``clip-ViT-B-32``), which maps both text and images into the *same*
    vector space, so text->image and image->image retrieval run over the ordinary
    single-vector TurboVec scan with no storage or scoring change. The model is
    loaded lazily on first use, and Pillow (the optional ``lodedb[image]`` extra)
    is imported only when an image is encoded, so a plain ``import lodedb`` pulls
    neither sentence-transformers nor Pillow.

    ``embed_images`` is an additional capability beyond the text-only
    :class:`EngineEmbeddingBackend` protocol; the SDK's ``add_image`` /
    ``search_by_image`` verbs detect it by duck typing.
    """

    name = "clip"

    def __init__(
        self,
        *,
        model_name: str,
        native_dim: int,
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        """Initializes a lazy CLIP backend for shared image/text embedding."""

        if not model_name:
            raise ValueError("model_name is required")
        if native_dim <= 0:
            raise ValueError("native_dim must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.model_name = model_name
        self.required_model_name = model_name
        self.native_dim = native_dim
        self.device = device
        self.batch_size = batch_size
        self._model: object | None = None

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Embeds text chunks into the shared CLIP space."""

        return self._encode(list(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embeds one text query into the shared CLIP space."""

        return self._encode([text])[0]

    def embed_images(self, images: Sequence[Any]) -> tuple[tuple[float, ...], ...]:
        """Embeds images into the shared CLIP space, aligned cross-modally with text.

        Each item may be a filesystem path (``str`` / ``os.PathLike``), raw image
        ``bytes``, or an already-opened PIL ``Image``. Images are loaded with
        Pillow (the ``lodedb[image]`` extra) only here, never on import.
        """

        loaded = [self._load_image(item) for item in images]
        return self._encode(loaded)

    def _encode(self, items: list[Any]) -> tuple[tuple[float, ...], ...]:
        """Encodes a batch of text or image items into normalized vectors."""

        if not items:
            return ()
        model = self._load_model()
        embeddings = model.encode(  # type: ignore[attr-defined]
            items,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.native_dim:
            returned_dim = array.shape[1] if array.ndim == 2 else "unknown"
            raise ValueError(
                f"{self.model_name} returned dim {returned_dim}; expected {self.native_dim}"
            )
        return tuple(tuple(float(value) for value in row) for row in array)

    def _load_model(self) -> object:
        """Loads the sentence-transformers CLIP model lazily."""

        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @staticmethod
    def _load_image(item: Any) -> Any:
        """Loads one image item (path / bytes / PIL image) into an RGB PIL image.

        Guards against oversized / decompression-bomb images: the pixel count is
        checked against ``LODEDB_MAX_IMAGE_PIXELS`` (default ~64 MP) from the header,
        before the full decode, so a single hostile file cannot exhaust memory.
        """

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - exercised via the [image] extra
            raise ImportError(
                "image embedding requires Pillow; install the optional extra: "
                "pip install 'lodedb[image]'"
            ) from exc

        limit = _max_image_pixels()
        if isinstance(item, Image.Image):
            _reject_oversized_image(item, limit)
            return item.convert("RGB")
        if isinstance(item, (bytes, bytearray)):
            source: Any = io.BytesIO(bytes(item))
        elif isinstance(item, (str, os.PathLike)):
            source = os.fspath(item)
        else:
            raise TypeError(
                f"image must be a path, bytes, or a PIL Image; got {type(item).__name__}"
            )
        # Close the file/stream handle promptly rather than relying on finalization;
        # convert() returns a new image holding its own decoded data.
        try:
            with Image.open(source) as image:
                _reject_oversized_image(image, limit)
                return image.convert("RGB")
        except Image.DecompressionBombError as exc:
            raise ValueError(f"image rejected as a decompression bomb: {exc}") from exc


# Default ceiling on the pixel count of an image fed to the CLIP backend (~64 MP),
# a guard against oversized / decompression-bomb inputs. Override with the env var.
_DEFAULT_MAX_IMAGE_PIXELS = 64_000_000


def _max_image_pixels() -> int:
    """Returns the image pixel-count ceiling, honoring ``LODEDB_MAX_IMAGE_PIXELS``."""

    raw = os.environ.get("LODEDB_MAX_IMAGE_PIXELS")
    if raw is None or not raw.strip():
        return _DEFAULT_MAX_IMAGE_PIXELS
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("LODEDB_MAX_IMAGE_PIXELS must be an integer") from exc
    if value <= 0:
        raise ValueError("LODEDB_MAX_IMAGE_PIXELS must be a positive integer")
    return value


def _reject_oversized_image(image: object, limit: int) -> None:
    """Raises ``ValueError`` if an image's pixel count exceeds ``limit``.

    Reads the dimensions from the decoded header (``size``), so an oversized image
    is rejected before its pixels are fully materialized.
    """

    width, height = image.size  # type: ignore[attr-defined]
    if width * height > limit:
        raise ValueError(
            f"image is {width}x{height} ({width * height} pixels), over the "
            f"{limit}-pixel limit; raise LODEDB_MAX_IMAGE_PIXELS to allow it"
        )


class ONNXRuntimeEmbeddingBackend:
    """Embeds text with an ONNX Runtime feature-extraction model.

    A drop-in for :class:`SentenceTransformerEmbeddingBackend` that runs the same
    model through ONNX Runtime instead of PyTorch. It produces vectors comparable
    to the sentence-transformers path for the same model (matching tokenizer,
    pooling, and L2 normalization), so an index built with one runtime stays
    usable by the other. ``onnxruntime`` and ``transformers`` are imported lazily
    so a plain ``import lodedb`` never pays for them.

    Pooling must match the source model: BGE uses ``"cls"`` (the model's own
    pooling), MiniLM uses ``"mean"`` over the attention mask.
    """

    name = "onnx_runtime"

    def __init__(
        self,
        *,
        model_name: str,
        native_dim: int,
        onnx_model_path: str | Path,
        tokenizer_name_or_path: str,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
        batch_size: int = 16,
        max_seq_length: int = 512,
        query_prefix: str = "",
        document_prefix: str = "",
        pooling: str = "cls",
        normalize: bool = True,
        output_name: str | None = None,
    ) -> None:
        """Stores ONNX Runtime configuration while keeping imports and model load lazy."""

        if not model_name:
            raise ValueError("model_name is required")
        if native_dim <= 0:
            raise ValueError("native_dim must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")
        if pooling not in {"cls", "mean"}:
            raise ValueError("pooling must be cls or mean")
        if not providers:
            raise ValueError("at least one ONNX Runtime execution provider is required")
        self.model_name = model_name
        self.required_model_name = model_name
        self.native_dim = native_dim
        self.onnx_model_path = Path(onnx_model_path)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.providers = tuple(providers)
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.pooling = pooling
        self.normalize = normalize
        self.output_name = output_name
        self._session: object | None = None
        self._tokenizer: object | None = None

    def embed_documents(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Embeds document chunks with the configured document prefix."""

        return self._encode(tuple(f"{self.document_prefix}{text}" for text in texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        """Embeds one retrieval query with the configured query prefix."""

        return self._encode((f"{self.query_prefix}{text}",))[0]

    def _encode(self, texts: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        """Tokenizes, runs ONNX inference, pools, normalizes, and validates embeddings."""

        if not texts:
            return ()
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            tokenized = self._tokenize(batch)
            outputs = self._session_run(tokenized)
            pooled = _pool_onnx_output(
                outputs,
                attention_mask=np.asarray(tokenized.get("attention_mask")),
                pooling=self.pooling,
                output_name=self.output_name,
            )
            rows.append(pooled)
        array = np.vstack(rows).astype(np.float32, copy=False)
        if self.normalize:
            array = _l2_normalize_rows(array)
        _validate_embedding_array(array, native_dim=self.native_dim, model_name=self.model_name)
        return tuple(tuple(float(value) for value in row) for row in array)

    def active_providers(self) -> tuple[str, ...]:
        """Returns the execution providers active in the loaded ONNX Runtime session."""

        session = self._load_session()
        return tuple(str(provider) for provider in session.get_providers())

    def _tokenize(self, texts: tuple[str, ...]) -> dict[str, np.ndarray]:
        """Returns NumPy tokenizer inputs compatible with ONNX Runtime."""

        tokenizer = self._load_tokenizer()
        tokenized = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="np",
        )
        return {key: np.asarray(value) for key, value in dict(tokenized).items()}

    def _session_run(self, tokenized: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Runs the ONNX session with only the inputs the graph declares."""

        session = self._load_session()
        input_names = {item.name for item in session.get_inputs()}
        run_inputs = {key: value for key, value in tokenized.items() if key in input_names}
        if not run_inputs:
            raise ValueError(f"{self.model_name} ONNX graph accepted no tokenizer inputs")
        output_names = [item.name for item in session.get_outputs()]
        output_values = session.run(output_names, run_inputs)
        return {
            name: np.asarray(value, dtype=np.float32)
            for name, value in zip(output_names, output_values, strict=True)
        }

    def _load_session(self) -> object:
        """Loads the ONNX Runtime session lazily so a plain import never needs it."""

        if self._session is None:
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - optional runtime
                raise RuntimeError(
                    "onnxruntime is required for the ONNX embedding runtime "
                    "(install it, or use a runtime that falls back to torch)."
                ) from exc
            _preload_cuda_execution_provider_dependencies(ort, providers=self.providers)
            self._session = ort.InferenceSession(
                str(self.onnx_model_path),
                providers=list(self.providers),
            )
        return self._session

    def _load_tokenizer(self) -> object:
        """Loads the Hugging Face tokenizer lazily to avoid import cost on a plain import."""

        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional runtime
                raise RuntimeError(
                    "transformers is required to tokenize for the ONNX embedding runtime."
                ) from exc
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        return self._tokenizer


def _preload_cuda_execution_provider_dependencies(
    ort: object, *, providers: tuple[str, ...]
) -> None:
    """Preloads CUDA provider libraries before ONNX Runtime creates a session."""

    if "CUDAExecutionProvider" not in providers:
        return
    preload_dlls = getattr(ort, "preload_dlls", None)
    if callable(preload_dlls):
        try:
            preload_dlls(cuda=True, cudnn=True, msvc=False)
        except TypeError:
            preload_dlls()
        except Exception:  # noqa: BLE001 - best-effort warmup; absence is fine
            pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.current_device()
    except Exception:  # noqa: BLE001 - best-effort warmup; absence is fine
        pass


def _pool_onnx_output(
    outputs: dict[str, np.ndarray],
    *,
    attention_mask: np.ndarray,
    pooling: str,
    output_name: str | None,
) -> np.ndarray:
    """Pools ONNX token embeddings into one sentence embedding per input row."""

    selected = _select_onnx_embedding_output(outputs, output_name=output_name)
    if selected.ndim == 2:
        return selected.astype(np.float32, copy=False)
    if selected.ndim != 3:
        raise ValueError("ONNX output must be a 2D sentence tensor or 3D token tensor")
    if pooling == "cls":
        return selected[:, 0, :].astype(np.float32, copy=False)
    if attention_mask.ndim != 2:
        raise ValueError("mean pooling requires a 2D attention_mask")
    mask = attention_mask.astype(np.float32)
    masked = selected * mask[:, :, None]
    denominator = np.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    return (masked.sum(axis=1) / denominator).astype(np.float32, copy=False)


def _select_onnx_embedding_output(
    outputs: dict[str, np.ndarray],
    *,
    output_name: str | None,
) -> np.ndarray:
    """Selects the configured ONNX output or the first tensor shaped like embeddings."""

    if output_name is not None:
        if output_name not in outputs:
            raise ValueError(f"ONNX output {output_name!r} was not returned")
        return outputs[output_name]
    for value in outputs.values():
        if value.ndim in {2, 3}:
            return value
    raise ValueError("ONNX session returned no embedding-shaped output")


def _l2_normalize_rows(array: np.ndarray) -> np.ndarray:
    """Returns row-wise L2-normalized float32 embeddings with zero rows preserved."""

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return (array / safe_norms).astype(np.float32, copy=False)


def _validate_embedding_array(
    array: np.ndarray,
    *,
    native_dim: int,
    model_name: str,
) -> None:
    """Raises a deterministic error when an embedding runtime returns the wrong shape."""

    if array.ndim != 2:
        raise ValueError(f"{model_name} returned rank {array.ndim}; expected a 2D tensor")
    if array.shape[1] != native_dim:
        raise ValueError(f"{model_name} returned dim {array.shape[1]}; expected {native_dim}")


def hash_embedding(text: str, dim: int) -> tuple[float, ...]:
    """Builds a deterministic normalized fixture embedding without external calls."""

    if dim <= 0:
        raise ValueError("dim must be positive")
    values = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        counter += 1
    array = np.asarray(values[:dim], dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        return tuple(0.0 for _ in range(dim))
    return tuple(float(value) for value in array / norm)


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Returns cosine similarity for normalized embeddings of the same dimension."""

    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    score = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return float(score)
