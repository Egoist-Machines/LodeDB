"""Embedding backends for the LodeDB engine."""

from __future__ import annotations

import hashlib
import io
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np


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
