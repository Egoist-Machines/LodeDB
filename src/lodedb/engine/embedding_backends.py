"""Embedding backends for the LodeDB engine."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

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
