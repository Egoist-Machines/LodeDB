"""Embedding device selection for the local layer.

This module selects the embedding *device*; the TurboVec vector scan is separate
(the CPU SIMD kernel, or the optional CUDA path) and is not affected here.

- ``device="mps"`` runs sentence-transformers on PyTorch's Metal Performance
  Shaders backend (the Apple GPU).
- ``device="cuda"`` runs it on an NVIDIA GPU; ``device="cpu"`` on the CPU.
- ``device="auto"`` prefers MPS on Apple Silicon, CUDA on NVIDIA, else CPU.

All paths go through the existing ``SentenceTransformerEmbeddingBackend`` and the
``EngineEmbeddingBackend`` protocol.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass

from lodedb.engine.embedding_backends import (
    EngineEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
)
from lodedb.local.presets import LocalModelPreset


@dataclass(frozen=True)
class LocalEmbeddingResolution:
    """Records the requested vs. effective embedding device."""

    requested_device: str
    backend_name: str
    effective_device: str
    fallback_used: bool
    fallback_reason: str

    def to_dict(self) -> dict[str, object]:
        """Serializes device-selection metadata for doctor/telemetry (no payloads)."""

        return {
            "requested_device": self.requested_device,
            "backend_name": self.backend_name,
            "effective_device": self.effective_device,
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
        }


def is_apple_silicon() -> bool:
    """Returns whether this process runs on macOS arm64 (Apple Silicon)."""

    return platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}


def torch_mps_available() -> bool:
    """Returns whether PyTorch reports a usable MPS (Metal) device."""

    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def torch_cuda_available() -> bool:
    """Returns whether PyTorch reports a usable CUDA device."""

    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def torch_cuda_build_version() -> str | None:
    """Returns the CUDA version PyTorch was built against (e.g. ``"12.1"``), else ``None``.

    ``None`` means torch is not importable *or* it is a CPU-only build: a CPU-only wheel
    reports ``torch.version.cuda is None`` even on a CUDA-capable machine, which is the
    default ``pip install torch`` on Windows. Distinct from :func:`torch_cuda_available`,
    which probes for a usable *device* at runtime.
    """

    try:
        import torch
    except ImportError:
        return None
    return getattr(getattr(torch, "version", None), "cuda", None)


def resolve_local_device(requested: str) -> str:
    """Resolves an ``auto`` device request to a concrete device for this machine.

    Returns one of ``"mps"``, ``"cuda"``, or ``"cpu"``.
    """

    choice = (requested or "auto").strip().lower()
    if choice not in {"auto", "cpu", "mps", "cuda"}:
        raise ValueError(f"unknown device {requested!r}; choose auto, cpu, mps, or cuda")
    if choice != "auto":
        return choice
    if torch_mps_available():
        return "mps"
    if torch_cuda_available():
        return "cuda"
    return "cpu"


def _sentence_transformer_backend(
    preset: LocalModelPreset,
    *,
    device: str,
    batch_size: int,
    max_seq_length: int,
) -> SentenceTransformerEmbeddingBackend:
    """Builds the sentence-transformers backend on a concrete device."""

    return SentenceTransformerEmbeddingBackend(
        model_name=preset.model_name,
        native_dim=preset.native_dim,
        device=device,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        query_prefix=preset.query_prefix,
        document_prefix=preset.document_prefix,
    )


def build_local_embedding_backend(
    preset: LocalModelPreset,
    *,
    device: str = "auto",
    batch_size: int = 32,
    max_seq_length: int = 256,
) -> tuple[EngineEmbeddingBackend, LocalEmbeddingResolution]:
    """Builds the sentence-transformers embedding backend for a preset/device.

    Returns the backend plus a :class:`LocalEmbeddingResolution` describing the
    device actually selected, so the SDK and ``doctor`` can report it.
    """

    resolved = resolve_local_device(device)
    backend = _sentence_transformer_backend(
        preset, device=resolved, batch_size=batch_size, max_seq_length=max_seq_length
    )
    return backend, LocalEmbeddingResolution(
        requested_device=device,
        backend_name=backend.name,
        effective_device=resolved,
        fallback_used=False,
        fallback_reason="",
    )
