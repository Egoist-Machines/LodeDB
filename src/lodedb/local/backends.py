"""Embedding runtime + device selection for the local layer.

This module selects the embedding *runtime* (ONNX Runtime or PyTorch) and the
*device* it runs on; the TurboVec vector scan is separate (the CPU SIMD kernel,
or the optional CUDA path) and is not affected here.

Runtime (``embedding_runtime``):

- ``"auto"`` (default) prefers the ONNX Runtime path when ``onnxruntime`` is
  installed and the model's ONNX artifact can be materialized, and otherwise
  falls back to PyTorch sentence-transformers. ONNX is typically faster for
  feature extraction and drops the heavy torch dependency from the hot path.
- ``"onnx"`` forces ONNX Runtime (errors if it cannot be set up).
- ``"torch"`` forces PyTorch sentence-transformers.

Device:

- ``device="mps"`` runs sentence-transformers on PyTorch's Metal Performance
  Shaders backend (the Apple GPU); for ONNX it prefers the Core ML provider.
- ``device="cuda"`` runs on an NVIDIA GPU; ``device="cpu"`` on the CPU.
- ``device="auto"`` prefers MPS on Apple Silicon, CUDA on NVIDIA, else CPU.

Both runtimes implement the ``EngineEmbeddingBackend`` protocol and produce
comparable vectors for the same model, so an index built with one stays usable
by the other.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
from dataclasses import dataclass

from lodedb.engine.embedding_backends import (
    ClipEmbeddingBackend,
    EngineEmbeddingBackend,
    ONNXRuntimeEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
)
from lodedb.local.onnx_artifacts import OnnxMaterializationError, materialize_onnx_model
from lodedb.local.presets import LocalModelPreset

logger = logging.getLogger("lodedb.local")


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


def onnxruntime_available() -> bool:
    """Returns whether ``onnxruntime`` is importable (without importing it)."""

    return importlib.util.find_spec("onnxruntime") is not None


def sentence_transformers_available() -> bool:
    """Returns whether ``sentence-transformers`` is importable (without importing it)."""

    return importlib.util.find_spec("sentence_transformers") is not None


def _missing_embedding_runtime_error(*, runtime: str, onnx_installed: bool) -> ModuleNotFoundError:
    """Builds a clear error for a text model requested with no usable embedding runtime.

    Built-in text embedding is opt-in (the ``[embeddings]`` / ``[torch]`` extras); without it
    LodeDB is a vector store. This points at the right extra rather than letting a deep
    ``ModuleNotFoundError`` surface later at encode time, and reminds the caller of the
    bring-your-own-vectors path.
    """

    if runtime == "torch":
        hint = "pip install 'lodedb[embeddings,torch]'"
    elif onnx_installed:
        # ONNX is installed but could not materialize this model's graph; torch is the fallback.
        hint = (
            "pip install 'lodedb[embeddings,torch]' (PyTorch fallback) or "
            "'lodedb[onnx-export]' (export ONNX for this model)"
        )
    else:
        hint = "pip install 'lodedb[embeddings]'"
    return ModuleNotFoundError(
        "text embedding needs an embedding runtime, which is not installed: "
        f"{hint} (or bring your own vectors via LodeDB.open_vector_store(...) / pass embedder=)."
    )


def _coreml_opt_in() -> bool:
    """Returns whether the opt-in ONNX Core ML provider is enabled (``LODEDB_ONNX_COREML``)."""

    return os.environ.get("LODEDB_ONNX_COREML", "").strip().lower() in {"1", "true", "yes", "auto"}


def _preferred_onnx_providers(device: str) -> tuple[str, ...]:
    """Returns the preferred ONNX provider order for a device, before availability filtering.

    CUDA hosts prefer the CUDA provider. The Apple **Core ML** provider is **off by default**:
    on the dynamic-shape preset graphs it fragments into many Core ML/CPU partitions and measured
    *slower* than the plain CPU provider for single-query embedding (about 16 ms vs 3 ms on an
    M-series CPU), so it is opt-in via ``LODEDB_ONNX_COREML=1`` — the same stance the repo takes on
    the MPS vector scan, which is also off by default because NEON was faster. Everything else (CPU,
    and MPS without the opt-in) uses the CPU provider, which is the fast path for these models.
    """

    if device == "cuda":
        return ("CUDAExecutionProvider", "CPUExecutionProvider")
    if device == "mps" and _coreml_opt_in():
        return ("CoreMLExecutionProvider", "CPUExecutionProvider")
    return ("CPUExecutionProvider",)


def _resolve_onnx_providers(device: str) -> tuple[str, ...]:
    """Filters the preferred provider order to those the installed onnxruntime actually offers.

    ONNX Runtime treats the list as a preference order; keeping only available providers means a
    missing accelerator EP degrades to CPU instead of erroring. CPU is always appended last.
    """

    preferred = _preferred_onnx_providers(device)
    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - if we cannot probe, CPU is always present
        return ("CPUExecutionProvider",)
    chosen = tuple(provider for provider in preferred if provider in available)
    if "CPUExecutionProvider" not in chosen:
        chosen = (*chosen, "CPUExecutionProvider")
    return chosen


def _onnx_backend(
    preset: LocalModelPreset,
    *,
    providers: tuple[str, ...],
    batch_size: int,
    max_seq_length: int,
) -> ONNXRuntimeEmbeddingBackend:
    """Materializes the preset's ONNX model and builds the ONNX Runtime backend.

    Raises :class:`OnnxMaterializationError` if no ONNX artifact can be obtained.
    """

    artifact = materialize_onnx_model(preset.model_name)
    return ONNXRuntimeEmbeddingBackend(
        model_name=preset.model_name,
        native_dim=preset.native_dim,
        onnx_model_path=artifact.model_path,
        tokenizer_name_or_path=str(artifact.tokenizer_dir),
        providers=providers,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        query_prefix=preset.query_prefix,
        document_prefix=preset.document_prefix,
        pooling=preset.pooling,
        normalize=True,
    )


def build_local_embedding_backend(
    preset: LocalModelPreset,
    *,
    device: str = "auto",
    batch_size: int = 32,
    max_seq_length: int = 256,
    embedding_runtime: str = "auto",
) -> tuple[EngineEmbeddingBackend, LocalEmbeddingResolution]:
    """Builds the embedding backend for a preset, runtime, and device.

    ``embedding_runtime`` is ``"auto"`` (prefer ONNX Runtime, fall back to torch),
    ``"onnx"`` (force ONNX Runtime), or ``"torch"`` (force sentence-transformers).
    Returns the backend plus a :class:`LocalEmbeddingResolution` describing the
    runtime/device actually selected (and any fallback), so the SDK and ``doctor``
    can report it.
    """

    runtime = (embedding_runtime or "auto").strip().lower()
    if runtime not in {"auto", "onnx", "torch"}:
        raise ValueError(
            f"unknown embedding_runtime {embedding_runtime!r}; choose auto, onnx, or torch"
        )
    resolved = resolve_local_device(device)

    # A multimodal preset ("clip") embeds text and images into one shared space via
    # sentence-transformers; it does not use the ONNX/torch text runtime selection.
    if preset.multimodal:
        if not sentence_transformers_available():
            raise ModuleNotFoundError(
                "the 'clip' preset needs the sentence-transformers stack, which is not "
                "installed: pip install 'lodedb[image]'"
            )
        backend = ClipEmbeddingBackend(
            model_name=preset.model_name,
            native_dim=preset.native_dim,
            device=resolved,
            batch_size=batch_size,
        )
        return backend, LocalEmbeddingResolution(
            requested_device=device,
            backend_name=backend.name,
            effective_device=resolved,
            fallback_used=False,
            fallback_reason="",
        )

    fallback_reason = ""
    if runtime in {"auto", "onnx"}:
        # Forcing ONNX without the runtime installed must raise the install hint up front,
        # not enter _onnx_backend and fail deep at materialization/encode time. (The "auto"
        # path instead falls through to the torch fallback below.)
        if runtime == "onnx" and not onnxruntime_available():
            raise _missing_embedding_runtime_error(runtime="onnx", onnx_installed=False)
        if runtime == "onnx" or onnxruntime_available():
            providers = _resolve_onnx_providers(resolved)
            # A CUDA device with no CUDAExecutionProvider means the CPU-only onnxruntime wheel is
            # installed: embedding runs on the CPU (typically 10-50x slower) with no error.
            cuda_fell_back_to_cpu = resolved == "cuda" and "CUDAExecutionProvider" not in providers
            try:
                backend = _onnx_backend(
                    preset,
                    providers=providers,
                    batch_size=batch_size,
                    max_seq_length=max_seq_length,
                )
                # Log/warn only now that ONNX is the committed runtime: in "auto" the build above
                # can still raise and fall back to torch (which may reach the GPU), so warning early
                # would misreport the device. Surfacing the CPU fallback is the single highest-value
                # fix reported from real GPU deployments that silently embedded on the CPU: a log
                # line reaches the operator where docs do not.
                logger.info(
                    "embedding runtime=onnx device=%s (requested %s) providers=%s",
                    resolved,
                    device,
                    ",".join(providers),
                )
                if cuda_fell_back_to_cpu:
                    logger.warning(
                        "LodeDB is embedding on the CPU: the device resolved to CUDA but ONNX "
                        "Runtime has no CUDAExecutionProvider (the default onnxruntime wheel is "
                        "CPU-only, typically 10-50x slower). Install onnxruntime-gpu to use the "
                        "NVIDIA GPU (`lodedb doctor` then lists CUDAExecutionProvider)."
                    )
                return backend, LocalEmbeddingResolution(
                    requested_device=device,
                    backend_name=backend.name,
                    effective_device="cpu" if cuda_fell_back_to_cpu else resolved,
                    fallback_used=cuda_fell_back_to_cpu,
                    fallback_reason=(
                        "cuda requested but onnxruntime has no CUDAExecutionProvider; embedding on "
                        "cpu (install onnxruntime-gpu)"
                        if cuda_fell_back_to_cpu
                        else ""
                    ),
                )
            except (OnnxMaterializationError, RuntimeError, ImportError) as exc:
                if runtime == "onnx":
                    raise
                fallback_reason = f"onnx runtime unavailable, using torch: {exc}"
        else:
            fallback_reason = "onnxruntime not installed; using torch"

    if not sentence_transformers_available():
        raise _missing_embedding_runtime_error(
            runtime=runtime, onnx_installed=onnxruntime_available()
        )
    backend = _sentence_transformer_backend(
        preset, device=resolved, batch_size=batch_size, max_seq_length=max_seq_length
    )
    return backend, LocalEmbeddingResolution(
        requested_device=device,
        backend_name=backend.name,
        effective_device=resolved,
        fallback_used=bool(fallback_reason),
        fallback_reason=fallback_reason,
    )
