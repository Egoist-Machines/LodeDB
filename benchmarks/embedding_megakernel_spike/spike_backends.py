"""Named single-query embedding baselines for the megakernel spike.

Issue #67 cites two M1 numbers: 5.73 ms (CPU embed) and 8.42 ms (MPS embed). The repo
routes an ONNX ``device="mps"`` request to the CPU provider (Core ML is opt-in because
it measured slower on the dynamic-shape preset graph), so the 8.42 ms MPS number is the
*torch* path, not ONNX. This module builds each runtime/provider path as a separately
named baseline so the two numbers are attributed to the right stack:

- ``onnx-cpu``    - ONNX Runtime, CPUExecutionProvider (the default single-query path).
- ``onnx-coreml`` - ONNX Runtime, CoreMLExecutionProvider (opt-in on Apple).
- ``torch-cpu``   - sentence-transformers on the CPU.
- ``torch-mps``   - sentence-transformers on the Apple GPU (the 8.42 ms path).
- ``onnx-cuda`` / ``torch-cuda`` - the CUDA equivalents, used by the Modal harness.

The ONNX and torch backends are the exact LodeDB classes the SDK uses. A small manual
HF pipeline (:class:`ManualTorchMiniLM`) mirrors sentence-transformers (AutoModel +
mean-pool + L2-normalize) so the torch/MPS/CUDA forward pass can be timed stage by
stage; its output is parity-checked against the SDK backend.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import numpy as np

from lodedb.engine.embedding_backends import (
    ONNXRuntimeEmbeddingBackend,
    SentenceTransformerEmbeddingBackend,
)
from lodedb.local.backends import is_apple_silicon
from lodedb.local.onnx_artifacts import materialize_onnx_model
from lodedb.local.presets import resolve_preset

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Deterministic, neutral fixture queries of a few lengths (short/medium retrieval
# phrasings). Used only as benchmark inputs; never written to any artifact.
FIXTURE_QUERIES: tuple[str, ...] = (
    "vector database",
    "how do embeddings work",
    "what is the capital of the country with the largest population",
    "local first storage engine for retrieval augmented generation pipelines",
    "compare quantized nearest neighbor search against exact scan latency tradeoffs",
    "single query embedding latency",
    "fused attention feed forward layer",
    "memory bandwidth roofline for transformer inference on batch size one",
)


@dataclass(frozen=True)
class NamedBackend:
    """A named embedding baseline plus its resolved runtime/provider metadata."""

    name: str
    runtime: str  # "onnx" | "torch"
    device: str
    providers: tuple[str, ...]
    backend: object

    def embed_query(self, text: str) -> tuple[float, ...]:
        return self.backend.embed_query(text)  # type: ignore[attr-defined]


def machine_info() -> dict[str, object]:
    """Recorded machine + toolchain block (versions, providers). No payloads."""

    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "apple_silicon": is_apple_silicon(),
        "cpu_count": os.cpu_count(),
    }
    try:
        info["ram_bytes"] = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        info["ram_bytes"] = None
    try:
        import onnxruntime as ort

        info["onnxruntime_version"] = ort.__version__
        info["onnxruntime_providers"] = list(ort.get_available_providers())
    except Exception:  # noqa: BLE001 - benchmark environment probe
        info["onnxruntime_version"] = None
        info["onnxruntime_providers"] = []
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["torch_mps"] = bool(torch.backends.mps.is_available())
        info["torch_cuda"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - benchmark environment probe
        info["torch_version"] = None
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception:  # noqa: BLE001 - benchmark environment probe
        info["transformers_version"] = None
    return info


def onnx_backend(providers: tuple[str, ...]) -> ONNXRuntimeEmbeddingBackend:
    """Builds the MiniLM ONNX backend with an explicit provider list (mirrors the SDK)."""

    preset = resolve_preset("minilm")
    artifact = materialize_onnx_model(preset.model_name)
    return ONNXRuntimeEmbeddingBackend(
        model_name=preset.model_name,
        native_dim=preset.native_dim,
        onnx_model_path=artifact.model_path,
        tokenizer_name_or_path=str(artifact.tokenizer_dir),
        providers=providers,
        batch_size=32,
        max_seq_length=256,
        query_prefix=preset.query_prefix,
        document_prefix=preset.document_prefix,
        pooling=preset.pooling,
        normalize=True,
    )


def torch_backend(device: str) -> SentenceTransformerEmbeddingBackend:
    """Builds the MiniLM sentence-transformers backend on a concrete device."""

    preset = resolve_preset("minilm")
    return SentenceTransformerEmbeddingBackend(
        model_name=preset.model_name,
        native_dim=preset.native_dim,
        device=device,
        batch_size=32,
        max_seq_length=256,
        query_prefix=preset.query_prefix,
        document_prefix=preset.document_prefix,
    )


def _available_ort_providers() -> set[str]:
    try:
        import onnxruntime as ort

        return set(ort.get_available_providers())
    except Exception:  # noqa: BLE001
        return {"CPUExecutionProvider"}


def build_named_backend(name: str) -> NamedBackend | None:
    """Builds one named baseline, or ``None`` if its runtime/device is unavailable."""

    available = _available_ort_providers()
    if name == "onnx-cpu":
        return NamedBackend(name, "onnx", "cpu", ("CPUExecutionProvider",),
                            onnx_backend(("CPUExecutionProvider",)))
    if name == "onnx-coreml":
        if "CoreMLExecutionProvider" not in available:
            return None
        providers = ("CoreMLExecutionProvider", "CPUExecutionProvider")
        return NamedBackend(name, "onnx", "coreml", providers, onnx_backend(providers))
    if name == "onnx-cuda":
        if "CUDAExecutionProvider" not in available:
            return None
        providers = ("CUDAExecutionProvider", "CPUExecutionProvider")
        return NamedBackend(name, "onnx", "cuda", providers, onnx_backend(providers))
    if name == "onnx-tensorrt":
        if "TensorrtExecutionProvider" not in available:
            return None
        providers = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
        return NamedBackend(name, "onnx", "tensorrt", providers, onnx_backend(providers))
    if name in {"torch-cpu", "torch-mps", "torch-cuda"}:
        device = name.split("-", 1)[1]
        if device == "mps" and not _torch_has("mps"):
            return None
        if device == "cuda" and not _torch_has("cuda"):
            return None
        return NamedBackend(name, "torch", device, (), torch_backend(device))
    raise ValueError(f"unknown baseline {name!r}")


def _torch_has(device: str) -> bool:
    try:
        import torch

        if device == "mps":
            return bool(torch.backends.mps.is_available())
        if device == "cuda":
            return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False
    return False


class ManualTorchMiniLM:
    """HF AutoModel + mean-pool + L2-normalize, exposing per-stage timing seams.

    sentence-transformers wraps tokenize/forward/pool/normalize in one ``encode`` call,
    which cannot be attributed. This reproduces the same computation (all-MiniLM-L6-v2 is
    a BERT encoder with mean pooling) so the torch/MPS/CUDA forward pass can be isolated
    from tokenization and pooling. Parity against the SDK backend is asserted by the
    caller before its timings are trusted.
    """

    def __init__(self, device: str, *, max_seq_length: int = 256) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(MINILM_MODEL)
        self.model = AutoModel.from_pretrained(MINILM_MODEL).to(device)
        self.model.eval()

    def sync(self) -> None:
        """Blocks until queued device work finishes, so timing captures real latency."""

        if self.device == "mps":
            self._torch.mps.synchronize()
        elif self.device == "cuda":
            self._torch.cuda.synchronize()

    def tokenize(self, text: str) -> dict:
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        ).to(self.device)

    def forward(self, tokens: dict):
        torch = self._torch
        with torch.no_grad():
            return self.model(**tokens)

    def pool_normalize(self, outputs, attention_mask):
        torch = self._torch
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def embed(self, text: str) -> tuple[float, ...]:
        tokens = self.tokenize(text)
        outputs = self.forward(tokens)
        vec = self.pool_normalize(outputs, tokens["attention_mask"])
        self.sync()
        return tuple(float(v) for v in vec[0].detach().to("cpu").numpy())


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity of two vectors (embeddings are already L2-normalized)."""

    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)
