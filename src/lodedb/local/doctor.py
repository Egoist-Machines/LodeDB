"""Capability report behind ``lodedb doctor``.

Reuses the existing :func:`turbovec_capability` probe and CPU-flag detection
rather than reimplementing any capability logic. Reports, honestly:

- whether this is Apple Silicon;
- the embedding device that ``device="auto"`` resolves to, plus MPS/CUDA
  availability;
- the compact (TurboVec) backend status and inferred native dispatch;
- whether the GPU-resident vector-scan path exists. That scan runs in the
  bundled native core (cudarc), so the probe is the native CUDA-driver check,
  not torch or CuPy; on Apple Silicon there is no CUDA driver and the NEON CPU
  kernel is the accelerated path.
"""

from __future__ import annotations

import platform
import shutil
from typing import Any

from lodedb.engine.turbovec_index import (
    detect_cpu_flags,
    turbovec_capability,
    turbovec_native_backend_from_flags,
)
from lodedb.local.backends import (
    is_apple_silicon,
    onnxruntime_available,
    resolve_local_device,
    torch_cuda_available,
    torch_cuda_build_version,
    torch_mps_available,
)

# Windows PyPI serves a CPU-only torch wheel by default; this index has the CUDA builds.
# cu121 is a broadly driver-compatible default; other toolkits live at /whl/cu124, etc.
_PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"


def _windows_gpu_embedding_hint() -> dict[str, Any] | None:
    """On Windows with a CPU-only PyTorch, returns how to switch to a CUDA build.

    PyPI serves the CPU-only torch wheel on Windows by default, so ``pip install lodedb``
    leaves embeddings on the CPU even on an NVIDIA machine, and no package metadata can
    redirect torch to the CUDA index. Returns ``None`` when it does not apply: off Windows,
    when torch is absent, or when torch is already a CUDA build.
    """

    if platform.system() != "Windows":
        return None
    try:
        import torch  # noqa: F401  (presence check; a broken install is a different problem)
    except ImportError:
        return None
    if torch_cuda_build_version() is not None:
        return None  # already a CUDA build
    nvidia_detected = shutil.which("nvidia-smi") is not None
    return {
        "torch_cuda_build": False,
        "nvidia_smi_detected": nvidia_detected,
        "index_url": _PYTORCH_CUDA_INDEX,
        "command": (
            f"pip install torch --force-reinstall --no-deps --index-url {_PYTORCH_CUDA_INDEX}"
        ),
    }


def _image_embedding_status() -> dict[str, Any]:
    """Returns whether the optional image+text (CLIP) embedding path is installed.

    The ``"clip"`` preset / ``add_image`` runs on the base sentence-transformers
    stack and adds only Pillow (the ``[image]`` extra) for decoding image files.
    """

    import importlib.util

    pillow_present = importlib.util.find_spec("PIL") is not None
    st_present = importlib.util.find_spec("sentence_transformers") is not None
    available = bool(pillow_present and st_present)
    if available:
        reason = "Pillow + sentence-transformers present (use model='clip')"
    else:
        missing = []
        if not pillow_present:
            missing.append("Pillow")
        if not st_present:
            missing.append("sentence-transformers")
        reason = f"image embedding requires {' + '.join(missing)} (pip install 'lodedb[image]')"
    return {
        "image_embedding_available": available,
        "pillow_present": pillow_present,
        "model": "sentence-transformers/clip-ViT-B-32",
        "reason": reason,
    }


def _gpu_vector_scan_status() -> dict[str, Any]:
    """Returns honest GPU-resident vector-scan availability from the native probe.

    The GPU-resident scan runs in the bundled native core (cudarc), so this
    reports the real CUDA-driver probe the native scan gates on rather than a
    torch/CuPy proxy: it needs neither. ``native_core_available`` distinguishes
    "no CUDA driver" from "the native extension did not load".
    """

    from lodedb.engine.native_adapter import NativeCoreAdapter

    adapter = NativeCoreAdapter()
    native_core_available = adapter.available
    available = bool(adapter.cuda_runtime_available())
    if available:
        reason = "native core + CUDA driver present"
    elif is_apple_silicon():
        reason = (
            "Apple Silicon: the CUDA GPU vector scan is unavailable here; the "
            "NEON CPU kernel is the accelerated path."
        )
    elif not native_core_available:
        reason = "GPU vector scan requires the bundled native core, which is not loaded"
    else:
        reason = "GPU vector scan requires a CUDA driver (none detected)"
    return {
        "gpu_vector_scan_available": available,
        "native_core_available": native_core_available,
        "reason": reason,
    }


def _embedding_runtime_status() -> dict[str, Any]:
    """Reports the embedding runtime ``embedding_runtime="auto"`` prefers, plus ONNX details.

    ``auto`` uses ONNX Runtime only when ``onnxruntime`` is installed *and* the model's
    ONNX graph can be obtained (cached, a prebuilt Hub snapshot, or an Optimum export),
    falling back to PyTorch sentence-transformers otherwise. This probe checks the former
    (a usable onnxruntime) without forcing a per-model download, so it reports a
    *preference*, not a guarantee; the ``note`` states the fallback condition.
    """

    onnx = onnxruntime_available()
    providers: list[str] = []
    if onnx:
        try:
            import onnxruntime as ort

            providers = list(ort.get_available_providers())
        except Exception:  # noqa: BLE001 - report none if probing fails
            providers = []
    note = (
        "auto prefers ONNX; falls back to PyTorch if a model's ONNX graph is unavailable"
        if onnx
        else "onnxruntime not installed; auto uses PyTorch"
    )
    return {
        "preferred": "onnx" if onnx else "torch",
        "onnxruntime_available": onnx,
        "onnx_providers": providers,
        "note": note,
    }


def local_capability_report(*, device: str = "auto") -> dict[str, Any]:
    """Builds the full local capability report (no payloads)."""

    apple_silicon = is_apple_silicon()
    mps = torch_mps_available()
    cuda = torch_cuda_available()
    effective_device = resolve_local_device(device)

    cpu_flags = detect_cpu_flags()
    capability = turbovec_capability()

    return {
        "platform": {
            "apple_silicon": apple_silicon,
        },
        "embedding": {
            "requested_device": device,
            "auto_resolves_to": effective_device,
            "mps_available": mps,
            "cuda_available": cuda,
            "runtime": _embedding_runtime_status(),
        },
        "image_embedding": _image_embedding_status(),
        "compact_backend": {
            **capability.to_dict(),
            "inferred_native_dispatch": turbovec_native_backend_from_flags(cpu_flags),
        },
        "gpu_vector_scan": _gpu_vector_scan_status(),
        "windows_gpu_hint": _windows_gpu_embedding_hint(),
    }


def format_capability_report(report: dict[str, Any]) -> str:
    """Renders a human-readable capability report for the CLI."""

    plat = report["platform"]
    emb = report["embedding"]
    img = report["image_embedding"]
    backend = report["compact_backend"]
    gpu = report["gpu_vector_scan"]
    lines = [
        "LodeDB doctor — local capability report",
        "=" * 42,
        f"Apple Silicon            : {plat['apple_silicon']}",
        "",
        "Embedding (accelerated stage)",
        f"  requested device       : {emb['requested_device']}",
        f"  auto resolves to       : {emb['auto_resolves_to']}",
        f"  MPS available          : {emb['mps_available']}",
        f"  CUDA available         : {emb['cuda_available']}",
        f"  runtime (auto prefers) : {emb['runtime']['preferred']}",
        f"    fallback             : {emb['runtime']['note']}",
        f"  onnxruntime available  : {emb['runtime']['onnxruntime_available']}",
        f"  onnx providers         : {', '.join(emb['runtime']['onnx_providers']) or 'none'}",
        "",
        "Image + text embedding (CLIP, optional [image] extra)",
        f"  available              : {img['image_embedding_available']}",
        f"  model                  : {img['model']}",
        f"  reason                 : {img['reason']}",
        "",
        "Compact storage backend (TurboVec)",
        f"  available              : {backend['available']}",
        f"  native dispatch        : {backend.get('native_backend', '?')}"
        f" (inferred {backend.get('inferred_native_dispatch', '?')})",
        f"  vendored target        : turbovec {backend.get('version', '?')}"
        f" (tag {backend.get('source_tag', '?')})",
    ]
    if backend["available"]:
        delta = backend.get("delta_persistence_available", False)
        recon = backend.get("reconstruction_available", False)
        lines += [
            f"  patched core           : "
            f"{'present' if (delta and recon) else 'MISSING (stock PyPI turbovec)'}",
            f"    delta persistence    : {delta}"
            f" ({'incremental .tvd deltas' if delta else 'unavailable — full .tvim rewrites'})",
            f"    exact reconstruction : {recon}"
            f" ({'CUDA exact serving' if recon else 'unavailable, CPU scan only'})",
        ]
    else:
        lines.append(f"  unavailable reason     : {backend.get('unavailable_reason', '')}")
    lines += [
        "",
        "GPU-resident vector scan (native core, CUDA driver only)",
        f"  available              : {gpu['gpu_vector_scan_available']}",
        f"  reason                 : {gpu['reason']}",
    ]
    hint = report.get("windows_gpu_hint")
    if hint:
        gpu_state = "detected" if hint["nvidia_smi_detected"] else "if you have one"
        lines += [
            "",
            "Windows GPU embeddings",
            "  PyTorch build          : CPU-only (no CUDA)",
            f"  NVIDIA GPU             : {gpu_state}",
            f"  to embed on the GPU    : {hint['command']}",
            "    or run `lodedb doctor --fix`; see https://pytorch.org/get-started/locally/",
            "    for the index matching your CUDA version (cu121, cu124, ...).",
        ]
    return "\n".join(lines)
