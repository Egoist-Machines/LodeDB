"""Capability report behind ``lodedb doctor``.

Reuses the existing :func:`turbovec_capability` probe and CPU-flag detection
rather than reimplementing any capability logic. Reports, honestly:

- whether this is Apple Silicon;
- the embedding device that ``device="auto"`` resolves to, plus MPS/CUDA
  availability;
- the compact (TurboVec) backend status and inferred native dispatch;
- whether the GPU-resident vector-scan path exists — which is **CUDA/CuPy
  only**; on Apple Silicon that CUDA path is unavailable, but an opt-in Metal
  (MPS) exact scan is reported separately (off by default; NEON is the default).
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


def _gpu_vector_scan_status() -> dict[str, Any]:
    """Returns honest GPU-resident vector-scan availability (CUDA/CuPy only)."""

    cuda = torch_cuda_available()
    cupy_present = False
    try:
        import importlib.util

        cupy_present = importlib.util.find_spec("cupy") is not None
    except Exception:  # noqa: BLE001 - absence is the common, fine case
        cupy_present = False
    available = bool(cuda and cupy_present)
    if available:
        reason = "CUDA + CuPy present"
    elif is_apple_silicon():
        reason = (
            "Apple Silicon: the CUDA/CuPy GPU vector scan is unavailable here; an "
            "opt-in Metal (MPS) exact scan is available instead (see mps_vector_scan). "
            "NEON is the default and was faster on measured Apple hardware."
        )
    else:
        missing = []
        if not cuda:
            missing.append("CUDA")
        if not cupy_present:
            missing.append("CuPy")
        reason = f"GPU vector scan requires {' + '.join(missing) or 'CUDA + CuPy'}"
    return {
        "gpu_vector_scan_available": available,
        "cuda_available": cuda,
        "cupy_present": cupy_present,
        "reason": reason,
    }


def _mps_vector_scan_status() -> dict[str, Any]:
    """Returns honest opt-in Apple-GPU (MPS) exact-scan availability.

    The MPS scan is opt-in and never the default: NEON is the default on Apple
    Silicon and was faster across batch sizes on the hardware measured.
    """

    from lodedb.engine.mps_turbovec import mps_exact_scan_available

    available, reason = mps_exact_scan_available()
    return {
        "mps_exact_scan_available": available,
        "opt_in": True,
        "default_enabled": False,
        "reason": (
            reason
            or "available; opt-in via LODEDB_MPS_DIRECT_TURBOVEC=auto|required "
            "(NEON is the default and faster on measured Apple hardware)"
        ),
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
        "compact_backend": {
            **capability.to_dict(),
            "inferred_native_dispatch": turbovec_native_backend_from_flags(cpu_flags),
        },
        "gpu_vector_scan": _gpu_vector_scan_status(),
        "mps_vector_scan": _mps_vector_scan_status(),
        "windows_gpu_hint": _windows_gpu_embedding_hint(),
    }


def format_capability_report(report: dict[str, Any]) -> str:
    """Renders a human-readable capability report for the CLI."""

    plat = report["platform"]
    emb = report["embedding"]
    backend = report["compact_backend"]
    gpu = report["gpu_vector_scan"]
    mps_scan = report["mps_vector_scan"]
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
            f" ({'MPS/CUDA exact serving' if recon else 'unavailable — CPU scan only'})",
        ]
    else:
        lines.append(f"  unavailable reason     : {backend.get('unavailable_reason', '')}")
    lines += [
        "",
        "GPU-resident vector scan (CUDA/CuPy only)",
        f"  available              : {gpu['gpu_vector_scan_available']}",
        f"  reason                 : {gpu['reason']}",
        "",
        "Apple GPU exact scan (MPS, opt-in)",
        f"  available              : {mps_scan['mps_exact_scan_available']}",
        f"  default enabled        : {mps_scan['default_enabled']}",
        f"  reason                 : {mps_scan['reason']}",
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
