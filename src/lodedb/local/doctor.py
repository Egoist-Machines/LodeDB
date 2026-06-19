"""Capability report behind ``lodedb doctor``.

Reuses the existing :func:`turbovec_capability` probe and CPU-flag detection
rather than reimplementing any capability logic. Reports, honestly:

- whether this is Apple Silicon;
- the embedding device that ``device="auto"`` resolves to, plus MPS/CUDA
  availability;
- the compact (TurboVec) backend status and inferred native dispatch;
- whether the GPU-resident vector-scan path exists — which is **CUDA/CuPy
  only**; on Apple Silicon it is always reported as unavailable, and embedding
  is the only accelerated stage.
"""

from __future__ import annotations

from typing import Any

from lodedb.engine.turbovec_index import (
    detect_cpu_flags,
    turbovec_capability,
    turbovec_native_backend_from_flags,
)
from lodedb.local.backends import (
    is_apple_silicon,
    resolve_local_device,
    torch_cuda_available,
    torch_mps_available,
)


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
            "Apple Silicon: GPU vector scan is CUDA/CuPy-only and not implemented "
            "on Metal; the TurboVec scan runs on the CPU kernel here. Embedding is "
            "the only stage accelerated on this machine."
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
        },
        "compact_backend": {
            **capability.to_dict(),
            "inferred_native_dispatch": turbovec_native_backend_from_flags(cpu_flags),
        },
        "gpu_vector_scan": _gpu_vector_scan_status(),
    }


def format_capability_report(report: dict[str, Any]) -> str:
    """Renders a human-readable capability report for the CLI."""

    plat = report["platform"]
    emb = report["embedding"]
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
    ]
    return "\n".join(lines)
