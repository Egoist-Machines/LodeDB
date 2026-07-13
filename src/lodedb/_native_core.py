"""Private native-core shim backed by the bundled Rust extension.

This module is intentionally not public API. It keeps the rollout import path
(``lodedb._native_core``) stable while the root wheel still builds one Rust
extension module, ``lodedb._turbovec``.
"""

from __future__ import annotations

import warnings

from lodedb import _turbovec as _turbovec
from lodedb._turbovec import (  # type: ignore[attr-defined]
    CoreAppender,
    CoreCheckpointer,
    CoreEngine,
    build_embedded_documents_payload,
    core_document_to_json,
    cuda_runtime_available,
    decode_wal_segment,
    encode_wal_segment,
    native_core_abi_version,
    native_core_version,
    plan_segment_documents,
    round_trip_core_json,
    storage_schema_version,
)

_native_build_profile = getattr(_turbovec, "native_build_profile", None)


def native_build_profile() -> str:
    """Returns the loaded extension's Cargo profile, or ``"unknown"`` for older builds."""

    if not callable(_native_build_profile):
        return "unknown"
    try:
        profile = _native_build_profile()
    except Exception:  # noqa: BLE001 - diagnostics must not prevent native-core loading
        return "unknown"
    return profile if profile in {"debug", "release"} else "unknown"


def _warn_if_debug_build(profile: str) -> None:
    """Warns when the loaded extension uses Cargo's unoptimized debug profile."""

    if profile == "debug":
        warnings.warn(
            "The bundled native extension was compiled without optimizations, so vector search "
            "runs roughly 100x slower. Rebuild it from the repository root with "
            "`uv run --with maturin maturin develop --release`.",
            RuntimeWarning,
            stacklevel=2,
        )


__version__ = native_core_version()

_warn_if_debug_build(native_build_profile())

__all__ = [
    "CoreAppender",
    "CoreCheckpointer",
    "CoreEngine",
    "__version__",
    "build_embedded_documents_payload",
    "core_document_to_json",
    "cuda_runtime_available",
    "decode_wal_segment",
    "encode_wal_segment",
    "native_build_profile",
    "native_core_abi_version",
    "native_core_version",
    "plan_segment_documents",
    "round_trip_core_json",
    "storage_schema_version",
]
