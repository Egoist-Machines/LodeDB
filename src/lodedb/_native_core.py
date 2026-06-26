"""Private native-core shim backed by the bundled Rust extension.

This module is intentionally not public API. It keeps the rollout import path
(``lodedb._native_core``) stable while the root wheel still builds one Rust
extension module, ``lodedb._turbovec``.
"""

from __future__ import annotations

from lodedb._turbovec import (  # type: ignore[attr-defined]
    CoreEngine,
    core_document_to_json,
    native_core_version,
    round_trip_core_json,
    storage_schema_version,
)

__version__ = native_core_version()

__all__ = [
    "CoreEngine",
    "__version__",
    "core_document_to_json",
    "native_core_version",
    "round_trip_core_json",
    "storage_schema_version",
]
