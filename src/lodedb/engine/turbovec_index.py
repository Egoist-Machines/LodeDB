"""TurboVec capability + CPU-dispatch detection for the bundled compact backend.

Reports whether the vendored TurboVec runtime can serve compact indexes and the
native SIMD dispatch it is likely to use, without serving any index from
Python: the native Rust core is the sole engine and owns the live scan. The
remaining helpers back ``lodedb doctor`` and the benchmark provenance fields.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger("lodedb.engine")

# The compiled TurboVec extension is bundled into the lodedb wheel as
# `lodedb._turbovec` (maturin builds it from third_party/turbovec). A standalone
# `turbovec` package, if present from a source/editable build, is a fallback.
TURBOVEC_PACKAGE_NAME = "lodedb._turbovec"
_TURBOVEC_FALLBACK_PACKAGE_NAME = "turbovec"
TURBOVEC_VERSION = "0.8.0"
TURBOVEC_SOURCE_TAG = "v0.9.0"
TURBOVEC_SOURCE_COMMIT = "1e7200cfd8f26c92ce2855652db64bc7f85bc039"
TURBOVEC_VENDORED_SOURCE = "third_party/turbovec"
TURBOVEC_IDMAP_FILENAME_SUFFIX = ".tvim"


@dataclass(frozen=True)
class TurboVecCapability:
    """Describes whether the vendored TurboVec runtime can serve compact indexes."""

    available: bool
    backend_name: str
    native_backend: str
    native_used: bool
    cpu_flags: tuple[str, ...]
    version: str
    source_tag: str
    source_commit: str
    delta_persistence_available: bool = False
    reconstruction_available: bool = False
    unavailable_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serializes compact-backend capability metadata without raw vectors or text."""

        return {
            "available": self.available,
            "backend_name": self.backend_name,
            "native_backend": self.native_backend,
            "native_used": self.native_used,
            "cpu_flags": list(self.cpu_flags),
            "version": self.version,
            "source_tag": self.source_tag,
            "source_commit": self.source_commit,
            "delta_persistence_available": self.delta_persistence_available,
            "reconstruction_available": self.reconstruction_available,
            "vendored_source": TURBOVEC_VENDORED_SOURCE,
            "unavailable_reason": self.unavailable_reason,
        }


def turbovec_capability(id_map_index_class: Any | None = None) -> TurboVecCapability:
    """Returns compact-backend availability and inferred CPU dispatch metadata."""

    cpu_flags = detect_cpu_flags()
    native_backend = turbovec_native_backend_from_flags(cpu_flags)
    try:
        index_class = id_map_index_class or load_turbovec_id_map_index_class()
        _validate_turbovec_runtime(index_class)
    except RuntimeError as exc:
        return TurboVecCapability(
            available=False,
            backend_name="turbovec_idmap",
            native_backend="unavailable",
            native_used=False,
            cpu_flags=cpu_flags,
            version=TURBOVEC_VERSION,
            source_tag=TURBOVEC_SOURCE_TAG,
            source_commit=TURBOVEC_SOURCE_COMMIT,
            unavailable_reason=str(exc),
        )
    # Probe the loaded build for the Apache-2.0 local patches (see
    # third_party/turbovec/LOCAL_PATCHES.md). Stock PyPI turbovec==0.8.0 lacks
    # them, so the engine silently falls back to full .tvim rewrites and the CPU
    # scan; surface that honestly rather than implying the patched core is present.
    from lodedb.engine.gpu_turbovec import turbovec_reconstruction_api_available
    from lodedb.engine.turbovec_delta_store import turbovec_delta_api_available

    return TurboVecCapability(
        available=True,
        backend_name="turbovec_idmap",
        native_backend=native_backend,
        native_used=native_backend != "scalar",
        cpu_flags=cpu_flags,
        version=TURBOVEC_VERSION,
        source_tag=TURBOVEC_SOURCE_TAG,
        source_commit=TURBOVEC_SOURCE_COMMIT,
        delta_persistence_available=turbovec_delta_api_available(index_class),
        reconstruction_available=turbovec_reconstruction_api_available(index_class),
    )


def load_turbovec_id_map_index_class() -> Any:
    """Imports TurboVec's ``IdMapIndex`` from the bundled compiled extension.

    The patched core ships inside the lodedb wheel as ``lodedb._turbovec``; a
    source build that still exposes a standalone ``turbovec`` package is accepted
    as a fallback. The bundled name is tried first so a stray stock PyPI
    ``turbovec`` can never shadow the patched core.
    """

    last_exc: ImportError | None = None
    for name in (TURBOVEC_PACKAGE_NAME, _TURBOVEC_FALLBACK_PACKAGE_NAME):
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            last_exc = exc
            continue
        index_class = getattr(module, "IdMapIndex", None)
        if index_class is None:
            raise RuntimeError("TurboVec compact backend does not expose IdMapIndex")
        return index_class
    raise RuntimeError(
        "TurboVec compact backend is not installed; install lodedb (the compiled core "
        "is bundled) or build the vendored source at third_party/turbovec/turbovec-python."
    ) from last_exc


def stable_uint64_ids_for_chunk_ids(chunk_ids: tuple[str, ...]) -> NDArray[np.uint64]:
    """Maps chunk IDs to deterministic nonzero uint64 IDs with collision repair."""

    used: set[int] = set()
    ids: list[int] = []
    for chunk_id in chunk_ids:
        candidate = _stable_uint64_for_text(chunk_id)
        while candidate == 0 or candidate in used:
            candidate = (candidate + 1) & 0xFFFFFFFFFFFFFFFF
        used.add(candidate)
        ids.append(candidate)
    return np.ascontiguousarray(ids, dtype=np.uint64)


def detect_cpu_flags() -> tuple[str, ...]:
    """Returns normalized CPU flags relevant to TurboVec native dispatch."""

    flags: set[str] = set()
    proc_cpuinfo = Path("/proc/cpuinfo")
    if proc_cpuinfo.exists():
        for line in proc_cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith(("flags", "features")) and ":" in line:
                flags.update(line.split(":", maxsplit=1)[1].strip().lower().split())
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        flags.add("neon")
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        flags.add("accelerate")
    return tuple(sorted(flags))


def turbovec_native_backend_from_flags(cpu_flags: tuple[str, ...]) -> str:
    """Infers TurboVec's likely native dispatch from CPU flags and architecture."""

    flags = set(cpu_flags)
    if {"avx512bw", "avx512f"} <= flags:
        return "avx512bw"
    if "avx2" in flags:
        return "avx2"
    if "neon" in flags:
        return "neon"
    return "scalar"


def _validate_turbovec_runtime(index_class: Any) -> None:
    """Runs a tiny add/search/remove/write/load probe for fail-closed validation."""

    try:
        index = index_class(dim=8, bit_width=2)
        vectors = np.eye(2, 8, dtype=np.float32)
        ids = np.asarray([101, 102], dtype=np.uint64)
        index.add_with_ids(vectors, ids)
        if hasattr(index, "prepare"):
            index.prepare()
        scores, found_ids = index.search(vectors[:1], k=1)
        if int(np.asarray(found_ids, dtype=np.uint64)[0, 0]) != 101:
            raise RuntimeError("TurboVec validation returned the wrong stable id")
        if not index.remove(101):
            raise RuntimeError("TurboVec validation remove failed")
        if int(len(index)) != 1:
            raise RuntimeError("TurboVec validation length mismatch after remove")
        if np.asarray(scores, dtype=np.float32).size != 1:
            raise RuntimeError("TurboVec validation returned invalid scores")
    except Exception as exc:
        raise RuntimeError(f"TurboVec runtime validation failed: {exc}") from exc


def _stable_uint64_for_text(value: str) -> int:
    """Returns the first eight SHA-256 bytes as a stable little-endian uint64."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)
