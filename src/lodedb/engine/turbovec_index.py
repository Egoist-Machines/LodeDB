"""TurboVec serving adapter for direct engine route profiles."""

from __future__ import annotations

import hashlib
import importlib
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

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


@dataclass(frozen=True)
class TurboVecSearchResult:
    """Stores compact stable-id search results and safe runtime telemetry."""

    stable_ids: NDArray[np.uint64]
    scores: NDArray[np.float32]
    native_backend: str
    native_used: bool


@dataclass(frozen=True)
class TurboVecServingIndex:
    """Stores a TurboVec IdMapIndex plus engine stable-id lookup metadata."""

    index: Any
    chunk_ids_by_stable_id: dict[int, str]
    document_ids_by_stable_id: dict[int, str]
    dim: int
    bit_width: int
    generation: int
    native_backend: str
    native_used: bool
    build_seconds: float

    def search(
        self,
        query_embedding: tuple[float, ...],
        *,
        top_k: int,
        allowlist_chunk_ids: tuple[str, ...] = (),
    ) -> TurboVecSearchResult:
        """Searches TurboVec by stable id and optionally restricts to chunk IDs."""

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dim:
            raise ValueError("query dimension does not match TurboVec index")
        kwargs: dict[str, Any] = {}
        if allowlist_chunk_ids:
            allowed = _allowlist_stable_ids(
                allowlist_chunk_ids,
                chunk_ids_by_stable_id=self.chunk_ids_by_stable_id,
            )
            if allowed.size == 0:
                return TurboVecSearchResult(
                    stable_ids=np.empty((1, 0), dtype=np.uint64),
                    scores=np.empty((1, 0), dtype=np.float32),
                    native_backend=self.native_backend,
                    native_used=self.native_used,
                )
            kwargs["allowlist"] = allowed
        effective_top_k = min(int(top_k), int(len(self.index)))
        if "allowlist" in kwargs:
            effective_top_k = min(effective_top_k, int(kwargs["allowlist"].size))
        if effective_top_k <= 0:
            return TurboVecSearchResult(
                stable_ids=np.empty((1, 0), dtype=np.uint64),
                scores=np.empty((1, 0), dtype=np.float32),
                native_backend=self.native_backend,
                native_used=self.native_used,
            )
        scores, stable_ids = self.index.search(query, k=effective_top_k, **kwargs)
        return TurboVecSearchResult(
            stable_ids=np.asarray(stable_ids, dtype=np.uint64),
            scores=np.asarray(scores, dtype=np.float32),
            native_backend=self.native_backend,
            native_used=self.native_used,
        )

    def search_batch(
        self,
        query_embeddings: NDArray[np.float32],
        *,
        top_k: int,
    ) -> TurboVecSearchResult:
        """Searches a whole query batch in one native call (no allowlist support).

        One call amortizes the per-query LUT/dispatch overhead the vendored
        kernel pays on entry; the binding already accepts 2D query batches.
        Result rows align with input query rows.
        """

        if top_k <= 0:
            raise ValueError("top_k must be positive")
        queries = np.ascontiguousarray(query_embeddings, dtype=np.float32)
        if queries.ndim != 2 or queries.shape[1] != self.dim:
            raise ValueError("query batch must be 2D and match the TurboVec dimension")
        effective_top_k = min(int(top_k), int(len(self.index)))
        if effective_top_k <= 0:
            return TurboVecSearchResult(
                stable_ids=np.empty((queries.shape[0], 0), dtype=np.uint64),
                scores=np.empty((queries.shape[0], 0), dtype=np.float32),
                native_backend=self.native_backend,
                native_used=self.native_used,
            )
        scores, stable_ids = self.index.search(queries, k=effective_top_k)
        return TurboVecSearchResult(
            stable_ids=np.asarray(stable_ids, dtype=np.uint64),
            scores=np.asarray(scores, dtype=np.float32),
            native_backend=self.native_backend,
            native_used=self.native_used,
        )

    def write(self, path: str | Path) -> dict[str, Any]:
        """Persists the TurboVec index payload and returns safe write metrics."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self.index.write(str(output))
        persist_ms = (time.perf_counter() - started) * 1000.0
        return {
            "compact_backend": "turbovec_idmap",
            "snapshot_bytes": output.stat().st_size,
            "persist_ms": persist_ms,
            "raw_payload_text_present": False,
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


def require_turbovec_available(id_map_index_class: Any | None = None) -> TurboVecCapability:
    """Raises a clear compact-backend error unless TurboVec validates at runtime."""

    capability = turbovec_capability(id_map_index_class=id_map_index_class)
    if not capability.available:
        raise RuntimeError(f"TurboVec compact backend unavailable: {capability.unavailable_reason}")
    return capability


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


def build_turbovec_serving_index(
    chunks: tuple[Any, ...],
    *,
    native_dim: int,
    bit_width: int,
    generation: int,
    id_map_index_class: Any | None = None,
    progress_label: str | None = None,
) -> TurboVecServingIndex:
    """Builds an IdMapIndex from engine chunks and stable uint64 chunk IDs."""

    if native_dim <= 0:
        raise ValueError("native_dim must be positive")
    if bit_width not in {2, 4}:
        raise ValueError("TurboVec bit_width must be 2 or 4")
    index_class = id_map_index_class or load_turbovec_id_map_index_class()
    capability = require_turbovec_available(id_map_index_class=index_class)
    _log_turbovec_build_progress(
        progress_label,
        phase="embedding_matrix",
        event="start",
        chunk_count=len(chunks),
        native_dim=native_dim,
        bit_width=bit_width,
        generation=generation,
        backend=capability.native_backend,
    )
    phase_started = time.perf_counter()
    embeddings = _chunk_embedding_matrix(chunks, native_dim=native_dim)
    _log_turbovec_build_progress(
        progress_label,
        phase="embedding_matrix",
        event="end",
        chunk_count=len(chunks),
        native_dim=native_dim,
        bit_width=bit_width,
        generation=generation,
        backend=capability.native_backend,
        elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
    )
    _log_turbovec_build_progress(
        progress_label,
        phase="stable_ids",
        event="start",
        chunk_count=len(chunks),
        native_dim=native_dim,
        bit_width=bit_width,
        generation=generation,
        backend=capability.native_backend,
    )
    phase_started = time.perf_counter()
    stable_ids = stable_uint64_ids_for_chunk_ids(tuple(str(chunk.chunk_id) for chunk in chunks))
    _log_turbovec_build_progress(
        progress_label,
        phase="stable_ids",
        event="end",
        chunk_count=len(chunks),
        native_dim=native_dim,
        bit_width=bit_width,
        generation=generation,
        backend=capability.native_backend,
        elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
    )
    started = time.perf_counter()
    index = index_class(dim=native_dim, bit_width=int(bit_width))
    if embeddings.shape[0]:
        _log_turbovec_build_progress(
            progress_label,
            phase="add_with_ids",
            event="start",
            chunk_count=len(chunks),
            native_dim=native_dim,
            bit_width=bit_width,
            generation=generation,
            backend=capability.native_backend,
        )
        phase_started = time.perf_counter()
        index.add_with_ids(embeddings, stable_ids)
        _log_turbovec_build_progress(
            progress_label,
            phase="add_with_ids",
            event="end",
            chunk_count=len(chunks),
            native_dim=native_dim,
            bit_width=bit_width,
            generation=generation,
            backend=capability.native_backend,
            elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
        )
    if hasattr(index, "prepare"):
        _log_turbovec_build_progress(
            progress_label,
            phase="prepare",
            event="start",
            chunk_count=len(chunks),
            native_dim=native_dim,
            bit_width=bit_width,
            generation=generation,
            backend=capability.native_backend,
        )
        phase_started = time.perf_counter()
        index.prepare()
        _log_turbovec_build_progress(
            progress_label,
            phase="prepare",
            event="end",
            chunk_count=len(chunks),
            native_dim=native_dim,
            bit_width=bit_width,
            generation=generation,
            backend=capability.native_backend,
            elapsed_ms=(time.perf_counter() - phase_started) * 1000.0,
        )
    build_seconds = time.perf_counter() - started
    return TurboVecServingIndex(
        index=index,
        chunk_ids_by_stable_id={
            int(stable_id): str(chunk.chunk_id)
            for stable_id, chunk in zip(stable_ids, chunks, strict=True)
        },
        document_ids_by_stable_id={
            int(stable_id): str(chunk.document_id)
            for stable_id, chunk in zip(stable_ids, chunks, strict=True)
        },
        dim=native_dim,
        bit_width=int(bit_width),
        generation=int(generation),
        native_backend=capability.native_backend,
        native_used=capability.native_used,
        build_seconds=build_seconds,
    )


def _log_turbovec_build_progress(
    progress_label: str | None,
    *,
    phase: str,
    event: str,
    chunk_count: int,
    native_dim: int,
    bit_width: int,
    generation: int,
    backend: str,
    elapsed_ms: float | None = None,
) -> None:
    """Emits raw-payload-free progress for direct TurboVec index construction."""

    if progress_label is None:
        return
    elapsed = "" if elapsed_ms is None else f" elapsed_ms={elapsed_ms:.3f}"
    print(
        "turbovec_build: "
        f"label={progress_label} phase={phase} event={event} "
        f"chunks={chunk_count} native_dim={native_dim} bit_width={bit_width} "
        f"generation={generation} backend={backend}{elapsed}",
        flush=True,
    )


def load_turbovec_serving_index(
    path: str | Path,
    chunks: tuple[Any, ...],
    *,
    generation: int,
    id_map_index_class: Any | None = None,
    post_load: Any | None = None,
) -> TurboVecServingIndex:
    """Loads a persisted TurboVec IdMapIndex and attaches redacted chunk metadata.

    ``post_load``, when given, is called with the raw loaded ``IdMapIndex``
    before `prepare()` and metadata attachment — the `.tvim-delta` replay
    hook uses it to apply journaled mutations so the loaded index matches
    the live pre-persist state.
    """

    index_class = id_map_index_class or load_turbovec_id_map_index_class()
    capability = require_turbovec_available(id_map_index_class=index_class)
    stable_ids = stable_uint64_ids_for_chunk_ids(tuple(str(chunk.chunk_id) for chunk in chunks))
    started = time.perf_counter()
    index = index_class.load(str(path))
    if post_load is not None:
        post_load(index)
    if hasattr(index, "prepare"):
        index.prepare()
    build_seconds = time.perf_counter() - started
    return TurboVecServingIndex(
        index=index,
        chunk_ids_by_stable_id={
            int(stable_id): str(chunk.chunk_id)
            for stable_id, chunk in zip(stable_ids, chunks, strict=True)
        },
        document_ids_by_stable_id={
            int(stable_id): str(chunk.document_id)
            for stable_id, chunk in zip(stable_ids, chunks, strict=True)
        },
        dim=int(index.dim),
        bit_width=int(index.bit_width),
        generation=int(generation),
        native_backend=capability.native_backend,
        native_used=capability.native_used,
        build_seconds=build_seconds,
    )


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


def _chunk_embedding_matrix(chunks: tuple[Any, ...], *, native_dim: int) -> NDArray[np.float32]:
    """Returns a contiguous embedding matrix for engine chunk records."""

    if not chunks:
        return np.empty((0, native_dim), dtype=np.float32)
    matrix = np.asarray([chunk.embedding for chunk in chunks], dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != native_dim:
        raise ValueError("chunk embeddings do not match native_dim")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _allowlist_stable_ids(
    chunk_ids: tuple[str, ...],
    *,
    chunk_ids_by_stable_id: dict[int, str],
) -> NDArray[np.uint64]:
    """Converts chunk-id allowlists into active TurboVec stable IDs."""

    allowed_chunks = set(chunk_ids)
    stable_ids = [
        stable_id
        for stable_id, chunk_id in chunk_ids_by_stable_id.items()
        if chunk_id in allowed_chunks
    ]
    return np.ascontiguousarray(stable_ids, dtype=np.uint64)


def _stable_uint64_for_text(value: str) -> int:
    """Returns the first eight SHA-256 bytes as a stable little-endian uint64."""

    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)
