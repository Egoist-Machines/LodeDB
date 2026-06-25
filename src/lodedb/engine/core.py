"""In-process LodeDB engine: per-client isolated document indexing, search, and persistence."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import shutil
import threading
import uuid
import weakref
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from lodedb.engine import _filter_plan
from lodedb.engine._atomic_io import durable_replace
from lodedb.engine._commit_manifest import (
    COMMIT_MANIFEST_SUFFIX,
    DEFAULT_EPOCHS_RETAINED,
    base_json_path,
    base_tvim_path,
    base_tvlex_path,
    base_tvtext_path,
    build_commit_body,
    commit_manifest_path,
    generation_dir,
    is_commit_manifest_name,
    list_base_epochs,
    read_commit_manifest,
    write_commit_manifest,
)
from lodedb.engine._filelock import WriterLock, lodedb_lock_timeout_from_env
from lodedb.engine._lexical import (
    Bm25Index,
    build_chunk_texts,
    build_chunk_token_lists,
    reciprocal_rank_fusion,
    tokenize,
)
from lodedb.engine._predicate import compile_metadata_filter, validate_metadata_filter
from lodedb.engine.document_text_store import (
    DocumentTextStore,
    read_legacy_text_sidecar,
)
from lodedb.engine.embedding_backends import (
    EngineEmbeddingBackend,
    HashEmbeddingBackend,
)
from lodedb.engine.lexical_index_store import LexicalIndexStore
from lodedb.engine.route_registry import (
    SUPPORTED_ROUTE_CLASSES,
    RouteDecision,
    RouteRegistry,
)
from lodedb.engine.runtime_policy import (
    CommitMode,
    GpuDirectTurboVecPolicy,
    MpsDirectTurboVecPolicy,
    TvimDeltaPersistencePolicy,
    commit_mode_from_env,
    gpu_direct_turbovec_max_batch_from_env,
    gpu_direct_turbovec_policy_from_env,
    gpu_direct_turbovec_should_use,
    gpu_memory_budget_bytes_from_env,
    mps_direct_turbovec_max_batch_from_env,
    mps_direct_turbovec_policy_from_env,
    mps_direct_turbovec_should_use,
    mps_memory_budget_bytes_from_env,
    tvim_delta_persistence_policy_from_env,
)
from lodedb.engine.state_journal_store import StateJournalStore
from lodedb.engine.turbovec_delta_store import (
    TvimDeltaStore,
    turbovec_delta_api_available,
)
from lodedb.engine.turbovec_index import (
    TurboVecServingIndex,
    build_turbovec_serving_index,
    load_turbovec_serving_index,
    require_turbovec_available,
    stable_uint64_ids_for_chunk_ids,
    turbovec_capability,
)
from lodedb.engine.wal_store import (
    DEFAULT_CHECKPOINT_BYTES,
    DEFAULT_CHECKPOINT_OPS,
    WalStore,
    wal_path,
)

# The optional CUDA path (`gpu_turbovec`) is imported lazily inside the methods
# that use it, so importing the engine never requires CuPy or a GPU. The
# type-only import below keeps annotations precise with no runtime dependency.
if TYPE_CHECKING:
    from lodedb.engine.gpu_turbovec import GpuDirectTurboVecSession
    from lodedb.engine.mps_turbovec import MpsDirectTurboVecSession

logger = logging.getLogger("lodedb.engine")

DIRECT_TURBOVEC_STORAGE_PROFILE = "turbovec_direct"
# Cap on the per-index in-memory query-latency ring. Latency samples are
# runtime telemetry (stats/audit percentiles), not durable state, so a long
# query stream keeps only the most recent samples rather than growing without
# bound or bloating the JSON snapshot/journal headers.
QUERY_LATENCY_SAMPLE_CAP = 1024
DEFAULT_INDEX_ID = "default"
DEFAULT_INDEX_NAME = "Default index"
# Cap on rows sampled for the (deferred, query-warm) quantization-drift metric.
_DRIFT_SAMPLE_LIMIT = 16
ACTIVE_INDEX_STATUS = "ready"
LEGACY_INDEX_TIMESTAMP = "1970-01-01T00:00:00+00:00"
QUERY_INCLUDE_METADATA = "metadata"
# Retrieval modes for ``query``/``query_batch``. ``vector`` is the default and
# leaves the existing scan untouched. ``hybrid`` fuses the vector scan with a
# lexical BM25 ranker via Reciprocal Rank Fusion; ``lexical`` returns the BM25
# ranking alone. The lexical and hybrid modes are a pure-Python CPU post-step
# that requires retained raw text (the BM25 index is rebuilt from it).
RETRIEVAL_MODE_VECTOR = "vector"
RETRIEVAL_MODE_HYBRID = "hybrid"
RETRIEVAL_MODE_LEXICAL = "lexical"
RETRIEVAL_MODES = frozenset(
    {RETRIEVAL_MODE_VECTOR, RETRIEVAL_MODE_HYBRID, RETRIEVAL_MODE_LEXICAL}
)
_LEXICAL_MODES = frozenset({RETRIEVAL_MODE_HYBRID, RETRIEVAL_MODE_LEXICAL})
# A lexical/hybrid query pulls a widened candidate pool from each ranker before
# fusing, so the fused top-k is not capped by either ranker's own top-k. The
# pool is ``max(k * factor, floor)`` chunks; RRF (c=60) gives negligible weight
# past a few dozen ranks, so a bounded widening is sufficient and keeps the
# lexical pass O(matching chunks).
_LEXICAL_POOL_FACTOR = 5
_LEXICAL_POOL_FLOOR = 50
# When the cached BM25 index is stale by only a small chunk-level delta, fold
# just the changed documents in place instead of a full O(total tokens) rebuild.
# Changed-document ids are tracked at mutation time; once their count exceeds a
# fraction of the corpus a full rebuild is cheaper than touching the postings a
# document at a time, so the incremental path is bounded to genuinely small
# changes (the common incremental-``add`` case).
_LEXICAL_INCREMENTAL_MAX_FRACTION = 0.25
# Floor on the pending changed-document count before a full rebuild is forced,
# so a tiny corpus does not rebuild on its first few writes (a 4-doc corpus
# would otherwise hit the fraction at one change). The pending sets are bounded
# at record time by max(this floor, the fraction of the corpus), so a long run
# of writes with no intervening hybrid query cannot grow them without bound.
_LEXICAL_REBUILD_FLOOR = 64
PROJECTION_INITIAL_SOURCE = "initial_identity_prefix"
PROJECTION_REFIT_SOURCE_INT8 = "int8_rerank"
PROJECTION_REFIT_SOURCE_TRANSIENT_FLOAT = "transient_full_float"
CASCADE_RERANK_MATERIALIZATION_BLOCK_ROWS = 16_384


@dataclass(frozen=True)
class EngineSecurityConfig:
    """Stores the local engine's binding and telemetry settings."""

    bind_host: str
    route_profile: str = ""
    telemetry_mode: str = "metrics_only"
    allow_raw_result_text: bool = False
    persist_lexical_index: bool = False


@dataclass(frozen=True)
class EngineRequestContext:
    """Carries the client id and clock for one engine request."""

    client_id: str
    now: datetime


@dataclass(frozen=True)
class EngineResponse:
    """Represents an endpoint-shaped status code and JSON-like response body."""

    status_code: int
    body: dict[str, Any]


@dataclass(frozen=True)
class EngineDocument:
    """Stores one source document supplied to the local engine."""

    document_id: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EngineVectorDocument:
    """Stores one pre-embedded document supplied to the local engine (vector-in).

    Unlike :class:`EngineDocument`, the caller supplies the embedding vector
    directly: the engine skips chunking and embedding and stores the vector as a
    single chunk. ``text`` is optional retained payload text only; when present
    and raw-text storage is enabled it is stored in the dedicated text sidecar,
    never in the redacted snapshot or metadata. The vector must match the
    index's ``native_dim`` and is expected to be L2-normalized (the SDK
    normalizes by default) so cosine scoring stays comparable with the text path.
    """

    document_id: str
    vector: tuple[float, ...]
    metadata: dict[str, str] = field(default_factory=dict)
    text: str | None = None


@dataclass(frozen=True)
class EngineQuery:
    """Stores one retrieval query supplied to the local engine.

    Exactly one of ``text`` or ``embedding`` carries the query: when
    ``embedding`` is set (vector-in search) the engine skips embedding and uses
    it directly; otherwise ``text`` is embedded with the index backend.
    """

    text: str
    top_k: int = 10
    filter: dict[str, Any] | None = None
    include: tuple[str, ...] = ()
    route_drifted: bool = False
    route_failed: bool = False
    high_risk: bool = False
    # Retrieval mode: ``vector`` (default), ``hybrid`` (BM25 + RRF), or
    # ``lexical`` (BM25 only). See ``RETRIEVAL_MODES``.
    mode: str = RETRIEVAL_MODE_VECTOR
    embedding: tuple[float, ...] | None = None


@dataclass(frozen=True)
class EngineRoutePolicy:
    """Constrains one client-facing engine route profile and its index shape."""

    profile: str
    label: str
    client_note: str
    model: str
    provider: str
    task: str
    native_dim: int
    method_template: str
    experimental: bool = False
    index_backend: str = DIRECT_TURBOVEC_STORAGE_PROFILE
    turbovec_bit_width: int | None = None

    def validate_index_request(
        self,
        *,
        model: str,
        provider: str,
        task: str,
        native_dim: int,
        route: RouteDecision,
    ) -> str | None:
        """Returns an error message when an index request does not match this profile."""

        if model != self.model:
            return f"model must match route profile {self.profile}"
        if provider != self.provider:
            return f"provider must match route profile {self.profile}"
        if task != self.task:
            return f"task must match route profile {self.profile}"
        if native_dim != self.native_dim:
            return f"native_dim must match route profile {self.profile}"
        if self.index_backend != DIRECT_TURBOVEC_STORAGE_PROFILE:
            if route.route_classification not in SUPPORTED_ROUTE_CLASSES:
                return f"route profile {self.profile} is not supported by the route registry"
            if route.method_template != self.method_template:
                return f"route registry selection must match route profile {self.profile}"
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serializes only client-safe route profile metadata."""

        return {
            "profile": self.profile,
            "label": self.label,
            "client_note": self.client_note,
            "index_backend": self.index_backend,
            "turbovec_bit_width": self.turbovec_bit_width,
            "experimental": self.experimental,
        }


@dataclass(frozen=True)
class IndexedChunk:
    """Stores one indexed chunk and its transient full embedding.

    The embedding is only populated between ingest and the direct TurboVec
    sync; ready states discard it (`_discard_direct_turbovec_transient_embeddings`).
    """

    chunk_id: str
    document_id: str
    content_hash: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class _PendingBuildChunk:
    """Carries chunk metadata while cold build batches raw chunk texts for embedding."""

    chunk_id: str
    document_id: str
    content_hash: str


@dataclass(frozen=True)
class _PlannedDocumentMutation:
    """Stores one validated document mutation plan before batch embeddings are applied."""

    document: EngineDocument
    content_hash: str
    new_chunk_ids: tuple[str, ...]
    planned_chunks: tuple[tuple[str, str, IndexedChunk | None, int | None], ...]
    unchanged: bool = False


@dataclass(frozen=True)
class _TurboVecChangeset:
    """Exact set of chunk-level changes a mutation verb applied to ``state.chunks``.

    Threaded from the upsert/delete verbs into ``_sync_direct_turbovec_index`` so
    the sync can reconcile the serving index in O(changed) instead of re-deriving
    the full diff against the whole corpus on every mutation. ``added_chunks`` are
    the :class:`IndexedChunk` objects the verb just wrote; ``removed_chunk_ids``
    are the chunk ids it removed. The changeset is only a hint — the sync still
    truth-checks every id against the post-mutation ``state.chunks`` and the live
    id map, so an over-reporting changeset (e.g. a chunk added then orphaned in
    one batch) stays correct without scanning the corpus.
    """

    added_chunks: tuple[IndexedChunk, ...] = ()
    removed_chunk_ids: tuple[str, ...] = ()


@dataclass
class ClientIndexState:
    """Stores per-client index state without crossing client boundaries."""

    client_id: str
    client_id_hash: str
    index_id: str
    index_key: str
    model: str
    provider: str
    task: str
    native_dim: int
    name: str = DEFAULT_INDEX_NAME
    metadata: dict[str, str] = field(default_factory=dict)
    status: str = ACTIVE_INDEX_STATUS
    created_at: str = LEGACY_INDEX_TIMESTAMP
    updated_at: str = LEGACY_INDEX_TIMESTAMP
    chunks: dict[str, IndexedChunk] = field(default_factory=dict)
    document_hashes: dict[str, str] = field(default_factory=dict)
    document_chunk_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    document_metadata: dict[str, dict[str, str]] = field(default_factory=dict)
    # Opt-in raw document text keyed by document id. Populated only when the
    # engine runs with ``EngineSecurityConfig.allow_raw_result_text`` and held
    # in its own ``.tvtext`` sidecar (never the redacted ``.json`` snapshot,
    # the ``.jsd`` journal, telemetry, or audit output). Default builds leave
    # it empty so the raw-payload-free guarantees of every other path hold.
    document_text: dict[str, str] = field(default_factory=dict)
    # Opt-in persisted lexical postings keyed by document id: per-document, the
    # per-chunk token lists (aligned with ``document_chunk_ids``) captured at
    # ingest time. Populated only when the engine runs with
    # ``EngineSecurityConfig.persist_lexical_index`` and held in its own
    # ``.tvlex`` sidecar (never the redacted ``.json`` snapshot, the ``.jsd``
    # journal, telemetry, or audit output). Default builds leave it empty so the
    # raw-payload-free guarantees of every other path hold.
    document_tokens: dict[str, list[list[str]]] = field(default_factory=dict)
    embedded_chunk_count: int = 0
    cache_reuse_count: int = 0
    delete_count: int = 0
    deleted_chunk_count: int = 0
    query_count: int = 0
    fallback_count: int = 0
    fallback_reasons: dict[str, int] = field(default_factory=dict)
    # Bounded ring of recent per-query latencies (most recent
    # QUERY_LATENCY_SAMPLE_CAP). Runtime telemetry only: excluded from the
    # persisted snapshot/journal header, so stats report a recent window and a
    # restart resets it.
    query_latency_ms: deque[float] = field(
        default_factory=lambda: deque(maxlen=QUERY_LATENCY_SAMPLE_CAP)
    )
    storage_profile: str = DIRECT_TURBOVEC_STORAGE_PROFILE
    # Direct TurboVec quantization width, fixed at index creation from the
    # active route profile (legacy snapshots default to 4-bit).
    turbovec_bit_width: int = 4


@dataclass(frozen=True)
class _EngineTurboVecQueryResult:
    """Carries TurboVec stable-id search output back to engine materialization."""

    index: TurboVecServingIndex
    stable_ids: np.ndarray
    scores: np.ndarray
    native_used: bool
    native_backend: str
    retrieval_mode: str = "direct_turbovec"
    fallback_used: bool = False
    stage_one_backend: str = ""
    compact_route_fallback: bool = False
    # The ``gpu_*`` fields below are the SHARED resident-scan telemetry namespace,
    # populated by both the CUDA path and the opt-in Apple-GPU (MPS) path via
    # ``_try_query_resident_direct_batch``. The name is historical (CUDA shipped
    # first) and intentionally kept stable -- it is the recorded key across the GPU
    # benchmark history and harnesses. Which backend produced a row is disambiguated
    # by ``stage_one_backend`` (e.g. ``mps_torch_exact_direct``), so on Apple Silicon
    # ``gpu_estimated_bytes`` reports MPS unified-memory bytes.
    gpu_stage_one_status: str = "not_applicable"
    gpu_estimated_bytes: int = 0
    gpu_budget_bytes: int = 0
    gpu_copy_back_bytes: int = 0
    gpu_fallback_reason: str = ""
    gpu_query_count: int = 0
    gpu_resident_upload_build_ms: float = 0.0
    gpu_stage_one_search_ms: float = 0.0
    gpu_device_to_host_copy_ms: float = 0.0
    gpu_stage_one_tile_count: int = 0


@dataclass(frozen=True)
class _PreparedQuery:
    """Stores validated query state before embedding and direct execution."""

    query: EngineQuery
    query_filter: dict[str, Any]
    includes: tuple[str, ...]
    route: RouteDecision
    mode: str = RETRIEVAL_MODE_VECTOR


class _MetadataPostingIndex:
    """Generation-keyed posting index that resolves a filter to chunk ids fast.

    Maps each ``(metadata_key, value)`` pair to the documents carrying it and
    each document to its chunk ids, so a filtered allowlist is built in
    O(matching documents + their chunks) instead of scanning the whole corpus.
    It is cached per index generation (rebuilt lazily on the first filtered
    query after a mutation, exactly like the resident GPU session), so it adds
    nothing to the write/commit path.
    """

    __slots__ = ("generation", "_postings", "_chunks_by_document", "_fields", "_all_docs")

    def __init__(
        self,
        generation: int,
        postings: dict[tuple[str, str], set[str]],
        chunks_by_document: dict[str, list[str]],
        fields: dict[str, _filter_plan.FieldIndex],
        all_docs: set[str],
    ) -> None:
        self.generation = generation
        self._postings = postings
        self._chunks_by_document = chunks_by_document
        self._fields = fields
        self._all_docs = all_docs

    def allowlist(self, query_filter: Mapping[str, Any]) -> tuple[str, ...]:
        """Returns the eligible chunk ids for a validated metadata/document filter."""

        document_set: set[str] | None = None
        metadata = query_filter.get("metadata")
        if metadata:
            for key, value in metadata.items():
                posting = self._postings.get((str(key), str(value)))
                if not posting:
                    return ()
                document_set = set(posting) if document_set is None else (document_set & posting)
                if not document_set:
                    return ()
        document_ids = query_filter.get("document_ids")
        if document_ids is not None:
            requested = {str(document_id) for document_id in document_ids}
            document_set = requested if document_set is None else (document_set & requested)
        if document_set is None:  # empty filter: callers guard, but match-all is correct
            document_set = set(self._chunks_by_document)
        chunk_ids: list[str] = []
        for document_id in document_set:
            chunk_ids.extend(self._chunks_by_document.get(document_id, ()))
        return tuple(chunk_ids)

    def document_allowlist(self, query_filter: Mapping[str, Any]) -> set[str]:
        """Resolves a validated filter to the matching document-id set, no scan.

        Uses set operations and bisect over the per-field value indexes
        (``_filter_plan``), so resolution is O(matches + log V) rather than the
        O(corpus) compiled-matcher scan. Handles the full predicate grammar and
        the ``document_ids`` allowlist, and is shared by filtered search (then
        expanded to chunk ids) and filtered enumeration / count.
        """

        metadata = query_filter.get("metadata")
        if metadata:
            document_set = _filter_plan.resolve(metadata, self._fields, self._all_docs)
        else:
            document_set = set(self._all_docs)
        document_ids = query_filter.get("document_ids")
        if document_ids is not None:
            requested = {str(document_id) for document_id in document_ids}
            document_set &= requested
        return document_set

    def chunk_allowlist(self, query_filter: Mapping[str, Any]) -> tuple[str, ...]:
        """Returns eligible chunk ids for a validated filter via the planner."""

        chunk_ids: list[str] = []
        for document_id in self.document_allowlist(query_filter):
            chunk_ids.extend(self._chunks_by_document.get(document_id, ()))
        return tuple(chunk_ids)


class _LexicalServingIndex:
    """Generation-keyed BM25 index over the corpus chunks for hybrid retrieval.

    Built lazily on the first lexical or hybrid query of a generation and reused
    after that, exactly like the metadata posting index and the resident GPU
    session — so it never touches the write/commit path. When a later generation
    differs from the cached one, only the documents changed since the cache was
    last stamped (tracked at mutation time) are folded into the existing
    :class:`Bm25Index` via document-group replace/remove and the cached
    generation is re-stamped, instead of rebuilding from scratch. It is held in
    memory only: a BM25 inverted index is payload-derived and must never reach
    the redacted artifacts or telemetry.
    """

    __slots__ = ("generation", "bm25")

    def __init__(self, generation: int, bm25: Bm25Index) -> None:
        self.generation = generation
        # The BM25 index owns the chunk-id -> stable-position mapping, so a
        # metadata allowlist (a set of chunk ids) maps to the positions BM25 may
        # score, keeping the lexical ranker constrained to the same subset as the
        # vector ranker. Reading positions from the index (rather than a snapshot
        # taken at build time) keeps the allowlist correct after an in-place
        # incremental update reassigns positions for added units.
        self.bm25 = bm25

    def allowed_positions(self, allowed_chunk_ids: tuple[str, ...]) -> set[int]:
        """Maps an allowlist of chunk ids to BM25 unit positions (drops unknown ids)."""

        position_of = self.bm25.position_of
        positions: set[int] = set()
        for chunk_id in allowed_chunk_ids:
            position = position_of(chunk_id)
            if position is not None:
                positions.add(position)
        return positions


@dataclass(frozen=True)
class _ResidentDirectBackend:
    """Describes one resident-scan backend (GPU/MPS) for the shared batch dispatcher.

    Captures only what differs between the CUDA and Apple-GPU direct paths so
    :meth:`LodeEngine._try_query_resident_direct_batch` can drive both. ``kind`` is
    the human label used in ``required`` error messages; ``reason_prefix`` namespaces
    the visible fallback reasons (``gpu_direct_*`` / ``mps_direct_*``). ``build_session``
    and ``estimated_bytes`` close over the backend's session class and serving index.
    """

    kind: str
    reason_prefix: str
    policy: Any
    dependency_available: bool
    api_available: bool
    unavailable_reason: str
    default_unavailable_reason: str
    should_use: Callable[..., bool]
    max_batch: int | None
    max_top_k: int
    sessions: dict[str, Any]
    memory_budget_bytes: int | None
    build_session: Callable[[], Any]
    estimated_bytes: Callable[[Any], int]


def _synchronized(method):
    """Serializes a public engine operation on the per-engine in-process lock.

    The local dev server shares one engine across request threads, and a
    mutation rebuilds the cached columnar index that a concurrent query reads,
    so every externally reachable operation (mutations and queries alike) runs
    under one reentrant lock. This guards in-process threads; the cross-process
    single-writer guarantee is the separate file lock. The lock favors
    correctness over read parallelism — see the README concurrency notes.
    """

    @functools.wraps(method)
    def _locked(self, *args, **kwargs):
        with self._op_lock:
            return method(self, *args, **kwargs)

    return _locked


class LodeEngine:
    """OSS engine: create/upsert/delete/query/stats over TurboVec + ``.tvim``/``.tvd``/``.jsd``.

    Auth-free and local by design: it runs in-process with no authentication,
    bound to loopback, with metrics-only telemetry.
    """

    def __init__(
        self,
        *,
        security: EngineSecurityConfig,
        route_registry: RouteRegistry,
        chunk_character_limit: int = 900,
        persistence_dir: str | Path | None = None,
        read_only: bool = False,
        fsync_on_commit: bool = False,
        commit_mode: CommitMode | None = None,
        embedding_backend: EngineEmbeddingBackend | None = None,
        route_policy: EngineRoutePolicy | None = None,
        gpu_memory_budget_bytes: int | None = None,
        turbovec_id_map_index_class: Any | None = None,
        tvim_delta_persistence_policy: TvimDeltaPersistencePolicy | None = None,
        gpu_direct_turbovec_policy: GpuDirectTurboVecPolicy | None = None,
        gpu_direct_turbovec_max_batch: int | None = None,
        mps_direct_turbovec_policy: MpsDirectTurboVecPolicy | None = None,
        mps_direct_turbovec_max_batch: int | None = None,
        mps_memory_budget_bytes: int | None = None,
    ) -> None:
        """Initializes engine state and optionally loads local persisted indexes."""

        if not is_private_bind_host(security.bind_host):
            raise ValueError("engine bind_host must be loopback or private network address")
        if security.telemetry_mode != "metrics_only":
            raise ValueError("engine telemetry_mode must be metrics_only")
        if chunk_character_limit <= 0:
            raise ValueError("chunk_character_limit must be positive")
        self.security = security
        self.route_registry = route_registry
        self.chunk_character_limit = chunk_character_limit
        self.persistence_dir = Path(persistence_dir) if persistence_dir is not None else None
        # A read-only handle never mutates or persists, and takes no writer lock
        # (so it neither blocks nor is blocked by a live writer).
        self._read_only = bool(read_only)
        # fsync each published file + its directory on commit (power-loss
        # durability) vs. the default atomic-but-fast rename.
        self._fsync_on_commit = bool(fsync_on_commit)
        # Per-mutation commit strategy. ``generation`` (default) publishes a
        # crash-atomic MVCC generation on every mutation; ``wal`` appends one
        # framed record to ``<key>.wal`` per mutation and checkpoints into a
        # generation periodically (opt-in, single-writer; see wal_store).
        self._commit_mode = (
            commit_mode if commit_mode is not None else commit_mode_from_env()
        )
        # Open WAL stores per index key (writer handles only). Populated lazily on
        # the first WAL append/load; never created for a read-only handle or in
        # generation mode.
        self._wal_stores: dict[str, WalStore] = {}
        # Ops/bytes thresholds that trigger a WAL -> generation checkpoint.
        self._wal_checkpoint_ops = DEFAULT_CHECKPOINT_OPS
        self._wal_checkpoint_bytes = DEFAULT_CHECKPOINT_BYTES
        # Set while replaying WAL records on open so the replayed mutations
        # re-drive the engine verbs without re-appending to the WAL.
        self._wal_replaying = False
        # The logical mutation a verb stages for the next WAL append, keyed by
        # index key: ``(op, payload)``. Consumed by ``_append_wal_record`` so the
        # single ``_persist_state`` funnel records the right verb in WAL mode.
        self._pending_wal_records: dict[str, tuple[str, dict[str, Any]]] = {}
        # Single-writer lock timeout (seconds), read once at construction.
        self._persist_lock_timeout = lodedb_lock_timeout_from_env()
        self.embedding_backend = embedding_backend
        self.route_policy = route_policy
        self.gpu_memory_budget_bytes = (
            gpu_memory_budget_bytes
            if gpu_memory_budget_bytes is not None
            else gpu_memory_budget_bytes_from_env()
        )
        self.turbovec_id_map_index_class = turbovec_id_map_index_class
        self.tvim_delta_persistence_policy = (
            tvim_delta_persistence_policy
            if tvim_delta_persistence_policy is not None
            else tvim_delta_persistence_policy_from_env()
        )
        self.gpu_direct_turbovec_policy = (
            gpu_direct_turbovec_policy
            if gpu_direct_turbovec_policy is not None
            else gpu_direct_turbovec_policy_from_env()
        )
        # Optional batch cap for the auto policy (default None = no cap): with
        # the torch.topk top-k the GPU scan beats the CPU kernel at every batch
        # >= 2, so memory admission is the only gate. Operators can still set a
        # cap to force a CPU flip above some batch (e.g. for memory headroom).
        self.gpu_direct_turbovec_max_batch = (
            gpu_direct_turbovec_max_batch
            if gpu_direct_turbovec_max_batch is not None
            else gpu_direct_turbovec_max_batch_from_env()
        )
        self._gpu_direct_turbovec_sessions: dict[str, GpuDirectTurboVecSession] = {}
        # Opt-in Apple-GPU (MPS) exact scan; OFF by default (NEON is the default
        # and faster on measured Apple hardware). Mirrors the CUDA policy shape.
        self.mps_direct_turbovec_policy = (
            mps_direct_turbovec_policy
            if mps_direct_turbovec_policy is not None
            else mps_direct_turbovec_policy_from_env()
        )
        self.mps_direct_turbovec_max_batch = (
            mps_direct_turbovec_max_batch
            if mps_direct_turbovec_max_batch is not None
            else mps_direct_turbovec_max_batch_from_env()
        )
        self.mps_memory_budget_bytes = (
            mps_memory_budget_bytes
            if mps_memory_budget_bytes is not None
            else mps_memory_budget_bytes_from_env()
        )
        self._mps_direct_turbovec_sessions: dict[str, MpsDirectTurboVecSession] = {}
        self._pending_tvim_deltas: dict[str, dict[str, tuple[int, ...]]] = {}
        self._pending_state_journal_documents: dict[str, dict[str, dict[str, None]]] = {}
        self._turbovec_drift_telemetry: dict[str, dict[str, float]] = {}
        # Recently-added rows (id + embedding) awaiting an opportunistic
        # quantization-drift sample. Sampling needs a search, which needs the
        # warm SIMD layout, so it is deferred off the commit path to the next
        # query that warms the layout (see _sample_pending_drift). Embeddings
        # are buffered here because the transient copies in state.chunks are
        # zeroed right after the commit.
        self._pending_drift_samples: dict[str, list[tuple[str, tuple[float, ...]]]] = {}
        self._tvim_persist_telemetry: dict[str, dict[str, float | str]] = {}
        self._state_load_telemetry: dict[str, dict[str, float]] = {}
        self._indexes: dict[str, ClientIndexState] = {}
        self._turbovec_indexes: dict[str, TurboVecServingIndex] = {}
        self._index_generations: dict[str, int] = {}
        # Generation-keyed metadata posting index per index key, for O(matches)
        # filter allowlists. Rebuilt lazily when the generation advances.
        self._metadata_posting_indexes: dict[str, _MetadataPostingIndex] = {}
        # Generation-keyed in-memory BM25 index per index key, for lexical/hybrid
        # retrieval. Built lazily from retained raw text on the first lexical
        # query of a generation; never persisted (a BM25 index is payload-derived).
        self._lexical_indexes: dict[str, _LexicalServingIndex] = {}
        # Documents changed since the cached lexical index was last stamped, per
        # index key: ``{"upserted": set, "deleted": set}``. Accumulated at
        # mutation time (later-wins) so the next lexical/hybrid query folds in
        # only those documents (O(changed)) instead of diffing the whole corpus.
        self._pending_lexical_documents: dict[str, dict[str, set[str]]] = {}
        # Index keys whose pending lexical delta grew past the rebuild bound, so
        # the next lexical/hybrid query must do one full rebuild rather than a
        # document-at-a-time fold. Set at record time; cleared on rebuild.
        self._lexical_full_rebuild: set[str] = set()
        # Live base epoch per index key: which ``<key>.gen/g<epoch>.*`` artifacts
        # the committed root manifest points at. Set on load/commit; absent for a
        # not-yet-persisted or legacy (pre-commit-manifest) index.
        self._base_epochs: dict[str, int] = {}
        self._audit_events: list[dict[str, Any]] = []
        self._metrics: list[dict[str, Any]] = []
        # Serializes public operations within one process. The local dev server
        # shares one engine across request threads, and a mutation rebuilds the
        # cached columnar view a concurrent query reads, so reads and writes
        # alike run under this reentrant lock (see ``@_synchronized``).
        self._op_lock = threading.RLock()
        self._writer_lock: WriterLock | None = None
        if self.persistence_dir is not None:
            if self._read_only:
                # Read-only handles take NO writer lock — they neither block nor
                # are blocked by a live writer. Each index loads the single
                # consistent generation named by its atomic ``<key>.commit.json``
                # root manifest, so a reader never observes a torn cross-file
                # mix (no retry needed; that was the Stage 1 stopgap).
                self._load_persisted_indexes()
            else:
                self.persistence_dir.mkdir(parents=True, exist_ok=True)
                # Acquire the single-writer lock before loading. A second process
                # opening the same path waits for this handle to close and then
                # fails fast (ConcurrentWriterError) once the timeout elapses.
                self._acquire_writer_lock()
                try:
                    self._load_persisted_indexes()
                except BaseException:
                    self._release_writer_lock()
                    raise

    @_synchronized
    def checkpoint(self) -> None:
        """Folds any outstanding WAL into a committed generation (no-op otherwise).

        The explicit durability checkpoint behind the SDK ``persist()`` in WAL
        commit mode: it commits a fresh generation for every index that has
        WAL-logged-but-not-yet-checkpointed mutations and truncates the WAL. In
        ``generation`` mode there is nothing buffered, so it does nothing. A
        read-only handle never checkpoints.
        """

        if self._read_only:
            return
        self._checkpoint_all_wals()

    def _acquire_writer_lock(self) -> None:
        """Takes the exclusive single-writer lock for this engine's directory."""

        if self.persistence_dir is None or self._writer_lock is not None:
            return
        lock = WriterLock(self.persistence_dir)
        lock.acquire(self._persist_lock_timeout)
        self._writer_lock = lock
        # Best-effort release if the handle is dropped without close(); the OS
        # also releases the advisory lock when the process exits.
        self._writer_lock_finalizer = weakref.finalize(self, lock.release)

    def _release_writer_lock(self) -> None:
        """Releases the single-writer lock so another handle can open the path."""

        lock = self._writer_lock
        if lock is None:
            return
        self._writer_lock = None
        finalizer = getattr(self, "_writer_lock_finalizer", None)
        if finalizer is not None:
            finalizer.detach()
            self._writer_lock_finalizer = None
        lock.release()

    def close(self) -> None:
        """Checkpoints any open WAL, then releases the single-writer lock.

        In ``wal`` commit mode a clean close folds the outstanding WAL into a
        fresh committed generation and truncates it, so the next open finds an
        up-to-date base and an empty WAL (replay only ever has to recover an
        *unclean* shutdown). A best-effort guard keeps a checkpoint failure from
        leaking the writer lock — the WAL is still durable and will replay on the
        next open. After this the engine must not persist again.
        """

        try:
            if not self._read_only:
                self._checkpoint_all_wals()
        finally:
            self._release_writer_lock()

    def _require_writable(self) -> None:
        """Raises if this engine was opened read-only (mutations are forbidden).

        Defense in depth: the SDK already blocks mutating verbs on a read-only
        handle, so this should never fire on the normal path.
        """

        if self._read_only:
            raise RuntimeError("LodeDB is open read-only; reopen without read_only=True to mutate")

    @property
    def audit_events(self) -> tuple[dict[str, Any], ...]:
        """Returns redacted audit events emitted by endpoint handlers."""

        with self._op_lock:
            return tuple(dict(event) for event in self._audit_events)

    @property
    def metrics(self) -> tuple[dict[str, Any], ...]:
        """Returns metrics-only telemetry emitted by endpoint handlers."""

        with self._op_lock:
            return tuple(dict(metric) for metric in self._metrics)

    @_synchronized
    def create_index(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        model: str | None = None,
        provider: str | None = None,
        task: str | None = None,
        native_dim: int | None = None,
        extreme_compact: bool | None = None,
    ) -> EngineResponse:
        """Creates an isolated client index from the active route profile."""

        try:
            requested_index_id = _requested_index_id(index_id, name=name, metadata=metadata)
            safe_metadata = _validate_metadata(metadata or {})
            safe_name = _validate_index_name(name)
        except ValueError as exc:
            return self._error(400, str(exc), "create_index", context.client_id)
        if extreme_compact is not None:
            return self._error(
                400,
                "unsupported field: extreme_compact",
                "create_index",
                context.client_id,
            )
        try:
            model, provider, task, native_dim = self._index_shape_from_profile(
                model=model,
                provider=provider,
                task=task,
                native_dim=native_dim,
            )
        except ValueError as exc:
            return self._error(400, str(exc), "create_index", context.client_id)
        if native_dim <= 0:
            return self._error(
                400,
                "native_dim must be positive",
                "create_index",
                context.client_id,
            )
        if _is_blank(model) or _is_blank(provider) or _is_blank(task):
            return self._error(
                400,
                "model, provider, and task are required",
                "create_index",
                context.client_id,
            )
        if self.embedding_backend is not None and native_dim != self.embedding_backend.native_dim:
            return self._error(
                400,
                "native_dim must match configured embedding backend",
                "create_index",
                context.client_id,
            )
        if self.embedding_backend is not None and self.embedding_backend.required_model_name:
            if model != self.embedding_backend.required_model_name:
                return self._error(
                    400,
                    "model must match configured embedding backend",
                    "create_index",
                    context.client_id,
                )
        route = self.route_registry.select_route(model=model, provider=provider, task=task)
        if self.route_policy is not None:
            route_error = self.route_policy.validate_index_request(
                model=model,
                provider=provider,
                task=task,
                native_dim=native_dim,
                route=route,
            )
            if route_error is not None:
                return self._error(400, route_error, "create_index", context.client_id)
        if (
            self.route_policy is not None
            and self.route_policy.index_backend == DIRECT_TURBOVEC_STORAGE_PROFILE
        ):
            try:
                require_turbovec_available(id_map_index_class=self.turbovec_id_map_index_class)
            except RuntimeError as exc:
                return self._error(503, str(exc), "create_index", context.client_id)
        client_id_hash = sha256_text(context.client_id)
        state_key = index_state_key_for_client_hash(client_id_hash, requested_index_id)
        existing = self._indexes.get(state_key)
        if existing is not None and existing.chunks:
            return self._error(
                409,
                "index already contains documents",
                "create_index",
                context.client_id,
            )
        timestamp = _utc_now_iso(context.now)
        state = ClientIndexState(
            client_id=context.client_id,
            client_id_hash=client_id_hash,
            index_id=requested_index_id,
            index_key=state_key,
            model=model,
            provider=provider,
            task=task,
            native_dim=native_dim,
            name=safe_name,
            metadata=safe_metadata,
            created_at=(existing.created_at if existing is not None else timestamp),
            updated_at=timestamp,
            storage_profile=storage_profile_for_route_policy(self.route_policy),
            turbovec_bit_width=(
                int(self.route_policy.turbovec_bit_width)
                if self.route_policy is not None
                and self.route_policy.turbovec_bit_width is not None
                else 4
            ),
        )
        self._indexes[state_key] = state
        self._mark_index_changed(state_key)
        self._persist_state(state)
        self._emit(
            "index_created",
            context.client_id,
            {
                "index_id": state.index_id,
                "native_dim": native_dim,
                "storage_profile": state.storage_profile,
                "compact_backend": _compact_backend_name(state),
            },
        )
        return EngineResponse(201, _index_resource_payload(self, state, status="created"))

    @_synchronized
    def list_indexes(self, *, context: EngineRequestContext) -> EngineResponse:
        """Lists redacted index resources owned by the authenticated client."""

        client_id_hash = sha256_text(context.client_id)
        indexes = [
            _index_resource_payload(self, state)
            for state in sorted(
                self._indexes.values(),
                key=lambda item: (item.created_at, item.index_id),
            )
            if state.client_id_hash == client_id_hash
        ]
        return EngineResponse(200, {"status": "ok", "indexes": indexes})

    @_synchronized
    def get_index(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Returns one redacted index resource owned by the authenticated client."""

        state = self._index_for_context(context, index_id=index_id)
        if isinstance(state, EngineResponse):
            return state
        return EngineResponse(200, _index_resource_payload(self, state))

    @_synchronized
    def update_index(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
        name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EngineResponse:
        """Updates mutable redacted index metadata without touching indexed chunks."""

        state = self._index_for_context(context, index_id=index_id)
        if isinstance(state, EngineResponse):
            return state
        try:
            if name is not None:
                state.name = _validate_index_name(name)
            if metadata is not None:
                state.metadata = _validate_metadata(metadata)
        except ValueError as exc:
            return self._error(400, str(exc), "update_index", context.client_id)
        state.updated_at = _utc_now_iso(context.now)
        self._persist_state(state)
        self._emit("index_updated", context.client_id, {"index_id": state.index_id})
        return EngineResponse(200, _index_resource_payload(self, state))

    @_synchronized
    def delete_index(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Deletes one authenticated client index and any local snapshot sidecars."""

        self._require_writable()
        state = self._index_for_context(context, index_id=index_id)
        if isinstance(state, EngineResponse):
            return state
        self._indexes.pop(state.index_key, None)
        self._mark_index_changed(state.index_key)
        self._pending_tvim_deltas.pop(state.index_key, None)
        self._pending_state_journal_documents.pop(state.index_key, None)
        self._metadata_posting_indexes.pop(state.index_key, None)
        self._lexical_indexes.pop(state.index_key, None)
        self._pending_lexical_documents.pop(state.index_key, None)
        self._lexical_full_rebuild.discard(state.index_key)
        self._delete_persisted_state(state)
        self._emit(
            "index_deleted",
            context.client_id,
            {
                "index_id": state.index_id,
                "document_count": len(state.document_hashes),
                "chunk_count": len(state.chunks),
            },
        )
        return EngineResponse(
            200,
            {
                "status": "deleted",
                "index_id": state.index_id,
                "document_count": len(state.document_hashes),
                "chunk_count": len(state.chunks),
            },
        )

    @_synchronized
    def build_documents(
        self,
        *,
        context: EngineRequestContext,
        documents: tuple[EngineDocument, ...],
        index_id: str | None = None,
        embed_batch_size: int | None = None,
    ) -> EngineResponse:
        """Builds the initial cold corpus for an empty client index."""

        state = self._index_for_context(context, index_id=index_id, operation="build_documents")
        if isinstance(state, EngineResponse):
            return state
        if state.chunks or state.document_hashes:
            return self._error(
                409,
                "client index already contains documents; use upsert for incremental changes",
                "build_documents",
                context.client_id,
            )
        result = self._build_initial_documents(
            context=context,
            state=state,
            documents=documents,
            embed_batch_size=embed_batch_size,
        )
        if isinstance(result, EngineResponse):
            return result
        self._capture_document_text(state, documents)
        self._capture_lexical_tokens(state, documents)
        self._stage_wal_record(
            context, state, "upsert_documents", _documents_wal_payload(documents)
        )
        self._finalize_document_ingest(
            context=context,
            state=state,
            result=result,
            event_name="documents_built",
        )
        return EngineResponse(
            200,
            {
                "status": "built",
                **_document_ingest_response_payload(state, result),
            },
        )

    @_synchronized
    def upsert_documents(
        self,
        *,
        context: EngineRequestContext,
        documents: tuple[EngineDocument, ...],
        index_id: str | None = None,
        embed_batch_size: int | None = None,
    ) -> EngineResponse:
        """Upserts documents and embeds only chunks whose content hash changed."""

        state = self._index_for_context(context, index_id=index_id, operation="upsert_documents")
        if isinstance(state, EngineResponse):
            return state
        ingest = self._ingest_documents(
            context=context,
            state=state,
            documents=documents,
            operation="upsert_documents",
            embed_batch_size=embed_batch_size,
        )
        if isinstance(ingest, EngineResponse):
            return ingest
        result, changeset = ingest
        self._capture_document_text(state, documents)
        self._capture_lexical_tokens(state, documents)
        self._stage_wal_record(
            context, state, "upsert_documents", _documents_wal_payload(documents)
        )
        self._finalize_document_ingest(
            context=context,
            state=state,
            result=result,
            event_name="documents_upserted",
            changeset=changeset,
        )
        return EngineResponse(
            200,
            {
                "status": "upserted",
                **_document_ingest_response_payload(state, result),
            },
        )

    @_synchronized
    def upsert_vectors(
        self,
        *,
        context: EngineRequestContext,
        vectors: tuple[EngineVectorDocument, ...],
        index_id: str | None = None,
    ) -> EngineResponse:
        """Upserts pre-embedded documents (vector-in), skipping chunking/embedding.

        The caller supplies embedding vectors directly; each is stored as a
        single chunk. If a vector document carries optional ``text``, that text
        is captured only in the raw-text sidecar when enabled; it is never
        embedded, chunked, or written to redacted metadata. Vectors must match
        the index ``native_dim``. Reuses the same atomic-commit + O(changed)
        direct TurboVec sync path as :meth:`upsert_documents`, so vector-in
        inherits crash-atomic commits and incremental persistence unchanged.
        """

        state = self._index_for_context(context, index_id=index_id, operation="upsert_vectors")
        if isinstance(state, EngineResponse):
            return state
        ingest = self._ingest_vectors(
            context=context,
            state=state,
            vectors=vectors,
            operation="upsert_vectors",
        )
        if isinstance(ingest, EngineResponse):
            return ingest
        result, changeset = ingest
        self._capture_vector_document_text(state, vectors)
        self._stage_wal_record(context, state, "upsert_vectors", _vectors_wal_payload(vectors))
        self._finalize_document_ingest(
            context=context,
            state=state,
            result=result,
            event_name="vectors_upserted",
            changeset=changeset,
        )
        return EngineResponse(
            200,
            {
                "status": "upserted",
                **_document_ingest_response_payload(state, result),
            },
        )

    @_synchronized
    def update_document_payload(
        self,
        *,
        context: EngineRequestContext,
        document_id: str,
        metadata: Mapping[str, Any] | None = None,
        text: str | None = None,
        clear_text: bool = False,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Updates an existing document's metadata and/or retained raw text.

        This is intentionally narrow: it does not alter vectors or chunk
        membership, and it never embeds. It exists for adapters that maintain
        host-framework payloads beside precomputed vectors, such as mem0's
        payload-only entity-link updates.
        """

        state = self._index_for_context(
            context, index_id=index_id, operation="update_document_payload"
        )
        if isinstance(state, EngineResponse):
            return state
        document_id = str(document_id).strip()
        if not document_id:
            return self._error(
                400, "document_id is required", "update_document_payload", context.client_id
            )
        if document_id not in state.document_hashes:
            return self._error(
                404, "document not found", "update_document_payload", context.client_id
            )

        changed = False
        if metadata is not None:
            try:
                state.document_metadata[document_id] = _validate_metadata(metadata)
            except ValueError as exc:
                return self._error(400, str(exc), "update_document_payload", context.client_id)
            self._metadata_posting_indexes.pop(state.index_key, None)
            changed = True

        if text is not None or clear_text:
            if not self.raw_text_storage_enabled:
                return self._error(
                    400,
                    "raw document text storage is not enabled for this index",
                    "update_document_payload",
                    context.client_id,
                )
            if clear_text:
                state.document_text.pop(document_id, None)
            else:
                if not isinstance(text, str):
                    return self._error(
                        400,
                        "text must be a string",
                        "update_document_payload",
                        context.client_id,
                    )
                state.document_text[document_id] = text
            changed = True

        if changed:
            state.updated_at = _utc_now_iso(context.now)
            self._record_pending_state_journal_documents(
                state, upserted_document_ids=(document_id,)
            )
            # Run the incremental TurboVec sync even though no vectors changed: it is
            # also what marks this commit delta-eligible (it sets the pending TVIM
            # delta that _delta_append_ok requires), so _persist_state appends an
            # O(changed) journal delta instead of rewriting the full O(corpus) base.
            self._sync_direct_turbovec_index(state)
            self._stage_wal_record(
                context,
                state,
                "update_document_payload",
                {
                    "document_id": document_id,
                    "metadata": (
                        None
                        if metadata is None
                        else {str(k): str(v) for k, v in dict(metadata).items()}
                    ),
                    "text": (None if text is None else str(text)),
                    "clear_text": bool(clear_text),
                },
            )
            self._persist_state(state)

        return EngineResponse(
            200,
            {
                "status": "updated",
                "document_id": document_id,
                "metadata_updated": metadata is not None,
                "text_updated": text is not None or clear_text,
            },
        )

    def _build_initial_documents(
        self,
        *,
        context: EngineRequestContext,
        state: ClientIndexState,
        documents: tuple[EngineDocument, ...],
        embed_batch_size: int | None = None,
    ) -> dict[str, float | int] | EngineResponse:
        """Cold-builds an empty index with backend-sized document embedding batches."""

        if not documents:
            return self._error(
                400,
                "at least one document is required",
                "build_documents",
                context.client_id,
            )
        validation_error = self._validate_build_documents(
            context=context,
            documents=documents,
        )
        if validation_error is not None:
            return validation_error
        backend = self._embedding_backend_for_state(state)
        try:
            embedding_batch_size = _resolved_embedding_batch_size(
                backend,
                embed_batch_size=embed_batch_size,
            )
        except ValueError as exc:
            return self._error(400, str(exc), "build_documents", context.client_id)
        result: dict[str, float | int] = {
            "document_count": len(documents),
            "embedded_chunks": 0,
            "reused_chunks": 0,
            "deleted_chunks": 0,
            "embedding_time_ms": 0.0,
            "embedding_batch_size": embedding_batch_size,
        }
        pending_texts: list[str] = []
        pending_chunks: list[_PendingBuildChunk] = []

        def flush_pending() -> None:
            """Embeds pending build chunks and writes them into the client state."""

            if not pending_texts:
                return
            embedding_started = perf_counter()
            embeddings = backend.embed_documents(tuple(pending_texts))
            result["embedding_time_ms"] += (perf_counter() - embedding_started) * 1000.0
            for chunk, embedding in zip(pending_chunks, embeddings, strict=True):
                state.chunks[chunk.chunk_id] = IndexedChunk(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    content_hash=chunk.content_hash,
                    embedding=embedding,
                )
            result["embedded_chunks"] += len(pending_chunks)
            pending_texts.clear()
            pending_chunks.clear()

        for document_index, document in enumerate(documents, start=1):
            content_hash = sha256_text(document.text)
            chunks = chunk_text(document.text, self.chunk_character_limit)
            new_chunk_ids: list[str] = []
            seen_hash_counts: dict[str, int] = {}
            for chunk_body in chunks:
                chunk_hash = normalized_chunk_hash(chunk_body)
                occurrence = seen_hash_counts.get(chunk_hash, 0)
                seen_hash_counts[chunk_hash] = occurrence + 1
                chunk_id = _chunk_id_for_hash(
                    document.document_id,
                    chunk_hash=chunk_hash,
                    occurrence=occurrence,
                )
                pending_texts.append(chunk_body)
                pending_chunks.append(
                    _PendingBuildChunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        content_hash=chunk_hash,
                    )
                )
                new_chunk_ids.append(chunk_id)
                if len(pending_texts) >= embedding_batch_size:
                    flush_pending()
            state.document_hashes[document.document_id] = content_hash
            state.document_chunk_ids[document.document_id] = tuple(new_chunk_ids)
            state.document_metadata[document.document_id] = dict(document.metadata)
            if document_index != len(documents):
                _log_large_ingest_progress(
                    operation="build_documents",
                    processed_documents=document_index,
                    total_documents=len(documents),
                    embedded_chunks=int(result["embedded_chunks"]),
                    reused_chunks=0,
                )
        flush_pending()
        _log_large_ingest_progress(
            operation="build_documents",
            processed_documents=len(documents),
            total_documents=len(documents),
            embedded_chunks=int(result["embedded_chunks"]),
            reused_chunks=0,
        )
        return result

    def _validate_build_documents(
        self,
        *,
        context: EngineRequestContext,
        documents: tuple[EngineDocument, ...],
    ) -> EngineResponse | None:
        """Validates cold-build documents before mutating index state."""

        generic_error = self._validate_mutation_documents(
            context=context,
            documents=documents,
            operation="build_documents",
            unique_ids=True,
        )
        if generic_error is not None:
            return generic_error
        return None

    def _validate_mutation_documents(
        self,
        *,
        context: EngineRequestContext,
        documents: tuple[EngineDocument, ...],
        operation: str,
        unique_ids: bool,
    ) -> EngineResponse | None:
        """Validates document batch IDs, text, and metadata before any mutation."""

        seen_document_ids: set[str] = set()
        for document in documents:
            if _is_blank(document.document_id):
                return self._error(
                    400,
                    "document_id is required",
                    operation,
                    context.client_id,
                )
            if unique_ids and document.document_id in seen_document_ids:
                return self._error(
                    400,
                    "document_id must be unique for build",
                    operation,
                    context.client_id,
                )
            seen_document_ids.add(document.document_id)
            if _is_blank(document.text):
                return self._error(
                    400,
                    "document text is required",
                    operation,
                    context.client_id,
                )
            try:
                _validate_metadata(document.metadata)
            except ValueError as exc:
                return self._error(400, str(exc), operation, context.client_id)
        return None

    def _ingest_documents(
        self,
        *,
        context: EngineRequestContext,
        state: ClientIndexState,
        documents: tuple[EngineDocument, ...],
        operation: str,
        embed_batch_size: int | None = None,
    ) -> tuple[dict[str, float | int], _TurboVecChangeset] | EngineResponse:
        """Validates, chunks, embeds, and applies document changes without persisting.

        On success returns ``(result, changeset)`` where ``changeset`` is the exact
        chunk-level delta for the O(changed) direct TurboVec sync; validation
        failures still return an :class:`EngineResponse` so callers can short-circuit.
        """

        if not documents:
            return self._error(
                400,
                "at least one document is required",
                operation,
                context.client_id,
            )
        validation_error = self._validate_mutation_documents(
            context=context,
            documents=documents,
            operation=operation,
            unique_ids=False,
        )
        if validation_error is not None:
            return validation_error
        backend = self._embedding_backend_for_state(state)
        try:
            embedding_batch_size = _resolved_embedding_batch_size(
                backend,
                embed_batch_size=embed_batch_size,
            )
        except ValueError as exc:
            return self._error(400, str(exc), operation, context.client_id)
        result: dict[str, float | int] = {
            "document_count": len(documents),
            "embedded_chunks": 0,
            "reused_chunks": 0,
            "deleted_chunks": 0,
            "embedding_time_ms": 0.0,
            "embedding_batch_size": embedding_batch_size,
        }
        pending_texts: list[str] = []
        plans: list[_PlannedDocumentMutation] = []
        for document_index, document in enumerate(documents, start=1):
            content_hash = sha256_text(document.text)
            if state.document_hashes.get(document.document_id) == content_hash:
                plans.append(
                    _PlannedDocumentMutation(
                        document=document,
                        content_hash=content_hash,
                        new_chunk_ids=state.document_chunk_ids.get(document.document_id, ()),
                        planned_chunks=(),
                        unchanged=True,
                    )
                )
                continue
            chunks = chunk_text(document.text, self.chunk_character_limit)
            old_chunks = [
                state.chunks[chunk_id]
                for chunk_id in state.document_chunk_ids.get(document.document_id, ())
                if chunk_id in state.chunks
            ]
            reusable_chunks = _chunks_by_hash_occurrence(old_chunks)
            new_chunk_ids: list[str] = []
            planned_chunks: list[tuple[str, str, IndexedChunk | None, int | None]] = []
            seen_hash_counts: dict[str, int] = {}
            for chunk_body in chunks:
                chunk_hash = normalized_chunk_hash(chunk_body)
                occurrence = seen_hash_counts.get(chunk_hash, 0)
                seen_hash_counts[chunk_hash] = occurrence + 1
                chunk_id = _chunk_id_for_hash(
                    document.document_id,
                    chunk_hash=chunk_hash,
                    occurrence=occurrence,
                )
                reuse_queue = reusable_chunks.get(chunk_hash, [])
                reusable_chunk = reuse_queue.pop(0) if reuse_queue else None
                pending_index = None
                if reusable_chunk is None:
                    pending_index = len(pending_texts)
                    pending_texts.append(chunk_body)
                planned_chunks.append((chunk_id, chunk_hash, reusable_chunk, pending_index))
                new_chunk_ids.append(chunk_id)
            plans.append(
                _PlannedDocumentMutation(
                    document=document,
                    content_hash=content_hash,
                    new_chunk_ids=tuple(new_chunk_ids),
                    planned_chunks=tuple(planned_chunks),
                )
            )
            _log_large_ingest_progress(
                operation=operation,
                processed_documents=document_index,
                total_documents=len(documents),
                embedded_chunks=int(result["embedded_chunks"]),
                reused_chunks=int(result["reused_chunks"]),
            )
        embedding_started = perf_counter()
        embeddings = _embed_texts_in_batches(
            backend,
            tuple(pending_texts),
            batch_size=embedding_batch_size,
        )
        result["embedding_time_ms"] += (perf_counter() - embedding_started) * 1000.0
        # Collect the exact chunk-level delta this batch applies so the direct
        # TurboVec sync can reconcile in O(changed). Only genuinely new chunks
        # (reusable_chunk is None) are "added": reused chunks keep their
        # content-hash id, are already in the index, and would be filtered out by
        # the sync's truth check anyway.
        added_chunks: list[IndexedChunk] = []
        removed_chunk_ids: list[str] = []
        for document_index, plan in enumerate(plans, start=1):
            document = plan.document
            if plan.unchanged:
                result["reused_chunks"] += len(plan.new_chunk_ids)
                state.document_metadata[document.document_id] = dict(document.metadata)
                state.updated_at = _utc_now_iso(context.now)
                continue
            for chunk_id, chunk_hash, reusable_chunk, pending_index in plan.planned_chunks:
                if reusable_chunk is None:
                    if pending_index is None:
                        raise RuntimeError("pending embedding index is missing")
                    embedding = embeddings[pending_index]
                    result["embedded_chunks"] += 1
                else:
                    # Reused chunks keep the same content-hash chunk id, so the
                    # direct sync never re-adds them; ready states have already
                    # discarded their transient embeddings.
                    embedding = reusable_chunk.embedding
                    result["reused_chunks"] += 1
                new_chunk = IndexedChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    content_hash=chunk_hash,
                    embedding=embedding,
                )
                state.chunks[chunk_id] = new_chunk
                if reusable_chunk is None:
                    added_chunks.append(new_chunk)
            orphan_ids = _delete_orphaned_document_chunks(
                state,
                document.document_id,
                list(plan.new_chunk_ids),
            )
            result["deleted_chunks"] += len(orphan_ids)
            removed_chunk_ids.extend(orphan_ids)
            state.document_hashes[document.document_id] = plan.content_hash
            state.document_chunk_ids[document.document_id] = tuple(plan.new_chunk_ids)
            state.document_metadata[document.document_id] = dict(document.metadata)
            state.updated_at = _utc_now_iso(context.now)
            _log_large_ingest_progress(
                operation=operation,
                processed_documents=document_index,
                total_documents=len(documents),
                embedded_chunks=int(result["embedded_chunks"]),
                reused_chunks=int(result["reused_chunks"]),
            )
        # Unchanged documents are tracked too: metadata-only updates must
        # reach the direct-route JSON journal even without chunk changes.
        self._record_pending_state_journal_documents(
            state,
            upserted_document_ids=tuple(plan.document.document_id for plan in plans),
        )
        changeset = _TurboVecChangeset(
            added_chunks=tuple(added_chunks),
            removed_chunk_ids=tuple(removed_chunk_ids),
        )
        return result, changeset

    def _ingest_vectors(
        self,
        *,
        context: EngineRequestContext,
        state: ClientIndexState,
        vectors: tuple[EngineVectorDocument, ...],
        operation: str,
    ) -> tuple[dict[str, float | int], _TurboVecChangeset] | EngineResponse:
        """Validates and applies pre-embedded documents without embedding/chunking.

        Mirrors the apply half of :meth:`_ingest_documents` but treats each
        supplied vector as a single, already-embedded chunk: it validates the
        whole batch up front (so a bad vector fails atomically with nothing
        applied), writes one :class:`IndexedChunk` per document carrying the
        caller's vector as its transient embedding, and journals the mutation.
        The shared :meth:`_finalize_document_ingest` then runs the same direct
        TurboVec sync + atomic commit the text path uses. On success returns
        ``(result, changeset)`` for the O(changed) sync.
        """

        if not vectors:
            return self._error(
                400, "at least one vector document is required", operation, context.client_id
            )
        native_dim = state.native_dim
        # Validate + coerce the whole batch first; only then mutate state, so a
        # rejected vector never leaves the index half-updated.
        planned: list[tuple[str, str, tuple[float, ...], dict[str, str]]] = []
        for vector_doc in vectors:
            document_id = str(vector_doc.document_id).strip()
            if not document_id:
                return self._error(
                    400, "vector document id must be non-blank", operation, context.client_id
                )
            array = np.asarray(vector_doc.vector, dtype=np.float32)
            if array.ndim != 1 or array.shape[0] != native_dim:
                return self._error(
                    400,
                    f"vector dimension {tuple(array.shape)} does not match index "
                    f"native_dim {native_dim}",
                    operation,
                    context.client_id,
                )
            if not bool(np.all(np.isfinite(array))):
                return self._error(
                    400, "vector must contain only finite values", operation, context.client_id
                )
            try:
                metadata = _validate_metadata(vector_doc.metadata)
            except ValueError as exc:
                return self._error(400, str(exc), operation, context.client_id)
            content_hash = _vector_content_hash(array)
            planned.append((document_id, content_hash, tuple(array.tolist()), metadata))

        result: dict[str, float | int] = {
            "document_count": len(vectors),
            "embedded_chunks": 0,
            "reused_chunks": 0,
            "deleted_chunks": 0,
            "embedding_time_ms": 0.0,
            "embedding_batch_size": 0,
        }
        upserted_ids: list[str] = []
        added_chunks: list[IndexedChunk] = []
        removed_chunk_ids: list[str] = []
        for document_id, content_hash, embedding, metadata in planned:
            upserted_ids.append(document_id)
            if state.document_hashes.get(document_id) == content_hash:
                # Identical vector already stored: a metadata-only refresh, no
                # re-encode (parallels the unchanged-document branch of the text path).
                result["reused_chunks"] += 1
                state.document_metadata[document_id] = dict(metadata)
                state.updated_at = _utc_now_iso(context.now)
                continue
            chunk_id = _chunk_id_for_hash(document_id, chunk_hash=content_hash, occurrence=0)
            new_chunk = IndexedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                content_hash=content_hash,
                embedding=embedding,
            )
            state.chunks[chunk_id] = new_chunk
            added_chunks.append(new_chunk)
            result["embedded_chunks"] += 1
            orphan_ids = _delete_orphaned_document_chunks(state, document_id, [chunk_id])
            result["deleted_chunks"] += len(orphan_ids)
            removed_chunk_ids.extend(orphan_ids)
            state.document_hashes[document_id] = content_hash
            state.document_chunk_ids[document_id] = (chunk_id,)
            state.document_metadata[document_id] = dict(metadata)
            state.updated_at = _utc_now_iso(context.now)
        self._record_pending_state_journal_documents(
            state, upserted_document_ids=tuple(upserted_ids)
        )
        changeset = _TurboVecChangeset(
            added_chunks=tuple(added_chunks),
            removed_chunk_ids=tuple(removed_chunk_ids),
        )
        return result, changeset

    def _finalize_document_ingest(
        self,
        *,
        context: EngineRequestContext,
        state: ClientIndexState,
        result: dict[str, float | int],
        event_name: str,
        changeset: _TurboVecChangeset | None = None,
    ) -> None:
        """Updates counters, syncs the direct index, and persists once.

        ``changeset`` carries the exact chunk-level delta from the upsert verbs so
        the TurboVec sync runs in O(changed); when it is ``None`` (e.g. the cold
        build path) the sync falls back to the full corpus diff.
        """

        embedded = int(result["embedded_chunks"])
        reused = int(result["reused_chunks"])
        deleted = int(result["deleted_chunks"])
        state.embedded_chunk_count += embedded
        state.cache_reuse_count += reused
        state.deleted_chunk_count += deleted
        if _state_uses_direct_turbovec(state):
            _log_document_ingest_finalize_progress(
                event_name,
                phase="turbovec_sync",
                event="started",
                document_count=int(result["document_count"]),
                chunk_count=len(state.chunks),
            )
            sync_started = perf_counter()
            synced_chunk_ids = self._sync_direct_turbovec_index(state, changeset=changeset)
            _log_document_ingest_finalize_progress(
                event_name,
                phase="turbovec_sync",
                event="completed",
                document_count=int(result["document_count"]),
                chunk_count=len(state.chunks),
                elapsed_ms=(perf_counter() - sync_started) * 1000.0,
            )
            _discard_direct_turbovec_transient_embeddings(state, synced_chunk_ids)
            self._emit(
                event_name,
                context.client_id,
                {
                    "document_count": int(result["document_count"]),
                    "embedded_chunks": embedded,
                    "reused_chunks": reused,
                    "deleted_chunks": deleted,
                    "embedding_time_ms": float(result["embedding_time_ms"]),
                    **self._turbovec_drift_fields(state),
                },
            )
            state.updated_at = _utc_now_iso(context.now)
            _log_document_ingest_finalize_progress(
                event_name,
                phase="snapshot_persist",
                event="started",
                document_count=int(result["document_count"]),
                chunk_count=len(state.chunks),
            )
            snapshot_started = perf_counter()
            self._persist_state(state)
            _log_document_ingest_finalize_progress(
                event_name,
                phase="snapshot_persist",
                event="completed",
                document_count=int(result["document_count"]),
                chunk_count=len(state.chunks),
                elapsed_ms=(perf_counter() - snapshot_started) * 1000.0,
            )
            return

    @_synchronized
    def delete_documents(
        self,
        *,
        context: EngineRequestContext,
        document_ids: tuple[str, ...],
        index_id: str | None = None,
    ) -> EngineResponse:
        """Deletes documents from only the authenticated client's index."""

        state = self._index_for_context(context, index_id=index_id, operation="delete_documents")
        if isinstance(state, EngineResponse):
            return state
        if not document_ids:
            return self._error(
                400,
                "at least one document_id is required",
                "delete_documents",
                context.client_id,
            )
        deleted_chunks = 0
        removed_chunk_ids: list[str] = []
        unique_document_ids = tuple(dict.fromkeys(document_ids))
        for document_id in document_ids:
            if _is_blank(document_id):
                return self._error(
                    400,
                    "document_id is required",
                    "delete_documents",
                    context.client_id,
                )
        for document_id in unique_document_ids:
            removed = self._delete_document_chunks(state, document_id)
            deleted_chunks += len(removed)
            removed_chunk_ids.extend(removed)
            state.document_hashes.pop(document_id, None)
            state.document_chunk_ids.pop(document_id, None)
            state.document_metadata.pop(document_id, None)
            state.document_text.pop(document_id, None)
            state.document_tokens.pop(document_id, None)
        self._record_pending_state_journal_documents(
            state,
            deleted_document_ids=unique_document_ids,
        )
        state.delete_count += len(unique_document_ids)
        state.deleted_chunk_count += deleted_chunks
        state.updated_at = _utc_now_iso(context.now)
        self._sync_direct_turbovec_index(
            state,
            changeset=_TurboVecChangeset(removed_chunk_ids=tuple(removed_chunk_ids)),
        )
        self._emit(
            "documents_deleted",
            context.client_id,
            {
                "document_count": len(unique_document_ids),
                "deleted_chunks": deleted_chunks,
            },
        )
        self._stage_wal_record(
            context, state, "delete_documents", {"document_ids": list(unique_document_ids)}
        )
        self._persist_state(state)
        return EngineResponse(
            200,
            {
                "status": "deleted",
                "document_count": len(unique_document_ids),
                "deleted_chunks": deleted_chunks,
            },
        )

    @_synchronized
    def query(
        self,
        *,
        context: EngineRequestContext,
        query: EngineQuery,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Returns top chunk IDs and scores without exposing raw query or chunk text."""

        state = self._index_for_context(context, index_id=index_id, operation="query")
        if isinstance(state, EngineResponse):
            return state
        if query.top_k <= 0:
            return self._error(400, "top_k must be positive", "query", context.client_id)
        if query.embedding is None and _is_blank(query.text):
            return self._error(400, "query text is required", "query", context.client_id)
        try:
            query_filter = _validate_query_filter(query.filter)
            includes = _validate_query_includes(query.include)
            mode = _validate_query_mode(query.mode)
        except ValueError as exc:
            return self._error(400, str(exc), "query", context.client_id)
        lexical_unavailable = self._require_lexical_capable(mode)
        if lexical_unavailable is not None:
            return self._error(400, lexical_unavailable, "query", context.client_id)

        started_at = perf_counter()
        route = self.route_registry.select_route(
            model=state.model,
            provider=state.provider,
            task=state.task,
            drifted=query.route_drifted,
            failed=query.route_failed,
            high_risk=query.high_risk,
        )
        if query.embedding is not None:
            # Vector-in search: the caller supplied the query embedding, so skip
            # the embedder entirely and validate the dimension here for a clean
            # 400 (rather than a deep RuntimeError -> 503 from the scan kernel).
            if len(query.embedding) != state.native_dim:
                return self._error(
                    400,
                    f"query vector dimension {len(query.embedding)} does not match index "
                    f"native_dim {state.native_dim}",
                    "query",
                    context.client_id,
                )
            query_embedding = tuple(query.embedding)
            query_embedding_latency_ms = 0.0
        else:
            embedding_started = perf_counter()
            query_embedding = self._embedding_backend_for_state(state).embed_query(query.text)
            query_embedding_latency_ms = (perf_counter() - embedding_started) * 1000.0
        try:
            search_started = perf_counter()
            if mode == RETRIEVAL_MODE_VECTOR:
                query_result = self._query_serving_index(
                    state,
                    route=route,
                    query_embedding=query_embedding,
                    query_filter=query_filter,
                    top_k=query.top_k,
                )
            else:
                query_result = self._query_serving_index_hybrid(
                    state,
                    query_embedding=query_embedding,
                    query_text=query.text,
                    query_filter=query_filter,
                    top_k=query.top_k,
                    mode=mode,
                )
            query_search_latency_ms = (perf_counter() - search_started) * 1000.0
        except RuntimeError as exc:
            return self._error(503, str(exc), "query", context.client_id)
        fallback_used = query_result.fallback_used
        state.query_count += 1
        if fallback_used:
            state.fallback_count += 1
            state.fallback_reasons[route.reason] = state.fallback_reasons.get(route.reason, 0) + 1
        results = _materialize_query_results(
            query_result,
            state=state,
            query_filter=query_filter,
            include=includes,
            top_k=query.top_k,
        )
        latency_ms = (perf_counter() - started_at) * 1000.0
        state.query_latency_ms.append(latency_ms)
        self._emit(
            "query_completed",
            context.client_id,
            {
                "top_k": query.top_k,
                "result_count": len(results),
                "fallback_used": fallback_used,
                "retrieval_mode": query_result.retrieval_mode,
                "latency_ms": latency_ms,
                "query_embedding_latency_ms": query_embedding_latency_ms,
                "query_search_latency_ms": query_search_latency_ms,
                "native_query_used": query_result.native_used,
                "compact_backend": _compact_backend_name(state),
                "native_backend": query_result.native_backend,
                "stage_one_backend": query_result.stage_one_backend,
                "compact_route_fallback": query_result.compact_route_fallback,
                "gpu_stage_one_status": query_result.gpu_stage_one_status,
                "gpu_estimated_bytes": query_result.gpu_estimated_bytes,
                "gpu_budget_bytes": query_result.gpu_budget_bytes,
                "gpu_fallback_reason": query_result.gpu_fallback_reason,
            },
        )
        return EngineResponse(
            200,
            {
                "status": "ok",
                "route_profile": _client_route_profile(self.security, self.route_policy),
                "fallback_used": fallback_used,
                "latency_ms": latency_ms,
                "query_embedding_latency_ms": query_embedding_latency_ms,
                "query_search_latency_ms": query_search_latency_ms,
                "results": results,
            },
        )

    @_synchronized
    def query_batch(
        self,
        *,
        context: EngineRequestContext,
        queries: tuple[EngineQuery, ...],
        index_id: str | None = None,
    ) -> EngineResponse:
        """Executes a validated query batch while preserving per-query result order."""

        state = self._index_for_context(context, index_id=index_id, operation="query_batch")
        if isinstance(state, EngineResponse):
            return state
        if not queries:
            return self._error(
                400,
                "queries must be a nonempty list",
                "query_batch",
                context.client_id,
            )
        prepared: list[_PreparedQuery] = []
        for item in queries:
            if item.top_k <= 0:
                return self._error(400, "top_k must be positive", "query_batch", context.client_id)
            if item.embedding is None and _is_blank(item.text):
                return self._error(400, "query text is required", "query_batch", context.client_id)
            if item.embedding is not None and len(item.embedding) != state.native_dim:
                return self._error(
                    400,
                    f"query vector dimension {len(item.embedding)} does not match index "
                    f"native_dim {state.native_dim}",
                    "query_batch",
                    context.client_id,
                )
            try:
                query_filter = _validate_query_filter(item.filter)
                includes = _validate_query_includes(item.include)
                mode = _validate_query_mode(item.mode)
            except ValueError as exc:
                return self._error(400, str(exc), "query_batch", context.client_id)
            lexical_unavailable = self._require_lexical_capable(mode)
            if lexical_unavailable is not None:
                return self._error(400, lexical_unavailable, "query_batch", context.client_id)
            route = self.route_registry.select_route(
                model=state.model,
                provider=state.provider,
                task=state.task,
                drifted=item.route_drifted,
                failed=item.route_failed,
                high_risk=item.high_risk,
            )
            prepared.append(
                _PreparedQuery(
                    query=item,
                    query_filter=query_filter,
                    includes=includes,
                    route=route,
                    mode=mode,
                )
            )

        started_at = perf_counter()
        backend = self._embedding_backend_for_state(state)
        query_results: list[_EngineTurboVecQueryResult | None]
        query_results = [None] * len(prepared)
        embedding_latencies: list[float] = []
        search_latencies: list[float] = [0.0 for _item in prepared]
        query_embeddings: list[tuple[float, ...]] = []
        for prepared_item in prepared:
            item_query = prepared_item.query
            if item_query.embedding is not None:
                # Vector-in: dimension already validated above; use it directly.
                query_embeddings.append(tuple(item_query.embedding))
                embedding_latencies.append(0.0)
                continue
            embedding_started = perf_counter()
            query_embeddings.append(backend.embed_query(item_query.text))
            embedding_latencies.append((perf_counter() - embedding_started) * 1000.0)
        try:
            self._execute_prepared_query_batch(
                state=state,
                prepared=tuple(prepared),
                query_embeddings=tuple(query_embeddings),
                query_results=query_results,
                search_latencies=search_latencies,
            )
        except RuntimeError as exc:
            return self._error(503, str(exc), "query_batch", context.client_id)
        response_items: list[dict[str, Any]] = []
        batch_result_counts: list[int] = []
        for index, (prepared_item, query_result) in enumerate(
            zip(prepared, query_results, strict=True)
        ):
            if query_result is None:
                raise RuntimeError("query batch execution left a missing result")
            results = _materialize_query_results(
                query_result,
                state=state,
                query_filter=prepared_item.query_filter,
                include=prepared_item.includes,
                top_k=prepared_item.query.top_k,
            )
            latency_ms = embedding_latencies[index] + search_latencies[index]
            state.query_latency_ms.append(latency_ms)
            state.query_count += 1
            if query_result.fallback_used:
                state.fallback_count += 1
                reason = prepared_item.route.reason
                state.fallback_reasons[reason] = state.fallback_reasons.get(reason, 0) + 1
            batch_result_counts.append(len(results))
            response_items.append(
                {
                    "status": "ok",
                    "fallback_used": query_result.fallback_used,
                    "latency_ms": latency_ms,
                    "query_embedding_latency_ms": embedding_latencies[index],
                    "query_search_latency_ms": search_latencies[index],
                    "results": results,
                }
            )
        batch_latency_ms = (perf_counter() - started_at) * 1000.0
        self._emit(
            "query_batch_completed",
            context.client_id,
            {
                "query_count": len(prepared),
                "batch_query_count": len(prepared),
                "latency_ms": batch_latency_ms,
                "fallback_used": any(item["fallback_used"] for item in response_items),
                "result_count": sum(batch_result_counts),
                "compact_backend": _compact_backend_name(state),
                "native_query_used": any(
                    result.native_used for result in query_results if result is not None
                ),
                "native_backend": _combined_backend(query_results, field_name="native_backend"),
                "stage_one_backend": _combined_backend(
                    query_results,
                    field_name="stage_one_backend",
                ),
                "gpu_stage_one_status": _combined_backend(
                    query_results,
                    field_name="gpu_stage_one_status",
                ),
                "gpu_estimated_bytes": _max_query_result_int(
                    query_results,
                    field_name="gpu_estimated_bytes",
                ),
                "gpu_budget_bytes": _max_query_result_int(
                    query_results,
                    field_name="gpu_budget_bytes",
                ),
                "gpu_copy_back_bytes": _sum_query_result_int(
                    query_results,
                    field_name="gpu_copy_back_bytes",
                ),
                "gpu_fallback_reason": _combined_backend(
                    query_results,
                    field_name="gpu_fallback_reason",
                ),
                "gpu_query_count": _max_query_result_int(
                    query_results,
                    field_name="gpu_query_count",
                ),
                "gpu_resident_upload_build_ms": _max_query_result_float(
                    query_results,
                    field_name="gpu_resident_upload_build_ms",
                ),
                "gpu_stage_one_search_ms": _max_query_result_float(
                    query_results,
                    field_name="gpu_stage_one_search_ms",
                ),
                "gpu_device_to_host_copy_ms": _max_query_result_float(
                    query_results,
                    field_name="gpu_device_to_host_copy_ms",
                ),
                "gpu_stage_one_tile_count": _max_query_result_int(
                    query_results,
                    field_name="gpu_stage_one_tile_count",
                ),
            },
        )
        return EngineResponse(
            200,
            {
                "status": "ok",
                "route_profile": _client_route_profile(self.security, self.route_policy),
                "query_count": len(prepared),
                "latency_ms": batch_latency_ms,
                "queries": response_items,
            },
        )

    @_synchronized
    def list_documents(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
        filter: dict[str, Any] | None = None,
        after: str | None = None,
        limit: int | None = None,
    ) -> EngineResponse:
        """Lists redacted document records for one authenticated client index.

        With ``filter`` (a validated metadata / ``document_ids`` filter), only the
        matching documents are materialized, resolved through the per-field
        planner in O(matches) rather than scanning the corpus. ``after``/``limit``
        provide a stable-id keyset cursor for streaming large result sets.
        """

        state = self._index_for_context(context, index_id=index_id, operation="list_documents")
        if isinstance(state, EngineResponse):
            return state
        if filter is None:
            document_ids = sorted(state.document_hashes)
        else:
            try:
                query_filter = _validate_query_filter(filter)
            except ValueError as exc:
                return self._error(400, str(exc), "list_documents", context.client_id)
            index = self._metadata_posting_index(state)
            document_ids = sorted(index.document_allowlist(query_filter))
        if after is not None:
            cursor = str(after)
            document_ids = [document_id for document_id in document_ids if document_id > cursor]
        if limit is not None:
            document_ids = document_ids[: max(0, int(limit))]
        documents = [_document_resource_payload(state, document_id) for document_id in document_ids]
        return EngineResponse(200, {"status": "ok", "documents": documents})

    @_synchronized
    def count_documents(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> EngineResponse:
        """Returns the document count, optionally for a filter, materializing nothing.

        ``count_documents(filter=...)`` resolves the matching document set through
        the planner and returns its size, without building any record payload.
        """

        state = self._index_for_context(context, index_id=index_id, operation="count_documents")
        if isinstance(state, EngineResponse):
            return state
        if filter is None:
            return EngineResponse(200, {"status": "ok", "count": len(state.document_hashes)})
        try:
            query_filter = _validate_query_filter(filter)
        except ValueError as exc:
            return self._error(400, str(exc), "count_documents", context.client_id)
        count = len(self._metadata_posting_index(state).document_allowlist(query_filter))
        return EngineResponse(200, {"status": "ok", "count": count})

    @_synchronized
    def get_document(
        self,
        *,
        context: EngineRequestContext,
        document_id: str,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Returns one redacted document record from an authenticated client index."""

        state = self._index_for_context(context, index_id=index_id, operation="get_document")
        if isinstance(state, EngineResponse):
            return state
        if _is_blank(document_id):
            return self._error(400, "document_id is required", "get_document", context.client_id)
        if document_id not in state.document_hashes:
            return self._error(404, "document not found", "get_document", context.client_id)
        return EngineResponse(200, _document_resource_payload(state, document_id))

    @_synchronized
    def get_document_text(
        self,
        *,
        context: EngineRequestContext,
        document_id: str,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Returns one stored document's raw text (opt-in raw-text storage only).

        Requires the engine to run with raw-text storage enabled
        (``EngineSecurityConfig.allow_raw_result_text``); otherwise it reports a
        clear error rather than implying text was retained. The retrieved text
        is returned only to the local caller and is never logged, journaled, or
        added to telemetry/audit output.
        """

        state = self._index_for_context(context, index_id=index_id, operation="get_document_text")
        if isinstance(state, EngineResponse):
            return state
        if _is_blank(document_id):
            return self._error(
                400, "document_id is required", "get_document_text", context.client_id
            )
        if not self.raw_text_storage_enabled:
            return self._error(
                400,
                "raw document text storage is not enabled for this index",
                "get_document_text",
                context.client_id,
            )
        if document_id not in state.document_hashes:
            return self._error(404, "document not found", "get_document_text", context.client_id)
        if document_id not in state.document_text:
            return self._error(
                404, "document text not stored", "get_document_text", context.client_id
            )
        self._emit("document_text_read", context.client_id, {"document_count": 1})
        return EngineResponse(
            200,
            {
                "status": "ok",
                "document_id": document_id,
                "text": state.document_text[document_id],
            },
        )

    @_synchronized
    def get_document_texts(
        self,
        *,
        context: EngineRequestContext,
        document_ids: tuple[str, ...],
        index_id: str | None = None,
    ) -> EngineResponse:
        """Returns stored raw text for several document ids (opt-in storage only).

        Unknown or not-stored ids are simply omitted from the returned map, so a
        batch retrieval never fails because one id is missing. As with the
        single-id path, this requires raw-text storage to be enabled and only
        ever returns text to the local caller.
        """

        state = self._index_for_context(context, index_id=index_id, operation="get_document_texts")
        if isinstance(state, EngineResponse):
            return state
        if not self.raw_text_storage_enabled:
            return self._error(
                400,
                "raw document text storage is not enabled for this index",
                "get_document_texts",
                context.client_id,
            )
        texts: dict[str, str] = {}
        for document_id in document_ids:
            if _is_blank(document_id):
                return self._error(
                    400, "document_id is required", "get_document_texts", context.client_id
                )
            if document_id in state.document_text:
                texts[document_id] = state.document_text[document_id]
        self._emit("document_text_read", context.client_id, {"document_count": len(texts)})
        return EngineResponse(200, {"status": "ok", "documents": texts})

    @_synchronized
    def stats(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Returns per-client index, recompute, and storage-pressure counters."""

        state = self._index_for_context(context, index_id=index_id, operation="stats")
        if isinstance(state, EngineResponse):
            return state
        route = self.route_registry.select_route(
            model=state.model,
            provider=state.provider,
            task=state.task,
        )
        recompute_fraction = _state_recompute_fraction(state)
        return EngineResponse(
            200,
            {
                "client_id": context.client_id,
                "index_id": state.index_id,
                "route_profile": _client_route_profile(self.security, self.route_policy),
                "document_count": len(state.document_hashes),
                "chunk_count": len(state.chunks),
                "embedded_chunk_count": state.embedded_chunk_count,
                "cache_reuse_count": state.cache_reuse_count,
                "delete_count": state.delete_count,
                "deleted_chunk_count": state.deleted_chunk_count,
                "embedded_chunks": state.embedded_chunk_count,
                "reused_chunks": state.cache_reuse_count,
                "deleted_chunks": state.deleted_chunk_count,
                "recompute_fraction": recompute_fraction,
                "query_latency": _latency_payload(state.query_latency_ms),
                "storage": self._storage_payload_for_state(state, route=route),
                "storage_profile": state.storage_profile,
                "compact_backend": _compact_backend_name(state),
                "native_backend": _native_backend_for_state(self, state),
                "native_used": _native_used_for_state(self, state),
                "raw_payload_text_present": False,
            },
        )

    @_synchronized
    def audit(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Returns metrics-only recompute and fallback audit fields for one client index."""

        state = self._index_for_context(context, index_id=index_id, operation="audit")
        if isinstance(state, EngineResponse):
            return state
        recompute_fraction = _state_recompute_fraction(state)
        fallback_rate = (
            0.0 if state.query_count == 0 else state.fallback_count / float(state.query_count)
        )
        return EngineResponse(
            200,
            {
                "client_id": context.client_id,
                "index_id": state.index_id,
                "status": "ok",
                "route_profile": _client_route_profile(self.security, self.route_policy),
                "document_count": len(state.document_hashes),
                "chunk_count": len(state.chunks),
                "embedded_chunk_count": state.embedded_chunk_count,
                "cache_reuse_count": state.cache_reuse_count,
                "delete_count": state.delete_count,
                "deleted_chunk_count": state.deleted_chunk_count,
                "embedded_chunks": state.embedded_chunk_count,
                "reused_chunks": state.cache_reuse_count,
                "deleted_chunks": state.deleted_chunk_count,
                "query_count": state.query_count,
                "fallback_count": state.fallback_count,
                "fallback_rate": fallback_rate,
                "fallback_reasons": dict(sorted(state.fallback_reasons.items())),
                "recompute_fraction": recompute_fraction,
                "query_latency": _latency_payload(state.query_latency_ms),
                "storage_profile": state.storage_profile,
                "compact_backend": _compact_backend_name(state),
                "native_backend": _native_backend_for_state(self, state),
                "native_used": _native_used_for_state(self, state),
                "raw_payload_text_present": False,
            },
        )

    def _index_for_context(
        self,
        context: EngineRequestContext,
        *,
        index_id: str | None = None,
        operation: str = "index",
    ) -> ClientIndexState | EngineResponse:
        """Returns the authenticated client's requested index or a redacted 404."""

        try:
            requested_index_id = normalize_index_id(index_id)
        except ValueError as exc:
            return self._error(400, str(exc), operation, context.client_id)
        state_key = index_state_key(context.client_id, requested_index_id)
        state = self._indexes.get(state_key)
        if state is None:
            return self._error(404, "client index not found", operation, context.client_id)
        return state

    def _storage_payload_for_state(
        self,
        state: ClientIndexState,
        *,
        route: RouteDecision,
    ) -> dict[str, Any]:
        """Returns theoretical and actual persisted storage accounting for one index."""

        payload = _physical_storage_payload(
            state,
            route=route,
            route_profile=_client_route_profile(self.security, self.route_policy),
        )
        payload.update(_persisted_storage_bytes(self, state))
        return payload

    def _delete_persisted_state(self, state: ClientIndexState) -> None:
        """Deletes JSON and native sidecar snapshots for one index when present."""

        if self.persistence_dir is None:
            return
        key = state.index_key
        # Stage-2 artifacts: the root commit-manifest pointer and the per-index
        # generation directory holding every base epoch + its journals.
        self._commit_manifest_path(key).unlink(missing_ok=True)
        shutil.rmtree(generation_dir(self.persistence_dir, key), ignore_errors=True)
        self._base_epochs.pop(key, None)
        # Legacy (pre-commit-manifest) top-level base files, journals, and the
        # legacy raw-text sidecar. The generation-addressed text sidecars are
        # inside the per-index directory removed above.
        self._gc_legacy_files(key)

    def _index_shape_from_profile(
        self,
        *,
        model: str | None,
        provider: str | None,
        task: str | None,
        native_dim: int | None,
    ) -> tuple[str, str, str, int]:
        """Returns the internal index shape, deriving it from the active route profile."""

        if self.route_policy is None:
            if model is None or provider is None or task is None or native_dim is None:
                raise ValueError("model, provider, task, and native_dim are required")
            return model, provider, task, native_dim
        if any(value is not None for value in (model, provider, task, native_dim)):
            raise ValueError("index shape is derived from route profile")
        return (
            self.route_policy.model,
            self.route_policy.provider,
            self.route_policy.task,
            self.route_policy.native_dim,
        )

    def _delete_document_chunks(
        self, state: ClientIndexState, document_id: str
    ) -> tuple[str, ...]:
        """Removes all chunks for one document from a client-local index.

        Returns the removed chunk ids so ``delete_documents`` can build the
        direct TurboVec changeset; the deleted-chunk count is ``len(...)``.
        """

        chunk_ids = state.document_chunk_ids.get(document_id, ())
        for chunk_id in chunk_ids:
            state.chunks.pop(chunk_id, None)
        return tuple(chunk_ids)

    def _embedding_backend_for_state(self, state: ClientIndexState) -> EngineEmbeddingBackend:
        """Returns the configured local backend or a deterministic fixture backend."""

        if self.embedding_backend is not None:
            if self.embedding_backend.native_dim != state.native_dim:
                raise ValueError("configured embedding backend dimension does not match index")
            return self.embedding_backend
        return HashEmbeddingBackend(native_dim=state.native_dim)

    def _query_serving_index(
        self,
        state: ClientIndexState,
        *,
        route: RouteDecision,
        query_embedding: tuple[float, ...],
        query_filter: Mapping[str, Any],
        top_k: int,
    ) -> _EngineTurboVecQueryResult:
        """Searches the direct TurboVec backend for one prepared query."""

        del route
        return self._query_direct_turbovec_index(
            state,
            query_embedding=query_embedding,
            top_k=top_k,
            query_filter=query_filter,
        )

    def _query_direct_turbovec_index(
        self,
        state: ClientIndexState,
        *,
        query_embedding: tuple[float, ...],
        top_k: int,
        query_filter: Mapping[str, Any],
    ) -> _EngineTurboVecQueryResult:
        """Scores one query directly against full-dimensional TurboVec storage."""

        index = self._turbovec_index_for_state(state)
        if query_filter:
            # Push the document/metadata filter into the native allowlist so
            # filtered queries rank only eligible rows instead of widening
            # top_k to the full corpus and sorting everything. The batch path
            # (`_run_direct_batch_group`) shares this same allowlist strategy.
            allowed_chunk_ids = self._build_filter_allowlist(state, query_filter)
            if not allowed_chunk_ids:
                return _empty_turbovec_result(index)
            result = index.search(
                query_embedding,
                top_k=top_k,
                allowlist_chunk_ids=allowed_chunk_ids,
            )
        else:
            result = index.search(query_embedding, top_k=top_k)
        # The search above (re)built the SIMD layout, so any deferred
        # quantization-drift rows can be sampled cheaply now.
        self._sample_pending_drift(state)
        return _EngineTurboVecQueryResult(
            index=index,
            stable_ids=result.stable_ids.reshape(-1),
            scores=result.scores.reshape(-1),
            native_used=result.native_used,
            native_backend=result.native_backend,
            retrieval_mode="direct_turbovec",
            fallback_used=False,
            compact_route_fallback=False,
        )

    def _try_query_gpu_direct_turbovec_batch(
        self,
        state: ClientIndexState,
        *,
        serving: TurboVecServingIndex,
        batch: np.ndarray,
        top_k: int,
        allowlist_stable_ids: np.ndarray | None = None,
    ) -> tuple[Any | None, dict[str, Any]]:
        """Attempts the GPU-resident exact batch path for one direct-route batch.

        Thin CUDA adapter: probes CuPy, builds a :class:`_ResidentDirectBackend`,
        and delegates to :meth:`_try_query_resident_direct_batch` (which documents
        the ``(result, telemetry_fields)`` contract and the ``off``/``auto``/
        ``required`` semantics). ``off`` short-circuits here so the CuPy probe never
        runs on the default-on path when an operator has explicitly disabled it.
        """

        from lodedb.engine.gpu_turbovec import (
            GPU_DIRECT_TURBOVEC_MAX_TOP_K,
            GpuDirectTurboVecSession,
            gpu_direct_turbovec_dependencies,
            turbovec_reconstruction_api_available,
        )

        policy = self.gpu_direct_turbovec_policy
        if policy == GpuDirectTurboVecPolicy.OFF:
            return None, {}
        # Probe CuPy only once policy is on (the dependency import is not free).
        dependencies = gpu_direct_turbovec_dependencies()
        backend = _ResidentDirectBackend(
            kind="GPU",
            reason_prefix="gpu_direct",
            policy=policy,
            dependency_available=bool(dependencies.available),
            api_available=turbovec_reconstruction_api_available(serving.index),
            unavailable_reason=dependencies.unavailable_reason or "",
            default_unavailable_reason="gpu_dependencies_unavailable",
            should_use=gpu_direct_turbovec_should_use,
            max_batch=self.gpu_direct_turbovec_max_batch,
            max_top_k=GPU_DIRECT_TURBOVEC_MAX_TOP_K,
            sessions=self._gpu_direct_turbovec_sessions,
            memory_budget_bytes=self.gpu_memory_budget_bytes,
            build_session=lambda: GpuDirectTurboVecSession.build(
                index=serving.index,
                generation=serving.generation,
                dependencies=dependencies,
                memory_budget_bytes=self.gpu_memory_budget_bytes,
            ),
            estimated_bytes=lambda session: int(session.estimated_gpu_bytes),
        )
        return self._try_query_resident_direct_batch(
            backend,
            state,
            serving=serving,
            batch=batch,
            top_k=top_k,
            allowlist_stable_ids=allowlist_stable_ids,
        )

    def _try_query_resident_direct_batch(
        self,
        backend: _ResidentDirectBackend,
        state: ClientIndexState,
        *,
        serving: TurboVecServingIndex,
        batch: np.ndarray,
        top_k: int,
        allowlist_stable_ids: np.ndarray | None = None,
    ) -> tuple[Any | None, dict[str, Any]]:
        """Backend-agnostic engine for one resident (GPU/MPS) direct-route batch.

        Shared by :meth:`_try_query_gpu_direct_turbovec_batch` and
        :meth:`_try_query_mps_direct_turbovec_batch`, which pass a
        :class:`_ResidentDirectBackend` describing the device specifics. Returns
        ``(result, telemetry_fields)``: on success the fields carry admission
        accounting for the "used" rows; on fallback the result is ``None`` and the
        fields annotate the CPU rows with a visible status/reason. ``required`` raises
        instead of falling back (except single queries, which stay CPU by design).
        The resident session is generation-keyed: a mutation or budget change
        invalidates it and the next batch re-uploads lazily.
        """

        policy = backend.policy
        policy_type = type(policy)
        if policy == policy_type.OFF:
            return None, {}
        available = bool(backend.dependency_available and backend.api_available)
        try:
            should_use = backend.should_use(
                policy=policy,
                dependency_available=available,
                query_batch_size=int(batch.shape[0]),
                maximum_batch_size=backend.max_batch,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"{backend.kind} direct TurboVec serving is required but unavailable: "
                + (
                    backend.unavailable_reason
                    or "the loaded TurboVec backend lacks the reconstruction APIs"
                )
            ) from exc
        if not should_use:
            status = "bypassed"
            reason = ""
            if int(batch.shape[0]) < 2:
                reason = f"{backend.reason_prefix}_batch_below_minimum"
            elif (
                policy == policy_type.AUTO
                and backend.max_batch is not None
                and int(batch.shape[0]) > backend.max_batch
            ):
                # Auto flips to the CPU kernel above the resident-favorable window:
                # the shared-top-k scan is faster at large batch.
                reason = f"{backend.reason_prefix}_batch_above_window"
            elif not backend.dependency_available:
                status = "failed_over"
                reason = backend.unavailable_reason or backend.default_unavailable_reason
            elif not backend.api_available:
                status = "failed_over"
                reason = "turbovec_reconstruction_api_unavailable"
            return None, {"gpu_stage_one_status": status, "gpu_fallback_reason": reason}
        if int(top_k) > backend.max_top_k:
            # Widened post-filter top-k requests would sort and copy back
            # huge candidate sets; the CPU kernel handles them instead.
            if policy == policy_type.REQUIRED:
                raise RuntimeError(
                    f"{backend.kind} direct TurboVec serving is required but the effective "
                    f"top_k {int(top_k)} exceeds the {backend.kind} limit {backend.max_top_k}"
                )
            return None, {
                "gpu_stage_one_status": "bypassed",
                "gpu_fallback_reason": f"{backend.reason_prefix}_top_k_exceeds_limit",
            }
        session = backend.sessions.get(state.index_key)
        if (
            session is None
            or session.generation != serving.generation
            # A budget change forces re-admission, matching the V1 session
            # cache: a lowered budget must evict an oversized resident copy.
            or session.memory_budget_bytes != backend.memory_budget_bytes
        ):
            try:
                session = backend.build_session()
            except MemoryError as exc:
                if policy == policy_type.REQUIRED:
                    raise RuntimeError(
                        f"{backend.kind} direct TurboVec memory admission rejected: {exc}"
                    ) from exc
                return None, {
                    "gpu_stage_one_status": "memory_rejected",
                    "gpu_fallback_reason": str(exc),
                    "gpu_budget_bytes": int(backend.memory_budget_bytes or 0),
                }
            except RuntimeError as exc:
                if policy == policy_type.REQUIRED:
                    raise
                return None, {
                    "gpu_stage_one_status": "failed_over",
                    "gpu_fallback_reason": f"{backend.reason_prefix}_build_failed: {exc}",
                }
            backend.sessions[state.index_key] = session
        try:
            result = session.search_batch(
                batch, top_k=int(top_k), allowlist_stable_ids=allowlist_stable_ids
            )
        except Exception as exc:  # noqa: BLE001 - auto policy must fall back visibly.
            backend.sessions.pop(state.index_key, None)
            if policy == policy_type.REQUIRED:
                raise RuntimeError(
                    f"{backend.kind} direct TurboVec batch search failed: {exc}"
                ) from exc
            return None, {
                "gpu_stage_one_status": "failed_over",
                "gpu_fallback_reason": f"{backend.reason_prefix}_runtime_error: {exc}",
            }
        return result, {
            "gpu_estimated_bytes": int(backend.estimated_bytes(session)),
            "gpu_budget_bytes": int(backend.memory_budget_bytes or 0),
            "gpu_resident_upload_build_ms": float(session.upload_build_ms),
        }

    def _try_query_mps_direct_turbovec_batch(
        self,
        state: ClientIndexState,
        *,
        serving: TurboVecServingIndex,
        batch: np.ndarray,
        top_k: int,
        allowlist_stable_ids: np.ndarray | None = None,
    ) -> tuple[Any | None, dict[str, Any]]:
        """Attempts the opt-in MPS-resident exact batch path for one direct-route batch.

        Thin Apple-GPU adapter: probes Metal, builds a :class:`_ResidentDirectBackend`,
        and delegates to :meth:`_try_query_resident_direct_batch`. Off by default;
        NEON stays the default on Apple Silicon. Reuses the ``gpu_*`` telemetry
        fields, disambiguated from the CUDA path by ``stage_one_backend``.
        """

        from lodedb.engine.gpu_turbovec import turbovec_reconstruction_api_available
        from lodedb.engine.mps_turbovec import (
            MPS_DIRECT_TURBOVEC_MAX_TOP_K,
            MpsDirectTurboVecSession,
            mps_exact_scan_available,
        )

        policy = self.mps_direct_turbovec_policy
        if policy == MpsDirectTurboVecPolicy.OFF:
            return None, {}
        # Probe Metal only once policy is on, mirroring the GPU adapter.
        mps_ok, mps_reason = mps_exact_scan_available()
        backend = _ResidentDirectBackend(
            kind="MPS",
            reason_prefix="mps_direct",
            policy=policy,
            dependency_available=bool(mps_ok),
            api_available=turbovec_reconstruction_api_available(serving.index),
            unavailable_reason=mps_reason or "",
            default_unavailable_reason="mps_unavailable",
            should_use=mps_direct_turbovec_should_use,
            max_batch=self.mps_direct_turbovec_max_batch,
            max_top_k=MPS_DIRECT_TURBOVEC_MAX_TOP_K,
            sessions=self._mps_direct_turbovec_sessions,
            memory_budget_bytes=self.mps_memory_budget_bytes,
            build_session=lambda: MpsDirectTurboVecSession.build(
                index=serving.index,
                generation=serving.generation,
                memory_budget_bytes=self.mps_memory_budget_bytes,
            ),
            estimated_bytes=lambda session: int(session.estimated_mps_bytes),
        )
        return self._try_query_resident_direct_batch(
            backend,
            state,
            serving=serving,
            batch=batch,
            top_k=top_k,
            allowlist_stable_ids=allowlist_stable_ids,
        )

    def _execute_prepared_query_batch(
        self,
        *,
        state: ClientIndexState,
        prepared: tuple[_PreparedQuery, ...],
        query_embeddings: tuple[tuple[float, ...], ...],
        query_results: list[_EngineTurboVecQueryResult | None],
        search_latencies: list[float],
    ) -> None:
        """Executes prepared direct-route queries, batching multi-query requests.

        Three partitions by mode. Pure-vector queries keep the batched group
        path. Hybrid queries that share a filter (two or more in a group) run
        their VECTOR component through the same batched group scan -- widened to
        the lexical pool -- and then fuse with BM25 + RRF per query on the CPU, so
        a batch of hybrid queries finally rides the GPU/MPS vector scan instead of
        scanning one query at a time. A lone hybrid query (the only one in its
        filter group) and every lexical query run on the individual CPU post-step
        path, keeping single-query parity exact. Request order is preserved for
        every result regardless of partition.
        """

        lexical_positions: list[int] = []
        hybrid_positions: list[int] = []
        vector_positions: list[int] = []
        for position, prepared_item in enumerate(prepared):
            if prepared_item.mode == RETRIEVAL_MODE_VECTOR:
                vector_positions.append(position)
            elif prepared_item.mode == RETRIEVAL_MODE_HYBRID:
                hybrid_positions.append(position)
            else:
                lexical_positions.append(position)

        # Lexical-only queries have no vector component to batch; run each as the
        # individual CPU post-step (BM25 alone).
        for position in lexical_positions:
            self._run_individual_hybrid(
                state, prepared=prepared, query_embeddings=query_embeddings,
                position=position, query_results=query_results,
                search_latencies=search_latencies,
            )

        # Hybrid queries: group by filter so a filter-homogeneous group of two or
        # more rides one batched vector scan; a singleton group falls back to the
        # individual path so its result is byte-for-byte the single-query result.
        if hybrid_positions:
            serving = self._turbovec_index_for_state(state)
            batch = np.asarray(query_embeddings, dtype=np.float32)
            hybrid_groups: dict[tuple[Any, ...], list[int]] = {}
            for position in hybrid_positions:
                hybrid_groups.setdefault(
                    _filter_signature(prepared[position].query_filter), []
                ).append(position)
            for positions in hybrid_groups.values():
                if len(positions) < 2:
                    self._run_individual_hybrid(
                        state, prepared=prepared, query_embeddings=query_embeddings,
                        position=positions[0], query_results=query_results,
                        search_latencies=search_latencies,
                    )
                    continue
                self._run_batched_hybrid_group(
                    state=state,
                    serving=serving,
                    batch=batch,
                    positions=positions,
                    prepared=prepared,
                    query_results=query_results,
                    search_latencies=search_latencies,
                )

        if not vector_positions:
            return
        if len(vector_positions) > 1:
            serving = self._turbovec_index_for_state(state)
            batch = np.asarray(query_embeddings, dtype=np.float32)
            # Group queries that share an identical filter and run each group as
            # one batched native call with a single shared allowlist. The common
            # ``search_many(filter=...)`` shape collapses to one group; per-query
            # top_k widths are sliced from the group's shared top-k, valid because
            # the kernel's ordering is deterministic for any prefix width.
            groups: dict[tuple[Any, ...], list[int]] = {}
            for position in vector_positions:
                groups.setdefault(
                    _filter_signature(prepared[position].query_filter), []
                ).append(position)
            for positions in groups.values():
                self._run_direct_batch_group(
                    state=state,
                    serving=serving,
                    batch=batch,
                    positions=positions,
                    prepared=prepared,
                    query_results=query_results,
                    search_latencies=search_latencies,
                )
            return
        position = vector_positions[0]
        prepared_item = prepared[position]
        started = perf_counter()
        query_results[position] = self._query_serving_index(
            state,
            route=prepared_item.route,
            query_embedding=query_embeddings[position],
            query_filter=prepared_item.query_filter,
            top_k=prepared_item.query.top_k,
        )
        search_latencies[position] = (perf_counter() - started) * 1000.0
        return

    def _run_individual_hybrid(
        self,
        state: ClientIndexState,
        *,
        prepared: tuple[_PreparedQuery, ...],
        query_embeddings: tuple[tuple[float, ...], ...],
        position: int,
        query_results: list[_EngineTurboVecQueryResult | None],
        search_latencies: list[float],
    ) -> None:
        """Runs one lexical/hybrid position on the individual CPU post-step path."""

        prepared_item = prepared[position]
        started = perf_counter()
        query_results[position] = self._query_serving_index_hybrid(
            state,
            query_embedding=query_embeddings[position],
            query_text=prepared_item.query.text,
            query_filter=prepared_item.query_filter,
            top_k=prepared_item.query.top_k,
            mode=prepared_item.mode,
        )
        search_latencies[position] = (perf_counter() - started) * 1000.0

    def _run_batched_hybrid_group(
        self,
        *,
        state: ClientIndexState,
        serving: TurboVecServingIndex,
        batch: np.ndarray,
        positions: list[int],
        prepared: tuple[_PreparedQuery, ...],
        query_results: list[_EngineTurboVecQueryResult | None],
        search_latencies: list[float],
    ) -> None:
        """Batches the vector component of a filter-homogeneous hybrid group, then fuses.

        Every position in ``positions`` is ``mode='hybrid'`` and shares the same
        filter. Their vector scan runs as one grouped (GPU/MPS or CPU) call via
        :meth:`_run_direct_batch_group`, each widened to that position's lexical
        pool, so the group is scanned once at ``max`` pool width and each position
        sliced to its own pool. The pool-wide vector result is then fused per
        position with the BM25 + RRF pass, constrained to the same metadata
        allowlist the scan used, so each fused result equals the single-query
        result for the same pool width on the same kernel.
        """

        query_filter = prepared[positions[0]].query_filter
        allowed_chunk_ids: tuple[str, ...] | None = None
        if query_filter:
            allowed_chunk_ids = self._build_filter_allowlist(state, query_filter)
            if not allowed_chunk_ids:
                # A filter that matches nothing constrains both rankers to empty,
                # exactly as the single hybrid path returns for an empty allowlist.
                for position in positions:
                    query_results[position] = _empty_turbovec_result(serving)
                    search_latencies[position] = 0.0
                return

        widths = {
            position: self._lexical_pool_width(prepared[position].query.top_k)
            for position in positions
        }
        # Run the vector component for the whole group in one batched scan; this
        # writes a pool-wide direct_turbovec result at each position.
        self._run_direct_batch_group(
            state=state,
            serving=serving,
            batch=batch,
            positions=positions,
            prepared=prepared,
            query_results=query_results,
            search_latencies=search_latencies,
            widths=widths,
        )
        # Fuse each position's pool-wide vector result with its BM25 ranking.
        for position in positions:
            base_result = query_results[position]
            if base_result is None:  # defensive: scan always writes a result
                base_result = _empty_turbovec_result(serving)
            fuse_started = perf_counter()
            query_results[position] = self._fuse_precomputed_vector_result(
                state,
                serving=serving,
                base_result=base_result,
                query_text=prepared[position].query.text,
                allowed_chunk_ids=allowed_chunk_ids,
                pool=widths[position],
                mode=RETRIEVAL_MODE_HYBRID,
            )
            search_latencies[position] += (perf_counter() - fuse_started) * 1000.0

    def _run_direct_batch_group(
        self,
        *,
        state: ClientIndexState,
        serving: TurboVecServingIndex,
        batch: np.ndarray,
        positions: list[int],
        prepared: tuple[_PreparedQuery, ...],
        query_results: list[_EngineTurboVecQueryResult | None],
        search_latencies: list[float],
        widths: dict[int, int] | None = None,
    ) -> None:
        """Runs one filter-homogeneous slice of a batch as a single native call.

        Every query in ``positions`` shares ``query_filter``. A filtered group
        pushes one shared allowlist into the scan (no top_k widening); an
        unfiltered group keeps the resident GPU/MPS fast path. ``widths`` maps
        each position to the number of rows that position wants from the scan,
        defaulting to its own ``query.top_k`` (so a pure-vector batch is
        byte-for-byte unchanged); a hybrid caller passes the widened lexical pool
        width so the vector component of the hybrid query rides this same batched
        scan. The group's shared scan width is ``max(widths.values())`` and each
        position is sliced to its own width, written back at its original
        position to preserve batch order.
        """

        from lodedb.engine.gpu_turbovec import GPU_DIRECT_TURBOVEC_BACKEND
        from lodedb.engine.mps_turbovec import MPS_DIRECT_TURBOVEC_BACKEND

        if widths is None:
            widths = {position: prepared[position].query.top_k for position in positions}
        query_filter = prepared[positions[0]].query_filter
        group_batch = np.ascontiguousarray(batch[positions])
        shared_top_k = max(widths[position] for position in positions)
        started = perf_counter()

        if query_filter:
            allowed_chunk_ids = self._build_filter_allowlist(state, query_filter)
            if not allowed_chunk_ids:
                elapsed_ms = (perf_counter() - started) * 1000.0
                for position in positions:
                    query_results[position] = _empty_turbovec_result(serving)
                    search_latencies[position] = elapsed_ms / len(positions)
                return
            # Try the resident (GPU/MPS) allowlist scan first; when no resident
            # backend is admitted, the CPU SIMD kernel honours the same allowlist
            # inside the scan -- either way top_k stays k, never the corpus size.
            resident = self._try_resident_allowlist_batch(
                state,
                serving=serving,
                group_batch=group_batch,
                top_k=shared_top_k,
                allowed_chunk_ids=allowed_chunk_ids,
            )
            if resident is not None:
                resident_batch, backend, fields = resident
                self._scatter_resident_batch_group(
                    positions=positions,
                    widths=widths,
                    serving=serving,
                    query_results=query_results,
                    search_latencies=search_latencies,
                    resident_batch=resident_batch,
                    backend=backend,
                    fields=fields,
                    started=started,
                )
                return
            batch_result = serving.search_batch(
                group_batch, top_k=shared_top_k, allowlist_chunk_ids=allowed_chunk_ids
            )
            self._scatter_cpu_batch_group(
                positions=positions,
                widths=widths,
                serving=serving,
                query_results=query_results,
                search_latencies=search_latencies,
                batch_result=batch_result,
                started=started,
                extra_fields={
                    "gpu_stage_one_status": "bypassed",
                    "gpu_fallback_reason": "filtered_batch_cpu_allowlist",
                },
            )
            return

        gpu_batch, gpu_fields = self._try_query_gpu_direct_turbovec_batch(
            state, serving=serving, batch=group_batch, top_k=shared_top_k
        )
        if gpu_batch is not None:
            self._scatter_resident_batch_group(
                positions=positions,
                widths=widths,
                serving=serving,
                query_results=query_results,
                search_latencies=search_latencies,
                resident_batch=gpu_batch,
                backend=GPU_DIRECT_TURBOVEC_BACKEND,
                fields=gpu_fields,
                started=started,
            )
            return
        mps_batch, mps_fields = self._try_query_mps_direct_turbovec_batch(
            state, serving=serving, batch=group_batch, top_k=shared_top_k
        )
        if mps_batch is not None:
            self._scatter_resident_batch_group(
                positions=positions,
                widths=widths,
                serving=serving,
                query_results=query_results,
                search_latencies=search_latencies,
                resident_batch=mps_batch,
                backend=MPS_DIRECT_TURBOVEC_BACKEND,
                fields=mps_fields,
                started=started,
            )
            return
        # Surface whichever resident backend bypassed/failed over so the "why this
        # batch fell back to the CPU kernel" status stays visible on the CPU rows.
        resident_fallback_fields = {**gpu_fields, **mps_fields}
        batch_result = serving.search_batch(group_batch, top_k=shared_top_k)
        self._scatter_cpu_batch_group(
            positions=positions,
            widths=widths,
            serving=serving,
            query_results=query_results,
            search_latencies=search_latencies,
            batch_result=batch_result,
            started=started,
            extra_fields=resident_fallback_fields,
        )

    def _try_resident_allowlist_batch(
        self,
        state: ClientIndexState,
        *,
        serving: TurboVecServingIndex,
        group_batch: np.ndarray,
        top_k: int,
        allowed_chunk_ids: tuple[str, ...],
    ) -> tuple[Any, str, dict[str, Any]] | None:
        """Resident (GPU/MPS) allowlist scan for a filtered group.

        Returns ``(batch_result, backend_label, telemetry_fields)`` when a
        resident backend served the filtered batch, else ``None`` so the caller
        falls back to the CPU allowlist kernel. The resident scan honours the
        allowlist via a per-tile ``-inf`` score mask, so top_k stays k and a
        filtered batch never widens past the resident cap.
        """

        from lodedb.engine.gpu_turbovec import GPU_DIRECT_TURBOVEC_BACKEND
        from lodedb.engine.mps_turbovec import MPS_DIRECT_TURBOVEC_BACKEND

        allowed_stable_ids = serving.stable_ids_for_chunks(allowed_chunk_ids)
        if allowed_stable_ids.size == 0:
            return None
        gpu_batch, gpu_fields = self._try_query_gpu_direct_turbovec_batch(
            state,
            serving=serving,
            batch=group_batch,
            top_k=top_k,
            allowlist_stable_ids=allowed_stable_ids,
        )
        if gpu_batch is not None:
            return gpu_batch, GPU_DIRECT_TURBOVEC_BACKEND, gpu_fields
        mps_batch, mps_fields = self._try_query_mps_direct_turbovec_batch(
            state,
            serving=serving,
            batch=group_batch,
            top_k=top_k,
            allowlist_stable_ids=allowed_stable_ids,
        )
        if mps_batch is not None:
            return mps_batch, MPS_DIRECT_TURBOVEC_BACKEND, mps_fields
        return None

    def _scatter_resident_batch_group(
        self,
        *,
        positions: list[int],
        widths: dict[int, int],
        serving: TurboVecServingIndex,
        query_results: list[_EngineTurboVecQueryResult | None],
        search_latencies: list[float],
        resident_batch: Any,
        backend: str,
        fields: dict[str, Any],
        started: float,
    ) -> None:
        """Writes one resident (GPU/MPS) group result back at original positions.

        Each position is sliced to ``widths[position]`` rows from the shared
        (wider) scan; a pure-vector caller passes its ``top_k`` and a hybrid
        caller passes the widened lexical pool width, so the same group scan
        serves both without rerunning the kernel.
        """

        per_query_ms = ((perf_counter() - started) * 1000.0) / max(len(positions), 1)
        for local_index, position in enumerate(positions):
            take = min(int(widths[position]), resident_batch.stable_ids.shape[1])
            query_results[position] = _EngineTurboVecQueryResult(
                index=serving,
                stable_ids=resident_batch.stable_ids[local_index, :take].reshape(-1),
                scores=resident_batch.scores[local_index, :take].reshape(-1),
                native_used=True,
                native_backend=backend,
                retrieval_mode="direct_turbovec",
                fallback_used=False,
                compact_route_fallback=False,
                stage_one_backend=backend,
                gpu_stage_one_status="used",
                gpu_query_count=len(positions),
                gpu_copy_back_bytes=int(resident_batch.copy_back_bytes),
                gpu_stage_one_search_ms=float(resident_batch.search_ms),
                gpu_device_to_host_copy_ms=float(resident_batch.device_to_host_copy_ms),
                gpu_stage_one_tile_count=int(resident_batch.tile_count),
                **fields,
            )
            search_latencies[position] = per_query_ms

    def _scatter_cpu_batch_group(
        self,
        *,
        positions: list[int],
        widths: dict[int, int],
        serving: TurboVecServingIndex,
        query_results: list[_EngineTurboVecQueryResult | None],
        search_latencies: list[float],
        batch_result: Any,
        started: float,
        extra_fields: dict[str, Any],
    ) -> None:
        """Writes one CPU-kernel group result back at original positions.

        Each position is sliced to ``widths[position]`` rows from the shared
        (wider) scan, mirroring :meth:`_scatter_resident_batch_group`.
        """

        per_query_ms = ((perf_counter() - started) * 1000.0) / max(len(positions), 1)
        for local_index, position in enumerate(positions):
            take = min(int(widths[position]), batch_result.stable_ids.shape[1])
            query_results[position] = _EngineTurboVecQueryResult(
                index=serving,
                stable_ids=batch_result.stable_ids[local_index, :take].reshape(-1),
                scores=batch_result.scores[local_index, :take].reshape(-1),
                native_used=batch_result.native_used,
                native_backend=batch_result.native_backend,
                retrieval_mode="direct_turbovec",
                fallback_used=False,
                compact_route_fallback=False,
                **extra_fields,
            )
            search_latencies[position] = per_query_ms

    def _turbovec_index_for_state(self, state: ClientIndexState) -> TurboVecServingIndex:
        """Returns a current TurboVec IdMapIndex view for one client state."""

        generation = self._index_generations.get(state.index_key, 0)
        cached = self._turbovec_indexes.get(state.index_key)
        if cached is not None and cached.generation == generation:
            return cached
        raise RuntimeError("direct TurboVec snapshot is not loaded")

    def _metadata_posting_index(self, state: ClientIndexState) -> _MetadataPostingIndex:
        """Returns the generation-current metadata posting index, building it lazily.

        Keyed by the same generation as the serving index, so any mutation (which
        advances the generation) transparently invalidates it; the rebuild is an
        O(corpus) pass that happens on the first filtered query of a generation
        and is then reused, never touching the write/commit path.
        """

        generation = self._index_generations.get(state.index_key, 0)
        cached = self._metadata_posting_indexes.get(state.index_key)
        if cached is not None and cached.generation == generation:
            return cached
        # Build the per-field value indexes once; the exact-match (key, value)
        # postings are derived from them (shared doc sets, read-only use), so the
        # metadata is scanned a single time for both the fast exact path and the
        # predicate planner.
        fields, all_docs = _filter_plan.build_field_indexes(state.document_metadata)
        postings: dict[tuple[str, str], set[str]] = {
            (key, value): docs
            for key, field in fields.items()
            for value, docs in field.value_docs.items()
        }
        chunks_by_document: dict[str, list[str]] = {}
        for chunk_id, chunk in state.chunks.items():
            chunks_by_document.setdefault(str(chunk.document_id), []).append(str(chunk_id))
        index = _MetadataPostingIndex(generation, postings, chunks_by_document, fields, all_docs)
        self._metadata_posting_indexes[state.index_key] = index
        return index

    def _lexical_document_units(
        self, state: ClientIndexState, document_id: str
    ) -> list[tuple[str, list[str]]]:
        """Returns one document's ``(chunk_id, tokens)`` units for an incremental fold.

        Used only by the O(changed) incremental path of
        :meth:`_lexical_serving_index`, so it materializes a single changed
        document's chunks: on the persisted-token path it zips the captured
        per-chunk tokens against the recorded chunk ids; on the raw-text path it
        re-chunks the document's stored text the same way ingest does and
        tokenizes only those chunks (so tokenization stays on the query, never on
        the write path). A document with no stored tokens/text contributes no
        units, which :meth:`Bm25Index.replace_group` treats as a removal.
        """

        chunk_ids = state.document_chunk_ids.get(document_id, ())
        if not chunk_ids:
            return []
        if self.lexical_index_enabled and state.document_tokens:
            chunks = state.document_tokens.get(document_id)
            if not chunks:
                return []
            return [
                (str(chunk_id), [str(token) for token in tokens])
                for chunk_id, tokens in zip(chunk_ids, chunks, strict=False)
            ]
        text = state.document_text.get(document_id)
        if text is None:
            return []
        pieces = chunk_text(text, self.chunk_character_limit)
        return [
            (str(chunk_id), tokenize(piece))
            for chunk_id, piece in zip(chunk_ids, pieces, strict=False)
        ]

    def _lexical_serving_index(self, state: ClientIndexState) -> _LexicalServingIndex:
        """Returns the generation-current BM25 index, building it lazily for the mode.

        Keyed by the same generation as the serving index. When lexical-index
        persistence is on, the index is built from the persisted per-chunk token
        lists with no raw-text dependency and no re-tokenization; otherwise it is
        built from the retained raw document text re-chunked the same way ingest
        chunks it. Both share the exact chunk id space the vector scan ranks over.

        On a generation miss only the documents changed since the cache was last
        stamped (tracked at mutation time in ``_pending_lexical_documents``) are
        folded into the cached :class:`Bm25Index`: each deleted document's group
        is dropped and each upserted document's group is replaced, so the work is
        O(changed documents), not O(corpus). The raw-text path tokenizes only the
        changed documents' chunks; unchanged chunks keep their postings. When the
        pending delta grew past the rebuild bound (``_lexical_full_rebuild``) or
        the cache is cold, a full bulk build runs instead. Either way the served
        index is observably identical to a fresh build over the same final unit
        set. The index is held in memory only — a BM25 inverted index is
        payload-derived. Callers must have verified the mode is serviceable (see
        :meth:`_require_lexical_capable`); this assumes it.
        """

        key = state.index_key
        generation = self._index_generations.get(key, 0)
        cached = self._lexical_indexes.get(key)
        if cached is not None and cached.generation == generation:
            return cached

        if cached is not None and key not in self._lexical_full_rebuild:
            pending = self._pending_lexical_documents.get(key)
            upserted = pending["upserted"] if pending is not None else set()
            deleted = pending["deleted"] if pending is not None else set()
            bm25 = cached.bm25
            for document_id in deleted:
                bm25.remove_group(document_id)
            for document_id in upserted:
                bm25.replace_group(
                    document_id, self._lexical_document_units(state, document_id)
                )
            self._pending_lexical_documents.pop(key, None)
            cached.generation = generation
            return cached

        use_tokens = bool(self.lexical_index_enabled and state.document_tokens)
        if use_tokens:
            chunk_ids, token_lists, group_ids = build_chunk_token_lists(
                state.document_tokens,
                state.document_chunk_ids,
            )
            bm25 = Bm25Index.from_token_lists(
                chunk_ids, token_lists, group_ids=group_ids
            )
        else:
            chunk_ids, chunk_texts, group_ids = build_chunk_texts(
                state.document_text,
                state.document_chunk_ids,
                chunk_text,
                self.chunk_character_limit,
            )
            bm25 = Bm25Index(chunk_ids, chunk_texts, group_ids=group_ids)
        index = _LexicalServingIndex(generation, bm25)
        self._lexical_indexes[key] = index
        self._pending_lexical_documents.pop(key, None)
        self._lexical_full_rebuild.discard(key)
        return index

    def _require_lexical_capable(self, mode: str) -> str | None:
        """Returns an actionable error message when a lexical mode is unavailable.

        Lexical and hybrid retrieval build a BM25 index from either the persisted
        lexical postings (``index_text=True``) or the retained raw text
        (``store_text=True``), so a serviceable mode needs at least one of those.
        Returns ``None`` when the mode is serviceable.
        """

        if (
            mode in _LEXICAL_MODES
            and not self.lexical_index_enabled
            and not self.raw_text_storage_enabled
        ):
            return (
                f"mode={mode!r} requires a lexical source; open LodeDB with "
                "index_text=True (persists the BM25 postings) or store_text=True "
                "(the BM25 index is rebuilt from the raw-text store)"
            )
        return None

    def _lexical_chunk_ranking(
        self,
        state: ClientIndexState,
        *,
        query_text: str,
        limit: int,
        allowed_chunk_ids: tuple[str, ...] | None,
    ) -> list[str]:
        """Returns BM25-ranked chunk ids for one query, constrained to the allowlist."""

        lexical = self._lexical_serving_index(state)
        allowed_positions = (
            lexical.allowed_positions(allowed_chunk_ids)
            if allowed_chunk_ids is not None
            else None
        )
        if allowed_positions is not None and not allowed_positions:
            return []
        ranked = lexical.bm25.rank(
            query_text, limit=limit, allowed_indices=allowed_positions
        )
        return [chunk_id for chunk_id, _score in ranked]

    def _vector_chunk_ranking(
        self,
        vector_result: _EngineTurboVecQueryResult,
    ) -> list[str]:
        """Returns the vector scan's chunk ids in rank order (best first)."""

        chunk_ids_by_stable_id = vector_result.index.chunk_ids_by_stable_id
        return [
            chunk_ids_by_stable_id[int(stable_id)]
            for stable_id in vector_result.stable_ids
        ]

    def _fused_query_result(
        self,
        *,
        serving: TurboVecServingIndex,
        vector_chunk_ids: list[str],
        lexical_chunk_ids: list[str],
        mode: str,
        base_result: _EngineTurboVecQueryResult,
    ) -> _EngineTurboVecQueryResult:
        """Fuses the vector and lexical chunk rankings into a result for materialization.

        ``hybrid`` fuses both rankings with RRF; ``lexical`` uses the BM25 order
        alone (still expressed as an RRF-style descending score so the public row
        shape is unchanged). The fused chunk ids are mapped back to stable ids so
        the existing :func:`_materialize_query_results` collapses chunks to
        documents exactly as the vector path does.
        """

        if mode == RETRIEVAL_MODE_LEXICAL:
            fused = reciprocal_rank_fusion((lexical_chunk_ids,))
        else:
            fused = reciprocal_rank_fusion((vector_chunk_ids, lexical_chunk_ids))
        reverse = serving._stable_id_by_chunk_id  # type: ignore[attr-defined]
        stable_ids: list[int] = []
        scores: list[float] = []
        for chunk_id, score in fused:
            stable_id = reverse.get(chunk_id)
            if stable_id is None:
                continue
            stable_ids.append(int(stable_id))
            scores.append(float(score))
        return _EngineTurboVecQueryResult(
            index=serving,
            stable_ids=np.asarray(stable_ids, dtype=np.uint64),
            scores=np.asarray(scores, dtype=np.float32),
            native_used=base_result.native_used,
            native_backend=base_result.native_backend,
            retrieval_mode=mode,
            fallback_used=base_result.fallback_used,
            compact_route_fallback=base_result.compact_route_fallback,
        )

    def _query_serving_index_hybrid(
        self,
        state: ClientIndexState,
        *,
        query_embedding: tuple[float, ...],
        query_text: str,
        query_filter: Mapping[str, Any],
        top_k: int,
        mode: str,
    ) -> _EngineTurboVecQueryResult:
        """Runs a lexical or hybrid query as a pure-CPU post-step over the vector scan.

        Pulls a widened candidate pool from each ranker (so the fused top-k is not
        capped by either ranker's own top-k), constrains BOTH rankers to the same
        metadata allowlist, fuses with RRF, and returns a result whose stable ids
        carry the fused chunk order for the shared materialization step. The
        vector scan itself is untouched: this never enters the GPU/MPS or kernel
        paths and adds nothing to the write/commit path.
        """

        serving = self._turbovec_index_for_state(state)
        pool = self._lexical_pool_width(top_k)
        allowed_chunk_ids: tuple[str, ...] | None = None
        if query_filter:
            allowed_chunk_ids = self._build_filter_allowlist(state, query_filter)
            if not allowed_chunk_ids:
                return _empty_turbovec_result(serving)

        if mode == RETRIEVAL_MODE_LEXICAL:
            base_result = _empty_turbovec_result(serving)
        else:
            base_result = self._query_direct_turbovec_index(
                state,
                query_embedding=query_embedding,
                top_k=pool,
                query_filter=query_filter,
            )
        return self._fuse_precomputed_vector_result(
            state,
            serving=serving,
            base_result=base_result,
            query_text=query_text,
            allowed_chunk_ids=allowed_chunk_ids,
            pool=pool,
            mode=mode,
        )

    @staticmethod
    def _lexical_pool_width(top_k: int) -> int:
        """Returns the widened candidate-pool width pulled from each ranker.

        ``max(top_k * factor, floor)`` chunks, so the fused top-k is not capped by
        either ranker's own top-k. Shared by the single and batched hybrid paths
        so both widen the vector scan identically (the batched path must scan at
        this width for its per-query fusion to match the single path exactly).
        """

        return max(top_k * _LEXICAL_POOL_FACTOR, _LEXICAL_POOL_FLOOR)

    def _fuse_precomputed_vector_result(
        self,
        state: ClientIndexState,
        *,
        serving: TurboVecServingIndex,
        base_result: _EngineTurboVecQueryResult,
        query_text: str,
        allowed_chunk_ids: tuple[str, ...] | None,
        pool: int,
        mode: str,
    ) -> _EngineTurboVecQueryResult:
        """Fuses an already-computed vector result with the BM25 + RRF lexical pass.

        The single hybrid path computes ``base_result`` one query at a time; the
        batched path computes it for many queries in one grouped (GPU/MPS) scan
        and calls this per query. Both then run the identical lexical ranking
        (constrained to the same ``allowed_chunk_ids`` the vector scan was, so the
        metadata allowlist governs both rankers) and RRF fusion, so a batched
        hybrid query returns byte-for-byte the same ranking and scores as the
        repeated single query for the same pool width on the same kernel. For
        ``lexical`` mode ``base_result`` is the empty vector result and only the
        BM25 order contributes.
        """

        vector_chunk_ids = (
            [] if mode == RETRIEVAL_MODE_LEXICAL else self._vector_chunk_ranking(base_result)
        )
        lexical_chunk_ids = self._lexical_chunk_ranking(
            state,
            query_text=query_text,
            limit=pool,
            allowed_chunk_ids=allowed_chunk_ids,
        )
        return self._fused_query_result(
            serving=serving,
            vector_chunk_ids=vector_chunk_ids,
            lexical_chunk_ids=lexical_chunk_ids,
            mode=mode,
            base_result=base_result,
        )

    def _build_filter_allowlist(
        self, state: ClientIndexState, query_filter: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Returns the chunk ids eligible under a query filter (the native allowlist).

        Shared by the single-query and batch paths so filters are pushed into the
        TurboVec scan instead of widening top_k to the corpus and post-filtering.
        Exact-match filters resolve through the metadata posting index in
        O(matching docs + chunks). Predicate filters (comparison operators or
        logical composition) can't be expressed as equality postings, so they
        resolve via a compiled-predicate corpus scan instead.
        """

        index = self._metadata_posting_index(state)
        if _is_predicate_filter(query_filter.get("metadata")):
            # Predicate filters resolve through the per-field planner (set ops +
            # bisect, O(matches + log V)) instead of the O(corpus) compiled scan.
            # _scan_filter_allowlist is retained as the parity oracle in tests.
            return index.chunk_allowlist(query_filter)
        return index.allowlist(query_filter)

    def _scan_filter_allowlist(
        self, state: ClientIndexState, query_filter: Mapping[str, Any]
    ) -> tuple[str, ...]:
        """Resolves a predicate filter to eligible chunk ids by a compiled scan.

        O(corpus), but the predicate is compiled once (operators dispatched,
        operands pre-parsed -- see ``_predicate.compile_metadata_filter``) and, on
        the batch path, this runs once per filter group rather than per query.
        """

        matches = _compile_query_filter(query_filter)
        document_metadata = state.document_metadata
        return tuple(
            str(chunk.chunk_id)
            for chunk in state.chunks.values()
            if matches(chunk.document_id, document_metadata.get(chunk.document_id, {}))
        )

    def _sync_direct_turbovec_index(
        self, state: ClientIndexState, changeset: _TurboVecChangeset | None = None
    ) -> tuple[str, ...]:
        """Applies transient full-embedding mutations to a direct TurboVec index.

        Returns the chunk ids whose transient full embeddings are now encoded in
        the index and can therefore be discarded by the caller (the rows added
        this sync), so the discard is O(changed) rather than O(corpus).

        When ``changeset`` is supplied and an index already exists, the diff is
        taken straight from the verb's exact ``(added_chunks, removed_chunk_ids)``
        in O(changed) instead of re-deriving it against the whole corpus. The
        changeset is only a hint: every removed id is confirmed present in the
        live id map and gone from ``state.chunks``, and every added chunk is
        confirmed still in ``state.chunks`` and not already indexed, so an
        over-reporting changeset can never corrupt the index — it degrades at
        worst to the same set the full scan would have produced. When
        ``changeset`` is ``None`` (or no prior index exists) the original
        full-corpus diff runs verbatim as a fallback.
        """

        if not _state_uses_direct_turbovec(state):
            return ()
        # Try to patch the GPU-resident dequantized copy in-place; if it fails,
        # pop it and the next eligible batch lazily re-uploads against the new generation.
        gpu_session = self._gpu_direct_turbovec_sessions.get(state.index_key)
        mps_session = self._mps_direct_turbovec_sessions.get(state.index_key)
        previous = self._turbovec_indexes.get(state.index_key)
        # Incremental reconciliation needs a prior index to mutate; the cold-build
        # branch below ignores the changeset and re-encodes the whole corpus.
        incremental = previous is not None and changeset is not None
        # The full-diff fallback needs the corpus-wide id set; the incremental
        # path uses O(1) ``state.chunks`` membership instead, so only pay for the
        # set when falling back.
        current_chunk_ids = set(state.chunks) if not incremental else frozenset()
        generation = self._index_generations.get(state.index_key, 0) + 1
        progress_label = _direct_turbovec_progress_label(
            self,
            state,
            generation=generation,
        )
        if previous is None:
            full_chunks = tuple(state.chunks.values())
            if not _chunks_have_full_embeddings(full_chunks, native_dim=state.native_dim):
                raise RuntimeError("direct TurboVec cannot be rebuilt without full embeddings")
            built = build_turbovec_serving_index(
                full_chunks,
                native_dim=state.native_dim,
                bit_width=_turbovec_bit_width_for_state(state),
                generation=generation,
                id_map_index_class=self.turbovec_id_map_index_class,
                progress_label=progress_label,
            )
            self._turbovec_indexes[state.index_key] = built
            self._index_generations[state.index_key] = generation
            self._pending_tvim_deltas.pop(state.index_key, None)
            # A cold rebuild re-encodes every chunk, so every chunk's transient
            # embedding is now in the index and can be discarded.
            return tuple(state.chunks)
        if incremental:
            assert changeset is not None  # narrowed by ``incremental``
            # O(changed) removals: map each reported removed chunk id to its
            # derived stable id and keep only those the index actually still holds
            # and that are truly gone from state.chunks. The chunk-id match guards
            # against a (vanishingly unlikely) stable-id collision aliasing a
            # different live chunk, and the ``seen`` set dedupes a chunk reported
            # twice in one batch so remove_many's count check stays exact.
            removed_seen: set[int] = set()
            removed_stable_list: list[int] = []
            removed_chunk_ids_applied: list[str] = []
            for chunk_id in changeset.removed_chunk_ids:
                if chunk_id in state.chunks:
                    continue  # re-added within the same batch: not a net removal
                stable_id = int(
                    stable_uint64_ids_for_chunk_ids((str(chunk_id),))[0]
                )
                if stable_id in removed_seen:
                    continue
                if previous.chunk_ids_by_stable_id.get(stable_id) != str(chunk_id):
                    continue  # not indexed under this id (already gone or collision)
                removed_seen.add(stable_id)
                removed_stable_list.append(stable_id)
                removed_chunk_ids_applied.append(str(chunk_id))
            removed_stable_ids = tuple(removed_stable_list)
        else:
            removed_stable_ids = tuple(
                int(stable_id)
                for stable_id, chunk_id in previous.chunk_ids_by_stable_id.items()
                if chunk_id not in current_chunk_ids
            )
        if removed_stable_ids:
            _log_direct_turbovec_update_progress(
                progress_label,
                phase="remove",
                event="start",
                chunk_count=len(removed_stable_ids),
                native_dim=state.native_dim,
                bit_width=_turbovec_bit_width_for_state(state),
                generation=generation,
            )
            started = perf_counter()
        if removed_stable_ids and hasattr(previous.index, "remove_many"):
            removed_count = int(
                previous.index.remove_many(np.asarray(removed_stable_ids, dtype=np.uint64))
            )
            if removed_count != len(removed_stable_ids):
                raise RuntimeError(
                    "direct TurboVec batched removal count mismatch: "
                    f"expected {len(removed_stable_ids)}, removed {removed_count}"
                )
            for stable_id in removed_stable_ids:
                previous.chunk_ids_by_stable_id.pop(int(stable_id), None)
                previous.document_ids_by_stable_id.pop(int(stable_id), None)
        else:
            for stable_id in removed_stable_ids:
                previous.index.remove(int(stable_id))
                previous.chunk_ids_by_stable_id.pop(int(stable_id), None)
                previous.document_ids_by_stable_id.pop(int(stable_id), None)
        if removed_stable_ids:
            _log_direct_turbovec_update_progress(
                progress_label,
                phase="remove",
                event="end",
                chunk_count=len(removed_stable_ids),
                native_dim=state.native_dim,
                bit_width=_turbovec_bit_width_for_state(state),
                generation=generation,
                elapsed_ms=(perf_counter() - started) * 1000.0,
            )
        if incremental:
            assert changeset is not None  # narrowed by ``incremental``
            # O(changed) additions: keep each reported added chunk that survived
            # to state.chunks and is not already indexed (after the removals
            # above). The stable-id and chunk-id truth checks make a chunk that
            # was added then orphaned in the same batch, or one already present,
            # drop out — yielding exactly the set the full scan would, in the same
            # relative order, so the batched stable-id assignment below matches.
            added_seen: set[str] = set()
            incremental_new: list[Any] = []
            for chunk in changeset.added_chunks:
                chunk_id = str(chunk.chunk_id)
                if chunk_id in added_seen:
                    continue
                if chunk_id not in state.chunks:
                    continue  # added then removed within the batch
                candidate_stable_id = int(
                    stable_uint64_ids_for_chunk_ids((chunk_id,))[0]
                )
                if candidate_stable_id in previous.chunk_ids_by_stable_id:
                    continue  # already indexed (e.g. unchanged reused chunk)
                added_seen.add(chunk_id)
                incremental_new.append(chunk)
            new_chunks = tuple(incremental_new)
        else:
            indexed_chunk_ids = set(previous.chunk_ids_by_stable_id.values())
            new_chunks = tuple(
                chunk
                for chunk in state.chunks.values()
                if chunk.chunk_id not in indexed_chunk_ids
            )
        if new_chunks:
            if not _chunks_have_full_embeddings(new_chunks, native_dim=state.native_dim):
                raise RuntimeError("direct TurboVec new chunks require transient full embeddings")
            vectors = np.asarray([chunk.embedding for chunk in new_chunks], dtype=np.float32)
            stable_ids = stable_uint64_ids_for_chunk_ids(
                tuple(str(chunk.chunk_id) for chunk in new_chunks)
            )
            _log_direct_turbovec_update_progress(
                progress_label,
                phase="add_with_ids",
                event="start",
                chunk_count=len(new_chunks),
                native_dim=state.native_dim,
                bit_width=_turbovec_bit_width_for_state(state),
                generation=generation,
            )
            started = perf_counter()
            previous.index.add_with_ids(vectors, stable_ids)
            _log_direct_turbovec_update_progress(
                progress_label,
                phase="add_with_ids",
                event="end",
                chunk_count=len(new_chunks),
                native_dim=state.native_dim,
                bit_width=_turbovec_bit_width_for_state(state),
                generation=generation,
                elapsed_ms=(perf_counter() - started) * 1000.0,
            )
            # Do NOT eagerly prepare() here. The add invalidated TurboVec's
            # derived SIMD "blocked" layout; rebuilding it now (an O(corpus)
            # repack) only for the next mutation to invalidate it again makes
            # every incremental commit O(corpus). `search` rebuilds the layout
            # lazily on the next query, so a burst of commits pays a single
            # repack at the next query instead of one per commit. (Cold builds
            # still prepare once in build_turbovec_serving_index; the packed
            # codes persistence exports come from add_with_ids above, not the
            # blocked layout.)
            for stable_id, chunk in zip(stable_ids, new_chunks, strict=True):
                previous.chunk_ids_by_stable_id[int(stable_id)] = str(chunk.chunk_id)
                previous.document_ids_by_stable_id[int(stable_id)] = str(chunk.document_id)
        if gpu_session is not None:
            try:
                upsert_stable_ids = tuple(int(uid) for uid in stable_ids) if new_chunks else ()
                gpu_session.patch(
                    index=previous.index,
                    removed_ids=removed_stable_ids,
                    upsert_ids=upsert_stable_ids,
                    generation=generation,
                )
            except Exception:
                # Patch failed (e.g. MemoryError from over-allocation); safely fail closed
                # and let the next query rebuild the GPU array using safe O(N) allocation
                self._gpu_direct_turbovec_sessions.pop(state.index_key, None)
        if mps_session is not None:
            try:
                mps_upsert_stable_ids = tuple(int(uid) for uid in stable_ids) if new_chunks else ()
                mps_session.patch(
                    index=previous.index,
                    removed_ids=removed_stable_ids,
                    upsert_ids=mps_upsert_stable_ids,
                    generation=generation,
                )
            except Exception:
                # Fail closed: drop the resident copy so the next batch rebuilds.
                self._mps_direct_turbovec_sessions.pop(state.index_key, None)
        if incremental:
            # Reuse the existing serving index in place rather than constructing a
            # fresh one. The forward id maps were already mutated above (removals
            # popped, adds inserted); patch the derived reverse chunk-id -> stable-id
            # map for the same O(changed) ids and bump the generation. Building a new
            # frozen TurboVecServingIndex would re-derive that reverse map from the
            # whole forward map in TurboVecServingIndex.__post_init__ — an O(corpus)
            # cost on every mutation, which is the floor this incremental path exists
            # to remove. Safe because every engine access is serialized by
            # @_synchronized, this engine is the sole in-process holder of the object,
            # and concurrent readers are separate processes loading from disk.
            reverse_map: dict[str, int] = previous._stable_id_by_chunk_id
            for removed_chunk_id in removed_chunk_ids_applied:
                reverse_map.pop(removed_chunk_id, None)
            if new_chunks:
                for stable_id, chunk in zip(stable_ids, new_chunks, strict=True):
                    reverse_map[str(chunk.chunk_id)] = int(stable_id)
            object.__setattr__(previous, "generation", generation)
            self._turbovec_indexes[state.index_key] = previous
        else:
            self._turbovec_indexes[state.index_key] = TurboVecServingIndex(
                index=previous.index,
                chunk_ids_by_stable_id=dict(previous.chunk_ids_by_stable_id),
                document_ids_by_stable_id=dict(previous.document_ids_by_stable_id),
                dim=previous.dim,
                bit_width=previous.bit_width,
                generation=generation,
                native_backend=previous.native_backend,
                native_used=previous.native_used,
                build_seconds=previous.build_seconds,
            )
        self._index_generations[state.index_key] = generation
        pending = self._pending_tvim_deltas.setdefault(
            state.index_key, {"upserted": (), "removed": ()}
        )
        pending["upserted"] = tuple(pending["upserted"]) + tuple(
            int(stable_id) for stable_id in (stable_ids if new_chunks else ())
        )
        pending["removed"] = tuple(pending["removed"]) + tuple(
            int(stable_id) for stable_id in removed_stable_ids
        )
        # Buffer the new rows for a deferred, query-warm drift sample rather than
        # searching here — a per-commit drift search would force the O(corpus)
        # layout rebuild this method now avoids. _sample_pending_drift consumes
        # the buffer on the next query that warms the layout.
        self._buffer_pending_drift(state, new_chunks)
        return tuple(str(chunk.chunk_id) for chunk in new_chunks)

    def _buffer_pending_drift(self, state: ClientIndexState, new_chunks: tuple[Any, ...]) -> None:
        """Records recently-added rows for a deferred, query-warm drift sample.

        Sampling drift needs a self-score search, which needs the warm SIMD
        layout, so it is kept off the commit path. The embeddings are buffered
        here (not read from ``state.chunks``) because the transient copies there
        are zeroed right after the commit. The buffer keeps only the most recent
        rows so a long ingest burst stays O(1) in memory.
        """

        if not new_chunks:
            return
        buffer = self._pending_drift_samples.setdefault(state.index_key, [])
        for chunk in new_chunks:
            buffer.append((str(chunk.chunk_id), tuple(float(v) for v in chunk.embedding)))
        if len(buffer) > _DRIFT_SAMPLE_LIMIT:
            del buffer[:-_DRIFT_SAMPLE_LIMIT]

    def _sample_pending_drift(self, state: ClientIndexState) -> None:
        """Samples buffered rows' quantization drift; called when the layout is warm.

        Invoked from the query path after a search has (re)built the SIMD
        layout, so these self-score searches are cheap. For each sampled new row
        the exact self inner product is its embedding norm squared; the TurboVec
        self-score gap measures how far 4-bit quantization moved the row. Emits
        the mean relative gap and clears the buffer.
        """

        samples = self._pending_drift_samples.get(state.index_key)
        if not samples:
            return
        serving = self._turbovec_indexes.get(state.index_key)
        if serving is None:
            self._pending_drift_samples.pop(state.index_key, None)
            return
        ratios: list[float] = []
        for chunk_id, embedding in samples:
            vector = np.asarray(embedding, dtype=np.float32)
            expected = float(np.dot(vector, vector))
            result = serving.search(
                tuple(float(value) for value in vector),
                top_k=1,
                allowlist_chunk_ids=(chunk_id,),
            )
            if result.scores.size:
                observed = float(result.scores.reshape(-1)[0])
                ratios.append(abs(observed - expected) / max(abs(expected), 1e-9))
        self._turbovec_drift_telemetry[state.index_key] = {
            "turbovec_drift_sample_rows": float(len(ratios)),
            "turbovec_self_score_drift_ratio": float(np.mean(ratios)) if ratios else 0.0,
        }
        self._pending_drift_samples.pop(state.index_key, None)

    def _turbovec_drift_fields(self, state: ClientIndexState) -> dict[str, float | bool]:
        """Returns drift and calibration-lifecycle telemetry for mutation events.

        ``turbovec_calibration_rebuild_recommended`` flags indexes that grew
        past the TQ+ sample threshold while frozen on identity calibration
        (a trickle-ingested index permanently missing the TQ+ recall lift).
        """

        fields: dict[str, float | bool] = dict(
            self._turbovec_drift_telemetry.get(
                state.index_key,
                {
                    "turbovec_drift_sample_rows": 0.0,
                    "turbovec_self_score_drift_ratio": 0.0,
                },
            )
        )
        serving = self._turbovec_indexes.get(state.index_key)
        if serving is not None and hasattr(serving.index, "calibration_fitted"):
            fitted = bool(serving.index.calibration_fitted())
            fields["turbovec_calibration_fitted"] = fitted
            fields["turbovec_calibration_rebuild_recommended"] = (
                not fitted and len(state.chunks) >= 1000
            )
        return fields

    def _mark_index_changed(self, client_id_hash: str) -> None:
        """Invalidates the cached columnar view after index contents change."""

        self._index_generations[client_id_hash] = self._index_generations.get(client_id_hash, 0) + 1
        self._turbovec_indexes.pop(client_id_hash, None)
        self._gpu_direct_turbovec_sessions.pop(client_id_hash, None)
        self._mps_direct_turbovec_sessions.pop(client_id_hash, None)

    # -- write-ahead log (opt-in commit_mode="wal") -------------------------

    def _stage_wal_record(
        self,
        context: EngineRequestContext,
        state: ClientIndexState,
        op: str,
        payload: dict[str, Any],
    ) -> None:
        """Stages the logical mutation a verb will append to the WAL on persist.

        A no-op outside WAL mode (so the verbs stay branch-light) and while
        replaying. The record envelope captures the live ``client_id``/``index_id``
        so replay resolves the same index the live call did (the persisted state
        snapshot drops the raw ``client_id``, so it cannot be recovered from the
        loaded state alone). ``_persist_state`` then appends the staged record
        instead of publishing a generation.
        """

        if self._commit_mode != CommitMode.WAL or self._wal_replaying:
            return
        envelope = {
            "client_id": context.client_id,
            "index_id": state.index_id,
            **payload,
        }
        self._pending_wal_records[state.index_key] = (op, envelope)

    def _wal_store_for(self, index_key: str) -> WalStore:
        """Returns (opening lazily) the WAL store for one index key."""

        store = self._wal_stores.get(index_key)
        if store is None:
            store = WalStore(
                wal_path(self.persistence_dir, index_key), fsync=self._fsync_on_commit
            )
            self._wal_stores[index_key] = store
        return store

    def _append_wal_record(self, state: ClientIndexState) -> None:
        """Appends the staged logical mutation to the WAL and maybe checkpoints.

        The in-memory index was already synced before this runs, so the record
        only has to make the mutation recoverable. After the append, if the WAL
        backlog crosses the checkpoint threshold the WAL is folded into a fresh
        committed generation and truncated, bounding replay time and file size.
        """

        record = self._pending_wal_records.pop(state.index_key, None)
        if record is None:
            # No logical mutation was staged (e.g. an internal persist with no
            # verb). Fall back to a generation publish so nothing is lost.
            self._persist_generation(state)
            return
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        op, payload = record
        store = self._wal_store_for(state.index_key)
        store.append(op, payload)
        if store.should_checkpoint(
            checkpoint_ops=self._wal_checkpoint_ops,
            checkpoint_bytes=self._wal_checkpoint_bytes,
        ):
            self._checkpoint_wal(state)

    def _checkpoint_wal(self, state: ClientIndexState) -> None:
        """Folds the WAL into a fresh committed generation, then truncates it.

        The generation is committed via the atomic root-manifest swap *before*
        the WAL is truncated, so a crash between the two simply replays a few
        already-applied (idempotent) records on the next open rather than losing
        them. Marks delta-append ineligible so the checkpoint always writes a
        fresh base epoch (the live deltas may not reflect every WAL mutation).
        """

        store = self._wal_stores.get(state.index_key)
        if store is None or store.op_count == 0:
            return
        # Drop any pending delta accumulators: a WAL run may have advanced the
        # in-memory index past what the live base+deltas describe, so the
        # checkpoint must write a self-contained fresh base, not append a delta.
        self._pending_state_journal_documents.pop(state.index_key, None)
        self._pending_tvim_deltas.pop(state.index_key, None)
        self._base_epochs.pop(state.index_key, None)
        self._persist_generation(state)
        store.truncate()

    def _checkpoint_all_wals(self) -> None:
        """Checkpoints every open WAL into a generation (used by persist/close)."""

        if self._commit_mode != CommitMode.WAL:
            return
        for index_key, store in list(self._wal_stores.items()):
            if store.op_count == 0:
                continue
            state = self._indexes.get(index_key)
            if state is not None:
                self._checkpoint_wal(state)

    def _replay_wal(self, state: ClientIndexState) -> dict[str, float]:
        """Replays this index's intact WAL records onto the loaded generation.

        Each record re-invokes the public engine verb that produced it, so the
        recovered state is rebuilt by the identical ingest/sync path — never a
        parallel decoder. A torn trailing record (writer crashed mid-append) was
        already dropped by the store, so replay applies exactly the mutations
        that were durably logged. Runs under ``_wal_replaying`` so the replayed
        verbs do not re-append to (or checkpoint) the WAL.
        """

        store = self._wal_store_for(state.index_key)
        records = store.read_records()
        if not records:
            return {"wal_records_replayed": 0.0}
        started = perf_counter()
        self._wal_replaying = True
        try:
            for record in records:
                self._apply_wal_record(state, record.op, record.payload)
        finally:
            self._wal_replaying = False
        return {
            "wal_records_replayed": float(len(records)),
            "wal_replay_ms": float((perf_counter() - started) * 1000.0),
        }

    def _apply_wal_record(
        self,
        state: ClientIndexState,
        op: str,
        payload: dict[str, Any],
    ) -> None:
        """Re-drives one logical WAL mutation through its public engine verb.

        The record envelope carries the live ``client_id``/``index_id`` so the
        re-invoked verb resolves the same index the original call did (replay is
        otherwise indistinguishable from a fresh public call, which is what keeps
        the recovered state byte-identical to what produced the log).
        """

        context = EngineRequestContext(
            client_id=str(payload.get("client_id", state.client_id)),
            now=datetime.now(tz=UTC),
        )
        index_id = str(payload.get("index_id", state.index_id))
        if op == "upsert_documents":
            documents = tuple(
                EngineDocument(
                    document_id=str(item["document_id"]),
                    text=str(item["text"]),
                    metadata={str(k): str(v) for k, v in dict(item.get("metadata", {})).items()},
                )
                for item in payload.get("documents", [])
            )
            response = self.upsert_documents(
                context=context, documents=documents, index_id=index_id
            )
        elif op == "upsert_vectors":
            vectors = tuple(
                EngineVectorDocument(
                    document_id=str(item["document_id"]),
                    vector=tuple(float(value) for value in item["vector"]),
                    metadata={str(k): str(v) for k, v in dict(item.get("metadata", {})).items()},
                    text=(None if item.get("text") is None else str(item["text"])),
                )
                for item in payload.get("vectors", [])
            )
            response = self.upsert_vectors(context=context, vectors=vectors, index_id=index_id)
        elif op == "delete_documents":
            document_ids = tuple(str(value) for value in payload.get("document_ids", []))
            response = self.delete_documents(
                context=context, document_ids=document_ids, index_id=index_id
            )
        elif op == "update_document_payload":
            response = self.update_document_payload(
                context=context,
                document_id=str(payload["document_id"]),
                metadata=(
                    None
                    if payload.get("metadata") is None
                    else {str(k): str(v) for k, v in dict(payload["metadata"]).items()}
                ),
                text=(None if payload.get("text") is None else str(payload["text"])),
                clear_text=bool(payload.get("clear_text", False)),
                index_id=index_id,
            )
        else:
            raise RuntimeError(f"unknown WAL record op during replay: {op!r}")
        if isinstance(response, EngineResponse) and int(response.status_code) >= 400:
            raise RuntimeError(
                f"WAL replay of {op!r} failed: {response.body.get('error', 'engine error')}"
            )

    def _persist_state(self, state: ClientIndexState) -> None:
        """Persists one mutation, dispatching on the configured commit mode.

        In the default ``wal`` mode this appends the in-flight logical mutation to the index WAL
        (one framed ``write`` + an optional fsync) and checkpoints into a
        generation only when the backlog crosses a threshold — so the common
        single-add commit avoids the multi-file generation publish entirely. WAL
        replay on open re-drives the verb, so ``_persist_state`` is a no-op while
        replaying (the in-memory state is already being rebuilt). In
        ``generation`` mode, this publishes a new crash-atomic, MVCC-readable
        generation on every mutation (see :meth:`_persist_generation`).
        """

        if self.persistence_dir is None:
            return
        self._require_writable()
        if self._commit_mode == CommitMode.WAL and not self._wal_replaying:
            self._append_wal_record(state)
            return
        if self._wal_replaying:
            # Replaying a WAL record only rebuilds in-memory state; the durable
            # base is the committed generation already on disk plus the WAL itself.
            return
        self._persist_generation(state)

    def _persist_generation(self, state: ClientIndexState) -> None:
        """Commits one index generation atomically via its root commit manifest.

        Each commit writes its durable artifacts under the per-index
        ``<key>.gen/`` directory keyed by base epoch — the JSON/vector bases plus
        their delta journals, and (when ``store_text`` is on) the raw-text base +
        ``.txd`` journal — then atomically swaps the ``<key>.commit.json``
        pointer. That swap is the single commit point, so a crash leaves the
        previously committed generation (text included) fully intact.
        """

        if self.persistence_dir is None:
            return
        self._require_writable()
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        generation = self._index_generations.get(state.index_key, 0)
        pending_documents = self._pending_state_journal_documents.pop(state.index_key, None)
        pending_tvim = self._pending_tvim_deltas.pop(state.index_key, None)
        if state.chunks:
            self._commit_direct_route(
                state,
                generation=generation,
                pending_documents=pending_documents,
                pending_tvim=pending_tvim,
            )
        else:
            self._commit_empty_index(state, generation=generation)

    def _write_state_json(self, state: ClientIndexState, *, path: Path, generation: int) -> int:
        """Atomically writes the full JSON state snapshot and returns its byte size."""

        temporary_path = path.with_name(path.name + ".tmp")
        temporary_path.write_text(
            json.dumps(
                _state_to_payload(
                    state,
                    columnar_generation=generation,
                    route_profile=_client_route_profile(self.security, self.route_policy),
                ),
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        durable_replace(temporary_path, path, fsync=self._fsync_on_commit)
        return int(path.stat().st_size)

    def _record_pending_state_journal_documents(
        self,
        state: ClientIndexState,
        *,
        upserted_document_ids: tuple[str, ...] = (),
        deleted_document_ids: tuple[str, ...] = (),
    ) -> None:
        """Accumulates document changes for the next journal append and lexical fold.

        Later mentions win within one batch window: an upsert clears a prior
        pending delete for the same document and vice versa, so replay (and the
        next lexical fold) applies only the final outcome. The direct-route
        journal is tracked only on direct-route states (standard cascade is
        never journaled); the lexical changed-document mirror is tracked whenever
        a lexical/hybrid index could be served, independent of the storage route.
        """

        if _state_uses_direct_turbovec(state):
            pending = self._pending_state_journal_documents.setdefault(
                state.index_key, {"upserted": {}, "deleted": {}}
            )
            for document_id in upserted_document_ids:
                pending["deleted"].pop(document_id, None)
                pending["upserted"][document_id] = None
            for document_id in deleted_document_ids:
                pending["upserted"].pop(document_id, None)
                pending["deleted"][document_id] = None

        if not (self.lexical_index_enabled or self.raw_text_storage_enabled):
            return
        self._record_pending_lexical_documents(
            state,
            upserted_document_ids=upserted_document_ids,
            deleted_document_ids=deleted_document_ids,
        )

    def _record_pending_lexical_documents(
        self,
        state: ClientIndexState,
        *,
        upserted_document_ids: tuple[str, ...],
        deleted_document_ids: tuple[str, ...],
    ) -> None:
        """Mirrors changed-document ids into the O(changed) lexical fold sets.

        Same later-wins reconciliation as the journal accumulator: an upsert id
        clears a pending lexical delete and joins ``"upserted"``; a delete id
        clears a pending lexical upsert and joins ``"deleted"``. The combined set
        size is then bounded by ``max(_LEXICAL_REBUILD_FLOOR, fraction of the
        corpus)``; once it is exceeded the index key is marked for a full rebuild
        and the pending sets are cleared, so a long run of writes with no
        intervening hybrid query cannot grow the sets without bound.
        """

        key = state.index_key
        if key in self._lexical_full_rebuild:
            # Already destined for a full rebuild; tracking individual documents
            # would be wasted work (and bounded-out again immediately).
            return
        pending = self._pending_lexical_documents.setdefault(
            key, {"upserted": set(), "deleted": set()}
        )
        upserted = pending["upserted"]
        deleted = pending["deleted"]
        for document_id in upserted_document_ids:
            document_id = str(document_id)
            deleted.discard(document_id)
            upserted.add(document_id)
        for document_id in deleted_document_ids:
            document_id = str(document_id)
            upserted.discard(document_id)
            deleted.add(document_id)
        bound = max(
            _LEXICAL_REBUILD_FLOOR,
            int(_LEXICAL_INCREMENTAL_MAX_FRACTION * max(len(state.document_chunk_ids), 1)),
        )
        if len(upserted) + len(deleted) > bound:
            self._lexical_full_rebuild.add(key)
            self._pending_lexical_documents.pop(key, None)

    def _delta_append_ok(
        self,
        key: str,
        *,
        current_epoch: int | None,
        serving: Any,
        pending_documents: dict[str, dict[str, None]] | None,
        pending_tvim: dict[str, Any] | None,
    ) -> bool:
        """Returns whether this mutation can append a delta onto the live epoch.

        Delta-append needs the ``auto`` policy, a live base epoch whose
        journaled artifacts are present and patched-API-capable, an actual
        change to journal, and neither store due for compaction. Anything else
        (the ``off`` policy, a cold build, a compaction threshold) falls back to
        a fresh base at a new epoch. ``pending_tvim is not None`` proves an
        incremental sync ran this cycle, so the live index is base+deltas
        consistent; a cold rebuild pops the entry and must rewrite the base.
        """

        if self.tvim_delta_persistence_policy != TvimDeltaPersistencePolicy.AUTO:
            return False
        if current_epoch is None or pending_tvim is None:
            return False
        tvim_changed = bool(pending_tvim["upserted"] or pending_tvim["removed"])
        documents_changed = pending_documents is not None and bool(
            pending_documents["upserted"] or pending_documents["deleted"]
        )
        if not (tvim_changed or documents_changed):
            return False
        base_json = self._epoch_json_path(key, current_epoch)
        base_tvim = self._epoch_tvim_path(key, current_epoch)
        journal = StateJournalStore(base_json)
        tvim_store = TvimDeltaStore(base_tvim)
        # The raw-text journal shares the epoch; a full text base also forces a
        # base rewrite so the text deltas are compacted alongside the index.
        text_compaction_due = False
        if self.raw_text_storage_enabled:
            text_store = self._text_store(key, current_epoch)
            text_compaction_due = text_store.has_manifest() and text_store.should_compact()
        # The lexical-postings journal shares the epoch too; compact it alongside
        # the index when its delta backlog is due.
        lexical_compaction_due = False
        if self.lexical_index_enabled:
            lexical_store = self._lexical_index_store(key, current_epoch)
            lexical_compaction_due = (
                lexical_store.has_manifest() and lexical_store.should_compact()
            )
        return (
            base_json.exists()
            and journal.has_manifest()
            and base_tvim.exists()
            and tvim_store.has_manifest()
            and turbovec_delta_api_available(serving.index)
            and not tvim_store.should_compact()
            and not journal.should_compact()
            and not text_compaction_due
            and not lexical_compaction_due
        )

    def _commit_direct_route(
        self,
        state: ClientIndexState,
        *,
        generation: int,
        pending_documents: dict[str, dict[str, None]] | None,
        pending_tvim: dict[str, Any] | None,
    ) -> None:
        """Persists a non-empty direct-route index and seals it via the root manifest.

        A delta-eligible mutation appends one ``.jsd`` document delta and one
        ``.tvd`` vector delta onto the live base epoch; otherwise a base rewrite
        writes a fresh base at a NEW epoch (cold build, compaction, or the
        non-journaled ``off`` policy). Either way the commit is sealed by the
        atomic ``<key>.commit.json`` swap, after which base rewrites GC the
        superseded epochs and any migrated legacy artifacts.
        """

        key = state.index_key
        fsync = self._fsync_on_commit
        try:
            serving = self._turbovec_index_for_state(state)
            current_epoch = self._base_epochs.get(key)
            if self._delta_append_ok(
                key,
                current_epoch=current_epoch,
                serving=serving,
                pending_documents=pending_documents,
                pending_tvim=pending_tvim,
            ):
                assert current_epoch is not None and pending_tvim is not None
                base_json = self._epoch_json_path(key, current_epoch)
                base_tvim = self._epoch_tvim_path(key, current_epoch)
                journal = StateJournalStore(base_json, fsync=fsync)
                tvim_store = TvimDeltaStore(base_tvim, fsync=fsync)
                json_write = journal.append_delta(
                    upserted_documents=[
                        _state_journal_document_entry(state, document_id)
                        for document_id in (pending_documents or {}).get("upserted", {})
                    ],
                    deleted_document_ids=list((pending_documents or {}).get("deleted", {})),
                    state_header=_state_header_payload(
                        state,
                        columnar_generation=generation,
                        route_profile=_client_route_profile(self.security, self.route_policy),
                    ),
                    document_count_after=len(state.document_hashes),
                    chunk_count_after=len(state.chunks),
                    generation=generation,
                )
                # Journal ids in their exact live mutation order (deduped,
                # order-preserving). swap_remove makes slot layout depend on
                # removal order, and exactly-tied scores (duplicate chunk text
                # quantizing to identical codes) are tie-broken by slot order —
                # sorted replay produced a different tie order than the live
                # index on GovReport-scale corpora.
                tvim_write = tvim_store.append_delta(
                    serving.index,
                    upsert_stable_ids=np.asarray(
                        list(dict.fromkeys(pending_tvim["upserted"])), dtype=np.uint64
                    ),
                    removed_stable_ids=np.asarray(
                        list(dict.fromkeys(pending_tvim["removed"])), dtype=np.uint64
                    ),
                    generation=generation,
                )
                text_manifest = self._journal_text(
                    state,
                    epoch=current_epoch,
                    pending_documents=pending_documents,
                    base_rewrite=False,
                )
                lexical_manifest = self._journal_lexical(
                    state,
                    epoch=current_epoch,
                    pending_documents=pending_documents,
                    base_rewrite=False,
                )
                self._write_root_commit_manifest(
                    state,
                    epoch=current_epoch,
                    generation=generation,
                    journal=journal,
                    tvim_store=tvim_store,
                    text_manifest=text_manifest,
                    lexical_manifest=lexical_manifest,
                )
                self._tvim_persist_telemetry[key] = {
                    "tvim_persist_mode": "delta_append",
                    "tvim_persist_bytes": float(tvim_write.file_bytes),
                    "tvim_persist_write_ms": float(tvim_write.write_ms),
                    **tvim_store.storage_file_bytes(),
                    "json_persist_mode": "delta_append",
                    "json_persist_bytes": float(json_write.file_bytes),
                    "json_persist_write_ms": float(json_write.write_ms),
                    **journal.storage_file_bytes(),
                }
                return
            # Base rewrite at a NEW epoch (cold build, compaction, or off policy).
            epoch = generation
            base_json = self._epoch_json_path(key, epoch)
            base_tvim = self._epoch_tvim_path(key, epoch)
            base_json.parent.mkdir(parents=True, exist_ok=True)
            journal = StateJournalStore(base_json, fsync=fsync)
            tvim_store = TvimDeltaStore(base_tvim, fsync=fsync)
            json_started = perf_counter()
            json_bytes = float(self._write_state_json(state, path=base_json, generation=generation))
            journal.record_base(
                document_count=len(state.document_hashes),
                chunk_count=len(state.chunks),
            )
            json_write_ms = float((perf_counter() - json_started) * 1000.0)
            tvim_write = tvim_store.persist_base(serving.index)
            self._base_epochs[key] = epoch
            text_manifest = self._journal_text(
                state, epoch=epoch, pending_documents=None, base_rewrite=True
            )
            lexical_manifest = self._journal_lexical(
                state, epoch=epoch, pending_documents=None, base_rewrite=True
            )
            self._write_root_commit_manifest(
                state,
                epoch=epoch,
                generation=generation,
                journal=journal,
                tvim_store=tvim_store,
                text_manifest=text_manifest,
                lexical_manifest=lexical_manifest,
            )
            self._gc_after_base_rewrite(key, live_epoch=epoch)
            self._gc_legacy_files(key)
            self._tvim_persist_telemetry[key] = {
                "tvim_persist_mode": "base_rewrite",
                "tvim_persist_bytes": float(tvim_write.file_bytes),
                "tvim_persist_write_ms": float(tvim_write.write_ms),
                **tvim_store.storage_file_bytes(),
                "json_persist_mode": "base_rewrite",
                "json_persist_bytes": json_bytes,
                "json_persist_write_ms": json_write_ms,
                **journal.storage_file_bytes(),
            }
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("TurboVec compact snapshot persistence failed") from exc

    def _commit_empty_index(
        self,
        state: ClientIndexState,
        *,
        generation: int,
    ) -> None:
        """Commits an empty index: a base-only JSON state at a new epoch, no vectors."""

        key = state.index_key
        epoch = generation
        try:
            base_json = self._epoch_json_path(key, epoch)
            base_json.parent.mkdir(parents=True, exist_ok=True)
            journal = StateJournalStore(base_json, fsync=self._fsync_on_commit)
            self._write_state_json(state, path=base_json, generation=generation)
            journal.record_base(document_count=len(state.document_hashes), chunk_count=0)
            self._base_epochs[key] = epoch
            # An empty index holds no documents, so it holds no raw text or
            # lexical postings either.
            text_manifest = self._journal_text(
                state, epoch=epoch, pending_documents=None, base_rewrite=True
            )
            lexical_manifest = self._journal_lexical(
                state, epoch=epoch, pending_documents=None, base_rewrite=True
            )
            self._write_root_commit_manifest(
                state,
                epoch=epoch,
                generation=generation,
                journal=journal,
                tvim_store=None,
                text_manifest=text_manifest,
                lexical_manifest=lexical_manifest,
            )
            self._gc_after_base_rewrite(key, live_epoch=epoch)
            self._gc_legacy_files(key)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("empty index snapshot persistence failed") from exc
        self._tvim_persist_telemetry[key] = {
            "tvim_persist_mode": "empty",
            "json_persist_mode": "base_rewrite",
        }

    def _write_root_commit_manifest(
        self,
        state: ClientIndexState,
        *,
        epoch: int,
        generation: int,
        journal: StateJournalStore,
        tvim_store: TvimDeltaStore | None,
        text_manifest: dict[str, Any] | None,
        lexical_manifest: dict[str, Any] | None = None,
    ) -> None:
        """Seals one committed generation by atomically swapping the root manifest.

        ``text_manifest`` is the raw-text journal manifest (base + ``.txd``
        deltas) written for this same generation, so text is pinned by — and
        rolls back with — the committed generation; ``None`` when raw-text
        storage is off or the index holds no text. ``lexical_manifest`` is the
        analogous lexical-postings journal manifest (base + ``.lxd`` deltas),
        ``None`` when lexical-index persistence is off or the index holds no
        tokens. Both pin their sidecar to this generation, so each commits and
        rolls back atomically with it.
        """

        key = state.index_key
        body = build_commit_body(
            index_key=key,
            generation=generation,
            base_epoch=epoch,
            document_count=len(state.document_hashes),
            chunk_count=len(state.chunks),
            json_manifest=journal.current_manifest(),
            tvim_manifest=tvim_store.current_manifest() if tvim_store is not None else None,
            tvim_present=tvim_store is not None,
            tvtext_manifest=text_manifest,
            tvlex_manifest=lexical_manifest,
        )
        write_commit_manifest(self._commit_manifest_path(key), body, fsync=self._fsync_on_commit)

    def _gc_after_base_rewrite(self, key: str, *, live_epoch: int) -> None:
        """Removes superseded base epochs, keeping the most recent few for readers."""

        epochs = sorted(list_base_epochs(self.persistence_dir, key), reverse=True)
        keep = set(epochs[:DEFAULT_EPOCHS_RETAINED]) | {live_epoch}
        for epoch in epochs:
            if epoch not in keep:
                self._remove_epoch_artifacts(key, epoch)

    def _remove_epoch_artifacts(self, key: str, epoch: int) -> None:
        """Deletes one base epoch's JSON/TurboVec/text/lexical bases and journal dirs."""

        base_json = self._epoch_json_path(key, epoch)
        base_tvim = self._epoch_tvim_path(key, epoch)
        shutil.rmtree(StateJournalStore(base_json).journal_dir, ignore_errors=True)
        shutil.rmtree(TvimDeltaStore(base_tvim).delta_dir, ignore_errors=True)
        self._text_store(key, epoch).remove_all()
        self._lexical_index_store(key, epoch).remove_all()
        base_json.unlink(missing_ok=True)
        base_tvim.unlink(missing_ok=True)

    def _sweep_pre_journal_text_sidecars(self, key: str) -> None:
        """Removes stray pre-journal single-file ``text-g*.tvtext`` sidecars.

        The pre-journal Stage 2 layout wrote one ``text-g<gen>.tvtext`` file per
        commit; the journaled layout supersedes them with ``g<epoch>.tvtext`` +
        ``.txd`` deltas. Once an index has migrated to the journal, these stray
        single files are dead weight, so drop them.
        """

        directory = generation_dir(self.persistence_dir, key)
        if not directory.is_dir():
            return
        for path in directory.glob("text-g*.tvtext"):
            path.unlink(missing_ok=True)

    def _gc_legacy_files(self, key: str) -> None:
        """Removes pre-commit-manifest top-level artifacts once migrated (no-op if none).

        Includes the legacy top-level ``<key>.tvtext``: raw text now lives in the
        per-index ``<key>.gen/`` directory governed by the commit manifest.
        """

        legacy_json = self._state_path(key)
        legacy_tvim = self._turbovec_snapshot_path(key)
        legacy_text = self._legacy_text_sidecar_path(key)
        if not (legacy_json.exists() or legacy_tvim.exists() or legacy_text.exists()):
            return
        shutil.rmtree(StateJournalStore(legacy_json).journal_dir, ignore_errors=True)
        shutil.rmtree(TvimDeltaStore(legacy_tvim).delta_dir, ignore_errors=True)
        legacy_json.unlink(missing_ok=True)
        legacy_tvim.unlink(missing_ok=True)
        legacy_text.unlink(missing_ok=True)

    def _load_persisted_indexes(self) -> None:
        """Loads persisted indexes, each from its atomic root commit manifest.

        Stage-2 indexes load the single consistent generation named by
        ``<key>.commit.json`` (so a lock-free reader never sees a torn cross-file
        mix). A pre-commit-manifest top-level ``<key>.json`` (v0.1.x) is loaded
        directly as a fallback and migrates to the new layout on its next write.
        """

        if self.persistence_dir is None:
            return
        loaded_keys: set[str] = set()
        for commit_path in sorted(self.persistence_dir.glob(f"*{COMMIT_MANIFEST_SUFFIX}")):
            key = commit_path.name[: -len(COMMIT_MANIFEST_SUFFIX)]
            body = read_commit_manifest(commit_path)
            if body is None:
                continue
            if not self._read_only:
                # Writer recovery: heal the per-store manifests back to the
                # committed root (dropping any segment a crashed commit left
                # uncommitted) and GC superseded epochs + migrated legacy files.
                self._recover_to_commit(key, body)
            self._load_index_from_commit(key, body)
            loaded_keys.add(key)
        for path in sorted(self.persistence_dir.glob("*.json")):
            if is_commit_manifest_name(path.name) or path.stem in loaded_keys:
                continue
            self._load_legacy_index(path)
        # WAL recovery (writer handles only). The durable base on disk is always a
        # committed generation; this replays any <key>.wal tail on top so a
        # WAL-committed-but-not-yet-checkpointed mutation survives the reopen, then
        # folds it into a fresh generation so the handle always lands on a clean,
        # consistent generation (and WAL mode starts a fresh log from there). A
        # torn trailing record (crash mid-append) was already dropped by the store.
        # This runs regardless of the configured commit mode and of whether the
        # prior writer used WAL or generation; with no WAL present (the common
        # case) it is one cheap stat per index and a no-op. Read-only handles never
        # replay (the WAL is the writer's private log).
        if not self._read_only:
            self._recover_wals()

    def _recover_wals(self) -> None:
        """Replays each index's WAL tail, then folds it into a fresh generation.

        The durable base loaded above is always a committed generation; any
        ``<key>.wal`` holds only mutations a prior writer logged but had not yet
        checkpointed (a clean close folds and truncates the WAL, so one is present
        only after an unclean shutdown). Replay re-drives the same engine verbs to
        rebuild the in-memory state, then the recovered WAL is always folded into a
        fresh generation and truncated, so every open lands on a clean, consistent
        generation regardless of commit mode: WAL mode then starts a new log from
        that generation, and generation mode is simply left with no stray log. A
        torn trailing record was dropped by the store, so exactly the durably
        logged mutations are recovered.
        """

        for index_key in list(self._indexes):
            state = self._indexes.get(index_key)
            if state is None:
                continue
            store = self._wal_store_for(index_key)
            if not store.exists():
                continue
            stats = self._replay_wal(state)
            if stats.get("wal_records_replayed"):
                self._state_load_telemetry.setdefault(index_key, {}).update(stats)
                # Normalize to a clean generation on open: fold the recovered WAL
                # into a fresh generation and truncate it, in either commit mode.
                self._checkpoint_wal(state)
            else:
                # A WAL with no intact records (only a torn partial frame): nothing
                # to recover, but drop the stray bytes so a later append cannot turn
                # them into interior corruption.
                store.truncate()

    def _load_index_from_commit(self, key: str, body: dict[str, Any]) -> None:
        """Loads one index from the consistent generation named by its root manifest."""

        epoch = int(body["base_epoch"])
        base_json = self._epoch_json_path(key, epoch)
        payload = json.loads(base_json.read_text(encoding="utf-8"))
        journal = StateJournalStore(base_json)
        json_manifest = body.get("json")
        journal_stats: dict[str, float] | None = None
        if json_manifest is not None:
            journal.validate_base_checksum(manifest=json_manifest)
            journal_stats = journal.replay_onto_payload(payload, manifest=json_manifest)
        state = _state_from_payload(payload)
        if journal_stats is not None:
            self._state_load_telemetry[state.index_key] = dict(journal_stats)
        generation = int(body.get("generation", payload.get("columnar_generation", 0)))
        # The raw-text journal is pinned by the root manifest at this same
        # generation, so a torn commit's uncommitted text is never loaded.
        tvtext = body.get("tvtext")
        if isinstance(tvtext, dict) and "base" in tvtext:
            self._load_document_text(state, epoch=epoch, text_manifest=tvtext)
        elif isinstance(tvtext, dict) and tvtext.get("present"):
            # Pre-journal single-file layout: load the per-generation sidecar
            # once; the index migrates it into the text journal on its next write.
            self._load_pre_journal_text(
                state, generation=generation, expected_sha=tvtext.get("sha256")
            )
        # The lexical-postings journal is pinned by the root at the same epoch, so
        # a torn commit's uncommitted postings are never loaded.
        tvlex = body.get("tvlex")
        if isinstance(tvlex, dict) and "base" in tvlex:
            self._load_lexical(state, epoch=epoch, lexical_manifest=tvlex)
        self._indexes[state.index_key] = state
        self._index_generations[state.index_key] = generation
        self._base_epochs[state.index_key] = epoch
        if body.get("tvim_present"):
            self._load_turbovec_snapshot(
                state,
                generation=generation,
                path=self._epoch_tvim_path(key, epoch),
                tvim_manifest=body.get("tvim"),
            )
        else:
            # Empty index: no vector base — build an empty serving index.
            self._load_turbovec_snapshot(
                state, generation=generation, path=self._epoch_tvim_path(key, epoch)
            )
        self._emit_state_metric("index_loaded", state, {"chunk_count": len(state.chunks)})

    def _load_legacy_index(self, path: Path) -> None:
        """Loads one pre-commit-manifest top-level snapshot (v0.1.x layout)."""

        payload = json.loads(path.read_text(encoding="utf-8"))
        journal = StateJournalStore(path)
        journal_stats: dict[str, float] | None = None
        if journal.has_manifest():
            journal.validate_base_checksum()
            journal_stats = journal.replay_onto_payload(payload)
        state = _state_from_payload(payload)
        if journal_stats is not None:
            self._state_load_telemetry[state.index_key] = dict(journal_stats)
        if self.raw_text_storage_enabled:
            # v0.1.x kept raw text in a single top-level ``<key>.tvtext`` file;
            # load it once, then the next write migrates it into the journal.
            state.document_text = read_legacy_text_sidecar(
                self._legacy_text_sidecar_path(state.index_key)
            )
        self._indexes[state.index_key] = state
        generation = int(payload.get("columnar_generation", 0))
        self._index_generations[state.index_key] = generation
        self._load_turbovec_snapshot(
            state, generation=generation, path=self._turbovec_snapshot_path(state.index_key)
        )
        self._emit_state_metric("index_loaded", state, {"chunk_count": len(state.chunks)})

    def _recover_to_commit(self, key: str, body: dict[str, Any]) -> None:
        """Reconciles on-disk state to the committed root before a writer loads it.

        Rewrites each per-store manifest to the committed copy (dropping any
        segment a crashed commit appended but never committed), re-points the
        live base epoch, and GCs superseded epochs plus any migrated legacy
        artifacts. This turns a torn commit into a clean roll-back to the last
        good generation instead of a fail-closed open.
        """

        epoch = int(body["base_epoch"])
        json_manifest = body.get("json")
        if json_manifest is not None:
            StateJournalStore(
                self._epoch_json_path(key, epoch), fsync=self._fsync_on_commit
            ).restore_manifest(json_manifest)
        if body.get("tvim_present") and body.get("tvim") is not None:
            TvimDeltaStore(
                self._epoch_tvim_path(key, epoch), fsync=self._fsync_on_commit
            ).restore_manifest(body["tvim"])
        tvtext = body.get("tvtext")
        if isinstance(tvtext, dict) and "base" in tvtext:
            # Heal the text journal back to the committed manifest (dropping any
            # .txd segment a crashed commit appended but never committed).
            self._text_store(key, epoch).restore_manifest(tvtext)
            self._sweep_pre_journal_text_sidecars(key)
        tvlex = body.get("tvlex")
        if isinstance(tvlex, dict) and "base" in tvlex:
            # Heal the lexical-postings journal back to the committed manifest
            # (dropping any .lxd segment a crashed commit never committed).
            self._lexical_index_store(key, epoch).restore_manifest(tvlex)
        self._base_epochs[key] = epoch
        self._gc_after_base_rewrite(key, live_epoch=epoch)
        self._gc_legacy_files(key)

    def _state_path(self, client_id_hash: str) -> Path:
        """Returns the local snapshot path for one hashed index key."""

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return self.persistence_dir / f"{client_id_hash}.json"

    def _turbovec_snapshot_path(self, client_id_hash: str) -> Path:
        """Returns the legacy (pre-commit-manifest) TurboVec sidecar path."""

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return self.persistence_dir / f"{client_id_hash}.tvim"

    def _commit_manifest_path(self, client_id_hash: str) -> Path:
        """Returns the root commit-manifest pointer path for one index key."""

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return commit_manifest_path(self.persistence_dir, client_id_hash)

    def _epoch_json_path(self, client_id_hash: str, epoch: int) -> Path:
        """Returns the JSON state base path for one index key and base epoch."""

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return base_json_path(self.persistence_dir, client_id_hash, epoch)

    def _epoch_tvim_path(self, client_id_hash: str, epoch: int) -> Path:
        """Returns the TurboVec base path for one index key and base epoch."""

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return base_tvim_path(self.persistence_dir, client_id_hash, epoch)

    def _live_base_paths(self, client_id_hash: str) -> tuple[Path, Path]:
        """Returns the (JSON, TurboVec) base paths for the live committed epoch.

        Falls back to the legacy top-level paths for an index that has not yet
        migrated to the commit-manifest layout.
        """

        epoch = self._base_epochs.get(client_id_hash)
        if epoch is None:
            return self._state_path(client_id_hash), self._turbovec_snapshot_path(client_id_hash)
        return self._epoch_json_path(client_id_hash, epoch), self._epoch_tvim_path(
            client_id_hash, epoch
        )

    def _text_base_path(self, client_id_hash: str, epoch: int) -> Path:
        """Returns the raw-text base path for one index key and base epoch.

        The raw-text store is governed by the same root manifest as the index:
        each base epoch holds a ``<key>.gen/g<epoch>.tvtext`` full map plus a
        ``.txd`` delta journal, and the root pins its manifest, so a torn commit
        rolls text back to the committed generation (rather than leaving an
        uncommitted overwrite visible to ``get``).
        """

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return base_tvtext_path(self.persistence_dir, client_id_hash, epoch)

    def _text_store(self, client_id_hash: str, epoch: int) -> DocumentTextStore:
        """Returns the journaled raw-text store for one index key and base epoch."""

        return DocumentTextStore(
            self._text_base_path(client_id_hash, epoch), fsync=self._fsync_on_commit
        )

    def _lexical_base_path(self, client_id_hash: str, epoch: int) -> Path:
        """Returns the lexical-index base path for one index key and base epoch.

        The lexical-index store is governed by the same root manifest as the
        index: each base epoch holds a ``<key>.gen/g<epoch>.tvlex`` full
        ``document_id -> token lists`` map plus a ``.lxd`` delta journal, and the
        root pins its manifest, so a torn commit rolls the postings back to the
        committed generation (rather than leaving an uncommitted overwrite).
        """

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return base_tvlex_path(self.persistence_dir, client_id_hash, epoch)

    def _lexical_index_store(self, client_id_hash: str, epoch: int) -> LexicalIndexStore:
        """Returns the journaled lexical-index store for one index key and base epoch."""

        return LexicalIndexStore(
            self._lexical_base_path(client_id_hash, epoch), fsync=self._fsync_on_commit
        )

    def _legacy_text_sidecar_path(self, client_id_hash: str) -> Path:
        """Returns the legacy (pre-commit-manifest) top-level raw-text sidecar path."""

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return self.persistence_dir / f"{client_id_hash}.tvtext"

    @property
    def raw_text_storage_enabled(self) -> bool:
        """Returns whether this engine retains raw document text for retrieval.

        Off by default; turning it on (``EngineSecurityConfig.allow_raw_result_text``)
        is the explicit opt-in documented in the README/architecture notes. It
        only affects the dedicated ``.tvtext`` sidecar and the
        ``get_document_text`` retrieval path — never telemetry, audit, the
        redacted ``.json`` snapshot, or query result rows.
        """

        return bool(self.security.allow_raw_result_text)

    @property
    def lexical_index_enabled(self) -> bool:
        """Returns whether this engine persists the lexical (BM25) postings.

        Off by default; turning it on (``EngineSecurityConfig.persist_lexical_index``)
        is the explicit opt-in documented in the README/architecture notes. It
        captures each document's per-chunk tokens at ingest time and keeps them
        in a dedicated ``.tvlex`` sidecar (base + ``.lxd`` journal), so hybrid and
        lexical search survive a reopen without re-tokenizing or even retaining
        the raw text. Like the raw-text sidecar, it never touches telemetry,
        audit, the redacted ``.json`` snapshot, or query result rows.
        """

        return bool(self.security.persist_lexical_index)

    def _capture_document_text(
        self,
        state: ClientIndexState,
        documents: tuple[EngineDocument, ...],
    ) -> None:
        """Records raw document text for retrieval when raw-text storage is on."""

        if not self.raw_text_storage_enabled:
            return
        for document in documents:
            state.document_text[document.document_id] = document.text

    def _capture_vector_document_text(
        self,
        state: ClientIndexState,
        documents: tuple[EngineVectorDocument, ...],
    ) -> None:
        """Records optional vector-in payload text without embedding it."""

        if not self.raw_text_storage_enabled:
            return
        for document in documents:
            if document.text is None:
                state.document_text.pop(document.document_id, None)
            else:
                state.document_text[document.document_id] = document.text

    def _capture_lexical_tokens(
        self,
        state: ClientIndexState,
        documents: tuple[EngineDocument, ...],
    ) -> None:
        """Records each document's per-chunk tokens when lexical-index persistence is on.

        Tokens are captured here, at ingest time, while the raw text is in hand,
        so the persisted lexical index is genuinely independent of
        ``store_text``: it is built from these tokens on reopen with no raw-text
        dependency and no re-tokenization. The per-chunk token lists align with
        the document's recorded ``document_chunk_ids`` because both come from the
        same deterministic chunker.
        """

        if not self.lexical_index_enabled:
            return
        for document in documents:
            state.document_tokens[document.document_id] = [
                tokenize(chunk)
                for chunk in chunk_text(document.text, self.chunk_character_limit)
            ]

    def _journal_text(
        self,
        state: ClientIndexState,
        *,
        epoch: int,
        pending_documents: dict[str, dict[str, None]] | None,
        base_rewrite: bool,
    ) -> dict[str, Any] | None:
        """Journals raw text for this commit; returns the manifest the root pins.

        On a base rewrite the full ``document_id -> text`` map is written as a
        fresh base at this epoch; on a delta append only the upserted texts and
        deleted ids of this batch are journaled (O(changed)). Returns the text
        journal manifest for the root to embed, or ``None`` when raw-text storage
        is disabled or the index holds no text — so a reader skips text entirely.
        """

        if self.persistence_dir is None or not self.raw_text_storage_enabled:
            return None
        store = self._text_store(state.index_key, epoch)
        if base_rewrite or not store.has_manifest():
            if not state.document_text:
                return None
            store.record_base(state.document_text)
        else:
            upserted = {
                document_id: state.document_text[document_id]
                for document_id in (pending_documents or {}).get("upserted", {})
                if document_id in state.document_text
            }
            deleted = list((pending_documents or {}).get("deleted", {}))
            store.append_delta(
                upserted=upserted,
                deleted=deleted,
                document_count_after=len(state.document_text),
            )
        return store.current_manifest()

    def _journal_lexical(
        self,
        state: ClientIndexState,
        *,
        epoch: int,
        pending_documents: dict[str, dict[str, None]] | None,
        base_rewrite: bool,
    ) -> dict[str, Any] | None:
        """Journals the lexical postings for this commit; returns the manifest the root pins.

        The exact parallel of :meth:`_journal_text` over the per-chunk token
        lists captured at ingest. On a base rewrite the full
        ``document_id -> token lists`` map is written as a fresh base at this
        epoch; on a delta append only the upserted token lists and deleted ids of
        this batch are journaled (O(changed)). Returns the lexical journal
        manifest for the root to embed, or ``None`` when lexical-index
        persistence is disabled or the index holds no tokens.
        """

        if self.persistence_dir is None or not self.lexical_index_enabled:
            return None
        store = self._lexical_index_store(state.index_key, epoch)
        if base_rewrite or not store.has_manifest():
            if not state.document_tokens:
                return None
            store.record_base(state.document_tokens)
        else:
            upserted = {
                document_id: state.document_tokens[document_id]
                for document_id in (pending_documents or {}).get("upserted", {})
                if document_id in state.document_tokens
            }
            deleted = list((pending_documents or {}).get("deleted", {}))
            store.append_delta(
                upserted=upserted,
                deleted=deleted,
                document_count_after=len(state.document_tokens),
            )
        return store.current_manifest()

    def _load_document_text(
        self,
        state: ClientIndexState,
        *,
        epoch: int,
        text_manifest: dict[str, Any] | None,
    ) -> None:
        """Loads the committed raw-text journal into state when present.

        ``text_manifest`` is the journal manifest the root commit pins (base +
        ``.txd`` deltas); ``None`` means the committed generation has no text.
        Fails closed if a pinned base/segment is missing or fails its checksum,
        rather than serving stale or partial text.
        """

        if self.persistence_dir is None or not self.raw_text_storage_enabled:
            return
        if not text_manifest:
            return
        state.document_text = self._text_store(state.index_key, epoch).load(manifest=text_manifest)

    def _load_lexical(
        self,
        state: ClientIndexState,
        *,
        epoch: int,
        lexical_manifest: dict[str, Any] | None,
    ) -> None:
        """Loads the committed lexical-postings journal into state when present.

        ``lexical_manifest`` is the journal manifest the root commit pins (base +
        ``.lxd`` deltas); ``None`` means the committed generation has no persisted
        postings. Fails closed if a pinned base/segment is missing or fails its
        checksum, rather than serving a stale or partial index.
        """

        if self.persistence_dir is None or not self.lexical_index_enabled:
            return
        if not lexical_manifest:
            return
        state.document_tokens = self._lexical_index_store(state.index_key, epoch).load(
            manifest=lexical_manifest
        )

    def _load_pre_journal_text(
        self, state: ClientIndexState, *, generation: int, expected_sha: str | None
    ) -> None:
        """Loads a pre-journal single-file ``text-g<gen>.tvtext`` sidecar (compat).

        The pre-journal Stage 2 layout pinned a per-generation single file by its
        file sha. Verify that sha against the committed root, then read it; the
        index migrates this into the text journal on its next write.
        """

        if self.persistence_dir is None or not self.raw_text_storage_enabled:
            return
        path = generation_dir(self.persistence_dir, state.index_key) / f"text-g{generation}.tvtext"
        if not path.is_file():
            if expected_sha is None:
                return
            raise RuntimeError("committed raw-text sidecar is missing")
        if expected_sha is not None and _sha256_file(path) != expected_sha:
            raise RuntimeError("committed raw-text sidecar failed manifest checksum")
        state.document_text = read_legacy_text_sidecar(path)

    def _load_turbovec_snapshot(
        self,
        state: ClientIndexState,
        *,
        generation: int,
        path: Path | None = None,
        tvim_manifest: dict[str, Any] | None = None,
    ) -> None:
        """Loads and validates a direct TurboVec base for one persisted index.

        ``path`` is the resolved base file (the epoch base under the commit
        manifest, or the legacy top-level sidecar); ``tvim_manifest`` overrides
        the on-disk delta manifest so the committed segment set is replayed
        rather than whatever a crashed writer last left on disk.
        """

        if self.persistence_dir is None:
            return
        if path is None:
            path = self._turbovec_snapshot_path(state.index_key)
        if not path.exists():
            if state.chunks:
                raise RuntimeError("direct TurboVec snapshot is required but missing")
            self._turbovec_indexes[state.index_key] = build_turbovec_serving_index(
                (),
                native_dim=state.native_dim,
                bit_width=_turbovec_bit_width_for_state(state),
                generation=generation,
                id_map_index_class=self.turbovec_id_map_index_class,
                progress_label=_direct_turbovec_progress_label(
                    self,
                    state,
                    generation=generation,
                ),
            )
            return
        try:
            store = TvimDeltaStore(path)
            post_load = None
            if tvim_manifest is not None or store.has_manifest():
                store.validate_base_checksum(manifest=tvim_manifest)

                def post_load(
                    raw_index: Any,
                    _store: TvimDeltaStore = store,
                    _manifest: dict[str, Any] | None = tvim_manifest,
                ) -> None:
                    # Replay APIs are only required when segments exist; a
                    # base-only manifest (every auto base rewrite) must load
                    # on unpatched backends too.
                    if _store.has_pending_segments(
                        manifest=_manifest
                    ) and not turbovec_delta_api_available(raw_index):
                        raise RuntimeError(
                            "TurboVec delta manifest present but the loaded backend "
                            "lacks the delta replay APIs"
                        )
                    _store.replay_onto(raw_index, manifest=_manifest)

            loaded = load_turbovec_serving_index(
                path,
                tuple(state.chunks.values()),
                generation=generation,
                id_map_index_class=self.turbovec_id_map_index_class,
                post_load=post_load,
            )
            _validate_direct_turbovec_snapshot(state, loaded)
        except (OSError, RuntimeError, ValueError) as exc:
            self._turbovec_indexes.pop(state.index_key, None)
            raise RuntimeError(f"TurboVec compact snapshot load failed: {exc}") from exc
        self._turbovec_indexes[state.index_key] = loaded

    def _emit(self, event: str, client_id: str, fields: dict[str, Any]) -> None:
        """Emits redacted logs and numeric/bool/string metrics without raw payloads."""

        safe_fields = sanitize_observability_fields(fields)
        event_payload = {"event": event, "client_id_hash": sha256_text(client_id), **safe_fields}
        self._audit_events.append(event_payload)
        self._metrics.append(event_payload)

    def _emit_state_metric(
        self,
        event: str,
        state: ClientIndexState,
        fields: dict[str, Any],
    ) -> None:
        """Emits metrics for persisted-load work when only the client hash is stored."""

        safe_fields = sanitize_observability_fields(fields)
        event_payload = {"event": event, "client_id_hash": state.client_id_hash, **safe_fields}
        self._audit_events.append(event_payload)
        self._metrics.append(event_payload)

    def _error(
        self,
        status_code: int,
        message: str,
        event: str,
        client_id: str,
    ) -> EngineResponse:
        """Builds an error response and records a redacted audit event."""

        self._emit(f"{event}_failed", client_id, {"status_code": status_code})
        return EngineResponse(status_code, {"status": "error", "error": message})


def sha256_text(value: str) -> str:
    """Returns a stable SHA-256 hex digest for a UTF-8 string."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _materialize_query_results(
    query_result: _EngineTurboVecQueryResult,
    *,
    state: ClientIndexState,
    query_filter: dict[str, Any],
    include: tuple[str, ...],
    top_k: int,
) -> list[dict[str, Any]]:
    """Converts direct TurboVec search output into redacted public result rows."""

    candidates = [
        {
            "chunk_id": query_result.index.chunk_ids_by_stable_id[int(stable_id)],
            "document_id": query_result.index.document_ids_by_stable_id[int(stable_id)],
            "score": float(score),
        }
        for stable_id, score in zip(
            query_result.stable_ids,
            query_result.scores,
            strict=True,
        )
    ]
    filtered: list[dict[str, Any]] = []
    matches_filter = _compile_query_filter(query_filter)
    document_metadata = state.document_metadata
    for candidate in candidates:
        document_id = str(candidate["document_id"])
        if not matches_filter(document_id, document_metadata.get(document_id, {})):
            continue
        if QUERY_INCLUDE_METADATA in include:
            candidate["metadata"] = dict(document_metadata.get(document_id, {}))
        filtered.append(candidate)
        if len(filtered) >= top_k:
            break
    return filtered


def _compact_backend_name(state: ClientIndexState) -> str:
    """Returns the safe compact-backend label for stats and audit output."""

    if _state_uses_direct_turbovec(state):
        return "turbovec_idmap"
    return "columnar"


def _state_uses_direct_turbovec(state: ClientIndexState) -> bool:
    """Returns whether a state is served by direct full-dimensional TurboVec."""

    return state.storage_profile == DIRECT_TURBOVEC_STORAGE_PROFILE


def _direct_turbovec_progress_label(
    engine: LodeEngine,
    state: ClientIndexState,
    *,
    generation: int,
) -> str:
    """Builds a raw-payload-free direct TurboVec progress label."""

    route_profile = _client_route_profile(engine.security, engine.route_policy)
    return (
        f"route={route_profile},storage={state.storage_profile},"
        f"native_dim={state.native_dim},generation={generation}"
    )


def _log_direct_turbovec_update_progress(
    progress_label: str,
    *,
    phase: str,
    event: str,
    chunk_count: int,
    native_dim: int,
    bit_width: int,
    generation: int,
    elapsed_ms: float | None = None,
) -> None:
    """Emits raw-payload-free progress for incremental direct TurboVec updates."""

    logger.info(
        "turbovec_update label=%s phase=%s event=%s chunks=%d native_dim=%d "
        "bit_width=%d generation=%d elapsed_ms=%s",
        progress_label,
        phase,
        event,
        chunk_count,
        native_dim,
        bit_width,
        generation,
        None if elapsed_ms is None else round(elapsed_ms, 3),
    )


def _native_backend_for_state(engine: LodeEngine, state: ClientIndexState) -> str:
    """Returns safe native backend evidence for the direct TurboVec backend."""

    del state
    capability = turbovec_capability(id_map_index_class=engine.turbovec_id_map_index_class)
    return capability.native_backend


def _native_used_for_state(engine: LodeEngine, state: ClientIndexState) -> bool:
    """Returns whether the native TurboVec kernel is active for serving."""

    del state
    capability = turbovec_capability(id_map_index_class=engine.turbovec_id_map_index_class)
    return capability.native_used


def _combined_backend(
    query_results: list[_EngineTurboVecQueryResult | None],
    *,
    field_name: str,
) -> str:
    """Returns a stable backend label for a batch of query result telemetry."""

    values = sorted(
        {
            str(getattr(result, field_name, ""))
            for result in query_results
            if result is not None and str(getattr(result, field_name, ""))
        }
    )
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    return "mixed"


def _max_query_result_int(
    query_results: list[_EngineTurboVecQueryResult | None],
    *,
    field_name: str,
) -> int:
    """Returns the maximum integer telemetry value across a query result batch."""

    return max(
        (int(getattr(result, field_name, 0)) for result in query_results if result is not None),
        default=0,
    )


def _sum_query_result_int(
    query_results: list[_EngineTurboVecQueryResult | None],
    *,
    field_name: str,
) -> int:
    """Returns the summed integer telemetry value across a query result batch."""

    return sum(
        int(getattr(result, field_name, 0)) for result in query_results if result is not None
    )


def _max_query_result_float(
    query_results: list[_EngineTurboVecQueryResult | None],
    *,
    field_name: str,
) -> float:
    """Returns the maximum floating-point telemetry value across a query result batch."""

    return max(
        (float(getattr(result, field_name, 0.0)) for result in query_results if result is not None),
        default=0.0,
    )


def _turbovec_bit_width_for_state(state: ClientIndexState) -> int:
    """Returns the compact bit width fixed at index creation for direct routes.

    Direct routes default to 4-bit; the opt-in 2-bit storage tier persists
    its width in the snapshot header so restarts and snapshot validation
    stay consistent even if the deployment's route profile changes.
    """

    width = int(getattr(state, "turbovec_bit_width", 4) or 4)
    if width not in {2, 4}:
        raise ValueError(f"unsupported direct TurboVec bit width: {width}")
    return width


def _validate_direct_turbovec_snapshot(
    state: ClientIndexState,
    index: TurboVecServingIndex,
) -> None:
    """Rejects direct TurboVec sidecars that do not match persisted JSON metadata."""

    if index.dim != state.native_dim:
        raise ValueError(
            f"direct TurboVec snapshot dim {index.dim} does not match native_dim {state.native_dim}"
        )
    expected_bit_width = _turbovec_bit_width_for_state(state)
    if index.bit_width != expected_bit_width:
        raise ValueError(
            f"direct TurboVec snapshot bit_width {index.bit_width} is unsupported; "
            f"expected {expected_bit_width}"
        )
    expected_stable_ids = stable_uint64_ids_for_chunk_ids(
        tuple(str(chunk.chunk_id) for chunk in state.chunks.values())
    )
    if len(index.index) != expected_stable_ids.size:
        raise ValueError(
            f"direct TurboVec snapshot vector count {len(index.index)} does not match "
            f"JSON chunk count {expected_stable_ids.size}"
        )
    missing = [
        int(stable_id)
        for stable_id in expected_stable_ids
        if not _turbovec_index_contains_stable_id(index.index, int(stable_id))
    ]
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        suffix = ", ..." if len(missing) > 5 else ""
        raise ValueError(
            "direct TurboVec snapshot stable IDs do not match JSON chunks: "
            f"missing {preview}{suffix}"
        )


def _turbovec_index_contains_stable_id(index: Any, stable_id: int) -> bool:
    """Returns whether a loaded TurboVec IdMapIndex contains one stable uint64 ID."""

    contains = getattr(index, "contains", None)
    if callable(contains):
        return bool(contains(stable_id))
    return stable_id in index


def _sha256_file(path: Path) -> str:
    """Returns the SHA-256 hex digest for a local snapshot file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256_hex(value: str) -> bool:
    """Returns whether a value is a lowercase SHA-256 hex digest."""

    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _snapshot_forbidden_keys(payload: Any) -> set[str]:
    """Finds forbidden snapshot field names that could carry raw payload text."""

    forbidden_names = {
        "client_id",
        "document_text",
        "query_text",
        "chunk_text",
        "raw_document",
        "raw_query",
        "raw_chunk",
        "raw_payload",
        "text",
    }
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            key_text = str(key)
            if key_text in forbidden_names:
                found.add(key_text)
            found.update(_snapshot_forbidden_keys(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_snapshot_forbidden_keys(value))
    return found


def client_index_key(client_id: str) -> str:
    """Returns the stable in-memory and persisted index key for one client ID."""

    return sha256_text(client_id)


def normalize_index_id(index_id: str | None) -> str:
    """Returns a validated public index ID, defaulting legacy callers to default."""

    if index_id is None:
        return DEFAULT_INDEX_ID
    if not isinstance(index_id, str):
        raise ValueError("index_id must be a string")
    value = index_id.strip()
    if not value:
        raise ValueError("index_id is required")
    if value == DEFAULT_INDEX_ID:
        return value
    if re.fullmatch(r"idx_[0-9a-f]{24}", value):
        return value
    raise ValueError("index_id is invalid")


def index_state_key(client_id: str, index_id: str | None = None) -> str:
    """Returns the private state key for a client/index pair."""

    return index_state_key_for_client_hash(sha256_text(client_id), normalize_index_id(index_id))


def index_state_key_for_client_hash(client_id_hash: str, index_id: str) -> str:
    """Returns the persisted state key while preserving legacy default snapshot names."""

    if index_id == DEFAULT_INDEX_ID:
        return client_id_hash
    return sha256_text(f"{client_id_hash}:{index_id}")


def generated_index_id() -> str:
    """Returns a stable-format random public index ID for managed resources."""

    return f"idx_{uuid.uuid4().hex[:24]}"


def _requested_index_id(
    index_id: str | None,
    *,
    name: str | None,
    metadata: Mapping[str, Any] | None,
) -> str:
    """Resolves create-index compatibility defaults and managed generated IDs."""

    if index_id is not None:
        return normalize_index_id(index_id)
    if name is None and not metadata:
        return DEFAULT_INDEX_ID
    return generated_index_id()


def _validate_index_name(name: str | None) -> str:
    """Returns a public index display name after validation."""

    if name is None:
        return DEFAULT_INDEX_NAME
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    stripped = name.strip()
    if not stripped:
        raise ValueError("name must be nonblank")
    if len(stripped) > 120:
        raise ValueError("name must be 120 characters or fewer")
    return stripped


def _utc_now_iso(now: datetime) -> str:
    """Formats an aware timestamp as a simple UTC ISO string."""

    return now.astimezone(UTC).isoformat()


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, str]:
    """Validates filterable metadata as a small flat string-valued mapping."""

    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")
    safe: dict[str, str] = {}
    forbidden = _snapshot_forbidden_keys(dict(metadata))
    if forbidden:
        keys = ", ".join(sorted(forbidden))
        raise ValueError(f"metadata contains reserved raw-payload keys: {keys}")
    for key, value in metadata.items():
        key_text = str(key).strip()
        if not key_text:
            raise ValueError("metadata keys must be nonblank")
        if isinstance(value, bool | int | float | str) or value is None:
            safe[key_text] = "" if value is None else str(value)
            continue
        raise ValueError("metadata values must be scalar strings, numbers, booleans, or null")
    return safe


def client_state_snapshot_payload(state: ClientIndexState) -> dict[str, Any]:
    """Returns the redacted persisted snapshot payload for one client index."""

    return _state_to_payload(state)


def storage_profile_for_route_policy(route_policy: EngineRoutePolicy | None) -> str:
    """Returns the storage profile implied by the selected route policy."""

    if route_policy is not None and route_policy.index_backend != DIRECT_TURBOVEC_STORAGE_PROFILE:
        raise ValueError(f"unsupported index backend: {route_policy.index_backend}")
    return DIRECT_TURBOVEC_STORAGE_PROFILE


def normalized_chunk_hash(text: str) -> str:
    """Hashes normalized chunk text so harmless whitespace changes can reuse embeddings."""

    return sha256_text(" ".join(text.split()))


def _vector_content_hash(vector: np.ndarray) -> str:
    """Hashes a float32 embedding so re-adding an identical vector is a no-op.

    The vector-in counterpart to :func:`sha256_text` over document text: it lets
    :meth:`LodeEngine._ingest_vectors` detect an unchanged vector and skip the
    re-encode, keeping repeated upserts of the same vector O(1).
    """

    return hashlib.sha256(np.asarray(vector, dtype=np.float32).tobytes()).hexdigest()


def _chunks_by_hash_occurrence(chunks: list[IndexedChunk]) -> dict[str, list[IndexedChunk]]:
    """Groups old chunks by normalized hash while preserving document occurrence order."""

    grouped: dict[str, list[IndexedChunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.content_hash, []).append(chunk)
    return grouped


def _chunk_id_for_hash(document_id: str, *, chunk_hash: str, occurrence: int) -> str:
    """Builds a stable chunk ID from document ID, normalized chunk hash, and occurrence."""

    return f"{document_id}:{chunk_hash[:12]}:{occurrence:04d}"


def _delete_orphaned_document_chunks(
    state: ClientIndexState,
    document_id: str,
    new_chunk_ids: list[str],
) -> tuple[str, ...]:
    """Deletes old chunk rows for a document that are absent after a chunk-level upsert.

    Returns the chunk ids it removed so the caller can thread them into the
    direct TurboVec changeset (the count is just ``len(...)`` of this).
    """

    new_ids = set(new_chunk_ids)
    orphan_ids = tuple(
        chunk_id
        for chunk_id in state.document_chunk_ids.get(document_id, ())
        if chunk_id not in new_ids
    )
    for chunk_id in orphan_ids:
        state.chunks.pop(chunk_id, None)
    return orphan_ids


def _recompute_fraction(*, embedded: int, reused: int) -> float:
    """Returns the changed-chunk recompute fraction for one upsert response."""

    accounted = embedded + reused
    return 0.0 if accounted == 0 else embedded / float(accounted)


def _document_ingest_response_payload(
    state: ClientIndexState,
    result: Mapping[str, float | int],
) -> dict[str, Any]:
    """Formats redacted build/upsert counters for public engine responses."""

    embedded = int(result["embedded_chunks"])
    reused = int(result["reused_chunks"])
    payload = {
        "document_count": int(result["document_count"]),
        "embedded_chunks": embedded,
        "reused_chunks": reused,
        "deleted_chunks": int(result["deleted_chunks"]),
        "embedding_time_ms": float(result["embedding_time_ms"]),
        "recompute_fraction": _recompute_fraction(embedded=embedded, reused=reused),
    }
    if "embedding_batch_size" in result:
        payload["embedding_batch_size"] = int(result["embedding_batch_size"])
    return payload


def _documents_wal_payload(documents: tuple[EngineDocument, ...]) -> dict[str, Any]:
    """Frames an ``upsert``/``build`` document batch as a WAL record payload.

    Captures exactly the public-call inputs (id, text, redacted metadata) so
    replay re-embeds and re-indexes through the identical ingest path. Text is
    included because the WAL is the writer's own durable log; it is never read by
    a redacted/telemetry path and never leaves the process.
    """

    return {
        "documents": [
            {
                "document_id": document.document_id,
                "text": document.text,
                "metadata": dict(document.metadata),
            }
            for document in documents
        ]
    }


def _vectors_wal_payload(vectors: tuple[EngineVectorDocument, ...]) -> dict[str, Any]:
    """Frames a vector-in upsert batch as a WAL record payload.

    Stores each vector verbatim (already L2-normalized at the SDK boundary) plus
    redacted metadata and the optional retained text, so replay reproduces the
    same encoded rows the live commit did.
    """

    return {
        "vectors": [
            {
                "document_id": vector.document_id,
                "vector": [float(value) for value in vector.vector],
                "metadata": dict(vector.metadata),
                "text": vector.text,
            }
            for vector in vectors
        ]
    }


def _embedding_batch_size_for_backend(backend: EngineEmbeddingBackend) -> int:
    """Returns the configured maximum document-embedding batch size for a backend."""

    batch_size = getattr(backend, "batch_size", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        return 1
    return max(1, batch_size)


def _resolved_embedding_batch_size(
    backend: EngineEmbeddingBackend,
    *,
    embed_batch_size: int | None,
) -> int:
    """Returns the effective embedding batch size after caller validation."""

    if embed_batch_size is None:
        return _embedding_batch_size_for_backend(backend)
    if isinstance(embed_batch_size, bool) or not isinstance(embed_batch_size, int):
        raise ValueError("embed_batch_size must be an integer")
    if embed_batch_size <= 0:
        raise ValueError("embed_batch_size must be positive")
    return embed_batch_size


def _embed_texts_in_batches(
    backend: EngineEmbeddingBackend,
    texts: tuple[str, ...],
    *,
    batch_size: int,
) -> tuple[tuple[float, ...], ...]:
    """Embeds texts in backend-sized batches while preserving request order."""

    if not texts:
        return ()
    embeddings: list[tuple[float, ...]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        embeddings.extend(backend.embed_documents(tuple(batch)))
    return tuple(embeddings)


def _validate_query_filter(filter_payload: dict[str, Any] | None) -> dict[str, Any]:
    """Validates the public query filter object for exact document metadata matching."""

    if filter_payload is None:
        return {}
    if not isinstance(filter_payload, dict):
        raise ValueError("filter must be an object")
    allowed = {"metadata", "document_ids"}
    unknown = set(filter_payload) - allowed
    if unknown:
        raise ValueError("filter contains unsupported fields")
    validated: dict[str, Any] = {}
    if "metadata" in filter_payload:
        validated["metadata"] = validate_metadata_filter(filter_payload["metadata"])
    if "document_ids" in filter_payload:
        document_ids = filter_payload["document_ids"]
        if not isinstance(document_ids, list) or not document_ids:
            raise ValueError("filter.document_ids must be a nonempty list")
        cleaned_ids: list[str] = []
        for item in document_ids:
            if not isinstance(item, str) or _is_blank(item):
                raise ValueError("filter.document_ids must contain nonblank strings")
            cleaned_ids.append(item.strip())
        validated["document_ids"] = tuple(dict.fromkeys(cleaned_ids))
    return validated


def _validate_query_includes(include: tuple[str, ...]) -> tuple[str, ...]:
    """Validates optional safe include fields for query responses."""

    if not include:
        return ()
    cleaned: list[str] = []
    for item in include:
        if not isinstance(item, str):
            raise ValueError("include values must be strings")
        value = item.strip()
        if value != QUERY_INCLUDE_METADATA:
            raise ValueError("include may only contain metadata")
        cleaned.append(value)
    return tuple(dict.fromkeys(cleaned))


def _validate_query_mode(mode: Any) -> str:
    """Validates the retrieval mode and returns the canonical lowercase value."""

    if mode is None:
        return RETRIEVAL_MODE_VECTOR
    if not isinstance(mode, str):
        raise ValueError("mode must be a string")
    value = mode.strip().lower()
    if not value:
        return RETRIEVAL_MODE_VECTOR
    if value not in RETRIEVAL_MODES:
        allowed = ", ".join(sorted(RETRIEVAL_MODES))
        raise ValueError(f"mode must be one of: {allowed}")
    return value


def _filter_signature(query_filter: Mapping[str, Any]) -> tuple[Any, ...]:
    """Returns a hashable key grouping batch queries that share an identical filter."""

    metadata = query_filter.get("metadata") or {}
    document_ids = query_filter.get("document_ids") or ()
    return (_hashable_filter(metadata), tuple(document_ids))


def _hashable_filter(value: Any) -> Any:
    """Canonicalizes a (possibly predicate) validated filter into a hashable key.

    Predicate filters nest dicts (operator maps) and lists (``$and``/``$or``),
    which are unhashable; this maps each dict to a sorted item tuple and each list
    to a tuple so the batch grouper can key on the filter. Equal filters yield
    equal keys, so identical filters still collapse into one batch group.
    """

    if isinstance(value, Mapping):
        return tuple(sorted((key, _hashable_filter(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable_filter(item) for item in value)
    return value


def _empty_turbovec_result(index: TurboVecServingIndex) -> _EngineTurboVecQueryResult:
    """Returns an empty direct-TurboVec result (filter matched no eligible rows)."""

    return _EngineTurboVecQueryResult(
        index=index,
        stable_ids=np.empty(0, dtype=np.uint64),
        scores=np.empty(0, dtype=np.float32),
        native_used=index.native_used,
        native_backend=index.native_backend,
        retrieval_mode="direct_turbovec",
        fallback_used=False,
        compact_route_fallback=False,
    )


def _normalize_rows_for_engine(matrix: np.ndarray) -> np.ndarray:
    """Returns row-normalized float32 data for engine-side rerank matrices."""

    rows = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


def _is_predicate_filter(metadata: Mapping[str, Any] | None) -> bool:
    """Returns whether a validated metadata filter uses operators or logical composition.

    Such filters can't be resolved by the exact-match posting index (which only
    does ``(field, value)`` equality lookups), so they take the compiled-scan
    path. A flat map of bare-scalar equalities returns False (posting-resolvable).
    """

    if not metadata:
        return False
    return any(
        key.startswith("$") or isinstance(value, Mapping) for key, value in metadata.items()
    )


def _compile_query_filter(
    query_filter: Mapping[str, Any],
) -> Callable[[str, Mapping[str, str]], bool]:
    """Compiles a validated query filter into a reusable per-document matcher.

    Hoists the per-document-invariant work out of the corpus scan / post-filter:
    the optional ``document_ids`` allowlist becomes one set and the metadata filter
    is compiled once (operators dispatched, operands pre-parsed). The returned
    ``(document_id, document_metadata) -> bool`` does no re-walking, set rebuilding,
    or operand parsing per call.
    """

    document_ids = query_filter.get("document_ids")
    allowed_ids = set(document_ids) if document_ids is not None else None
    metadata = query_filter.get("metadata")
    metadata_predicate = compile_metadata_filter(metadata) if metadata is not None else None

    def _matches(document_id: str, document_metadata: Mapping[str, str]) -> bool:
        if allowed_ids is not None and document_id not in allowed_ids:
            return False
        if metadata_predicate is None:
            return True
        return metadata_predicate(document_metadata)

    return _matches


def _document_resource_payload(state: ClientIndexState, document_id: str) -> dict[str, Any]:
    """Builds a redacted public document resource without text, chunks, or embeddings."""

    return {
        "document_id": document_id,
        "metadata": dict(state.document_metadata.get(document_id, {})),
        "chunk_count": len(state.document_chunk_ids.get(document_id, ())),
        "content_hash": state.document_hashes.get(document_id, ""),
    }


def _index_resource_payload(
    engine: LodeEngine,
    state: ClientIndexState,
    *,
    status: str | None = None,
) -> dict[str, Any]:
    """Builds a redacted public index resource payload."""

    return {
        "index_id": state.index_id,
        "name": state.name,
        "status": status or state.status,
        "route_profile": _client_route_profile(engine.security, engine.route_policy),
        "storage_profile": state.storage_profile,
        "document_count": len(state.document_hashes),
        "chunk_count": len(state.chunks),
        "metadata": dict(state.metadata),
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }


def _log_large_ingest_progress(
    *,
    operation: str,
    processed_documents: int,
    total_documents: int,
    embedded_chunks: int,
    reused_chunks: int,
) -> None:
    """Emits raw-payload-free progress for large engine build/upsert operations."""

    if total_documents < 1_000:
        return
    if processed_documents not in {1, total_documents} and processed_documents % 500 != 0:
        return
    logger.info(
        "%s processed_documents=%d total_documents=%d embedded_chunks=%d reused_chunks=%d",
        operation,
        processed_documents,
        total_documents,
        embedded_chunks,
        reused_chunks,
    )


def _log_document_ingest_finalize_progress(
    operation: str,
    *,
    phase: str,
    event: str,
    document_count: int,
    chunk_count: int,
    elapsed_ms: float | None = None,
) -> None:
    """Emits raw-payload-free progress for large build/upsert finalization phases."""

    if document_count < 1_000 and chunk_count < 100_000:
        return
    logger.info(
        "%s phase=%s event=%s documents=%d chunks=%d elapsed_ms=%s",
        operation,
        phase,
        event,
        document_count,
        chunk_count,
        None if elapsed_ms is None else round(elapsed_ms, 3),
    )


def _state_recompute_fraction(state: ClientIndexState) -> float:
    """Returns the cumulative chunk-level recompute fraction for one client index."""

    return _recompute_fraction(
        embedded=state.embedded_chunk_count,
        reused=state.cache_reuse_count,
    )


def _chunks_have_full_embeddings(chunks: tuple[IndexedChunk, ...], *, native_dim: int) -> bool:
    """Returns whether chunks still carry transient full-dimensional embeddings."""

    for chunk in chunks:
        vector = np.asarray(chunk.embedding, dtype=np.float32)
        if vector.ndim != 1 or vector.shape[0] != native_dim:
            return False
    return True


def _discard_direct_turbovec_transient_embeddings(
    state: ClientIndexState, chunk_ids: tuple[str, ...]
) -> None:
    """Drops the transient full vectors of the rows just synced into the index.

    Only those rows still carry a non-zero transient embedding (prior rows were
    zeroed by their own commit), so this is O(changed), not O(corpus). The
    encoded codes already live in the index; the redacted ``.json`` state and
    persistence read the index, not these embeddings.
    """

    if not chunk_ids:
        return
    zero = tuple(0.0 for _item in range(state.native_dim))
    for chunk_id in chunk_ids:
        chunk = state.chunks.get(chunk_id)
        if chunk is None:
            continue
        state.chunks[chunk_id] = IndexedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content_hash=chunk.content_hash,
            embedding=zero,
        )


def audit_persisted_index_snapshots(persistence_dir: str | Path) -> dict[str, Any]:
    """Validates local index snapshots and returns a raw-payload-free audit manifest.

    Audits each index at the consistent generation named by its
    ``<key>.commit.json`` root manifest (epoch-addressed artifacts under
    ``<key>.gen/``), and falls back to any legacy top-level ``<key>.json`` base
    written before the commit-manifest layout.
    """

    directory = Path(persistence_dir)
    if not directory.exists():
        raise FileNotFoundError(f"index snapshot directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"index snapshot path is not a directory: {directory}")
    snapshot_rows: list[dict[str, Any]] = []
    audited_keys: set[str] = set()
    for commit_path in sorted(directory.glob(f"*{COMMIT_MANIFEST_SUFFIX}")):
        key = commit_path.name[: -len(COMMIT_MANIFEST_SUFFIX)]
        body = read_commit_manifest(commit_path)
        if body is None:
            continue
        epoch = int(body["base_epoch"])
        snapshot_rows.append(
            _audit_snapshot_row(
                base_json_path(directory, key, epoch),
                tvim_path=base_tvim_path(directory, key, epoch),
                expected_key=key,
                file_name=commit_path.name,
                json_manifest=body.get("json"),
                tvim_manifest=body.get("tvim"),
            )
        )
        audited_keys.add(key)
    for path in sorted(directory.glob("*.json")):
        if is_commit_manifest_name(path.name) or path.stem in audited_keys:
            continue
        snapshot_rows.append(
            _audit_snapshot_row(
                path,
                tvim_path=directory / f"{path.stem}.tvim",
                expected_key=path.stem,
                file_name=path.name,
            )
        )
    return {
        "artifact_type": "index_snapshot_audit",
        "status": "passed",
        "snapshot_directory": str(directory),
        "snapshot_count": len(snapshot_rows),
        "raw_document_text_present": False,
        "raw_query_text_present": False,
        "raw_client_id_present": False,
        "client_ids_hashed": True,
        "snapshot_files": snapshot_rows,
    }


def _audit_snapshot_row(
    json_path: Path,
    *,
    tvim_path: Path,
    expected_key: str,
    file_name: str,
    json_manifest: dict[str, Any] | None = None,
    tvim_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validates one index snapshot (base + journals + vector sidecar) for audit.

    ``json_manifest``/``tvim_manifest`` are the per-store manifests embedded in
    the root commit manifest; passing them audits the **committed** generation
    rather than whatever a crashed writer last left in the on-disk per-store
    manifests (legacy snapshots pass ``None`` and use their on-disk manifests).
    """

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    journal = StateJournalStore(json_path)
    journal_audit: dict[str, Any] = {}
    if json_manifest is not None or journal.has_manifest():
        # Journal segments are part of the persisted state: validate their
        # checksums, scan their bodies for raw-payload leaks, and replay them so
        # the audited counts reflect the committed served state.
        journal.validate_base_checksum(manifest=json_manifest)
        for body in journal.iter_segment_bodies(manifest=json_manifest):
            segment_forbidden = _snapshot_forbidden_keys(body)
            if segment_forbidden:
                keys = ", ".join(sorted(segment_forbidden))
                raise ValueError(f"{file_name} journal contains forbidden raw-payload keys: {keys}")
        replay_stats = journal.replay_onto_payload(payload, manifest=json_manifest)
        journal_audit = {
            **journal.storage_file_bytes(manifest=json_manifest),
            "json_delta_segments_replayed": replay_stats["json_delta_segments_replayed"],
        }
    forbidden_keys = _snapshot_forbidden_keys(payload)
    if forbidden_keys:
        keys = ", ".join(sorted(forbidden_keys))
        raise ValueError(f"{file_name} contains forbidden raw-payload keys: {keys}")
    if _snapshot_contains_full_float_embeddings(payload):
        raise ValueError(f"{file_name} contains full float embedding arrays")
    if _snapshot_contains_per_vector_json_arrays(payload):
        raise ValueError(f"{file_name} contains per-vector JSON arrays")
    state = _state_from_payload(payload)
    if not _is_sha256_hex(expected_key):
        raise ValueError(f"snapshot index key must be a SHA-256 hash: {file_name}")
    if state.index_key != expected_key:
        raise ValueError(f"snapshot index_key does not match its key: {file_name}")
    if not tvim_path.exists() and state.chunks:
        raise ValueError(f"{file_name} is missing required TurboVec sidecar")
    sidecar_audit: dict[str, Any] = (
        {
            "file_name": tvim_path.name,
            "file_sha256": _sha256_file(tvim_path),
            "storage_profile": DIRECT_TURBOVEC_STORAGE_PROFILE,
            "sidecar_bytes": tvim_path.stat().st_size,
        }
        if tvim_path.exists()
        else {}
    )
    tvim_store = TvimDeltaStore(tvim_path)
    if sidecar_audit and (tvim_manifest is not None or tvim_store.has_manifest()):
        # Delta segments extend the base `.tvim`; account for them so the audit
        # covers every persisted byte of the committed direct route.
        sidecar_audit.update(tvim_store.storage_file_bytes(manifest=tvim_manifest))
    return {
        "file_name": file_name,
        "client_id_hash": state.client_id_hash,
        "index_id": state.index_id,
        "index_key": state.index_key,
        "file_sha256": _sha256_file(json_path),
        "document_count": len(state.document_hashes),
        "chunk_count": len(state.chunks),
        "embedding_count": len(state.chunks),
        "json_bytes": json_path.stat().st_size,
        "state_journal": journal_audit,
        "vector_sidecar": sidecar_audit,
        "raw_payload_text_fields_present": False,
    }


def _snapshot_contains_full_float_embeddings(payload: Any) -> bool:
    """Returns whether a snapshot uses retired per-chunk full-float embedding arrays."""

    if not isinstance(payload, dict):
        return False
    for chunk_payload in payload.get("chunks", ()):
        if isinstance(chunk_payload, dict) and "embedding" in chunk_payload:
            return True
    return False


def _snapshot_contains_per_vector_json_arrays(payload: Any) -> bool:
    """Returns whether JSON chunk rows contain retired per-dimension vector arrays."""

    if not isinstance(payload, dict):
        return False
    vector_keys = {
        "stage_one_embedding",
        "refit_embedding",
        "compact_embedding",
        "turbovec_codes",
        "rerank_vector",
    }
    for chunk_payload in payload.get("chunks", ()):
        if not isinstance(chunk_payload, dict):
            continue
        for key in vector_keys:
            value = chunk_payload.get(key)
            if isinstance(value, list):
                return True
            if isinstance(value, dict) and any(isinstance(item, list) for item in value.values()):
                return True
    return False


def _chunk_row_payload(chunk: IndexedChunk) -> dict[str, Any]:
    """Serializes one chunk's redacted snapshot row (references only, no vectors)."""

    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "content_hash": chunk.content_hash,
    }


def _state_header_payload(
    state: ClientIndexState,
    *,
    columnar_generation: int = 0,
    route_profile: str = "custom",
) -> dict[str, Any]:
    """Serializes the scalar snapshot header shared by full rewrites and journal deltas.

    Document collections are deliberately excluded: the journal replays
    documents individually rather than re-journaling the full set per batch.
    """

    return {
        "schema_version": 1,
        "client_id_hash": state.client_id_hash,
        "index_id": state.index_id,
        "index_key": state.index_key,
        "name": state.name,
        "metadata": dict(sorted(state.metadata.items())),
        "status": state.status,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "columnar_generation": int(columnar_generation),
        "model": state.model,
        "provider": state.provider,
        "task": state.task,
        "native_dim": state.native_dim,
        "route_profile": route_profile,
        "storage_profile": state.storage_profile,
        "turbovec_bit_width": int(state.turbovec_bit_width),
        "embedded_chunk_count": state.embedded_chunk_count,
        "cache_reuse_count": state.cache_reuse_count,
        "delete_count": state.delete_count,
        "deleted_chunk_count": state.deleted_chunk_count,
        "query_count": state.query_count,
        "fallback_count": state.fallback_count,
        "fallback_reasons": dict(sorted(state.fallback_reasons.items())),
    }


def _state_journal_document_entry(state: ClientIndexState, document_id: str) -> dict[str, Any]:
    """Serializes one upserted document's redacted rows for a journal delta."""

    chunk_ids = state.document_chunk_ids.get(document_id, ())
    return {
        "document_id": document_id,
        "document_hash": state.document_hashes.get(document_id, ""),
        "chunk_ids": [str(chunk_id) for chunk_id in chunk_ids],
        "metadata": dict(sorted(state.document_metadata.get(document_id, {}).items())),
        "chunks": [
            _chunk_row_payload(state.chunks[chunk_id])
            for chunk_id in chunk_ids
            if chunk_id in state.chunks
        ],
    }


def _state_to_payload(
    state: ClientIndexState,
    *,
    columnar_generation: int = 0,
    route_profile: str = "custom",
) -> dict[str, Any]:
    """Serializes one client index state without raw document, chunk, or query text."""

    payload = _state_header_payload(
        state,
        columnar_generation=columnar_generation,
        route_profile=route_profile,
    )
    payload["chunks"] = [
        _chunk_row_payload(chunk)
        for chunk in sorted(state.chunks.values(), key=lambda item: item.chunk_id)
    ]
    payload["document_hashes"] = dict(sorted(state.document_hashes.items()))
    payload["document_chunk_ids"] = {
        document_id: list(chunk_ids)
        for document_id, chunk_ids in sorted(state.document_chunk_ids.items())
    }
    payload["document_metadata"] = {
        document_id: dict(sorted(metadata.items()))
        for document_id, metadata in sorted(state.document_metadata.items())
    }
    return payload


def _state_from_payload(payload: Any) -> ClientIndexState:
    """Restores one client index state from a redacted local snapshot payload."""

    if not isinstance(payload, dict):
        raise ValueError("persisted engine state must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported engine state schema version")
    legacy_client_id = str(payload.get("client_id", ""))
    client_id_hash = str(payload.get("client_id_hash") or sha256_text(legacy_client_id))
    index_id = normalize_index_id(str(payload.get("index_id", DEFAULT_INDEX_ID)))
    index_key = str(
        payload.get("index_key") or index_state_key_for_client_hash(client_id_hash, index_id)
    )
    state = ClientIndexState(
        client_id="",
        client_id_hash=client_id_hash,
        index_id=index_id,
        index_key=index_key,
        model=str(payload["model"]),
        provider=str(payload["provider"]),
        task=str(payload["task"]),
        native_dim=int(payload["native_dim"]),
        name=str(payload.get("name", DEFAULT_INDEX_NAME)),
        metadata=_validate_metadata(dict(payload.get("metadata", {}) or {})),
        status=str(payload.get("status", ACTIVE_INDEX_STATUS)),
        created_at=str(payload.get("created_at", LEGACY_INDEX_TIMESTAMP)),
        updated_at=str(payload.get("updated_at", LEGACY_INDEX_TIMESTAMP)),
        storage_profile=_storage_profile_from_payload(payload.get("storage_profile", "standard")),
        turbovec_bit_width=int(payload.get("turbovec_bit_width", 4)),
        embedded_chunk_count=int(payload.get("embedded_chunk_count", 0)),
        cache_reuse_count=int(payload.get("cache_reuse_count", 0)),
        delete_count=int(payload.get("delete_count", 0)),
        deleted_chunk_count=int(payload.get("deleted_chunk_count", 0)),
        query_count=int(payload.get("query_count", 0)),
        fallback_count=int(payload.get("fallback_count", 0)),
        fallback_reasons={
            str(key): int(value)
            for key, value in dict(payload.get("fallback_reasons", {}) or {}).items()
        },
        query_latency_ms=deque(
            (float(value) for value in payload.get("query_latency_ms", ())),
            maxlen=QUERY_LATENCY_SAMPLE_CAP,
        ),
    )
    for chunk_payload in payload.get("chunks", ()):
        chunk = _chunk_from_payload(chunk_payload, native_dim=state.native_dim)
        state.chunks[chunk.chunk_id] = chunk
    state.document_hashes = {
        str(key): str(value)
        for key, value in dict(payload.get("document_hashes", {}) or {}).items()
    }
    state.document_chunk_ids = {
        str(key): tuple(str(chunk_id) for chunk_id in value)
        for key, value in dict(payload.get("document_chunk_ids", {}) or {}).items()
    }
    state.document_metadata = {
        str(key): _validate_metadata(dict(value or {}))
        for key, value in dict(payload.get("document_metadata", {}) or {}).items()
    }
    return state


def _chunk_from_payload(payload: Any, *, native_dim: int) -> IndexedChunk:
    """Restores one indexed chunk reference row from a redacted snapshot.

    Persisted rows never carry vectors: the transient embedding restores as
    zeros and the served codes live in the `.tvim` sidecar.
    """

    if not isinstance(payload, dict):
        raise ValueError("persisted chunk must be a JSON object")
    return IndexedChunk(
        chunk_id=str(payload["chunk_id"]),
        document_id=str(payload["document_id"]),
        content_hash=str(payload["content_hash"]),
        embedding=tuple(0.0 for _item in range(native_dim)),
    )


def is_private_bind_host(host: str) -> bool:
    """Returns whether an engine bind host is loopback or private network only."""

    try:
        parsed = ip_address(host)
    except ValueError:
        return host == "localhost"
    if parsed.is_unspecified:
        return False
    return parsed.is_loopback or parsed.is_private


def sanitize_observability_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Keeps telemetry fields numeric, boolean, or low-cardinality status strings."""

    safe: dict[str, Any] = {}
    for key, value in fields.items():
        if isinstance(value, bool | int | float):
            safe[key] = value
        elif isinstance(value, str) and key in {
            "status",
            "route_classification",
            "storage_profile",
            "compact_backend",
            "native_backend",
            "retrieval_mode",
            "stage_one_backend",
            # Shared resident-scan telemetry namespace (CUDA + MPS); see
            # _EngineTurboVecQueryResult. Keep both backends' string statuses here.
            "gpu_stage_one_status",
            "gpu_fallback_reason",
        }:
            safe[key] = value
    return safe


def chunk_text(text: str, character_limit: int) -> tuple[str, ...]:
    """Splits text into deterministic non-empty chunks bounded by character count."""

    if character_limit <= 0:
        raise ValueError("character_limit must be positive")
    stripped = text.strip()
    if not stripped:
        return ()
    return tuple(
        stripped[start : start + character_limit]
        for start in range(0, len(stripped), character_limit)
    )


def _is_blank(value: str) -> bool:
    """Returns whether a caller-supplied string is empty after trimming whitespace."""

    return not value.strip()


def _client_route_profile(
    security: EngineSecurityConfig,
    route_policy: EngineRoutePolicy | None,
) -> str:
    """Returns the only route selector that may be exposed to clients."""

    if security.route_profile:
        return security.route_profile
    return route_policy.profile if route_policy is not None else "custom"


def _latency_payload(samples_ms: list[float]) -> dict[str, Any]:
    """Summarizes per-client query latency samples without storing query text."""

    if not samples_ms:
        return {
            "sample_count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    samples = np.asarray(samples_ms, dtype=np.float64)
    return {
        "sample_count": int(samples.size),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "p99_ms": float(np.percentile(samples, 99)),
    }


def _physical_storage_payload(
    state: ClientIndexState,
    *,
    route: RouteDecision,
    route_profile: str,
) -> dict[str, Any]:
    """Returns physical required-vector accounting for the persisted representation."""

    del route_profile
    chunk_count = len(state.chunks)
    if _state_uses_direct_turbovec(state):
        bit_width = _turbovec_bit_width_for_state(state)
        code_bpe = float(state.native_dim * bit_width / 8.0)
        id_bpe = 8.0
        norm_bpe = 4.0
        stored_bpe = code_bpe + id_bpe + norm_bpe
        stored_bytes = float(chunk_count) * stored_bpe
        return {
            "stored_bytes": stored_bytes,
            "required_vector_payload_bytes": int(stored_bytes),
            "required_vector_payload_bytes_per_embedding": stored_bpe,
            "code_bytes_per_embedding": code_bpe,
            "id_bytes_per_embedding": id_bpe,
            "norm_bytes_per_embedding": norm_bpe,
            "stored_bytes_per_embedding_estimate": stored_bpe,
            "stage_one_bytes": 0,
            "turbovec_bytes": int(chunk_count * (code_bpe + norm_bpe)),
            "rerank_bytes": 0,
            "rerank_scale_bytes": 0,
            "refit_source_bytes": 0,
            "row_id_bytes": int(chunk_count * id_bpe),
            "fixed_projection_bytes": 0,
            "json_metadata_bytes": 0,
            "persisted_total_bytes": 0,
            "persisted_total_bytes_per_embedding": 0.0,
            "ram_index_bytes": stored_bytes,
            "metadata_bytes": float(chunk_count * id_bpe),
            "stage_one_ram_index_bytes": 0.0,
            "storage_profile": state.storage_profile,
        }
    raise ValueError("storage accounting requires a direct TurboVec state")


def _persisted_storage_bytes(
    engine: LodeEngine,
    state: ClientIndexState,
) -> dict[str, Any]:
    """Returns persisted JSON and sidecar bytes when local snapshots are configured."""

    if engine.persistence_dir is None:
        return {
            "actual_persisted_bytes_available": False,
            "persisted_json_bytes": 0,
            "persisted_sidecar_bytes": 0,
            "persisted_total_bytes": 0,
            "persisted_total_bytes_per_embedding": 0.0,
        }
    json_path, tvim_path = engine._live_base_paths(state.index_key)
    # The direct surface persists exactly one binary sidecar: the `.tvim`
    # TurboVec snapshot.
    sidecar_paths = (tvim_path,)
    json_bytes = json_path.stat().st_size if json_path.exists() else 0
    sidecar_bytes = sum(path.stat().st_size for path in sidecar_paths if path.exists())
    # Delta journal directories extend the base files under the auto policy;
    # both report zero when absent so cold full-rewrite states are unaffected.
    journal_store = StateJournalStore(json_path)
    json_journal_bytes = 0.0
    if journal_store.has_manifest():
        accounting = journal_store.storage_file_bytes()
        json_journal_bytes = accounting["json_delta_bytes"] + accounting["json_manifest_bytes"]
    tvim_store = TvimDeltaStore(tvim_path)
    tvim_delta_bytes = 0.0
    if tvim_store.has_manifest():
        accounting = tvim_store.storage_file_bytes()
        tvim_delta_bytes = accounting["tvim_delta_bytes"] + accounting["tvim_manifest_bytes"]
    # The opt-in raw-text store (base + .txd journal) is byte-counted (never
    # inspected) so the storage report stays honest about on-disk usage; it is
    # zero unless raw-text storage is enabled and at least one document is stored.
    live_epoch = engine._base_epochs.get(state.index_key)
    if live_epoch is not None:
        raw_text_bytes = int(
            engine._text_store(state.index_key, live_epoch).storage_file_bytes()[
                "raw_text_sidecar_bytes"
            ]
        )
    else:
        legacy_text = engine._legacy_text_sidecar_path(state.index_key)
        raw_text_bytes = int(legacy_text.stat().st_size) if legacy_text.is_file() else 0
    total_bytes = int(
        json_bytes + sidecar_bytes + json_journal_bytes + tvim_delta_bytes + raw_text_bytes
    )
    chunk_count = len(state.chunks)
    return {
        "actual_persisted_bytes_available": True,
        "persisted_json_bytes": int(json_bytes),
        "json_metadata_bytes": int(json_bytes),
        "persisted_json_journal_bytes": int(json_journal_bytes),
        "persisted_tvim_delta_bytes": int(tvim_delta_bytes),
        "persisted_sidecar_bytes": int(sidecar_bytes),
        "persisted_raw_text_sidecar_bytes": raw_text_bytes,
        "persisted_total_bytes": total_bytes,
        "persisted_total_bytes_per_embedding": (
            0.0 if chunk_count == 0 else float(total_bytes) / float(chunk_count)
        ),
    }


def _storage_profile_from_payload(value: object) -> str:
    """Restores a persisted redacted storage profile with legacy standard fallback."""

    text = str(value)
    if text == DIRECT_TURBOVEC_STORAGE_PROFILE:
        return text
    raise ValueError(
        "storage_profile must be turbovec_direct; standard-cascade and "
        "extreme_compact snapshots are retired"
    )
