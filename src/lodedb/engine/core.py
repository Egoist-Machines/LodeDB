"""In-process LodeDB engine: per-client isolated document indexing, search, and persistence."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from ipaddress import ip_address
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from lodedb.engine.document_text_store import DocumentTextStore
from lodedb.engine.embedding_backends import (
    EngineEmbeddingBackend,
    HashEmbeddingBackend,
)
from lodedb.engine.route_registry import (
    SUPPORTED_ROUTE_CLASSES,
    RouteDecision,
    RouteRegistry,
)
from lodedb.engine.runtime_policy import (
    GpuDirectTurboVecPolicy,
    TvimDeltaPersistencePolicy,
    gpu_direct_turbovec_max_batch_from_env,
    gpu_direct_turbovec_policy_from_env,
    gpu_direct_turbovec_should_use,
    gpu_memory_budget_bytes_from_env,
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

# The optional CUDA path (`gpu_turbovec`) is imported lazily inside the methods
# that use it, so importing the engine never requires CuPy or a GPU. The
# type-only import below keeps annotations precise with no runtime dependency.
if TYPE_CHECKING:
    from lodedb.engine.gpu_turbovec import GpuDirectTurboVecSession

DIRECT_TURBOVEC_STORAGE_PROFILE = "turbovec_direct"
# Cap on the per-index in-memory query-latency ring. Latency samples are
# runtime telemetry (stats/audit percentiles), not durable state, so a long
# query stream keeps only the most recent samples rather than growing without
# bound or bloating the JSON snapshot/journal headers.
QUERY_LATENCY_SAMPLE_CAP = 1024
DEFAULT_INDEX_ID = "default"
DEFAULT_INDEX_NAME = "Default index"
ACTIVE_INDEX_STATUS = "ready"
LEGACY_INDEX_TIMESTAMP = "1970-01-01T00:00:00+00:00"
QUERY_INCLUDE_METADATA = "metadata"
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
class EngineQuery:
    """Stores one retrieval query supplied to the local engine."""

    text: str
    top_k: int = 10
    filter: dict[str, Any] | None = None
    include: tuple[str, ...] = ()
    route_drifted: bool = False
    route_failed: bool = False
    high_risk: bool = False


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
        embedding_backend: EngineEmbeddingBackend | None = None,
        route_policy: EngineRoutePolicy | None = None,
        gpu_memory_budget_bytes: int | None = None,
        turbovec_id_map_index_class: Any | None = None,
        tvim_delta_persistence_policy: TvimDeltaPersistencePolicy | None = None,
        gpu_direct_turbovec_policy: GpuDirectTurboVecPolicy | None = None,
        gpu_direct_turbovec_max_batch: int | None = None,
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
        self._pending_tvim_deltas: dict[str, dict[str, tuple[int, ...]]] = {}
        self._pending_state_journal_documents: dict[str, dict[str, dict[str, None]]] = {}
        self._turbovec_drift_telemetry: dict[str, dict[str, float]] = {}
        self._tvim_persist_telemetry: dict[str, dict[str, float | str]] = {}
        self._state_load_telemetry: dict[str, dict[str, float]] = {}
        self._indexes: dict[str, ClientIndexState] = {}
        self._turbovec_indexes: dict[str, TurboVecServingIndex] = {}
        self._index_generations: dict[str, int] = {}
        self._audit_events: list[dict[str, Any]] = []
        self._metrics: list[dict[str, Any]] = []
        if self.persistence_dir is not None:
            self.persistence_dir.mkdir(parents=True, exist_ok=True)
            self._load_persisted_indexes()

    @property
    def audit_events(self) -> tuple[dict[str, Any], ...]:
        """Returns redacted audit events emitted by endpoint handlers."""

        return tuple(dict(event) for event in self._audit_events)

    @property
    def metrics(self) -> tuple[dict[str, Any], ...]:
        """Returns metrics-only telemetry emitted by endpoint handlers."""

        return tuple(dict(metric) for metric in self._metrics)

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

    def delete_index(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Deletes one authenticated client index and any local snapshot sidecars."""

        state = self._index_for_context(context, index_id=index_id)
        if isinstance(state, EngineResponse):
            return state
        self._indexes.pop(state.index_key, None)
        self._mark_index_changed(state.index_key)
        self._pending_tvim_deltas.pop(state.index_key, None)
        self._pending_state_journal_documents.pop(state.index_key, None)
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
        result = self._ingest_documents(
            context=context,
            state=state,
            documents=documents,
            operation="upsert_documents",
            embed_batch_size=embed_batch_size,
        )
        if isinstance(result, EngineResponse):
            return result
        self._capture_document_text(state, documents)
        self._finalize_document_ingest(
            context=context,
            state=state,
            result=result,
            event_name="documents_upserted",
        )
        return EngineResponse(
            200,
            {
                "status": "upserted",
                **_document_ingest_response_payload(state, result),
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
    ) -> dict[str, float | int] | EngineResponse:
        """Validates, chunks, embeds, and applies document changes without persisting."""

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
                state.chunks[chunk_id] = IndexedChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    content_hash=chunk_hash,
                    embedding=embedding,
                )
            result["deleted_chunks"] += _delete_orphaned_document_chunks(
                state,
                document.document_id,
                list(plan.new_chunk_ids),
            )
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
        return result

    def _finalize_document_ingest(
        self,
        *,
        context: EngineRequestContext,
        state: ClientIndexState,
        result: dict[str, float | int],
        event_name: str,
    ) -> None:
        """Updates counters, syncs the direct index, and persists once."""

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
            self._sync_direct_turbovec_index(state)
            _log_document_ingest_finalize_progress(
                event_name,
                phase="turbovec_sync",
                event="completed",
                document_count=int(result["document_count"]),
                chunk_count=len(state.chunks),
                elapsed_ms=(perf_counter() - sync_started) * 1000.0,
            )
            _discard_direct_turbovec_transient_embeddings(state)
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
            deleted_chunks += self._delete_document_chunks(state, document_id)
            state.document_hashes.pop(document_id, None)
            state.document_chunk_ids.pop(document_id, None)
            state.document_metadata.pop(document_id, None)
            state.document_text.pop(document_id, None)
        self._record_pending_state_journal_documents(
            state,
            deleted_document_ids=unique_document_ids,
        )
        state.delete_count += len(unique_document_ids)
        state.deleted_chunk_count += deleted_chunks
        state.updated_at = _utc_now_iso(context.now)
        self._sync_direct_turbovec_index(state)
        self._emit(
            "documents_deleted",
            context.client_id,
            {
                "document_count": len(unique_document_ids),
                "deleted_chunks": deleted_chunks,
            },
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
        if _is_blank(query.text):
            return self._error(400, "query text is required", "query", context.client_id)
        try:
            query_filter = _validate_query_filter(query.filter)
            includes = _validate_query_includes(query.include)
        except ValueError as exc:
            return self._error(400, str(exc), "query", context.client_id)

        started_at = perf_counter()
        route = self.route_registry.select_route(
            model=state.model,
            provider=state.provider,
            task=state.task,
            drifted=query.route_drifted,
            failed=query.route_failed,
            high_risk=query.high_risk,
        )
        embedding_started = perf_counter()
        query_embedding = self._embedding_backend_for_state(state).embed_query(query.text)
        query_embedding_latency_ms = (perf_counter() - embedding_started) * 1000.0
        try:
            search_started = perf_counter()
            query_result = self._query_serving_index(
                state,
                route=route,
                query_embedding=query_embedding,
                query_filter=query_filter,
                top_k=query.top_k,
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
            if _is_blank(item.text):
                return self._error(400, "query text is required", "query_batch", context.client_id)
            try:
                query_filter = _validate_query_filter(item.filter)
                includes = _validate_query_includes(item.include)
            except ValueError as exc:
                return self._error(400, str(exc), "query_batch", context.client_id)
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
            embedding_started = perf_counter()
            query_embeddings.append(backend.embed_query(prepared_item.query.text))
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

    def list_documents(
        self,
        *,
        context: EngineRequestContext,
        index_id: str | None = None,
    ) -> EngineResponse:
        """Lists redacted document records for one authenticated client index."""

        state = self._index_for_context(context, index_id=index_id, operation="list_documents")
        if isinstance(state, EngineResponse):
            return state
        documents = [
            _document_resource_payload(state, document_id)
            for document_id in sorted(state.document_hashes)
        ]
        return EngineResponse(200, {"status": "ok", "documents": documents})

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

        state = self._index_for_context(
            context, index_id=index_id, operation="get_document_texts"
        )
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
        for path in (
            self._state_path(state.index_key),
            self._turbovec_snapshot_path(state.index_key),
        ):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
        # Delta journals sit in sidecar directories beside the base files.
        StateJournalStore(self._state_path(state.index_key)).remove_all()
        TvimDeltaStore(self._turbovec_snapshot_path(state.index_key)).remove_all()
        # The opt-in raw-text sidecar (present only when raw-text storage was
        # enabled) is removed with the rest of the index's local artifacts.
        self._document_text_store(state.index_key).remove()

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

    def _delete_document_chunks(self, state: ClientIndexState, document_id: str) -> int:
        """Removes all chunks for one document from a client-local index."""

        chunk_ids = state.document_chunk_ids.get(document_id, ())
        for chunk_id in chunk_ids:
            state.chunks.pop(chunk_id, None)
        return len(chunk_ids)

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
            # top_k to the full corpus and sorting everything.
            allowed_chunk_ids = tuple(
                str(chunk.chunk_id)
                for chunk in state.chunks.values()
                if _document_matches_query_filter(state, chunk.document_id, query_filter)
            )
            if not allowed_chunk_ids:
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
            result = index.search(
                query_embedding,
                top_k=top_k,
                allowlist_chunk_ids=allowed_chunk_ids,
            )
        else:
            result = index.search(query_embedding, top_k=top_k)
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
    ) -> tuple[Any | None, dict[str, Any]]:
        """Attempts the GPU-resident exact batch path for one direct-route batch.

        Returns ``(result, telemetry_fields)``: on success the fields carry
        admission accounting for the "used" rows; on fallback the result is
        ``None`` and the fields annotate the CPU rows with a visible
        status/reason. ``required`` raises instead of falling back (except
        for single queries, which stay CPU by design); ``off`` returns empty
        fields so CPU rows keep today's defaults. The resident session is
        generation-keyed: mutations invalidate it and the next batch
        re-uploads lazily.
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
        dependencies = gpu_direct_turbovec_dependencies()
        api_available = turbovec_reconstruction_api_available(serving.index)
        available = bool(dependencies.available and api_available)
        try:
            should_use = gpu_direct_turbovec_should_use(
                policy=policy,
                dependency_available=available,
                query_batch_size=int(batch.shape[0]),
                maximum_batch_size=self.gpu_direct_turbovec_max_batch,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "GPU direct TurboVec serving is required but unavailable: "
                + (
                    dependencies.unavailable_reason
                    or "the loaded TurboVec backend lacks the reconstruction APIs"
                )
            ) from exc
        if not should_use:
            status = "bypassed"
            reason = ""
            if int(batch.shape[0]) < 2:
                reason = "gpu_direct_batch_below_minimum"
            elif (
                policy == GpuDirectTurboVecPolicy.AUTO
                and self.gpu_direct_turbovec_max_batch is not None
                and int(batch.shape[0]) > self.gpu_direct_turbovec_max_batch
            ):
                # Auto flips to the CPU kernel above the GPU-favorable window:
                # the shared-top-k scan is faster at large batch.
                reason = "gpu_direct_batch_above_window"
            elif not dependencies.available:
                status = "failed_over"
                reason = dependencies.unavailable_reason or "gpu_dependencies_unavailable"
            elif not api_available:
                status = "failed_over"
                reason = "turbovec_reconstruction_api_unavailable"
            return None, {"gpu_stage_one_status": status, "gpu_fallback_reason": reason}
        if int(top_k) > GPU_DIRECT_TURBOVEC_MAX_TOP_K:
            # Widened post-filter top-k requests would sort and copy back
            # huge candidate sets; the CPU kernel handles them instead.
            if policy == GpuDirectTurboVecPolicy.REQUIRED:
                raise RuntimeError(
                    "GPU direct TurboVec serving is required but the effective "
                    f"top_k {int(top_k)} exceeds the GPU limit "
                    f"{GPU_DIRECT_TURBOVEC_MAX_TOP_K}"
                )
            return None, {
                "gpu_stage_one_status": "bypassed",
                "gpu_fallback_reason": "gpu_direct_top_k_exceeds_limit",
            }
        session = self._gpu_direct_turbovec_sessions.get(state.index_key)
        if (
            session is None
            or session.generation != serving.generation
            # A budget change forces re-admission, matching the V1 session
            # cache: a lowered budget must evict an oversized resident copy.
            or session.memory_budget_bytes != self.gpu_memory_budget_bytes
        ):
            try:
                session = GpuDirectTurboVecSession.build(
                    index=serving.index,
                    generation=serving.generation,
                    dependencies=dependencies,
                    memory_budget_bytes=self.gpu_memory_budget_bytes,
                )
            except MemoryError as exc:
                if policy == GpuDirectTurboVecPolicy.REQUIRED:
                    raise RuntimeError(
                        f"GPU direct TurboVec memory admission rejected: {exc}"
                    ) from exc
                return None, {
                    "gpu_stage_one_status": "memory_rejected",
                    "gpu_fallback_reason": str(exc),
                    "gpu_budget_bytes": int(self.gpu_memory_budget_bytes or 0),
                }
            except RuntimeError as exc:
                if policy == GpuDirectTurboVecPolicy.REQUIRED:
                    raise
                return None, {
                    "gpu_stage_one_status": "failed_over",
                    "gpu_fallback_reason": f"gpu_direct_build_failed: {exc}",
                }
            self._gpu_direct_turbovec_sessions[state.index_key] = session
        try:
            result = session.search_batch(batch, top_k=int(top_k))
        except Exception as exc:  # noqa: BLE001 - auto policy must fall back visibly.
            self._gpu_direct_turbovec_sessions.pop(state.index_key, None)
            if policy == GpuDirectTurboVecPolicy.REQUIRED:
                raise RuntimeError(
                    f"GPU direct TurboVec batch search failed: {exc}"
                ) from exc
            return None, {
                "gpu_stage_one_status": "failed_over",
                "gpu_fallback_reason": f"gpu_direct_runtime_error: {exc}",
            }
        return result, {
            "gpu_estimated_bytes": int(session.estimated_gpu_bytes),
            "gpu_budget_bytes": int(self.gpu_memory_budget_bytes or 0),
            "gpu_resident_upload_build_ms": float(session.upload_build_ms),
        }

    def _execute_prepared_query_batch(
        self,
        *,
        state: ClientIndexState,
        prepared: tuple[_PreparedQuery, ...],
        query_embeddings: tuple[tuple[float, ...], ...],
        query_results: list[_EngineTurboVecQueryResult | None],
        search_latencies: list[float],
    ) -> None:
        """Executes prepared direct-route queries, batching multi-query requests."""

        from lodedb.engine.gpu_turbovec import GPU_DIRECT_TURBOVEC_BACKEND

        if len(prepared) > 1:
            # One native call scores the whole batch; per-query widths are
            # sliced from the shared top-k, which is valid because the
            # kernel's ordering is deterministic for any prefix width.
            serving = self._turbovec_index_for_state(state)
            per_query_top_k = [
                _search_top_k_for_filter(
                    state, prepared_item.query.top_k, prepared_item.query_filter
                )
                for prepared_item in prepared
            ]
            batch = np.asarray(query_embeddings, dtype=np.float32)
            started = perf_counter()
            gpu_batch, gpu_fields = self._try_query_gpu_direct_turbovec_batch(
                state,
                serving=serving,
                batch=batch,
                top_k=max(per_query_top_k),
            )
            if gpu_batch is not None:
                per_query_ms = ((perf_counter() - started) * 1000.0) / max(
                    len(prepared), 1
                )
                for index, width in enumerate(per_query_top_k):
                    take = min(int(width), gpu_batch.stable_ids.shape[1])
                    query_results[index] = _EngineTurboVecQueryResult(
                        index=serving,
                        stable_ids=gpu_batch.stable_ids[index, :take].reshape(-1),
                        scores=gpu_batch.scores[index, :take].reshape(-1),
                        native_used=True,
                        native_backend=GPU_DIRECT_TURBOVEC_BACKEND,
                        retrieval_mode="direct_turbovec",
                        fallback_used=False,
                        compact_route_fallback=False,
                        stage_one_backend=GPU_DIRECT_TURBOVEC_BACKEND,
                        gpu_stage_one_status="used",
                        gpu_query_count=len(prepared),
                        gpu_copy_back_bytes=int(gpu_batch.copy_back_bytes),
                        gpu_stage_one_search_ms=float(gpu_batch.search_ms),
                        gpu_device_to_host_copy_ms=float(
                            gpu_batch.device_to_host_copy_ms
                        ),
                        gpu_stage_one_tile_count=int(gpu_batch.tile_count),
                        **gpu_fields,
                    )
                    search_latencies[index] = per_query_ms
                return
            batch_result = serving.search_batch(batch, top_k=max(per_query_top_k))
            per_query_ms = ((perf_counter() - started) * 1000.0) / max(len(prepared), 1)
            for index, width in enumerate(per_query_top_k):
                take = min(int(width), batch_result.stable_ids.shape[1])
                query_results[index] = _EngineTurboVecQueryResult(
                    index=serving,
                    stable_ids=batch_result.stable_ids[index, :take].reshape(-1),
                    scores=batch_result.scores[index, :take].reshape(-1),
                    native_used=batch_result.native_used,
                    native_backend=batch_result.native_backend,
                    retrieval_mode="direct_turbovec",
                    fallback_used=False,
                    compact_route_fallback=False,
                    **gpu_fields,
                )
                search_latencies[index] = per_query_ms
            return
        for index, (prepared_item, query_embedding) in enumerate(
            zip(prepared, query_embeddings, strict=True)
        ):
            started = perf_counter()
            query_results[index] = self._query_serving_index(
                state,
                route=prepared_item.route,
                query_embedding=query_embedding,
                query_filter=prepared_item.query_filter,
                top_k=prepared_item.query.top_k,
            )
            search_latencies[index] = (perf_counter() - started) * 1000.0
        return

    def _turbovec_index_for_state(self, state: ClientIndexState) -> TurboVecServingIndex:
        """Returns a current TurboVec IdMapIndex view for one client state."""

        generation = self._index_generations.get(state.index_key, 0)
        cached = self._turbovec_indexes.get(state.index_key)
        if cached is not None and cached.generation == generation:
            return cached
        raise RuntimeError("direct TurboVec snapshot is not loaded")

    def _sync_direct_turbovec_index(self, state: ClientIndexState) -> None:
        """Applies transient full-embedding mutations to a direct TurboVec index."""

        if not _state_uses_direct_turbovec(state):
            return
        # Try to patch the GPU-resident dequantized copy in-place; if it fails,
        # pop it and the next eligible batch lazily re-uploads against the new generation.
        gpu_session = self._gpu_direct_turbovec_sessions.get(state.index_key)
        previous = self._turbovec_indexes.get(state.index_key)
        current_chunk_ids = set(state.chunks)
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
            return
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
                previous.index.remove_many(
                    np.asarray(removed_stable_ids, dtype=np.uint64)
                )
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
        indexed_chunk_ids = set(previous.chunk_ids_by_stable_id.values())
        new_chunks = tuple(
            chunk for chunk in state.chunks.values() if chunk.chunk_id not in indexed_chunk_ids
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
            if hasattr(previous.index, "prepare"):
                _log_direct_turbovec_update_progress(
                    progress_label,
                    phase="prepare",
                    event="start",
                    chunk_count=len(new_chunks),
                    native_dim=state.native_dim,
                    bit_width=_turbovec_bit_width_for_state(state),
                    generation=generation,
                )
                started = perf_counter()
                previous.index.prepare()
                _log_direct_turbovec_update_progress(
                    progress_label,
                    phase="prepare",
                    event="end",
                    chunk_count=len(new_chunks),
                    native_dim=state.native_dim,
                    bit_width=_turbovec_bit_width_for_state(state),
                    generation=generation,
                    elapsed_ms=(perf_counter() - started) * 1000.0,
                )
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
                )
            except Exception:
                # Patch failed (e.g. MemoryError from over-allocation); safely fail closed
                # and let the next query rebuild the GPU array using safe O(N) allocation
                self._gpu_direct_turbovec_sessions.pop(state.index_key, None)
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
        self._record_turbovec_drift_telemetry(
            state,
            new_chunks=new_chunks,
            stable_ids=stable_ids if new_chunks else None,
        )

    def _record_turbovec_drift_telemetry(
        self,
        state: ClientIndexState,
        *,
        new_chunks: tuple[Any, ...],
        stable_ids: Any | None,
    ) -> None:
        """Samples allowlisted self-scores of new rows as a quantization-drift signal.

        For each sampled newly added chunk the exact self inner product is its
        embedding norm squared; the TurboVec self-score gap measures how much
        4-bit quantization moved that row. The mean relative gap is emitted on
        mutation events so operators can watch encode drift over time.
        """

        if not new_chunks or stable_ids is None:
            self._turbovec_drift_telemetry[state.index_key] = {
                "turbovec_drift_sample_rows": 0.0,
                "turbovec_self_score_drift_ratio": 0.0,
            }
            return
        serving = self._turbovec_indexes.get(state.index_key)
        if serving is None:
            return
        sample_count = min(16, len(new_chunks))
        ratios: list[float] = []
        for chunk in new_chunks[:sample_count]:
            embedding = np.asarray(chunk.embedding, dtype=np.float32)
            expected = float(np.dot(embedding, embedding))
            result = serving.search(
                tuple(float(value) for value in embedding),
                top_k=1,
                allowlist_chunk_ids=(str(chunk.chunk_id),),
            )
            if result.scores.size:
                observed = float(result.scores.reshape(-1)[0])
                ratios.append(abs(observed - expected) / max(abs(expected), 1e-9))
        self._turbovec_drift_telemetry[state.index_key] = {
            "turbovec_drift_sample_rows": float(len(ratios)),
            "turbovec_self_score_drift_ratio": float(np.mean(ratios)) if ratios else 0.0,
        }

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

    def _persist_state(self, state: ClientIndexState) -> None:
        """Writes one client index snapshot locally without raw document or query text.

        The redacted ``.json``/``.tvim``/``.jsd`` artifacts never carry raw
        text. Raw document text, when the opt-in is enabled, is written to its
        own ``.tvtext`` sidecar here so it stays durable alongside — but
        separate from — the redacted snapshot.
        """

        if self.persistence_dir is None:
            return
        self._persist_document_text(state)
        path = self._state_path(state.index_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        generation = self._index_generations.get(state.index_key, 0)
        pending_documents = self._pending_state_journal_documents.pop(state.index_key, None)
        if state.chunks:
            self._persist_direct_route_snapshots(
                state,
                path=path,
                generation=generation,
                pending_documents=pending_documents,
            )
            return
        # Empty direct index: legacy full JSON rewrite plus sidecar and
        # journal cleanup so a later cold build starts from a clean layout.
        self._write_state_json(state, path=path, generation=generation)
        self._pending_tvim_deltas.pop(state.index_key, None)
        StateJournalStore(path).remove_all()
        try:
            snapshot_path = self._turbovec_snapshot_path(state.index_key)
            snapshot_path.unlink(missing_ok=True)
            TvimDeltaStore(snapshot_path).remove_all()
        except OSError as exc:
            raise RuntimeError("stale direct TurboVec snapshot removal failed") from exc

    def _write_state_json(self, state: ClientIndexState, *, path: Path, generation: int) -> int:
        """Atomically writes the full legacy JSON snapshot and returns its byte size."""

        temporary_path = path.with_suffix(".json.tmp")
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
        temporary_path.replace(path)
        return int(path.stat().st_size)

    def _record_pending_state_journal_documents(
        self,
        state: ClientIndexState,
        *,
        upserted_document_ids: tuple[str, ...] = (),
        deleted_document_ids: tuple[str, ...] = (),
    ) -> None:
        """Accumulates direct-route document changes for the next journal append.

        Later mentions win within one batch window: an upsert clears a prior
        pending delete for the same document and vice versa, so replay applies
        only the final outcome. Standard-cascade routes are never tracked.
        """

        if not _state_uses_direct_turbovec(state):
            return
        pending = self._pending_state_journal_documents.setdefault(
            state.index_key, {"upserted": {}, "deleted": {}}
        )
        for document_id in upserted_document_ids:
            pending["deleted"].pop(document_id, None)
            pending["upserted"][document_id] = None
        for document_id in deleted_document_ids:
            pending["upserted"].pop(document_id, None)
            pending["deleted"][document_id] = None

    def _persist_direct_route_snapshots(
        self,
        state: ClientIndexState,
        *,
        path: Path,
        generation: int,
        pending_documents: dict[str, dict[str, None]] | None,
    ) -> None:
        """Persists the JSON state and `.tvim` sidecar for one direct-route index.

        Under the default ``off`` policy both files are fully rewritten
        (byte-identical legacy layout, no journal directories). Under ``auto``
        a mutation batch appends one checksumed document delta to the JSON
        journal and one encoded-row delta to the `.tvim` store using a single
        shared decision, so both sidecars journal and compact together. Cold
        builds, missing manifests, unavailable patched APIs, and either
        store's compaction threshold fall back visibly to base rewrites of
        both files.
        """

        try:
            serving = self._turbovec_index_for_state(state)
            tvim_path = self._turbovec_snapshot_path(state.index_key)
            tvim_store = TvimDeltaStore(tvim_path)
            journal = StateJournalStore(path)
            pending_tvim = self._pending_tvim_deltas.pop(state.index_key, None)
            if self.tvim_delta_persistence_policy != TvimDeltaPersistencePolicy.AUTO:
                json_bytes = self._write_state_json(state, path=path, generation=generation)
                journal.remove_all()
                temporary_path = tvim_path.with_name(tvim_path.name + ".tmp")
                serving.write(temporary_path)
                temporary_path.replace(tvim_path)
                if tvim_store.has_manifest():
                    tvim_store.remove_all()
                self._tvim_persist_telemetry[state.index_key] = {
                    "tvim_persist_mode": "full_rewrite",
                    "tvim_persist_bytes": float(tvim_path.stat().st_size),
                    "json_persist_mode": "full_rewrite",
                    "json_persist_bytes": float(json_bytes),
                }
                return
            tvim_changed = pending_tvim is not None and bool(
                pending_tvim["upserted"] or pending_tvim["removed"]
            )
            documents_changed = pending_documents is not None and bool(
                pending_documents["upserted"] or pending_documents["deleted"]
            )
            # `pending_tvim is not None` proves an incremental sync ran this
            # cycle, so the live index is base+deltas-consistent; cold rebuilds
            # pop the pending entry and must rewrite both bases.
            delta_eligible = (
                pending_tvim is not None
                and (tvim_changed or documents_changed)
                and path.exists()
                and journal.has_manifest()
                and tvim_path.exists()
                and tvim_store.has_manifest()
                and turbovec_delta_api_available(serving.index)
                and not tvim_store.should_compact()
                and not journal.should_compact()
            )
            if delta_eligible:
                assert pending_tvim is not None
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
                json_mode = "delta_append"
                json_bytes = float(json_write.file_bytes)
                json_write_ms = float(json_write.write_ms)
                # Journal ids in their exact live mutation order (deduped,
                # order-preserving). swap_remove makes slot layout depend on
                # removal order, and exactly-tied scores (duplicate chunk
                # text quantizing to identical codes) are tie-broken by slot
                # order — sorted replay produced a different tie order than
                # the live index on GovReport-scale corpora.
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
                tvim_mode = "delta_append"
            else:
                json_started = perf_counter()
                json_bytes = float(
                    self._write_state_json(state, path=path, generation=generation)
                )
                journal.record_base(
                    document_count=len(state.document_hashes),
                    chunk_count=len(state.chunks),
                )
                json_write_ms = float((perf_counter() - json_started) * 1000.0)
                json_mode = "base_rewrite"
                tvim_write = tvim_store.persist_base(serving.index)
                tvim_mode = "base_rewrite"
            self._tvim_persist_telemetry[state.index_key] = {
                "tvim_persist_mode": tvim_mode,
                "tvim_persist_bytes": float(tvim_write.file_bytes),
                "tvim_persist_write_ms": float(tvim_write.write_ms),
                **tvim_store.storage_file_bytes(),
                "json_persist_mode": json_mode,
                "json_persist_bytes": json_bytes,
                "json_persist_write_ms": json_write_ms,
                **journal.storage_file_bytes(),
            }
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError("TurboVec compact snapshot persistence failed") from exc

    def _load_persisted_indexes(self) -> None:
        """Loads persisted client index snapshots from the local engine directory."""

        if self.persistence_dir is None:
            return
        for path in sorted(self.persistence_dir.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            journal = StateJournalStore(path)
            journal_stats: dict[str, float] | None = None
            if journal.has_manifest():
                # A journal manifest means document deltas are part of the
                # persisted state regardless of the current write policy;
                # replay validates checksums and counts and fails closed.
                journal.validate_base_checksum()
                journal_stats = journal.replay_onto_payload(payload)
            state = _state_from_payload(payload)
            if journal_stats is not None:
                self._state_load_telemetry[state.index_key] = dict(journal_stats)
            self._load_document_text(state)
            self._indexes[state.index_key] = state
            generation = int(payload.get("columnar_generation", 0))
            self._index_generations[state.index_key] = generation
            self._load_turbovec_snapshot(state, generation=generation)
            self._emit_state_metric(
                "index_loaded",
                state,
                {"chunk_count": len(state.chunks)},
            )

    def _state_path(self, client_id_hash: str) -> Path:
        """Returns the local snapshot path for one hashed index key."""

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return self.persistence_dir / f"{client_id_hash}.json"

    def _turbovec_snapshot_path(self, client_id_hash: str) -> Path:
        """Returns the TurboVec IdMapIndex snapshot sidecar path for one client hash."""

        if self.persistence_dir is None:
            raise ValueError("persistence_dir is not configured")
        return self.persistence_dir / f"{client_id_hash}.tvim"

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

    def _document_text_store(self, client_id_hash: str) -> DocumentTextStore:
        """Returns the opt-in raw-text sidecar store for one client index key."""

        return DocumentTextStore(self._state_path(client_id_hash))

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

    def _persist_document_text(self, state: ClientIndexState) -> None:
        """Writes (or clears) the raw-text sidecar for one index when enabled."""

        if self.persistence_dir is None:
            return
        store = self._document_text_store(state.index_key)
        if self.raw_text_storage_enabled:
            store.write(state.document_text)
        else:
            # A disabled engine never authored text; clear any stale sidecar so
            # toggling the flag off does not silently keep old payloads on disk.
            store.remove()

    def _load_document_text(self, state: ClientIndexState) -> None:
        """Loads the raw-text sidecar into state when raw-text storage is on.

        Fails closed: a corrupt or checksum-mismatched sidecar raises rather
        than serving partial text. When raw-text storage is off the sidecar is
        ignored entirely (and never read), so the redacted load path is
        unchanged.
        """

        if self.persistence_dir is None or not self.raw_text_storage_enabled:
            return
        state.document_text = self._document_text_store(state.index_key).load()

    def _load_turbovec_snapshot(
        self,
        state: ClientIndexState,
        *,
        generation: int,
    ) -> None:
        """Loads and validates a direct TurboVec sidecar for one persisted index."""

        if self.persistence_dir is None:
            return
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
            if store.has_manifest():
                store.validate_base_checksum()

                def post_load(raw_index: Any, _store: TvimDeltaStore = store) -> None:
                    # Replay APIs are only required when segments exist; a
                    # base-only manifest (every auto base rewrite) must load
                    # on unpatched backends too.
                    if _store.has_pending_segments() and not turbovec_delta_api_available(
                        raw_index
                    ):
                        raise RuntimeError(
                            "TurboVec delta manifest present but the loaded backend "
                            "lacks the delta replay APIs"
                        )
                    _store.replay_onto(raw_index)

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
    for candidate in candidates:
        document_id = str(candidate["document_id"])
        if not _document_matches_query_filter(state, document_id, query_filter):
            continue
        if QUERY_INCLUDE_METADATA in include:
            candidate["metadata"] = dict(state.document_metadata.get(document_id, {}))
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

    elapsed = "" if elapsed_ms is None else f" elapsed_ms={elapsed_ms:.3f}"
    print(
        "turbovec_update: "
        f"label={progress_label} phase={phase} event={event} chunks={chunk_count} "
        f"native_dim={native_dim} bit_width={bit_width} generation={generation}{elapsed}",
        flush=True,
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
        (
            int(getattr(result, field_name, 0))
            for result in query_results
            if result is not None
        ),
        default=0,
    )


def _sum_query_result_int(
    query_results: list[_EngineTurboVecQueryResult | None],
    *,
    field_name: str,
) -> int:
    """Returns the summed integer telemetry value across a query result batch."""

    return sum(
        int(getattr(result, field_name, 0))
        for result in query_results
        if result is not None
    )


def _max_query_result_float(
    query_results: list[_EngineTurboVecQueryResult | None],
    *,
    field_name: str,
) -> float:
    """Returns the maximum floating-point telemetry value across a query result batch."""

    return max(
        (
            float(getattr(result, field_name, 0.0))
            for result in query_results
            if result is not None
        ),
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
            f"direct TurboVec snapshot dim {index.dim} does not match native_dim "
            f"{state.native_dim}"
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
) -> int:
    """Deletes old chunk rows for a document that are absent after a chunk-level upsert."""

    new_ids = set(new_chunk_ids)
    orphan_ids = [
        chunk_id
        for chunk_id in state.document_chunk_ids.get(document_id, ())
        if chunk_id not in new_ids
    ]
    for chunk_id in orphan_ids:
        state.chunks.pop(chunk_id, None)
    return len(orphan_ids)


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
        metadata = filter_payload["metadata"]
        if not isinstance(metadata, Mapping) or not metadata:
            raise ValueError("filter.metadata must be a nonempty object")
        validated["metadata"] = _validate_metadata(metadata)
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


def _search_top_k_for_filter(
    state: ClientIndexState,
    top_k: int,
    query_filter: Mapping[str, Any],
) -> int:
    """Returns a larger candidate count when filtering may discard top results."""

    if query_filter:
        return max(top_k, len(state.chunks))
    return top_k


def _normalize_rows_for_engine(matrix: np.ndarray) -> np.ndarray:
    """Returns row-normalized float32 data for engine-side rerank matrices."""

    rows = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norms, out=np.zeros_like(rows), where=norms > 0)


def _document_matches_query_filter(
    state: ClientIndexState,
    document_id: str,
    query_filter: Mapping[str, Any],
) -> bool:
    """Returns whether a document satisfies exact ID and metadata query filters."""

    document_ids = query_filter.get("document_ids")
    if document_ids is not None and document_id not in set(document_ids):
        return False
    metadata = query_filter.get("metadata")
    if metadata is None:
        return True
    document_metadata = state.document_metadata.get(document_id, {})
    return all(document_metadata.get(str(key)) == str(value) for key, value in metadata.items())


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
    print(
        f"{operation}: processed {processed_documents}/{total_documents} documents "
        f"({embedded_chunks} embedded chunks, {reused_chunks} reused chunks)",
        flush=True,
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
    elapsed = "" if elapsed_ms is None else f" elapsed_ms={elapsed_ms:.3f}"
    print(
        f"{operation}: phase={phase} event={event} "
        f"documents={document_count} chunks={chunk_count}{elapsed}",
        flush=True,
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


def _discard_direct_turbovec_transient_embeddings(state: ClientIndexState) -> None:
    """Drops transient full vectors after direct TurboVec has persisted them in `.tvim`."""

    zero = tuple(0.0 for _item in range(state.native_dim))
    for chunk_id, chunk in list(state.chunks.items()):
        state.chunks[chunk_id] = IndexedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            content_hash=chunk.content_hash,
            embedding=zero,
        )


def audit_persisted_index_snapshots(persistence_dir: str | Path) -> dict[str, Any]:
    """Validates local index snapshots and returns a raw-payload-free audit manifest."""

    directory = Path(persistence_dir)
    if not directory.exists():
        raise FileNotFoundError(f"index snapshot directory not found: {directory}")
    if not directory.is_dir():
        raise ValueError(f"index snapshot path is not a directory: {directory}")
    snapshot_rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        journal = StateJournalStore(path)
        journal_audit: dict[str, Any] = {}
        if journal.has_manifest():
            # Journal segments are part of the persisted state: validate
            # their checksums, scan their bodies for raw-payload leaks, and
            # replay them so the audited counts reflect the served state.
            journal.validate_base_checksum()
            for body in journal.iter_segment_bodies():
                segment_forbidden = _snapshot_forbidden_keys(body)
                if segment_forbidden:
                    keys = ", ".join(sorted(segment_forbidden))
                    raise ValueError(
                        f"{path.name} journal contains forbidden raw-payload keys: {keys}"
                    )
            replay_stats = journal.replay_onto_payload(payload)
            journal_audit = {
                **journal.storage_file_bytes(),
                "json_delta_segments_replayed": replay_stats["json_delta_segments_replayed"],
            }
        forbidden_keys = _snapshot_forbidden_keys(payload)
        if forbidden_keys:
            keys = ", ".join(sorted(forbidden_keys))
            raise ValueError(f"{path.name} contains forbidden raw-payload keys: {keys}")
        if _snapshot_contains_full_float_embeddings(payload):
            raise ValueError(f"{path.name} contains full float embedding arrays")
        if _snapshot_contains_per_vector_json_arrays(payload):
            raise ValueError(f"{path.name} contains per-vector JSON arrays")
        state = _state_from_payload(payload)
        if not _is_sha256_hex(path.stem):
            raise ValueError(f"snapshot filename must be a SHA-256 index key: {path.name}")
        if state.index_key != path.stem:
            raise ValueError(f"snapshot index_key does not match filename: {path.name}")
        tvim_path = directory / f"{state.index_key}.tvim"
        if not tvim_path.exists() and state.chunks:
            raise ValueError(f"{path.name} is missing required TurboVec sidecar")
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
        if sidecar_audit and tvim_store.has_manifest():
            # Delta segments extend the base `.tvim`; account for them so
            # the audit covers every persisted byte of the direct route.
            sidecar_audit.update(tvim_store.storage_file_bytes())
        snapshot_rows.append(
            {
                "file_name": path.name,
                "client_id_hash": state.client_id_hash,
                "index_id": state.index_id,
                "index_key": state.index_key,
                "file_sha256": _sha256_file(path),
                "document_count": len(state.document_hashes),
                "chunk_count": len(state.chunks),
                "embedding_count": len(state.chunks),
                "json_bytes": path.stat().st_size,
                "state_journal": journal_audit,
                "vector_sidecar": sidecar_audit,
                "raw_payload_text_fields_present": False,
            }
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
    json_path = engine._state_path(state.index_key)
    # The direct surface persists exactly one binary sidecar: the `.tvim`
    # TurboVec snapshot.
    sidecar_paths = (engine._turbovec_snapshot_path(state.index_key),)
    json_bytes = json_path.stat().st_size if json_path.exists() else 0
    sidecar_bytes = sum(path.stat().st_size for path in sidecar_paths if path.exists())
    # Delta journal directories extend the base files under the auto policy;
    # both report zero when absent so cold full-rewrite states are unaffected.
    journal_store = StateJournalStore(json_path)
    json_journal_bytes = 0.0
    if journal_store.has_manifest():
        accounting = journal_store.storage_file_bytes()
        json_journal_bytes = accounting["json_delta_bytes"] + accounting["json_manifest_bytes"]
    tvim_store = TvimDeltaStore(engine._turbovec_snapshot_path(state.index_key))
    tvim_delta_bytes = 0.0
    if tvim_store.has_manifest():
        accounting = tvim_store.storage_file_bytes()
        tvim_delta_bytes = accounting["tvim_delta_bytes"] + accounting["tvim_manifest_bytes"]
    # The opt-in raw-text sidecar is byte-counted (never inspected) so the
    # storage report stays honest about on-disk usage; it is zero unless
    # raw-text storage is enabled and at least one document is stored.
    raw_text_bytes = int(
        engine._document_text_store(state.index_key).storage_file_bytes()["raw_text_sidecar_bytes"]
    )
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
