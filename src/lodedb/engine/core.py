"""Shared LodeDB engine dataclasses, route policy, and persisted-snapshot auditing.

The native core (``lodedb._native_core``) is the sole reader/writer for a
LodeDB handle; this module holds the engine-facing data contracts the SDK and
adapter exchange (the ``Engine*`` dataclasses, :class:`EngineRoutePolicy`), the
small validation/identity helpers they share, and the disk-only snapshot auditor
(:func:`audit_persisted_index_snapshots`) used by the migration tooling.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from ipaddress import ip_address
from pathlib import Path
from typing import Any

from lodedb.engine._commit_manifest import (
    COMMIT_MANIFEST_SUFFIX,
    base_json_path,
    base_tvim_path,
    is_commit_manifest_name,
    read_commit_manifest,
)
from lodedb.engine.route_registry import SUPPORTED_ROUTE_CLASSES, RouteDecision
from lodedb.engine.state_journal_store import StateJournalStore
from lodedb.engine.turbovec_delta_store import TvimDeltaStore

DIRECT_TURBOVEC_STORAGE_PROFILE = "turbovec_direct"

_NON_INDEX_JSON_FILES = frozenset({"collection.json", "migration.json"})

QUERY_LATENCY_SAMPLE_CAP = 1024

DEFAULT_INDEX_ID = "default"

DEFAULT_INDEX_NAME = "Default index"

ACTIVE_INDEX_STATUS = "ready"

LEGACY_INDEX_TIMESTAMP = "1970-01-01T00:00:00+00:00"

RETRIEVAL_MODE_VECTOR = "vector"


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


def sha256_text(value: str) -> str:
    """Returns a stable SHA-256 hex digest for a UTF-8 string."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def index_state_key_for_client_hash(client_id_hash: str, index_id: str) -> str:
    """Returns the persisted state key while preserving legacy default snapshot names."""

    if index_id == DEFAULT_INDEX_ID:
        return client_id_hash
    return sha256_text(f"{client_id_hash}:{index_id}")


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


def storage_profile_for_route_policy(route_policy: EngineRoutePolicy | None) -> str:
    """Returns the storage profile implied by the selected route policy."""

    if route_policy is not None and route_policy.index_backend != DIRECT_TURBOVEC_STORAGE_PROFILE:
        raise ValueError(f"unsupported index backend: {route_policy.index_backend}")
    return DIRECT_TURBOVEC_STORAGE_PROFILE


def normalized_chunk_hash(text: str) -> str:
    """Hashes normalized chunk text so harmless whitespace changes can reuse embeddings."""

    return sha256_text(" ".join(text.split()))


def _chunk_id_for_hash(document_id: str, *, chunk_hash: str, occurrence: int) -> str:
    """Builds a stable chunk ID from document ID, normalized chunk hash, and occurrence."""

    return f"{document_id}:{chunk_hash[:12]}:{occurrence:04d}"


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
        if (
            is_commit_manifest_name(path.name)
            or path.name in _NON_INDEX_JSON_FILES
            or path.stem in audited_keys
        ):
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


def _storage_profile_from_payload(value: object) -> str:
    """Restores a persisted redacted storage profile with legacy standard fallback."""

    text = str(value)
    if text == DIRECT_TURBOVEC_STORAGE_PROFILE:
        return text
    raise ValueError(
        "storage_profile must be turbovec_direct; standard-cascade and "
        "extreme_compact snapshots are retired"
    )
