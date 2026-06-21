"""Segmented base+delta journaling for direct-route JSON state snapshots.

Direct TurboVec route states keep document/chunk metadata in one JSON
snapshot (`<index_key>.json`). Rewriting that file costs O(corpus) per
mutation batch (~450 MB at 1M chunks), which dominates mutation persist
time once `.tvim` writes are delta-journaled. This store keeps the
existing JSON snapshot as the base segment and appends small `.jsd`
delta segments under `<index_key>.json-delta/`, so persisting a mutation
batch costs O(changed documents): each delta journals the upserted
documents' current redacted rows (document hash, ordered chunk ids,
chunk reference rows, validated metadata), the deleted document ids, and
the refreshed scalar state header (counters, timestamps, generation).

Loads replay deltas in order onto the parsed base payload before state
restoration. Every segment is checksumed (whole file in the manifest,
body separately in the segment header); replay fails closed on
checksum, sequence, or count mismatches so a corrupt or
mismatched journal can never serve quietly. The base JSON snapshot stays
byte-identical to the legacy full-rewrite layout, and the `off` policy
never creates the journal directory.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durable_replace

STATE_JOURNAL_DIR_SUFFIX = ".json-delta"
STATE_JOURNAL_MANIFEST_NAME = "manifest.json"
STATE_JOURNAL_SEGMENT_SUFFIX = ".jsd"
STATE_JOURNAL_MAGIC = b"EEJSD001"
STATE_JOURNAL_SCHEMA_VERSION = 1
_HEADER_LENGTH = struct.Struct("<Q")

DEFAULT_MAX_DELTA_DOCUMENT_FRACTION = 0.25
DEFAULT_MAX_DELTA_SEGMENTS = 64

# Top-level snapshot keys replaced wholesale by each delta's state header;
# document collections replay per document instead, and per-query latency
# samples are in-memory runtime telemetry that must never ride in (and grow)
# the delta headers.
STATE_HEADER_EXCLUDED_KEYS = (
    "chunks",
    "document_hashes",
    "document_chunk_ids",
    "document_metadata",
    "query_latency_ms",
)


@dataclass(frozen=True)
class StateJournalWrite:
    """Reports one persisted journal artifact's name, bytes, and elapsed time."""

    file_name: str
    file_bytes: int
    write_ms: float
    kind: str


class StateJournalStore:
    """Manages the base JSON snapshot plus ordered `.jsd` document deltas."""

    def __init__(self, base_path: str | Path, *, fsync: bool = False) -> None:
        """Binds the store to the base `.json` path; the journal dir sits beside it.

        ``fsync`` makes each published segment and manifest power-loss durable
        (the engine's ``durability="fsync"`` mode); the default keeps the fast
        atomic-rename path.
        """

        self.base_path = Path(base_path)
        self._fsync = bool(fsync)
        self.journal_dir = self.base_path.with_name(
            self.base_path.name + STATE_JOURNAL_DIR_SUFFIX
        )

    @property
    def manifest_path(self) -> Path:
        """Returns the manifest path that anchors journal validity."""

        return self.journal_dir / STATE_JOURNAL_MANIFEST_NAME

    def has_manifest(self) -> bool:
        """Returns whether a journal manifest exists for this base path."""

        return self.manifest_path.is_file()

    def record_base(
        self,
        *,
        document_count: int,
        chunk_count: int,
    ) -> StateJournalWrite:
        """Anchors an already-written base JSON snapshot and clears the backlog.

        The caller writes the base file itself (keeping legacy full-rewrite
        bytes identical); this records its checksum and resets the deltas.
        """

        started = time.perf_counter()
        if not self.base_path.is_file():
            raise RuntimeError("state journal base snapshot is missing")
        file_bytes = self.base_path.stat().st_size
        previous = self._read_manifest_optional()
        next_seq = int(previous.get("next_seq", 0)) if previous else 0
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        stale_segments = self._manifest_segment_names(previous)
        self._write_manifest(
            {
                "schema_version": STATE_JOURNAL_SCHEMA_VERSION,
                "base": {
                    "file_name": self.base_path.name,
                    "sha256": _sha256_file(self.base_path),
                    "file_bytes": int(file_bytes),
                    "document_count": int(document_count),
                    "chunk_count": int(chunk_count),
                },
                "deltas": [],
                "next_seq": next_seq + 1,
            }
        )
        for stale in stale_segments:
            (self.journal_dir / stale).unlink(missing_ok=True)
        return StateJournalWrite(
            file_name=self.base_path.name,
            file_bytes=int(file_bytes),
            write_ms=float((time.perf_counter() - started) * 1000.0),
            kind="base",
        )

    def append_delta(
        self,
        *,
        upserted_documents: list[dict[str, Any]],
        deleted_document_ids: list[str],
        state_header: dict[str, Any],
        document_count_after: int,
        chunk_count_after: int,
        generation: int,
    ) -> StateJournalWrite:
        """Journals one mutation batch's document rows and refreshed header."""

        started = time.perf_counter()
        manifest = self._read_manifest()
        sequence = int(manifest.get("next_seq", 0))
        body = {
            "schema_version": STATE_JOURNAL_SCHEMA_VERSION,
            "upserted_documents": upserted_documents,
            "deleted_document_ids": list(deleted_document_ids),
            "state_header": dict(state_header),
        }
        body_blob = json.dumps(body, sort_keys=True).encode("utf-8")
        header = {
            "schema_version": STATE_JOURNAL_SCHEMA_VERSION,
            "kind": "delta",
            "seq": sequence,
            "generation_after": int(generation),
            "document_count_after": int(document_count_after),
            "chunk_count_after": int(chunk_count_after),
            "body_bytes": len(body_blob),
            "body_sha256": hashlib.sha256(body_blob).hexdigest(),
        }
        header_blob = json.dumps(header, sort_keys=True).encode("utf-8")
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        segment_name = f"delta-{sequence:08d}{STATE_JOURNAL_SEGMENT_SUFFIX}"
        segment_path = self.journal_dir / segment_name
        temporary = segment_path.with_name(segment_path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(STATE_JOURNAL_MAGIC)
            handle.write(_HEADER_LENGTH.pack(len(header_blob)))
            handle.write(header_blob)
            handle.write(body_blob)
        durable_replace(temporary, segment_path, fsync=self._fsync)
        file_bytes = segment_path.stat().st_size
        manifest["deltas"] = list(manifest.get("deltas", [])) + [
            {
                "file_name": segment_name,
                "sha256": _sha256_file(segment_path),
                "file_bytes": int(file_bytes),
                "seq": sequence,
                "upserted_documents": len(upserted_documents),
                "deleted_documents": len(deleted_document_ids),
            }
        ]
        manifest["next_seq"] = sequence + 1
        self._write_manifest(manifest)
        return StateJournalWrite(
            file_name=segment_name,
            file_bytes=int(file_bytes),
            write_ms=float((time.perf_counter() - started) * 1000.0),
            kind="delta",
        )

    def should_compact(
        self,
        *,
        max_delta_document_fraction: float = DEFAULT_MAX_DELTA_DOCUMENT_FRACTION,
        max_delta_segments: int = DEFAULT_MAX_DELTA_SEGMENTS,
    ) -> bool:
        """Returns whether the journal backlog warrants folding into a new base."""

        manifest = self._read_manifest_optional()
        if not manifest:
            return False
        deltas = manifest.get("deltas", [])
        if not deltas:
            return False
        if len(deltas) >= max_delta_segments:
            return True
        delta_documents = sum(
            int(delta.get("upserted_documents", 0)) + int(delta.get("deleted_documents", 0))
            for delta in deltas
        )
        base_documents = max(int(manifest.get("base", {}).get("document_count", 1)), 1)
        return delta_documents >= base_documents * max_delta_document_fraction

    def replay_onto_payload(self, payload: dict[str, Any]) -> dict[str, float]:
        """Replays manifest deltas onto a parsed base payload, failing closed.

        Mutates ``payload`` in place to the post-journal state: document
        collections replay per document and the scalar header is replaced by
        the last delta's header.
        """

        started = time.perf_counter()
        manifest = self._read_manifest()
        chunks_by_id: dict[str, dict[str, Any]] = {
            str(row["chunk_id"]): row for row in payload.get("chunks", ())
        }
        document_hashes = {
            str(key): str(value)
            for key, value in dict(payload.get("document_hashes", {}) or {}).items()
        }
        document_chunk_ids = {
            str(key): list(value)
            for key, value in dict(payload.get("document_chunk_ids", {}) or {}).items()
        }
        document_metadata = {
            str(key): dict(value or {})
            for key, value in dict(payload.get("document_metadata", {}) or {}).items()
        }
        previous_seq = -1
        replayed_upserts = 0
        replayed_deletes = 0
        last_state_header: dict[str, Any] | None = None
        for entry in manifest.get("deltas", []):
            sequence = int(entry.get("seq", -1))
            if sequence <= previous_seq:
                raise RuntimeError("state journal manifest has out-of-order segments")
            previous_seq = sequence
            segment = _read_journal_segment(self._validated_segment_path(entry))
            header = segment["header"]
            body = segment["body"]
            for document_id in body.get("deleted_document_ids", ()):
                document_id = str(document_id)
                for chunk_id in document_chunk_ids.pop(document_id, ()):
                    chunks_by_id.pop(str(chunk_id), None)
                document_hashes.pop(document_id, None)
                document_metadata.pop(document_id, None)
                replayed_deletes += 1
            for document in body.get("upserted_documents", ()):
                document_id = str(document["document_id"])
                for chunk_id in document_chunk_ids.get(document_id, ()):
                    chunks_by_id.pop(str(chunk_id), None)
                for row in document.get("chunks", ()):
                    chunks_by_id[str(row["chunk_id"])] = dict(row)
                document_hashes[document_id] = str(document["document_hash"])
                document_chunk_ids[document_id] = [
                    str(chunk_id) for chunk_id in document.get("chunk_ids", ())
                ]
                document_metadata[document_id] = dict(document.get("metadata", {}) or {})
                replayed_upserts += 1
            expected_documents = int(header.get("document_count_after", -1))
            if expected_documents >= 0 and len(document_hashes) != expected_documents:
                raise RuntimeError("state journal replay rejected: document count mismatch")
            expected_chunks = int(header.get("chunk_count_after", -1))
            if expected_chunks >= 0 and len(chunks_by_id) != expected_chunks:
                raise RuntimeError("state journal replay rejected: chunk count mismatch")
            last_state_header = dict(body.get("state_header", {}) or {})
        if last_state_header is not None:
            for key in STATE_HEADER_EXCLUDED_KEYS:
                last_state_header.pop(key, None)
            payload.update(last_state_header)
        payload["chunks"] = sorted(chunks_by_id.values(), key=lambda row: str(row["chunk_id"]))
        payload["document_hashes"] = dict(sorted(document_hashes.items()))
        payload["document_chunk_ids"] = {
            document_id: list(chunk_ids)
            for document_id, chunk_ids in sorted(document_chunk_ids.items())
        }
        payload["document_metadata"] = {
            document_id: dict(sorted(metadata.items()))
            for document_id, metadata in sorted(document_metadata.items())
        }
        return {
            "json_replay_ms": float((time.perf_counter() - started) * 1000.0),
            "json_delta_segments_replayed": float(len(manifest.get("deltas", []))),
            "json_replayed_upserted_documents": float(replayed_upserts),
            "json_replayed_deleted_documents": float(replayed_deletes),
        }

    def validate_base_checksum(self) -> None:
        """Fails closed when the base JSON does not match the manifest digest."""

        manifest = self._read_manifest()
        base_entry = manifest.get("base", {})
        recorded = str(base_entry.get("sha256", ""))
        if recorded and _sha256_file(self.base_path) != recorded:
            raise RuntimeError("state journal base snapshot failed manifest checksum")

    def iter_segment_bodies(self) -> list[dict[str, Any]]:
        """Returns validated segment bodies in order for audit inspection."""

        manifest = self._read_manifest()
        bodies: list[dict[str, Any]] = []
        for entry in manifest.get("deltas", []):
            segment = _read_journal_segment(self._validated_segment_path(entry))
            bodies.append(segment["body"])
        return bodies

    def storage_file_bytes(self) -> dict[str, float]:
        """Returns manifest/base/delta byte accounting for telemetry."""

        manifest = self._read_manifest_optional() or {}
        base_bytes = float(manifest.get("base", {}).get("file_bytes", 0))
        deltas = manifest.get("deltas", [])
        delta_bytes = float(sum(int(delta.get("file_bytes", 0)) for delta in deltas))
        manifest_bytes = (
            float(self.manifest_path.stat().st_size) if self.manifest_path.is_file() else 0.0
        )
        return {
            "json_base_bytes": base_bytes,
            "json_delta_bytes": delta_bytes,
            "json_delta_segments": float(len(deltas)),
            "json_manifest_bytes": manifest_bytes,
        }

    def remove_all(self) -> None:
        """Removes the manifest and all journal segments (e.g. policy off)."""

        if not self.journal_dir.is_dir():
            return
        for path in self.journal_dir.glob(f"*{STATE_JOURNAL_SEGMENT_SUFFIX}"):
            path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)

    def _validated_segment_path(self, entry: dict[str, Any]) -> Path:
        """Returns a manifest segment's path after existence and checksum checks."""

        file_name = str(entry.get("file_name", ""))
        path = self.journal_dir / file_name
        if not file_name or not path.is_file():
            raise RuntimeError(f"state journal segment is missing: {file_name}")
        if _sha256_file(path) != str(entry.get("sha256", "")):
            raise RuntimeError(f"state journal segment failed checksum: {file_name}")
        return path

    def _read_manifest(self) -> dict[str, Any]:
        """Reads and validates the manifest, failing closed when absent."""

        manifest = self._read_manifest_optional()
        if manifest is None:
            raise RuntimeError("state journal manifest is missing")
        return manifest

    def _read_manifest_optional(self) -> dict[str, Any] | None:
        """Reads the manifest when present, validating its schema version."""

        if not self.manifest_path.is_file():
            return None
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != STATE_JOURNAL_SCHEMA_VERSION:
            raise RuntimeError("unsupported state journal manifest schema version")
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """Atomically replaces the manifest after its segments are durable."""

        temporary = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        durable_replace(temporary, self.manifest_path, fsync=self._fsync)

    def _manifest_segment_names(self, manifest: dict[str, Any] | None) -> list[str]:
        """Lists journal segment file names referenced by a manifest."""

        if not manifest:
            return []
        return [
            str(delta["file_name"])
            for delta in manifest.get("deltas", [])
            if delta.get("file_name")
        ]


def _read_journal_segment(path: Path) -> dict[str, Any]:
    """Parses and checksum-validates one `.jsd` journal segment file."""

    data = path.read_bytes()
    prefix = len(STATE_JOURNAL_MAGIC) + _HEADER_LENGTH.size
    if len(data) < prefix or data[: len(STATE_JOURNAL_MAGIC)] != STATE_JOURNAL_MAGIC:
        raise RuntimeError(f"not a state journal segment: {path.name}")
    header_length = _HEADER_LENGTH.unpack_from(data, len(STATE_JOURNAL_MAGIC))[0]
    header_stop = prefix + header_length
    if len(data) < header_stop:
        raise RuntimeError(f"state journal segment header is truncated: {path.name}")
    header = json.loads(data[prefix:header_stop].decode("utf-8"))
    if int(header.get("schema_version", -1)) != STATE_JOURNAL_SCHEMA_VERSION:
        raise RuntimeError("unsupported state journal segment schema version")
    body_blob = data[header_stop : header_stop + int(header.get("body_bytes", -1))]
    if len(body_blob) != int(header.get("body_bytes", -1)):
        raise RuntimeError(f"state journal segment body is truncated: {path.name}")
    if hashlib.sha256(body_blob).hexdigest() != str(header.get("body_sha256", "")):
        raise RuntimeError(f"state journal segment body failed checksum: {path.name}")
    body = json.loads(body_blob.decode("utf-8"))
    required = {"upserted_documents", "deleted_document_ids", "state_header"}
    missing = required - set(body)
    if missing:
        raise RuntimeError(f"state journal segment is missing fields: {sorted(missing)}")
    return {"header": header, "body": body}


def _sha256_file(path: Path) -> str:
    """Returns the sha256 hex digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
