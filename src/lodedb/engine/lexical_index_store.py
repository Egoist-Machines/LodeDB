"""Opt-in durable BM25 postings store with base + delta journaling.

The lexical (BM25) inverted index behind ``mode="hybrid"``/``mode="lexical"`` is
payload-derived: it is built from the per-chunk tokens of each document. By
default that index is rebuilt in memory from the retained raw text on the first
lexical query of a generation, which makes hybrid search depend on
``store_text=True`` and pay a re-tokenization cost on every reopen. Some
applications want LodeDB to keep the lexical index durable instead, so hybrid
and lexical search survive a reopen without re-reading or even retaining the raw
text (see ``README``/``docs/architecture.md``: lexical-index persistence is an
explicit opt-in).

This store provides that opt-in surface. When lexical-index persistence is
enabled, the per-document, per-chunk token lists captured at ``add`` time are
kept in a dedicated ``.tvlex`` base plus a ``.tvlex-delta/`` journal of ``.lxd``
segments, mirroring the state/vector/raw-text journals: a base holds the full
``document_id -> list[list[str]]`` map at a generation, and each mutation batch
appends one small delta segment (the upserted token lists and the deleted ids)
so a commit costs O(changed), not O(corpus). Loads replay the deltas onto the
base. Every base and segment is checksum-guarded and the journal fails closed (a
corrupt or mismatched artifact raises on load rather than serving a partial
index).

The store lives *beside* the other snapshot artifacts but is intentionally
**separate** from them: the metrics-only telemetry, the redacted JSON snapshot,
and :func:`audit_persisted_index_snapshots` never read it, so persisting the
lexical index does not weaken the raw-payload-free guarantees of those paths. It
is the lexical counterpart of the raw-text ``.tvtext`` sidecar and is
intentionally a faithful parallel of :class:`DocumentTextStore`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durable_replace

LEXICAL_INDEX_SIDECAR_SUFFIX = ".tvlex"
LEXICAL_INDEX_DELTA_DIR_SUFFIX = ".tvlex-delta"
LEXICAL_INDEX_MANIFEST_NAME = "manifest.json"
LEXICAL_INDEX_SEGMENT_SUFFIX = ".lxd"
LEXICAL_INDEX_SCHEMA_VERSION = 1

DEFAULT_MAX_DELTA_DOCUMENT_FRACTION = 0.25
DEFAULT_MAX_DELTA_SEGMENTS = 64

# One document's tokens: a list of chunks, each a list of tokens, aligned with
# the document's recorded chunk ids.
TokenLists = list[list[str]]


@dataclass(frozen=True)
class LexicalIndexWrite:
    """Reports one persisted lexical-index artifact's name, bytes, and kind."""

    file_name: str
    file_bytes: int
    kind: str


class LexicalIndexStore:
    """Manages the base ``.tvlex`` map plus ordered ``.lxd`` token deltas.

    The base is a single checksummed JSON object mapping ``document_id`` to a
    list of per-chunk token lists; each delta segment records the upserted token
    lists and deleted ids of one commit.
    """

    def __init__(self, base_path: str | Path, *, fsync: bool = False) -> None:
        """Binds the store to its base ``.tvlex`` path; the journal sits beside it."""

        self.base_path = Path(base_path)
        self._fsync = bool(fsync)
        self.delta_dir = self.base_path.with_name(
            self.base_path.name + LEXICAL_INDEX_DELTA_DIR_SUFFIX
        )

    @property
    def manifest_path(self) -> Path:
        """Returns the manifest path that anchors journal validity."""

        return self.delta_dir / LEXICAL_INDEX_MANIFEST_NAME

    def exists(self) -> bool:
        """Returns whether a base ``.tvlex`` file is present."""

        return self.base_path.is_file()

    def has_manifest(self) -> bool:
        """Returns whether a lexical-index journal manifest exists for this base path."""

        return self.manifest_path.is_file()

    def record_base(self, tokens: dict[str, TokenLists]) -> LexicalIndexWrite:
        """Writes the full ``document_id -> token lists`` map as a fresh base; clears deltas."""

        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        file_bytes = self._write_base(tokens)
        previous = self._read_manifest_optional()
        next_seq = int(previous.get("next_seq", 0)) if previous else 0
        stale_segments = self._manifest_segment_names(previous)
        self.delta_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(
            {
                "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
                "base": {
                    "file_name": self.base_path.name,
                    "sha256": _sha256_file(self.base_path),
                    "file_bytes": int(file_bytes),
                    "document_count": len(tokens),
                },
                "deltas": [],
                "next_seq": next_seq + 1,
            }
        )
        for stale in stale_segments:
            (self.delta_dir / stale).unlink(missing_ok=True)
        return LexicalIndexWrite(self.base_path.name, int(file_bytes), "base")

    def append_delta(
        self,
        *,
        upserted: dict[str, TokenLists],
        deleted: list[str],
        document_count_after: int,
    ) -> LexicalIndexWrite:
        """Journals one mutation batch's upserted token lists and deleted ids."""

        manifest = self._read_manifest()
        sequence = int(manifest.get("next_seq", 0))
        body = {
            "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
            "upserted": {str(k): _normalize_token_lists(v) for k, v in upserted.items()},
            "deleted": [str(d) for d in deleted],
        }
        body_blob = json.dumps(body, sort_keys=True).encode("utf-8")
        segment = {
            "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
            "seq": sequence,
            "document_count_after": int(document_count_after),
            "body_sha256": hashlib.sha256(body_blob).hexdigest(),
            "body": body,
        }
        self.delta_dir.mkdir(parents=True, exist_ok=True)
        segment_name = f"lexical-{sequence:08d}{LEXICAL_INDEX_SEGMENT_SUFFIX}"
        segment_path = self.delta_dir / segment_name
        temporary = segment_path.with_name(segment_path.name + ".tmp")
        temporary.write_text(json.dumps(segment, sort_keys=True), encoding="utf-8")
        durable_replace(temporary, segment_path, fsync=self._fsync)
        file_bytes = segment_path.stat().st_size
        manifest["deltas"] = list(manifest.get("deltas", [])) + [
            {
                "file_name": segment_name,
                "sha256": _sha256_file(segment_path),
                "file_bytes": int(file_bytes),
                "seq": sequence,
                "upserted": len(upserted),
                "deleted": len(deleted),
            }
        ]
        manifest["next_seq"] = sequence + 1
        self._write_manifest(manifest)
        return LexicalIndexWrite(segment_name, int(file_bytes), "delta")

    def should_compact(
        self,
        *,
        manifest: dict[str, Any] | None = None,
        max_delta_document_fraction: float = DEFAULT_MAX_DELTA_DOCUMENT_FRACTION,
        max_delta_segments: int = DEFAULT_MAX_DELTA_SEGMENTS,
    ) -> bool:
        """Returns whether the lexical-index journal backlog warrants a new base."""

        manifest = self._read_manifest_optional() if manifest is None else manifest
        if not manifest:
            return False
        deltas = manifest.get("deltas", [])
        if not deltas:
            return False
        if len(deltas) >= max_delta_segments:
            return True
        delta_documents = sum(
            int(delta.get("upserted", 0)) + int(delta.get("deleted", 0)) for delta in deltas
        )
        base_documents = max(int(manifest.get("base", {}).get("document_count", 1)), 1)
        return delta_documents >= base_documents * max_delta_document_fraction

    def load(self, *, manifest: dict[str, Any] | None = None) -> dict[str, TokenLists]:
        """Replays the journal onto the base map, failing closed on any mismatch."""

        manifest = self._read_manifest() if manifest is None else manifest
        self.validate_base_checksum(manifest=manifest)
        tokens = self._read_base()
        previous_seq = -1
        for entry in manifest.get("deltas", []):
            sequence = int(entry.get("seq", -1))
            if sequence <= previous_seq:
                raise RuntimeError("lexical index manifest has out-of-order segments")
            previous_seq = sequence
            body = _read_lexical_segment(self._validated_segment_path(entry))
            for document_id in body.get("deleted", ()):
                tokens.pop(str(document_id), None)
            for document_id, token_lists in body.get("upserted", {}).items():
                tokens[str(document_id)] = _normalize_token_lists(token_lists)
        return tokens

    def validate_base_checksum(self, *, manifest: dict[str, Any] | None = None) -> None:
        """Fails closed when the base does not match the manifest digest."""

        manifest = self._read_manifest() if manifest is None else manifest
        recorded = str(manifest.get("base", {}).get("sha256", ""))
        if recorded and (not self.base_path.is_file() or _sha256_file(self.base_path) != recorded):
            raise RuntimeError("lexical index base failed manifest checksum")

    def current_manifest(self) -> dict[str, Any] | None:
        """Returns the on-disk manifest (for embedding in the root commit manifest)."""

        return self._read_manifest_optional()

    def restore_manifest(self, manifest: dict[str, Any]) -> None:
        """Heals the on-disk manifest to a committed snapshot, dropping orphan segments."""

        self.delta_dir.mkdir(parents=True, exist_ok=True)
        referenced = set(self._manifest_segment_names(manifest))
        self._write_manifest(manifest)
        for path in self.delta_dir.glob(f"*{LEXICAL_INDEX_SEGMENT_SUFFIX}"):
            if path.name not in referenced:
                path.unlink(missing_ok=True)

    def storage_file_bytes(self, *, manifest: dict[str, Any] | None = None) -> dict[str, float]:
        """Returns base/delta/manifest byte accounting for storage telemetry."""

        manifest = (self._read_manifest_optional() if manifest is None else manifest) or {}
        base_bytes = float(manifest.get("base", {}).get("file_bytes", 0))
        deltas = manifest.get("deltas", [])
        delta_bytes = float(sum(int(delta.get("file_bytes", 0)) for delta in deltas))
        manifest_bytes = (
            float(self.manifest_path.stat().st_size) if self.manifest_path.is_file() else 0.0
        )
        return {"lexical_index_sidecar_bytes": base_bytes + delta_bytes + manifest_bytes}

    def remove_all(self) -> None:
        """Removes the base, manifest, and all delta segments for this store."""

        self.base_path.unlink(missing_ok=True)
        if not self.delta_dir.is_dir():
            return
        for path in self.delta_dir.glob(f"*{LEXICAL_INDEX_SEGMENT_SUFFIX}"):
            path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)

    def _write_base(self, tokens: dict[str, TokenLists]) -> int:
        """Atomically writes the full ``document_id -> token lists`` map; returns byte size."""

        body = {
            "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
            "documents": {
                str(key): _normalize_token_lists(value) for key, value in sorted(tokens.items())
            },
        }
        body_blob = json.dumps(body, sort_keys=True).encode("utf-8")
        payload = {
            "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
            "body_sha256": hashlib.sha256(body_blob).hexdigest(),
            "body": body,
        }
        temporary = self.base_path.with_name(self.base_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        durable_replace(temporary, self.base_path, fsync=self._fsync)
        return int(self.base_path.stat().st_size)

    def _read_base(self) -> dict[str, TokenLists]:
        """Reads and checksum-validates the base map, failing closed on mismatch."""

        if not self.base_path.is_file():
            return {}
        try:
            payload = json.loads(self.base_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            raise RuntimeError("lexical index base is corrupt") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("lexical index base is not a JSON object")
        if int(payload.get("schema_version", -1)) != LEXICAL_INDEX_SCHEMA_VERSION:
            raise RuntimeError("unsupported lexical index base schema version")
        body = payload.get("body")
        if not isinstance(body, dict):
            raise RuntimeError("lexical index base body is missing")
        body_blob = json.dumps(body, sort_keys=True).encode("utf-8")
        if hashlib.sha256(body_blob).hexdigest() != str(payload.get("body_sha256", "")):
            raise RuntimeError("lexical index base failed checksum")
        documents = body.get("documents", {})
        if not isinstance(documents, dict):
            raise RuntimeError("lexical index base documents must be an object")
        return {str(key): _normalize_token_lists(value) for key, value in documents.items()}

    def _validated_segment_path(self, entry: dict[str, Any]) -> Path:
        """Returns a manifest segment's path after existence and checksum checks."""

        file_name = str(entry.get("file_name", ""))
        path = self.delta_dir / file_name
        if not file_name or not path.is_file():
            raise RuntimeError(f"lexical index segment is missing: {file_name}")
        if _sha256_file(path) != str(entry.get("sha256", "")):
            raise RuntimeError(f"lexical index segment failed checksum: {file_name}")
        return path

    def _read_manifest(self) -> dict[str, Any]:
        """Reads and validates the manifest, failing closed when absent."""

        manifest = self._read_manifest_optional()
        if manifest is None:
            raise RuntimeError("lexical index manifest is missing")
        return manifest

    def _read_manifest_optional(self) -> dict[str, Any] | None:
        """Reads the manifest when present, validating its schema version."""

        if not self.manifest_path.is_file():
            return None
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != LEXICAL_INDEX_SCHEMA_VERSION:
            raise RuntimeError("unsupported lexical index manifest schema version")
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """Atomically replaces the manifest after its segments are durable."""

        temporary = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        durable_replace(temporary, self.manifest_path, fsync=self._fsync)

    def _manifest_segment_names(self, manifest: dict[str, Any] | None) -> list[str]:
        """Lists lexical-index segment file names referenced by a manifest."""

        if not manifest:
            return []
        return [
            str(delta["file_name"])
            for delta in manifest.get("deltas", [])
            if delta.get("file_name")
        ]


def _normalize_token_lists(value: Any) -> TokenLists:
    """Coerces a stored/loaded value into a list of per-chunk token lists.

    Each chunk is a list of string tokens; a malformed inner value raises so the
    journal fails closed rather than serving a partial index.
    """

    if not isinstance(value, list):
        raise RuntimeError("lexical index token lists must be a list of chunks")
    chunks: TokenLists = []
    for chunk in value:
        if not isinstance(chunk, list):
            raise RuntimeError("lexical index chunk must be a list of tokens")
        chunks.append([str(token) for token in chunk])
    return chunks


def _read_lexical_segment(path: Path) -> dict[str, Any]:
    """Parses and checksum-validates one ``.lxd`` lexical segment file."""

    try:
        segment = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(f"lexical index segment is corrupt: {path.name}") from exc
    if int(segment.get("schema_version", -1)) != LEXICAL_INDEX_SCHEMA_VERSION:
        raise RuntimeError("unsupported lexical index segment schema version")
    body = segment.get("body")
    if not isinstance(body, dict):
        raise RuntimeError(f"lexical index segment body is missing: {path.name}")
    body_blob = json.dumps(body, sort_keys=True).encode("utf-8")
    if hashlib.sha256(body_blob).hexdigest() != str(segment.get("body_sha256", "")):
        raise RuntimeError(f"lexical index segment failed checksum: {path.name}")
    if not isinstance(body.get("upserted", {}), dict) or not isinstance(
        body.get("deleted", []), list
    ):
        raise RuntimeError(f"lexical index segment has malformed body: {path.name}")
    return body


def _sha256_file(path: Path) -> str:
    """Returns the sha256 hex digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
