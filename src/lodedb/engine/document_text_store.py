"""Opt-in durable raw document-text sidecar for direct-route index snapshots.

By default LodeDB is raw-payload-free: the `.json` state snapshot, the `.jsd`
state journal, the `.tvim`/`.tvd` vector sidecars, telemetry, and audit events
all store ids, counts, hashes, metadata, and compact codes — never raw document
or query text. Some applications, however, want LodeDB itself to be the durable
store for the source text keyed by document id, instead of maintaining a second
store of their own (see ``README``/``docs/architecture.md``: raw-text
persistence is an explicit opt-in).

This store provides exactly that opt-in surface. When (and only when) raw-text
storage is enabled for an engine, the original text supplied to ``add`` is kept
in a dedicated ``<index_key>.tvtext`` sidecar that maps ``document_id -> text``.
The sidecar lives *beside* the other snapshot artifacts but is intentionally
**separate** from them: the metrics-only telemetry, the redacted JSON snapshot,
and :func:`audit_persisted_index_snapshots` never read it, so enabling raw-text
retrieval does not weaken the raw-payload-free guarantees of those paths — it
just adds one clearly named file whose entire purpose is to hold the text the
caller explicitly asked LodeDB to retain.

The file is a single JSON object written atomically (temp file + ``os.replace``)
and checksum-guarded by a small manifest, mirroring the fail-closed convention
of :mod:`lodedb.engine.state_journal_store`: a corrupt or mismatched sidecar
raises on load rather than serving partial text.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durable_replace

DOCUMENT_TEXT_SIDECAR_SUFFIX = ".tvtext"
DOCUMENT_TEXT_SCHEMA_VERSION = 1


class DocumentTextStore:
    """Manages one index's opt-in ``document_id -> raw text`` sidecar file."""

    def __init__(self, base_path: str | Path, *, fsync: bool = False) -> None:
        """Binds the store to a base ``.json`` path; the sidecar sits beside it.

        ``fsync`` makes the published sidecar power-loss durable (the engine's
        ``durability="fsync"`` mode); the default keeps the fast atomic-rename
        path.
        """

        base = Path(base_path)
        self.base_path = base
        self._fsync = bool(fsync)
        self.sidecar_path = base.with_name(base.stem + DOCUMENT_TEXT_SIDECAR_SUFFIX)

    def exists(self) -> bool:
        """Returns whether a raw-text sidecar file is present for this index."""

        return self.sidecar_path.is_file()

    def write(self, texts: dict[str, str]) -> int:
        """Atomically writes the full ``document_id -> text`` map; returns byte size.

        An empty map removes the sidecar so an index with raw-text storage on
        but no documents leaves no stray file behind.
        """

        if not texts:
            self.remove()
            return 0
        body = {
            "schema_version": DOCUMENT_TEXT_SCHEMA_VERSION,
            "documents": {str(key): str(value) for key, value in sorted(texts.items())},
        }
        body_blob = json.dumps(body, sort_keys=True).encode("utf-8")
        payload = {
            "schema_version": DOCUMENT_TEXT_SCHEMA_VERSION,
            "body_sha256": hashlib.sha256(body_blob).hexdigest(),
            "body": body,
        }
        temporary = self.sidecar_path.with_name(self.sidecar_path.name + ".tmp")
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        durable_replace(temporary, self.sidecar_path, fsync=self._fsync)
        return int(self.sidecar_path.stat().st_size)

    def load(self) -> dict[str, str]:
        """Loads and checksum-validates the sidecar, failing closed on mismatch."""

        if not self.sidecar_path.is_file():
            return {}
        try:
            payload = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
            # Fail closed like the other persistence sidecars: a corrupt file
            # raises a RuntimeError on reopen rather than serving partial text.
            raise RuntimeError("document text sidecar is corrupt") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("document text sidecar is not a JSON object")
        if int(payload.get("schema_version", -1)) != DOCUMENT_TEXT_SCHEMA_VERSION:
            raise RuntimeError("unsupported document text sidecar schema version")
        body = payload.get("body")
        if not isinstance(body, dict):
            raise RuntimeError("document text sidecar body is missing")
        body_blob = json.dumps(body, sort_keys=True).encode("utf-8")
        if hashlib.sha256(body_blob).hexdigest() != str(payload.get("body_sha256", "")):
            raise RuntimeError("document text sidecar failed checksum")
        documents = body.get("documents", {})
        if not isinstance(documents, dict):
            raise RuntimeError("document text sidecar documents must be an object")
        return {str(key): str(value) for key, value in documents.items()}

    def remove(self) -> None:
        """Removes the sidecar file when present (e.g. raw-text storage disabled)."""

        self.sidecar_path.unlink(missing_ok=True)

    def storage_file_bytes(self) -> dict[str, Any]:
        """Returns raw-text sidecar byte accounting for storage telemetry."""

        size = float(self.sidecar_path.stat().st_size) if self.sidecar_path.is_file() else 0.0
        return {"raw_text_sidecar_bytes": size}
