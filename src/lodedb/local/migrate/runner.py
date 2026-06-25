"""The migration runner: build a plan, dry-run it, run it safely, validate it.

The runner is the inspect -> plan -> dry-run -> run -> validate spine shared by both
issues. It is deliberately small and explicit about safety:

- **Source is read-only.** It only ever constructs a
  :class:`~lodedb.local.migrate.sources.base.SourceExport` and iterates it.
- **Write to a temp, then move.** A real run writes the new store to
  ``<target>.tmp``, reopens it read-only, validates it, runs the persisted-index
  audit, and only then atomically moves ``<target>.tmp`` into ``<target>``.
- **Never clobber silently.** A run refuses when the target already holds a LodeDB
  store unless ``--overwrite-target`` (or ``--resume``) is set.
- **Payload-free manifest.** It writes ``migration.json`` with counts, dimensions,
  fingerprints, options, skipped-row reasons (by id hash), validation results,
  versions, and timestamps — never raw text, vectors, payloads, or credentials.

The runner picks the *target writer* from the plan's route and mode: a framework
route replays through that framework's shipped LodeDB adapter (so the on-disk shape
matches what the app will read), and a direct provider writes vectors through
``LodeDB.open_vector_store`` (vector-preserve) or text through ``LodeDB`` (text
replay).
"""

from __future__ import annotations

import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lodedb.engine.core import audit_persisted_index_snapshots
from lodedb.local.db import LodeDB
from lodedb.local.migrate.detect import Detection, inspect_project
from lodedb.local.migrate.plan import (
    MigrationPlan,
    build_switch_snippet,
    default_rollback,
)
from lodedb.local.migrate.report import (
    fingerprint_text,
    hash_id,
    sample_indices,
)
from lodedb.local.migrate.sources.base import (
    MODE_TEXT_REPLAY,
    MODE_VECTOR_PRESERVE,
    ExportedRow,
    SourceExport,
)

# LodeDB vector indexes require a dimension that is a positive multiple of 8.
_VECTOR_DIM_MULTIPLE = 8


class MigrationError(RuntimeError):
    """Raised when a migration cannot proceed safely (e.g. target collision)."""


@dataclass
class MigrationResult:
    """The payload-free outcome of a run (also the basis of ``migration.json``)."""

    status: str  # "dry-run" | "migrated" | "failed"
    route: str
    mode: str
    framework: str | None
    provider: str | None
    source_kind: str
    source_location_fingerprint: str
    target_path: str
    embedding_dim: int | None
    source_count: int | None
    store_subdir: str | None = None
    written_count: int = 0
    skipped: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    versions: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_manifest(self) -> dict[str, Any]:
        """Renders the payload-free ``migration.json`` manifest."""

        return {
            "artifact_type": "lodedb_migration",
            "status": self.status,
            "route": self.route,
            "mode": self.mode,
            "framework": self.framework,
            "provider": self.provider,
            "source": {
                "kind": self.source_kind,
                "location_fingerprint": self.source_location_fingerprint,
                "count": self.source_count,
            },
            "target": {
                "path": str(self.target_path),
                "store_subdir": self.store_subdir,
                "embedding_dim": self.embedding_dim,
                "written_count": self.written_count,
            },
            "skipped": self.skipped,
            "validation": self.validation,
            "warnings": self.warnings,
            "versions": self.versions,
            "options": self.options,
            "timings": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "seconds": round(self.finished_at - self.started_at, 6)
                if self.finished_at
                else 0.0,
            },
        }


def _lodedb_versions() -> dict[str, Any]:
    """Returns the redacted version block for the manifest."""

    import lodedb

    return {"lodedb": getattr(lodedb, "__version__", "unknown"), "plan_version": 1}


def target_has_store(target: Path) -> bool:
    """Returns True if ``target`` already contains a LodeDB store (commit manifest)."""

    if not target.is_dir():
        return False
    return any(target.glob("*.commit.json")) or any(target.glob("*.json"))


def build_plan(
    detection: Detection,
    *,
    target: str | Path,
    model: str = "minilm",
    device: str = "auto",
    mode: str | None = None,
    embedding_dim: int | None = None,
    collection: str | None = None,
    table: str | None = None,
    source: str | None = None,
    store_text: bool = True,
) -> MigrationPlan:
    """Builds a :class:`MigrationPlan` from a detection plus explicit options.

    The plan's mode follows the route by default (text-replay for LangChain /
    LlamaIndex, vector-preserve for mem0 and direct pgvector) and can be overridden
    with ``mode`` only where it is meaningful.
    """

    framework = detection.framework
    provider = detection.provider
    resolved_mode = mode or _default_mode(framework, provider)
    src_kind = _source_kind(framework, provider)
    location = source or detection.source_path or ""
    snippet = build_switch_snippet(
        framework=framework,
        provider=provider,
        mode=resolved_mode,
        target_path=str(target),
        model=model,
        embedding_dim=embedding_dim,
        collection=collection,
    )
    warnings = list(detection.warnings)
    unsupported: list[str] = []
    if framework in ("langchain", "llama-index") and resolved_mode == MODE_TEXT_REPLAY:
        unsupported.append(
            "ranking parity is not promised: LodeDB re-embeds text, so order may differ from the "
            "source embeddings (count, metadata/filter, and stored-text parity are validated)"
        )
    return MigrationPlan(
        route=detection.route,
        mode=resolved_mode,
        framework=framework,
        provider=provider,
        source_location_fingerprint=fingerprint_text(location) if location else "",
        source_kind=src_kind,
        target_path=str(target),
        collection=collection,
        table=table,
        document_count_estimate=detection.signals.get("source_count"),
        embedding_dim=embedding_dim,
        model=model if resolved_mode == MODE_TEXT_REPLAY else None,
        device=device,
        package_manager=detection.package_manager,
        install_command=detection.install_command,
        install_extra=detection.install_extra,
        store_text=store_text,
        unsupported_fields=unsupported,
        warnings=warnings,
        switch_snippet=snippet,
        rollback=default_rollback(framework, provider),
        source_options=_source_options(
            location=location, collection=collection, table=table, embedding_dim=embedding_dim
        ),
    )


def _default_mode(framework: str | None, provider: str | None) -> str:
    """Returns the default replay mode for a route."""

    if framework in ("langchain", "llama-index"):
        return MODE_TEXT_REPLAY
    if framework == "mem0" or provider == "pgvector":
        return MODE_VECTOR_PRESERVE
    return MODE_TEXT_REPLAY


def _source_kind(framework: str | None, provider: str | None) -> str:
    """Returns the concrete source-store kind the default importer reads."""

    if framework == "langchain":
        return "in-memory"
    if framework == "llama-index":
        return "simple"
    if framework == "mem0":
        return "qdrant"
    return provider or "unknown"


def _source_options(
    *, location: str, collection: str | None, table: str | None, embedding_dim: int | None
) -> dict[str, Any]:
    """Records the source-open options the plan needs to re-open the source on ``run``.

    A non-secret location (a filesystem path, a local Qdrant path, a collection
    name) is stored verbatim so ``run`` can re-open it. A *credentialed* location
    (a DSN such as ``postgresql://user:pw@host/db``) is never written to the plan:
    only its redacted form is kept, and ``location_required`` is set so ``run`` knows
    it must be re-supplied with ``--source`` / ``$DATABASE_URL``. This keeps the JSON
    plan, like the Markdown plan and the manifest, free of credentials.
    """

    from lodedb.local.migrate.report import is_local_source, redact_connection_string

    secret = location.startswith(("postgres://", "postgresql://")) and not is_local_source(
        location
    )
    # A credentialed URL (any user:pw@) is secret even when local.
    if "@" in location and "://" in location:
        secret = True
    return {
        "location": "" if secret else location,
        "location_redacted": redact_connection_string(location) if location else "",
        "location_required": secret,
        "collection": collection,
        "table": table,
        "embedding_dim": embedding_dim,
    }


def open_source(
    plan: MigrationPlan,
    *,
    allow_remote: bool = False,
    pg_connect: Any | None = None,
    source_factory: Any | None = None,
    location_override: str | None = None,
    **column_overrides: Any,
) -> SourceExport:
    """Opens the read-only source export described by a plan.

    ``source_factory`` (an open :class:`SourceExport` or a zero-arg callable that
    returns one) overrides construction entirely, which the tests use to inject
    fixtures without a real Qdrant/Postgres. Otherwise the importer is chosen from
    the plan's route/provider. ``location_override`` supplies a source location the
    plan deliberately did not store (a credentialed DSN), re-passed at ``run`` time.
    """

    if source_factory is not None:
        return source_factory() if callable(source_factory) else source_factory

    opts = plan.source_options
    location = location_override or opts.get("location") or ""
    if not location and opts.get("location_required"):
        raise MigrationError(
            "this plan's source is a credentialed connection that was not stored in the plan; "
            "re-supply it with `--source <connection>` (e.g. --source \"$DATABASE_URL\")"
        )

    if plan.framework == "langchain":
        from lodedb.local.migrate.sources.langchain_inmemory import LangChainInMemoryExport

        return LangChainInMemoryExport(location)
    if plan.framework == "llama-index":
        from lodedb.local.migrate.sources.llama_index_simple import LlamaIndexSimpleExport

        return LlamaIndexSimpleExport(location)
    if plan.framework == "mem0":
        from lodedb.local.migrate.sources.mem0_qdrant import Mem0QdrantExport

        return Mem0QdrantExport(
            collection_name=plan.collection or "mem0",
            path=location or None,
            embedding_model_dims=opts.get("embedding_dim"),
            allow_remote=allow_remote,
        )
    if plan.provider == "pgvector":
        from lodedb.local.migrate.sources.pgvector import PgVectorExport

        return PgVectorExport(
            dsn=location,
            table=plan.table or column_overrides.pop("table", None) or "documents",
            vector_dim=opts.get("embedding_dim"),
            allow_remote=allow_remote,
            connect=pg_connect,
            **column_overrides,
        )
    raise MigrationError(
        f"no source importer for route={plan.route!r} framework={plan.framework!r} "
        f"provider={plan.provider!r}"
    )


def run_migration(
    plan: MigrationPlan,
    *,
    target: str | Path | None = None,
    dry_run: bool = True,
    overwrite_target: bool = False,
    resume: bool = False,
    allow_remote: bool = False,
    embedding_backend: Any | None = None,
    source: SourceExport | None = None,
    source_factory: Any | None = None,
    source_location: str | None = None,
    pg_connect: Any | None = None,
    **source_kwargs: Any,
) -> MigrationResult:
    """Executes a plan end to end (or, with ``dry_run``, validates feasibility).

    Returns a :class:`MigrationResult`. On a real run it writes the new store to
    ``<target>.tmp``, validates a read-only reopen, runs the persisted-index audit,
    writes ``migration.json``, and moves the temp into place. ``embedding_backend``
    injects a deterministic embedder for text-replay (used by tests/offline runs).
    """

    target_path = Path(target or plan.target_path)
    started = time.time()

    if not dry_run and not resume and target_has_store(target_path):
        if not overwrite_target:
            raise MigrationError(
                f"target {target_path} already contains a LodeDB store; pass --overwrite-target "
                "to replace it or --resume to write into it"
            )

    export = source if source is not None else open_source(
        plan,
        allow_remote=allow_remote,
        pg_connect=pg_connect,
        source_factory=source_factory,
        location_override=source_location,
        **source_kwargs,
    )

    result = MigrationResult(
        status="dry-run" if dry_run else "migrated",
        route=plan.route,
        mode=export.mode,
        framework=plan.framework,
        provider=plan.provider,
        source_kind=export.provider,
        source_location_fingerprint=plan.source_location_fingerprint
        or fingerprint_text(export.location),
        target_path=str(target_path),
        store_subdir=(plan.collection or "mem0") if plan.framework == "mem0" else None,
        embedding_dim=export.vector_dim or plan.embedding_dim,
        source_count=export.count,
        warnings=list(plan.warnings) + list(export.warnings),
        versions=_lodedb_versions(),
        options={
            "dry_run": dry_run,
            "mode": export.mode,
            "store_text": plan.store_text,
            "model": plan.model,
            "device": plan.device,
            "overwrite_target": overwrite_target,
            "resume": resume,
        },
        started_at=started,
    )

    try:
        if export.mode == MODE_VECTOR_PRESERVE:
            _validate_vector_dim(result.embedding_dim)
        if dry_run:
            # A dry run reads enough of the source to confirm it opens and that the
            # first rows are shaped correctly, without writing a target.
            result.written_count, preview_skips = _dry_run_probe(export, plan)
            result.skipped = preview_skips
            result.finished_at = time.time()
            return result

        write_dir = _temp_target(target_path)
        if write_dir.exists():
            shutil.rmtree(write_dir)
        try:
            written, skipped = _write_target(
                export, plan, write_dir, embedding_backend=embedding_backend
            )
            result.written_count = written
            result.skipped = skipped
            result.validation = _validate_target(
                export,
                plan,
                write_dir,
                written=written,
                skipped_count=len(skipped),
                embedding_backend=embedding_backend,
            )
            _write_manifest(write_dir, result)
            _atomic_move(write_dir, target_path, resume=resume, overwrite=overwrite_target)
        except Exception:
            # Leave the source untouched; clean up the partial temp on failure.
            if write_dir.exists():
                shutil.rmtree(write_dir, ignore_errors=True)
            raise
        result.finished_at = time.time()
        # Rewrite the manifest at the final path so its timings/validation are complete.
        _write_manifest(target_path, result)
        return result
    finally:
        export.close()


# -- writing ----------------------------------------------------------------


def _open_text_target(plan: MigrationPlan, write_dir: Path, embedding_backend: Any | None) -> Any:
    """Opens a text-replay target, wrapped in the route's framework adapter when present."""

    if plan.framework == "langchain":
        from lodedb.local.integrations.langchain import LodeDBVectorStore

        db = LodeDB(
            path=write_dir,
            model=plan.model or "minilm",
            device=plan.device,
            store_text=plan.store_text,
            _embedding_backend=embedding_backend,
        )
        return _LangChainTargetWriter(LodeDBVectorStore(db), db)
    if plan.framework == "llama-index":
        from lodedb.local.integrations.llama_index import LodeDBVectorStore

        db = LodeDB(
            path=write_dir,
            model=plan.model or "minilm",
            device=plan.device,
            store_text=plan.store_text,
            _embedding_backend=embedding_backend,
        )
        return _LlamaIndexTargetWriter(LodeDBVectorStore(db), db)
    # Direct provider, text-owned mode.
    db = LodeDB(
        path=write_dir,
        model=plan.model or "bge",
        device=plan.device,
        store_text=plan.store_text,
        _embedding_backend=embedding_backend,
    )
    return _PlainTextTargetWriter(db)


def _open_vector_target(plan: MigrationPlan, write_dir: Path, embedding_dim: int) -> Any:
    """Opens a vector-preserve target, via the mem0 adapter or a vector-only handle."""

    if plan.framework == "mem0":
        from lodedb.local.integrations.mem0 import LodeDBVectorStore

        store = LodeDBVectorStore(
            collection_name=plan.collection or "mem0",
            path=str(write_dir),
            embedding_model_dims=embedding_dim,
            store_payloads=plan.store_text,
        )
        return _Mem0TargetWriter(store)
    db = LodeDB.open_vector_store(write_dir, vector_dim=embedding_dim, store_text=plan.store_text)
    return _VectorTargetWriter(db)


def _write_target(
    export: SourceExport,
    plan: MigrationPlan,
    write_dir: Path,
    *,
    embedding_backend: Any | None,
) -> tuple[int, list[dict[str, Any]]]:
    """Replays the source into a fresh LodeDB target; returns ``(written, skipped)``."""

    if export.mode == MODE_VECTOR_PRESERVE:
        writer = _open_vector_target(plan, write_dir, int(export.vector_dim or plan.embedding_dim))
    else:
        writer = _open_text_target(plan, write_dir, embedding_backend)

    written = 0
    skipped: list[dict[str, Any]] = []
    try:
        for row in export.iter_rows():
            reason = writer.write(row)
            if reason is None:
                written += 1
            else:
                skipped.append({"id_hash": hash_id(row.id), "reason": reason})
        writer.persist()
    finally:
        writer.close()
    return written, skipped


# -- target writers ---------------------------------------------------------


class _PlainTextTargetWriter:
    """Writes text-replay rows through a plain text LodeDB handle."""

    def __init__(self, db: LodeDB) -> None:
        self._db = db

    def write(self, row: ExportedRow) -> str | None:
        if not row.text or not row.text.strip():
            return "missing-text"
        metadata = _stringify_metadata(row.metadata)
        self._db.add(row.text, id=row.id, metadata=metadata)
        return None

    def persist(self) -> None:
        self._db.persist()

    def close(self) -> None:
        self._db.close()


class _LangChainTargetWriter:
    """Writes text-replay rows through the LangChain adapter."""

    def __init__(self, store: Any, db: LodeDB) -> None:
        self._store = store
        self._db = db

    def write(self, row: ExportedRow) -> str | None:
        if not row.text or not row.text.strip():
            return "missing-text"
        self._store.add_texts(
            [row.text], metadatas=[_stringify_metadata(row.metadata)], ids=[row.id]
        )
        return None

    def persist(self) -> None:
        self._db.persist()

    def close(self) -> None:
        self._db.close()


class _LlamaIndexTargetWriter:
    """Writes text-replay rows through the LlamaIndex adapter (rebuilds SOURCE)."""

    def __init__(self, store: Any, db: LodeDB) -> None:
        self._store = store
        self._db = db

    def write(self, row: ExportedRow) -> str | None:
        if not row.text or not row.text.strip():
            return "missing-text"
        from llama_index.core.schema import NodeRelationship, RelatedNodeInfo, TextNode

        node = TextNode(
            id_=row.id, text=row.text, metadata=_stringify_metadata(row.metadata)
        )
        if row.ref_doc_id is not None:
            node.relationships[NodeRelationship.SOURCE] = RelatedNodeInfo(node_id=row.ref_doc_id)
        self._store.add([node])
        return None

    def persist(self) -> None:
        self._db.persist()

    def close(self) -> None:
        self._db.close()


class _VectorTargetWriter:
    """Writes vector-preserve rows through a vector-only LodeDB handle."""

    def __init__(self, db: LodeDB) -> None:
        self._db = db

    def write(self, row: ExportedRow) -> str | None:
        if row.vector is None:
            return "missing-vector"
        if len(row.vector) != self._db._vector_dim:
            return "dimension-mismatch"
        text = row.text
        if text is None and row.raw_payload:
            import json

            text = json.dumps(row.raw_payload, sort_keys=True, separators=(",", ":"), default=str)
        self._db.add_vectors(
            row.vector, id=row.id, metadata=_stringify_metadata(row.metadata), text=text
        )
        return None

    def persist(self) -> None:
        self._db.persist()

    def close(self) -> None:
        self._db.close()


class _Mem0TargetWriter:
    """Writes vector-preserve rows through the mem0 adapter (preserves payloads)."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._dim = int(store.embedding_model_dims)

    def write(self, row: ExportedRow) -> str | None:
        if row.vector is None:
            return "missing-vector"
        if len(row.vector) != self._dim:
            return "dimension-mismatch"
        payload = dict(row.raw_payload or {})
        self._store.insert(vectors=[row.vector], ids=[row.id], payloads=[payload])
        return None

    def persist(self) -> None:
        self._store.client.persist()

    def close(self) -> None:
        self._store.close()


# -- validation -------------------------------------------------------------


def _validate_target(
    export: SourceExport,
    plan: MigrationPlan,
    write_dir: Path,
    *,
    written: int,
    skipped_count: int,
    embedding_backend: Any | None,
) -> dict[str, Any]:
    """Validates the freshly written target by reopening it read-only.

    Checks count parity (or documented skips), id/metadata survival for a sample,
    stored-text fetch after reopen when text is retained, and the persisted-index
    audit. ``written``/``skipped_count`` are the run's own counts; count parity
    holds when every source row was either written or recorded as skipped.
    """

    checks: dict[str, Any] = {}
    store_dir = _store_dir(plan, write_dir)
    read_back = _read_back_counts(plan, store_dir, embedding_backend)
    source_count = export.count
    checks["count"] = {
        "source": source_count,
        "target": read_back["count"],
        "written": written,
        "skipped": skipped_count,
    }
    # Parity: the reopened store holds what we wrote, and written + skipped accounts
    # for every source row (when the source count is known).
    accounted = written + skipped_count
    checks["count_parity"] = read_back["count"] == written and (
        source_count is None or accounted == source_count
    )

    # Persisted-index audit on the written store.
    try:
        audit = audit_persisted_index_snapshots(store_dir)
        checks["audit"] = {
            "status": audit.get("status"),
            "snapshot_count": audit.get("snapshot_count"),
            "raw_document_text_present": audit.get("raw_document_text_present"),
        }
        checks["audit_passed"] = audit.get("status") == "passed" and not audit.get(
            "raw_document_text_present", False
        )
    except Exception as exc:  # noqa: BLE001 - record the failure rather than crash validate
        checks["audit"] = {"error": type(exc).__name__}
        checks["audit_passed"] = False

    checks["sample"] = read_back["sample"]
    checks["passed"] = bool(checks["count_parity"]) and bool(checks.get("audit_passed"))
    return checks


def _read_back_counts(
    plan: MigrationPlan, write_dir: Path, embedding_backend: Any | None
) -> dict[str, Any]:
    """Reopens the written store read-only and gathers count/sample parity facts."""

    if plan.mode == MODE_VECTOR_PRESERVE:
        db = LodeDB.open_vector_store(
            write_dir,
            vector_dim=int(plan.embedding_dim or 8),
            read_only=True,
            store_text=plan.store_text,
        )
    else:
        db = LodeDB(
            path=write_dir,
            model=plan.model or "minilm",
            read_only=True,
            store_text=plan.store_text,
            _embedding_backend=embedding_backend,
        )
    try:
        records = db.list_documents()
        ids = [record["id"] for record in records]
        sample_idx = sample_indices(len(ids), limit=int(plan.thresholds.get("sample_size", 25)))
        sample_ids = [ids[i] for i in sample_idx]
        text_recovered = 0
        if plan.store_text and sample_ids:
            try:
                texts = db.get_texts(sample_ids)
                text_recovered = sum(1 for value in texts.values() if value)
            except ValueError:
                text_recovered = 0
        return {
            "count": len(ids),
            "skipped": 0,
            "sample": {
                "size": len(sample_ids),
                "ids_present": len(sample_ids),
                "text_recovered": text_recovered,
                "id_hashes": [hash_id(i) for i in sample_ids[:5]],
            },
        }
    finally:
        db.close()


def _dry_run_probe(export: SourceExport, plan: MigrationPlan) -> tuple[int, list[dict[str, Any]]]:
    """Reads a bounded prefix of the source to confirm it opens and rows are shaped right.

    Returns ``(rows_seen, skipped)``: a dry run writes nothing, so ``rows_seen`` is
    the count it inspected (up to the source count), and ``skipped`` flags any
    early rows that would not migrate (missing text in text-replay, missing/
    mismatched vector in vector-preserve).
    """

    seen = 0
    skipped: list[dict[str, Any]] = []
    limit = 200  # bounded peek; the real run streams everything
    for row in export.iter_rows():
        seen += 1
        reason = _would_skip(row, export.mode, export.vector_dim or plan.embedding_dim)
        if reason is not None:
            skipped.append({"id_hash": hash_id(row.id), "reason": reason})
        if seen >= limit:
            break
    return seen, skipped


def _would_skip(row: ExportedRow, mode: str, dim: int | None) -> str | None:
    """Returns the skip reason a row would get, or ``None`` if it would migrate."""

    if mode == MODE_VECTOR_PRESERVE:
        if row.vector is None:
            return "missing-vector"
        if dim is not None and len(row.vector) != dim:
            return "dimension-mismatch"
        return None
    if not row.text or not row.text.strip():
        return "missing-text"
    return None


# -- io helpers -------------------------------------------------------------


def _temp_target(target: Path) -> Path:
    """Returns the ``<target>.tmp`` staging path next to the requested target."""

    return target.parent / (target.name + ".tmp")


def _store_dir(plan: MigrationPlan, base: Path) -> Path:
    """Returns the actual LodeDB store directory under a migration target base.

    The mem0 adapter nests its index under ``<path>/<collection_name>``, so for that
    route the store the audit/read-back must point at is the nested directory; every
    other route writes the index directly at ``base``. ``migration.json`` is always
    written at ``base`` (the directory the user names as ``--target``).
    """

    if plan.framework == "mem0":
        return base / (plan.collection or "mem0")
    return base


def _atomic_move(write_dir: Path, target: Path, *, resume: bool, overwrite: bool) -> None:
    """Moves the validated temp store into the final target path.

    ``resume``/``overwrite`` replace an existing target; otherwise the move assumes
    a fresh target (the caller already refused a collision).
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (overwrite or resume):
        backup = target.parent / (target.name + ".replaced")
        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
        target.rename(backup)
        try:
            write_dir.rename(target)
        except OSError:
            shutil.move(str(write_dir), str(target))
        shutil.rmtree(backup, ignore_errors=True)
        return
    try:
        write_dir.rename(target)
    except OSError:
        shutil.move(str(write_dir), str(target))


def _write_manifest(directory: Path, result: MigrationResult) -> None:
    """Writes the payload-free ``migration.json`` manifest into ``directory``."""

    import json

    from lodedb.local.migrate.report import assert_payload_free

    directory.mkdir(parents=True, exist_ok=True)
    manifest = result.to_manifest()
    assert_payload_free(manifest, where="migration.json")
    (directory / "migration.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_vector_dim(dim: int | None) -> None:
    """Raises :class:`MigrationError` unless ``dim`` is a positive multiple of 8."""

    if dim is None or dim <= 0 or dim % _VECTOR_DIM_MULTIPLE != 0:
        raise MigrationError(
            f"vector-preserve mode needs an embedding dimension that is a positive multiple of 8 "
            f"for LodeDB vector indexes; got {dim!r}"
        )


def _stringify_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keeps only scalar metadata values LodeDB can store, dropping nested values.

    LodeDB metadata is a scalar string map; list/dict values (which belong in the
    raw-text sidecar) are dropped here so the metadata write never fails, and their
    presence is the caller's concern (mem0 carries them in the payload).
    """

    out: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)):
            out[str(key)] = value
    return out


# -- top-level convenience --------------------------------------------------


def inspect_and_plan(
    project: str | Path,
    *,
    target: str | Path,
    framework: str | None = None,
    provider: str | None = None,
    model: str = "minilm",
    device: str = "auto",
    mode: str | None = None,
    embedding_dim: int | None = None,
    collection: str | None = None,
    table: str | None = None,
    source: str | None = None,
    store_text: bool = True,
) -> tuple[Detection, MigrationPlan | None]:
    """Inspects a project and, when the route is actionable, builds a plan.

    Returns ``(detection, plan)``; ``plan`` is ``None`` for an ambiguous/none route
    (the caller surfaces the detection's guidance instead).
    """

    detection = inspect_project(project, framework=framework, provider=provider)
    if detection.route not in ("framework", "provider"):
        return detection, None
    if detection.route == "provider" and detection.provider != "pgvector":
        # Other direct providers are detected and reported, but migration is a follow-up.
        return detection, None
    plan = build_plan(
        detection,
        target=target,
        model=model,
        device=device,
        mode=mode,
        embedding_dim=embedding_dim,
        collection=collection,
        table=table,
        source=source,
        store_text=store_text,
    )
    return detection, plan


def iter_export_rows(export: SourceExport) -> Iterator[ExportedRow]:
    """Public, read-only pass-through over a source export's rows (for callers/tests)."""

    yield from export.iter_rows()
