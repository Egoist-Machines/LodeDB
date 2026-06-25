"""The ``lodedb migrate`` Typer sub-app: inspect | plan | run | validate.

Wires the migration toolkit onto the CLI, mirroring the ``lodedb mcp`` sub-app
pattern (a ``typer.Typer`` added to the root app under a name). No migration logic
lives here: each command is a thin shell over
:mod:`lodedb.local.migrate.detect` / :mod:`lodedb.local.migrate.plan` /
:mod:`lodedb.local.migrate.runner`, and prints payload-free JSON or writes the
plan/manifest artifacts. ``install-agent`` is exposed as ``inspect`` with the
framework-handoff routing baked into detection: when a framework owns the store the
result's ``route`` is ``framework`` and ``next`` points at the #34 path.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from lodedb.local.migrate.detect import inspect_project
from lodedb.local.migrate.plan import MigrationPlan
from lodedb.local.migrate.runner import (
    MigrationError,
    build_plan,
    run_migration,
)

migrate_app = typer.Typer(
    no_args_is_help=True,
    help="Migrate an existing vector store (LangChain/LlamaIndex/mem0 or a direct provider "
    "such as pgvector) onto a local LodeDB path. Plan-first and non-destructive: the source "
    "store is never modified or deleted.",
)

_PROJECT_OPTION = typer.Option(Path("."), "--project", help="Project checkout to inspect.")
_TARGET_OPTION = typer.Option(
    Path("./data/lodedb"), "--target", help="New on-disk LodeDB directory to migrate into."
)
_FRAMEWORK_OPTION = typer.Option(
    None, "--framework", help="langchain | llama-index | mem0 (override auto-detection)."
)
_PROVIDER_OPTION = typer.Option(
    None, "--provider", help="pgvector | qdrant | chroma | lancedb | sqlite-vec | faiss | auto."
)
_SOURCE_OPTION = typer.Option(
    None, "--source", help="Persisted store path or connection string for the source."
)
_TABLE_OPTION = typer.Option(None, "--table", help="Source table name (pgvector).")
_COLLECTION_OPTION = typer.Option(
    None, "--collection", help="Source collection name (mem0/qdrant)."
)
_MODEL_OPTION = typer.Option(
    "minilm", "--model", "-m", help="LodeDB embedding preset for text-replay migrations."
)
_DEVICE_OPTION = typer.Option("auto", "--device", "-d", help="auto | cpu | mps | cuda.")
_MODE_OPTION = typer.Option(
    None, "--mode", help="vector-preserve | text-replay | auto (defaults follow the route)."
)
_EMBEDDING_DIM_OPTION = typer.Option(
    None, "--embedding-dim", help="Embedding dimension for vector-preserve mode."
)
_VECTOR_DIM_OPTION = typer.Option(
    None, "--vector-dim", help="Alias of --embedding-dim for direct-provider sources."
)
_OUT_OPTION = typer.Option(
    Path("lodedb-migration-plan.md"), "--out", help="Markdown plan path (a .json is written too)."
)
_PLAN_OPTION = typer.Option(..., "--plan", help="The migration plan JSON produced by `plan`.")
_DRY_RUN_OPTION = typer.Option(
    True,
    "--dry-run/--write",
    help="Default is a dry run: read the source and confirm feasibility without writing "
    "anything. Pass --write to perform the migration into the target path.",
)
_TARGET_OVERRIDE_OPTION = typer.Option(
    None, "--target", help="Override the plan's target path."
)
_OVERWRITE_OPTION = typer.Option(
    False, "--overwrite-target", help="Replace an existing LodeDB store at the target path."
)
_RESUME_OPTION = typer.Option(False, "--resume", help="Write into an existing target path.")
_ALLOW_REMOTE_OPTION = typer.Option(
    False, "--allow-remote-source", help="Permit connecting to a non-local source host."
)
_LOCAL_ONLY_OPTION = typer.Option(
    False, "--local-only", help="Refuse non-local sources (default behavior; explicit form)."
)
_STORE_TEXT_OPTION = typer.Option(
    True, "--store-text/--no-store-text", help="Retain source text in LodeDB (default on)."
)
_MANIFEST_OPTION = typer.Option(
    ..., "--manifest", help="A migration.json manifest (or its target directory) to validate."
)
_JSON_OPTION = typer.Option(False, "--json", help="Emit machine-readable JSON.")


@migrate_app.command("inspect")
def migrate_inspect(
    project: Path = _PROJECT_OPTION,
    framework: str | None = _FRAMEWORK_OPTION,
    provider: str | None = _PROVIDER_OPTION,
    json_out: bool = _JSON_OPTION,
) -> None:
    """Detect the framework/provider in a project and print the routing decision.

    This is the ``install-agent`` router: a framework owner (LangChain/LlamaIndex/
    mem0) routes to the #34 path even when backed by pgvector/Qdrant; otherwise a
    direct provider (pgvector first) routes to the #35 path. Ambiguous projects stop
    and ask for ``--framework``/``--provider``. The report is payload-free.
    """

    try:
        detection = inspect_project(project, framework=framework, provider=provider)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = detection.to_dict()
    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return
    typer.echo(f"route     : {payload['route']}")
    if payload.get("framework"):
        typer.echo(f"framework : {payload['framework']}")
    if payload.get("provider"):
        typer.echo(f"provider  : {payload['provider']}")
    typer.echo(f"install   : {payload.get('install_command')}")
    if payload.get("frameworks_seen"):
        typer.echo(f"frameworks: {', '.join(payload['frameworks_seen'])}")
    if payload.get("providers_seen"):
        typer.echo(f"providers : {', '.join(payload['providers_seen'])}")
    if payload.get("next"):
        typer.echo(f"next      : {payload['next']}")
    for warning in payload.get("warnings", []):
        typer.echo(f"warning   : {warning}")


@migrate_app.command("plan")
def migrate_plan(
    project: Path = _PROJECT_OPTION,
    target: Path = _TARGET_OPTION,
    framework: str | None = _FRAMEWORK_OPTION,
    provider: str | None = _PROVIDER_OPTION,
    source: str | None = _SOURCE_OPTION,
    table: str | None = _TABLE_OPTION,
    collection: str | None = _COLLECTION_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    mode: str | None = _MODE_OPTION,
    embedding_dim: int | None = _EMBEDDING_DIM_OPTION,
    vector_dim: int | None = _VECTOR_DIM_OPTION,
    store_text: bool = _STORE_TEXT_OPTION,
    out: Path = _OUT_OPTION,
) -> None:
    """Produce a payload-free Markdown + JSON migration plan for review.

    Writes ``--out`` (Markdown) and a sibling ``.json`` (the input to ``run``). The
    plan states the route, mode, source fingerprint, target, counts, embedding
    dimension, install command, validation thresholds, rollback, and the exact
    code/config switch snippet. It never writes data.
    """

    try:
        detection = inspect_project(project, framework=framework, provider=provider)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if detection.route not in ("framework", "provider"):
        typer.echo(json.dumps(detection.to_dict(), indent=2, sort_keys=True))
        raise typer.Exit(code=2)
    if detection.route == "provider" and detection.provider != "pgvector":
        typer.echo(
            f"direct {detection.provider} migration is a follow-up; pgvector is the supported "
            "direct provider in this release."
        )
        raise typer.Exit(code=2)

    plan = build_plan(
        detection,
        target=target,
        model=model,
        device=device,
        mode=mode,
        embedding_dim=embedding_dim or vector_dim,
        collection=collection,
        table=table,
        source=source,
        store_text=store_text,
    )
    md_path = out
    json_path = out.with_suffix(".json")
    md_path.write_text(plan.to_markdown(), encoding="utf-8")
    json_path.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    typer.echo(f"wrote plan: {md_path}")
    typer.echo(f"wrote plan: {json_path}")
    typer.echo(f"route={plan.route} mode={plan.mode} target={plan.target_path}")


@migrate_app.command("run")
def migrate_run(
    plan_path: Path = _PLAN_OPTION,
    target: Path | None = _TARGET_OVERRIDE_OPTION,
    source: str | None = _SOURCE_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    overwrite_target: bool = _OVERWRITE_OPTION,
    resume: bool = _RESUME_OPTION,
    allow_remote: bool = _ALLOW_REMOTE_OPTION,
    local_only: bool = _LOCAL_ONLY_OPTION,
) -> None:
    """Execute a migration plan: dry-run by default, or write with ``--write``.

    The default is a dry run that reads the source and confirms feasibility without
    writing anything; pass ``--write`` to perform the migration. A real run writes to
    ``<target>.tmp``, reopens it read-only, validates, runs the persisted-index audit,
    writes ``migration.json``, then moves it into place. The source store is read-only
    throughout. Prints a payload-free result summary.

    When the plan's source is a credentialed connection (a DSN), the plan does not
    store it; re-supply it here with ``--source "$DATABASE_URL"``.
    """

    if local_only and allow_remote:
        raise typer.BadParameter("--local-only and --allow-remote-source are mutually exclusive")
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"could not read plan {plan_path}: {exc}") from exc
    plan = MigrationPlan.from_dict(data)
    try:
        result = run_migration(
            plan,
            target=target,
            dry_run=dry_run,
            overwrite_target=overwrite_target,
            resume=resume,
            allow_remote=allow_remote,
            source_location=source,
        )
    except MigrationError as exc:
        raise typer.BadParameter(str(exc)) from exc
    manifest = result.to_manifest()
    typer.echo(json.dumps(manifest, indent=2, sort_keys=True))
    if result.status == "migrated" and not result.validation.get("passed", False):
        # A completed-but-failing validation is a non-zero exit so an agent stops.
        raise typer.Exit(code=1)


@migrate_app.command("validate")
def migrate_validate(
    manifest: Path = _MANIFEST_OPTION,
    json_out: bool = _JSON_OPTION,
) -> None:
    """Re-validate a migrated target from its ``migration.json`` manifest.

    Re-runs the persisted-index audit on the target and reports the recorded
    validation plus a fresh audit, payload-free. Exits non-zero if the audit fails.
    """

    manifest_path = manifest
    if manifest_path.is_dir():
        manifest_path = manifest_path / "migration.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"migration.json not found at {manifest_path}")
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise typer.BadParameter(f"could not read manifest {manifest_path}: {exc}") from exc

    target_block = data.get("target", {})
    target = Path(target_block.get("path", manifest_path.parent))
    subdir = target_block.get("store_subdir")
    store_dir = target / subdir if subdir else target
    from lodedb.engine.core import audit_persisted_index_snapshots

    report: dict[str, object] = {
        "manifest": str(manifest_path),
        "recorded_status": data.get("status"),
        "recorded_validation": data.get("validation", {}),
    }
    try:
        audit = audit_persisted_index_snapshots(store_dir)
        report["audit"] = {
            "status": audit.get("status"),
            "snapshot_count": audit.get("snapshot_count"),
            "raw_document_text_present": audit.get("raw_document_text_present"),
        }
        ok = audit.get("status") == "passed" and not audit.get("raw_document_text_present", False)
    except Exception as exc:  # noqa: BLE001 - report failure rather than crash
        report["audit"] = {"error": type(exc).__name__}
        ok = False
    report["passed"] = bool(ok) and bool(data.get("validation", {}).get("passed", True))

    if json_out:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(f"manifest        : {report['manifest']}")
        typer.echo(f"recorded status : {report['recorded_status']}")
        typer.echo(f"audit status    : {report['audit'].get('status', report['audit'])}")
        typer.echo(f"passed          : {report['passed']}")
    if not report["passed"]:
        raise typer.Exit(code=1)
