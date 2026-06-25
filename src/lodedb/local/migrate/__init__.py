"""LodeDB migration toolkit (``lodedb migrate``).

Takes a project on an existing LangChain / LlamaIndex / mem0 store, or a direct
vector provider (pgvector first), and moves it onto LodeDB without hand-written
export scripts, guesswork about which optional extra to install, or risk to
production data. The flow is inspect -> plan -> dry-run -> run -> validate, and the
source store is never modified or deleted.

Routing follows both issues: a framework owner (LangChain / LlamaIndex / mem0) wins
over any direct provider beneath it and is migrated through that framework's shipped
LodeDB adapter; a direct provider is migrated provider-first into a local LodeDB
path. Every artifact stays payload-free (counts, bytes, timings, id hashes,
dimensions, versions, warnings) and credentials are redacted from logs, plans, and
manifests.
"""

from lodedb.local.migrate.detect import Detection, inspect_project
from lodedb.local.migrate.plan import MigrationPlan, build_switch_snippet
from lodedb.local.migrate.runner import (
    MigrationError,
    MigrationResult,
    build_plan,
    inspect_and_plan,
    run_migration,
    target_has_store,
)

__all__ = [
    "Detection",
    "MigrationError",
    "MigrationPlan",
    "MigrationResult",
    "build_plan",
    "build_switch_snippet",
    "inspect_and_plan",
    "inspect_project",
    "run_migration",
    "target_has_store",
]
