"""Tests for the direct pgvector exporter and the ``lodedb migrate`` CLI surface.

pgvector is driven through a fake psycopg-shaped DB-API connection (no Postgres),
so the read-only export contract, column auto-detection, dimension discovery, DSN
redaction, and remote-host gating are all exercised in-process. The CLI is driven
through Typer's ``CliRunner`` against synthetic project directories and a fixture
plan.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from lodedb.local.cli import app
from lodedb.local.db import LodeDB
from lodedb.local.migrate import MigrationError, build_plan, run_migration
from lodedb.local.migrate.detect import Detection
from lodedb.local.migrate.plan import MigrationPlan
from lodedb.local.migrate.sources.pgvector import PgVectorExport

runner = CliRunner()
DIM = 8


def _pgvec_text(i: int) -> str:
    """Returns a pgvector text literal ``[..]`` one-hot vector of width DIM."""

    values = [0.0] * DIM
    values[i] = 1.0
    return "[" + ",".join(str(v) for v in values) + "]"


class _FakeCursor:
    """A minimal psycopg-shaped cursor over an in-memory table of dict rows."""

    def __init__(self, rows: list[dict], *, recorder: list[str]) -> None:
        self._rows = rows
        self._result: list[tuple] = []
        self._recorder = recorder

    def execute(self, sql: str, params=()) -> None:
        normalized = " ".join(sql.split())
        self._recorder.append(normalized)
        if "information_schema.columns" in normalized:
            self._result = [
                ("id", "integer"),
                ("content", "text"),
                ("embedding", "USER-DEFINED"),
                ("metadata", "jsonb"),
            ]
        elif "pg_attribute" in normalized:
            self._result = [(DIM,)]  # declared vector(DIM) via atttypmod
        elif normalized.startswith("SELECT COUNT(*)"):
            self._result = [(len(self._rows),)]
        elif "ORDER BY" in normalized:
            limit = params[-1]
            after = params[0] if len(params) == 2 else None
            ordered = sorted(self._rows, key=lambda r: r["id"])
            if after is not None:
                ordered = [r for r in ordered if r["id"] > after]
            ordered = ordered[:limit]
            self._result = [
                (r["id"], r["embedding"], r["content"], r["metadata"]) for r in ordered
            ]
        else:
            self._result = []

    def fetchall(self):
        return list(self._result)

    def fetchone(self):
        return self._result[0] if self._result else None

    def close(self):
        return None


class _FakeConn:
    """A psycopg-shaped connection that hands out :class:`_FakeCursor` cursors."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.statements: list[str] = []

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self._rows, recorder=self.statements)

    def close(self):
        return None


def _fake_rows() -> list[dict]:
    return [
        {"id": 1, "content": "first doc", "embedding": _pgvec_text(0), "metadata": {"t": "t1"}},
        {"id": 2, "content": "second doc", "embedding": _pgvec_text(1), "metadata": {"t": "t2"}},
        {"id": 3, "content": "third doc", "embedding": _pgvec_text(2), "metadata": {"t": "t1"}},
    ]


def test_pgvector_export_detects_columns_dimension_and_redacts_dsn():
    """The exporter auto-detects columns, reads the dimension, and never stores the DSN."""

    conn = _FakeConn(_fake_rows())
    export = PgVectorExport(
        dsn="postgresql://user:pw@localhost:5432/app",
        table="documents",
        connect=lambda dsn: conn,
    )
    assert export.provider == "pgvector"
    assert export.framework is None
    assert export.vector_dim == DIM
    assert export.count == 3
    # The stored location is the redacted DSN (no credentials/host).
    assert export.location == "postgresql://<redacted>"
    assert export.notes["text_column"] == "content"
    assert export.notes["vector_column"] == "embedding"

    rows = list(export.iter_rows())
    assert [r.id for r in rows] == ["1", "2", "3"]
    assert rows[0].text == "first doc"
    assert rows[0].metadata == {"t": "t1"}
    assert len(rows[0].vector) == DIM
    export.close()

    # Only read-only statements were issued (no writes/DDL).
    forbidden = ("INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER", "CREATE")
    assert all(not s.upper().startswith(forbidden) for s in conn.statements)


def test_pgvector_export_refuses_remote_host_without_override():
    """A non-local Postgres host is refused unless allow_remote is set."""

    from lodedb.local.migrate.sources.base import SourceExportError

    with pytest.raises(SourceExportError):
        PgVectorExport(
            dsn="postgresql://prod.db.example.com/app",
            table="documents",
            connect=lambda dsn: _FakeConn(_fake_rows()),
        )
    # With the override it constructs (the fake conn returns a usable schema).
    export = PgVectorExport(
        dsn="postgresql://prod.db.example.com/app",
        table="documents",
        allow_remote=True,
        connect=lambda dsn: _FakeConn(_fake_rows()),
    )
    assert export.vector_dim == DIM
    export.close()


def test_pgvector_export_rejects_unsafe_identifier():
    """A table/column name that is not a plain identifier is rejected before any query."""

    from lodedb.local.migrate.sources.base import SourceExportError

    with pytest.raises(SourceExportError):
        PgVectorExport(
            dsn="postgresql://localhost/app",
            table="documents; DROP TABLE users",
            connect=lambda dsn: _FakeConn(_fake_rows()),
        )


def test_pgvector_migration_into_fresh_lodedb(tmp_path):
    """A direct pgvector fixture migrates into a fresh vector-only LodeDB store."""

    conn = _FakeConn(_fake_rows())
    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(
        det,
        target=tmp_path / "pg",
        embedding_dim=DIM,
        table="documents",
        source="postgresql://localhost/app",
    )
    export = PgVectorExport(
        dsn="postgresql://localhost/app", table="documents", connect=lambda dsn: conn
    )
    result = run_migration(plan, dry_run=False, source=export)
    assert result.status == "migrated"
    assert result.written_count == 3
    assert result.validation["passed"] is True
    # The manifest records a fingerprint, never the DSN.
    manifest = json.loads((tmp_path / "pg" / "migration.json").read_text(encoding="utf-8"))
    assert "localhost" not in json.dumps(manifest)
    assert manifest["source"]["location_fingerprint"]

    db = LodeDB.open_vector_store(tmp_path / "pg", vector_dim=DIM, read_only=True)
    try:
        hits = db.search_by_vector(
            [1.0] + [0.0] * (DIM - 1), k=3, filter={"metadata": {"t": "t1"}}
        )
        assert {h.id for h in hits} == {"1", "3"}
        assert db.get("1") == "first doc"
    finally:
        db.close()


# --------------------------------------------------------------------------------------
# CLI surface.
# --------------------------------------------------------------------------------------


def test_cli_inspect_routes_pgvector_project(tmp_path):
    """`lodedb migrate inspect --json` reports the direct pgvector route."""

    (tmp_path / "requirements.txt").write_text("psycopg[binary]\npgvector\n", encoding="utf-8")
    (tmp_path / "store.py").write_text(
        "import psycopg\n# embedding vector(1536)\n", encoding="utf-8"
    )
    result = runner.invoke(
        app, ["migrate", "inspect", "--project", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["route"] == "provider"
    assert payload["provider"] == "pgvector"
    assert payload["install_command"].endswith("lodedb")


def test_cli_inspect_routes_framework_project_to_handoff(tmp_path):
    """`lodedb migrate inspect` hands a mem0 project off to the #34 framework path."""

    (tmp_path / "app.py").write_text(
        "from mem0 import Memory\nMemory.from_config({})\n", encoding="utf-8"
    )
    result = runner.invoke(app, ["migrate", "inspect", "--project", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["route"] == "framework"
    assert payload["framework"] == "mem0"
    assert "#34" in payload["next"]


def test_cli_plan_writes_markdown_and_json_redacted(tmp_path):
    """`lodedb migrate plan` writes both artifacts and redacts the DSN from Markdown."""

    (tmp_path / "store.py").write_text(
        "import psycopg\n# embedding vector(1536)\n", encoding="utf-8"
    )
    out = tmp_path / "plan.md"
    result = runner.invoke(
        app,
        [
            "migrate",
            "plan",
            "--project",
            str(tmp_path),
            "--provider",
            "pgvector",
            "--target",
            str(tmp_path / "lodedb"),
            "--embedding-dim",
            "1536",
            "--table",
            "docs",
            "--source",
            "postgresql://u:pw@localhost/app",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.is_file()
    plan_json = out.with_suffix(".json")
    assert plan_json.is_file()
    data = json.loads(plan_json.read_text(encoding="utf-8"))
    assert data["mode"] == "vector-preserve"
    assert data["switch_snippet"]
    # The DSN never reaches either artifact.
    assert "u:pw@localhost" not in out.read_text(encoding="utf-8")
    assert "u:pw@localhost" not in json.dumps(data)


def test_cli_plan_on_ambiguous_project_exits_nonzero(tmp_path):
    """`lodedb migrate plan` refuses an ambiguous project (asks for --framework)."""

    (tmp_path / "a.py").write_text("import langchain_core\nimport llama_index\n", encoding="utf-8")
    result = runner.invoke(
        app, ["migrate", "plan", "--project", str(tmp_path), "--target", str(tmp_path / "x")]
    )
    assert result.exit_code == 2


def test_cli_run_and_validate_roundtrip(tmp_path, monkeypatch):
    """`lodedb migrate run` + `validate` drive a plan to a validated store.

    The source open is monkeypatched to an in-memory fixture export so the CLI path
    runs without a real source, exercising the JSON plan -> run -> manifest ->
    validate flow end to end.
    """

    from lodedb.local.migrate import runner as runner_mod
    from lodedb.local.migrate.sources.base import ExportedRow, SourceExport

    class _Fixture(SourceExport):
        def __init__(self):
            super().__init__(
                framework=None,
                provider="pgvector",
                mode="vector-preserve",
                location="postgresql://localhost/app",
                vector_dim=DIM,
                count=2,
            )

        def iter_rows(self):
            yield ExportedRow(id="1", vector=[1.0] + [0.0] * (DIM - 1), metadata={"t": "a"})
            yield ExportedRow(id="2", vector=[0.0, 1.0] + [0.0] * (DIM - 2), metadata={"t": "b"})

    monkeypatch.setattr(runner_mod, "open_source", lambda plan, **kw: _Fixture())

    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(
        det,
        target=tmp_path / "pg",
        embedding_dim=DIM,
        table="docs",
        source="postgresql://localhost/app",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

    # dry-run first
    dry = runner.invoke(app, ["migrate", "run", "--plan", str(plan_path), "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert json.loads(dry.output)["status"] == "dry-run"

    # real run (dry-run is the default, so --write performs the migration)
    run_result = runner.invoke(app, ["migrate", "run", "--plan", str(plan_path), "--write"])
    assert run_result.exit_code == 0, run_result.output
    manifest = json.loads(run_result.output)
    assert manifest["status"] == "migrated"
    assert manifest["target"]["written_count"] == 2

    # validate from the manifest
    validate = runner.invoke(
        app, ["migrate", "validate", "--manifest", str(tmp_path / "pg"), "--json"]
    )
    assert validate.exit_code == 0, validate.output
    report = json.loads(validate.output)
    assert report["passed"] is True
    assert report["audit"]["status"] == "passed"


def test_cli_run_defaults_to_dry_run_and_writes_nothing(tmp_path, monkeypatch):
    """`migrate run` with no flag is a dry run by default and never writes a target.

    The issues require "default to dry run"; the destructive direction must be opt-in
    via ``--write``. A bare ``run`` reports ``dry-run`` and leaves no target on disk.
    """

    from lodedb.local.migrate import runner as runner_mod
    from lodedb.local.migrate.sources.base import ExportedRow, SourceExport

    class _Fixture(SourceExport):
        def __init__(self):
            super().__init__(
                framework=None,
                provider="pgvector",
                mode="vector-preserve",
                location="postgresql://localhost/app",
                vector_dim=DIM,
                count=1,
            )

        def iter_rows(self):
            yield ExportedRow(id="1", vector=[1.0] + [0.0] * (DIM - 1), metadata={"t": "a"})

    monkeypatch.setattr(runner_mod, "open_source", lambda plan, **kw: _Fixture())
    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(det, target=tmp_path / "pg", embedding_dim=DIM, table="docs")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")

    result = runner.invoke(app, ["migrate", "run", "--plan", str(plan_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "dry-run"
    assert not (tmp_path / "pg").exists()
    assert not (tmp_path / "pg.tmp").exists()


def test_cli_run_local_only_and_allow_remote_conflict(tmp_path):
    """`--local-only` and `--allow-remote-source` cannot be combined."""

    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(det, target=tmp_path / "pg", embedding_dim=DIM, table="docs")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    result = runner.invoke(
        app,
        ["migrate", "run", "--plan", str(plan_path), "--local-only", "--allow-remote-source"],
    )
    assert result.exit_code != 0


def test_plan_does_not_store_credentialed_dsn(tmp_path):
    """A credentialed DSN is never written into the JSON plan; run must re-supply it."""

    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(
        det,
        target=tmp_path / "pg",
        embedding_dim=DIM,
        table="docs",
        source="postgresql://user:pw@db.example.com/app",
    )
    data = plan.to_dict()
    # The raw DSN is absent everywhere in the serialized plan.
    assert "user:pw@db.example.com" not in json.dumps(data)
    assert data["source_options"]["location"] == ""
    assert data["source_options"]["location_required"] is True

    # run without --source on such a plan fails with a clear, credential-free message.
    reloaded = MigrationPlan.from_dict(data)
    with pytest.raises(MigrationError):
        run_migration(reloaded, dry_run=True, allow_remote=True)


def test_run_with_resupplied_source_location(tmp_path):
    """A credentialed-source plan runs when the location is re-supplied at run time."""

    det = Detection(route="provider", provider="pgvector")
    plan = build_plan(
        det,
        target=tmp_path / "pg",
        embedding_dim=DIM,
        table="documents",
        source="postgresql://user:pw@localhost/app",
    )
    reloaded = MigrationPlan.from_dict(plan.to_dict())
    conn = _FakeConn(_fake_rows())
    result = run_migration(
        reloaded,
        dry_run=False,
        source_location="postgresql://user:pw@localhost/app",
        pg_connect=lambda dsn: conn,
    )
    assert result.status == "migrated"
    assert result.written_count == 3
