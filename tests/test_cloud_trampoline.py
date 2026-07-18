"""The `lodedb cloud` trampoline: forwards argv to the first-party cloud CLI
(`lodedb.cloud.cli`, whose modules pull the [cloud] extra's dependencies),
and answers a clear install hint — not a traceback — when those dependencies
are absent. Both paths are forced deterministically, so the suite never
depends on the venv's extras."""

from __future__ import annotations

import sys

import typer
from typer.testing import CliRunner

from lodedb.local.cli import app

runner = CliRunner()


def _all_output(result) -> str:
    """stdout + stderr regardless of how this click version splits them."""
    output = result.output
    try:
        output += result.stderr
    except (ValueError, AttributeError):
        pass
    return output


def test_missing_cloud_deps_print_install_hint(monkeypatch):
    # A None entry makes `import lodedb.cloud.cli` raise ImportError even
    # though the extra's dependencies are installed in the dev venv.
    monkeypatch.setitem(sys.modules, "lodedb.cloud.cli", None)
    result = runner.invoke(app, ["cloud", "sync", "./somewhere"])
    assert result.exit_code == 1
    assert 'pip install "lodedb[cloud]"' in _all_output(result)


def test_argv_forwards_to_the_cloud_cli(monkeypatch):
    """Everything after `lodedb cloud` reaches the cloud app untouched —
    subcommand, positionals, and options the trampoline knows nothing about."""
    captured: dict[str, object] = {}
    fake_app = typer.Typer()

    @fake_app.command("sync")
    def sync(target: str, note: str = typer.Option("", "--note")) -> None:
        captured["target"] = target
        captured["note"] = note

    @fake_app.command("status")
    def status() -> None:  # second command keeps the fake app a group
        captured["status"] = True

    monkeypatch.setattr("lodedb.cloud.cli.app", fake_app)

    result = runner.invoke(app, ["cloud", "sync", "./my-store", "--note", "hi"])
    assert result.exit_code == 0, _all_output(result)
    assert captured == {"target": "./my-store", "note": "hi"}
