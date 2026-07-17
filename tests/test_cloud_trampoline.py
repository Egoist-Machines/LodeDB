"""The `lodedb cloud` trampoline: forwards argv to the optional `orecloud`
CLI ([cloud] extra), and answers a clear install hint — not a traceback —
when the client is absent. Both paths are forced deterministically, so the
suite never depends on whether orecloud happens to be installed."""

from __future__ import annotations

import sys
import types

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


def test_missing_client_prints_install_hint(monkeypatch):
    # A None entry makes `import orecloud` raise ImportError even when the
    # real package is installed in the venv.
    monkeypatch.setitem(sys.modules, "orecloud", None)
    monkeypatch.setitem(sys.modules, "orecloud.cli", None)
    result = runner.invoke(app, ["cloud", "sync", "./somewhere"])
    assert result.exit_code == 1
    assert 'pip install "lodedb[cloud]"' in _all_output(result)


def test_argv_forwards_to_the_cloud_cli(monkeypatch):
    """Everything after `lodedb cloud` reaches the orecloud app untouched —
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

    fake_cli = types.ModuleType("orecloud.cli")
    fake_cli.app = fake_app
    fake_pkg = types.ModuleType("orecloud")
    fake_pkg.cli = fake_cli
    monkeypatch.setitem(sys.modules, "orecloud", fake_pkg)
    monkeypatch.setitem(sys.modules, "orecloud.cli", fake_cli)

    result = runner.invoke(app, ["cloud", "sync", "./my-store", "--note", "hi"])
    assert result.exit_code == 0, _all_output(result)
    assert captured == {"target": "./my-store", "note": "hi"}
