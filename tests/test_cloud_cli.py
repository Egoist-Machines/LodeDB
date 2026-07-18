"""CLI smoke tests (`lodedb cloud ...` via typer's CliRunner): each verb runs the
same round trip the API tests cover, plus clean nonzero-exit error handling.

CliRunner pipes stdout (not a TTY), so these tests exercise the agent-facing
default: JSON on stdout. `--no-json` pins the human render."""

import json

from typer.testing import CliRunner

from lodedb.cloud.cli import app

runner = CliRunner()


def test_cli_round_trip(committed_store, tmp_path):
    source, key = committed_store
    remote = tmp_path / "remote"
    restored = tmp_path / "restored"

    result = runner.invoke(app, ["keys", str(source)])
    assert result.exit_code == 0
    assert json.loads(result.output) == [key]

    result = runner.invoke(app, ["push", str(source), str(remote), key])
    assert result.exit_code == 0
    assert json.loads(result.output)["pointer_published"] is True

    result = runner.invoke(app, ["status", str(source), str(remote), key])
    assert result.exit_code == 0
    assert json.loads(result.output)["in_sync"] is True

    result = runner.invoke(app, ["verify", str(remote), key])
    assert result.exit_code == 0

    result = runner.invoke(app, ["pull", str(remote), str(restored), key])
    assert result.exit_code == 0
    assert "document_count" in json.loads(result.output)


def test_cli_human_output_on_request(committed_store, tmp_path):
    """`--no-json` forces the aligned field/value render even when piped."""
    source, key = committed_store
    remote = tmp_path / "remote"
    result = runner.invoke(app, ["--no-json", "push", str(source), str(remote), key])
    assert result.exit_code == 0
    assert "pointer_published  True" in result.output


def test_cli_missing_generation_exits_nonzero(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["verify", str(empty), "no-such-key"])
    assert result.exit_code == 1


def test_cli_bad_scheme_exits_nonzero(tmp_path):
    result = runner.invoke(app, ["verify", "ftp://nope/x", "idx"])
    assert result.exit_code == 1
