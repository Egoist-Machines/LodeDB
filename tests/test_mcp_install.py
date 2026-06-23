"""Tests for ``lodedb mcp install`` / ``lodedb mcp uninstall``.

The config-editing helpers (JSON for Claude Desktop / Cursor / LM Studio, TOML for
Codex) are pure string transforms and are exercised directly. The launch-command
resolver and per-OS config-path discovery are unit-tested with monkeypatched
``platform``/``shutil.which``. The CLI surface is driven through Typer's
``CliRunner`` against temp config files (the ``claude-code`` path stubs out the
subprocess so no real ``claude`` is invoked).
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

import lodedb.local.mcp_install as mi
from lodedb.local.cli import app
from lodedb.local.mcp_install import (
    MCPInstallError,
    MCPOptions,
    ServerInvocation,
    build_server_entry,
    client_config_path,
    expand_clients,
    install_client,
    remove_json_server,
    remove_toml_server,
    resolve_server_invocation,
    upsert_json_server,
    upsert_toml_server,
)

runner = CliRunner()

_INVOCATION = ServerInvocation(command="lodedb", args=["mcp"], how="test")


# --------------------------------------------------------------------------------------
# Option -> args, and the server entry.
# --------------------------------------------------------------------------------------


def test_options_render_only_non_default_flags():
    """``--path`` is always emitted; other flags appear only when they differ from defaults."""

    assert MCPOptions(path="./data").to_args() == ["--path", "./data"]
    assert MCPOptions(
        path="/d", model="bge", device="cuda", store_text=False, exclude_text=True
    ).to_args() == [
        "--path",
        "/d",
        "--model",
        "bge",
        "--device",
        "cuda",
        "--no-store-text",
        "--exclude-text",
    ]


def test_build_server_entry_concatenates_invocation_and_options():
    """The written entry is ``{command, args}`` with the invocation args before the options."""

    entry = build_server_entry(_INVOCATION, MCPOptions(path="./data", model="bge"))
    assert entry == {"command": "lodedb", "args": ["mcp", "--path", "./data", "--model", "bge"]}


# --------------------------------------------------------------------------------------
# Launch-command resolution.
# --------------------------------------------------------------------------------------


def test_resolve_prefers_lodedb_on_path(monkeypatch):
    """When ``lodedb`` is on PATH the resolver uses it verbatim."""

    monkeypatch.setattr(
        mi.shutil, "which", lambda name: "/usr/bin/lodedb" if name == "lodedb" else None
    )
    inv = resolve_server_invocation()
    assert inv.command == "lodedb"
    assert inv.args == ["mcp"]


def test_resolve_falls_back_to_uv_run_when_not_on_path(monkeypatch, tmp_path):
    """Off PATH but inside a checkout with ``uv``, the resolver uses ``uv run --project``."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: "/opt/uv" if name == "uv" else None)
    monkeypatch.setattr(mi, "_project_root_for_uv", lambda: tmp_path)
    inv = resolve_server_invocation()
    assert inv.command == "uv"
    assert inv.args == ["run", "--project", str(tmp_path), "lodedb", "mcp"]


def test_resolve_prefer_uv_overrides_path(monkeypatch, tmp_path):
    """``prefer_uv`` forces the ``uv run`` form even when ``lodedb`` is on PATH."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(mi, "_project_root_for_uv", lambda: tmp_path)
    inv = resolve_server_invocation(prefer_uv=True)
    assert inv.command == "uv"
    assert inv.args[:3] == ["run", "--project", str(tmp_path)]


def test_resolve_falls_back_to_entry_point_then_python_module(monkeypatch, tmp_path):
    """With no PATH ``lodedb`` and no uv project: absolute entry point, else ``python -m``."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: None)
    monkeypatch.setattr(mi, "_project_root_for_uv", lambda: None)

    entry = tmp_path / "lodedb"
    entry.write_text("#!/bin/sh\n")
    monkeypatch.setattr(mi, "_entry_point_path", lambda: entry)
    inv = resolve_server_invocation()
    assert inv.command == str(entry)
    assert inv.args == ["mcp"]

    monkeypatch.setattr(mi, "_entry_point_path", lambda: None)
    inv = resolve_server_invocation()
    assert inv.command == mi.sys.executable
    assert inv.args == ["-m", "lodedb", "mcp"]


# --------------------------------------------------------------------------------------
# Per-OS config-path discovery.
# --------------------------------------------------------------------------------------


def test_claude_desktop_path_macos_windows_and_linux_error(monkeypatch):
    """Claude Desktop resolves macOS/Windows paths and raises a clear error on Linux."""

    monkeypatch.setattr(mi.platform, "system", lambda: "Darwin")
    assert (
        client_config_path("claude-desktop")
        .as_posix()
        .endswith("Library/Application Support/Claude/claude_desktop_config.json")
    )

    monkeypatch.setattr(mi.platform, "system", lambda: "Windows")
    monkeypatch.setenv("APPDATA", str(Path("/Roaming")))
    win = client_config_path("claude-desktop")
    assert win.name == "claude_desktop_config.json"
    assert "Claude" in win.parts

    monkeypatch.setattr(mi.platform, "system", lambda: "Linux")
    with pytest.raises(MCPInstallError, match="no official Linux build"):
        client_config_path("claude-desktop")


def test_codex_path_honors_codex_home(monkeypatch, tmp_path):
    """Codex's config path follows ``$CODEX_HOME`` and defaults to ``~/.codex``."""

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    assert client_config_path("codex") == tmp_path / "config.toml"
    monkeypatch.delenv("CODEX_HOME", raising=False)
    assert client_config_path("codex").name == "config.toml"


def test_cursor_project_vs_global_path(tmp_path):
    """Cursor resolves a project-level ``.cursor/mcp.json`` and a global one in ``~``."""

    assert client_config_path("cursor", project=tmp_path) == tmp_path / ".cursor" / "mcp.json"
    assert client_config_path("cursor").name == "mcp.json"


def test_lm_studio_path():
    """LM Studio resolves ``~/.lmstudio/mcp.json``."""

    assert client_config_path("lm-studio").as_posix().endswith(".lmstudio/mcp.json")


# --------------------------------------------------------------------------------------
# JSON config editing (Claude Desktop, Cursor, LM Studio).
# --------------------------------------------------------------------------------------


def test_upsert_json_creates_then_updates_in_place():
    """Upsert creates ``mcpServers.lodedb`` on a blank file and replaces it on re-run."""

    entry1 = {"command": "lodedb", "args": ["mcp", "--path", "./data"]}
    text = upsert_json_server(None, entry1)
    assert json.loads(text)["mcpServers"]["lodedb"] == entry1

    entry2 = {"command": "lodedb", "args": ["mcp", "--path", "./other"]}
    text2 = upsert_json_server(text, entry2)
    servers = json.loads(text2)["mcpServers"]
    assert servers["lodedb"] == entry2
    assert list(servers).count("lodedb") == 1  # updated, not duplicated


def test_upsert_json_preserves_other_servers_and_keys():
    """Upsert never clobbers other servers or unrelated top-level keys."""

    existing = json.dumps({"mcpServers": {"other": {"command": "x", "args": []}}, "theme": "dark"})
    out = json.loads(upsert_json_server(existing, {"command": "lodedb", "args": ["mcp"]}))
    assert out["mcpServers"]["other"] == {"command": "x", "args": []}
    assert out["theme"] == "dark"
    assert out["mcpServers"]["lodedb"] == {"command": "lodedb", "args": ["mcp"]}


def test_remove_json_server_reports_presence_and_keeps_others():
    """Remove deletes only ``lodedb`` and reports whether it was there."""

    existing = json.dumps(
        {"mcpServers": {"lodedb": {"command": "lodedb", "args": []}, "other": {"command": "x"}}}
    )
    text, removed = remove_json_server(existing)
    assert removed is True
    servers = json.loads(text)["mcpServers"]
    assert "lodedb" not in servers and "other" in servers

    _, removed_again = remove_json_server(text)
    assert removed_again is False


def test_upsert_json_rejects_non_object_config():
    """A malformed (non-object) config raises a clear error rather than silently overwriting."""

    with pytest.raises(MCPInstallError):
        upsert_json_server("[1, 2, 3]", {"command": "lodedb", "args": []})


# --------------------------------------------------------------------------------------
# TOML config editing (Codex).
# --------------------------------------------------------------------------------------


def test_upsert_toml_creates_block_on_blank_file():
    """Upsert writes a ``[mcp_servers.lodedb]`` table that round-trips through tomllib."""

    text = upsert_toml_server(None, {"command": "lodedb", "args": ["mcp", "--path", "./data"]})
    parsed = tomllib.loads(text)
    assert parsed["mcp_servers"]["lodedb"] == {
        "command": "lodedb",
        "args": ["mcp", "--path", "./data"],
    }


def test_upsert_toml_replaces_in_place_and_preserves_rest():
    """Re-running upsert replaces the lodedb block once and keeps other tables + comments."""

    existing = (
        "# my config\n"
        'model = "gpt-5"\n\n'
        "[mcp_servers.context7]\n"
        'command = "npx"\n'
        'args = ["-y", "ctx"]\n\n'
        "[mcp_servers.context7.env]\n"
        'KEY = "secret"\n\n'
        "[mcp_servers.lodedb]\n"
        'command = "OLD"\n'
        'args = ["mcp", "--path", "OLD"]\n\n'
        "[tui]\n"
        'theme = "dark"\n'
    )
    out = upsert_toml_server(existing, {"command": "lodedb", "args": ["mcp", "--path", "./new"]})
    parsed = tomllib.loads(out)
    # lodedb updated, everything else intact.
    assert parsed["mcp_servers"]["lodedb"] == {
        "command": "lodedb",
        "args": ["mcp", "--path", "./new"],
    }
    assert parsed["mcp_servers"]["context7"]["env"] == {"KEY": "secret"}
    assert parsed["tui"] == {"theme": "dark"}
    assert parsed["model"] == "gpt-5"
    assert "# my config" in out  # comment preserved
    assert out.count("[mcp_servers.lodedb]") == 1  # no duplicate block


def test_upsert_toml_appends_when_absent():
    """When no lodedb block exists, a new one is appended without touching the rest."""

    existing = '[mcp_servers.context7]\ncommand = "npx"\nargs = []\n'
    out = upsert_toml_server(existing, {"command": "lodedb", "args": ["mcp"]})
    parsed = tomllib.loads(out)
    assert set(parsed["mcp_servers"]) == {"context7", "lodedb"}


def test_remove_toml_server_drops_block_and_subtables():
    """Remove drops the lodedb table and its ``.env`` sub-table, keeping other servers."""

    existing = (
        "[mcp_servers.lodedb]\n"
        'command = "lodedb"\n'
        'args = ["mcp"]\n\n'
        "[mcp_servers.lodedb.env]\n"
        'A = "1"\n\n'
        "[mcp_servers.other]\n"
        'command = "x"\n'
        "args = []\n"
    )
    out, removed = remove_toml_server(existing)
    assert removed is True
    parsed = tomllib.loads(out)
    assert "lodedb" not in parsed["mcp_servers"]
    assert parsed["mcp_servers"]["other"] == {"command": "x", "args": []}

    _, removed_again = remove_toml_server(out)
    assert removed_again is False


def test_toml_quotes_special_characters():
    """Backslashes/quotes in a Windows-style command are escaped to valid TOML."""

    weird = {"command": r"C:\Program Files\lodedb.exe", "args": ['a "b"']}
    out = upsert_toml_server(None, weird)
    assert tomllib.loads(out)["mcp_servers"]["lodedb"] == weird


# --------------------------------------------------------------------------------------
# install_client orchestration (no real files / subprocess except via tmp_path).
# --------------------------------------------------------------------------------------


def test_install_client_writes_json_file_for_cursor(tmp_path):
    """install_client writes a Cursor config and is idempotent across repeats."""

    cfg = tmp_path / "mcp.json"
    result = install_client(
        "cursor", options=MCPOptions(path="./data"), invocation=_INVOCATION, config_path=cfg
    )
    assert result.method == "config" and result.changed
    assert json.loads(cfg.read_text())["mcpServers"]["lodedb"]["command"] == "lodedb"

    # Re-run with a different option: still exactly one entry, now updated.
    install_client(
        "cursor",
        options=MCPOptions(path="./data", model="bge"),
        invocation=_INVOCATION,
        config_path=cfg,
    )
    servers = json.loads(cfg.read_text())["mcpServers"]
    assert servers["lodedb"]["args"][-2:] == ["--model", "bge"]


def test_install_client_dry_run_does_not_write(tmp_path):
    """A dry-run install computes the entry but leaves the file untouched."""

    cfg = tmp_path / "mcp.json"
    result = install_client("cursor", invocation=_INVOCATION, config_path=cfg, dry_run=True)
    assert result.dry_run and result.entry is not None
    assert not cfg.exists()


def test_uninstall_client_removes_entry(tmp_path):
    """install_client(action='uninstall') drops the entry and reports the change."""

    cfg = tmp_path / "mcp.json"
    install_client("cursor", invocation=_INVOCATION, config_path=cfg)
    result = install_client("cursor", action="uninstall", config_path=cfg)
    assert result.changed
    assert "lodedb" not in json.loads(cfg.read_text())["mcpServers"]


def test_install_claude_code_builds_add_argv_with_separator(monkeypatch):
    """claude-code install shells out to ``claude mcp add lodedb -- lodedb mcp ...``."""

    calls: list[list[str]] = []
    monkeypatch.setattr(mi.shutil, "which", lambda name: "/usr/bin/claude")
    result = install_client(
        "claude-code",
        options=MCPOptions(path="./data"),
        invocation=_INVOCATION,
        _runner=lambda argv: calls.append(argv) or 0,
    )
    assert result.method == "cli" and result.changed
    assert calls == [["claude", "mcp", "add", "lodedb", "--", "lodedb", "mcp", "--path", "./data"]]


def test_install_claude_code_errors_when_cli_missing(monkeypatch):
    """A real (non-dry-run) claude-code install errors clearly if ``claude`` is absent."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: None)
    with pytest.raises(MCPInstallError, match="`claude` CLI was not found"):
        install_client("claude-code", invocation=_INVOCATION)


def test_install_claude_code_dry_run_skips_subprocess(monkeypatch):
    """A dry-run claude-code install builds the argv without needing ``claude`` on PATH."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: None)
    result = install_client("claude-code", invocation=_INVOCATION, dry_run=True)
    assert result.cli_command[:4] == ["claude", "mcp", "add", "lodedb"]


def test_unknown_client_raises():
    """An unrecognized client name is rejected."""

    with pytest.raises(MCPInstallError, match="unknown client"):
        install_client("emacs", invocation=_INVOCATION)


def test_expand_clients_handles_all():
    """``all`` expands to the direct-config clients; a single name passes through."""

    assert expand_clients("all") == list(mi.INSTALL_ALL_CLIENTS)
    assert "claude-code" not in expand_clients("all")  # CLI client is explicit-only
    assert expand_clients("cursor") == ["cursor"]


# --------------------------------------------------------------------------------------
# CLI surface.
# --------------------------------------------------------------------------------------


def test_cli_mcp_help_lists_install_and_uninstall():
    """``lodedb mcp --help`` advertises the install/uninstall subcommands."""

    result = runner.invoke(app, ["mcp", "--help"])
    assert result.exit_code == 0, result.output
    assert "install" in result.output
    assert "uninstall" in result.output


def test_cli_install_cursor_writes_and_prints_entry(tmp_path, monkeypatch):
    """``lodedb mcp install --client cursor --config ...`` writes the entry and prints it."""

    monkeypatch.setattr(
        mi.shutil, "which", lambda name: "/usr/bin/lodedb" if name == "lodedb" else None
    )
    cfg = tmp_path / "mcp.json"
    result = runner.invoke(
        app,
        ["mcp", "install", "--client", "cursor", "--config", str(cfg), "--path", "./data"],
    )
    assert result.exit_code == 0, result.output
    assert "wrote the lodedb entry" in result.output
    assert str(cfg) in result.output
    assert json.loads(cfg.read_text())["mcpServers"]["lodedb"]["command"] == "lodedb"


def test_cli_install_dry_run_writes_nothing(tmp_path, monkeypatch):
    """``--dry-run`` prints the entry but does not create the file."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: "/usr/bin/lodedb")
    cfg = tmp_path / "mcp.json"
    result = runner.invoke(
        app,
        ["mcp", "install", "-c", "cursor", "--config", str(cfg), "--dry-run"],
    )
    assert result.exit_code == 0, result.output
    assert not cfg.exists()
    assert "dry run" in result.output


def test_cli_install_all_rejects_explicit_config(tmp_path):
    """``--client all`` cannot be combined with a single ``--config`` path."""

    result = runner.invoke(
        app, ["mcp", "install", "--client", "all", "--config", str(tmp_path / "x.json")]
    )
    assert result.exit_code != 0
    assert "all" in result.output


def test_cli_install_passes_through_options(tmp_path, monkeypatch):
    """The CLI forwards --model/--device/--exclude-text into the written args."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: "/usr/bin/lodedb")
    cfg = tmp_path / "mcp.json"
    result = runner.invoke(
        app,
        [
            "mcp",
            "install",
            "-c",
            "cursor",
            "--config",
            str(cfg),
            "--path",
            "/d",
            "--model",
            "bge",
            "--device",
            "cuda",
            "--exclude-text",
        ],
    )
    assert result.exit_code == 0, result.output
    args = json.loads(cfg.read_text())["mcpServers"]["lodedb"]["args"]
    # The CLI resolves --path to an absolute path (drive-anchored on Windows, so `/d` becomes
    # e.g. `D:\d`); assert the structure and the pass-through flags rather than pinning the
    # platform-specific path string.
    assert args[:2] == ["mcp", "--path"]
    assert Path(args[2]).is_absolute()
    assert args[3:] == ["--model", "bge", "--device", "cuda", "--exclude-text"]


def test_cli_install_resolves_relative_path_to_absolute(tmp_path, monkeypatch):
    """A relative ``--path`` is resolved to absolute in the written entry.

    A coding assistant launches the server with its own working directory, so a relative
    data path in the entry would point somewhere unintended. The CLI resolves it against
    the install-time CWD before writing.
    """

    monkeypatch.setattr(mi.shutil, "which", lambda name: "/usr/bin/lodedb")
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "mcp.json"
    result = runner.invoke(
        app, ["mcp", "install", "-c", "cursor", "--config", str(cfg), "--path", "./data"]
    )
    assert result.exit_code == 0, result.output
    args = json.loads(cfg.read_text())["mcpServers"]["lodedb"]["args"]
    path_value = args[args.index("--path") + 1]
    assert Path(path_value).is_absolute()
    assert path_value.endswith("data")


def test_cli_uninstall_removes_entry(tmp_path, monkeypatch):
    """``lodedb mcp uninstall`` removes a previously written entry."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: "/usr/bin/lodedb")
    cfg = tmp_path / "mcp.json"
    runner.invoke(app, ["mcp", "install", "-c", "cursor", "--config", str(cfg)])
    result = runner.invoke(app, ["mcp", "uninstall", "-c", "cursor", "--config", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "lodedb" not in json.loads(cfg.read_text())["mcpServers"]


def test_cli_install_codex_writes_toml(tmp_path, monkeypatch):
    """``--client codex`` writes a TOML ``[mcp_servers.lodedb]`` table that parses."""

    monkeypatch.setattr(mi.shutil, "which", lambda name: "/usr/bin/lodedb")
    cfg = tmp_path / "config.toml"
    result = runner.invoke(
        app, ["mcp", "install", "-c", "codex", "--config", str(cfg), "--path", "./data"]
    )
    assert result.exit_code == 0, result.output
    parsed = tomllib.loads(cfg.read_text())
    assert parsed["mcp_servers"]["lodedb"]["command"] == "lodedb"
