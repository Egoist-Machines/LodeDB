"""``lodedb mcp install`` — one-command MCP registration for coding assistants.

Edits the host's JSON/TOML config for Claude Desktop, Cursor, LM Studio, and
Codex; for Claude Code it shells out to ``claude mcp add``.  Reuses the same
``--path``, ``--model``, ``--device``, and ``--no-store-text`` options from
``lodedb mcp`` so the generated entry matches the current environment.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer

# -- per-OS config discovery -------------------------------------------------


def _config_paths(client: str) -> list[Path]:
    """Returns default config file paths for *client*, or all JSON clients."""

    home = Path.home()
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA", home / "AppData" / "Roaming"))
        claude_dir = appdata / "Claude"
    elif sys.platform == "darwin":
        claude_dir = home / "Library" / "Application Support" / "Claude"
    else:
        claude_dir = home / ".config" / "Claude"

    paths: dict[str, list[Path]] = {
        "claude-desktop": [claude_dir / "claude_desktop_config.json"],
        "cursor": [home / ".cursor" / "mcp.json"],
        "lm-studio": [home / ".lmstudio" / "mcp.json"],
    }
    if client == "all":
        return [p for group in paths.values() for p in group]
    return paths.get(client, [])


# -- entry builders ----------------------------------------------------------


def _server_entry(opts: dict[str, Any]) -> dict[str, Any]:
    """Builds the ``{command, args}`` entry for the current venv."""

    args = ["-m", "lodedb", "mcp"]

    path = opts.get("path")
    if path:
        args.extend(["--path", str(Path(path).absolute())])

    model = opts.get("model")
    if model and model != "minilm":
        args.extend(["--model", model])

    device = opts.get("device")
    if device and device != "auto":
        args.extend(["--device", device])

    if not opts.get("store_text", True):
        args.append("--no-store-text")

    return {"command": sys.executable, "args": args}


def _toml_block(entry: dict[str, Any]) -> str:
    """Formats *entry* as a ``[mcpServers.lodedb]`` TOML fragment."""

    cmd = json.dumps(entry["command"])
    args = ", ".join(json.dumps(a) for a in entry["args"])
    return f"[mcpServers.lodedb]\ncommand = {cmd}\nargs = [{args}]\n"


def _claude_code_cmd(opts: dict[str, Any]) -> list[str]:
    """Builds the ``claude mcp add`` argv, or returns ``[]`` if not on PATH."""

    exe = shutil.which("claude")
    if not exe:
        return []
    cmd = [exe, "mcp", "add", "lodedb", "--", sys.executable, "-m", "lodedb", "mcp"]
    path = opts.get("path")
    if path:
        cmd.extend(["--path", str(Path(path).absolute())])
    model = opts.get("model")
    if model and model != "minilm":
        cmd.extend(["--model", model])
    device = opts.get("device")
    if device and device != "auto":
        cmd.extend(["--device", device])
    if not opts.get("store_text", True):
        cmd.append("--no-store-text")
    return cmd


# -- config writers -----------------------------------------------------------


def _write_json(path: Path, entry: dict[str, Any]) -> bool:
    """Upserts ``mcpServers.lodedb`` in a JSON config, preserving neighbours."""

    if not path.parent.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            typer.echo(f"cannot create {path.parent}: {exc}", err=True)
            return False

    data: dict[str, Any] = {}
    if path.exists():
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            try:
                data = json.loads(raw)
            except Exception as exc:
                typer.echo(f"{path} is not valid JSON: {exc}", err=True)
                return False

    data.setdefault("mcpServers", {})["lodedb"] = entry

    try:
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        typer.echo(f"cannot write {path}: {exc}", err=True)
        return False
    return True


def _append_toml(path: Path, entry: dict[str, Any]) -> bool:
    """Appends a ``[mcpServers.lodedb]`` block if not already present."""

    if not path.parent.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            typer.echo(f"cannot create {path.parent}: {exc}", err=True)
            return False

    if path.exists() and "[mcpServers.lodedb]" in path.read_text(encoding="utf-8"):
        return True  # idempotent

    try:
        with path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + _toml_block(entry))
    except Exception as exc:
        typer.echo(f"cannot write {path}: {exc}", err=True)
        return False
    return True


# -- public entry point -------------------------------------------------------


def install(
    client: str,
    dry_run: bool,
    config_override: Path | None,
    **opts: Any,
) -> None:
    """Registers the LodeDB MCP server with one or all supported clients."""

    entry = _server_entry(opts)
    entry_json = json.dumps({"mcpServers": {"lodedb": entry}}, indent=2)

    # Claude Code — shell out to `claude mcp add`
    if client in ("claude-code", "all"):
        cmd = _claude_code_cmd(opts)
        if cmd:
            if dry_run:
                typer.echo(f"[dry run] {' '.join(cmd)}")
            else:
                typer.echo(f"running: {' '.join(cmd)}")
                subprocess.run(cmd, check=False)
        elif client == "claude-code":
            typer.echo("claude CLI not found in PATH", err=True)
            return
        if client == "claude-code":
            return

    # Codex — TOML config
    if client in ("codex", "all"):
        codex_cfg = (
            config_override
            if config_override and client == "codex"
            else Path.home() / ".codex" / "config.toml"
        )
        block = _toml_block(entry)
        if dry_run:
            typer.echo(f"\n[dry run] {codex_cfg}:\n{block}")
        elif _append_toml(codex_cfg, entry):
            typer.echo(f"wrote {codex_cfg}:\n{block}")
        if client == "codex":
            return

    # JSON clients — Claude Desktop, Cursor, LM Studio
    paths = [config_override] if config_override and client != "all" else _config_paths(client)
    if not paths and client != "all":
        typer.echo(f"unknown client: {client}", err=True)
        return

    for cfg in paths:
        if dry_run:
            typer.echo(f"\n[dry run] {cfg}:\n{entry_json}")
        elif _write_json(cfg, entry):
            typer.echo(f"wrote {cfg}:\n{entry_json}")
