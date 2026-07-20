"""One-command registration of the ``lodedb mcp`` server with coding assistants.

Backs ``lodedb mcp install`` / ``lodedb mcp uninstall``. Registering the stdio MCP
server is otherwise a manual, per-host step: each client keeps its config in a
different place and format (``claude_desktop_config.json``, ``.cursor/mcp.json``,
LM Studio's ``mcp.json``, ``~/.codex/config.toml``), and getting ``command`` right
is error-prone when ``lodedb`` is not on ``PATH``, the common case inside a
``uv``/virtualenv install. This module resolves the right invocation for the
current environment and writes (or removes) the ``lodedb`` entry for a chosen
client.

No storage logic lives here. The written entry simply launches ``lodedb mcp`` with
the options already on that command (``--path``, ``--model``, ``--device``,
``--exclude-text``, ``--no-store-text``), so the server itself is unchanged. Edits
are idempotent (an existing ``lodedb`` entry is updated in place, never duplicated)
and never touch other servers in the file.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The server name every client config is keyed under, and the name passed to
# `claude mcp add` / `codex mcp add`. Stable so installs are idempotent.
SERVER_NAME = "lodedb"

# Clients understood by `--client`. "all" is expanded to every direct-config client
# (Claude Code is excluded from "all" because it is registered via its own CLI, not a
# file this module owns; it is still installable explicitly).
DIRECT_CONFIG_CLIENTS = ("claude-desktop", "cursor", "lm-studio", "codex")
CLI_CLIENTS = ("claude-code",)
KNOWN_CLIENTS = CLI_CLIENTS + DIRECT_CONFIG_CLIENTS
INSTALL_ALL_CLIENTS = DIRECT_CONFIG_CLIENTS


class MCPInstallError(RuntimeError):
    """Raised when a client config cannot be located or written."""


@dataclass(frozen=True)
class ServerInvocation:
    """How to launch ``lodedb mcp`` on this machine: ``command`` plus a leading args list.

    ``args`` already contains the ``mcp`` subcommand and any pass-through options, so a
    client entry is just ``{"command": command, "args": args}``. ``how`` is a short,
    human-readable note on why this form was chosen (printed by the CLI).
    """

    command: str
    args: list[str]
    how: str


def _project_root_for_uv() -> Path | None:
    """Returns the LodeDB project root if this process runs from a source checkout.

    Walks up from this module looking for a ``pyproject.toml`` whose project name is
    ``lodedb`` (a ``uv``/editable install from a clone). Returns ``None`` for a wheel
    install, where there is no project to ``uv run --project`` against.
    """

    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            return None
        # Cheap, dependency-free check: the name line in [project]. Good enough to tell a
        # LodeDB checkout from an unrelated pyproject we happened to walk into.
        if 'name = "lodedb"' in text or "name = 'lodedb'" in text:
            return parent
        return None
    return None


def _entry_point_path() -> Path | None:
    """Returns the absolute path of the ``lodedb`` entry-point script, if it exists.

    Console scripts are installed next to the interpreter (``.venv/bin/lodedb`` or, on
    Windows, ``Scripts\\lodedb.exe``). Used as a PATH-independent absolute command when
    ``lodedb`` is not on ``PATH`` and there is no ``uv`` project to run against.
    """

    bindir = Path(sys.executable).resolve().parent
    candidates = [bindir / "lodedb"]
    if platform.system() == "Windows":
        candidates = [bindir / "lodedb.exe", bindir / "lodedb"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def resolve_server_invocation(*, prefer_uv: bool = False) -> ServerInvocation:
    """Resolves how to launch the MCP server so the written ``command`` works as-is.

    Order, picking the first that applies:

    1. ``lodedb`` on ``PATH``, the cleanest entry (``command="lodedb"``). Skipped when
       ``prefer_uv`` is set.
    2. A source checkout with ``uv`` available: ``uv run --project <root> lodedb`` so the
       entry resolves inside the project's environment without ``PATH``.
    3. The absolute path to the installed ``lodedb`` entry-point script (e.g. the venv's
       ``bin/lodedb``).
    4. Last resort: ``<this python> -m lodedb`` (``lodedb.__main__`` runs the CLI).

    ``prefer_uv`` forces the ``uv run`` form when in a checkout (useful when the on-``PATH``
    ``lodedb`` belongs to a *different* environment than the one being configured).
    """

    on_path = shutil.which("lodedb")
    if on_path and not prefer_uv:
        return ServerInvocation(
            command="lodedb",
            args=["mcp"],
            how="`lodedb` found on PATH",
        )

    root = _project_root_for_uv()
    uv_path = shutil.which("uv")
    if root is not None and uv_path is not None:
        return ServerInvocation(
            command="uv",
            args=["run", "--project", str(root), "lodedb", "mcp"],
            how=f"`uv run` against the source checkout at {root}",
        )

    if on_path:  # prefer_uv was set but there is no usable uv project; PATH still works.
        return ServerInvocation(command="lodedb", args=["mcp"], how="`lodedb` found on PATH")

    entry = _entry_point_path()
    if entry is not None:
        return ServerInvocation(
            command=str(entry),
            args=["mcp"],
            how=f"absolute path to the installed entry point ({entry})",
        )

    return ServerInvocation(
        command=sys.executable,
        args=["-m", "lodedb", "mcp"],
        how=f"`{Path(sys.executable).name} -m lodedb` (entry point not found on PATH)",
    )


@dataclass(frozen=True)
class MCPOptions:
    """The pass-through options shared with ``lodedb mcp`` (mirrors the CLI flags)."""

    path: str = "./data"
    model: str = "minilm"
    device: str = "auto"
    store_text: bool = True
    exclude_text: bool = False

    def to_args(self) -> list[str]:
        """Renders the options as ``lodedb mcp`` CLI args (only non-defaults are emitted).

        ``--path`` is always emitted (it is the one option a user almost always means to
        pin); the rest are added only when they differ from the server's own defaults, so
        a written entry stays minimal and readable.
        """

        args = ["--path", self.path]
        if self.model != "minilm":
            args += ["--model", self.model]
        if self.device != "auto":
            args += ["--device", self.device]
        if not self.store_text:
            args.append("--no-store-text")
        if self.exclude_text:
            args.append("--exclude-text")
        return args


def build_server_entry(invocation: ServerInvocation, options: MCPOptions) -> dict[str, Any]:
    """Builds the ``{command, args}`` entry written under a client's server name."""

    return {"command": invocation.command, "args": [*invocation.args, *options.to_args()]}


# --------------------------------------------------------------------------------------
# Per-client config-file discovery (per-OS).
# --------------------------------------------------------------------------------------


def _home() -> Path:
    return Path.home()


def _claude_desktop_config_path() -> Path:
    """Returns Claude Desktop's ``claude_desktop_config.json`` for this OS.

    macOS / Windows only. Claude Desktop has no official Linux build, so Linux raises a
    clear :class:`MCPInstallError` (use ``--config`` to override if you run a community
    build).
    """

    system = platform.system()
    if system == "Darwin":
        return _home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    if system == "Windows":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else _home() / "AppData" / "Roaming"
        return base / "Claude" / "claude_desktop_config.json"
    raise MCPInstallError(
        "Claude Desktop has no official Linux build, so its config path is unknown here. "
        "Use --config <path> to point at one explicitly, or install for claude-code instead."
    )


def _cursor_config_path(*, project: Path | None = None) -> Path:
    """Returns Cursor's ``mcp.json``: project ``./.cursor/mcp.json`` or global ``~/.cursor``."""

    if project is not None:
        return project / ".cursor" / "mcp.json"
    return _home() / ".cursor" / "mcp.json"


def _lm_studio_config_path() -> Path:
    """Returns LM Studio's ``~/.lmstudio/mcp.json`` (same location on every OS)."""

    return _home() / ".lmstudio" / "mcp.json"


def _codex_config_path() -> Path:
    """Returns Codex's ``config.toml`` (``$CODEX_HOME`` or ``~/.codex``)."""

    codex_home = os.environ.get("CODEX_HOME")
    base = Path(codex_home) if codex_home else _home() / ".codex"
    return base / "config.toml"


def client_config_path(client: str, *, project: Path | None = None) -> Path:
    """Returns the default config path for a direct-config client on this OS."""

    if client == "claude-desktop":
        return _claude_desktop_config_path()
    if client == "cursor":
        return _cursor_config_path(project=project)
    if client == "lm-studio":
        return _lm_studio_config_path()
    if client == "codex":
        return _codex_config_path()
    raise MCPInstallError(f"{client!r} does not use a config file this command can locate")


# --------------------------------------------------------------------------------------
# JSON config editing (Claude Desktop, Cursor, LM Studio).
# --------------------------------------------------------------------------------------


def upsert_json_server(
    config_text: str | None, entry: dict[str, Any], *, name: str = SERVER_NAME
) -> str:
    """Returns config JSON with ``mcpServers[name]`` set to ``entry``, preserving the rest.

    Idempotent: replaces an existing ``name`` entry rather than duplicating it, and leaves
    every other server (and any other top-level key) untouched. ``config_text`` is the
    current file contents (``None`` or empty for a new file). Raises :class:`MCPInstallError`
    if the file exists but is not a JSON object.
    """

    data = _load_json_object(config_text)
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[name] = entry
    data["mcpServers"] = servers
    return json.dumps(data, indent=2) + "\n"


def remove_json_server(config_text: str | None, *, name: str = SERVER_NAME) -> tuple[str, bool]:
    """Returns ``(config_json, removed)`` with ``mcpServers[name]`` deleted if present."""

    data = _load_json_object(config_text)
    servers = data.get("mcpServers")
    removed = isinstance(servers, dict) and name in servers
    if removed:
        del servers[name]
    return json.dumps(data, indent=2) + "\n", removed


def _load_json_object(config_text: str | None) -> dict[str, Any]:
    """Parses existing config text into a dict (empty for a new/blank file)."""

    if not config_text or not config_text.strip():
        return {}
    try:
        data = json.loads(config_text)
    except json.JSONDecodeError as exc:
        raise MCPInstallError(f"existing config is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise MCPInstallError("existing config is not a JSON object")
    return data


# --------------------------------------------------------------------------------------
# TOML config editing (Codex). tomli_w is not a dependency, so we edit the
# `[mcp_servers.<name>]` block textually and preserve the rest of the file verbatim.
# --------------------------------------------------------------------------------------


def _toml_quote(value: str) -> str:
    """Renders a Python string as a TOML basic string (escaped, double-quoted)."""

    escaped = (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
    )
    return f'"{escaped}"'


def _render_codex_block(entry: dict[str, Any], *, name: str) -> str:
    """Renders the ``[mcp_servers.<name>]`` table for a ``{command, args}`` entry."""

    args = ", ".join(_toml_quote(arg) for arg in entry["args"])
    return f"[mcp_servers.{name}]\ncommand = {_toml_quote(entry['command'])}\nargs = [{args}]\n"


def _toml_table_header(line: str) -> str | None:
    """Returns the table path of a TOML header line (``[a.b]`` -> ``a.b``), else ``None``."""

    stripped = line.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        # Covers both [table] and [[array.of.tables]]; we only key off the leading path.
        return stripped.strip("[]").strip()
    return None


def _codex_block_span(lines: list[str], *, name: str) -> tuple[int, int] | None:
    """Finds the line span ``[start, end)`` of the ``[mcp_servers.<name>]`` table.

    The block runs from its header to the next table header at any level (including the
    ``env`` sub-table ``[mcp_servers.<name>.env]``, which is absorbed so it is rewritten
    too). Returns ``None`` if the table is absent.
    """

    target = f"mcp_servers.{name}"
    start = None
    for i, line in enumerate(lines):
        header = _toml_table_header(line)
        if header is None:
            continue
        if start is None and (header == target):
            start = i
            continue
        if start is not None:
            # Sub-tables of this server (e.g. `.env`) belong to the block; any other
            # table header ends it.
            if header == target or header.startswith(target + "."):
                continue
            return (start, i)
    if start is not None:
        return (start, len(lines))
    return None


def upsert_toml_server(
    config_text: str | None, entry: dict[str, Any], *, name: str = SERVER_NAME
) -> str:
    """Returns Codex ``config.toml`` with ``[mcp_servers.<name>]`` set to ``entry``.

    Idempotent and surgical: replaces an existing ``[mcp_servers.<name>]`` block (and its
    sub-tables) in place, or appends a new block; all other content and comments are kept
    verbatim. ``config_text`` is the current file contents (``None`` for a new file).
    """

    block = _render_codex_block(entry, name=name)
    if not config_text or not config_text.strip():
        return block
    lines = config_text.splitlines(keepends=True)
    span = _codex_block_span(lines, name=name)
    if span is None:
        sep = "" if config_text.endswith("\n") else "\n"
        return config_text + sep + "\n" + block
    start, end = span
    replacement = block if block.endswith("\n") else block + "\n"
    # Keep one blank line before whatever followed the old block (e.g. the next table),
    # so a rewrite does not run the new block straight into the following header.
    if end < len(lines) and lines[end].strip() != "":
        replacement += "\n"
    new_lines = lines[:start] + [replacement] + lines[end:]
    return "".join(new_lines)


def remove_toml_server(config_text: str | None, *, name: str = SERVER_NAME) -> tuple[str, bool]:
    """Returns ``(config_toml, removed)`` with the ``[mcp_servers.<name>]`` block removed."""

    if not config_text or not config_text.strip():
        return (config_text or ""), False
    lines = config_text.splitlines(keepends=True)
    span = _codex_block_span(lines, name=name)
    if span is None:
        return config_text, False
    start, end = span
    # Drop one trailing blank line left behind by the removed block, if any.
    if end < len(lines) and lines[end].strip() == "":
        end += 1
    return "".join(lines[:start] + lines[end:]), True


# --------------------------------------------------------------------------------------
# Orchestration: install / uninstall one client, returning a structured result.
# --------------------------------------------------------------------------------------


@dataclass
class ClientResult:
    """Outcome of installing/uninstalling one client (printed and used by tests)."""

    client: str
    action: str  # "install" | "uninstall"
    method: str  # "config" | "cli"
    path: str | None = None  # the config file written (direct-config clients)
    entry: dict[str, Any] | None = None  # the server entry (install)
    cli_command: list[str] | None = None  # the CLI argv (claude-code)
    changed: bool = True  # whether anything would change / was changed
    dry_run: bool = False
    note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def _write_text(path: Path, text: str, *, dry_run: bool) -> None:
    """Writes ``text`` to ``path`` (parents created) unless ``dry_run``."""

    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_text_or_none(path: Path) -> str | None:
    """Returns a file's text, or ``None`` if it does not exist."""

    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _claude_code_argv(entry: dict[str, Any], *, name: str, action: str) -> list[str]:
    """Builds the ``claude mcp add|remove`` argv for the resolved entry.

    Uses the ``--`` separator before the launch command so Claude Code does not parse the
    server's own ``--path`` flag as one of its options.
    """

    if action == "uninstall":
        return ["claude", "mcp", "remove", name]
    return ["claude", "mcp", "add", name, "--", entry["command"], *entry["args"]]


def _install_claude_code(
    entry: dict[str, Any], *, action: str, dry_run: bool, runner: Any | None
) -> ClientResult:
    """Registers (or removes) the server with Claude Code via the ``claude`` CLI."""

    argv = _claude_code_argv(entry, name=SERVER_NAME, action=action)
    result = ClientResult(
        client="claude-code",
        action=action,
        method="cli",
        entry=entry if action == "install" else None,
        cli_command=argv,
        dry_run=dry_run,
    )
    if dry_run:
        result.note = "dry run: would run `" + " ".join(argv) + "`"
        return result
    if shutil.which("claude") is None:
        raise MCPInstallError(
            "the `claude` CLI was not found on PATH; install Claude Code, or write the entry "
            "by hand. Run with --dry-run to print the exact `claude mcp add` command."
        )
    run = runner or _run_subprocess
    code = run(argv)
    result.changed = code == 0
    if code != 0:
        raise MCPInstallError(
            "`" + " ".join(argv) + f"` exited with status {code}. "
            "Run with --dry-run to print the command and run it yourself."
        )
    result.note = "ran `" + " ".join(argv) + "`"
    return result


def _run_subprocess(argv: list[str]) -> int:
    """Runs ``argv`` and returns its exit code (isolated so tests can stub it)."""

    import subprocess

    return subprocess.run(argv).returncode  # noqa: S603 - argv is built from fixed parts


def _install_direct_config(
    client: str,
    entry: dict[str, Any],
    *,
    action: str,
    config_path: Path,
    dry_run: bool,
) -> ClientResult:
    """Installs/uninstalls a direct-config client by editing its JSON/TOML file."""

    is_toml = client == "codex"
    current = _read_text_or_none(config_path)
    if action == "uninstall":
        if is_toml:
            new_text, changed = remove_toml_server(current)
        else:
            new_text, changed = remove_json_server(current)
        if changed:
            _write_text(config_path, new_text, dry_run=dry_run)
        return ClientResult(
            client=client,
            action=action,
            method="config",
            path=str(config_path),
            changed=changed,
            dry_run=dry_run,
            note=("removed the lodedb entry" if changed else "no lodedb entry to remove"),
        )

    new_text = upsert_toml_server(current, entry) if is_toml else upsert_json_server(current, entry)
    _write_text(config_path, new_text, dry_run=dry_run)
    return ClientResult(
        client=client,
        action=action,
        method="config",
        path=str(config_path),
        entry=entry,
        dry_run=dry_run,
        note=("dry run: not written" if dry_run else "wrote the lodedb entry"),
    )


def install_client(
    client: str,
    *,
    action: str = "install",
    options: MCPOptions | None = None,
    invocation: ServerInvocation | None = None,
    config_path: Path | None = None,
    project: Path | None = None,
    prefer_uv: bool = False,
    dry_run: bool = False,
    _runner: Any | None = None,
) -> ClientResult:
    """Installs or uninstalls the LodeDB MCP server for one client.

    ``action`` is ``"install"`` or ``"uninstall"``. ``options`` are the pass-through
    ``lodedb mcp`` flags; ``invocation`` overrides the auto-resolved launch command;
    ``config_path`` overrides the client's default config location (``--config``);
    ``project`` selects Cursor's project-level ``.cursor/mcp.json``. ``dry_run`` computes
    the result without writing or shelling out.
    """

    if client not in KNOWN_CLIENTS:
        raise MCPInstallError(
            f"unknown client {client!r}; choose one of: {', '.join(KNOWN_CLIENTS)}, all"
        )
    options = options or MCPOptions()
    invocation = invocation or resolve_server_invocation(prefer_uv=prefer_uv)
    entry = build_server_entry(invocation, options)

    if client == "claude-code":
        return _install_claude_code(entry, action=action, dry_run=dry_run, runner=_runner)

    path = config_path or client_config_path(client, project=project)
    return _install_direct_config(client, entry, action=action, config_path=path, dry_run=dry_run)


def expand_clients(client: str) -> list[str]:
    """Expands ``"all"`` to the direct-config clients; otherwise returns ``[client]``."""

    if client == "all":
        return list(INSTALL_ALL_CLIENTS)
    return [client]
