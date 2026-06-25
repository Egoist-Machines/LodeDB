"""`lodedb` CLI: serve | index | query | benchmark | doctor.

Thin wrapper over the local SDK (:class:`LodeDB`) and the existing engine — no
retrieval/storage logic is duplicated here. ``doctor`` reuses
:func:`local_capability_report`; ``index``/``query``/``benchmark`` drive
``LodeDB``; ``serve`` runs a minimal local HTTP loop over the same in-process
engine the SDK uses (loopback, no auth).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from lodedb.local.backends import resolve_local_device
from lodedb.local.db import LodeDB
from lodedb.local.doctor import format_capability_report, local_capability_report

app = typer.Typer(
    help="LodeDB — local-first embedded vector DB with optional CUDA batch search. "
    "Your data stays on your machine.",
    no_args_is_help=True,
)

_PATH_OPTION = typer.Option(Path("./data"), "--path", "-p", help="On-disk LodeDB directory.")
_MODEL_OPTION = typer.Option(
    "minilm", "--model", "-m", help="Preset: minilm (fast) | bge (quality)."
)
_DEVICE_OPTION = typer.Option(
    "auto", "--device", "-d", help="auto | cpu | mps | cuda (embedding only)."
)
_RUNTIME_OPTION = typer.Option(
    "auto",
    "--runtime",
    "-r",
    help="Embedding runtime: auto (prefer ONNX, fall back to torch) | onnx | torch.",
)
_TEXTS_ARGUMENT = typer.Argument(None, help="Document texts to add (or use --file).")
_FILE_OPTION = typer.Option(None, "--file", "-f", help="A text file: one document per line.")
_BENCH_PATH_OPTION = typer.Option(
    None, "--path", "-p", help="On-disk dir (default: ephemeral temp dir)."
)
_STORE_TEXT_OPTION = typer.Option(
    True,
    "--store-text/--no-store-text",
    "-t",
    help="Retain raw text so `lodedb get ID` can return it (default on; "
    "--no-store-text opts out).",
)
_EXCLUDE_TEXT_OPTION = typer.Option(
    False,
    "--exclude-text",
    help="MCP only: redact document text from the server. `lodedb_search` returns "
    "metrics only and the get-by-id tool is withdrawn, while text stays on disk for "
    "hybrid search. By default (text retained) search returns each hit's text.",
)
_DURABILITY_OPTION = typer.Option(
    None,
    "--durability",
    help="fast (default: atomic but not power-loss durable) | fsync (fsync each "
    "file + dir on commit, durable but slower). Unset reads LODEDB_DURABILITY.",
)
_COMMIT_MODE_OPTION = typer.Option(
    None,
    "--commit-mode",
    help="generation (default: publish a crash-atomic MVCC generation per commit) | "
    "wal (append per mutation, checkpoint periodically; ~10x faster single adds, "
    "single-writer). Unset reads LODEDB_COMMIT_MODE.",
)


@app.command()
def doctor(
    device: str = typer.Option("auto", "--device", "-d", help="Device to report resolution for."),
    json_out: bool = typer.Option(False, "--json", help="Emit the raw capability JSON."),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="If PyTorch is a CPU-only build on Windows, reinstall the CUDA build so "
        "embeddings can run on an NVIDIA GPU.",
    ),
) -> None:
    """Reports local capabilities: embedding device, backend, CUDA GPU scan.

    With ``--fix``, reinstalls the CUDA PyTorch build when this is a CPU-only torch on
    Windows: PyPI serves CPU-only torch there and no package metadata can redirect it, so
    it is the one thing the report cannot resolve on its own.
    """

    report = local_capability_report(device=device)
    if json_out:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(format_capability_report(report))
    if fix:
        _fix_windows_torch(report.get("windows_gpu_hint"))


def _fix_windows_torch(hint: dict | None) -> None:
    """Reinstalls the CUDA PyTorch build when doctor flagged a CPU-only torch on Windows."""

    if not hint:
        typer.echo("\nNothing to fix: --fix only applies to a CPU-only PyTorch on Windows.")
        return

    import subprocess
    import sys

    args = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "torch",
        "--force-reinstall",
        "--no-deps",
        "--index-url",
        hint["index_url"],
    ]
    typer.echo("\nReinstalling the CUDA PyTorch build:\n  " + " ".join(args))
    result = subprocess.run(args)  # noqa: S603 - fixed args, user opted in via --fix
    if result.returncode != 0:
        typer.echo(
            "\nCould not run pip automatically (a uv-managed venv may not include pip). "
            "Reinstall with your package manager, e.g.:\n"
            f"  {hint['command']}\n"
            f"  uv pip install torch --reinstall --index-url {hint['index_url']}"
        )
        return
    typer.echo("\nDone. Re-run `lodedb doctor` to confirm the embedding device resolves to cuda.")


@app.command()
def index(
    texts: list[str] = _TEXTS_ARGUMENT,
    path: Path = _PATH_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    runtime: str = _RUNTIME_OPTION,
    file: Path | None = _FILE_OPTION,
    store_text: bool = _STORE_TEXT_OPTION,
    durability: str | None = _DURABILITY_OPTION,
    commit_mode: str | None = _COMMIT_MODE_OPTION,
) -> None:
    """Adds documents to the local index (positional texts or --file lines).

    Raw text is retained by default so it can be retrieved later with
    ``lodedb get ID``; pass ``--no-store-text`` to opt out (telemetry and the
    redacted snapshot stay payload-free either way).
    """

    docs: list[str] = list(texts or [])
    if file is not None:
        docs.extend(
            line.strip()
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not docs:
        raise typer.BadParameter("provide document texts or --file")
    db = LodeDB(
        path=path,
        model=model,
        device=device,
        embedding_runtime=runtime,
        store_text=store_text,
        durability=durability,
        commit_mode=commit_mode,
    )
    ids = db.add_many([{"text": text} for text in docs])
    db.persist()
    typer.echo(
        json.dumps(
            {
                "added": len(ids),
                "ids": ids,
                "count": db.count(),
                "device": db.embedding_resolution.to_dict(),
            },
            indent=2,
        )
    )
    db.close()


@app.command()
def query(
    text: str = typer.Argument(..., help="The query text."),
    path: Path = _PATH_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    runtime: str = _RUNTIME_OPTION,
    k: int = typer.Option(10, "--k", "-k", help="Number of results."),
) -> None:
    """Searches the local index and prints redacted (score, id, metadata) rows.

    Opens the store read-only, so it can query a path even while a writer (e.g.
    ``lodedb serve``) holds it.
    """

    try:
        db = LodeDB(
            path=path, model=model, device=device, embedding_runtime=runtime, read_only=True
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        hits = db.search(text, k=k)
        typer.echo(
            json.dumps(
                {
                    "device": db.embedding_resolution.to_dict(),
                    "results": [
                        {"score": h.score, "id": h.id, "metadata": h.metadata} for h in hits
                    ],
                },
                indent=2,
            )
        )
    finally:
        db.close()


@app.command()
def get(
    id: str = typer.Argument(..., help="The document id to retrieve."),
    path: Path = _PATH_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    runtime: str = _RUNTIME_OPTION,
) -> None:
    """Prints the stored raw text for one document id.

    Returns text for any document indexed with retention on (the default). Exits
    with an error if the id is unknown or was indexed with ``--no-store-text``.
    """

    try:
        db = LodeDB(
            path=path,
            model=model,
            device=device,
            embedding_runtime=runtime,
            store_text=True,
            read_only=True,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    try:
        text = db.get(id)
    finally:
        db.close()
    if text is None:
        raise typer.BadParameter("document not found")
    typer.echo(json.dumps({"id": id, "text": text}, indent=2))


@app.command()
def benchmark(
    path: Path = _BENCH_PATH_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    runtime: str = _RUNTIME_OPTION,
    docs: int = typer.Option(2000, "--docs", help="Number of synthetic documents to index."),
    queries: int = typer.Option(200, "--queries", help="Number of queries to time."),
    k: int = typer.Option(10, "--k", "-k", help="top-k per query."),
) -> None:
    """Times embedding throughput and CPU vector-scan latency locally (p50/p95).

    Honest scope: on Apple Silicon embedding is MPS-accelerated and the
    TurboVec vector scan runs on the CPU kernel. No GPU-vector-search claims.
    """

    import tempfile

    from lodedb.local.benchmark import run_local_benchmark

    workdir = path or Path(tempfile.mkdtemp(prefix="lodedb-bench-"))
    summary = run_local_benchmark(
        path=workdir,
        model=model,
        device=device,
        embedding_runtime=runtime,
        doc_count=docs,
        query_count=queries,
        top_k=k,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


@app.command()
def serve(
    path: Path = _PATH_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    runtime: str = _RUNTIME_OPTION,
    host: str = typer.Option("127.0.0.1", "--host", help="Loopback/private bind host only."),
    port: int = typer.Option(8088, "--port", help="Local port."),
    store_text: bool = _STORE_TEXT_OPTION,
    durability: str | None = _DURABILITY_OPTION,
    commit_mode: str | None = _COMMIT_MODE_OPTION,
) -> None:
    """Serves the local index over a minimal loopback HTTP API (no auth).

    Endpoints: ``POST /add {"text","id"?,"metadata"?}``,
    ``POST /search {"query","k"?,"filter"?}``, ``POST /get {"id"}`` (on by
    default; disabled by ``--no-store-text``), ``GET /stats``, ``GET /healthz``.
    Bound to loopback by default; a dev convenience.
    """

    from lodedb.local.server import serve_local

    typer.echo(
        f"LodeDB local server on http://{host}:{port} "
        f"(model={model}, device={resolve_local_device(device)}, path={path})"
    )
    serve_local(
        path=path,
        model=model,
        device=device,
        embedding_runtime=runtime,
        host=host,
        port=port,
        store_text=store_text,
        durability=durability,
        commit_mode=commit_mode,
    )


mcp_app = typer.Typer(
    invoke_without_command=True,
    no_args_is_help=False,
    help="Serve LodeDB over MCP (stdio), or `install`/`uninstall` it for a coding assistant.",
)
app.add_typer(mcp_app, name="mcp")


@mcp_app.callback(invoke_without_command=True)
def mcp(
    ctx: typer.Context,
    path: Path = _PATH_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    runtime: str = _RUNTIME_OPTION,
    store_text: bool = _STORE_TEXT_OPTION,
    exclude_text: bool = _EXCLUDE_TEXT_OPTION,
) -> None:
    """Serve LodeDB as local agent memory over the Model Context Protocol (stdio).

    ``lodedb_search`` runs hybrid (BM25 + vector) search and returns each hit's stored text
    by default, so an agent can rank and answer in one call; pass ``--exclude-text`` to return
    metrics only (this also withdraws the get-by-id tool) or ``--no-store-text`` to keep no
    text on disk at all (search then falls back to a vector scan).
    Register with a coding agent in one step with ``lodedb mcp install --client <client>``, or
    by hand, e.g.:
    {"mcpServers": {"lodedb": {"command": "lodedb", "args": ["mcp", "--path", "./data"]}}}
    """

    # `lodedb mcp` with no subcommand serves the stdio server (back-compat); a subcommand
    # (install/uninstall) handles its own work and these serve options are ignored.
    if ctx.invoked_subcommand is not None:
        return

    from lodedb.local.mcp_server import build_mcp_server

    server, _db = build_mcp_server(
        path,
        model=model,
        device=device,
        embedding_runtime=runtime,
        store_text=store_text,
        exclude_text=exclude_text,
    )
    server.run(transport="stdio")


_CLIENT_OPTION = typer.Option(
    ...,
    "--client",
    "-c",
    help="claude-code | claude-desktop | cursor | lm-studio | codex | all.",
)
_CONFIG_OPTION = typer.Option(
    None,
    "--config",
    help="Override the client's config file path (per-OS default is used otherwise).",
)
_PROJECT_OPTION = typer.Option(
    None,
    "--project",
    help="Cursor only: write the project-level ./.cursor/mcp.json under this directory "
    "(default: the global ~/.cursor/mcp.json).",
)
_PREFER_UV_OPTION = typer.Option(
    False,
    "--prefer-uv",
    help="Force the `uv run --project <root>` launch form even if a `lodedb` is on PATH "
    "(use when the PATH `lodedb` is a different environment than this checkout).",
)
_DRY_RUN_OPTION = typer.Option(
    False, "--dry-run", help="Print what would be written or run, without changing anything."
)


@mcp_app.command("install")
def mcp_install(
    client: str = _CLIENT_OPTION,
    path: Path = _PATH_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    store_text: bool = _STORE_TEXT_OPTION,
    exclude_text: bool = _EXCLUDE_TEXT_OPTION,
    config: Path | None = _CONFIG_OPTION,
    project: Path | None = _PROJECT_OPTION,
    prefer_uv: bool = _PREFER_UV_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
) -> None:
    """Register the LodeDB MCP server with a coding assistant in one step.

    Writes the correct ``command``/``args`` entry for ``--client`` to that host's config —
    even when ``lodedb`` is not on ``PATH`` (it falls back to the ``uv run --project`` form,
    then an absolute path to the entry point). The edit is idempotent (an existing ``lodedb``
    entry is updated, never duplicated) and leaves other servers untouched. For
    ``claude-code`` it runs ``claude mcp add``; the others edit the JSON/TOML config directly.
    Passes through the ``lodedb mcp`` options (``--path``, ``--model``, ``--device``,
    ``--exclude-text``, ``--no-store-text``). ``--dry-run`` prints without writing.
    """

    _run_mcp_install(
        action="install",
        client=client,
        path=path,
        model=model,
        device=device,
        store_text=store_text,
        exclude_text=exclude_text,
        config=config,
        project=project,
        prefer_uv=prefer_uv,
        dry_run=dry_run,
    )


@mcp_app.command("uninstall")
def mcp_uninstall(
    client: str = _CLIENT_OPTION,
    config: Path | None = _CONFIG_OPTION,
    project: Path | None = _PROJECT_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
) -> None:
    """Remove the LodeDB MCP server entry from a coding assistant's config.

    The inverse of ``install``: drops the ``lodedb`` server from ``--client``'s config
    (or runs ``claude mcp remove lodedb`` for ``claude-code``) and leaves every other
    server in place. ``--dry-run`` prints without changing anything.
    """

    _run_mcp_install(
        action="uninstall",
        client=client,
        config=config,
        project=project,
        dry_run=dry_run,
    )


def _run_mcp_install(
    *,
    action: str,
    client: str,
    config: Path | None,
    project: Path | None,
    dry_run: bool,
    path: Path = Path("./data"),
    model: str = "minilm",
    device: str = "auto",
    store_text: bool = True,
    exclude_text: bool = False,
    prefer_uv: bool = False,
) -> None:
    """Shared driver for ``mcp install``/``mcp uninstall``: resolves, applies, and prints."""

    from lodedb.local.mcp_install import (
        MCPInstallError,
        MCPOptions,
        expand_clients,
        install_client,
        resolve_server_invocation,
    )

    targets = expand_clients(client.strip().lower())
    if config is not None and len(targets) > 1:
        raise typer.BadParameter("--config cannot be combined with --client all")

    # Resolve the data path to an absolute path against the install-time working
    # directory: a coding assistant launches the MCP server with its *own* CWD, so a
    # relative `--path` in the written entry would point somewhere unintended (and
    # silently open an empty store). Resolving here makes the entry work wherever the
    # client starts it.
    options = MCPOptions(
        path=str(path.expanduser().resolve()),
        model=model,
        device=device,
        store_text=store_text,
        exclude_text=exclude_text,
    )
    # Resolve once so every client in `all` gets the same launch command, and so the
    # chosen form can be reported up front.
    invocation = resolve_server_invocation(prefer_uv=prefer_uv)
    if action == "install":
        typer.echo(f"Launch command: {invocation.command} (resolved via {invocation.how})")

    failures = 0
    for target in targets:
        try:
            result = install_client(
                target,
                action=action,
                options=options,
                invocation=invocation,
                config_path=config,
                project=project,
                dry_run=dry_run,
            )
        except MCPInstallError as exc:
            failures += 1
            typer.echo(f"[{target}] skipped: {exc}")
            continue
        _print_client_result(result)

    if failures:
        raise typer.Exit(code=1)


def _print_client_result(result) -> None:
    """Prints a one-client install/uninstall outcome (entry + destination)."""

    from lodedb.local.mcp_install import SERVER_NAME

    header = f"[{result.client}]"
    if result.method == "cli":
        typer.echo(f"{header} {result.note}")
        if result.cli_command is not None and (result.dry_run or result.action == "install"):
            typer.echo("  command: " + " ".join(result.cli_command))
        return

    typer.echo(f"{header} {result.note}")
    typer.echo(f"  file: {result.path}")
    if result.entry is not None:
        typer.echo("  entry: " + json.dumps({SERVER_NAME: result.entry}))


def main() -> None:
    """Runs the LodeDB Typer application."""

    app()


if __name__ == "__main__":
    main()
