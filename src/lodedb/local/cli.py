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
_TEXTS_ARGUMENT = typer.Argument(None, help="Document texts to add (or use --file).")
_FILE_OPTION = typer.Option(None, "--file", "-f", help="A text file: one document per line.")
_BENCH_PATH_OPTION = typer.Option(
    None, "--path", "-p", help="On-disk dir (default: ephemeral temp dir)."
)
_STORE_TEXT_OPTION = typer.Option(
    True,
    "--store-text/--no-store-text",
    "-t",
    help="Retain raw text so `lodedb get ID` can return it (default on; --no-store-text opts out).",
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
    file: Path | None = _FILE_OPTION,
    store_text: bool = _STORE_TEXT_OPTION,
    durability: str | None = _DURABILITY_OPTION,
) -> None:
    """Adds documents to the local index (positional texts or --file lines).

    Raw text is retained by default so it can be retrieved later with
    ``lodedb get ID``; pass ``--no-store-text`` to opt out (telemetry and the
    redacted snapshot stay payload-free either way).
    """

    docs: list[str] = list(texts or [])
    if file is not None:
        docs.extend(
            line.strip() for line in file.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    if not docs:
        raise typer.BadParameter("provide document texts or --file")
    db = LodeDB(path=path, model=model, device=device, store_text=store_text, durability=durability)
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
    k: int = typer.Option(10, "--k", "-k", help="Number of results."),
) -> None:
    """Searches the local index and prints redacted (score, id, metadata) rows.

    Opens the store read-only, so it can query a path even while a writer (e.g.
    ``lodedb serve``) holds it.
    """

    try:
        db = LodeDB(path=path, model=model, device=device, read_only=True)
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
) -> None:
    """Prints the stored raw text for one document id.

    Returns text for any document indexed with retention on (the default). Exits
    with an error if the id is unknown or was indexed with ``--no-store-text``.
    """

    try:
        db = LodeDB(path=path, model=model, device=device, store_text=True, read_only=True)
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
    host: str = typer.Option("127.0.0.1", "--host", help="Loopback/private bind host only."),
    port: int = typer.Option(8088, "--port", help="Local port."),
    store_text: bool = _STORE_TEXT_OPTION,
    durability: str | None = _DURABILITY_OPTION,
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
        host=host,
        port=port,
        store_text=store_text,
        durability=durability,
    )


mcp_app = typer.Typer(help="MCP server commands.")
app.add_typer(mcp_app, name="mcp", invoke_without_command=True)


@mcp_app.callback(invoke_without_command=True)
def mcp_default(
    ctx: typer.Context,
    path: Path = _PATH_OPTION,
    model: str = _MODEL_OPTION,
    device: str = _DEVICE_OPTION,
    store_text: bool = _STORE_TEXT_OPTION,
    exclude_text: bool = _EXCLUDE_TEXT_OPTION,
) -> None:
    """Serve LodeDB as local agent memory over the Model Context Protocol (stdio).

    ``lodedb_search`` runs hybrid (BM25 + vector) search and returns each hit's stored text
    by default, so an agent can rank and answer in one call; pass ``--exclude-text`` to return
    metrics only (this also withdraws the get-by-id tool) or ``--no-store-text`` to keep no
    text on disk at all (search then falls back to a vector scan).

    Register LodeDB with a coding agent by running: ``lodedb mcp install``
    """
    if ctx.invoked_subcommand is None:
        from lodedb.local.mcp_server import build_mcp_server

        server, _db = build_mcp_server(
            path, model=model, device=device, store_text=store_text, exclude_text=exclude_text
        )
        server.run(transport="stdio")


_CLIENT_OPTION = typer.Option(
    "all",
    "--client",
    help="claude-code | claude-desktop | codex | cursor | lm-studio | all",
)
_DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="Prints without writing.")
_CONFIG_OPTION = typer.Option(None, "--config", help="Override standard config file path.")


@mcp_app.command("install")
def mcp_install_cmd(
    ctx: typer.Context,
    client: str = _CLIENT_OPTION,
    dry_run: bool = _DRY_RUN_OPTION,
    config: Path | None = _CONFIG_OPTION,
) -> None:
    """Registers LodeDB's MCP server with a coding assistant in one step."""
    from lodedb.local.mcp_install import install

    parent_params = ctx.parent.params if ctx.parent else {}
    install(
        client=client,
        dry_run=dry_run,
        config_override=config,
        **parent_params,
    )


def main() -> None:
    """Runs the LodeDB Typer application."""

    app()


if __name__ == "__main__":
    main()
