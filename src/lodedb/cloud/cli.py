"""The `lodedb cloud` command line: the six client verbs over the native
transfer core (`lodedb._turbovec.cloud`),
plus the cloud verbs (login, tenancy, managed transfer).

Thin by design — argument parsing and report formatting only. Every operation
(and all of its validation) lives in the Rust core or `lodedb.cloud.transfer`, so
the CLI cannot offer a code path the library does not have.

Agent-native output contract:
- Results print as JSON when stdout is not a terminal (or with `--json`);
  the human render is the TTY default. Progress/context lines move to stderr
  in JSON mode so stdout stays parseable.
- Errors print to stderr as `error: …` (+ a `hint: …` line naming the exact
  next command where one exists), with an exit code per failure class:
  1 unexpected, 2 usage/local config, 3 auth (401/403), 4 not found (404),
  5 refused (402/409/422 — understood and denied), 6 transient (425/429/503
  — retry with backoff).
- Confirmation prompts never hang a pipe: without a TTY they fail
  immediately, naming `--yes`.

Transfer verbs accept three REMOTE spellings: a directory path or `s3://`
URL (the dumb targets, handled entirely by the Rust core), an explicit
`orecloud://org/environment[/store]` managed target, or the literal word
`cloud` — the managed remote recorded in LOCAL_DIR/orecloud.toml by
`lodedb cloud init`/`lodedb cloud link`.
"""

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Annotated, TypeVar

import typer

from lodedb._turbovec import cloud as _core

app = typer.Typer(
    name="cloud",
    help="Durable backup/restore for LodeDB indexes, to a directory or object storage.",
    no_args_is_help=True,
    add_completion=False,
)

T = TypeVar("T")

# Exit-code classes (see the module docstring).
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_AUTH = 3
EXIT_NOT_FOUND = 4
EXIT_REFUSED = 5
EXIT_RETRY = 6

# Tri-state output mode set by the app callback: True/False from
# --json/--no-json, None = auto (JSON exactly when stdout is not a TTY).
_json_output: bool | None = None


@app.callback()
def _main(
    json_output: Annotated[
        bool | None,
        typer.Option(
            "--json/--no-json",
            help="Force JSON (or human) output; default: JSON when stdout is not a terminal.",
        ),
    ] = None,
) -> None:
    # Records the output-mode override for this invocation (global option:
    # `lodedb cloud --json store list`).
    global _json_output
    _json_output = json_output


def _machine_output() -> bool:
    """True when results should print as JSON (forced, or piped stdout)."""
    if _json_output is not None:
        return _json_output
    return not sys.stdout.isatty()


def _emit(data: object, human: Callable[[], None]) -> None:
    """Prints one result twice over: `data` as JSON for programs, or the
    `human` renderer for a terminal."""
    if _machine_output():
        typer.echo(json.dumps(data, indent=2, default=str))
    else:
        human()


def _emit_report(report: Mapping[str, object]) -> None:
    """A report dict as JSON or aligned `field  value` lines."""
    _emit(dict(report), lambda: _echo_report(report))


def _note(message: str) -> None:
    """Progress/context lines: stdout for people, stderr in JSON mode so
    stdout stays pure JSON."""
    typer.echo(message, err=_machine_output())

# Shared option declarations for the payload opt-ins (redacted-by-default
# posture: raw text and lexical terms leave the machine only on explicit ask).
IncludeText = Annotated[
    bool,
    typer.Option("--include-text", help="Also ship the raw-text store (payload-bearing)."),
]
IncludeLexical = Annotated[
    bool,
    typer.Option(
        "--include-lexical",
        help="Also ship the lexical-index store (tokenised text; payload-bearing).",
    ),
]


def _run(operation: Callable[[], T]) -> T:
    """Runs one binding operation, mapping its exceptions onto stderr + exit 1."""
    try:
        return operation()
    except (FileNotFoundError, OSError, RuntimeError) as error:
        typer.secho(f"error: {error}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error


def _echo_report(report: Mapping[str, object]) -> None:
    """Prints a report dict as aligned `field  value` lines."""
    width = max(len(field) for field in report)
    for field, value in report.items():
        typer.echo(f"{field:<{width}}  {value}")


@app.command()
def keys(local_dir: str) -> None:
    """List the index keys with a committed generation in LOCAL_DIR."""
    found = list(_run(lambda: _core.keys(local_dir)))

    def human() -> None:
        for key in found:
            typer.echo(key)

    _emit(found, human)


@app.command()
def status(
    local_dir: str,
    remote: str,
    key: str,
    include_text: IncludeText = False,
    include_lexical: IncludeLexical = False,
) -> None:
    """Compare LOCAL_DIR against REMOTE for a push of KEY (read-only on both ends)."""
    managed = _managed(local_dir, remote)
    if managed is not None:
        client, target, host = managed
        with client:
            report = _cloud(
                lambda: managed_status(
                    client,
                    local_dir,
                    target,
                    key,
                    host=host,
                    include_text=include_text,
                    include_lexical=include_lexical,
                )
            )
        _emit_report(report)
        return
    report = _run(
        lambda: _core.status(
            local_dir, remote, key, include_text=include_text, include_lexical=include_lexical
        )
    )
    _emit_report(report)


@app.command()
def push(
    local_dir: str,
    remote: str,
    key: str,
    include_text: IncludeText = False,
    include_lexical: IncludeLexical = False,
) -> None:
    """Push KEY's committed generation from LOCAL_DIR to REMOTE (redacted by default)."""
    managed = _managed(local_dir, remote)
    if managed is not None:
        client, target, host = managed
        with client:
            result = _cloud(
                lambda: managed_push(
                    client,
                    local_dir,
                    target,
                    key,
                    host=host,
                    include_text=include_text,
                    include_lexical=include_lexical,
                )
            )
        _emit_report(result)
        return
    result = _run(
        lambda: _core.push(
            local_dir, remote, key, include_text=include_text, include_lexical=include_lexical
        )
    )
    _emit_report(result)


@app.command()
def sync(
    local_dir: str,
    remote: str,
    key: str,
    include_text: IncludeText = False,
    include_lexical: IncludeLexical = False,
    force_push: Annotated[
        bool,
        typer.Option(
            "--force-push", help="Keep the local copy: push it even over a diverged remote."
        ),
    ] = False,
    force_pull: Annotated[
        bool,
        typer.Option(
            "--force-pull", help="Keep the remote copy: pull it even over a diverged local."
        ),
    ] = False,
) -> None:
    """Sync KEY between LOCAL_DIR and REMOTE: fast-forward in whichever direction moved.

    Refuses (with a nonzero exit) when the two ends diverged past the last
    recorded sync, or when there is no sync record and the ends differ —
    resolve with exactly one of --force-push / --force-pull.
    """
    if force_push and force_pull:
        typer.secho(
            "error: --force-push and --force-pull are mutually exclusive",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=EXIT_USAGE)
    managed = _managed(local_dir, remote)
    if managed is not None:
        client, target, host = managed
        with client:
            outcome = _cloud(
                lambda: managed_sync(
                    client,
                    local_dir,
                    target,
                    key,
                    host=host,
                    include_text=include_text,
                    include_lexical=include_lexical,
                    force_push=force_push,
                    force_pull=force_pull,
                )
            )
    else:
        outcome = _run(
            lambda: _core.sync(
                local_dir,
                remote,
                key,
                include_text=include_text,
                include_lexical=include_lexical,
                force_push=force_push,
                force_pull=force_pull,
            )
        )
    if outcome.get("sidecar_corrupt"):
        typer.secho(
            "warning: the sync sidecar was present but corrupt and has been ignored",
            fg=typer.colors.YELLOW,
            err=True,
        )
    _emit_report(outcome)


@app.command()
def pull(remote: str, local_dir: str, key: str) -> None:
    """Restore KEY's committed generation from REMOTE into LOCAL_DIR and verify it opens."""
    managed = _managed(local_dir, remote)
    if managed is not None:
        client, target, host = managed
        with client:
            outcome = _cloud(
                lambda: managed_pull(client, target, local_dir, key, host=host)
            )
        _emit_report(outcome)
        return
    outcome = _run(lambda: _core.pull(remote, local_dir, key))
    _emit_report(outcome)


@app.command()
def verify(target: str, key: str) -> None:
    """Re-hash every artifact KEY's committed generation pins in TARGET."""
    report = _run(lambda: _core.verify(target, key))
    _emit_report(report)


# ---------------------------------------------------------------------------
# Cloud verbs: the managed control plane (login, identity, tokens, tenancy).
# These speak HTTP via lodedb.cloud.transfer; everything above stays pure-local.
# ---------------------------------------------------------------------------

import platform  # noqa: E402
import webbrowser  # noqa: E402
from datetime import UTC  # noqa: E402

from lodedb.cloud import _config  # noqa: E402
from lodedb.cloud.transfer import (  # noqa: E402
    CloudClient,
    CloudError,
    LoginHandoff,
    ManagedRemote,
    managed_pull,
    managed_push,
    managed_status,
    managed_sync,
)

auth_app = typer.Typer(help="Credential helpers for external tools.", no_args_is_help=True)
tokens_app = typer.Typer(help="Mint, list, and revoke API tokens.", no_args_is_help=True)
environments_app = typer.Typer(
    help="List the fixed environments (production and testing).", no_args_is_help=True
)
store_app = typer.Typer(help="Register and list managed stores.", no_args_is_help=True)
mcp_app = typer.Typer(
    help="Connect AI agents to a hosted store over MCP.", no_args_is_help=True
)
org_app = typer.Typer(
    help="Org lifecycle: delete, restore, trash, and export (offboarding).",
    no_args_is_help=True,
)
app.add_typer(auth_app, name="auth")
app.add_typer(tokens_app, name="tokens")
app.add_typer(environments_app, name="environments")
app.add_typer(store_app, name="store")
app.add_typer(mcp_app, name="mcp")
app.add_typer(org_app, name="org")


def _fail(message: str, code: int = EXIT_UNEXPECTED, hint: str | None = None) -> typer.Exit:
    """Prints `error: message` (+ optional `hint: …` naming the next command)
    to stderr and returns the Exit to raise."""
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    if hint:
        typer.secho(f"hint: {hint}", err=True)
    return typer.Exit(code=code)


def _classify(error: CloudError) -> tuple[int, str | None]:
    """Maps a control-plane refusal to (exit code, fix hint). The hint names
    the exact next command — errors double as prompts for agents."""
    status = error.status_code
    if status == 401:
        return EXIT_AUTH, (
            "the credential was rejected — run `lodedb cloud login --host <control-plane-url>`, "
            "or set both ORECLOUD_HOST and ORECLOUD_TOKEN"
        )
    if status == 403:
        return EXIT_AUTH, (
            "this credential lacks a needed scope — mint one with "
            "`lodedb cloud tokens mint --scope <scope>` (scopes: admin, write, "
            "read:search, read:text)"
        )
    if status == 404:
        return EXIT_NOT_FOUND, (
            "check the target spelling with `lodedb cloud environments list` / "
            "`lodedb cloud store list`"
        )
    if status == 402:
        return EXIT_REFUSED, (
            "a plan limit was reached — reads keep serving; upgrade or free the "
            "resource (see `lodedb cloud org export` and the console's Usage page)"
        )
    if status in (409, 422):
        return EXIT_REFUSED, None
    if status in (425, 429):
        return EXIT_RETRY, "transient — retry after a moment"
    if status == 503:
        return EXIT_RETRY, (
            "retry with backoff; if the message names a missing runtime, that is "
            "a deployment fix, not a retry"
        )
    return EXIT_UNEXPECTED, None


def _confirm(prompt: str) -> None:
    """`typer.confirm` with an agent guard: when stdin is not a TTY there is
    nobody to answer — fail immediately, naming `--yes`, instead of hanging
    the calling program."""
    if not sys.stdin.isatty():
        raise _fail(
            "confirmation needed but stdin is not interactive — re-run with --yes",
            code=EXIT_USAGE,
        )
    typer.confirm(prompt, abort=True)


def _load_credentials() -> "_config.Credentials | None":
    try:
        return _config.load_credentials()
    except _config.CredentialsError as error:
        raise _fail(str(error), code=EXIT_USAGE) from error


def _client() -> CloudClient:
    creds = _load_credentials()
    if creds is None:
        raise _fail("not logged in — run `lodedb cloud login --host <control-plane-url>`")
    return CloudClient(creds.host, creds.token)


def _cloud(operation: Callable[[], T]) -> T:
    # The same exception surface as _run(), plus CloudError (itself a
    # RuntimeError): managed operations interleave HTTP calls with local
    # Rust-core work (planning, materialisation), and both halves must
    # print as `error: …` rather than a traceback. Control-plane refusals
    # additionally classify into per-class exit codes with fix hints.
    try:
        return operation()
    except CloudError as error:
        code, hint = _classify(error)
        raise _fail(str(error), code=code, hint=hint) from error
    except (FileNotFoundError, OSError, RuntimeError) as error:
        raise _fail(str(error)) from error


# The tenancy flags share one story: the credential knows where it belongs.
_ORG_HELP = "Org slug; defaults to the token's binding, else your only org."
_ENVIRONMENT_HELP = (
    "Environment slug; defaults to the token's binding, else the org's only environment."
)
_TENANCY_HINT = (
    "name the target with --org/--environment (see `lodedb cloud environments list`)"
)


def _org_scope(client: CloudClient, org: str | None) -> str:
    """The org a command addresses: the explicit --org, else the token's
    binding (environment tokens), else the account's only org — failing
    with the actual choices when several orgs qualify."""
    if org:
        return org
    info = _cloud(client.token_self)
    if info["org"]:
        return info["org"]
    slugs = [row["slug"] for row in _cloud(client.me)["orgs"]]
    if not slugs:
        raise _fail("your account belongs to no org")
    if len(slugs) > 1:
        raise _fail(
            f"this account belongs to several orgs — pass --org (one of: {', '.join(slugs)})",
            code=EXIT_USAGE,
        )
    return slugs[0]


def _tenancy(
    client: CloudClient, org: str | None, environment: str | None
) -> tuple[str, str]:
    """The (org, environment) pair a command addresses — the SDK's
    resolution (explicit flags win; environment tokens supply their binding
    and refuse contradictions; personal tokens fall back to the only
    org/environment), re-raised with the CLI flags in the hint."""
    from lodedb.cloud.client import resolve_tenancy

    try:
        return resolve_tenancy(client, org, environment)
    except CloudError as error:
        code, _hint = _classify(error)
        raise _fail(str(error), code=code, hint=_TENANCY_HINT) from error


def _managed(local_dir: str, remote: str) -> tuple[CloudClient, ManagedRemote, str] | None:
    """Resolves REMOTE as a managed target: an explicit `orecloud://` URL or
    the literal `cloud` (the remote recorded in LOCAL_DIR/orecloud.toml).
    Returns None for dumb targets (paths, `s3://`)."""
    if remote == "cloud":
        try:
            config = _config.load_remote(local_dir)
        except _config.CredentialsError as error:
            raise _fail(str(error), code=EXIT_USAGE) from error
        if config is None:
            raise _fail(
                f"no {_config.REMOTE_FILE_NAME} in {local_dir!r} — run "
                "`lodedb cloud link` (or `lodedb cloud init`) first, or name the remote "
                "explicitly as orecloud://org/environment"
            )
        creds = _load_credentials()
        if creds is None:
            raise _fail("not logged in — run `lodedb cloud login --host <control-plane-url>`")
        if creds.host.rstrip("/") != config.host:
            raise _fail(
                f"this directory is linked to {config.host} but you are logged in to "
                f"{creds.host} — log in to the linked control plane or re-link"
            )
        target = ManagedRemote(config.org, config.environment, config.store)
        return CloudClient(creds.host, creds.token), target, creds.host
    target = _cloud(lambda: ManagedRemote.parse(remote))
    if target is None:
        return None
    creds = _load_credentials()
    if creds is None:
        raise _fail("not logged in — run `lodedb cloud login --host <control-plane-url>`")
    return CloudClient(creds.host, creds.token), target, creds.host


@app.command()
def login(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Control-plane URL, e.g. https://console.example.com."),
    ] = None,
    token: Annotated[
        str | None,
        typer.Option(
            "--token",
            help="Skip the browser handoff and store this token directly (CI).",
        ),
    ] = None,
    no_browser: Annotated[
        bool,
        typer.Option(
            "--no-browser",
            help="Don't open a browser; print the approval URL and code instead.",
        ),
    ] = False,
) -> None:
    """Log in to an OreCloud control plane and store the credential locally.

    Default flow: this machine generates a keypair, your browser approves the
    login, and the server returns the new token sealed to that keypair — the
    secret never crosses in the clear and is never stored server-side.
    """
    creds = _load_credentials()
    if host is None:
        host = creds.host if creds else None
    if host is None:
        raise _fail("--host is required for the first login")

    if token is None:
        token = _browser_login(host, open_browser=not no_browser)

    # Prove the credential works before persisting it — a typo'd --token must
    # not leave broken state behind.
    with CloudClient(host, token) as client:
        me = _cloud(client.me)
    path = _config.save_credentials(host, token)
    _emit(
        {"host": host, "email": me["user"]["email"], "credentials": str(path)},
        lambda: typer.echo(
            f"Logged in to {host} as {me['user']['email']} (credentials in {path})."
        ),
    )


def _browser_login(host: str, open_browser: bool = True) -> str:
    """The sealed-box browser handoff against `host`: prints the confirmation
    code and approval URL to stderr, blocks until a signed-in browser approves,
    and returns the minted token secret (sealed in transit to this process's
    keypair — it never crosses in the clear)."""
    with CloudClient(host) as anonymous:
        handoff = _cloud(
            lambda: LoginHandoff(
                anonymous, client_label=f"lodedb cloud CLI on {platform.node() or 'unknown host'}"
            )
        )
        _note(f"Confirmation code: {handoff.start.user_code}")
        _note(f"Approve this login at: {handoff.start.verification_url}")
        if open_browser:
            webbrowser.open(handoff.start.verification_url)
        _note("Waiting for approval…")
        return _cloud(handoff.wait)


def _ensure_logged_in(host: str | None) -> "_config.Credentials":
    """Stored credentials, or an inline sealed-box login when there are none —
    the single human touchpoint of `lodedb cloud init`. A `host` that contradicts
    the stored credential fails closed rather than silently re-aiming the
    command at another control plane."""
    creds = _load_credentials()
    if creds is not None:
        if host is not None and host.rstrip("/") != creds.host:
            raise _fail(
                f"already logged in to {creds.host} — run `lodedb cloud login --host {host}` "
                "to switch control planes first",
                code=EXIT_USAGE,
            )
        return creds
    if host is None:
        raise _fail(
            "not logged in — pass --host to log in as part of this command, "
            "or run `lodedb cloud login --host <control-plane-url>` first",
            code=EXIT_USAGE,
        )
    token = _browser_login(host)
    with CloudClient(host, token) as client:
        me = _cloud(client.me)
    _config.save_credentials(host, token)
    _note(f"logged in to {host} as {me['user']['email']}")
    return _config.Credentials(host=host.rstrip("/"), token=token, source="file")


@app.command()
def link(
    local_dir: str,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    store: Annotated[str, typer.Option(help="Store name.")] = "memory",
) -> None:
    """Record LOCAL_DIR's managed remote in orecloud.toml (committable, no
    secrets). Transfer verbs then accept the literal remote `cloud`."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
    creds = _load_credentials()
    assert creds is not None  # _client() above required it
    path = _config.save_remote(
        local_dir,
        _config.RemoteConfig(host=creds.host, org=org, environment=environment, store=store),
    )
    _emit(
        {"org": org, "environment": environment, "store": store, "path": str(path)},
        lambda: typer.echo(
            f"linked {local_dir} -> orecloud://{org}/{environment}/{store} ({path})"
        ),
    )


def _agents_repo_token(
    client: CloudClient,
    *,
    host: str,
    org: str,
    environment: str,
    local_dir: str,
    env_file: str | Path,
) -> tuple[dict[str, object], list[str]]:
    """The repo's deterministic `init-<dirname>` scoped key, written into the
    dotenv file. Reused when the file still holds a live token of that name;
    otherwise every stale same-name token is revoked and one fresh key is
    minted — re-running init never accumulates keys. Returns (report data,
    notes)."""
    from datetime import datetime

    from lodedb.cloud import _env_file as env

    name = f"init-{Path(local_dir).resolve().name}"[:200]

    def alive(row: Mapping[str, object]) -> bool:
        if row["revoked_at"] is not None:
            return False
        expires_at = row["expires_at"]
        if expires_at is None:
            return True
        return datetime.fromisoformat(str(expires_at)) > datetime.now(UTC)

    live = [
        row
        for row in _cloud(client.list_tokens)
        if row["name"] == name and row["org"] == org and row["environment"] == environment
        and alive(row)
    ]
    existing_secret = env.read_env_value(env_file, "ORECLOUD_TOKEN")
    if existing_secret:
        held = [row for row in live if existing_secret.startswith(str(row["prefix"]))]
        if held:
            # Reuse the key the file already holds; only heal the host line
            # (the token line stays untouched — we could not rewrite it anyway,
            # secrets are shown once).
            env.write_env_values(env_file, {"ORECLOUD_HOST": host})
            return (
                {"token": held[0], "env_file": str(env_file)},
                [f"reusing token {name} already in {env_file}"],
            )
    notes: list[str] = []
    for row in live:
        _cloud(lambda row=row: client.revoke_token(str(row["id"])))
        notes.append(f"revoked stale token {row['prefix']}… ({name})")
    # Same scope set `store create` mints: text access stays double-gated by
    # the store's own expose_text switch, so read:text here grants nothing on
    # stores that keep text closed.
    minted = _cloud(
        lambda: client.mint_token(
            "secret",
            ["write", "read:search", "read:text"],
            name=name,
            org=org,
            environment=environment,
        )
    )
    data: dict[str, object] = {"token": minted["token"]}
    data.update(_write_credential_env(env_file, host, minted["secret"]))
    return data, notes


@app.command()
def init(
    local_dir: str,
    environment: Annotated[
        str | None, typer.Option(help=_ENVIRONMENT_HELP)
    ] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    store: Annotated[str, typer.Option(help="Store name.")] = "memory",
    host: Annotated[
        str | None,
        typer.Option(
            "--host",
            help="Control-plane URL — when no credential is stored, init logs "
            "in first (browser approval, the one human step).",
        ),
    ] = None,
    agents: Annotated[
        bool,
        typer.Option(
            "--agents",
            help="Also mint a scoped key into --env-file and drop agent "
            "artifacts generated from this link: .claude/skills/orecloud/"
            "SKILL.md (regenerated each run), an ## OreCloud section in "
            "AGENTS.md (first run only), and an orecloud-control entry in "
            ".mcp.json (${ORECLOUD_TOKEN} auth).",
        ),
    ] = False,
    env_file: Annotated[
        str | None,
        typer.Option(
            "--env-file",
            help="Where --agents writes the minted credential "
            "(default: LOCAL_DIR/.env; the secret is never printed).",
        ),
    ] = None,
) -> None:
    """Link LOCAL_DIR to one of the org's environments — the one-command path
    from a local data directory to a managed remote. With --agents this is
    the whole setup for a coding agent: log in if needed (one browser
    approval), mint a scoped key into .env, and scaffold the repo's agent
    artifacts. Environments are fixed (production and testing, seeded at
    signup); init resolves the slug from the credential — or validates an
    explicit one — rather than creating anything."""
    creds = _ensure_logged_in(host)
    data: dict[str, object] = {}
    notes: list[str] = []
    with CloudClient(creds.host, creds.token) as client:
        if environment is None:
            org, environment = _tenancy(client, org, environment)
        else:
            org = _org_scope(client, org)
            available = sorted(
                p["slug"] for p in _cloud(lambda: client.list_environments(org))
            )
            if environment not in available:
                raise _fail(
                    f"no environment {environment!r} in org {org!r}",
                    code=EXIT_NOT_FOUND,
                    hint=f"pick one of: {', '.join(available)} (via --environment)",
                )
        if agents:
            token_data, notes = _run(
                lambda: _agents_repo_token(
                    client,
                    host=creds.host,
                    org=org,
                    environment=environment,
                    local_dir=local_dir,
                    env_file=env_file or Path(local_dir) / ".env",
                )
            )
            data.update(token_data)
    path = _config.save_remote(
        local_dir,
        _config.RemoteConfig(host=creds.host, org=org, environment=environment, store=store),
    )
    data.update(org=org, environment=environment, store=store, path=str(path))
    written: list[str] = []
    if agents:
        from lodedb.cloud._agents_scaffold import scaffold_agent_artifacts

        written, scaffold_notes = scaffold_agent_artifacts(
            local_dir, host=creds.host, org=org, environment=environment, store=store
        )
        notes.extend(scaffold_notes)
        data["agents"] = {"written": written, "notes": notes}

    def human() -> None:
        typer.echo(f"linked {local_dir} -> orecloud://{org}/{environment}/{store} ({path})")
        if "env_file" in data:
            typer.echo(f"wrote credential to {data['env_file']}")
        for name in written:
            typer.echo(f"wrote {name}")
        for note in notes:
            typer.echo(f"note: {note}")

    _emit(data, human)


@auth_app.command("print-headers")
def auth_print_headers() -> None:
    """Print the stored credential as an MCP headers object:
    {"Authorization": "Bearer …"}.

    The headersHelper contract (Claude Code and friends run a command to
    obtain MCP auth headers), so the browser-approved login feeds MCP
    servers without anyone pasting a key into a config file. Always JSON —
    that IS the output contract."""
    creds = _load_credentials()
    if creds is None:
        raise _fail(
            "not logged in — run `lodedb cloud login --host <control-plane-url>`",
            code=EXIT_AUTH,
        )
    typer.echo(json.dumps({"Authorization": f"Bearer {creds.token}"}))


@app.command()
def logout() -> None:
    """Forget the locally stored credential (the token itself stays valid —
    revoke it with `lodedb cloud tokens revoke` to kill it server-side)."""
    deleted = _config.delete_credentials()
    _emit(
        {"logged_out": deleted},
        lambda: typer.echo("Logged out." if deleted else "No stored credentials."),
    )


@app.command()
def whoami() -> None:
    """Show who (a personal token: the account) or what (an environment
    token: its kind, scopes, and binding) the stored credential is."""
    with _client() as client:
        try:
            info = client.token_self()
        except CloudError as error:
            # A control plane predating /v1/tokens/self answers 404; the
            # personal-token rendering below still works there via /v1/auth/me.
            if error.status_code != 404:
                code, hint = _classify(error)
                raise _fail(str(error), code=code, hint=hint) from error
            info = None
        me = _cloud(client.me) if info is None or info["kind"] == "personal" else None

    if me is not None:
        data = {
            "kind": info["kind"] if info else "personal",
            "email": me["user"]["email"],
            "auth": me["auth"],
            "orgs": [{"slug": org["slug"], "role": org["role"]} for org in me["orgs"]],
        }

        def human() -> None:
            typer.echo(f"{me['user']['email']} ({me['auth']})")
            for org in me["orgs"]:
                typer.echo(f"  org {org['slug']} ({org['role']})")
    else:
        data = {
            "kind": info["kind"],
            "org": info["org"],
            "environment": info["environment"],
            "scopes": info["scopes"],
            "name": info["name"],
        }

        def human() -> None:
            typer.echo(
                f"{info['kind']} token {info['prefix']}… bound to "
                f"{info['org']}/{info['environment']} (scopes: {', '.join(info['scopes'])})"
            )

    _emit(data, human)


@tokens_app.command("list")
def tokens_list() -> None:
    """List your tokens (personal, plus your orgs' environment tokens)."""
    with _client() as client:
        rows = _cloud(client.list_tokens)

    def human() -> None:
        if not rows:
            typer.echo("No tokens.")
            return
        for row in rows:
            state = "revoked" if row["revoked_at"] else "active"
            typer.echo(
                f"{row['id']}  {row['prefix']}…  {row['kind']:<11}  {state:<7}  "
                f"scopes={','.join(row['scopes'])}  {row['name']}"
            )

    _emit(rows, human)


def _write_credential_env(env_file: str | Path, host: str, secret: str) -> dict[str, object]:
    """Write ORECLOUD_HOST + ORECLOUD_TOKEN into the dotenv file (the secret
    never touches stdout — printed output becomes part of an agent's
    transcript) and make sure the file is git-ignored. Returns the env_file
    path and any gitignore note, for merging into the command's output."""
    from lodedb.cloud import _env_file as env

    path = env.write_env_values(env_file, {"ORECLOUD_HOST": host, "ORECLOUD_TOKEN": secret})
    data: dict[str, object] = {"env_file": str(path)}
    note = env.ensure_gitignored(path)
    if note:
        data["env_note"] = note
        _note(note)
    return data


@tokens_app.command("mint")
def tokens_mint(
    kind: Annotated[str, typer.Option(help="personal | secret | publishable.")] = "personal",
    scope: Annotated[
        list[str], typer.Option("--scope", help="Repeatable: admin, write, read:search, read:text.")
    ] = ["admin"],  # noqa: B006 — typer requires a literal default
    name: Annotated[str, typer.Option(help="Label shown in listings.")] = "",
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    expires_in_days: Annotated[int | None, typer.Option(min=1)] = None,
    env_file: Annotated[
        str | None,
        typer.Option(
            "--env-file",
            help="Write the secret into this dotenv file (ORECLOUD_HOST + "
            "ORECLOUD_TOKEN) instead of printing it — output then carries "
            "only the token metadata.",
        ),
    ] = None,
) -> None:
    """Mint a token. The secret is printed once and never retrievable again —
    or, with --env-file, written straight to the file and never printed.
    Environment tokens (secret/publishable) default to the resolved
    org/environment; personal tokens take neither flag."""
    creds = _load_credentials()
    if creds is None:
        raise _fail("not logged in — run `lodedb cloud login --host <control-plane-url>`")
    with CloudClient(creds.host, creds.token) as client:
        if kind != "personal" and (org is None or environment is None):
            org, environment = _tenancy(client, org, environment)
        minted = _cloud(
            lambda: client.mint_token(
                kind,
                scope,
                name=name,
                org=org,
                environment=environment,
                expires_in_days=expires_in_days,
            )
        )
    if env_file is not None:
        data: dict[str, object] = {"token": minted["token"]}
        data.update(_run(lambda: _write_credential_env(env_file, creds.host, minted["secret"])))
        _emit(data, lambda: typer.echo(f"wrote token to {data['env_file']}"))
    else:
        _emit(minted, lambda: typer.echo(minted["secret"]))
    typer.secho(
        f"(token {minted['token']['id']}, prefix {minted['token']['prefix']}… — "
        f"{'in ' + str(env_file) if env_file else 'store it now'}; "
        "it cannot be shown again)",
        err=True,
    )


@tokens_app.command("revoke")
def tokens_revoke(token_id: str) -> None:
    """Revoke a token by id (see `lodedb cloud tokens list`)."""
    with _client() as client:
        row = _cloud(lambda: client.revoke_token(token_id))
    _emit(row, lambda: typer.echo(f"revoked {row['prefix']}… at {row['revoked_at']}"))


@mcp_app.command("install")
def mcp_install(
    store: Annotated[str, typer.Argument(help="Store name — the end user's id.")],
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    token: Annotated[
        str | None,
        typer.Option(
            help="Embed this token in the printed configs (otherwise a "
            "placeholder is printed — mint one with `lodedb cloud tokens mint`)."
        ),
    ] = None,
) -> None:
    """Print ready-to-paste MCP configs (Claude Code, Cursor, VS Code) for
    one end user's store. The endpoint speaks Streamable HTTP with Bearer
    auth; the token's scopes decide which tools the agent sees. The store
    needs no create step — it provisions on the agent's first write."""
    import json as json_module
    from urllib.parse import quote

    with _client() as client:
        org, environment = _tenancy(client, org, environment)
    creds = _load_credentials()
    assert creds is not None  # _client() above required it
    # One URL per end user's store — always the 3-segment form. Slugs are
    # [a-z0-9-] but store names (end-user ids) are free-form: quote them.
    url = (
        f"{creds.host}/mcp/{quote(org, safe='')}/"
        f"{quote(environment, safe='')}/{quote(store, safe='')}"
    )
    bearer = token or "<your-token>"

    claude_command = (
        f'claude mcp add --transport http orecloud "{url}" '
        f'--header "Authorization: Bearer {bearer}"'
    )
    cursor_config = {
        "mcpServers": {"orecloud": {"url": url, "headers": {"Authorization": f"Bearer {bearer}"}}}
    }
    vscode_config = {
        "servers": {
            "orecloud": {
                "type": "http",
                "url": url,
                "headers": {"Authorization": "Bearer ${input:orecloud-token}"},
            }
        },
        "inputs": [
            {
                "id": "orecloud-token",
                "type": "promptString",
                "password": True,
                "description": "OreCloud API token",
            }
        ],
    }

    def human() -> None:
        typer.secho("MCP endpoint:", bold=True)
        typer.echo(f"  {url}\n")
        typer.secho("Claude Code:", bold=True)
        typer.echo(f"  {claude_command}\n")
        typer.secho("Cursor (~/.cursor/mcp.json):", bold=True)
        typer.echo(json_module.dumps(cursor_config, indent=2) + "\n")
        typer.secho("VS Code (.vscode/mcp.json):", bold=True)
        typer.echo(json_module.dumps(vscode_config, indent=2))

    _emit(
        {
            "url": url,
            "claude_code": claude_command,
            "cursor": cursor_config,
            "vscode": vscode_config,
        },
        human,
    )
    if token is None:
        typer.secho(
            "\n(mint a key first, e.g. `lodedb cloud tokens mint --kind secret "
            f"--scope read:search --scope read:text --org {org} --environment {environment}`)",
            err=True,
        )


@environments_app.command("list")
def environments_list(
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
) -> None:
    """List environments in an org."""
    with _client() as client:
        org = _org_scope(client, org)
        rows = _cloud(lambda: client.list_environments(org))

    def human() -> None:
        for row in rows:
            typer.echo(f"{org}/{row['slug']}  {row['name']}")

    _emit(rows, human)


@store_app.command("list")
def store_list(
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    after: Annotated[
        str | None, typer.Option(help="Keyset cursor: the last store name of the previous page.")
    ] = None,
    limit: Annotated[int, typer.Option(help="Page size (1-200).")] = 50,
) -> None:
    """One page of an environment's stores (a store is one end user)."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        out = _cloud(lambda: client.list_stores(org, environment, after=after, limit=limit))

    def human() -> None:
        rows = out["stores"]
        if not rows:
            typer.echo("No stores registered.")
            return
        for row in rows:
            text_flag = "text" if row["expose_text"] else "no-text"
            typer.echo(
                f"{row['store']}/{row['key']}  mode={row['mode']}  {text_flag}  "
                f"last write {row['last_write_at']}"
            )
        typer.echo(f"({out['count']} stores in the environment)")

    _emit(out, human)


@store_app.command("create")
def store_create(
    store: Annotated[str, typer.Argument(help="Store name.")] = "default",
    key: Annotated[
        str | None,
        typer.Option("--key", help="Index key; omit for the LodeDB default (the usual case)."),
    ] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    mode: Annotated[
        str,
        typer.Option(
            help="cloud_writer (write via the API — the quickstart path) | "
            "local_push (a LodeDB directory you push)."
        ),
    ] = "cloud_writer",
    preset: Annotated[
        str | None,
        typer.Option(
            help="Embedding preset (minilm | bge | clip); cloud_writer "
            "defaults to minilm unless --vector-dim is given."
        ),
    ] = None,
    vector_dim: Annotated[
        int | None,
        typer.Option(
            "--vector-dim",
            min=1,
            max=4096,
            help="Bring-your-own-vectors store at this dimensionality: you "
            "supply vectors (add_vectors/search_by_vector); the server never "
            "embeds. Mutually exclusive with --preset.",
        ),
    ] = None,
    expose_text: Annotated[
        bool, typer.Option("--expose-text", help="Allow read:text-scoped reads to return text.")
    ] = False,
    connect_key: Annotated[
        bool,
        typer.Option(
            "--connect-key/--no-connect-key",
            help="Mint a secret key and print a ready-to-run Python snippet.",
        ),
    ] = True,
) -> None:
    """Register a store and print everything Python needs to use it.

    The default is the API-first path: a cloud_writer store with the minilm
    preset plus a freshly minted secret key, so the printed snippet runs
    with zero edits. Pushing an existing LodeDB directory instead? Use
    `--mode local_push` (identity then comes from the pushed store).
    """
    if preset is not None and vector_dim is not None:
        raise _fail(
            "pass --preset or --vector-dim, not both (a store embeds "
            "server-side OR takes your vectors)",
            code=EXIT_USAGE,
        )
    if mode == "cloud_writer" and preset is None and vector_dim is None:
        preset = "minilm"
    minted = None
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        row = _cloud(
            lambda: client.create_store(
                org,
                environment,
                store,
                key,
                mode=mode,
                expose_text=expose_text,
                preset=preset,
                vector_dim=vector_dim,
            )
        )
        if connect_key and mode == "cloud_writer":
            scopes = ["write", "read:search"] + (["read:text"] if expose_text else [])
            minted = _cloud(
                lambda: client.mint_token(
                    "secret",
                    scopes,
                    name=f"cli connect ({environment})",
                    org=org,
                    environment=environment,
                )
            )
    creds = _load_credentials()
    host = creds.host if creds else "https://<your-orecloud-host>"
    target = f"{org}/{environment}/{row['store']}"

    def human() -> None:
        suffix = f", preset={preset}" if preset else ""
        typer.echo(
            f"registered {org}/{environment}/{row['store']} (key {row['key']}, "
            f"mode={row['mode']}{suffix})"
        )
        if minted is None:
            return
        # The key was minted bound to this environment, so the snippet never
        # restates the tenancy — Client resolves it from the credential.
        typer.echo()
        typer.echo("Connect from Python (the key is shown once — this is your copy):")
        typer.echo()
        typer.echo("    from lodedb.cloud import Client")
        typer.echo(
            f'    client = Client(token="{minted["secret"]}", host="{host}")'
        )
        typer.echo(f'    idx = client.store("{row["store"]}")')
        typer.echo('    idx.add_many([{"text": "hello lodedb"}]); print(idx.search("hello", k=3))')

    _emit(
        {
            "store": row,
            "target": target,
            "host": host,
            # The connect key, shown once, exactly like the human snippet.
            "secret": minted["secret"] if minted else None,
            "token": minted["token"] if minted else None,
        },
        human,
    )


def _resolve_index_key(
    client: CloudClient, org: str, environment: str, store: str, key: str | None
) -> str:
    """The explicit key, or the store's only index key when it is unambiguous
    (mirrors `store create`'s omit-the-key ergonomics). Exact-name lookup —
    never a page walk (an environment holds one store per end user)."""
    if key is not None:
        return key
    rows = _cloud(lambda: client.list_stores(org, environment, store=store))["stores"]
    if len(rows) == 1:
        return rows[0]["key"]
    if not rows:
        raise _fail(f"no such store {org}/{environment}/{store}")
    keys = ", ".join(row["key"] for row in rows)
    raise _fail(f"{org}/{environment}/{store} holds several index keys — pass --key ({keys})")


@store_app.command("delete")
def store_delete(
    store: Annotated[str, typer.Argument(help="Store name.")],
    key: Annotated[
        str | None,
        typer.Option(
            "--key",
            help="Delete only this index key (advanced; without it the whole store goes).",
        ),
    ] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Soft-delete a store (restorable for the grace period). With --key,
    delete only that index key inside the store."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        if key is None:
            if not yes:
                _confirm(f"Delete store {org}/{environment}/{store}?")
            out = _cloud(lambda: client.delete_store(org, environment, store))
            restore_hint = (
                f"  lodedb cloud store restore {out['slug']} "
                f"--environment {environment} --org {org}"
            )
        else:
            if not yes:
                _confirm(f"Delete index key {key} in store {org}/{environment}/{store}?")
            out = _cloud(lambda: client.delete_store_key(org, environment, store, key))
            restore_hint = (
                f"  lodedb cloud store restore {store} --key {out['slug']} "
                f"--environment {environment} --org {org}"
            )
    _emit(
        out,
        lambda: typer.echo(
            f"deleted (restorable until {out['purge_after']}):\n{restore_hint}"
        ),
    )


@store_app.command("restore")
def store_restore(
    parked: Annotated[
        str,
        typer.Argument(
            help="The parked store name `store delete` printed "
            "(or, with --key, the live store name)."
        ),
    ],
    key: Annotated[
        str | None,
        typer.Option("--key", help="Restore only this parked index key inside store PARKED."),
    ] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
) -> None:
    """Restore a soft-deleted store (or, with --key, one index key) inside
    its grace period."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        if key is None:
            out = _cloud(lambda: client.restore_store(org, environment, parked))
            _emit(out, lambda: typer.echo(f"restored {org}/{environment}/{out['slug']}"))
            return
        row = _cloud(lambda: client.restore_store_key(org, environment, parked, key))
    _emit(
        row, lambda: typer.echo(f"restored {org}/{environment}/{row['store']} (key {row['key']})")
    )


@store_app.command("history")
def store_history(
    store: Annotated[str, typer.Argument(help="Store name.")] = "default",
    key: Annotated[
        str | None,
        typer.Option("--key", help="Index key; omit when the store holds exactly one."),
    ] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
) -> None:
    """The store's restore window: every snapshot still held, newest first."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        key = _resolve_index_key(client, org, environment, store, key)
        rows = _cloud(lambda: client.store_history(org, environment, store, key))

    def human() -> None:
        if not rows:
            typer.echo("No snapshots yet.")
            return
        for row in rows:
            marker = "  <- current" if row["current"] else ""
            # Full ids: rollback takes the exact digest, so truncating here
            # would print a command the user can't complete.
            typer.echo(
                f"{row['snapshot_id']}  gen {row['generation']:<4} "
                f"{row['created_at']}{marker}"
            )
        typer.echo(
            f"\nRoll back with: lodedb cloud store rollback <snapshot-id> {store}"
        )

    _emit(rows, human)


@store_app.command("rollback")
def store_rollback(
    snapshot_id: Annotated[str, typer.Argument(help="Target snapshot id (from `store history`).")],
    store: Annotated[str, typer.Argument(help="Store name.")] = "default",
    key: Annotated[
        str | None,
        typer.Option("--key", help="Index key; omit when the store holds exactly one."),
    ] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Move the store back to a retained snapshot. Reversible: the replaced
    head stays in the restore window for the retention period."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        key = _resolve_index_key(client, org, environment, store, key)
        if not yes:
            _confirm(f"Roll {org}/{environment}/{store} back to {snapshot_id[:12]}…?")
        out = _cloud(
            lambda: client.rollback_store(org, environment, store, snapshot_id, key=key)
        )
    previous = out["previous_snapshot_id"]
    _emit(
        out,
        lambda: typer.echo(
            f"rolled back to {out['snapshot_id'][:12]} (gen {out['generation']})"
            + (f"; the replaced head {previous[:12]} stays restorable" if previous else "")
        ),
    )


@environments_app.command("delete")
def environments_delete(
    slug: str,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Soft-delete an environment and everything in it (restorable for the grace period)."""
    with _client() as client:
        org = _org_scope(client, org)
        if not yes:
            _confirm(f"Delete environment {org}/{slug} and every store in it?")
        out = _cloud(lambda: client.delete_environment(org, slug))
    _emit(
        out,
        lambda: typer.echo(
            f"deleted (restorable until {out['purge_after']}):\n"
            f"  lodedb cloud environments restore {out['slug']} --org {org}"
        ),
    )


@environments_app.command("restore")
def environments_restore(
    parked_slug: Annotated[
        str, typer.Argument(help="The parked slug `environments delete` printed.")
    ],
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
) -> None:
    """Restore a soft-deleted environment inside its grace period."""
    with _client() as client:
        org = _org_scope(client, org)
        out = _cloud(lambda: client.restore_environment(org, parked_slug))
    _emit(out, lambda: typer.echo(f"restored {org}/{out['slug']}"))


@org_app.command("delete")
def org_delete(
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Soft-delete an org and everything in it (restorable for the grace period)."""
    with _client() as client:
        org = _org_scope(client, org)
        if not yes:
            _confirm(f"Delete org {org} with all environments, stores, and tokens?")
        out = _cloud(lambda: client.delete_org(org))
    _emit(
        out,
        lambda: typer.echo(
            f"deleted (restorable until {out['purge_after']}):\n"
            f"  lodedb cloud org restore {out['slug']}"
        ),
    )


@org_app.command("restore")
def org_restore(
    parked_slug: Annotated[str, typer.Argument(help="The parked slug `org delete` printed.")],
) -> None:
    """Restore a soft-deleted org inside its grace period."""
    with _client() as client:
        out = _cloud(lambda: client.restore_org(parked_slug))
    _emit(out, lambda: typer.echo(f"restored {out['slug']}"))


@org_app.command("trash")
def org_trash(
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
) -> None:
    """List soft-deleted resources still restorable under an org."""
    with _client() as client:
        org = _org_scope(client, org)
        items = _cloud(lambda: client.list_trash(org))

    def human() -> None:
        if not items:
            typer.echo("Trash is empty.")
            return
        for item in items:
            where = "/".join(
                part for part in (item.get("environment"), item.get("store")) if part
            )
            scope = f" in {where}" if where else ""
            typer.echo(
                f"{item['kind']} {item['original']}{scope} — restore with "
                f"'{item['slug']}' until {item['purge_after']}"
            )

    _emit(items, human)


@org_app.command("export")
def org_export(
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
) -> None:
    """Print the org export manifest (JSON): every live environment and store
    with its head snapshot. Pull each store's bytes with `lodedb cloud pull`."""
    import json as _json

    with _client() as client:
        org = _org_scope(client, org)
        manifest = _cloud(lambda: client.export_org(org))
    typer.echo(_json.dumps(manifest, indent=2))


# ------------------------------------------------- memory verbs (Phase 8d)
# A store is one end user's LodeDB instance, so these take the store name
# (= the user's id) directly; `store list` is the user listing and
# `store delete` the forget-the-user verb.

def _scope_fields(agent: str | None, run: str | None) -> dict:
    """The optional narrowing axes as request fields."""
    fields: dict[str, str] = {}
    if agent is not None:
        fields["agent_id"] = agent
    if run is not None:
        fields["run_id"] = run
    return fields


@store_app.command("browse")
def store_browse(
    store: Annotated[str, typer.Argument(help="Store name (the end user's id).")],
    key: Annotated[str | None, typer.Option(help="Index key.")] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    after: Annotated[
        str | None,
        typer.Option(help="Keyset cursor: the last document id of the previous page."),
    ] = None,
    limit: Annotated[int, typer.Option(help="Page size (1-100).")] = 25,
    include_text: Annotated[
        bool,
        typer.Option("--include-text", help="Return stored text (needs read:text + expose_text)."),
    ] = False,
    agent: Annotated[str | None, typer.Option(help="Narrow to one agent's memories.")] = None,
    run: Annotated[str | None, typer.Option(help="Narrow to one session's memories.")] = None,
) -> None:
    """One page of a user's memories (ids + metadata, text on request)."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        payload = {
            "store": store,
            "key": key,
            "after": after,
            "limit": limit,
            "include_text": include_text,
            **_scope_fields(agent, run),
        }
        out = _cloud(lambda: client.browse_documents(org, environment, payload))

    def human() -> None:
        documents = out["documents"]
        if not documents:
            typer.echo("no memories")
            return
        for doc in documents:
            line = doc["id"]
            if include_text and doc.get("text"):
                line += f"  {' '.join(doc['text'].split())[:80]}"
            typer.echo(line)
        typer.echo(f"(page of {len(documents)}; next --after {documents[-1]['id']})")

    _emit(out, human)


@store_app.command("export")
def store_export(
    store: Annotated[str, typer.Argument(help="Store name (the end user's id).")],
    key: Annotated[str | None, typer.Option(help="Index key.")] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    include_text: Annotated[
        bool,
        typer.Option(
            "--include-text/--no-include-text",
            help="Include stored text (needs read:text + expose_text).",
        ),
    ] = True,
) -> None:
    """Export ALL of a user's memories as JSON (pages the browse endpoint
    to exhaustion) — the per-end-user data-export verb."""
    import json as _json

    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        documents: list[dict] = []
        after: str | None = None
        while True:
            payload = {
                "store": store,
                "key": key,
                "after": after,
                "limit": 100,
                "include_text": include_text,
            }
            # bind=payload: the lambda runs inside this iteration, but binding
            # explicitly keeps the closure honest (and ruff B023 quiet).
            page = _cloud(lambda bind=payload: client.browse_documents(org, environment, bind))
            documents.extend(page["documents"])
            if len(page["documents"]) < 100:
                break
            after = page["documents"][-1]["id"]
    typer.echo(_json.dumps({"store": store, "documents": documents}, indent=2))


@store_app.command("recall")
def store_recall(
    store: Annotated[str, typer.Argument(help="Store name (the end user's id).")],
    text: Annotated[str, typer.Argument(help="Raw text — a whole user message, not a query.")],
    key: Annotated[str | None, typer.Option(help="Index key.")] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    k: Annotated[int, typer.Option(help="How many memories to return.")] = 10,
    include_text: Annotated[
        bool, typer.Option("--include-text", help="Return stored text inline.")
    ] = False,
    agent: Annotated[str | None, typer.Option(help="Narrow to one agent's memories.")] = None,
    run: Annotated[str | None, typer.Option(help="Narrow to one session's memories.")] = None,
) -> None:
    """Non-exact retrieval from raw text: the server derives sub-queries
    (windows + key phrases) and fuses their rankings."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        payload = {
            "store": store,
            "key": key,
            "text": text,
            "k": k,
            "include_text": include_text,
            **_scope_fields(agent, run),
        }
        out = _cloud(lambda: client.recall(org, environment, payload))

    def human() -> None:
        if not out["hits"]:
            typer.echo("no memories recalled")
            return
        for hit in out["hits"]:
            line = f"{hit['score']:.4f}  {hit['id']}"
            if include_text and hit.get("text"):
                line += f"  {' '.join(hit['text'].split())[:80]}"
            typer.echo(line)
            typer.echo(f"        matched: {', '.join(hit['matched'])}")
        typer.echo(f"(sub-queries: {', '.join(out['queries'])})")

    _emit(out, human)


@store_app.command("delete-memories")
def store_delete_memories(
    store: Annotated[str, typer.Argument(help="Store name (the end user's id).")],
    key: Annotated[str | None, typer.Option(help="Index key.")] = None,
    environment: Annotated[str | None, typer.Option(help=_ENVIRONMENT_HELP)] = None,
    org: Annotated[str | None, typer.Option(help=_ORG_HELP)] = None,
    agent: Annotated[
        str | None, typer.Option(help="Narrow the deletion to one agent's memories.")
    ] = None,
    run: Annotated[
        str | None, typer.Option(help="Narrow the deletion to one session's memories.")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete a user's memories in place (expired ones included), keeping
    the store registered. Async like every cloud write. To forget the user
    entirely — row, entitlement slot and all — use `store delete`."""
    with _client() as client:
        org, environment = _tenancy(client, org, environment)
        if not yes:
            narrowed = "".join(
                f" {label} {value}" for label, value in (("agent", agent), ("run", run)) if value
            )
            _confirm(
                f"Delete ALL memories{narrowed} in {org}/{environment}/{store}?"
            )
        payload = {"store": store, "key": key, **_scope_fields(agent, run)}
        out = _cloud(lambda: client.delete_memories(org, environment, payload))
    _emit(
        out,
        lambda: typer.echo(
            f"accepted: {out['document_count']} memories queued for removal "
            f"({len(out['write_ids'])} writes)"
        ),
    )
