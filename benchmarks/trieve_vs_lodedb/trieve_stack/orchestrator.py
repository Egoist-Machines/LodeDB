"""Boot the full Trieve stack natively in one container and bootstrap org+dataset.

Modal's docker-in-docker is CPU-only, so we do not use docker-compose; every Trieve
dependency runs as a native child process in this container. Startup order (each
health-waited before the next):

1. Postgres  (initdb, create db ``trieve`` + role postgres/password)
2. Redis     (requirepass = PLAN password)
3. Qdrant    (standalone binary, HTTP :6333 / gRPC :6334, api key)
4. model server (dense/sparse/rerank + OIDC discovery stub on :7070)
5. trieve-server + ingestion-worker  (from ``$TRIEVE_SERVER_BIN`` / ``$INGESTION_WORKER_BIN``)

trieve-server panics at boot unless Postgres, Redis, and OIDC discovery are all up,
so the model server (which serves the OIDC stub) must be started before it. Migrations
run automatically at boot via ``embed_migrations!``. After boot we seed via REST using
``ADMIN_API_KEY``: create the org, then the dataset whose ``server_configuration``
points every model URL at our localhost model server.

``boot_and_bootstrap`` returns ``{base_url, dataset_id, org_id}``.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# Ports (kept in sync with the env block in trieve_modal / PLAN.md).
POSTGRES_PORT = 5432
REDIS_PORT = 6379
QDRANT_HTTP_PORT = 6333
QDRANT_GRPC_PORT = 6334
MODEL_SERVER_PORT = int(os.environ.get("MODEL_SERVER_PORT", "7070"))
TRIEVE_SERVER_PORT = 8090
TRIEVE_BASE_URL = f"http://localhost:{TRIEVE_SERVER_PORT}"

REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "thisredispasswordisverysecureandcomplex")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "qdrant_pass")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "admin")

# Filesystem roots (overridable so a smoke run can use a scratch dir).
DATA_ROOT = Path(os.environ.get("TRIEVE_STACK_DATA_ROOT", "/var/lib/trieve"))
PG_DATA_DIR = DATA_ROOT / "pgdata"
QDRANT_STORAGE_DIR = DATA_ROOT / "qdrant"
LOG_DIR = DATA_ROOT / "logs"


# -- process + readiness helpers -------------------------------------------


def _log(message: str) -> None:
    """Prints a timestamped orchestrator log line."""

    print(f"[orchestrator] {message}", flush=True)


def _open_log(name: str) -> Any:
    """Opens (truncating) a per-service log file under the log dir."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return open(LOG_DIR / f"{name}.log", "w")


def _spawn(name: str, argv: list[str], *, env: dict[str, str] | None = None,
           cwd: str | None = None) -> subprocess.Popen[bytes]:
    """Starts a child process, tee-ing stdout+stderr to its log file."""

    _log(f"starting {name}: {' '.join(argv)}")
    log_file = _open_log(name)
    process = subprocess.Popen(
        argv,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
    )
    return process


def _http_ok(url: str, *, headers: dict[str, str] | None = None, timeout: float = 3.0) -> bool:
    """Returns True if a GET to ``url`` returns a 2xx/3xx status."""

    request = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 400
    except urllib.error.HTTPError as exc:
        # A 401/403 still proves the port is serving; readiness callers decide.
        return 200 <= exc.code < 500
    except Exception:
        return False


def _wait_until(name: str, predicate: Any, *, timeout: float, interval: float = 1.0) -> None:
    """Polls ``predicate`` until true or raises after ``timeout`` seconds."""

    deadline = time.time() + timeout
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            if predicate():
                _log(f"{name} ready after {attempt} check(s)")
                return
        except Exception as exc:  # noqa: BLE001 - readiness probes are best-effort
            if attempt % 10 == 0:
                _log(f"{name} probe error (attempt {attempt}): {exc}")
        time.sleep(interval)
    raise RuntimeError(f"{name} did not become ready within {timeout}s")


def _which(candidates: list[str]) -> str:
    """Returns the first candidate that resolves on PATH or as an absolute file."""

    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(f"none of {candidates} found on PATH")


# -- Postgres ---------------------------------------------------------------


def _pg_bindir() -> str:
    """Finds the Postgres bin dir (initdb/pg_ctl/psql) across Debian layouts."""

    override = os.environ.get("PG_BINDIR")
    if override:
        return override
    for base in sorted(Path("/usr/lib/postgresql").glob("*/bin"), reverse=True):
        if (base / "initdb").exists():
            return str(base)
    # Fall back to PATH resolution.
    initdb = shutil.which("initdb")
    if initdb:
        return str(Path(initdb).parent)
    raise FileNotFoundError("Postgres bin dir not found (set PG_BINDIR)")


def _as_postgres(argv: list[str]) -> list[str]:
    """Wraps a command to run as the non-root 'postgres' user.

    initdb and the postgres server refuse to run as root, and Modal containers run
    as root, so drop privileges via runuser (fall back to su). The postgres OS user
    and group are created by the postgresql apt package.
    """

    import shlex

    runuser = shutil.which("runuser")
    if runuser:
        return [runuser, "-u", "postgres", "--", *argv]
    su = shutil.which("su") or "/bin/su"
    return [su, "-s", "/bin/sh", "postgres", "-c", " ".join(shlex.quote(part) for part in argv)]


def start_postgres() -> subprocess.Popen[bytes]:
    """Inits (if needed) and starts Postgres, then creates the trieve db + role."""

    bindir = _pg_bindir()
    initdb = str(Path(bindir) / "initdb")
    postgres = str(Path(bindir) / "postgres")
    psql = str(Path(bindir) / "psql")

    PG_DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Postgres (initdb + server) refuses to run as root, so it runs as the 'postgres'
    # user, which must own its data dir and be able to traverse the parents (this is
    # also why the data dir lives under /var/lib, not /root whose 700 perms block it).
    os.chmod(DATA_ROOT, 0o755)
    shutil.chown(PG_DATA_DIR, user="postgres", group="postgres")
    if not (PG_DATA_DIR / "PG_VERSION").exists():
        _log("initdb (trust auth for the local single-container superuser)")
        subprocess.run(
            _as_postgres(
                [
                    initdb, "-D", str(PG_DATA_DIR), "-U", "postgres",
                    "--auth=trust", "--encoding=UTF8",
                ]
            ),
            check=True,
            stdout=subprocess.DEVNULL,
        )

    env = dict(os.environ)
    process = _spawn(
        "postgres",
        _as_postgres([
            postgres,
            "-D",
            str(PG_DATA_DIR),
            "-p",
            str(POSTGRES_PORT),
            "-c",
            "listen_addresses=127.0.0.1",
            # Bench-oriented: more memory/connections, fsync off (ephemeral container).
            "-c",
            "fsync=off",
            "-c",
            "synchronous_commit=off",
            "-c",
            "full_page_writes=off",
            "-c",
            "max_connections=200",
            "-c",
            "shared_buffers=2GB",
        ]),
        env=env,
    )

    def _pg_ready() -> bool:
        return (
            subprocess.run(
                [str(Path(bindir) / "pg_isready"), "-h", "127.0.0.1", "-p", str(POSTGRES_PORT)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )

    _wait_until("postgres", _pg_ready, timeout=90)

    # Set the postgres role password (trust auth means no password needed to connect
    # as superuser here) and create the trieve database. DATABASE_URL uses
    # postgres:password so mirror that.
    _log("configuring postgres role + trieve database")
    subprocess.run(
        [
            psql, "-h", "127.0.0.1", "-p", str(POSTGRES_PORT), "-U", "postgres",
            "-v", "ON_ERROR_STOP=0", "-c", "ALTER USER postgres WITH PASSWORD 'password';",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # CREATE DATABASE is not idempotent; ignore "already exists" on a warm volume.
    subprocess.run(
        [psql, "-h", "127.0.0.1", "-p", str(POSTGRES_PORT), "-U", "postgres",
         "-c", "CREATE DATABASE trieve;"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process


# -- Redis ------------------------------------------------------------------


def start_redis() -> subprocess.Popen[bytes]:
    """Starts redis-server with the PLAN password and waits for PONG."""

    redis_server = _which(["redis-server"])
    process = _spawn(
        "redis",
        [
            redis_server,
            "--port",
            str(REDIS_PORT),
            "--requirepass",
            REDIS_PASSWORD,
            "--bind",
            "127.0.0.1",
            "--save",
            "",  # no RDB snapshots; ephemeral bench store
            "--appendonly",
            "no",
        ],
    )

    redis_cli = _which(["redis-cli"])

    def _redis_ready() -> bool:
        result = subprocess.run(
            [redis_cli, "-p", str(REDIS_PORT), "-a", REDIS_PASSWORD, "ping"],
            capture_output=True,
            text=True,
        )
        return "PONG" in (result.stdout or "")

    _wait_until("redis", _redis_ready, timeout=30)
    return process


# -- Qdrant -----------------------------------------------------------------


def start_qdrant() -> subprocess.Popen[bytes]:
    """Starts the standalone Qdrant binary and waits for /readyz.

    Config overrides via QDRANT__ env vars (Qdrant's double-underscore convention):
    api key, storage path, and explicit HTTP/gRPC ports. No config.yaml needed; the
    binary uses built-in defaults for everything else.
    """

    qdrant = _which([os.environ.get("QDRANT_BIN", "qdrant"), "/usr/local/bin/qdrant", "qdrant"])
    QDRANT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update(
        {
            "QDRANT__SERVICE__API_KEY": QDRANT_API_KEY,
            "QDRANT__SERVICE__HTTP_PORT": str(QDRANT_HTTP_PORT),
            "QDRANT__SERVICE__GRPC_PORT": str(QDRANT_GRPC_PORT),
            "QDRANT__SERVICE__HOST": "0.0.0.0",
            "QDRANT__STORAGE__STORAGE_PATH": str(QDRANT_STORAGE_DIR),
            "QDRANT__TELEMETRY_DISABLED": "true",
        }
    )
    process = _spawn("qdrant", [qdrant], env=env)

    api_headers = {"api-key": QDRANT_API_KEY}

    def _qdrant_ready() -> bool:
        # /readyz needs no auth in 1.12; fall back to the authed root for older probes.
        if _http_ok(f"http://localhost:{QDRANT_HTTP_PORT}/readyz"):
            return True
        return _http_ok(f"http://localhost:{QDRANT_HTTP_PORT}/", headers=api_headers)

    _wait_until("qdrant", _qdrant_ready, timeout=90)
    return process


# -- model server -----------------------------------------------------------


def start_model_server() -> subprocess.Popen[bytes]:
    """Starts the dense/sparse/rerank + OIDC model server as a child process.

    Runs it as its own interpreter process (``python -m ...``) so a model crash does
    not take down the orchestrator, and its logs land in the log dir. The OIDC stub it
    serves must be reachable before trieve-server boots.
    """

    module_env = dict(os.environ)
    # Make the benchmark dir importable so `-m trieve_stack.model_server` resolves.
    bench_dir = str(Path(__file__).resolve().parent.parent)
    module_env["PYTHONPATH"] = os.pathsep.join(
        [bench_dir, module_env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    process = _spawn(
        "model_server",
        [sys.executable, "-m", "trieve_stack.model_server"],
        env=module_env,
        cwd=bench_dir,
    )

    def _model_ready() -> bool:
        base = f"http://localhost:{MODEL_SERVER_PORT}"
        return _http_ok(f"{base}/health") and _http_ok(f"{base}/.well-known/openid-configuration")

    # First request triggers CUDA model loads; give it a generous window.
    _wait_until("model_server", _model_ready, timeout=600)
    return process


# -- trieve-server + ingestion-worker --------------------------------------


def start_trieve_server() -> subprocess.Popen[bytes]:
    """Starts trieve-server (binds :8090); migrations + OIDC discovery run at boot."""

    server_bin = _which([os.environ["TRIEVE_SERVER_BIN"]])
    clone_dir = os.environ.get("TRIEVE_CLONE_DIR")  # cwd so it finds src/public etc.
    process = _spawn("trieve-server", [server_bin], env=dict(os.environ), cwd=clone_dir)

    def _server_ready() -> bool:
        # trieve-server serves an unauthenticated health endpoint at /api/health.
        return _http_ok(f"{TRIEVE_BASE_URL}/api/health")

    _wait_until("trieve-server", _server_ready, timeout=180)
    return process


def start_ingestion_worker() -> subprocess.Popen[bytes]:
    """Starts the ingestion-worker (drains the redis/broccoli queue into Qdrant+PG)."""

    worker_bin = _which([os.environ["INGESTION_WORKER_BIN"]])
    clone_dir = os.environ.get("TRIEVE_CLONE_DIR")
    # The worker has no HTTP surface; a short settle is enough (readiness is proven
    # later when ingested chunk_count starts climbing).
    process = _spawn("ingestion-worker", [worker_bin], env=dict(os.environ), cwd=clone_dir)
    time.sleep(2.0)
    return process


# -- REST bootstrap ---------------------------------------------------------


def _post_json(path: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """POSTs JSON to trieve-server and returns the parsed response (raises on non-2xx)."""

    import json

    body = json.dumps(payload).encode("utf-8")
    all_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(
        f"{TRIEVE_BASE_URL}{path}", data=body, headers=all_headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {path} -> {exc.code}: {detail}") from exc


def _dataset_server_configuration(embedding_size: int, model_name: str) -> dict[str, Any]:
    """Returns the dataset server_configuration pinning every URL at the model server.

    All keys are literal ALL-CAPS (Trieve's DatasetConfiguration uses the Rust field
    names verbatim, no serde rename). EMBEDDING_BASE_URL / RERANKER_BASE_URL are set
    explicitly to localhost:7070 so Trieve's sentinel-rewrite (empty ->
    OPENAI_BASE_URL; k8s sentinel -> RERANKER_SERVER_ORIGIN) never fires. SPLADE has
    no config URL; it always reads the SPARSE_SERVER_*_ORIGIN envs, which the env
    block also points at the model server. FULLTEXT_ENABLED true keeps the SPLADE leg
    of Trieve hybrid live (dense + SPLADE + cross-encoder rerank); BM25 stays off.
    """

    model_url = f"http://localhost:{MODEL_SERVER_PORT}"
    return {
        "EMBEDDING_BASE_URL": model_url,
        "EMBEDDING_MODEL_NAME": model_name,
        "EMBEDDING_SIZE": embedding_size,
        "EMBEDDING_QUERY_PREFIX": "",
        "DISTANCE_METRIC": "cosine",
        "SEMANTIC_ENABLED": True,
        "FULLTEXT_ENABLED": True,
        "BM25_ENABLED": False,
        "QDRANT_ONLY": False,
        "RERANKER_BASE_URL": "http://embedding-reranker.default.svc.cluster.local",
        "RERANKER_MODEL_NAME": "bge-reranker-base",
    }


def bootstrap_org_and_dataset(
    *, embedding_size: int = 384, model_name: str = "all-MiniLM-L6-v2"
) -> dict[str, str]:
    """Creates the bench org + dataset via REST; returns {base_url, org_id, dataset_id}.

    Auth is ``Authorization: Bearer <ADMIN_API_KEY>``. Org create needs no
    TR-Organization header (the admin key resolves to the seeded default owner via the
    key-hash path); dataset create is scoped to the new org via TR-Organization. The
    RERANKER_BASE_URL is left as the k8s sentinel so Trieve rewrites it to
    RERANKER_SERVER_ORIGIN (the model server), which is the cleanest way to hit the
    native cross-encoder branch.
    """

    admin_auth = {"Authorization": f"Bearer {ADMIN_API_KEY}"}

    _log("creating organization 'bench-org'")
    org = _post_json("/api/organization", {"name": "bench-org"}, admin_auth)
    # Response is the created Organization (may be nested under "organization").
    org_id = str(org.get("id") or org.get("organization", {}).get("id"))
    if not org_id or org_id == "None":
        raise RuntimeError(f"could not read org id from create response: {org}")
    _log(f"org_id={org_id}")

    dataset_headers = {**admin_auth, "TR-Organization": org_id}
    payload = {
        "dataset_name": "bench-ds",
        "tracking_id": "bench-ds-1",
        "server_configuration": _dataset_server_configuration(embedding_size, model_name),
    }
    _log("creating dataset 'bench-ds'")
    dataset = _post_json("/api/dataset", payload, dataset_headers)
    dataset_id = str(dataset.get("id") or dataset.get("dataset", {}).get("id"))
    if not dataset_id or dataset_id == "None":
        raise RuntimeError(f"could not read dataset id from create response: {dataset}")
    _log(f"dataset_id={dataset_id}")

    return {"base_url": TRIEVE_BASE_URL, "org_id": org_id, "dataset_id": dataset_id}


# -- top-level orchestration ------------------------------------------------


_STARTED: list[subprocess.Popen[bytes]] = []


def _shutdown() -> None:
    """Terminates all spawned children (best-effort, reverse start order)."""

    for process in reversed(_STARTED):
        try:
            process.send_signal(signal.SIGTERM)
        except Exception:
            pass


def boot_and_bootstrap(
    *, embedding_size: int = 384, model_name: str = "all-MiniLM-L6-v2"
) -> dict[str, str]:
    """Boots the whole stack in dependency order and returns the bench handle.

    Returns ``{base_url, org_id, dataset_id}``. Children are tracked so the caller can
    call ``teardown()`` (or rely on container exit) to reap them.
    """

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _STARTED.append(start_postgres())
    _STARTED.append(start_redis())
    _STARTED.append(start_qdrant())
    _STARTED.append(start_model_server())  # serves the OIDC stub trieve boots against
    _STARTED.append(start_trieve_server())
    _STARTED.append(start_ingestion_worker())
    handle = bootstrap_org_and_dataset(embedding_size=embedding_size, model_name=model_name)
    _log(f"stack ready: {handle}")
    return handle


def teardown() -> None:
    """Public teardown for callers that want to reap children explicitly."""

    _shutdown()


if __name__ == "__main__":
    # Manual boot for debugging inside a container; prints the handle as JSON.
    import json

    try:
        print(json.dumps(boot_and_bootstrap()))
    finally:
        pass
