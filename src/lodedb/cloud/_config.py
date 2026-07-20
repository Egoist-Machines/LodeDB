"""Where the CLI keeps its cloud credentials.

One JSON file, `~/.orecloud/credentials.json`, chmod 0600, holding the
control-plane host and the personal access token from `lodedb cloud login`.
A plain file (not the OS keychain) is deliberate for now: it works headless
(CI boxes, containers, SSH sessions) with the same permission story as
`~/.aws/credentials`. ORECLOUD_TOKEN overrides the file entirely, so
ephemeral environments never need to write it; the host defaults to the
hosted control plane, with ORECLOUD_HOST (or `--host`) overriding it for
staging and self-hosted deployments.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

# The hosted control plane. Baked in so nobody has to know the URL (the same
# way `gh` knows github.com); `--host` / ORECLOUD_HOST remain the override
# for staging and self-hosted control planes.
DEFAULT_HOST = "https://api.egoistmachines.com"


def config_dir() -> Path:
    # Computed per call, not at import: ORECLOUD_CONFIG_DIR must work when
    # set after import (tests, wrappers).
    return Path(os.environ.get("ORECLOUD_CONFIG_DIR", "~/.orecloud")).expanduser()


def credentials_file() -> Path:
    return config_dir() / "credentials.json"


@dataclass(frozen=True)
class Credentials:
    host: str
    token: str
    source: str  # "env" or "file", surfaced by whoami for debuggability


class CredentialsError(RuntimeError):
    """Credential configuration is wrong in a way that must not be guessed
    around (e.g. half-set environment overrides)."""


def load_credentials() -> Credentials | None:
    env_token = os.environ.get("ORECLOUD_TOKEN")
    env_host = os.environ.get("ORECLOUD_HOST")
    if env_token:
        # An env token never combines with the file's host: falling back to
        # the file could aim a CI job's mutating commands at whatever control
        # plane a developer last logged in to. Without ORECLOUD_HOST the
        # token targets the hosted default.
        return Credentials(
            host=(env_host or DEFAULT_HOST).rstrip("/"), token=env_token, source="env"
        )
    if env_host:
        # A host override with no matching token must fail closed: silently
        # pairing it with the file's token would aim that stored credential
        # at a control plane the login never targeted.
        raise CredentialsError(
            "ORECLOUD_HOST is set but ORECLOUD_TOKEN is not — set ORECLOUD_TOKEN "
            "to use environment credentials, or unset ORECLOUD_HOST to use the "
            "stored credentials file"
        )
    try:
        raw = json.loads(credentials_file().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    host, token = raw.get("host"), raw.get("token")
    if not host or not token:
        return None
    return Credentials(host=host.rstrip("/"), token=token, source="file")


def save_credentials(host: str, token: str) -> Path:
    target = credentials_file()
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # Write-then-rename with the final mode already set: the secret never
    # exists on disk world-readable, even briefly.
    tmp = target.with_suffix(".json.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"host": host.rstrip("/"), "token": token}, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, target)
    os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
    return target


def delete_credentials() -> bool:
    try:
        credentials_file().unlink()
        return True
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------- orecloud.toml

# The committable per-directory remote record: which control plane and
# org/environment/store a data directory syncs with. No secrets (credentials
# stay in ~/.orecloud), so the file is safe to commit next to the data.
REMOTE_FILE_NAME = "orecloud.toml"


@dataclass(frozen=True)
class RemoteConfig:
    host: str
    org: str
    environment: str
    store: str


def remote_config_file(local_dir: str | Path) -> Path:
    return Path(local_dir) / REMOTE_FILE_NAME


def load_remote(local_dir: str | Path) -> RemoteConfig | None:
    """The directory's recorded remote, or None. A present-but-malformed file
    raises; guessing around a broken record could aim a push at the wrong
    environment."""
    import tomllib

    path = remote_config_file(local_dir)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        table = tomllib.loads(raw.decode("utf-8")).get("remote", {})
        host, org = table["host"], table["org"]
        environment = table["environment"]
        store = table.get("store", "default")
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, KeyError, TypeError) as error:
        raise CredentialsError(f"{path} is malformed: {error}") from error
    if not all(isinstance(v, str) and v for v in (host, org, environment, store)):
        raise CredentialsError(f"{path} is malformed: empty or non-string fields")
    return RemoteConfig(host=host.rstrip("/"), org=org, environment=environment, store=store)


def save_remote(local_dir: str | Path, config: RemoteConfig) -> Path:
    def quote(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    path = remote_config_file(local_dir)
    path.write_text(
        "# OreCloud managed remote for this data directory (no secrets; safe to commit).\n"
        "[remote]\n"
        f"host = {quote(config.host.rstrip('/'))}\n"
        f"org = {quote(config.org)}\n"
        f"environment = {quote(config.environment)}\n"
        f"store = {quote(config.store)}\n"
    )
    return path
