"""Writing credentials into a dotenv file without ever printing them.

`lodedb cloud tokens mint --env-file` and `lodedb cloud init --agents` land the minted
secret here instead of on stdout: anything printed by an agent-driven CLI run
becomes a permanent part of the agent's transcript, while a file write keeps
the secret on disk only (the same philosophy as the sealed-box login, one hop
further). The writer is deliberately conservative: existing lines it does not
manage are preserved byte-for-byte, managed keys are replaced in place, and a
missing file is created 0600.
"""

from __future__ import annotations

import os
from pathlib import Path

# Patterns in any .gitignore between the env file and the repo root that
# already cover a file named `.env` (or `<name>.env`). Deliberately a simple
# whitelist rather than full gitignore semantics: a miss just means we append
# one more (harmless, idempotent-by-note) line.
_IGNORE_PATTERNS = ("{name}", "/{name}", "*.env", ".env*", "**/{name}")


def write_env_values(path: str | Path, values: dict[str, str]) -> Path:
    """Set `KEY=value` lines in the dotenv file at `path`, creating it if
    missing. Managed keys replace their existing line in place; every other
    line (comments, other keys, blanks) is preserved. The result is always
    owner-only (0600): secrets land here, and a pre-existing 0644 `.env` must
    not keep the minted token group/world-readable. Returns the path."""
    target = Path(path)
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    remaining = dict(values)
    updated: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else None
        if key in remaining:
            updated.append(f"{key}={remaining.pop(key)}")
        else:
            updated.append(line)
    updated.extend(f"{key}={value}" for key, value in remaining.items())
    content = "\n".join(updated) + "\n"
    # Atomic replace with the final mode already set: the secret never exists
    # on disk readable to anyone but the owner, even briefly, and an existing
    # permissive file's mode is not inherited. mkstemp (unique name, O_EXCL,
    # 0600) rather than a fixed sibling name: a predictable scratch path in a
    # shared directory could be pre-planted as a symlink, aiming the secret at
    # an attacker-chosen file.
    import tempfile

    fd, scratch = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        # The rename carries mkstemp's owner-only mode onto the target; no
        # post-publication chmod (a path-based chmod on the published name
        # would reintroduce the symlink race this scratch file exists to
        # avoid, and has no exact-mode meaning on Windows anyway).
        os.replace(scratch, target)
    except BaseException:
        try:
            os.unlink(scratch)
        except OSError:
            pass
        raise
    return target


def read_env_value(path: str | Path, key: str) -> str | None:
    """The value of `key` in the dotenv file, or None (missing file or key).
    Last assignment wins, matching how dotenv loaders behave."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    value = None
    for line in lines:
        if "=" in line and line.split("=", 1)[0].strip() == key:
            value = line.split("=", 1)[1].strip()
    return value


def _git_root(start: Path) -> Path | None:
    """The enclosing git working-tree root, or None when `start` is untracked
    territory. A `.git` file (worktrees, submodules) counts."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def ensure_gitignored(env_path: str | Path) -> str | None:
    """Make sure the env file cannot be committed: if it lives in a git repo
    and no .gitignore between it and the repo root covers it, append its name
    to the repo-root .gitignore (creating the file if needed). Returns a
    human-readable note describing what happened, or None when the file was
    already covered."""
    target = Path(env_path).resolve()
    root = _git_root(target.parent)
    if root is None:
        return f"{target.name} is not inside a git repository — keep it out of version control"
    covered = {pattern.format(name=target.name) for pattern in _IGNORE_PATTERNS}
    directories = [target.parent]
    while directories[-1] != root:
        directories.append(directories[-1].parent)
    for directory in directories:
        gitignore = directory / ".gitignore"
        try:
            lines = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()}
        except FileNotFoundError:
            continue
        if lines & covered:
            return None
    root_ignore = root / ".gitignore"
    existing = root_ignore.read_text(encoding="utf-8") if root_ignore.exists() else ""
    separator = "" if existing.endswith("\n") or not existing else "\n"
    root_ignore.write_text(f"{existing}{separator}{target.name}\n", encoding="utf-8")
    return f"added {target.name} to {root_ignore}"
