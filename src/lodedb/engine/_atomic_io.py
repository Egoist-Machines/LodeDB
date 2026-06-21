"""Atomic, optionally power-loss-durable file replacement (stdlib only).

LodeDB publishes every persisted file by writing a sibling ``*.tmp`` and then
``os.replace``-ing it into place. ``os.replace`` is atomic *within a filesystem*
on POSIX and Windows — a concurrent reader sees either the whole old file or the
whole new file, never a torn one — so **atomicity is always on** and free.

It is not, however, *durable*: after a power loss or kernel panic (as opposed to
a clean process crash) neither the temp file's bytes nor the rename are
guaranteed to have reached stable storage, and on some filesystems the rename
can be reordered ahead of the data write. Closing that gap requires fsyncing the
temp file *before* the rename and the containing directory *after* it. That is
the extra cost gated by the ``fsync`` flag (LodeDB's ``durability="fsync"``):
the default ``"fast"`` path keeps the sub-millisecond commit profile, and
``"fsync"`` trades throughput for power-loss durability.

This only makes each individual file durable; LodeDB's multi-file commit is made
crash-*atomic* separately (see the persistence layer). On network filesystems
fsync semantics are unreliable; LodeDB targets local disk.
"""

from __future__ import annotations

import os
from pathlib import Path

_FAST = "fast"
_FSYNC = "fsync"
_VALID_DURABILITY = (_FAST, _FSYNC)


def normalize_durability(value: str | None) -> bool:
    """Maps a durability mode string to ``fsync_on_commit`` (True/False).

    ``"fast"`` (or ``None``/empty) -> ``False`` (atomic but not power-loss
    durable); ``"fsync"`` -> ``True`` (fsync each file + its directory on
    commit). Any other value raises :class:`ValueError`.
    """

    if value is None:
        return False
    mode = value.strip().lower()
    if mode in ("", _FAST):
        return False
    if mode == _FSYNC:
        return True
    raise ValueError(f"durability must be one of {_VALID_DURABILITY}, got {value!r}")


def durability_from_env(env: dict[str, str] | None = None) -> bool:
    """Returns the default ``fsync_on_commit`` from ``LODEDB_DURABILITY``.

    Unset defaults to ``False`` (``"fast"``). Used only when no explicit
    ``durability=`` is passed to :class:`~lodedb.local.db.LodeDB`.
    """

    source = os.environ if env is None else env
    return normalize_durability(source.get("LODEDB_DURABILITY"))


def fsync_path(path: str | Path) -> None:
    """Flushes one already-written file's contents to stable storage."""

    fd = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir(path: str | Path) -> None:
    """Flushes a directory entry so a rename/create within it is durable.

    A no-op where directories cannot be opened/fsynced (e.g. Windows), which is
    acceptable: there the rename's durability is handled by the filesystem.
    """

    try:
        fd = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def durable_replace(tmp: str | Path, dst: str | Path, *, fsync: bool) -> None:
    """Atomically replaces ``dst`` with ``tmp``; durable when ``fsync`` is set.

    Atomicity (no torn pathnames) is always provided by ``os.replace``. When
    ``fsync`` is True the temp file is fsynced before the rename and the
    destination directory after it, so the committed file survives a power loss.
    """

    tmp = Path(tmp)
    dst = Path(dst)
    if fsync:
        fsync_path(tmp)
    os.replace(tmp, dst)
    if fsync:
        fsync_dir(dst.parent)
