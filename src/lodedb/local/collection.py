"""Named vector spaces: a directory of sibling LodeDB indexes under one root.

A :class:`LodeCollection` groups several independent LodeDB indexes ("spaces")
under a single directory, each in its own subdirectory and each free to use a
different model or dimension, for example a ``"text"`` space at ``model="minilm"``
beside an ``"image"`` space at ``model="clip"``. This is the local-first analogue
of a database with named vectors per record, with one deliberate difference:
spaces are searched **independently** and never scored against each other, because
vectors produced by different models do not share a comparable space.

The collection keeps a small ``collection.json`` manifest recording how each space
was opened (model, vector_dim, bit_width) and re-enforces it on the next open, so
a space cannot be silently reopened with a mismatching model or dimension. The
engine is unchanged: each space is an ordinary, fully crash-safe LodeDB index, and
the manifest is only a registry of siblings.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durable_replace
from lodedb.engine._filelock import WriterLock, lodedb_lock_timeout_from_env
from lodedb.local.db import LodeDB

_MANIFEST_NAME = "collection.json"
_MANIFEST_VERSION = 1
# Space names are used as directory names, so keep them to a safe, portable set
# and forbid anything that could escape the collection root.
_SPACE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
# Sentinel for space() arguments the caller did not pass: a recorded space then
# reopens from its manifest config instead of being compared against fresh defaults.
_UNSET: Any = object()


class LodeCollection:
    """A directory of named LodeDB vector spaces sharing one root path.

    Example::

        col = LodeCollection("./memory")
        notes = col.space("text", model="minilm")
        shots = col.space("image", model="clip")     # needs the [image] extra
        notes.add("the quick brown fox")
        shots.add_image("diagram.png", metadata={"path": "diagram.png"})
        col.close()

    Reopening ``col.space("image")`` returns the same configuration; opening it
    with a different ``model``/``vector_dim`` raises :class:`ValueError`.
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        """Opens (or creates) a collection root and loads its space manifest.

        ``read_only=True`` opens every space read-only and never creates the root
        or a missing space (see :meth:`space`); the root must already exist.
        """

        self.path = Path(path)
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_dir():
                raise FileNotFoundError(
                    f"LodeCollection(read_only=True) requires an existing directory: {self.path}"
                )
        else:
            self.path.mkdir(parents=True, exist_ok=True)
        self._spaces_config: dict[str, dict[str, Any]] = self._load_manifest()
        self._open: dict[str, LodeDB] = {}

    # -- public API ---------------------------------------------------------

    def space(
        self,
        name: str,
        *,
        model: str = _UNSET,
        vector_dim: int | None = _UNSET,
        bit_width: int = _UNSET,
        **kwargs: Any,
    ) -> LodeDB:
        """Opens (or creates) the named space and returns its :class:`LodeDB` handle.

        The space lives at ``<root>/<name>/``. A new space records its
        ``(model, vector_dim, bit_width)`` in the manifest (``model="minilm"`` and
        ``bit_width=4`` by default; pass ``vector_dim=`` for a bring-your-own-vectors
        space). **Reopening uses the recorded config**: ``col.space("name")`` with no
        config args reopens any recorded space (including vector-only / ``clip`` /
        non-default ones) without restating it; passing a config value that conflicts
        with what was recorded raises :class:`ValueError`. Extra keyword arguments
        (``store_text``, ``index_text``, ``device``, ``commit_mode``, ...) pass
        through to :class:`LodeDB`. The same handle is returned for repeated calls in
        one process.
        """

        safe = self._validate_name(name)
        recorded = self._spaces_config.get(safe)
        if recorded is not None:
            # Reopen from the recorded config; enforce only explicit, conflicting
            # overrides (checked before returning an already-open handle, so a
            # mismatch fails here rather than later on a vector insert).
            self._enforce_explicit_overrides(safe, recorded, model, vector_dim, bit_width)
            open_model = recorded["model"]
            open_vector_dim = recorded["vector_dim"]
            open_bit_width = int(recorded["bit_width"])
        elif self.read_only:
            raise FileNotFoundError(
                f"space {name!r} does not exist in this collection (read-only): {self.path}"
            )
        else:
            open_model = "minilm" if model is _UNSET else model
            open_vector_dim = None if vector_dim is _UNSET else vector_dim
            open_bit_width = 4 if bit_width is _UNSET else int(bit_width)

        if safe in self._open:
            return self._open[safe]

        db = LodeDB(
            path=self.path / safe,
            # A vector-only space ignores model; "minilm" is a harmless placeholder.
            model=open_model or "minilm",
            vector_dim=open_vector_dim,
            bit_width=open_bit_width,
            read_only=self.read_only,
            **kwargs,
        )
        self._open[safe] = db
        if recorded is None and not self.read_only:
            self._spaces_config[safe] = {
                # A vector-only space ignores the preset model, so record it as null.
                "model": None if open_vector_dim is not None else open_model,
                "vector_dim": open_vector_dim,
                "bit_width": open_bit_width,
            }
            self._write_manifest()
        return db

    def spaces(self) -> list[str]:
        """Returns the names of all spaces recorded in the collection, sorted."""

        return sorted(self._spaces_config)

    def space_config(self, name: str) -> dict[str, Any] | None:
        """Returns the recorded ``(model, vector_dim, bit_width)`` for a space, or ``None``."""

        recorded = self._spaces_config.get(self._validate_name(name))
        return dict(recorded) if recorded is not None else None

    def close(self) -> None:
        """Closes every open space handle; on-disk state stays durable."""

        for db in self._open.values():
            db.close()
        self._open.clear()

    def __enter__(self) -> LodeCollection:
        """Enters a context manager; spaces are opened lazily via :meth:`space`."""

        return self

    def __exit__(self, *exc: object) -> None:
        """Exits the context manager, closing any open spaces."""

        self.close()

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _validate_name(name: str) -> str:
        """Validates a space name as a safe, single-segment directory name."""

        if not isinstance(name, str) or not _SPACE_NAME_RE.match(name):
            raise ValueError(
                "space name must match [A-Za-z0-9][A-Za-z0-9_-]* "
                f"(letters, digits, '-', '_'; no path separators); got {name!r}"
            )
        return name

    @staticmethod
    def _enforce_explicit_overrides(
        name: str,
        recorded: dict[str, Any],
        model: Any,
        vector_dim: Any,
        bit_width: Any,
    ) -> None:
        """Raises if an explicitly-passed config value conflicts with the recorded space.

        Arguments left as ``_UNSET`` are taken from the manifest, so a recorded space
        reopens cleanly; only a value the caller actually passed is checked, so a
        genuine mismatch (different dim / model / bit_width) still fails fast.
        """

        if model is not _UNSET and model != recorded["model"]:
            raise ValueError(
                f"space {name!r} was created with model={recorded['model']!r}; "
                f"reopen requested model={model!r}"
            )
        if vector_dim is not _UNSET and vector_dim != recorded["vector_dim"]:
            raise ValueError(
                f"space {name!r} was created with vector_dim={recorded['vector_dim']!r}; "
                f"reopen requested vector_dim={vector_dim!r}"
            )
        if bit_width is not _UNSET and int(bit_width) != recorded["bit_width"]:
            raise ValueError(
                f"space {name!r} was created with bit_width={recorded['bit_width']!r}; "
                f"reopen requested bit_width={int(bit_width)!r}"
            )

    def _manifest_path(self) -> Path:
        """Returns the path to the collection manifest file."""

        return self.path / _MANIFEST_NAME

    def _load_manifest(self) -> dict[str, dict[str, Any]]:
        """Loads the space registry from the manifest, or returns an empty one."""

        manifest_path = self._manifest_path()
        if not manifest_path.is_file():
            return {}
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"corrupt collection manifest: {manifest_path}") from exc
        spaces = payload.get("spaces", {}) if isinstance(payload, dict) else {}
        if not isinstance(spaces, dict):
            raise ValueError(f"corrupt collection manifest (spaces): {manifest_path}")
        return {str(name): dict(config) for name, config in spaces.items()}

    def _write_manifest(self) -> None:
        """Atomically publishes the space registry, merging any concurrent spaces.

        Manifest mutations are serialized on a collection-root advisory lock, and
        the on-disk manifest is re-read under it (read-merge-write), so a space that
        another handle created since this one loaded is preserved rather than lost
        to last-writer-wins. (Within one process, spaces still follow LodeDB's
        single-writer-per-path model; this only guards the shared manifest file.)
        """

        lock = WriterLock(self.path)
        lock.acquire(timeout=lodedb_lock_timeout_from_env())
        try:
            merged = self._load_manifest()
            merged.update(self._spaces_config)
            self._spaces_config = merged
            manifest_path = self._manifest_path()
            body = json.dumps(
                {"version": _MANIFEST_VERSION, "spaces": self._spaces_config},
                indent=2,
                sort_keys=True,
            )
            fd, tmp_name = tempfile.mkstemp(dir=self.path, prefix=".collection-", suffix=".tmp")
            try:
                with open(fd, "w", encoding="utf-8") as handle:
                    handle.write(body)
                durable_replace(tmp_name, manifest_path, fsync=False)
            finally:
                Path(tmp_name).unlink(missing_ok=True)
        finally:
            lock.release()
