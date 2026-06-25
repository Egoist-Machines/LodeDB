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

import contextlib
import json
import re
import tempfile
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durability_from_env, durable_replace, normalize_durability
from lodedb.engine._filelock import WriterLock, lodedb_lock_timeout_from_env
from lodedb.local.db import LodeDB
from lodedb.local.presets import resolve_preset

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

    def __init__(
        self,
        path: str | Path,
        *,
        read_only: bool = False,
        durability: str | None = None,
    ) -> None:
        """Opens (or creates) a collection root and loads its space manifest.

        ``read_only=True`` opens every space read-only and never creates the root
        or a missing space (see :meth:`space`); the root must already exist.
        ``durability`` (``"fast"`` default, or ``"fsync"``) governs the collection
        manifest's commit and is the default for the spaces it opens, exactly like
        :class:`LodeDB`; unset reads ``LODEDB_DURABILITY``. With ``"fsync"`` the
        manifest is fsynced into place so a durable space is never left invisible
        after a power loss.
        """

        self.path = Path(path)
        self.read_only = bool(read_only)
        # Same resolution as LodeDB: explicit arg wins, else LODEDB_DURABILITY, else fast.
        self._durability_arg = durability
        self._fsync = (
            durability_from_env() if durability is None else normalize_durability(durability)
        )
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
        embedder: Any = _UNSET,
        store_text: bool = _UNSET,
        index_text: bool = _UNSET,
        **kwargs: Any,
    ) -> LodeDB:
        """Opens (or creates) the named space and returns its :class:`LodeDB` handle.

        The space lives at ``<root>/<name>/``. Each space has a *kind*, fixed at
        creation and recorded in the manifest:

        - **preset** (default): ``model=`` a preset (``"minilm"``/``"bge"``/``"clip"``).
        - **vector**: ``vector_dim=`` a bring-your-own-vectors index (no model).
        - **custom**: ``embedder=`` a custom ``EngineEmbeddingBackend``
          (its ``required_model_name`` is recorded as the space's identity).

        The collection **owns the privacy/indexing flags** ``store_text`` and
        ``index_text``: their value at creation is recorded in the manifest and
        re-applied on every reopen, so ``col.space("name")`` restores the exact
        configuration a space was created with (a ``store_text=False`` space never
        silently flips back to retaining raw text). **Reopening uses the recorded
        config**: a preset or vector space reopens with no config args; a *custom*
        space must be reopened with a matching ``embedder=`` (the collection cannot
        persist a backend object); the recorded identity is re-enforced on open.
        Passing a config value that conflicts with the recorded kind/config raises
        :class:`ValueError`. Other keyword arguments pass through to :class:`LodeDB`;
        the same handle is returned for repeated calls in one process.
        """

        safe = self._validate_name(name)
        recorded = self._spaces_config.get(safe)
        passed_embedder = None if embedder is _UNSET else embedder

        if recorded is not None:
            open_kwargs = self._reopen_space(
                safe, recorded, model, vector_dim, bit_width, passed_embedder,
                store_text, index_text,
            )
            new_config = None
        elif self.read_only:
            raise FileNotFoundError(
                f"space {name!r} does not exist in this collection (read-only): {self.path}"
            )
        else:
            open_kwargs, new_config = self._new_space(
                model, vector_dim, bit_width, passed_embedder, store_text, index_text
            )

        if safe in self._open:
            return self._open[safe]

        # Spaces inherit the collection's durability unless the caller overrides it.
        kwargs.setdefault("durability", self._durability_arg)
        db = LodeDB(path=self.path / safe, read_only=self.read_only, **open_kwargs, **kwargs)
        if new_config is not None and not self.read_only:
            # Publish to the manifest before caching the handle. If the publish
            # fails, the space was never registered, so close it (releasing its
            # writer lock) and restore the registry rather than leaving an
            # unpublished, locked space behind.
            previous = dict(self._spaces_config)
            self._spaces_config[safe] = new_config
            try:
                self._write_manifest()
            except Exception as exc:
                self._spaces_config = previous
                with contextlib.suppress(Exception):
                    db.close()
                raise RuntimeError(
                    f"failed to publish collection manifest for space {name!r}; "
                    "rolled back (the space was not registered and its lock was released)"
                ) from exc
        self._open[safe] = db
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
    def _privacy_flags(store_text: Any, index_text: Any) -> dict[str, bool]:
        """Resolves the collection-owned privacy/indexing flags (LodeDB defaults)."""

        return {
            "store_text": True if store_text is _UNSET else bool(store_text),
            "index_text": False if index_text is _UNSET else bool(index_text),
        }

    @staticmethod
    def _new_space(
        model: Any,
        vector_dim: Any,
        bit_width: Any,
        embedder: Any,
        store_text: Any,
        index_text: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns (LodeDB kwargs, manifest entry) for a not-yet-recorded space.

        The space's kind is fixed here: ``embedder=`` makes a custom space (its
        ``required_model_name`` is recorded as the identity), ``vector_dim=`` a
        vector space, otherwise a preset space. The privacy/indexing flags
        (``store_text``/``index_text``) are recorded so reopen restores them.
        """

        explicit_bw = None if bit_width is _UNSET else int(bit_width)
        if explicit_bw is not None and explicit_bw not in {2, 4}:
            raise ValueError(f"bit_width must be 2 or 4, got {bit_width!r}")
        privacy = LodeCollection._privacy_flags(store_text, index_text)
        if embedder is not None:
            identity = getattr(embedder, "required_model_name", None)
            if not identity:
                raise ValueError(
                    "embedder used as a collection space must declare a non-empty "
                    "required_model_name (recorded as the space's identity)"
                )
            bw = 4 if explicit_bw is None else explicit_bw
            config = {"kind": "custom", "model_identity": identity, "bit_width": bw, **privacy}
            return {"embedder": embedder, "bit_width": bw, **privacy}, config
        if vector_dim is not _UNSET and vector_dim is not None:
            bw = 4 if explicit_bw is None else explicit_bw
            config = {"kind": "vector", "vector_dim": vector_dim, "bit_width": bw, **privacy}
            return {"vector_dim": vector_dim, "bit_width": bw, **privacy}, config
        # Preset space: its width is fixed by the preset, so record the *effective*
        # width (never a caller value that would not take effect) and let LodeDB use
        # it (no bit_width passed). An explicit, conflicting width is rejected.
        resolved_model = "minilm" if model is _UNSET else model
        effective_bw = resolve_preset(resolved_model).turbovec_bit_width
        if explicit_bw is not None and explicit_bw != effective_bw:
            raise ValueError(
                f"model {resolved_model!r} is a {effective_bw}-bit preset; "
                "bit_width is fixed by the preset"
            )
        config = {"kind": "preset", "model": resolved_model, "bit_width": effective_bw, **privacy}
        return {"model": resolved_model, **privacy}, config

    def _reopen_space(
        self,
        name: str,
        recorded: dict[str, Any],
        model: Any,
        vector_dim: Any,
        bit_width: Any,
        embedder: Any,
        store_text: Any,
        index_text: Any,
    ) -> dict[str, Any]:
        """Returns the LodeDB kwargs to reopen a recorded space, enforcing its kind.

        Config args left as ``_UNSET`` are taken from the manifest; an explicit value
        that conflicts with the recorded kind/config raises :class:`ValueError`. A
        custom space must be reopened with a matching ``embedder=`` (its identity is
        re-enforced when the index opens). The recorded privacy/indexing flags are
        re-applied, so a ``store_text=False`` space never silently flips back on.
        """

        kind = recorded.get("kind", "preset")
        if bit_width is not _UNSET and int(bit_width) != recorded["bit_width"]:
            raise ValueError(
                f"space {name!r} was created with bit_width={recorded['bit_width']!r}; "
                f"reopen requested bit_width={int(bit_width)!r}"
            )
        self._check_flag(name, "store_text", recorded, store_text, True)
        self._check_flag(name, "index_text", recorded, index_text, False)
        privacy = {
            "store_text": recorded.get("store_text", True),
            "index_text": recorded.get("index_text", False),
        }
        if kind == "custom":
            if embedder is None:
                raise ValueError(
                    f"space {name!r} is a custom-embedder space (identity "
                    f"{recorded.get('model_identity')!r}); reopen it with a matching embedder="
                )
            if model is not _UNSET or (vector_dim is not _UNSET and vector_dim is not None):
                raise ValueError(
                    f"space {name!r} is a custom-embedder space; pass embedder=, "
                    "not model/vector_dim"
                )
            return {"embedder": embedder, "bit_width": recorded["bit_width"], **privacy}
        if embedder is not None:
            raise ValueError(
                f"space {name!r} is a {kind} space, not custom; do not pass embedder="
            )
        if kind == "vector":
            if vector_dim is not _UNSET and vector_dim != recorded["vector_dim"]:
                raise ValueError(
                    f"space {name!r} was created with vector_dim={recorded['vector_dim']!r}; "
                    f"reopen requested vector_dim={vector_dim!r}"
                )
            if model is not _UNSET:
                raise ValueError(f"space {name!r} is a vector-only space; do not pass model=")
            return {
                "vector_dim": recorded["vector_dim"],
                "bit_width": recorded["bit_width"],
                **privacy,
            }
        if model is not _UNSET and model != recorded["model"]:
            raise ValueError(
                f"space {name!r} was created with model={recorded['model']!r}; "
                f"reopen requested model={model!r}"
            )
        if vector_dim is not _UNSET and vector_dim is not None:
            raise ValueError(f"space {name!r} is a preset space; do not pass vector_dim=")
        # A preset's width is fixed by the preset, so don't pass bit_width to LodeDB.
        return {"model": recorded["model"], **privacy}

    @staticmethod
    def _check_flag(
        name: str, field: str, recorded: dict[str, Any], value: Any, default: bool
    ) -> None:
        """Raises if an explicit privacy/indexing flag conflicts with the recorded one."""

        if value is _UNSET:
            return
        recorded_value = recorded.get(field, default)
        if bool(value) != recorded_value:
            raise ValueError(
                f"space {name!r} was created with {field}={recorded_value!r}; "
                f"reopen requested {field}={bool(value)!r}"
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
                # fsync the manifest into place under durability="fsync", so a durable
                # space is never left invisible after a power loss.
                durable_replace(tmp_name, manifest_path, fsync=self._fsync)
            finally:
                Path(tmp_name).unlink(missing_ok=True)
        finally:
            lock.release()
