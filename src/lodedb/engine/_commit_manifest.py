"""Per-index atomic commit manifest over generation-addressed artifacts.

Stage 1 made each *file* publish atomically (temp + ``os.replace``) and gave the
single-writer guarantee, but a logical commit spans several files — the JSON
state base, its ``.jsd`` journal, the ``.tvim`` vector base, its ``.tvd``
journal — published in sequence. A crash (or a lock-free reader) between those
publishes could observe a torn cross-file set: the existing loader *detected*
that and failed closed, but could not *recover* to the last good state.

This module closes that gap. Every index keeps its durable artifacts under a
per-index directory ``<key>.gen/`` named by the **base epoch** (the generation
at which a full base was last written):

    <key>.gen/g<epoch>.json            JSON state base
    <key>.gen/g<epoch>.json.json-delta/  its ``.jsd`` document journal
    <key>.gen/g<epoch>.tvim            TurboVec vector base
    <key>.gen/g<epoch>.tvim.tvim-delta/  its ``.tvd`` vector journal
    <key>.gen/g<epoch>.tvtext          opt-in raw-text base (full id->text map)
    <key>.gen/g<epoch>.tvtext.tvtext-delta/  its ``.txd`` raw-text journal

and a single top-level pointer:

    <key>.commit.json                  the root manifest (the commit point)

The root manifest embeds the committed state of both per-store manifests plus
counts and the live base epoch. A commit writes any new epoch artifacts first,
then atomically swaps ``<key>.commit.json`` — that swap is the **only** thing
that commits a generation. Because bases are epoch-addressed they are never
overwritten in place, so a crashed commit leaves the previous epoch fully
intact; recovery just re-points at it and deletes the unreferenced artifacts.
A lock-free reader reads the root manifest once and loads exactly the artifacts
it names — a consistent generation, never a torn cross-file mix.

The opt-in raw-text store (``store_text=True``) is part of this atomic set: it
has its own base + ``.txd`` delta journal under the same ``<key>.gen/`` epoch
and its manifest is embedded in the root, so raw text (visible via the public
``get`` API) commits and rolls back with exactly the generation the root names —
never leaking an uncommitted overwrite. It remains a separate store: no
telemetry/redacted/audit path reads it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durable_replace

COMMIT_MANIFEST_SUFFIX = ".commit.json"
GEN_DIR_SUFFIX = ".gen"
COMMIT_MANIFEST_SCHEMA_VERSION = 1

# How many superseded base epochs to retain after a successful commit. Keeping a
# few old epochs means a lock-free reader holding a recently-committed manifest
# still finds its artifacts even though a new base has since been written; the
# writer GCs anything older under the single-writer lock.
DEFAULT_EPOCHS_RETAINED = 3


def commit_manifest_path(persistence_dir: str | Path, index_key: str) -> Path:
    """Returns the top-level root-manifest path for one index key."""

    return Path(persistence_dir) / f"{index_key}{COMMIT_MANIFEST_SUFFIX}"


def generation_dir(persistence_dir: str | Path, index_key: str) -> Path:
    """Returns the per-index directory holding epoch-addressed artifacts."""

    return Path(persistence_dir) / f"{index_key}{GEN_DIR_SUFFIX}"


def base_json_path(persistence_dir: str | Path, index_key: str, epoch: int) -> Path:
    """Returns the JSON state base path for one index key and base epoch."""

    return generation_dir(persistence_dir, index_key) / f"g{int(epoch)}.json"


def base_tvim_path(persistence_dir: str | Path, index_key: str, epoch: int) -> Path:
    """Returns the TurboVec vector base path for one index key and base epoch."""

    return generation_dir(persistence_dir, index_key) / f"g{int(epoch)}.tvim"


def base_tvtext_path(persistence_dir: str | Path, index_key: str, epoch: int) -> Path:
    """Returns the raw-text base path for one index key and base epoch."""

    return generation_dir(persistence_dir, index_key) / f"g{int(epoch)}.tvtext"


def is_commit_manifest_name(name: str) -> bool:
    """Returns whether a top-level file name is a root commit manifest."""

    return name.endswith(COMMIT_MANIFEST_SUFFIX)


def list_base_epochs(persistence_dir: str | Path, index_key: str) -> set[int]:
    """Returns the base epochs with a JSON base present under ``<key>.gen/``."""

    directory = generation_dir(persistence_dir, index_key)
    if not directory.is_dir():
        return set()
    epochs: set[int] = set()
    for path in directory.glob("g*.json"):
        stem = path.stem  # e.g. "g7"
        if stem.startswith("g") and stem[1:].isdigit():
            epochs.add(int(stem[1:]))
    return epochs


def read_commit_manifest(path: str | Path) -> dict[str, Any] | None:
    """Reads and validates a root commit manifest, or ``None`` when absent.

    Fails closed on an unsupported schema version or a body-checksum mismatch so
    a truncated/garbled pointer can never silently select a wrong generation.
    """

    path = Path(path)
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if int(document.get("schema_version", -1)) != COMMIT_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("unsupported commit manifest schema version")
    body = document.get("body")
    recorded = str(document.get("body_sha256", ""))
    if not isinstance(body, dict) or not recorded:
        raise RuntimeError("commit manifest is missing its body or checksum")
    if _body_sha256(body) != recorded:
        raise RuntimeError("commit manifest failed body checksum")
    return body


def write_commit_manifest(path: str | Path, body: dict[str, Any], *, fsync: bool) -> None:
    """Atomically writes the root manifest; this swap is the commit point.

    The body is checksum-wrapped and the file is published via the durable
    temp + ``os.replace`` path, so the pointer flips all-or-nothing.

    The body embeds both per-store manifests and is serialized **exactly once**
    per commit: that single ``json.dumps`` is the dominant cost of the write
    (it dwarfs the file I/O), so the one serialization is reused for both the
    integrity checksum and the on-disk payload — the wrapper is assembled around
    it rather than re-dumping the body inside a dict.
    """

    path = Path(path)
    body_json = json.dumps(body, sort_keys=True)
    body_sha = hashlib.sha256(body_json.encode("utf-8")).hexdigest()
    document_json = (
        '{"body":'
        + body_json
        + ',"body_sha256":'
        + json.dumps(body_sha)
        + ',"schema_version":'
        + str(int(COMMIT_MANIFEST_SCHEMA_VERSION))
        + "}"
    )
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(document_json, encoding="utf-8")
    # durable_replace already fsyncs the destination directory in fsync mode.
    durable_replace(temporary, path, fsync=fsync)


def build_commit_body(
    *,
    index_key: str,
    generation: int,
    base_epoch: int,
    document_count: int,
    chunk_count: int,
    json_manifest: dict[str, Any] | None,
    tvim_manifest: dict[str, Any] | None,
    tvim_present: bool,
    tvtext_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    """Assembles a root-manifest body capturing one consistent committed generation.

    ``tvtext_manifest`` is the raw-text journal manifest (base + delta segments)
    for this committed generation, or ``None`` when raw-text storage is disabled
    or the index holds no text. It is pinned by the root exactly like the JSON
    and TurboVec manifests, so raw text commits and rolls back atomically.
    """

    return {
        "index_key": index_key,
        "generation": int(generation),
        "base_epoch": int(base_epoch),
        "document_count": int(document_count),
        "chunk_count": int(chunk_count),
        "json": json_manifest,
        "tvim": tvim_manifest,
        "tvim_present": bool(tvim_present),
        "tvtext": tvtext_manifest,
    }


def _body_sha256(body: dict[str, Any]) -> str:
    """Returns the canonical sha256 of a manifest body for integrity checks."""

    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
