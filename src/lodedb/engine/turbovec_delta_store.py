"""Segmented base+delta persistence for direct TurboVec `.tvim` snapshots.

Full `.tvim` rewrites cost O(corpus) per mutation batch (~400-850 MB at
1M chunks). This store keeps the existing `<index_key>.tvim` file as the
base segment and appends small `.tvd` delta segments under
`<index_key>.tvim-delta/`, so persisting a mutation batch costs O(changed
rows): each delta journals the exact packed codes and scales exported from
the live index (vendored `export_encoded` local API) plus removed stable
ids. Loads replay deltas in order through `add_encoded`/`remove_many`, and
a compaction policy folds deltas back into a fresh base.

Every segment is checksumed and records the index's calibration
fingerprint; loads fail closed on checksum, sequence, fingerprint, or
count mismatches so a corrupt or mismatched delta can never serve quietly.
The base `.tvim` stays byte-identical to upstream format v3.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lodedb.engine._atomic_io import durable_replace

TVIM_DELTA_DIR_SUFFIX = ".tvim-delta"
TVIM_DELTA_MANIFEST_NAME = "manifest.json"
TVIM_DELTA_SEGMENT_SUFFIX = ".tvd"
TVIM_DELTA_MAGIC = b"EETVD001"
TVIM_DELTA_SCHEMA_VERSION = 1
_HEADER_LENGTH = struct.Struct("<Q")

DEFAULT_MAX_DELTA_ROW_FRACTION = 0.25
DEFAULT_MAX_DELTA_SEGMENTS = 64


def turbovec_delta_api_available(index: Any) -> bool:
    """Returns whether the loaded TurboVec build exposes the local delta APIs."""

    return all(
        hasattr(index, name)
        for name in ("export_encoded", "add_encoded", "remove_many", "calibration_fingerprint")
    )


@dataclass(frozen=True)
class TvimDeltaWrite:
    """Reports one persisted segment's name, payload bytes, and elapsed time."""

    file_name: str
    file_bytes: int
    write_ms: float
    kind: str


class TvimDeltaStore:
    """Manages the base `.tvim` plus ordered `.tvd` delta segments."""

    def __init__(self, base_path: str | Path, *, fsync: bool = False) -> None:
        """Binds the store to the base `.tvim` path; the delta dir sits beside it.

        ``fsync`` makes each published base/segment/manifest power-loss durable
        (the engine's ``durability="fsync"`` mode); the default keeps the fast
        atomic-rename path.
        """

        self.base_path = Path(base_path)
        self._fsync = bool(fsync)
        self.delta_dir = self.base_path.with_name(self.base_path.name + TVIM_DELTA_DIR_SUFFIX)

    @property
    def manifest_path(self) -> Path:
        """Returns the manifest path that anchors delta validity."""

        return self.delta_dir / TVIM_DELTA_MANIFEST_NAME

    def has_manifest(self) -> bool:
        """Returns whether a delta manifest exists for this base path."""

        return self.manifest_path.is_file()

    def persist_base(self, index: Any) -> TvimDeltaWrite:
        """Writes a fresh full base `.tvim` and clears the delta backlog."""

        started = time.perf_counter()
        self.base_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.base_path.with_name(self.base_path.name + ".tmp")
        index.write(str(temporary))
        durable_replace(temporary, self.base_path, fsync=self._fsync)
        file_bytes = self.base_path.stat().st_size
        previous = self._read_manifest_optional()
        next_seq = int(previous.get("next_seq", 0)) if previous else 0
        self.delta_dir.mkdir(parents=True, exist_ok=True)
        stale_segments = self._manifest_segment_names(previous)
        self._write_manifest(
            {
                "schema_version": TVIM_DELTA_SCHEMA_VERSION,
                "base": {
                    "file_name": self.base_path.name,
                    "sha256": _sha256_file(self.base_path),
                    "file_bytes": int(file_bytes),
                    "rows": int(len(index)),
                    "calibration_fingerprint": int(index.calibration_fingerprint())
                    if hasattr(index, "calibration_fingerprint")
                    else 0,
                },
                "deltas": [],
                "next_seq": next_seq + 1,
            }
        )
        for stale in stale_segments:
            (self.delta_dir / stale).unlink(missing_ok=True)
        return TvimDeltaWrite(
            file_name=self.base_path.name,
            file_bytes=int(file_bytes),
            write_ms=float((time.perf_counter() - started) * 1000.0),
            kind="base",
        )

    def append_delta(
        self,
        index: Any,
        *,
        upsert_stable_ids: NDArray[np.uint64],
        removed_stable_ids: NDArray[np.uint64],
        generation: int,
    ) -> TvimDeltaWrite:
        """Journals one mutation batch's exact codes and removals as a segment."""

        started = time.perf_counter()
        manifest = self._read_manifest()
        sequence = int(manifest.get("next_seq", 0))
        if upsert_stable_ids.shape[0]:
            codes, scales = index.export_encoded(upsert_stable_ids)
        else:
            bytes_per_vector = int(index.bytes_per_vector() or 0)
            codes = np.zeros((0, bytes_per_vector), dtype=np.uint8)
            scales = np.zeros(0, dtype=np.float32)
        arrays: list[tuple[str, NDArray[Any]]] = [
            ("upsert_stable_ids", np.ascontiguousarray(upsert_stable_ids, dtype=np.uint64)),
            ("upsert_codes", np.ascontiguousarray(codes, dtype=np.uint8)),
            ("upsert_scales", np.ascontiguousarray(scales, dtype=np.float32)),
            ("removed_stable_ids", np.ascontiguousarray(removed_stable_ids, dtype=np.uint64)),
        ]
        header = {
            "schema_version": TVIM_DELTA_SCHEMA_VERSION,
            "kind": "delta",
            "seq": sequence,
            "generation_after": int(generation),
            "calibration_fingerprint": int(index.calibration_fingerprint()),
            "rows_after": int(len(index)),
            "arrays": [
                {
                    "name": name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "nbytes": int(array.nbytes),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
                for name, array in arrays
            ],
        }
        header_blob = json.dumps(header, sort_keys=True).encode("utf-8")
        self.delta_dir.mkdir(parents=True, exist_ok=True)
        segment_name = f"delta-{sequence:08d}{TVIM_DELTA_SEGMENT_SUFFIX}"
        segment_path = self.delta_dir / segment_name
        temporary = segment_path.with_name(segment_path.name + ".tmp")
        with temporary.open("wb") as handle:
            handle.write(TVIM_DELTA_MAGIC)
            handle.write(_HEADER_LENGTH.pack(len(header_blob)))
            handle.write(header_blob)
            for _, array in arrays:
                handle.write(array.tobytes())
        durable_replace(temporary, segment_path, fsync=self._fsync)
        file_bytes = segment_path.stat().st_size
        manifest["deltas"] = list(manifest.get("deltas", [])) + [
            {
                "file_name": segment_name,
                "sha256": _sha256_file(segment_path),
                "file_bytes": int(file_bytes),
                "seq": sequence,
                "upsert_rows": int(upsert_stable_ids.shape[0]),
                "removed_rows": int(removed_stable_ids.shape[0]),
            }
        ]
        manifest["next_seq"] = sequence + 1
        self._write_manifest(manifest)
        return TvimDeltaWrite(
            file_name=segment_name,
            file_bytes=int(file_bytes),
            write_ms=float((time.perf_counter() - started) * 1000.0),
            kind="delta",
        )

    def has_pending_segments(self, *, manifest: dict[str, Any] | None = None) -> bool:
        """Returns whether the manifest references delta segments to replay.

        A base-only manifest (written by every auto-policy base rewrite)
        needs no replay APIs at load time, so unpatched backends can still
        restart from it after only base persists.
        """

        manifest = self._read_manifest_optional() if manifest is None else manifest
        return bool(manifest and manifest.get("deltas"))

    def should_compact(
        self,
        *,
        max_delta_row_fraction: float = DEFAULT_MAX_DELTA_ROW_FRACTION,
        max_delta_segments: int = DEFAULT_MAX_DELTA_SEGMENTS,
    ) -> bool:
        """Returns whether the delta backlog warrants folding into a new base."""

        manifest = self._read_manifest_optional()
        if not manifest:
            return False
        deltas = manifest.get("deltas", [])
        if not deltas:
            return False
        if len(deltas) >= max_delta_segments:
            return True
        delta_rows = sum(
            int(delta.get("upsert_rows", 0)) + int(delta.get("removed_rows", 0))
            for delta in deltas
        )
        base_rows = max(int(manifest.get("base", {}).get("rows", 1)), 1)
        return delta_rows >= base_rows * max_delta_row_fraction

    def replay_onto(
        self, index: Any, *, manifest: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """Replays manifest deltas onto a freshly loaded base index, failing closed.

        ``manifest`` overrides the on-disk manifest so a load can be driven by
        the committed copy embedded in the root manifest, ignoring any
        uncommitted segment a crashed writer left behind.
        """

        started = time.perf_counter()
        manifest = self._read_manifest() if manifest is None else manifest
        base_entry = manifest.get("base", {})
        recorded_fingerprint = int(base_entry.get("calibration_fingerprint", 0))
        if recorded_fingerprint and hasattr(index, "calibration_fingerprint"):
            actual = int(index.calibration_fingerprint())
            if actual != recorded_fingerprint:
                raise RuntimeError(
                    "TurboVec delta replay rejected: base calibration fingerprint mismatch"
                )
        previous_seq = -1
        replayed_upserts = 0
        replayed_removes = 0
        for delta in manifest.get("deltas", []):
            sequence = int(delta.get("seq", -1))
            if sequence <= previous_seq:
                raise RuntimeError("TurboVec delta manifest has out-of-order segments")
            previous_seq = sequence
            payload = _read_delta_segment(self._validated_segment_path(delta))
            header = payload["header"]
            if int(header.get("calibration_fingerprint", -1)) != int(
                index.calibration_fingerprint()
            ):
                raise RuntimeError(
                    "TurboVec delta replay rejected: segment calibration fingerprint mismatch"
                )
            removed_ids = payload["removed_stable_ids"]
            if removed_ids.shape[0]:
                removed = int(index.remove_many(removed_ids))
                if removed != removed_ids.shape[0]:
                    raise RuntimeError(
                        "TurboVec delta replay rejected: removed-id count mismatch"
                    )
                replayed_removes += removed
            upsert_ids = payload["upsert_stable_ids"]
            if upsert_ids.shape[0]:
                index.add_encoded(
                    upsert_ids,
                    payload["upsert_codes"],
                    payload["upsert_scales"],
                )
                replayed_upserts += int(upsert_ids.shape[0])
            expected_rows = int(header.get("rows_after", -1))
            if expected_rows >= 0 and len(index) != expected_rows:
                raise RuntimeError("TurboVec delta replay rejected: row count mismatch")
        return {
            "replay_ms": float((time.perf_counter() - started) * 1000.0),
            "delta_segments_replayed": float(len(manifest.get("deltas", []))),
            "replayed_upsert_rows": float(replayed_upserts),
            "replayed_removed_rows": float(replayed_removes),
        }

    def validate_base_checksum(self, *, manifest: dict[str, Any] | None = None) -> None:
        """Fails closed when the base `.tvim` does not match the manifest digest."""

        manifest = self._read_manifest() if manifest is None else manifest
        base_entry = manifest.get("base", {})
        recorded = str(base_entry.get("sha256", ""))
        if recorded and _sha256_file(self.base_path) != recorded:
            raise RuntimeError("TurboVec base snapshot failed manifest checksum")

    def storage_file_bytes(self) -> dict[str, float]:
        """Returns manifest/base/delta byte accounting for telemetry."""

        manifest = self._read_manifest_optional() or {}
        base_bytes = float(manifest.get("base", {}).get("file_bytes", 0))
        deltas = manifest.get("deltas", [])
        delta_bytes = float(sum(int(delta.get("file_bytes", 0)) for delta in deltas))
        manifest_bytes = (
            float(self.manifest_path.stat().st_size) if self.manifest_path.is_file() else 0.0
        )
        return {
            "tvim_base_bytes": base_bytes,
            "tvim_delta_bytes": delta_bytes,
            "tvim_delta_segments": float(len(deltas)),
            "tvim_manifest_bytes": manifest_bytes,
        }

    def remove_all(self) -> None:
        """Removes the manifest and all delta segments (e.g. empty index)."""

        if not self.delta_dir.is_dir():
            return
        for path in self.delta_dir.glob(f"*{TVIM_DELTA_SEGMENT_SUFFIX}"):
            path.unlink(missing_ok=True)
        self.manifest_path.unlink(missing_ok=True)

    def current_manifest(self) -> dict[str, Any] | None:
        """Returns the on-disk manifest (for embedding in the root commit manifest)."""

        return self._read_manifest_optional()

    def restore_manifest(self, manifest: dict[str, Any]) -> None:
        """Heals the on-disk manifest to a committed snapshot, dropping orphans.

        Writer recovery: a crashed commit may have appended a segment and bumped
        the on-disk manifest past the committed root. Rewriting the manifest to
        the committed copy and deleting any segment file it does not reference
        rolls this store back to the last good generation.
        """

        self.delta_dir.mkdir(parents=True, exist_ok=True)
        referenced = set(self._manifest_segment_names(manifest))
        self._write_manifest(manifest)
        for path in self.delta_dir.glob(f"*{TVIM_DELTA_SEGMENT_SUFFIX}"):
            if path.name not in referenced:
                path.unlink(missing_ok=True)

    def _validated_segment_path(self, entry: dict[str, Any]) -> Path:
        """Returns a manifest segment's path after existence and checksum checks."""

        file_name = str(entry.get("file_name", ""))
        path = self.delta_dir / file_name
        if not file_name or not path.is_file():
            raise RuntimeError(f"TurboVec delta segment is missing: {file_name}")
        if _sha256_file(path) != str(entry.get("sha256", "")):
            raise RuntimeError(f"TurboVec delta segment failed checksum: {file_name}")
        return path

    def _read_manifest(self) -> dict[str, Any]:
        """Reads and validates the manifest, failing closed when absent."""

        manifest = self._read_manifest_optional()
        if manifest is None:
            raise RuntimeError("TurboVec delta manifest is missing")
        return manifest

    def _read_manifest_optional(self) -> dict[str, Any] | None:
        """Reads the manifest when present, validating its schema version."""

        if not self.manifest_path.is_file():
            return None
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema_version", -1)) != TVIM_DELTA_SCHEMA_VERSION:
            raise RuntimeError("unsupported TurboVec delta manifest schema version")
        return manifest

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        """Atomically replaces the manifest after its segments are durable."""

        temporary = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        durable_replace(temporary, self.manifest_path, fsync=self._fsync)

    def _manifest_segment_names(self, manifest: dict[str, Any] | None) -> list[str]:
        """Lists delta segment file names referenced by a manifest."""

        if not manifest:
            return []
        return [
            str(delta["file_name"])
            for delta in manifest.get("deltas", [])
            if delta.get("file_name")
        ]


def _read_delta_segment(path: Path) -> dict[str, Any]:
    """Parses and checksum-validates one `.tvd` delta segment file."""

    data = path.read_bytes()
    prefix = len(TVIM_DELTA_MAGIC) + _HEADER_LENGTH.size
    if len(data) < prefix or data[: len(TVIM_DELTA_MAGIC)] != TVIM_DELTA_MAGIC:
        raise RuntimeError(f"not a TurboVec delta segment: {path.name}")
    header_length = _HEADER_LENGTH.unpack_from(data, len(TVIM_DELTA_MAGIC))[0]
    header_stop = prefix + header_length
    if len(data) < header_stop:
        raise RuntimeError(f"TurboVec delta segment header is truncated: {path.name}")
    header = json.loads(data[prefix:header_stop].decode("utf-8"))
    if int(header.get("schema_version", -1)) != TVIM_DELTA_SCHEMA_VERSION:
        raise RuntimeError("unsupported TurboVec delta segment schema version")
    arrays: dict[str, NDArray[Any]] = {}
    offset = header_stop
    for spec in header["arrays"]:
        nbytes = int(spec["nbytes"])
        blob = data[offset : offset + nbytes]
        if len(blob) != nbytes:
            raise RuntimeError(f"TurboVec delta array {spec['name']} is truncated")
        if hashlib.sha256(blob).hexdigest() != spec["sha256"]:
            raise RuntimeError(f"TurboVec delta array {spec['name']} failed checksum")
        arrays[str(spec["name"])] = (
            np.frombuffer(blob, dtype=np.dtype(spec["dtype"]))
            .reshape(tuple(int(size) for size in spec["shape"]))
            .copy()
        )
        offset += nbytes
    required = {"upsert_stable_ids", "upsert_codes", "upsert_scales", "removed_stable_ids"}
    missing = required - set(arrays)
    if missing:
        raise RuntimeError(f"TurboVec delta segment is missing arrays: {sorted(missing)}")
    return {"header": header, **arrays}


def _sha256_file(path: Path) -> str:
    """Returns the sha256 hex digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
