"""Default single-writer write-ahead log for the LodeDB commit path.

The classic generation commit path publishes a new immutable generation on every mutation
(an O(changed) ``.jsd``/``.tvd`` delta append plus the atomic ``<key>.commit.json``
root-manifest swap — see :mod:`lodedb.engine._commit_manifest`). That gives
lock-free MVCC readers a torn-free snapshot, but it costs several ``os.replace``
publishes and as many sha256 passes per single ``add`` — overkill for the common
single-process, single-writer deployment that just wants a durable append.

This module backs the default writer mode (``commit_mode="wal"``).
A mutation appends **one** length-prefixed, CRC32-framed record to a single
append-only ``<key>.wal`` file (one ``write`` + an optional ``fsync``) and skips
the generation publish entirely. The in-memory index is already up to date when
the record is written (the engine syncs it before persisting), so the WAL only
has to make that mutation *recoverable*. Periodically — on a size/op threshold,
on an explicit ``persist()``, or on ``close()`` — the writer checkpoints by
folding the WAL's effect into a fresh generation through the ordinary commit path
and then truncating the WAL.

Records are **logical**: each frames the public engine mutation that produced it
(``upsert_documents`` / ``upsert_vectors`` / ``delete_documents``) with its exact
inputs. On open, the engine loads the last committed generation as usual and then
*replays* any WAL records on top by re-invoking those same engine verbs — so the
recovered state is reconstructed by the identical ingest/sync code that produced
it, never a parallel decoder that could drift. A record whose frame is torn (a
writer killed mid-append) fails the length/CRC check; replay stops there and
treats everything up to it as the durable WAL-committed state, so a crash can
never surface a half-written mutation or corrupt the committed generation.

Crash-atomicity of a WAL *append* rests on three things: the single-writer file
lock (so only one process ever appends), the length+CRC frame (so a torn tail is
detected, not parsed), and — in ``durability="fsync"`` mode — fsyncing the file
after the write (so the bytes reach stable storage before the call returns). The
checkpoint reuses the generation path's existing crash-atomicity: the new
generation commits via the atomic root-manifest swap *before* the WAL is
truncated, so a crash between the two replays a few already-applied records
(idempotent upserts/deletes) rather than losing them.

This file format is independent of the generation artifacts and is only ever read
or written by a writable handle using WAL mode; the generation/MVCC reader path
never looks at it.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import fsync_dir

WAL_SUFFIX = ".wal"
WAL_MAGIC = b"EELWAL01"
WAL_SCHEMA_VERSION = 1
# Each record on disk: 4-byte big-endian body length, then the body, then a
# 4-byte big-endian CRC32 of (length-prefix || body). Framing the checksum over
# the length too means a corrupt length cannot point past a short read undetected.
_LEN = struct.Struct(">I")
_CRC = struct.Struct(">I")
_HEADER = struct.Struct(">8sI")  # magic + schema version

# Default checkpoint thresholds. A checkpoint folds the WAL into a fresh
# generation and truncates it, bounding both replay time on the next open and the
# WAL file size. Tuned conservatively: large enough that a burst of single adds
# stays on the cheap append path, small enough that the WAL never dwarfs the base.
DEFAULT_CHECKPOINT_OPS = 512
DEFAULT_CHECKPOINT_BYTES = 64 * 1024 * 1024


def wal_path(persistence_dir: str | Path, index_key: str) -> Path:
    """Returns the append-only WAL path for one index key."""

    return Path(persistence_dir) / f"{index_key}{WAL_SUFFIX}"


@dataclass(frozen=True)
class WalAppend:
    """Reports one appended WAL record's on-disk size and the running op count."""

    record_bytes: int
    op_count: int


@dataclass(frozen=True)
class WalRecord:
    """One decoded, checksum-validated WAL record (a logical mutation)."""

    op: str
    payload: dict[str, Any]


class WalCorruptionError(RuntimeError):
    """Raised when an *interior* WAL record is malformed (not a torn tail).

    A torn trailing record (writer killed mid-append) is expected and is
    silently dropped by :meth:`WalStore.read_records`; this is raised only when a
    record that is *followed by more bytes* fails its frame/checksum, which means
    real corruption rather than an interrupted final append.
    """


class WalStore:
    """Append-only, single-writer WAL over one ``<key>.wal`` file.

    Not safe for concurrent writers: correctness rests on the engine's
    single-writer file lock, exactly like the delta stores. ``fsync`` mirrors the
    engine's ``durability="fsync"`` mode — when set, each append is flushed to
    stable storage before it returns and the directory is flushed when the file
    is first created, so an appended record survives a power loss.
    """

    __slots__ = ("path", "_fsync", "_op_count", "_byte_count")

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        """Binds the store to a WAL path and reads its current op/byte counts."""

        self.path = Path(path)
        self._fsync = bool(fsync)
        self._op_count = 0
        self._byte_count = 0
        self._scan_existing()

    @property
    def op_count(self) -> int:
        """Returns the number of intact records currently in the WAL."""

        return self._op_count

    @property
    def byte_count(self) -> int:
        """Returns the on-disk size (in bytes) of the WAL body region."""

        return self._byte_count

    def exists(self) -> bool:
        """Returns whether a non-empty WAL file is present for this key."""

        return self.path.is_file() and self.path.stat().st_size > _HEADER.size

    def append(self, op: str, payload: dict[str, Any]) -> WalAppend:
        """Appends one logical-mutation record, fsyncing it when in fsync mode.

        The record body is ``op`` + a JSON payload, length-prefixed and
        CRC32-framed. The file's magic/version header is written lazily on the
        first append so an empty WAL leaves no file behind.
        """

        body = _encode_body(op, payload)
        frame = _LEN.pack(len(body)) + body
        frame += _CRC.pack(zlib.crc32(frame) & 0xFFFFFFFF)
        first_write = not self.path.exists()
        if first_write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        # Open for append; the OS guarantees each write lands at end-of-file even
        # under O_APPEND, and a single write of one frame is what makes a torn
        # tail (rather than an interleaved one) the only crash outcome.
        with self.path.open("ab") as handle:
            if first_write:
                handle.write(_HEADER.pack(WAL_MAGIC, WAL_SCHEMA_VERSION))
            handle.write(frame)
            handle.flush()
            if self._fsync:
                os.fsync(handle.fileno())
        if first_write and self._fsync:
            # Make the file's directory entry durable the first time we create it.
            fsync_dir(self.path.parent)
        self._op_count += 1
        self._byte_count += len(frame)
        return WalAppend(record_bytes=len(frame), op_count=self._op_count)

    def read_records(self) -> list[WalRecord]:
        """Returns the intact records in append order, dropping a torn final frame.

        Fails closed (:class:`WalCorruptionError`) on a malformed *interior*
        record — one followed by further bytes — because that is real corruption
        rather than an interrupted last append. A bad/short *trailing* frame is
        treated as the crash point and silently dropped, so replay recovers
        exactly the records that were fully written before the crash.
        """

        if not self.path.is_file():
            return []
        data = self.path.read_bytes()
        if len(data) < _HEADER.size:
            return []
        magic, version = _HEADER.unpack_from(data, 0)
        if magic != WAL_MAGIC:
            raise WalCorruptionError("not a LodeDB WAL file (bad magic)")
        if int(version) != WAL_SCHEMA_VERSION:
            raise WalCorruptionError(f"unsupported WAL schema version: {version}")
        records: list[WalRecord] = []
        offset = _HEADER.size
        total = len(data)
        while offset < total:
            if offset + _LEN.size > total:
                break  # torn length prefix -> crash tail
            body_len = _LEN.unpack_from(data, offset)[0]
            frame_end = offset + _LEN.size + body_len
            crc_end = frame_end + _CRC.size
            if crc_end > total:
                break  # body/crc truncated -> crash tail
            frame = data[offset:frame_end]
            recorded_crc = _CRC.unpack_from(data, frame_end)[0]
            if (zlib.crc32(frame) & 0xFFFFFFFF) != recorded_crc:
                # A bad CRC at the very end is a torn tail; in the interior it is
                # real corruption we must not silently skip past.
                if crc_end == total:
                    break
                raise WalCorruptionError("WAL record failed CRC32 (interior corruption)")
            body = frame[_LEN.size :]
            records.append(_decode_body(body))
            offset = crc_end
        return records

    def truncate(self) -> None:
        """Removes the WAL after a checkpoint folds it into a committed generation.

        The new generation is committed (root-manifest swap) *before* this runs,
        so deleting the WAL only drops records that are now durable in the base —
        and a crash before the delete just replays those already-applied records.
        """

        self.path.unlink(missing_ok=True)
        if self._fsync:
            fsync_dir(self.path.parent)
        self._op_count = 0
        self._byte_count = 0

    def should_checkpoint(
        self,
        *,
        checkpoint_ops: int = DEFAULT_CHECKPOINT_OPS,
        checkpoint_bytes: int = DEFAULT_CHECKPOINT_BYTES,
    ) -> bool:
        """Returns whether the WAL backlog warrants folding into a new generation."""

        if self._op_count <= 0:
            return False
        return self._op_count >= checkpoint_ops or self._byte_count >= checkpoint_bytes

    def _scan_existing(self) -> None:
        """Counts the intact records/bytes already on disk (e.g. after a reopen)."""

        records = self.read_records()
        self._op_count = len(records)
        # Recompute the durable body-region size from the intact frames so a torn
        # tail is excluded from the byte accounting that drives checkpointing.
        self._byte_count = 0
        for record in records:
            body = _encode_body(record.op, record.payload)
            self._byte_count += _LEN.size + len(body) + _CRC.size


def _encode_body(op: str, payload: dict[str, Any]) -> bytes:
    """Encodes one record body as ``op\\n`` + canonical JSON payload bytes."""

    if not isinstance(op, str) or not op:
        raise ValueError("WAL record op must be a non-empty string")
    head = op.encode("utf-8")
    if b"\n" in head:
        raise ValueError("WAL record op must not contain a newline")
    return head + b"\n" + json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _decode_body(body: bytes) -> WalRecord:
    """Decodes one record body produced by :func:`_encode_body`."""

    newline = body.find(b"\n")
    if newline < 0:
        raise WalCorruptionError("WAL record body is missing its op header")
    op = body[:newline].decode("utf-8")
    try:
        payload = json.loads(body[newline + 1 :].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise WalCorruptionError("WAL record payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise WalCorruptionError("WAL record payload must be a JSON object")
    return WalRecord(op=op, payload=payload)
