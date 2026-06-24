"""Frame-level tests for the append-only WAL store (lodedb.engine.wal_store).

The WAL frames each logical mutation as ``len || body || crc32(len||body)``. A
torn *trailing* frame (writer killed mid-append) must be dropped silently so
replay recovers everything written before the crash; a corrupt *interior* frame
must fail closed so a damaged log can never replay a wrong mutation. These tests
exercise the framing directly, independent of the engine.
"""

from __future__ import annotations

import struct
import zlib

import pytest

from lodedb.engine.wal_store import (
    WAL_MAGIC,
    WalCorruptionError,
    WalStore,
)


def _wal(tmp_path):
    return WalStore(tmp_path / "k.wal")


def test_empty_wal_has_no_records(tmp_path):
    store = _wal(tmp_path)
    assert store.op_count == 0
    assert store.read_records() == []
    assert not store.exists()  # no file written until the first append


def test_append_then_read_roundtrips_in_order(tmp_path):
    store = _wal(tmp_path)
    store.append("upsert_documents", {"documents": [{"document_id": "a"}]})
    store.append("delete_documents", {"document_ids": ["a"]})
    assert store.op_count == 2
    assert store.exists()
    records = store.read_records()
    assert [r.op for r in records] == ["upsert_documents", "delete_documents"]
    assert records[0].payload == {"documents": [{"document_id": "a"}]}
    assert records[1].payload == {"document_ids": ["a"]}


def test_reopen_recovers_existing_record_count(tmp_path):
    store = _wal(tmp_path)
    store.append("upsert_documents", {"n": 1})
    store.append("upsert_documents", {"n": 2})
    reopened = WalStore(tmp_path / "k.wal")
    assert reopened.op_count == 2
    assert [r.payload["n"] for r in reopened.read_records()] == [1, 2]


def test_torn_trailing_frame_is_dropped(tmp_path):
    """A half-written final frame (a crash mid-append) is silently discarded."""

    store = _wal(tmp_path)
    store.append("upsert_documents", {"n": 1})
    store.append("upsert_documents", {"n": 2})
    # Simulate a writer killed mid-append: a partial third frame.
    with (tmp_path / "k.wal").open("ab") as handle:
        handle.write(struct.pack(">I", 9999) + b"partial-body-no-crc")
    reopened = WalStore(tmp_path / "k.wal")
    assert reopened.op_count == 2  # only the two intact frames
    assert [r.payload["n"] for r in reopened.read_records()] == [1, 2]


def test_truncated_length_prefix_is_dropped(tmp_path):
    """A trailing fragment shorter than the length prefix is treated as a crash tail."""

    store = _wal(tmp_path)
    store.append("upsert_documents", {"n": 1})
    with (tmp_path / "k.wal").open("ab") as handle:
        handle.write(b"\x00\x00")  # 2 bytes, shorter than the 4-byte length prefix
    assert WalStore(tmp_path / "k.wal").op_count == 1


def test_interior_corruption_fails_closed(tmp_path):
    """A bad CRC on a record followed by more bytes is real corruption, not a tail."""

    store = _wal(tmp_path)
    store.append("upsert_documents", {"n": 1})
    store.append("upsert_documents", {"n": 2})
    raw = bytearray((tmp_path / "k.wal").read_bytes())
    # Flip a byte inside the first record's body (well past the 8-byte header and
    # 4-byte length prefix), leaving the second record intact after it.
    raw[len(WAL_MAGIC) + 4 + 12] ^= 0xFF
    (tmp_path / "k.wal").write_bytes(bytes(raw))
    with pytest.raises(WalCorruptionError):
        WalStore(tmp_path / "k.wal").read_records()


def test_bad_magic_fails_closed(tmp_path):
    (tmp_path / "k.wal").write_bytes(b"NOTAWALX" + b"\x00\x00\x00\x01" + b"junk")
    with pytest.raises(WalCorruptionError):
        WalStore(tmp_path / "k.wal").read_records()


def test_truncate_removes_file_and_resets_counts(tmp_path):
    store = _wal(tmp_path)
    store.append("upsert_documents", {"n": 1})
    assert store.exists()
    store.truncate()
    assert not (tmp_path / "k.wal").exists()
    assert store.op_count == 0
    assert store.byte_count == 0


def test_should_checkpoint_on_op_threshold(tmp_path):
    store = _wal(tmp_path)
    for i in range(5):
        store.append("upsert_documents", {"n": i})
    assert store.should_checkpoint(checkpoint_ops=5, checkpoint_bytes=10**12)
    assert not store.should_checkpoint(checkpoint_ops=6, checkpoint_bytes=10**12)


def test_should_checkpoint_on_byte_threshold(tmp_path):
    store = _wal(tmp_path)
    store.append("upsert_documents", {"blob": "x" * 1000})
    assert store.should_checkpoint(checkpoint_ops=10**9, checkpoint_bytes=10)
    assert not store.should_checkpoint(checkpoint_ops=10**9, checkpoint_bytes=10**9)


def test_fsync_mode_writes_durably(tmp_path):
    """fsync mode must still produce a readable, correctly framed WAL."""

    store = WalStore(tmp_path / "k.wal", fsync=True)
    store.append("upsert_documents", {"n": 1})
    store.append("delete_documents", {"document_ids": ["a"]})
    records = WalStore(tmp_path / "k.wal").read_records()
    assert [r.op for r in records] == ["upsert_documents", "delete_documents"]


def test_crc_covers_length_prefix(tmp_path):
    """Corrupting only the length prefix is caught by the CRC (framed over it)."""

    store = _wal(tmp_path)
    store.append("upsert_documents", {"n": 1})
    store.append("upsert_documents", {"n": 2})
    raw = bytearray((tmp_path / "k.wal").read_bytes())
    # Corrupt the first record's length prefix (the 4 bytes after the magic+ver
    # header). Because the CRC is computed over (length || body), this is caught
    # as interior corruption rather than silently mis-parsed.
    raw[len(WAL_MAGIC)] ^= 0x01
    (tmp_path / "k.wal").write_bytes(bytes(raw))
    with pytest.raises(WalCorruptionError):
        WalStore(tmp_path / "k.wal").read_records()


def test_crc_value_matches_manual_computation(tmp_path):
    """Sanity-check the on-disk framing against an independent CRC computation."""

    store = _wal(tmp_path)
    store.append("op", {"x": 1})
    raw = (tmp_path / "k.wal").read_bytes()
    body_region = raw[len(WAL_MAGIC) + 4 :]  # strip magic + version header
    length = struct.unpack_from(">I", body_region, 0)[0]
    frame = body_region[: 4 + length]
    recorded = struct.unpack_from(">I", body_region, 4 + length)[0]
    assert recorded == (zlib.crc32(frame) & 0xFFFFFFFF)
