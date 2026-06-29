"""Tests for durable raw-text retrieval in the local LodeDB SDK.

Raw text is retained by default (``store_text=True``) in a dedicated ``.tvtext``
base plus a ``.txd`` delta journal, so the original text is retrievable by id
(``get`` / ``get_text`` / ``get_texts``), durably across reopens and committed
O(changed) per write. The redacted artifacts stay payload-free
regardless — telemetry, audit, the ``.json`` snapshot, the ``.jsd`` journal, and
the ``.tvim``/``.tvd`` vector sidecars never carry raw document text. Opening
with ``store_text=False`` opts out of retaining text at all.

These exercise the feature with the same deterministic hash backend the rest of
the local suite uses, so they neither download models nor import torch.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from lodedb.engine.core import audit_persisted_index_snapshots
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB

_SECRET = "TOPSECRET document body that must only live in the text sidecar"


_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _open(
    tmp_path,
    *,
    store_text: bool,
    dim: int = 384,
    commit_mode: str | None = None,
    compression: bool | None = None,
) -> LodeDB:
    """Opens a LodeDB with an injected deterministic hash backend.

    ``compression`` is omitted from the call when ``None`` so a reopen can rely on
    the persisted value (the SDK default is ``True``).
    """

    kwargs: dict = {}
    if compression is not None:
        kwargs["compression"] = compression
    return LodeDB(
        path=tmp_path,
        model="minilm",
        store_text=store_text,
        commit_mode=commit_mode,
        _embedding_backend=HashEmbeddingBackend(native_dim=dim),
        **kwargs,
    )


def _text_base(tmp_path) -> Path:
    """Returns the single retained-text base file (``g*.tvtext``)."""

    bases = sorted(glob.glob(str(Path(tmp_path) / "**" / "g*.tvtext"), recursive=True))
    assert bases, "expected a raw-text base"
    return Path(bases[-1])


def test_store_text_enabled_by_default(tmp_path):
    """Text retention is on by default: get works and a sidecar is written."""

    db = LodeDB(
        path=tmp_path,
        model="minilm",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    assert db.store_text is True
    db.add("default-on document body", id="a")
    db.persist()
    assert db.get("a") == "default-on document body"
    assert glob.glob(str(Path(tmp_path) / "**" / "*.tvtext"), recursive=True)
    db.close()


def test_store_text_false_opts_out(tmp_path):
    """Opening with store_text=False: get_text/get_texts raise and no sidecar is written."""

    db = _open(tmp_path, store_text=False)
    db.add("plain document body", id="a")
    db.persist()
    with pytest.raises(ValueError, match="store_text=True"):
        db.get_text("a")
    with pytest.raises(ValueError, match="store_text=True"):
        db.get_texts(["a"])
    assert not glob.glob(str(Path(tmp_path) / "**" / "*.tvtext"), recursive=True)
    db.close()


def test_get_text_returns_stored_text(tmp_path):
    """With the opt-in, get_text returns the exact text passed to add."""

    db = _open(tmp_path, store_text=True)
    db.add("the quick brown fox jumps", id="fox", metadata={"topic": "animals"})
    assert db.get_text("fox") == "the quick brown fox jumps"
    db.close()


def test_text_commit_is_o_changed(tmp_path):
    """An incremental text commit appends a .txd delta, never rewriting the base."""

    # O(changed) per-commit deltas are a generation-mode property (the WAL default
    # buffers writes and folds them into a base at checkpoint, not per add).
    db = _open(tmp_path, store_text=True, commit_mode="generation")
    db.add_many([{"text": f"base body {i}", "id": f"b{i}"} for i in range(12)])  # cold base
    base_files = glob.glob(str(Path(tmp_path) / "**" / "g*.tvtext"), recursive=True)
    assert base_files, "expected a raw-text base"
    base = Path(sorted(base_files)[-1])
    base_bytes = base.read_bytes()

    db.add("one small delta", id="delta")  # a single-doc incremental commit
    # The base map is untouched; the change lands in a small .txd delta segment.
    assert base.read_bytes() == base_bytes, "base must not be rewritten on a delta commit"
    deltas = glob.glob(str(Path(tmp_path) / "**" / "*.txd"), recursive=True)
    assert deltas, "expected a .txd text delta segment"
    assert db.get("delta") == "one small delta"
    db.close()

    reopened = _open(tmp_path, store_text=True)
    try:  # replaying base + delta reconstructs every document's text
        assert reopened.get("b5") == "base body 5"
        assert reopened.get("delta") == "one small delta"
    finally:
        reopened.close()


def test_get_is_alias_for_get_text(tmp_path):
    """``get`` is the primary retrieval verb and behaves exactly like ``get_text``."""

    db = _open(tmp_path, store_text=True)
    db.add("the quick brown fox jumps", id="fox")
    assert db.get("fox") == db.get_text("fox") == "the quick brown fox jumps"
    assert db.get("absent") is None
    db.close()

    off = _open(tmp_path, store_text=False)
    with pytest.raises(ValueError, match="store_text=True"):
        off.get("fox")
    off.close()


def test_get_text_missing_id_returns_none(tmp_path):
    """A never-added id returns None (not an error) when the opt-in is on."""

    db = _open(tmp_path, store_text=True)
    db.add("present document", id="here")
    assert db.get_text("absent") is None
    db.close()


def test_get_texts_batch_omits_missing(tmp_path):
    """get_texts returns a {id: text} map and omits unknown ids."""

    db = _open(tmp_path, store_text=True)
    db.add_many(
        [
            {"text": "first body", "id": "one"},
            {"text": "second body", "id": "two"},
        ]
    )
    got = db.get_texts(["one", "two", "missing"])
    assert got == {"one": "first body", "two": "second body"}
    assert db.get_texts([]) == {}
    db.close()


def test_add_many_and_auto_ids_store_text(tmp_path):
    """Auto-generated ids also get their text stored and retrievable."""

    db = _open(tmp_path, store_text=True)
    ids = db.add_many([{"text": "auto one"}, {"text": "auto two"}])
    texts = db.get_texts(ids)
    assert {texts[ids[0]], texts[ids[1]]} == {"auto one", "auto two"}
    db.close()


def test_text_persists_and_reloads_across_reopen(tmp_path):
    """Stored text survives persist()/close and is retrievable after reopen."""

    db = _open(tmp_path, store_text=True)
    db.add("durable body alpha", id="a", metadata={"k": "a"})
    db.add("durable body beta", id="b", metadata={"k": "b"})
    db.persist()
    db.close()

    reopened = _open(tmp_path, store_text=True)
    assert reopened.count() == 2
    assert reopened.get_text("a") == "durable body alpha"
    assert reopened.get_texts(["a", "b"]) == {
        "a": "durable body alpha",
        "b": "durable body beta",
    }
    reopened.close()


def test_remove_drops_stored_text(tmp_path):
    """Removing a document also drops its stored text, durably."""

    db = _open(tmp_path, store_text=True)
    db.add("removable body", id="gone")
    db.add("kept body", id="kept")
    assert db.remove("gone") is True
    assert db.get_text("gone") is None
    assert db.get_text("kept") == "kept body"
    db.persist()
    db.close()

    reopened = _open(tmp_path, store_text=True)
    assert reopened.get_text("gone") is None
    assert reopened.get_text("kept") == "kept body"
    reopened.close()


def test_upsert_overwrites_stored_text(tmp_path):
    """Re-adding an id replaces its stored text with the new body."""

    db = _open(tmp_path, store_text=True)
    db.add("original body", id="doc")
    assert db.get_text("doc") == "original body"
    db.add("replacement body", id="doc")
    assert db.get_text("doc") == "replacement body"
    db.persist()
    db.close()

    reopened = _open(tmp_path, store_text=True)
    assert reopened.get_text("doc") == "replacement body"
    reopened.close()


def test_reopen_without_opt_in_ignores_sidecar(tmp_path):
    """A DB reopened without store_text neither reads nor exposes the sidecar."""

    db = _open(tmp_path, store_text=True)
    db.add("body for later", id="x")
    db.persist()
    db.close()

    plain = _open(tmp_path, store_text=False)
    assert plain.count() == 1  # redacted state still loads
    with pytest.raises(ValueError, match="store_text=True"):
        plain.get_text("x")
    plain.close()

    # The opt-in DB can still read it back afterwards.
    again = _open(tmp_path, store_text=True)
    assert again.get_text("x") == "body for later"
    again.close()


def test_raw_text_never_leaks_into_redacted_artifacts(tmp_path):
    """Enabling text storage keeps snapshot/journal/telemetry/audit payload-free."""

    db = _open(tmp_path, store_text=True)
    db.add(_SECRET, id="s", metadata={"topic": "x"})
    db.search("body", k=3)
    db.persist()

    # Redacted JSON snapshot/manifest and journal deltas carry no raw text.
    for json_file in glob.glob(str(Path(tmp_path) / "**" / "*.json"), recursive=True):
        assert _SECRET not in Path(json_file).read_text(encoding="utf-8")
    for jsd in glob.glob(str(Path(tmp_path) / "**" / "*.jsd"), recursive=True):
        assert _SECRET not in Path(jsd).read_bytes().decode("utf-8", "replace")

    # The persisted redacted artifacts carry no raw text: the disk auditor scans
    # the committed snapshot + journal deltas for forbidden raw-payload keys and
    # never inspects the text sidecar.
    report = audit_persisted_index_snapshots(tmp_path)
    assert report["status"] == "passed"
    assert report["raw_document_text_present"] is False
    assert _SECRET not in json.dumps(report)

    # stats stays payload-free and still reports raw_payload_text_present False.
    stats = db.stats()
    assert stats["raw_payload_text_present"] is False
    assert _SECRET not in json.dumps(stats)

    # The text lives only in the dedicated sidecar. It is zstd-compressed, so the
    # payload is not even plaintext on disk; round-tripping through the native
    # reader proves it is durably stored and retrievable.
    assert db.get("s") == _SECRET
    tvtext = Path(glob.glob(str(Path(tmp_path) / "**" / "*.tvtext"), recursive=True)[0])
    assert tvtext.stat().st_size > 0
    assert _SECRET not in tvtext.read_bytes().decode("utf-8", "replace")
    db.close()


def test_snapshot_auditor_still_passes_with_text_storage(tmp_path):
    """The redacted-snapshot auditor passes (it never inspects the text sidecar)."""

    db = _open(tmp_path, store_text=True)
    db.add(_SECRET, id="s")
    db.persist()
    db.close()

    report = audit_persisted_index_snapshots(tmp_path)
    assert report["status"] == "passed"
    assert report["raw_document_text_present"] is False
    assert report["snapshot_count"] == 1


def test_corrupt_text_sidecar_fails_closed(tmp_path):
    """A garbled .tvtext sidecar raises on reopen instead of serving partial text."""

    db = _open(tmp_path, store_text=True)
    db.add("body that will be corrupted", id="c")
    db.persist()
    db.close()

    sidecars = glob.glob(str(Path(tmp_path) / "**" / "*.tvtext"), recursive=True)
    assert sidecars, "expected a .tvtext sidecar"
    Path(sidecars[0]).write_text("not-valid-json {{{", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _open(tmp_path, store_text=True)


def test_text_store_checksum_mismatch_fails_closed(tmp_path):
    """A tampered (but valid-JSON) text base body fails the checksum guard."""

    db = _open(tmp_path, store_text=True)
    db.add("checksum target body", id="c")
    db.persist()
    db.close()

    # The journaled raw-text base at g<epoch>.tvtext is a checksummed id->text map,
    # now zstd-compressed. Overwrite it with an uncompressed base whose body no
    # longer matches its recorded checksum: the native reader still accepts
    # uncompressed bases (back-compat) and re-checks the body hash, so a tampered
    # body fails closed regardless of on-disk compression.
    base = Path(sorted(glob.glob(str(Path(tmp_path) / "**" / "g*.tvtext"), recursive=True))[-1])
    tampered = {
        "schema_version": 2,
        "body_sha256": "0" * 64,  # stale: does not match the body below
        "body": {"schema_version": 2, "documents": {"c": "tampered body"}},
    }
    base.write_text(json.dumps(tampered), encoding="utf-8")

    # Reopening the DB fails closed on the checksum rather than serving tampered text.
    with pytest.raises(RuntimeError, match="checksum"):
        _open(tmp_path, store_text=True)


def test_compression_enabled_by_default_writes_zstd_base(tmp_path):
    """The default (compression=True) writes a zstd-framed retained-text base."""

    db = LodeDB(
        path=tmp_path,
        model="minilm",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    assert db.compression is True
    db.add("compressed default body", id="a")
    db.persist()
    base = _text_base(tmp_path)
    assert base.read_bytes().startswith(_ZSTD_MAGIC), "default base must be zstd-compressed"
    assert db.get("a") == "compressed default body"
    db.close()


def test_compression_false_writes_plain_json_base(tmp_path):
    """compression=False writes a plain-JSON base (no zstd magic) that still reads."""

    db = _open(tmp_path, store_text=True, compression=False)
    assert db.compression is False
    db.add("uncompressed body", id="a")
    db.persist()
    base = _text_base(tmp_path)
    raw = base.read_bytes()
    assert not raw.startswith(_ZSTD_MAGIC), "compression=False must not write a zstd frame"
    # It is plain canonical JSON, so the wrapper parses and carries the document.
    assert b'"documents"' in raw
    assert db.get("a") == "uncompressed body"
    db.close()


def test_compression_setting_persists_and_reopen_wins(tmp_path):
    """A store created uncompressed stays uncompressed on a default reopen.

    The text-store manifest records ``compress``; the persisted value wins over
    the passed default, so reopening WITHOUT compression=False (SDK default True)
    keeps writing uncompressed bases, and all text still reads back.
    """

    # commit_mode="generation" so each add rewrites/extends the text store on disk
    # under the engine's effective (persisted) compression flag.
    db = _open(tmp_path, store_text=True, compression=False, commit_mode="generation")
    db.add("seed body", id="seed")
    db.persist()
    db.close()
    assert not _text_base(tmp_path).read_bytes().startswith(_ZSTD_MAGIC)

    # Reopen WITHOUT passing compression: the SDK default is True, but the
    # persisted False must win, so a fresh base written now stays uncompressed.
    reopened = _open(tmp_path, store_text=True, commit_mode="generation")
    assert reopened.compression is True  # the requested/seeded value
    # Enough writes to force a fresh base rewrite (compaction), proving NEW writes
    # honour the persisted flag rather than the requested default.
    reopened.add_many([{"text": f"more body {i}", "id": f"m{i}"} for i in range(16)])
    reopened.persist()
    base = _text_base(tmp_path)
    assert not base.read_bytes().startswith(
        _ZSTD_MAGIC
    ), "persisted compress=false must win on reopen (new base stays uncompressed)"
    assert reopened.get("seed") == "seed body"
    assert reopened.get("m5") == "more body 5"
    reopened.close()


def test_compression_true_persists_and_reopen_stays_compressed(tmp_path):
    """A store created compressed stays compressed across a reopen, and reads back."""

    db = _open(tmp_path, store_text=True, compression=True, commit_mode="generation")
    db.add("compressed seed", id="seed")
    db.persist()
    db.close()
    assert _text_base(tmp_path).read_bytes().startswith(_ZSTD_MAGIC)

    reopened = _open(tmp_path, store_text=True, commit_mode="generation")
    reopened.add_many([{"text": f"more body {i}", "id": f"m{i}"} for i in range(16)])
    reopened.persist()
    assert _text_base(tmp_path).read_bytes().startswith(_ZSTD_MAGIC)
    assert reopened.get("seed") == "compressed seed"
    assert reopened.get("m5") == "more body 5"
    reopened.close()
