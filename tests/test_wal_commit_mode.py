"""Opt-in WAL commit mode for LodeDB (``commit_mode="wal"``).

WAL mode appends one framed record per mutation to ``<key>.wal`` and checkpoints
into a generation periodically, instead of publishing a new generation on every
write. It is crash-atomic: the WAL is replayed on open (a torn trailing record
is discarded) and folded into a generation on a clean close. These tests cover
the knob plumbing, equivalence with the default generation mode, the checkpoint,
and crash recovery — including a hard ``os._exit`` kill of a writer mid-run.
"""

from __future__ import annotations

import glob
import multiprocessing as mp
import os
from pathlib import Path

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.engine.runtime_policy import (
    CommitMode,
    commit_mode_from_env,
    parse_commit_mode,
)
from lodedb.engine.wal_store import wal_path
from lodedb.local.db import LodeDB


def _be() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=384)


def _open(path, **kwargs) -> LodeDB:
    return LodeDB(path=path, model="minilm", _embedding_backend=_be(), **kwargs)


def _wal_files(path) -> list[str]:
    return glob.glob(os.path.join(str(path), "*.wal"))


# -- knob plumbing ----------------------------------------------------------


def test_parse_commit_mode_values():
    assert parse_commit_mode(None) is CommitMode.GENERATION
    assert parse_commit_mode("") is CommitMode.GENERATION
    assert parse_commit_mode("generation") is CommitMode.GENERATION
    assert parse_commit_mode("wal") is CommitMode.WAL
    assert parse_commit_mode("WAL") is CommitMode.WAL  # case-insensitive
    with pytest.raises(ValueError):
        parse_commit_mode("bogus")


def test_commit_mode_from_env(monkeypatch):
    monkeypatch.delenv("LODEDB_COMMIT_MODE", raising=False)
    assert commit_mode_from_env() is CommitMode.GENERATION
    monkeypatch.setenv("LODEDB_COMMIT_MODE", "wal")
    assert commit_mode_from_env() is CommitMode.WAL


def test_bad_commit_mode_rejected_at_open(tmp_path):
    with pytest.raises(ValueError):
        _open(tmp_path, commit_mode="bogus")


def test_cli_exposes_commit_mode_flag():
    """The ``index`` and ``serve`` commands surface a ``--commit-mode`` option."""

    from typer.testing import CliRunner

    from lodedb.local.cli import app

    runner = CliRunner()
    for command in ("index", "serve"):
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0
        assert "--commit-mode" in result.stdout


def test_default_mode_writes_no_wal(tmp_path):
    """The default (generation) mode never creates a WAL file."""

    db = _open(tmp_path)
    db.add("alpha", id="a")
    db.add("beta", id="b")
    db.close()
    assert _wal_files(tmp_path) == []


def test_env_default_can_select_wal(tmp_path, monkeypatch):
    monkeypatch.setenv("LODEDB_COMMIT_MODE", "wal")
    db = _open(tmp_path)  # no explicit commit_mode -> reads env
    db.add("alpha", id="a")
    # Before a checkpoint, the mutation lives in the WAL.
    assert _wal_files(tmp_path)
    db.close()


# -- functional equivalence with generation mode ----------------------------


_OPS = [
    ("add", "alpha one", "a"),
    ("add", "beta two", "b"),
    ("add", "gamma three", "c"),
    ("rm", "b", None),
    ("add", "alpha one revised", "a"),  # upsert-replace
    ("add", "delta four", "d"),
]


def _run_sequence(path, mode) -> dict:
    db = _open(path, commit_mode=mode)
    for kind, text, doc_id in _OPS:
        if kind == "add":
            db.add(text, id=doc_id)
        else:
            db.remove(text)
    db.close()
    reader = LodeDB.open_readonly(path, model="minilm", _embedding_backend=_be())
    try:
        return {
            "count": reader.count(),
            "ids": sorted(rec["id"] for rec in reader.list_documents()),
            "search_alpha": [(h.id, round(h.score, 6)) for h in reader.search("alpha", k=5)],
            "search_delta": [(h.id, round(h.score, 6)) for h in reader.search("delta", k=5)],
            "get_a": reader.get("a"),
            "get_d": reader.get("d"),
        }
    finally:
        reader.close()


def test_wal_mode_matches_generation_mode(tmp_path):
    """WAL mode and generation mode produce identical state and search scores."""

    generation = _run_sequence(tmp_path / "gen", CommitMode.GENERATION.value)
    wal = _run_sequence(tmp_path / "wal", CommitMode.WAL.value)
    assert wal == generation
    assert wal["ids"] == ["a", "c", "d"]
    assert wal["get_a"] == "alpha one revised"


# -- checkpoint -------------------------------------------------------------


def test_clean_close_checkpoints_and_truncates_wal(tmp_path):
    """A clean close folds the WAL into a generation and removes the WAL file."""

    db = _open(tmp_path, commit_mode="wal")
    db.add("alpha", id="a")
    db.add("beta", id="b")
    assert _wal_files(tmp_path)  # buffered in the WAL before close
    db.close()
    assert _wal_files(tmp_path) == []  # checkpoint truncated it
    # Reopen in generation mode: the data is fully in the committed generation.
    reader = LodeDB.open_readonly(tmp_path, model="minilm", _embedding_backend=_be())
    try:
        assert reader.count() == 2
        assert reader.get("a") == "alpha"
    finally:
        reader.close()


def test_persist_checkpoints_wal(tmp_path):
    """``persist()`` folds the WAL into a generation mid-session."""

    db = _open(tmp_path, commit_mode="wal")
    db.add("alpha", id="a")
    assert _wal_files(tmp_path)
    db.persist()
    assert _wal_files(tmp_path) == []
    # Further writes go back to the WAL until the next checkpoint.
    db.add("beta", id="b")
    assert _wal_files(tmp_path)
    assert db.count() == 2
    db.close()
    reader = LodeDB.open_readonly(tmp_path, model="minilm", _embedding_backend=_be())
    try:
        assert reader.count() == 2
    finally:
        reader.close()


def test_low_op_threshold_checkpoints_during_run(tmp_path):
    """When the op threshold is small, a run auto-checkpoints into a generation."""

    db = _open(tmp_path, commit_mode="wal")
    db._engine._wal_checkpoint_ops = 3  # fold every 3 logged mutations
    for i in range(7):
        db.add(f"doc number {i}", id=f"d{i}")
    # 7 adds with a 3-op threshold -> at least two folds happened; the residual
    # WAL holds fewer than 3 records (or none, on an exact multiple).
    assert db.count() == 7
    db.close()
    reader = LodeDB.open_readonly(tmp_path, model="minilm", _embedding_backend=_be())
    try:
        assert reader.count() == 7
    finally:
        reader.close()


# -- crash recovery (in-process: drop the handle without close) -------------


def test_recovery_after_unclean_shutdown(tmp_path):
    """Dropping a WAL-mode handle without close() still recovers via WAL replay."""

    db = _open(tmp_path, commit_mode="wal")
    db.add("the quick brown fox", id="a")
    db.add("a lazy dog", id="b")
    db.add("the fox runs", id="c")
    live = [(h.id, round(h.score, 6)) for h in db.search("fox", k=3)]
    # Simulate a crash: release the lock (so the test can reopen) WITHOUT
    # checkpointing, leaving the WAL on disk as a real crash would.
    db._engine._release_writer_lock()
    del db

    recovered = _open(tmp_path, commit_mode="wal")
    try:
        assert recovered.count() == 3
        assert recovered.get("a") == "the quick brown fox"
        assert recovered.get("c") == "the fox runs"
        # Replay reconstructs the identical index, so the ranking is unchanged.
        assert [(h.id, round(h.score, 6)) for h in recovered.search("fox", k=3)] == live
    finally:
        recovered.close()


def test_recovery_when_reopened_in_default_generation_mode(tmp_path):
    """A leftover WAL is recovered even if the next open omits ``commit_mode="wal"``.

    A writer that crashes in WAL mode before checkpointing leaves a ``<key>.wal``
    on disk. Reopening without re-specifying the mode (so it defaults to
    ``generation``) must still replay those durably-logged mutations rather than
    silently dropping them, and must fold the recovered WAL into a generation so
    the default commit path is not left with a log it never consults.
    """

    db = _open(tmp_path, commit_mode="wal")
    db.add("the quick brown fox", id="a")
    db.add("a lazy dog", id="b")
    db.add("the fox runs", id="c")
    live = [(h.id, round(h.score, 6)) for h in db.search("fox", k=3)]
    # Simulate a crash: leave the WAL on disk, release the lock so we can reopen.
    db._engine._release_writer_lock()
    del db
    assert _wal_files(tmp_path), "the crashed WAL-mode writer should leave a WAL"

    # Reopen in the DEFAULT generation mode (no commit_mode passed).
    recovered = _open(tmp_path)
    try:
        assert recovered.count() == 3
        assert recovered.get("a") == "the quick brown fox"
        assert recovered.get("c") == "the fox runs"
        assert [(h.id, round(h.score, 6)) for h in recovered.search("fox", k=3)] == live
    finally:
        recovered.close()
    # The recovered WAL was folded into a generation and truncated, so the
    # default path is left with no lingering log.
    assert _wal_files(tmp_path) == []


def test_torn_wal_tail_recovers_prior_records(tmp_path):
    """A torn trailing WAL record (crash mid-append) is dropped; the rest recover."""

    db = _open(tmp_path, commit_mode="wal")
    db.add("alpha one", id="a")
    db.add("beta two", id="b")
    db._engine._release_writer_lock()
    del db
    wal = wal_path(tmp_path, _only_wal_key(tmp_path))
    with wal.open("ab") as handle:
        handle.write(b"\x00\x00\x10\x00partial-frame-without-crc")

    recovered = _open(tmp_path, commit_mode="wal")
    try:
        assert recovered.count() == 2
        assert sorted(r["id"] for r in recovered.list_documents()) == ["a", "b"]
    finally:
        recovered.close()


def test_interior_wal_corruption_fails_closed(tmp_path):
    """A corrupt interior WAL record raises on reopen rather than losing data."""

    db = _open(tmp_path, commit_mode="wal")
    db.add("alpha one", id="a")
    db.add("beta two", id="b")
    db.add("gamma three", id="c")
    db._engine._release_writer_lock()
    del db
    wal = wal_path(tmp_path, _only_wal_key(tmp_path))
    raw = bytearray(wal.read_bytes())
    raw[40] ^= 0xFF  # flip a byte inside an early (non-trailing) record
    wal.write_bytes(bytes(raw))
    with pytest.raises(RuntimeError):
        _open(tmp_path, commit_mode="wal")


def _only_wal_key(path) -> str:
    files = _wal_files(path)
    assert len(files) == 1, f"expected exactly one WAL file, found {files}"
    return Path(files[0]).name[: -len(".wal")]


# -- crash recovery (subprocess: hard os._exit, no cleanup) -----------------


def _crash_writer(path_str: str) -> None:
    """Writes durable WAL records then hard-exits (no close/atexit/finalizers)."""

    db = LodeDB(
        path=path_str,
        model="minilm",
        commit_mode="wal",
        durability="fsync",  # records are guaranteed on disk before the kill
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    db.add("alpha durable", id="a")
    db.add("beta durable", id="b")
    db.add("gamma durable", id="c")
    # Hard kill: bypass close(), so no checkpoint and no lock release run — this
    # is the writer-killed-mid-commit case the WAL must survive.
    os._exit(0)


def test_recovery_after_hard_process_kill(tmp_path):
    """A writer hard-killed (os._exit) mid-run recovers its WAL on the next open."""

    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_crash_writer, args=(str(tmp_path),))
    proc.start()
    proc.join(timeout=120)
    assert proc.exitcode == 0  # os._exit(0)
    # The OS released the (advisory) writer lock when the process died, and the
    # fsync'd WAL is on disk with no checkpoint.
    assert _wal_files(tmp_path), "the killed writer should have left a WAL behind"

    recovered = _open(tmp_path, commit_mode="wal")
    try:
        assert recovered.count() == 3
        assert recovered.get("a") == "alpha durable"
        assert recovered.get("c") == "gamma durable"
        assert {h.id for h in recovered.search("durable", k=5)} == {"a", "b", "c"}
    finally:
        recovered.close()


# -- vector-in path ---------------------------------------------------------


def test_wal_mode_vector_in_recovers(tmp_path):
    """The vector-in path (add_vectors) is logged and replayed under WAL mode."""

    db = LodeDB.open_vector_store(
        tmp_path, vector_dim=8, commit_mode="wal", _embedding_backend=None
    )
    db.add_vectors([1.0, 0, 0, 0, 0, 0, 0, 0], id="x", text="vec x")
    db.add_vectors([0, 1.0, 0, 0, 0, 0, 0, 0], id="y", text="vec y")
    db._engine._release_writer_lock()
    del db

    recovered = LodeDB.open_vector_store(tmp_path, vector_dim=8, commit_mode="wal")
    try:
        assert recovered.count() == 2
        hits = recovered.search_by_vector([1.0, 0, 0, 0, 0, 0, 0, 0], k=2)
        assert hits[0].id == "x"
        assert recovered.get("x") == "vec x"
    finally:
        recovered.close()


# -- durability=fsync interplay ---------------------------------------------


def test_wal_with_fsync_roundtrips(tmp_path):
    """commit_mode=wal + durability=fsync reopens with data intact after a clean close."""

    db = _open(tmp_path, commit_mode="wal", durability="fsync")
    db.add("alpha", id="a")
    db.add("beta", id="b")
    db.close()
    reader = LodeDB.open_readonly(tmp_path, model="minilm", _embedding_backend=_be())
    try:
        assert reader.count() == 2
        assert reader.get("a") == "alpha"
        assert reader.get("b") == "beta"
    finally:
        reader.close()
