"""Default WAL commit mode for LodeDB (``commit_mode="wal"``).

WAL mode appends one framed record per mutation to ``<key>.wal`` and checkpoints
into a generation periodically, instead of publishing a new generation on every
write. It is crash-atomic: the WAL is replayed on open (a torn trailing record
is discarded) and folded into a generation on a clean close. These tests cover
the knob plumbing, equivalence with the classic generation mode, the checkpoint,
and crash recovery — including a hard ``os._exit`` kill of a writer mid-run.
"""

from __future__ import annotations

import gc
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
from lodedb.local.db import LodeDB


def _be() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=384)


def _open(path, **kwargs) -> LodeDB:
    return LodeDB(path=path, model="minilm", _embedding_backend=_be(), **kwargs)


def _wal_files(path) -> list[str]:
    return glob.glob(os.path.join(str(path), "*.wal"))


# -- knob plumbing ----------------------------------------------------------


def test_parse_commit_mode_values():
    assert parse_commit_mode(None) is CommitMode.WAL  # wal is the default
    assert parse_commit_mode("") is CommitMode.WAL
    assert parse_commit_mode("wal") is CommitMode.WAL
    assert parse_commit_mode("generation") is CommitMode.GENERATION
    assert parse_commit_mode("GENERATION") is CommitMode.GENERATION  # case-insensitive
    with pytest.raises(ValueError):
        parse_commit_mode("bogus")


def test_commit_mode_from_env(monkeypatch):
    monkeypatch.delenv("LODEDB_COMMIT_MODE", raising=False)
    assert commit_mode_from_env() is CommitMode.WAL  # wal is the default
    monkeypatch.setenv("LODEDB_COMMIT_MODE", "generation")
    assert commit_mode_from_env() is CommitMode.GENERATION
    monkeypatch.setenv("LODEDB_COMMIT_MODE", "wal")
    assert commit_mode_from_env() is CommitMode.WAL


def test_bad_commit_mode_rejected_at_open(tmp_path):
    with pytest.raises(ValueError):
        _open(tmp_path, commit_mode="bogus")


def test_cli_exposes_commit_mode_flag():
    """The ``index`` and ``serve`` commands register a ``--commit-mode`` option.

    Introspects the Click command tree rather than the rendered ``--help`` text:
    Typer renders help through Rich, which wraps/reflows option names at narrow or
    non-TTY widths (e.g. CI), so asserting on the rendered string is flaky.
    """

    import typer.main

    from lodedb.local.cli import app

    command = typer.main.get_command(app)
    for name in ("index", "serve"):
        opts = [opt for param in command.commands[name].params for opt in param.opts]
        assert "--commit-mode" in opts, f"{name} is missing the --commit-mode option"


def test_default_mode_is_wal(tmp_path):
    """The default commit mode is now WAL: a live add lands in a <key>.wal log."""

    db = _open(tmp_path)  # no explicit commit_mode -> default
    db.add("alpha", id="a")
    db.add("beta", id="b")
    # Before a checkpoint, the mutations live in the WAL.
    assert _wal_files(tmp_path)
    db.close()
    # A clean close folds the WAL into a generation and truncates it.
    assert _wal_files(tmp_path) == []


def test_generation_mode_writes_no_wal(tmp_path):
    """The opt-out generation mode never creates a WAL file."""

    db = _open(tmp_path, commit_mode="generation")
    db.add("alpha", id="a")
    db.add("beta", id="b")
    assert _wal_files(tmp_path) == []  # no WAL even before close
    db.close()
    assert _wal_files(tmp_path) == []


def test_env_can_select_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("LODEDB_COMMIT_MODE", "generation")
    db = _open(tmp_path)  # no explicit commit_mode -> reads env
    db.add("alpha", id="a")
    assert _wal_files(tmp_path) == []  # generation mode writes no WAL
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


def test_wal_run_folds_into_generation_on_close(tmp_path):
    """A WAL-mode run folds its log into a committed generation on close."""

    db = _open(tmp_path, commit_mode="wal")
    for i in range(7):
        db.add(f"doc number {i}", id=f"d{i}")
    assert db.count() == 7
    db.close()
    # close() checkpoints the WAL into a committed generation, leaving no log.
    assert _wal_files(tmp_path) == []
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
    # Simulate a crash: drop the handle WITHOUT checkpointing, leaving the WAL on
    # disk as a real crash would. The native engine (and its writer lock) is
    # released on its worker by the finalizer, so the reopen below can reacquire it.
    del db
    gc.collect()  # drop the native engine (releasing its writer lock) without a clean close

    recovered = _open(tmp_path, commit_mode="wal")
    try:
        assert recovered.count() == 3
        assert recovered.get("a") == "the quick brown fox"
        assert recovered.get("c") == "the fox runs"
        # Replay reconstructs the identical index, so the ranking is unchanged.
        assert [(h.id, round(h.score, 6)) for h in recovered.search("fox", k=3)] == live
    finally:
        recovered.close()


def test_recovery_when_reopened_in_generation_mode(tmp_path):
    """A leftover WAL is recovered even when the next open opts into generation mode.

    A writer that crashes before checkpointing leaves a ``<key>.wal`` on disk.
    Reopening with ``commit_mode="generation"`` must still replay those
    durably-logged mutations rather than silently dropping them, and must fold the
    recovered WAL into a generation so the generation path is not left with a log
    it never consults.
    """

    db = _open(tmp_path, commit_mode="wal")
    db.add("the quick brown fox", id="a")
    db.add("a lazy dog", id="b")
    db.add("the fox runs", id="c")
    live = [(h.id, round(h.score, 6)) for h in db.search("fox", k=3)]
    # Simulate a crash: drop the handle, leaving the WAL on disk; the native
    # engine's writer lock is released on its worker so we can reopen.
    del db
    gc.collect()  # drop the native engine (releasing its writer lock) without a clean close
    assert _wal_files(tmp_path), "the crashed WAL-mode writer should leave a WAL"

    # Reopen explicitly in generation mode (the opt-out path).
    recovered = _open(tmp_path, commit_mode="generation")
    try:
        assert recovered.count() == 3
        assert recovered.get("a") == "the quick brown fox"
        assert recovered.get("c") == "the fox runs"
        assert [(h.id, round(h.score, 6)) for h in recovered.search("fox", k=3)] == live
    finally:
        recovered.close()
    # The recovered WAL was folded into a generation and truncated, so the
    # generation path is left with no lingering log.
    assert _wal_files(tmp_path) == []


def test_open_normalizes_wal_into_generation(tmp_path):
    """Every writable open folds a leftover WAL into a clean generation tail.

    Reopening in WAL mode after an unclean shutdown still lands on a normalized
    committed generation (the recovered WAL is folded, not left on disk), and a
    subsequent add starts a fresh WAL on top of it.
    """

    db = _open(tmp_path, commit_mode="wal")
    db.add("alpha one", id="a")
    db.add("beta two", id="b")
    del db  # crash: drop the handle without a clean close, leaving the WAL on disk
    gc.collect()
    assert _wal_files(tmp_path)

    reopened = _open(tmp_path, commit_mode="wal")
    try:
        # Open folded the recovered WAL into a generation: no stray log remains.
        assert _wal_files(tmp_path) == []
        assert reopened.count() == 2
        # A fresh add starts a new WAL on top of the normalized generation.
        reopened.add("gamma three", id="c")
        assert _wal_files(tmp_path)
        assert reopened.count() == 3
    finally:
        reopened.close()
    assert _wal_files(tmp_path) == []


def test_torn_wal_tail_recovers_prior_records(tmp_path):
    """A torn trailing WAL record (crash mid-append) is dropped; the rest recover."""

    db = _open(tmp_path, commit_mode="wal")
    db.add("alpha one", id="a")
    db.add("beta two", id="b")
    del db
    gc.collect()  # drop the native engine (releasing its writer lock) without a clean close
    wal = Path(tmp_path) / f"{_only_wal_key(tmp_path)}.wal"
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
    del db
    gc.collect()  # drop the native engine (releasing its writer lock) without a clean close
    wal = Path(tmp_path) / f"{_only_wal_key(tmp_path)}.wal"
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
    del db
    gc.collect()  # drop the native engine (releasing its writer lock) without a clean close

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
