"""`LodeDB("orecloud://…")` dispatch: the constructor routes managed-cloud
targets to the optional `orecloud` companion ([cloud] extra) and answers a
clear install hint when it is absent, while every other target constructs the
local database exactly as before. Both cloud paths are forced with a fake (or
blocked) `orecloud` module, so the suite never depends on whether the real
client happens to be installed."""

from __future__ import annotations

import sys
import types

import pytest

from lodedb import LodeDB

CLOUD_TARGET = "orecloud://acme/prod/user-42"


def _fake_orecloud(monkeypatch, connect):
    """Installs a fake `orecloud` module whose `connect` is the given callable."""
    fake_pkg = types.ModuleType("orecloud")
    fake_pkg.connect = connect
    monkeypatch.setitem(sys.modules, "orecloud", fake_pkg)


def _recording_connect(captured: dict, handle: object):
    """A stand-in for `orecloud.connect` mirroring its real keyword surface;
    records what it was called with and returns `handle`."""

    def connect(
        target: str,
        *,
        token: str | None = None,
        host: str | None = None,
        key: str | None = None,
        warm: bool = True,
        timeout: float = 30.0,
        read_your_writes: bool = True,
        transport: object | None = None,
    ) -> object:
        captured["target"] = target
        captured["token"] = token
        captured["warm"] = warm
        return handle

    return connect


def test_missing_companion_raises_install_hint(monkeypatch):
    # A None entry makes `import orecloud` raise ImportError even when the
    # real package is installed in the venv.
    monkeypatch.setitem(sys.modules, "orecloud", None)
    with pytest.raises(ImportError, match=r'pip install "lodedb\[cloud\]"'):
        LodeDB(CLOUD_TARGET)


def test_cloud_target_returns_companion_handle(monkeypatch):
    """A cloud target hands back whatever the companion's connect() opens —
    LodeDB.__init__ never runs on it — with the options forwarded."""
    captured: dict[str, object] = {}
    handle = object()
    _fake_orecloud(monkeypatch, _recording_connect(captured, handle))

    db = LodeDB(CLOUD_TARGET, token="tok-1", warm=False)

    assert db is handle
    assert not isinstance(db, LodeDB)
    assert captured == {"target": CLOUD_TARGET, "token": "tok-1", "warm": False}


def test_cloud_target_accepts_path_keyword(monkeypatch):
    """`LodeDB(path="orecloud://…")` (the documented keyword spelling)
    dispatches too, and the `path` keyword itself is not forwarded."""
    captured: dict[str, object] = {}
    handle = object()
    _fake_orecloud(monkeypatch, _recording_connect(captured, handle))

    assert LodeDB(path=CLOUD_TARGET) is handle
    assert captured["target"] == CLOUD_TARGET


def test_local_only_options_are_rejected_up_front(monkeypatch):
    """Local construction options have no cloud meaning: the error names the
    offending keyword and lists what cloud targets do accept."""
    _fake_orecloud(monkeypatch, _recording_connect({}, object()))

    with pytest.raises(TypeError, match=r"do not accept model.*available options.*token"):
        LodeDB(CLOUD_TARGET, model="minilm")


def test_local_paths_still_construct_locally(tmp_path):
    """The dispatch leaves ordinary construction untouched: a filesystem path
    (str or Path) yields a real local LodeDB instance."""
    db = LodeDB(tmp_path / "store", vector_dim=8)
    try:
        assert isinstance(db, LodeDB)
    finally:
        db.close()
    db = LodeDB(str(tmp_path / "store"), vector_dim=8)
    try:
        assert isinstance(db, LodeDB)
    finally:
        db.close()
