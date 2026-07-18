"""The two constructor front doors to a managed-cloud store: `LodeDB.cloud()`
(the human-facing form, accepting a bare store id) and the
`LodeDB("orecloud://…")` config-string dispatch (one string field expressing
either a local path or a cloud store). Both funnel through
`lodedb.cloud.open_cloud_target`: lazy companion import with a clear install
hint when the [cloud] extra is absent, targeted rejection of local-only
options, and every other target constructing the local database exactly as
before. All cloud paths are forced with a fake (or blocked) `orecloud`
module, so the suite never depends on whether the real client is installed."""

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


def test_cloud_classmethod_accepts_a_bare_store_id(monkeypatch):
    """`LodeDB.cloud("user-42")` forwards the short target verbatim — the
    companion resolves org/environment from the credential — and returns the
    companion's handle through the same funnel as the config-string form."""
    captured: dict[str, object] = {}
    handle = object()
    _fake_orecloud(monkeypatch, _recording_connect(captured, handle))

    db = LodeDB.cloud("user-42", token="tok-1")

    assert db is handle
    assert captured["target"] == "user-42"
    assert captured["token"] == "tok-1"


def test_cloud_classmethod_accepts_the_url_form(monkeypatch):
    """The classmethod takes the full triple and the orecloud:// spelling too."""
    captured: dict[str, object] = {}
    handle = object()
    _fake_orecloud(monkeypatch, _recording_connect(captured, handle))

    assert LodeDB.cloud(CLOUD_TARGET) is handle
    assert captured["target"] == CLOUD_TARGET


def test_cloud_classmethod_missing_companion_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "orecloud", None)
    with pytest.raises(ImportError, match=r'pip install "lodedb\[cloud\]"'):
        LodeDB.cloud("user-42")


def test_cloud_classmethod_rejects_local_only_options(monkeypatch):
    _fake_orecloud(monkeypatch, _recording_connect({}, object()))
    with pytest.raises(TypeError, match=r"do not accept model"):
        LodeDB.cloud("user-42", model="minilm")


def test_bare_store_ids_do_not_dispatch_from_the_constructor(tmp_path, monkeypatch):
    """Only `LodeDB.cloud()` accepts scheme-less short forms: a bare id passed
    to the constructor is an ordinary (relative) local path, never a cloud
    target — so a fake companion must not be consulted."""

    def _exploding_connect(target, **options):
        raise AssertionError("the constructor must not dispatch scheme-less targets")

    _fake_orecloud(monkeypatch, _exploding_connect)
    monkeypatch.chdir(tmp_path)
    db = LodeDB("user-42", vector_dim=8)
    try:
        assert isinstance(db, LodeDB)
    finally:
        db.close()


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
