"""The two constructor front doors to a managed-cloud store: `LodeDB.cloud()`
(the human-facing form, accepting a bare store id) and the
`LodeDB("orecloud://…")` config-string dispatch (one string field expressing
either a local path or a cloud store). Both funnel through
`lodedb.cloud.open_cloud_target`: a lazy import of the first-party client
with a clear install hint when the [cloud] extra's dependencies are absent,
targeted rejection of local-only options, and every other target constructing
the local database exactly as before. The client's `connect` is faked (or its
module blocked), so the suite runs without any network."""

from __future__ import annotations

import sys

import pytest

from lodedb import LodeDB

CLOUD_TARGET = "orecloud://acme/prod/user-42"


def _fake_connect(monkeypatch, captured: dict, handle: object):
    """Replaces `lodedb.cloud.serving.connect` with a recorder mirroring its
    real keyword surface; the funnel re-fetches the attribute per call."""

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

    monkeypatch.setattr("lodedb.cloud.serving.connect", connect)


def test_missing_cloud_deps_raise_install_hint(monkeypatch):
    # A None entry makes `import lodedb.cloud.serving` raise ImportError even
    # though the [cloud] extra's dependencies are installed in the dev venv.
    monkeypatch.setitem(sys.modules, "lodedb.cloud.serving", None)
    with pytest.raises(ImportError, match=r'pip install "lodedb\[cloud\]"'):
        LodeDB(CLOUD_TARGET)


def test_cloud_target_returns_the_client_handle(monkeypatch):
    """A cloud target hands back whatever the client's connect() opens —
    LodeDB.__init__ never runs on it — with the options forwarded."""
    captured: dict[str, object] = {}
    handle = object()
    _fake_connect(monkeypatch, captured, handle)

    db = LodeDB(CLOUD_TARGET, token="tok-1", warm=False)

    assert db is handle
    assert not isinstance(db, LodeDB)
    assert captured == {"target": CLOUD_TARGET, "token": "tok-1", "warm": False}


def test_cloud_target_accepts_path_keyword(monkeypatch):
    """`LodeDB(path="orecloud://…")` (the documented keyword spelling)
    dispatches too, and the `path` keyword itself is not forwarded."""
    captured: dict[str, object] = {}
    handle = object()
    _fake_connect(monkeypatch, captured, handle)

    assert LodeDB(path=CLOUD_TARGET) is handle
    assert captured["target"] == CLOUD_TARGET


def test_local_only_options_are_rejected_up_front(monkeypatch):
    """Local construction options have no cloud meaning: the error names the
    offending keyword and lists what cloud targets do accept."""
    _fake_connect(monkeypatch, {}, object())

    with pytest.raises(TypeError, match=r"do not accept model.*available options.*token"):
        LodeDB(CLOUD_TARGET, model="minilm")


def test_cloud_classmethod_accepts_a_bare_store_id(monkeypatch):
    """`LodeDB.cloud("user-42")` forwards the short target verbatim — the
    client resolves org/environment from the credential — and returns the
    handle through the same funnel as the config-string form."""
    captured: dict[str, object] = {}
    handle = object()
    _fake_connect(monkeypatch, captured, handle)

    db = LodeDB.cloud("user-42", token="tok-1")

    assert db is handle
    assert captured["target"] == "user-42"
    assert captured["token"] == "tok-1"


def test_cloud_classmethod_accepts_the_url_form(monkeypatch):
    """The classmethod takes the full triple and the orecloud:// spelling too."""
    captured: dict[str, object] = {}
    handle = object()
    _fake_connect(monkeypatch, captured, handle)

    assert LodeDB.cloud(CLOUD_TARGET) is handle
    assert captured["target"] == CLOUD_TARGET


def test_cloud_classmethod_missing_deps_raise_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "lodedb.cloud.serving", None)
    with pytest.raises(ImportError, match=r'pip install "lodedb\[cloud\]"'):
        LodeDB.cloud("user-42")


def test_cloud_classmethod_rejects_local_only_options(monkeypatch):
    _fake_connect(monkeypatch, {}, object())
    with pytest.raises(TypeError, match=r"do not accept model"):
        LodeDB.cloud("user-42", model="minilm")


def test_bare_store_ids_do_not_dispatch_from_the_constructor(tmp_path, monkeypatch):
    """Only `LodeDB.cloud()` accepts scheme-less short forms: a bare id passed
    to the constructor is an ordinary (relative) local path, never a cloud
    target — so the client's connect must not be consulted."""

    def _exploding_connect(target, **options):
        raise AssertionError("the constructor must not dispatch scheme-less targets")

    monkeypatch.setattr("lodedb.cloud.serving.connect", _exploding_connect)
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
