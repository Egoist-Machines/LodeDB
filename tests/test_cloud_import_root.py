"""The `lodedb.cloud` import root: PEP 562 lazy resolution of the first-party
cloud client. HTTP-backed names (`Client`, `connect`, …) come from their
submodule with an install hint when the [cloud] extra's dependencies are
absent; the transfer verbs (`push`, `pull`, …) ride the bundled native
extension and need no extra at all. The laziness itself (importing
`lodedb.cloud` must not load httpx) is guarded by the fresh-subprocess probe
in `tests/test_import_boundary.py`."""

from __future__ import annotations

import sys

import pytest

# The tests below import (or monkeypatch into) the client modules, which
# pull httpx/pynacl, so skip cleanly without the [cloud] extra installed.
pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
pytest.importorskip("nacl", reason="needs the [cloud] extra's dependencies")

from lodedb import _turbovec, cloud


def test_http_names_resolve_to_their_submodules():
    from lodedb.cloud.client import Client
    from lodedb.cloud.serving import CloudStore, connect

    assert cloud.Client is Client
    assert cloud.connect is connect
    assert cloud.CloudStore is CloudStore
    assert cloud.CloudIndex is CloudStore  # back-compat alias


def test_transfer_verbs_resolve_to_the_native_extension():
    for name in ("keys", "pull", "push", "status", "sync", "verify"):
        assert getattr(cloud, name) is getattr(_turbovec.cloud, name)


def test_missing_cloud_deps_raise_install_hint(monkeypatch):
    # A None entry makes the submodule import raise ImportError even though
    # the extra's dependencies are installed in the dev venv.
    monkeypatch.setitem(sys.modules, "lodedb.cloud.client", None)
    with pytest.raises(ImportError, match=r'pip install "lodedb\[cloud\]"'):
        cloud.Client  # noqa: B018  # the attribute fetch is the behavior under test


def test_unknown_names_stay_attribute_errors():
    """A name outside the public surface is a plain AttributeError, not a
    misleading install hint."""
    with pytest.raises(AttributeError):
        cloud.NoSuchName  # noqa: B018


def test_underscore_probes_never_touch_the_lazy_table(monkeypatch):
    """Attribute machinery probes (`__path__`-style dunders) must resolve
    locally; blocked submodules stay untouched."""
    monkeypatch.setitem(sys.modules, "lodedb.cloud.client", None)
    monkeypatch.setitem(sys.modules, "lodedb.cloud.serving", None)
    with pytest.raises(AttributeError):
        cloud._private_probe  # noqa: B018
