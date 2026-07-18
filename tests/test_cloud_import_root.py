"""The `lodedb.cloud` import root: a PEP 562 proxy that forwards public names
to the optional `orecloud` companion, so docs can teach one import root
(`from lodedb.cloud import Client`). Forwarding and the install-hint path are
forced with a fake (or blocked) `orecloud`, so the suite never depends on
whether the real client is installed; the laziness itself (importing
`lodedb.cloud` must not load orecloud) is guarded by the fresh-subprocess
probe in `tests/test_import_boundary.py`."""

from __future__ import annotations

import sys
import types

import pytest

from lodedb import cloud


def test_public_names_forward_to_the_companion(monkeypatch):
    sentinel_client = object()
    fake_pkg = types.ModuleType("orecloud")
    fake_pkg.Client = sentinel_client
    monkeypatch.setitem(sys.modules, "orecloud", fake_pkg)

    assert cloud.Client is sentinel_client


def test_missing_companion_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "orecloud", None)
    with pytest.raises(ImportError, match=r'pip install "lodedb\[cloud\]"'):
        cloud.Client  # noqa: B018 — the attribute fetch is the behavior under test


def test_unknown_companion_names_stay_attribute_errors(monkeypatch):
    """A name the companion doesn't export is a plain AttributeError, not a
    misleading install hint."""
    monkeypatch.setitem(sys.modules, "orecloud", types.ModuleType("orecloud"))
    with pytest.raises(AttributeError):
        cloud.NoSuchName  # noqa: B018


def test_underscore_probes_never_import_the_companion(monkeypatch):
    """Attribute machinery probes (`__path__`-style dunders) must not trigger
    the companion import — a blocked orecloud stays untouched."""
    monkeypatch.setitem(sys.modules, "orecloud", None)
    with pytest.raises(AttributeError):
        cloud._private_probe  # noqa: B018
