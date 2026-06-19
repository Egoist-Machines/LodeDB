"""Unit tests for ``mps_exact_scan_available`` (always run; no real MPS needed).

The parity test in ``test_mps_exact_scan.py`` skips wholesale when MPS is
unavailable, so these isolate the availability probe itself with a fake ``torch``
and run on every host — including CI runners where ``torch.backends.mps.is_available()``
reports True but allocations fail. That mismatch must be reported as *unavailable*
so the parity test skips and production falls back to the CPU scan, rather than
crashing mid-build with a spurious "MPS backend out of memory".
"""

from __future__ import annotations

import types
from importlib import import_module

import lodedb.engine.mps_turbovec as mps_turbovec


class _FakeTensor:
    """Minimal stand-in supporting the probe's ``(t + t).sum().to("cpu")`` + ``float()``."""

    def __add__(self, other: object) -> _FakeTensor:
        return self

    def sum(self) -> _FakeTensor:
        return self

    def to(self, *args: object, **kwargs: object) -> _FakeTensor:
        return self

    def __float__(self) -> float:
        return 1.0


def _fake_torch(*, mps_available: bool, ones):
    """Builds a fake ``torch`` exposing only what the availability probe touches."""

    mps = types.SimpleNamespace(is_available=lambda: mps_available)
    return types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=mps),
        float16="float16",
        ones=ones,
    )


def _patch_torch(monkeypatch, fake) -> None:
    """Makes the module's ``import_module('torch')`` return ``fake``."""

    monkeypatch.setattr(
        mps_turbovec,
        "import_module",
        lambda name: fake if name == "torch" else import_module(name),
    )


def test_available_when_mps_allocation_succeeds(monkeypatch):
    """is_available() True and a working allocation -> available."""

    fake = _fake_torch(mps_available=True, ones=lambda *a, **k: _FakeTensor())
    _patch_torch(monkeypatch, fake)
    available, reason = mps_turbovec.mps_exact_scan_available()
    assert available is True
    assert reason == ""


def test_unavailable_when_mps_not_reported(monkeypatch):
    """is_available() False short-circuits to unavailable before any allocation."""

    def _should_not_be_called(*args: object, **kwargs: object):
        raise AssertionError("must not probe allocation when MPS is not reported")

    fake = _fake_torch(mps_available=False, ones=_should_not_be_called)
    _patch_torch(monkeypatch, fake)
    available, reason = mps_turbovec.mps_exact_scan_available()
    assert available is False
    assert "no usable MPS" in reason


def test_unavailable_when_advertised_but_allocation_fails(monkeypatch):
    """The CI case: MPS advertised but allocation raises -> reported unavailable."""

    def _raise_oom(*args: object, **kwargs: object):
        raise RuntimeError("MPS backend out of memory (simulated CI runner)")

    fake = _fake_torch(mps_available=True, ones=_raise_oom)
    _patch_torch(monkeypatch, fake)
    available, reason = mps_turbovec.mps_exact_scan_available()
    assert available is False
    assert "unusable" in reason
    assert "out of memory" in reason
