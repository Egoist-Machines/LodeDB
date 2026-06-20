"""Tests for local embedding device selection and the doctor report.

These avoid importing torch at module scope; the torch-dependent checks are
skipped when torch is unavailable so the module stays portable.
"""

from __future__ import annotations

import importlib.util

import pytest

from lodedb.local.backends import (
    build_local_embedding_backend,
    is_apple_silicon,
    resolve_local_device,
)
from lodedb.local.doctor import format_capability_report, local_capability_report
from lodedb.local.presets import resolve_preset


def test_resolve_device_rejects_unknown():
    """resolve_local_device rejects devices outside the supported set."""

    with pytest.raises(ValueError, match="unknown device"):
        resolve_local_device("tpu")


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_resolve_device_passes_through_explicit(device):
    """Explicit device requests are returned verbatim (not auto-resolved)."""

    assert resolve_local_device(device) == device


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_resolve_auto_picks_a_concrete_device():
    """auto resolves to a concrete device the machine actually has."""

    assert resolve_local_device("auto") in {"mps", "cuda", "cpu"}


def test_build_backend_rejects_unknown_device():
    """The backend builder rejects an unsupported device before any model load."""

    with pytest.raises(ValueError, match="unknown device"):
        build_local_embedding_backend(resolve_preset("minilm"), device="tpu")


def test_doctor_report_is_honest_about_gpu_scan():
    """The doctor report never claims GPU vector search on Apple Silicon."""

    report = local_capability_report(device="auto")
    assert "platform" in report and "compact_backend" in report
    gpu = report["gpu_vector_scan"]
    if is_apple_silicon():
        assert gpu["gpu_vector_scan_available"] is False
        assert "CUDA" in gpu["reason"]
    # The formatted report renders without error and mentions the CPU kernel.
    text = format_capability_report(report)
    assert "TurboVec" in text
    assert "CUDA/CuPy only" in text
    # The opt-in MPS scan is reported and is never the default.
    mps_scan = report["mps_vector_scan"]
    assert mps_scan["opt_in"] is True
    assert mps_scan["default_enabled"] is False
    assert "opt-in" in text
    if is_apple_silicon():
        assert mps_scan["mps_exact_scan_available"] is True


def test_doctor_report_is_honest_about_patched_core():
    """doctor reports whether the patched TurboVec APIs are actually present.

    Stock PyPI ``turbovec`` lacks the local delta/reconstruction patches; the
    report must say so (``patched core : MISSING``) rather than implying the
    patched build is present from a hardcoded source tag alone.
    """

    report = local_capability_report(device="auto")
    backend = report["compact_backend"]
    assert isinstance(backend["delta_persistence_available"], bool)
    assert isinstance(backend["reconstruction_available"], bool)

    text = format_capability_report(report)
    if backend["available"]:
        assert "patched core" in text
        both = backend["delta_persistence_available"] and backend["reconstruction_available"]
        assert ("present" in text) if both else ("MISSING" in text)
