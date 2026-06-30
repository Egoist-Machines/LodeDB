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


def test_preset_backend_errors_without_embedding_runtime(monkeypatch):
    """A preset text model with no embedding runtime installed raises a clear install hint.

    Built-in text embedding is opt-in (the [embeddings]/[torch] extras); requesting a preset
    must point at the extra rather than surfacing a deep ModuleNotFoundError at encode time.
    """

    import lodedb.local.backends as backends

    monkeypatch.setattr(backends, "onnxruntime_available", lambda: False)
    monkeypatch.setattr(backends, "sentence_transformers_available", lambda: False)
    with pytest.raises(ModuleNotFoundError, match=r"lodedb\[embeddings\]"):
        build_local_embedding_backend(resolve_preset("minilm"), device="cpu")


def test_forced_torch_runtime_errors_without_sentence_transformers(monkeypatch):
    """embedding_runtime='torch' without the [torch] tier points at lodedb[embeddings,torch]."""

    import lodedb.local.backends as backends

    monkeypatch.setattr(backends, "sentence_transformers_available", lambda: False)
    with pytest.raises(ModuleNotFoundError, match=r"lodedb\[embeddings,torch\]"):
        build_local_embedding_backend(
            resolve_preset("minilm"), device="cpu", embedding_runtime="torch"
        )


def test_forced_onnx_runtime_errors_without_onnxruntime(monkeypatch):
    """embedding_runtime='onnx' without the [embeddings] tier raises the hint up front.

    Forcing ONNX must not enter the materialize/encode path when onnxruntime is absent (which
    would fail deep, or hand back a backend that only blows up at first encode); it points at
    lodedb[embeddings].
    """

    import lodedb.local.backends as backends

    monkeypatch.setattr(backends, "onnxruntime_available", lambda: False)
    # A working torch tier must not mask the forced-onnx request — the hint still wins.
    monkeypatch.setattr(backends, "sentence_transformers_available", lambda: True)
    with pytest.raises(ModuleNotFoundError, match=r"lodedb\[embeddings\]"):
        build_local_embedding_backend(
            resolve_preset("minilm"), device="cpu", embedding_runtime="onnx"
        )


def test_clip_preset_errors_without_sentence_transformers(monkeypatch):
    """The clip preset without sentence-transformers points at the [image] extra."""

    import lodedb.local.backends as backends

    monkeypatch.setattr(backends, "sentence_transformers_available", lambda: False)
    with pytest.raises(ModuleNotFoundError, match=r"lodedb\[image\]"):
        build_local_embedding_backend(resolve_preset("clip"), device="cpu")


def test_doctor_report_is_honest_about_gpu_scan():
    """The doctor report never claims GPU vector search on Apple Silicon.

    The GPU-resident scan runs in the native core (cudarc), so availability is
    sourced from the native CUDA-driver probe, not a torch/CuPy proxy.
    """

    report = local_capability_report(device="auto")
    assert "platform" in report and "compact_backend" in report
    gpu = report["gpu_vector_scan"]
    assert "native_core_available" in gpu
    assert "cupy_present" not in gpu
    if is_apple_silicon():
        assert gpu["gpu_vector_scan_available"] is False
        assert "CUDA" in gpu["reason"]
    # The formatted report renders without error and mentions the CPU kernel.
    text = format_capability_report(report)
    assert "TurboVec" in text
    assert "GPU-resident vector scan (native core, CUDA driver only)" in text


def test_torch_cuda_build_version_returns_none_or_str():
    """The build-version probe returns None (CPU build or torch absent) or a version string."""

    from lodedb.local.backends import torch_cuda_build_version

    value = torch_cuda_build_version()
    assert value is None or isinstance(value, str)


def test_windows_gpu_hint_absent_off_windows(monkeypatch):
    """No Windows GPU hint is emitted on non-Windows platforms."""

    import lodedb.local.doctor as doctor

    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    assert doctor._windows_gpu_embedding_hint() is None
    assert local_capability_report(device="auto")["windows_gpu_hint"] is None


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_windows_gpu_hint_present_for_cpu_torch_on_windows(monkeypatch):
    """On Windows with a CPU-only torch build, the hint carries the CUDA reinstall command."""

    import lodedb.local.doctor as doctor

    monkeypatch.setattr(doctor.platform, "system", lambda: "Windows")
    monkeypatch.setattr(doctor, "torch_cuda_build_version", lambda: None)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: "C:\\nvidia-smi.exe")

    hint = doctor._windows_gpu_embedding_hint()
    assert hint is not None
    assert hint["torch_cuda_build"] is False
    assert hint["nvidia_smi_detected"] is True
    assert "download.pytorch.org/whl/cu" in hint["command"]

    report = local_capability_report(device="auto")
    text = format_capability_report(report)
    assert "Windows GPU embeddings" in text
    assert "force-reinstall" in text


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_windows_gpu_hint_absent_for_cuda_torch_build(monkeypatch):
    """No hint when torch is already a CUDA build, even on Windows."""

    import lodedb.local.doctor as doctor

    monkeypatch.setattr(doctor.platform, "system", lambda: "Windows")
    monkeypatch.setattr(doctor, "torch_cuda_build_version", lambda: "12.1")
    assert doctor._windows_gpu_embedding_hint() is None


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
