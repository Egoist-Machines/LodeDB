"""Capability and native-store reports behind ``lodedb doctor``.

Reuses the existing :func:`turbovec_capability` probe and CPU-flag detection
rather than reimplementing any capability logic. Reports, honestly:

- whether this is Apple Silicon;
- the embedding device that ``device="auto"`` resolves to, plus MPS/CUDA
  availability;
- the compact (TurboVec) backend status and inferred native dispatch;
- whether the GPU-resident vector-scan path exists. That scan runs in the
  bundled native core (cudarc), so the probe is the native CUDA-driver check,
  not torch or CuPy; on Apple Silicon there is no CUDA driver and the NEON CPU
  kernel is the accelerated path.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

from lodedb.engine.turbovec_index import (
    detect_cpu_flags,
    turbovec_capability,
    turbovec_native_backend_from_flags,
)
from lodedb.local.backends import (
    is_apple_silicon,
    onnxruntime_available,
    resolve_local_device,
    sentence_transformers_available,
    torch_cuda_available,
    torch_cuda_build_version,
    torch_mps_available,
)

# Windows PyPI serves a CPU-only torch wheel by default; this index has the CUDA builds.
# cu121 is a broadly driver-compatible default; other toolkits live at /whl/cu124, etc.
_PYTORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
_TVVF_BASE_MAGIC = b"EEVFB001"
_TVVF_DELTA_MAGIC = b"EEVFD001"
_TVVF_FNV_OFFSET = 0xCBF2_9CE4_8422_2325
_TVVF_FNV_PRIME = 0x0000_0100_0000_01B3
_TVVF_SCHEMA_VERSION = 1
_TVVF_HEADER_MAX_BYTES = 64 * 1024 * 1024
_TVVF_READ_BYTES = 64 * 1024


def _doctor_finding(
    *,
    status: str,
    index_key: str,
    message: str,
    coverage: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Builds the stable doctor finding shape for one TVVF sidecar."""

    finding: dict[str, Any] = {
        "name": "tvvf_sidecar",
        "status": status,
        "index_key": index_key,
        "message": message,
    }
    if coverage is not None:
        finding["coverage"] = coverage
    return finding


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer at least {minimum}")
    return value


def _require_file_name(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ValueError(f"{label} must be a file name")
    return value


def _fnv1a64_update(current: int, data: bytes) -> int:
    """Matches the native TVVF id-section FNV-1a checksum."""

    for byte in data:
        current ^= byte
        current = (current * _TVVF_FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return current


def _read_exact(file: BinaryIO, size: int, label: str) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError(f"{label} is truncated")
    return data


def _row_stride(dtype: str, dim: int) -> int:
    if dtype == "float16":
        return dim * 2
    if dtype == "float32":
        return dim * 4
    if dtype == "int8":
        # A 4-byte per-row scale precedes the dim int8 codes.
        return 4 + dim
    raise ValueError("TVVF header has an invalid dtype")


def _read_tvvf_segment(
    path: Path,
    *,
    magic: bytes,
    is_delta: bool,
    ids_of_interest: set[int] | None = None,
) -> dict[str, Any]:
    """Validates one segment header and ids while streaming its SHA-256.

    The full-file digest is deliberate here. A doctor run is the one operator
    command allowed to pay for hashing a very large (including 32 GB) sidecar.
    """

    file_size = path.stat().st_size
    sha = hashlib.sha256()
    with path.open("rb") as file:
        prefix = _read_exact(file, 16, f"TVVF segment {path.name}")
        sha.update(prefix)
        if prefix[:8] != magic:
            kind = "delta" if is_delta else "base"
            raise ValueError(f"TVVF segment {path.name} is not a {kind} segment")
        header_size = int.from_bytes(prefix[8:], "little")
        if header_size > _TVVF_HEADER_MAX_BYTES or 16 + header_size > file_size:
            raise ValueError(f"TVVF segment {path.name} has an invalid header length")
        header_bytes = _read_exact(file, header_size, f"TVVF header {path.name}")
        sha.update(header_bytes)
        try:
            header = _require_mapping(json.loads(header_bytes), f"TVVF header {path.name}")
        except json.JSONDecodeError as error:
            raise ValueError(f"TVVF header {path.name} is invalid JSON") from error
        if header.get("schema_version") != _TVVF_SCHEMA_VERSION:
            raise ValueError(f"TVVF header {path.name} has an unsupported schema version")
        dtype = header.get("dtype")
        if not isinstance(dtype, str):
            raise ValueError(f"TVVF header {path.name} is missing dtype")
        dim = _require_int(header.get("dim"), f"TVVF header {path.name} dim", minimum=1)
        row_count = _require_int(
            header.get("row_count"), f"TVVF header {path.name} row_count", minimum=0
        )
        if row_count > 0xFFFFFFFF:
            raise ValueError(f"TVVF header {path.name} row_count exceeds the format limit")
        ids_checksum = _require_int(
            header.get("ids_checksum"), f"TVVF header {path.name} ids_checksum", minimum=0
        )
        deleted_ids: list[int] = []
        if is_delta:
            raw_deleted = header.get("deleted_ids")
            if not isinstance(raw_deleted, list):
                raise ValueError(f"TVVF delta {path.name} is missing deleted_ids")
            deleted_ids = [
                _require_int(value, f"TVVF delta {path.name} deleted id", minimum=0)
                for value in raw_deleted
            ]
            if len(set(deleted_ids)) != len(deleted_ids):
                raise ValueError(f"TVVF delta {path.name} has duplicate deleted ids")
        stride = _row_stride(dtype, dim)
        expected_size = 16 + header_size + row_count * (8 + 4 + stride)
        if file_size != expected_size:
            raise ValueError(f"TVVF segment {path.name} has an invalid length")

        ids_checksum_actual = _TVVF_FNV_OFFSET
        ids: list[int] = []
        remaining = row_count * 8
        while remaining:
            chunk = _read_exact(file, min(remaining, _TVVF_READ_BYTES), f"TVVF ids {path.name}")
            sha.update(chunk)
            ids_checksum_actual = _fnv1a64_update(ids_checksum_actual, chunk)
            if ids_of_interest is None:
                ids.extend(value[0] for value in struct.iter_unpack("<Q", chunk))
            else:
                ids.extend(
                    value
                    for (value,) in struct.iter_unpack("<Q", chunk)
                    if value in ids_of_interest
                )
            remaining -= len(chunk)
        if ids_checksum_actual != ids_checksum:
            raise ValueError(f"TVVF ids section failed checksum for {path.name}")
        if ids_of_interest is None and len(set(ids)) != len(ids):
            raise ValueError(f"TVVF segment {path.name} contains duplicate stable ids")
        if is_delta and set(ids).intersection(deleted_ids):
            raise ValueError(f"TVVF delta {path.name} upserts and deletes the same stable id")

        remaining = file_size - file.tell()
        while remaining:
            chunk = _read_exact(
                file, min(remaining, _TVVF_READ_BYTES), f"TVVF payload {path.name}"
            )
            sha.update(chunk)
            remaining -= len(chunk)

    return {
        "dtype": dtype,
        "dim": dim,
        "row_count": row_count,
        "ids_checksum": ids_checksum,
        "ids": ids,
        "deleted_ids": deleted_ids,
        "file_bytes": file_size,
        "sha256": sha.hexdigest(),
    }


def _validate_tvvf_sidecar(index_key: str, root: Path, manifest: Any) -> dict[str, int]:
    """Checks a committed TVVF manifest and returns its metrics-only coverage."""

    manifest = _require_mapping(manifest, "committed TVVF manifest")
    if manifest.get("schema_version") != _TVVF_SCHEMA_VERSION:
        raise ValueError("TVVF manifest has an unsupported schema version")
    if manifest.get("index_key") != index_key:
        raise ValueError("TVVF manifest identity does not match the committed index key")
    vf_epoch = _require_int(manifest.get("vf_epoch"), "TVVF manifest vf_epoch", minimum=1)
    base_manifest = _require_mapping(manifest.get("base"), "TVVF manifest base")
    base_name = _require_file_name(base_manifest.get("file_name"), "TVVF base file_name")
    expected_base_name = f"vf{vf_epoch}.tvvf"
    if base_name != expected_base_name:
        raise ValueError("TVVF manifest identity does not match the base path")
    base_path = root / f"{index_key}.gen" / base_name
    deltas = manifest.get("deltas")
    if not isinstance(deltas, list):
        raise ValueError("TVVF manifest deltas must be an array")

    parsed_deltas: list[dict[str, Any]] = []
    touched_ids: set[int] = set()
    previous_sequence = -1
    delta_dir = base_path.with_name(f"{base_path.name}.tvvf-delta")
    for entry in deltas:
        entry = _require_mapping(entry, "TVVF delta manifest entry")
        sequence = _require_int(entry.get("seq"), "TVVF delta seq", minimum=0)
        if sequence <= previous_sequence:
            raise ValueError("TVVF manifest has out-of-order delta segments")
        previous_sequence = sequence
        name = _require_file_name(entry.get("file_name"), "TVVF delta file_name")
        expected_sha = entry.get("sha256")
        if not isinstance(expected_sha, str) or not expected_sha:
            raise ValueError("TVVF delta manifest is missing sha256")
        delta = _read_tvvf_segment(delta_dir / name, magic=_TVVF_DELTA_MAGIC, is_delta=True)
        if delta["sha256"] != expected_sha:
            raise ValueError(f"TVVF delta {name} failed full-file SHA-256 validation")
        if entry.get("file_bytes") != delta["file_bytes"]:
            raise ValueError(f"TVVF delta {name} file size does not match its manifest")
        if entry.get("upsert_rows") != delta["row_count"]:
            raise ValueError(f"TVVF delta {name} upsert row count does not match its manifest")
        if entry.get("deleted_rows") != len(delta["deleted_ids"]):
            raise ValueError(f"TVVF delta {name} deleted row count does not match its manifest")
        touched_ids.update(delta["ids"])
        touched_ids.update(delta["deleted_ids"])
        parsed_deltas.append(delta)

    base = _read_tvvf_segment(
        base_path,
        magic=_TVVF_BASE_MAGIC,
        is_delta=False,
        ids_of_interest=touched_ids,
    )
    if base["sha256"] != base_manifest.get("sha256"):
        raise ValueError("TVVF base failed full-file SHA-256 validation")
    if base_manifest.get("file_bytes") != base["file_bytes"]:
        raise ValueError("TVVF base file size does not match its manifest")
    for key in ("dtype", "dim", "row_count", "ids_checksum"):
        if base_manifest.get(key) != base[key]:
            raise ValueError(f"TVVF base {key} does not match its header")

    base_touched = set(base["ids"])
    live_rows = base["row_count"]
    latest: dict[int, bool] = {}
    for delta in parsed_deltas:
        if delta["dtype"] != base["dtype"] or delta["dim"] != base["dim"]:
            raise ValueError("TVVF delta dtype or dimension does not match the base")
        for stable_id in delta["deleted_ids"]:
            was_live = latest.get(stable_id, stable_id in base_touched)
            if was_live:
                live_rows -= 1
            latest[stable_id] = False
        for stable_id in delta["ids"]:
            was_live = latest.get(stable_id, stable_id in base_touched)
            if not was_live:
                live_rows += 1
            latest[stable_id] = True
    return {
        "sidecar_rows": max(live_rows, 0),
        "tombstones": sum(not present for present in latest.values()),
        "base_rows": base["row_count"],
        "delta_segments": len(parsed_deltas),
    }


def native_store_findings(path: str | Path) -> list[dict[str, Any]]:
    """Returns non-mutating TVVF findings for committed native stores at ``path``.

    Every failure becomes a finding so an operator can inspect or restore the
    sidecar while the vector store remains available. Doctor only reports. It
    never deletes, repairs, or otherwise changes sidecar files.
    """

    root = Path(path)
    findings: list[dict[str, Any]] = []
    if not root.is_dir():
        return [
            _doctor_finding(
                status="fail",
                index_key="",
                message=(
                    f"Native store path is not an existing directory: {root}. Nothing was "
                    "inspected. Doctor reports only and never removes files."
                ),
            )
        ]
    try:
        manifests: Iterable[Path] = sorted(root.glob("*.commit.json"))
    except OSError as error:
        return [
            _doctor_finding(
                status="fail",
                index_key="",
                message=(
                    f"Could not inspect native store: {error}. Doctor reports only and "
                    "never removes files."
                ),
            )
        ]
    for commit_path in manifests:
        index_key = commit_path.name.removesuffix(".commit.json")
        try:
            commit = _require_mapping(json.loads(commit_path.read_text(encoding="utf-8")), "commit")
            # Trust the embedded tvvf entry only after the root wrap validates the
            # same way the native loader does; a corrupt root is its own finding.
            if commit.get("schema_version") != 1:
                raise ValueError("unsupported commit manifest schema version")
            body = _require_mapping(commit.get("body"), "commit body")
            stored_sha = commit.get("body_sha256")
            body_sha = hashlib.sha256(
                json.dumps(body, sort_keys=True).encode("utf-8")
            ).hexdigest()
            if not isinstance(stored_sha, str) or stored_sha != body_sha:
                raise ValueError("commit manifest failed body checksum")
            tvvf_manifest = body.get("tvvf")
            if tvvf_manifest is None:
                continue
            coverage = _validate_tvvf_sidecar(index_key, root, tvvf_manifest)
        except Exception as error:  # noqa: BLE001 - doctor reports malformed stores as findings
            findings.append(
                _doctor_finding(
                    status="fail",
                    index_key=index_key,
                    message=(
                        f"Commit manifest or TVVF sidecar validation failed: {error}. Doctor "
                        "reports the problem and never "
                        "removes files. Full-file SHA-256 validation may read a 32 GB base."
                    ),
                )
            )
        else:
            findings.append(
                _doctor_finding(
                    status="ok",
                    index_key=index_key,
                    coverage=coverage,
                    message=(
                        "TVVF manifest identity, base header, ids checksum, and full-file SHA-256 "
                        "validated. Full-file SHA-256 validation may read a 32 GB base; "
                        "doctor only "
                        "reports and never removes files."
                    ),
                )
            )
    return findings


def _windows_gpu_embedding_hint() -> dict[str, Any] | None:
    """On Windows with a CPU-only PyTorch, returns how to switch to a CUDA build.

    PyPI serves the CPU-only torch wheel on Windows by default, so installing the PyTorch tier
    (``pip install 'lodedb[torch]'``) leaves embeddings on the CPU even on an NVIDIA machine, and
    no package metadata can redirect torch to the CUDA index. Returns ``None`` when it does not
    apply: off Windows, when torch is absent, or when torch is already a CUDA build.
    """

    if platform.system() != "Windows":
        return None
    try:
        import torch  # noqa: F401  (presence check; a broken install is a different problem)
    except ImportError:
        return None
    if torch_cuda_build_version() is not None:
        return None  # already a CUDA build
    nvidia_detected = shutil.which("nvidia-smi") is not None
    return {
        "torch_cuda_build": False,
        "nvidia_smi_detected": nvidia_detected,
        "index_url": _PYTORCH_CUDA_INDEX,
        "command": (
            f"pip install torch --force-reinstall --no-deps --index-url {_PYTORCH_CUDA_INDEX}"
        ),
    }


def _image_embedding_status() -> dict[str, Any]:
    """Returns whether the optional image+text (CLIP) embedding path is installed.

    The ``"clip"`` preset / ``add_image`` runs on the sentence-transformers stack (the
    ``[torch]`` tier) and adds Pillow for decoding image files; the ``[image]`` extra pulls
    both.
    """

    import importlib.util

    pillow_present = importlib.util.find_spec("PIL") is not None
    st_present = importlib.util.find_spec("sentence_transformers") is not None
    available = bool(pillow_present and st_present)
    if available:
        reason = "Pillow + sentence-transformers present (use model='clip')"
    else:
        missing = []
        if not pillow_present:
            missing.append("Pillow")
        if not st_present:
            missing.append("sentence-transformers")
        reason = f"image embedding requires {' + '.join(missing)} (pip install 'lodedb[image]')"
    return {
        "image_embedding_available": available,
        "pillow_present": pillow_present,
        "model": "sentence-transformers/clip-ViT-B-32",
        "reason": reason,
    }


def _gpu_vector_scan_status() -> dict[str, Any]:
    """Returns honest GPU-resident vector-scan availability from the native probe.

    The GPU-resident scan runs in the bundled native core (cudarc), so this
    reports the real CUDA-driver probe the native scan gates on rather than a
    torch/CuPy proxy: it needs neither. ``native_core_available`` distinguishes
    "no CUDA driver" from "the native extension did not load".
    """

    from lodedb.engine.native_adapter import NativeCoreAdapter

    adapter = NativeCoreAdapter()
    native_core_available = adapter.available
    available = bool(adapter.cuda_runtime_available())
    if available:
        reason = "native core + CUDA driver present"
    elif is_apple_silicon():
        reason = (
            "Apple Silicon: the CUDA GPU vector scan is unavailable here; the "
            "NEON CPU kernel is the accelerated path."
        )
    elif not native_core_available:
        reason = "GPU vector scan requires the bundled native core, which is not loaded"
    else:
        reason = "GPU vector scan requires a CUDA driver (none detected)"
    return {
        "gpu_vector_scan_available": available,
        "native_core_available": native_core_available,
        "reason": reason,
    }


def _embedding_runtime_status() -> dict[str, Any]:
    """Reports the embedding runtime ``embedding_runtime="auto"`` prefers, plus ONNX details.

    ``auto`` uses ONNX Runtime when ``onnxruntime`` is installed *and* the model's ONNX graph
    can be obtained (cached, a prebuilt Hub snapshot, or an Optimum export), falling back to
    PyTorch sentence-transformers otherwise. Built-in text embedding is the optional
    ``[embeddings]`` extra (ONNX) plus ``[torch]`` (the PyTorch tier); with neither installed
    LodeDB is a vector store (bring your own vectors / ``embedder=``). This probe reports a
    *preference*, not a guarantee, without forcing a per-model download.
    """

    onnx = onnxruntime_available()
    torch_runtime = sentence_transformers_available()
    providers: list[str] = []
    if onnx:
        try:
            import onnxruntime as ort

            providers = list(ort.get_available_providers())
        except Exception:  # noqa: BLE001 - report none if probing fails
            providers = []
    if onnx:
        preferred = "onnx"
        note = (
            "auto prefers ONNX; falls back to PyTorch if a model's ONNX graph is unavailable"
            if torch_runtime
            else "auto uses ONNX; install lodedb[torch] for the PyTorch fallback"
        )
    elif torch_runtime:
        preferred = "torch"
        note = "onnxruntime not installed; auto uses PyTorch (install lodedb[embeddings] for ONNX)"
    else:
        preferred = "none"
        note = (
            "no embedding runtime installed; text embedding is unavailable "
            "(pip install 'lodedb[embeddings]', or bring your own vectors)"
        )
    return {
        "preferred": preferred,
        "onnxruntime_available": onnx,
        "sentence_transformers_available": torch_runtime,
        "onnx_providers": providers,
        "note": note,
    }


def local_capability_report(
    *, device: str = "auto", path: str | Path | None = None
) -> dict[str, Any]:
    """Builds the local capability report, plus optional native-store findings."""

    apple_silicon = is_apple_silicon()
    mps = torch_mps_available()
    cuda = torch_cuda_available()
    effective_device = resolve_local_device(device)

    cpu_flags = detect_cpu_flags()
    capability = turbovec_capability()

    runtime_status = _embedding_runtime_status()
    # The silent-CPU-fallback footgun: the device resolves to CUDA but the installed onnxruntime
    # wheel is CPU-only, so text embedding runs on the CPU. Key on the *resolved* device, which
    # covers both device="auto" picking CUDA and an explicit --device cuda (the setup the docs
    # recommend when torch is absent and auto cannot see the GPU), and does not fire for a
    # deliberate --device cpu on a CUDA host. Flag it rather than leaving the operator to compare
    # the CUDA and providers lines by eye.
    cpu_fallback_warning = ""
    if (
        effective_device == "cuda"
        and runtime_status["onnxruntime_available"]
        and "CUDAExecutionProvider" not in runtime_status["onnx_providers"]
    ):
        cpu_fallback_warning = (
            "the device resolves to CUDA but ONNX Runtime has no CUDAExecutionProvider: text "
            "embedding runs on the CPU. Install onnxruntime-gpu (pip install onnxruntime-gpu) to "
            "use the GPU."
        )

    report = {
        "platform": {
            "apple_silicon": apple_silicon,
        },
        "embedding": {
            "requested_device": device,
            "auto_resolves_to": effective_device,
            "mps_available": mps,
            "cuda_available": cuda,
            "runtime": runtime_status,
            "cpu_fallback_warning": cpu_fallback_warning,
        },
        "image_embedding": _image_embedding_status(),
        "compact_backend": {
            **capability.to_dict(),
            "inferred_native_dispatch": turbovec_native_backend_from_flags(cpu_flags),
        },
        "gpu_vector_scan": _gpu_vector_scan_status(),
        "windows_gpu_hint": _windows_gpu_embedding_hint(),
    }
    if path is not None:
        report["store"] = {"path": str(path), "findings": native_store_findings(path)}
    return report


def format_capability_report(report: dict[str, Any]) -> str:
    """Renders a human-readable capability report for the CLI."""

    plat = report["platform"]
    emb = report["embedding"]
    img = report["image_embedding"]
    backend = report["compact_backend"]
    gpu = report["gpu_vector_scan"]
    lines = [
        "LodeDB doctor — local capability report",
        "=" * 42,
        f"Apple Silicon            : {plat['apple_silicon']}",
        "",
        "Embedding (accelerated stage)",
        f"  requested device       : {emb['requested_device']}",
        f"  auto resolves to       : {emb['auto_resolves_to']}",
        f"  MPS available          : {emb['mps_available']}",
        f"  CUDA available         : {emb['cuda_available']}",
        f"  runtime (auto prefers) : {emb['runtime']['preferred']}",
        f"    fallback             : {emb['runtime']['note']}",
        f"  onnxruntime available  : {emb['runtime']['onnxruntime_available']}",
        f"  sentence-transformers  : {emb['runtime']['sentence_transformers_available']}",
        f"  onnx providers         : {', '.join(emb['runtime']['onnx_providers']) or 'none'}",
        *([f"  ! {emb['cpu_fallback_warning']}"] if emb.get("cpu_fallback_warning") else []),
        "",
        "Image + text embedding (CLIP, optional [image] extra)",
        f"  available              : {img['image_embedding_available']}",
        f"  model                  : {img['model']}",
        f"  reason                 : {img['reason']}",
        "",
        "Compact storage backend (TurboVec)",
        f"  available              : {backend['available']}",
        f"  native dispatch        : {backend.get('native_backend', '?')}"
        f" (inferred {backend.get('inferred_native_dispatch', '?')})",
        f"  vendored target        : turbovec {backend.get('version', '?')}"
        f" (tag {backend.get('source_tag', '?')})",
    ]
    if backend["available"]:
        delta = backend.get("delta_persistence_available", False)
        recon = backend.get("reconstruction_available", False)
        lines += [
            f"  patched core           : "
            f"{'present' if (delta and recon) else 'MISSING (stock PyPI turbovec)'}",
            f"    delta persistence    : {delta}"
            f" ({'incremental .tvd deltas' if delta else 'unavailable — full .tvim rewrites'})",
            f"    exact reconstruction : {recon}"
            f" ({'CUDA exact serving' if recon else 'unavailable, CPU scan only'})",
        ]
    else:
        lines.append(f"  unavailable reason     : {backend.get('unavailable_reason', '')}")
    lines += [
        "",
        "GPU-resident vector scan (native core, CUDA driver only)",
        f"  available              : {gpu['gpu_vector_scan_available']}",
        f"  reason                 : {gpu['reason']}",
    ]
    hint = report.get("windows_gpu_hint")
    if hint:
        gpu_state = "detected" if hint["nvidia_smi_detected"] else "if you have one"
        lines += [
            "",
            "Windows GPU embeddings",
            "  PyTorch build          : CPU-only (no CUDA)",
            f"  NVIDIA GPU             : {gpu_state}",
            f"  to embed on the GPU    : {hint['command']}",
            "    or run `lodedb doctor --fix`; see https://pytorch.org/get-started/locally/",
            "    for the index matching your CUDA version (cu121, cu124, ...).",
        ]
    store = report.get("store")
    if isinstance(store, dict):
        findings = store.get("findings")
        lines += ["", "Native store checks", f"  path                   : {store.get('path', '')}"]
        if not findings:
            lines.append("  TVVF sidecars          : none committed")
        elif isinstance(findings, list):
            for finding in findings:
                if not isinstance(finding, dict):
                    continue
                coverage = finding.get("coverage")
                summary = ""
                if isinstance(coverage, dict):
                    summary = (
                        f" (live rows {coverage.get('sidecar_rows')}, "
                        f"tombstones {coverage.get('tombstones')})"
                    )
                lines += [
                    f"  {finding.get('name', 'finding')} [{finding.get('status', '?')}]"
                    f"{summary}",
                    f"    {finding.get('message', '')}",
                ]
    return "\n".join(lines)
