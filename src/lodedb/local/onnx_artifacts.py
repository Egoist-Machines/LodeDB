"""Materializes ONNX model + tokenizer artifacts for the local ONNX runtime.

The ONNX embedding runtime (:class:`~lodedb.engine.embedding_backends.ONNXRuntimeEmbeddingBackend`)
needs an on-disk ``model.onnx`` plus its tokenizer files. We can't ship those in
the wheel (hundreds of MB), so they are materialized on first use and cached,
the same way sentence-transformers downloads weights lazily.

Resolution order for a model id:

1. **Cached** — a previous materialization under the cache root; no network.
2. **Prebuilt snapshot** — many Hugging Face repos already ship an ``onnx/``
   export; fetch it with ``huggingface_hub`` (no Optimum / torch needed).
3. **Optimum export** — fall back to exporting the feature-extraction graph with
   Optimum, run as a **subprocess** so Optimum/torch never import into this
   process (keeping a plain ``import lodedb`` lean).

If none succeed (offline with nothing cached, or Optimum absent), this raises
:class:`OnnxMaterializationError`, which the backend resolver treats as a signal
to fall back to the PyTorch sentence-transformers runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from lodedb.engine._atomic_io import durable_replace

# Hugging Face files needed to run an ONNX feature-extraction model + tokenizer.
# Only the base fp32 ``model.onnx`` is fetched, not the optimized/quantized
# variants some repos also publish (e.g. ``model_O3.onnx``, ``model_qint8_*.onnx``),
# so a model with several ONNX exports does not pull hundreds of extra MB.
_SNAPSHOT_ALLOW_PATTERNS = (
    "onnx/model.onnx",
    "model.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "merges.txt",
    "sentencepiece.bpe.model",
    "sentence_bert_config.json",
)
_TOKENIZER_SUFFIXES = {".json", ".txt", ".model"}
_CACHE_MANIFEST_NAME = "artifact.json"
_CACHE_SCHEMA_VERSION = 1
_DEFAULT_ONNX_EXPORT_TIMEOUT_SECONDS = 30 * 60


class OnnxMaterializationError(RuntimeError):
    """Raised when no ONNX artifact can be obtained (cache, snapshot, or export)."""


@dataclass(frozen=True)
class OnnxArtifact:
    """An on-disk ONNX model and the directory holding its tokenizer files."""

    model_name: str
    model_path: Path
    tokenizer_dir: Path
    source: str  # "cached" | "snapshot" | "export"


def onnx_cache_root() -> Path:
    """Returns the cache root for materialized ONNX artifacts.

    Honors ``LODEDB_ONNX_CACHE``; otherwise ``~/.cache/lodedb/onnx``.
    """

    override = os.environ.get("LODEDB_ONNX_CACHE")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "lodedb" / "onnx"


def _model_cache_dir(model_name: str) -> Path:
    """Returns the per-model cache directory (model id sanitized into one path segment)."""

    sanitized = model_name.replace("/", "__").replace("\\", "__")
    return onnx_cache_root() / sanitized


def _find_onnx_file(root: Path) -> Path | None:
    """Returns the model ONNX file under ``root``, preferring ``model.onnx``."""

    direct = root / "model.onnx"
    if direct.is_file():
        return direct
    candidates = sorted(root.rglob("*.onnx"))
    return candidates[0] if candidates else None


def _has_tokenizer(root: Path) -> bool:
    """Returns whether ``root`` holds a usable tokenizer (fast or vocab-based)."""

    return (root / "tokenizer.json").is_file() or (root / "vocab.txt").is_file()


def materialize_onnx_model(model_name: str) -> OnnxArtifact:
    """Returns a cached/snapshotted/exported ONNX artifact for ``model_name``.

    Raises :class:`OnnxMaterializationError` if no artifact can be produced.
    """

    if not model_name:
        raise ValueError("model_name is required")
    cache_dir = _model_cache_dir(model_name)

    with _OnnxCacheLock(_cache_lock_path(cache_dir)):
        cached = _load_cached_artifact(model_name, cache_dir)
        if cached is not None:
            return cached

        snapshot = _try_snapshot(model_name, cache_dir)
        if snapshot is not None:
            return snapshot

        return _export_with_optimum(model_name, cache_dir)


class _OnnxCacheLock:
    """Small cross-process lock around one model cache materialization."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle = None

    def __enter__(self) -> _OnnxCacheLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        handle.write(b"\0")
        handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        self._handle = handle
        return self

    def __exit__(self, *exc: object) -> None:
        if self._handle is None:
            return
        if os.name == "nt":
            import msvcrt

            self._handle.seek(0)
            msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None


def _cache_lock_path(cache_dir: Path) -> Path:
    return cache_dir.with_name(cache_dir.name + ".lock")


def _load_cached_artifact(model_name: str, cache_dir: Path) -> OnnxArtifact | None:
    """Returns a valid manifest-backed cached artifact, or ``None``."""

    manifest_path = cache_dir / _CACHE_MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return None
    if manifest.get("model_name") != model_name:
        return None
    model_file = str(manifest.get("model_file", ""))
    if not model_file or "/" in model_file or "\\" in model_file:
        return None
    model_path = cache_dir / model_file
    if not model_path.is_file() or not _has_tokenizer(cache_dir):
        return None
    expected_sha = manifest.get("model_sha256")
    if not isinstance(expected_sha, str) or _sha256_file(model_path) != expected_sha:
        return None
    return OnnxArtifact(model_name, model_path, cache_dir, source="cached")


def _try_snapshot(model_name: str, cache_dir: Path) -> OnnxArtifact | None:
    """Copies a prebuilt Hugging Face ONNX snapshot into the cache, if the repo ships one."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return None
    try:
        snapshot = Path(
            snapshot_download(repo_id=model_name, allow_patterns=list(_SNAPSHOT_ALLOW_PATTERNS))
        )
    except Exception:  # noqa: BLE001 - offline / missing repo: fall through to export
        return None
    source_onnx = _find_onnx_file(snapshot)
    if source_onnx is None or not _has_tokenizer(snapshot):
        return None
    return _publish_artifact(model_name, cache_dir, source_onnx, snapshot, source="snapshot")


def _export_with_optimum(model_name: str, cache_dir: Path) -> OnnxArtifact:
    """Exports a feature-extraction ONNX graph with Optimum, as an isolated subprocess."""

    export_dir = cache_dir / "export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    optimum_cli = shutil.which("optimum-cli")
    command_prefix = (
        [optimum_cli, "export", "onnx"]
        if optimum_cli
        else [sys.executable, "-m", "optimum.exporters.onnx"]
    )
    command = [
        *command_prefix,
        "--model",
        model_name,
        "--library-name",
        "transformers",
        "--task",
        "feature-extraction",
        "--framework",
        "pt",
        "--opset",
        "17",
        str(export_dir),
    ]
    timeout_seconds = _onnx_export_timeout_seconds()
    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(export_dir, ignore_errors=True)
        raise OnnxMaterializationError(
            f"Optimum ONNX export timed out after {timeout_seconds:.0f}s for {model_name!r}."
        ) from exc
    except FileNotFoundError as exc:
        shutil.rmtree(export_dir, ignore_errors=True)
        raise OnnxMaterializationError(
            "Optimum is required to export an ONNX model and is not installed "
            f"(no prebuilt ONNX snapshot for {model_name!r})."
        ) from exc
    except subprocess.CalledProcessError as exc:
        shutil.rmtree(export_dir, ignore_errors=True)
        tail = (exc.stderr or "").strip().splitlines()[-5:]
        raise OnnxMaterializationError(
            f"Optimum ONNX export failed for {model_name!r}: {' '.join(tail)}"
        ) from exc

    exported_onnx = _find_onnx_file(export_dir)
    if exported_onnx is None:
        shutil.rmtree(export_dir, ignore_errors=True)
        raise OnnxMaterializationError(f"Optimum export produced no ONNX file for {model_name!r}")
    artifact = _publish_artifact(model_name, cache_dir, exported_onnx, export_dir, source="export")
    shutil.rmtree(export_dir, ignore_errors=True)
    return artifact


def _publish_artifact(
    model_name: str,
    cache_dir: Path,
    source_onnx: Path,
    tokenizer_source: Path,
    *,
    source: str,
) -> OnnxArtifact:
    """Atomically publishes model/tokenizer files, then commits a checksum manifest."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    _copy_tokenizer_files(tokenizer_source, cache_dir)
    model_path = cache_dir / "model.onnx"
    _copy_file_atomic(source_onnx, model_path)
    manifest = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "model_name": model_name,
        "model_file": model_path.name,
        "model_sha256": _sha256_file(model_path),
        "source": source,
    }
    _write_json_atomic(cache_dir / _CACHE_MANIFEST_NAME, manifest)
    return OnnxArtifact(model_name, model_path, cache_dir, source=source)


def _copy_tokenizer_files(source: Path, destination: Path) -> None:
    """Copies tokenizer/config files (json/txt/model) from a snapshot/export into the cache."""

    for path in source.rglob("*"):
        if not path.is_file() or path.suffix not in _TOKENIZER_SUFFIXES:
            continue
        target = destination / path.name
        if target.resolve() == path.resolve():
            continue
        _copy_file_atomic(path, target)


def _copy_file_atomic(source: Path, target: Path) -> None:
    """Copies one file to a sibling temp file before atomically publishing it."""

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copy2(source, tmp)
        durable_replace(tmp, target, fsync=False)
    finally:
        tmp.unlink(missing_ok=True)


def _write_json_atomic(target: Path, payload: dict[str, object]) -> None:
    """Writes JSON via the same sibling-temp + replace discipline as persisted state."""

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        durable_replace(tmp, target, fsync=False)
    finally:
        tmp.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _onnx_export_timeout_seconds() -> float:
    value = os.environ.get("LODEDB_ONNX_EXPORT_TIMEOUT_SECONDS")
    if value is None or value.strip() == "":
        return float(_DEFAULT_ONNX_EXPORT_TIMEOUT_SECONDS)
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError("LODEDB_ONNX_EXPORT_TIMEOUT_SECONDS must be a number") from exc
    if timeout <= 0:
        raise ValueError("LODEDB_ONNX_EXPORT_TIMEOUT_SECONDS must be greater than zero")
    return timeout
