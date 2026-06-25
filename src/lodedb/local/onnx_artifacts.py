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

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

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

    cached = _find_onnx_file(cache_dir)
    if cached is not None and _has_tokenizer(cache_dir):
        return OnnxArtifact(model_name, cached, cache_dir, source="cached")

    snapshot = _try_snapshot(model_name, cache_dir)
    if snapshot is not None:
        return snapshot

    return _export_with_optimum(model_name, cache_dir)


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
    cache_dir.mkdir(parents=True, exist_ok=True)
    _copy_tokenizer_files(snapshot, cache_dir)
    model_path = cache_dir / "model.onnx"
    shutil.copy2(source_onnx, model_path)
    return OnnxArtifact(model_name, model_path, cache_dir, source="snapshot")


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
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise OnnxMaterializationError(
            "Optimum is required to export an ONNX model and is not installed "
            f"(no prebuilt ONNX snapshot for {model_name!r})."
        ) from exc
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or "").strip().splitlines()[-5:]
        raise OnnxMaterializationError(
            f"Optimum ONNX export failed for {model_name!r}: {' '.join(tail)}"
        ) from exc

    exported_onnx = _find_onnx_file(export_dir)
    if exported_onnx is None:
        raise OnnxMaterializationError(f"Optimum export produced no ONNX file for {model_name!r}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    _copy_tokenizer_files(export_dir, cache_dir)
    model_path = cache_dir / "model.onnx"
    shutil.copy2(exported_onnx, model_path)
    shutil.rmtree(export_dir, ignore_errors=True)
    return OnnxArtifact(model_name, model_path, cache_dir, source="export")


def _copy_tokenizer_files(source: Path, destination: Path) -> None:
    """Copies tokenizer/config files (json/txt/model) from a snapshot/export into the cache."""

    for path in source.rglob("*"):
        if not path.is_file() or path.suffix not in _TOKENIZER_SUFFIXES:
            continue
        target = destination / path.name
        if target.resolve() == path.resolve():
            continue
        shutil.copy2(path, target)
