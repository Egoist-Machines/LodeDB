"""Minimal configuration helpers for the LodeDB engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    """Loads a YAML config file with a clear error when PyYAML is unavailable."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only without optional deps
        raise RuntimeError("Install PyYAML to load YAML configs.") from exc
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at top level of {path}")
    return loaded
