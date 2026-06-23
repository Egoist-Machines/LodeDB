"""Tests for ``lodedb mcp install`` config generation and idempotency.

Exercises helpers directly with ``tmp_path`` — no real config files touched.
"""

from __future__ import annotations

import json
from pathlib import Path

from lodedb.local.mcp_install import (
    _append_toml,
    _claude_code_cmd,
    _config_paths,
    _server_entry,
    _toml_block,
    _write_json,
    install,
)

# -- _server_entry -----------------------------------------------------------


class TestServerEntry:
    def test_defaults_omit_optional_flags(self) -> None:
        entry = _server_entry({"path": None, "model": "minilm", "device": "auto"})
        assert entry["args"] == ["-m", "lodedb", "mcp"]

    def test_path_resolved_to_absolute(self, tmp_path: Path) -> None:
        entry = _server_entry({"path": tmp_path / "db"})
        idx = entry["args"].index("--path")
        assert Path(entry["args"][idx + 1]).is_absolute()

    def test_non_default_model_forwarded(self) -> None:
        entry = _server_entry({"model": "bge"})
        assert "--model" in entry["args"]

    def test_non_default_device_forwarded(self) -> None:
        entry = _server_entry({"device": "cuda"})
        assert "--device" in entry["args"]

    def test_no_store_text(self) -> None:
        assert "--no-store-text" in _server_entry({"store_text": False})["args"]

    def test_store_text_true_omits_flag(self) -> None:
        assert "--no-store-text" not in _server_entry({"store_text": True})["args"]


# -- _write_json --------------------------------------------------------------


class TestWriteJson:
    def test_creates_new_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        entry = {"command": "python", "args": ["-m", "lodedb"]}
        assert _write_json(cfg, entry)
        assert json.loads(cfg.read_text("utf-8"))["mcpServers"]["lodedb"] == entry

    def test_preserves_other_servers(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"other": {"command": "x"}}}))
        _write_json(cfg, {"command": "python", "args": []})
        data = json.loads(cfg.read_text("utf-8"))
        assert "other" in data["mcpServers"]
        assert "lodedb" in data["mcpServers"]

    def test_updates_existing_lodedb(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        cfg.write_text(json.dumps({"mcpServers": {"lodedb": {"command": "old"}}}))
        _write_json(cfg, {"command": "new", "args": []})
        assert json.loads(cfg.read_text("utf-8"))["mcpServers"]["lodedb"]["command"] == "new"

    def test_refuses_malformed_json(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        cfg.write_text("{ not json }")
        assert not _write_json(cfg, {"command": "python", "args": []})
        assert "not json" in cfg.read_text("utf-8")  # original untouched

    def test_handles_empty_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "mcp.json"
        cfg.write_text("")
        assert _write_json(cfg, {"command": "python", "args": []})
        assert "lodedb" in json.loads(cfg.read_text("utf-8"))["mcpServers"]

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        cfg = tmp_path / "deep" / "nested" / "mcp.json"
        assert _write_json(cfg, {"command": "python", "args": []})
        assert cfg.exists()


# -- _append_toml -------------------------------------------------------------


class TestAppendToml:
    def test_creates_new_file(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        assert _append_toml(cfg, {"command": "python", "args": ["-m", "lodedb"]})
        assert "[mcpServers.lodedb]" in cfg.read_text("utf-8")

    def test_idempotent(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[mcpServers.lodedb]\ncommand = "old"\n')
        _append_toml(cfg, {"command": "new", "args": []})
        content = cfg.read_text("utf-8")
        assert content.count("[mcpServers.lodedb]") == 1
        assert '"old"' in content

    def test_preserves_existing(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[other]\nkey = "val"\n')
        _append_toml(cfg, {"command": "python", "args": []})
        content = cfg.read_text("utf-8")
        assert "[other]" in content
        assert "[mcpServers.lodedb]" in content


# -- _toml_block --------------------------------------------------------------


def test_toml_block_format() -> None:
    block = _toml_block({"command": "python", "args": ["-m", "lodedb"]})
    assert block.startswith("[mcpServers.lodedb]")
    assert 'command = "python"' in block


# -- _claude_code_cmd ---------------------------------------------------------


def test_claude_code_returns_empty_when_missing() -> None:
    import shutil

    if shutil.which("claude") is None:
        assert _claude_code_cmd({}) == []


# -- _config_paths ------------------------------------------------------------


class TestConfigPaths:
    def test_known_client(self) -> None:
        assert len(_config_paths("cursor")) == 1

    def test_unknown_client(self) -> None:
        assert _config_paths("vscode") == []

    def test_all_returns_multiple(self) -> None:
        assert len(_config_paths("all")) >= 3


# -- install (end-to-end with --config override) ------------------------------


class TestInstall:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test_mcp.json"
        install(
            client="lm-studio",
            dry_run=False,
            config_override=cfg,
            path=tmp_path / "db",
            model="minilm",
            device="auto",
            store_text=True,
        )
        data = json.loads(cfg.read_text("utf-8"))
        assert "lodedb" in data["mcpServers"]

    def test_dry_run_skips_write(self, tmp_path: Path) -> None:
        cfg = tmp_path / "test_mcp.json"
        install(client="cursor", dry_run=True, config_override=cfg)
        assert not cfg.exists()

    def test_unknown_client_does_not_crash(self) -> None:
        install(client="unknown-editor", dry_run=False, config_override=None)
