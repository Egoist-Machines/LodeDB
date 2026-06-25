"""Tests for the lodedb CLI surface and the local HTTP server.

`doctor` is exercised through Typer's CliRunner (no model load). The server is
exercised against a LodeDB opened with an injected hash backend, hitting the
real request handler through a loopback socket.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from typer.testing import CliRunner

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local import LodeDB
from lodedb.local.cli import app
from lodedb.local.server import build_local_handler

runner = CliRunner()


def _cli_json(output: str) -> dict:
    """Extracts the command's JSON object from clean CLI output."""

    return json.loads(output)


def test_cli_doctor_text_and_json():
    """`lodedb doctor` renders text and valid JSON capability reports."""

    text_result = runner.invoke(app, ["doctor"])
    assert text_result.exit_code == 0, text_result.output
    assert "LodeDB doctor" in text_result.output
    assert "GPU-resident vector scan" in text_result.output

    json_result = runner.invoke(app, ["doctor", "--json"])
    assert json_result.exit_code == 0, json_result.output
    report = json.loads(json_result.output)
    assert "compact_backend" in report
    assert "gpu_vector_scan" in report


def test_cli_doctor_fix_is_no_op_when_nothing_to_fix(monkeypatch):
    """`lodedb doctor --fix` prints a clear no-op message when there is no CPU torch to fix.

    The hint is forced absent so the test never shells out to pip on any platform.
    """

    import lodedb.local.doctor as doctor

    monkeypatch.setattr(doctor, "_windows_gpu_embedding_hint", lambda: None)
    result = runner.invoke(app, ["doctor", "--fix"])
    assert result.exit_code == 0, result.output
    assert "Nothing to fix" in result.output


def test_cli_help_lists_all_commands():
    """The CLI exposes serve | index | query | benchmark | doctor."""

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("serve", "index", "query", "get", "benchmark", "doctor", "mcp"):
        assert command in result.output


def _http_json(url: str, payload: dict | None = None) -> dict:
    """POSTs/GETs JSON to a local URL and returns the parsed response."""

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - local test URL
        return json.loads(response.read().decode("utf-8"))


def test_local_server_add_search_remove_round_trip(tmp_path):
    """The local server adds, searches, reports stats, and removes documents."""

    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    handler = build_local_handler(db)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        assert _http_json(f"{base}/healthz")["status"] == "ok"
        added = _http_json(f"{base}/add", {"text": "the quick brown fox", "id": "fox"})
        assert added["id"] == "fox" and added["count"] == 1
        _http_json(f"{base}/add", {"text": "a slow turtle", "id": "turtle"})

        searched = _http_json(f"{base}/search", {"query": "fox", "k": 5})
        ids = {row["id"] for row in searched["results"]}
        assert "fox" in ids

        stats = _http_json(f"{base}/stats")
        assert stats["document_count"] == 2
        assert stats["raw_payload_text_present"] is False

        removed = _http_json(f"{base}/remove", {"id": "turtle"})
        assert removed["removed"] is True and removed["count"] == 1
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_local_server_get_returns_stored_text_unless_opted_out(tmp_path):
    """POST /get returns raw text by default, and 400s when store_text is off."""

    def _serve(db):
        handler = build_local_handler(db)
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"

    # default (store_text on) -> /get returns the original text; missing id -> 404.
    db = LodeDB(
        path=tmp_path / "on",
        model="minilm",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    server, base = _serve(db)
    try:
        _http_json(f"{base}/add", {"text": "the quick brown fox", "id": "fox"})
        assert _http_json(f"{base}/get", {"id": "fox"}) == {
            "id": "fox",
            "text": "the quick brown fox",
        }
        try:
            _http_json(f"{base}/get", {"id": "absent"})
            raise AssertionError("expected 404 for unknown id")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        server.shutdown()
        server.server_close()
        db.close()

    # store_text=False -> /get is a clear 400 (text was deliberately not retained).
    off = LodeDB(
        path=tmp_path / "off",
        model="minilm",
        store_text=False,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    off.add("hidden body", id="h")
    server, base = _serve(off)
    try:
        try:
            _http_json(f"{base}/get", {"id": "h"})
            raise AssertionError("expected 400 when store_text is off")
        except urllib.error.HTTPError as exc:
            assert exc.code == 400
    finally:
        server.shutdown()
        server.server_close()
        off.close()


def test_cli_index_default_get_and_no_store_text_round_trip(tmp_path, monkeypatch):
    """`lodedb index` retains text by default; `--no-store-text` opts out.

    The CLI builds its own LodeDB, so we inject the deterministic hash backend by
    patching the symbol the CLI uses — keeping this test model-free like the rest
    of the suite while exercising the real command wiring end to end.
    """

    import lodedb.local.cli as cli_mod

    real_lodedb = cli_mod.LodeDB

    def _factory(*args, **kwargs):
        kwargs.setdefault("_embedding_backend", HashEmbeddingBackend(native_dim=384))
        return real_lodedb(*args, **kwargs)

    monkeypatch.setattr(cli_mod, "LodeDB", _factory)

    # Index with defaults (text retained), then retrieve it by the returned id.
    indexed = runner.invoke(app, ["index", "-p", str(tmp_path), "the quick brown fox"])
    assert indexed.exit_code == 0, indexed.output
    doc_id = _cli_json(indexed.output)["ids"][0]

    got = runner.invoke(app, ["get", doc_id, "-p", str(tmp_path)])
    assert got.exit_code == 0, got.output
    assert _cli_json(got.output) == {"id": doc_id, "text": "the quick brown fox"}

    # A document indexed WITH --no-store-text has no retrievable text.
    plain = runner.invoke(
        app, ["index", "--no-store-text", "-p", str(tmp_path / "plain"), "no stored body"]
    )
    plain_id = _cli_json(plain.output)["ids"][0]
    missing = runner.invoke(app, ["get", plain_id, "-p", str(tmp_path / "plain")])
    assert missing.exit_code != 0  # BadParameter -> "document not found"


def test_private_bind_hosts_are_intentional_but_unspecified_is_rejected():
    """serve_local allows loopback/private addresses, not public or all-interface binds."""

    from lodedb.engine.core import is_private_bind_host

    for host in ("127.0.0.1", "::1", "localhost", "10.0.0.5", "172.16.0.5", "192.168.1.5"):
        assert is_private_bind_host(host), host
    for host in ("8.8.8.8", "1.1.1.1", "0.0.0.0", "::"):
        assert not is_private_bind_host(host), host


def test_local_server_rejects_non_private_host(tmp_path):
    """serve_local refuses to bind to a public host."""

    import pytest

    from lodedb.local.server import serve_local

    with pytest.raises(ValueError, match="loopback or private"):
        serve_local(path=tmp_path, host="8.8.8.8", port=9099)
