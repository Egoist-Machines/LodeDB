"""Tests for the lodedb CLI surface and the local HTTP server.

`doctor` is exercised through Typer's CliRunner (no model load). The server is
exercised against a LodeDB opened with an injected hash backend, hitting the
real request handler through a loopback socket.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from typer.testing import CliRunner

from lodedb.engine.core import chunk_text
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
    assert "native_build_profile" in report["compact_backend"]
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


def _http_json(
    url: str,
    payload: dict | None = None,
    *,
    headers: dict[str, str] | None = None,
) -> dict:
    """POSTs/GETs JSON to a local URL and returns the parsed response."""

    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers or {},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310 - local test URL
        return json.loads(response.read().decode("utf-8"))


def _http_error(
    url: str,
    payload: dict,
    *,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict]:
    """POSTs JSON and returns an expected local HTTP error response."""

    with pytest.raises(urllib.error.HTTPError) as error:
        _http_json(url, payload, headers=headers)
    return error.value.code, json.loads(error.value.read().decode("utf-8"))


def _serve(db: LodeDB) -> tuple[ThreadingHTTPServer, str]:
    """Starts a loopback test server for one open database."""

    handler = build_local_handler(db)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _normalized_mean(vectors: tuple[tuple[float, ...], ...]) -> list[float]:
    """Returns the L2-normalized mean of one text's chunk embeddings."""

    pooled = [math.fsum(values) / len(vectors) for values in zip(*vectors, strict=True)]
    norm = math.sqrt(math.fsum(value * value for value in pooled))
    assert norm != 0.0
    return [value / norm for value in pooled]


def test_local_server_add_search_remove_round_trip(tmp_path):
    """The local server adds, searches, reports stats, and removes documents."""

    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    server, base = _serve(db)
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


def test_embed_texts_preserves_single_chunk_backend_output_without_storing(tmp_path):
    """embed_texts leaves a one-chunk document embedding unchanged."""

    backend = HashEmbeddingBackend(native_dim=384)
    db = LodeDB(path=tmp_path, model="minilm", _embedding_backend=backend)
    text = "  the quick brown fox  "
    try:
        expected = list(backend.embed_documents((text,))[0])
        assert db.embed_texts([text]) == [expected]
        assert db.count() == 0

        with pytest.raises(ValueError, match="non-empty list of strings"):
            db.embed_texts([])
        with pytest.raises(ValueError, match="each text must be a non-empty string"):
            db.embed_texts(["ok", " "])
        with pytest.raises(ValueError, match="each text must be a non-empty string"):
            db.embed_texts(["ok", 1])  # type: ignore[list-item]
    finally:
        db.close()


def test_embed_texts_mean_pools_long_text_chunks(tmp_path):
    """embed_texts mean-pools and normalizes one long document's chunk vectors."""

    chunk_limit = 16
    backend = HashEmbeddingBackend(native_dim=384)
    db = LodeDB(
        path=tmp_path,
        model="minilm",
        chunk_character_limit=chunk_limit,
        _embedding_backend=backend,
    )
    text = "The quick brown fox jumps over the slow turtle. " * 3
    try:
        chunks = chunk_text(text, chunk_limit)
        assert len(chunks) > 1
        expected = _normalized_mean(backend.embed_documents(chunks))
        assert db.embed_texts([text])[0] == pytest.approx(expected)
    finally:
        db.close()


def test_embed_texts_rejects_zero_norm_pooled_vector(tmp_path, monkeypatch):
    """embed_texts rejects a zero vector produced by mean-pooling chunks."""

    backend = HashEmbeddingBackend(native_dim=384)
    db = LodeDB(
        path=tmp_path,
        model="minilm",
        chunk_character_limit=16,
        _embedding_backend=backend,
    )

    def _zero_embed_documents(chunks: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        return tuple((0.0,) * 384 for _ in chunks)

    monkeypatch.setattr(backend, "embed_documents", _zero_embed_documents)
    try:
        with pytest.raises(ValueError, match="mean-pooled embedding has zero norm"):
            db.embed_texts(["A long text needs multiple chunks. " * 3])
    finally:
        db.close()


def test_embed_texts_mixed_batch_preserves_input_order(tmp_path, monkeypatch):
    """embed_texts flattens mixed batches while retaining their input order."""

    chunk_limit = 16
    backend = HashEmbeddingBackend(native_dim=384)
    db = LodeDB(
        path=tmp_path,
        model="minilm",
        chunk_character_limit=chunk_limit,
        _embedding_backend=backend,
    )
    texts = ["first short text", "A long text needs several chunks. " * 3, "last short text"]
    long_chunks = chunk_text(texts[1], chunk_limit)
    expected = [
        list(backend.embed_documents((texts[0],))[0]),
        _normalized_mean(backend.embed_documents(long_chunks)),
        list(backend.embed_documents((texts[2],))[0]),
    ]
    calls: list[tuple[str, ...]] = []
    embed_documents = backend.embed_documents

    def _record_embed_documents(chunks: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
        calls.append(chunks)
        return embed_documents(chunks)

    monkeypatch.setattr(backend, "embed_documents", _record_embed_documents)
    try:
        embeddings = db.embed_texts(texts)
        assert calls == [(texts[0], *long_chunks, texts[2])]
        for actual, expected_embedding in zip(embeddings, expected, strict=True):
            assert actual == pytest.approx(expected_embedding)
    finally:
        db.close()


def test_local_server_openai_embeddings_response_and_paths(tmp_path):
    """Both OpenAI embedding paths return document embeddings in OpenAI's wire shape."""

    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    server, base = _serve(db)
    texts = ["the quick brown fox", "a slow turtle"]
    try:
        response = _http_json(
            f"{base}/embeddings",
            {
                "model": "inkeep-client-model",
                "input": texts,
                "encoding_format": "float",
                "dimensions": 384,
            },
            headers={"Authorization": "Bearer ignored"},
        )
        assert response["object"] == "list"
        assert response["model"] == "inkeep-client-model"
        assert response["usage"] == {
            "prompt_tokens": sum(len(text) for text in texts) // 4,
            "total_tokens": sum(len(text) for text in texts) // 4,
        }
        assert [row["index"] for row in response["data"]] == [0, 1]
        assert [row["embedding"] for row in response["data"]] == db.embed_texts(texts)

        # Consume this exactly as OpenKnowledge does: defensively sort response rows,
        # then verify every vector is a float list at the expected native dimension.
        rows = sorted(response["data"], key=lambda row: row["index"])
        assert len(rows) == len(texts)
        for index, row in enumerate(rows):
            assert row["object"] == "embedding"
            assert row["index"] == index
            assert isinstance(row["embedding"], list)
            assert len(row["embedding"]) == 384
            assert all(isinstance(value, float) for value in row["embedding"])
        assert isinstance(response["usage"]["total_tokens"], int)
        assert response["usage"]["total_tokens"] >= 0

        single = _http_json(
            f"{base}/v1/embeddings",
            {"model": "another-client-model", "input": "one text", "encoding_format": "float"},
        )
        assert single["model"] == "another-client-model"
        assert single["data"] == [
            {
                "object": "embedding",
                "index": 0,
                "embedding": db.embed_texts(["one text"])[0],
            }
        ]
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_local_server_openai_embeddings_chunks_long_input(tmp_path):
    """POST /v1/embeddings returns the pooled vector for a long input."""

    db = LodeDB(
        path=tmp_path,
        model="minilm",
        chunk_character_limit=16,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    server, base = _serve(db)
    text = "A long endpoint input needs several chunks. " * 3
    try:
        response = _http_json(
            f"{base}/v1/embeddings", {"model": "client-model", "input": text}
        )
        embedding = response["data"][0]["embedding"]
        assert len(embedding) == 384
        assert embedding == db.embed_texts([text])[0]
    finally:
        server.shutdown()
        server.server_close()
        db.close()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"model": "client-model", "input": ["ok"], "dimensions": 385},
            "dimensions 385 does not match active model dimension 384",
        ),
        ({"model": "client-model", "input": []}, "input must be a non-empty list of strings"),
        ({"model": "client-model", "input": [" "]}, "each input item must be a non-empty string"),
        (
            {"model": "client-model", "input": ["ok", 1]},
            "each input item must be a non-empty string",
        ),
        (
            {"model": "client-model", "input": ["ok"], "encoding_format": "base64"},
            'encoding_format must be "float"',
        ),
    ],
)
def test_local_server_openai_embeddings_rejects_invalid_input(tmp_path, payload, message):
    """The OpenAI embedding endpoint rejects incompatible dimensions and invalid input."""

    db = LodeDB(
        path=tmp_path, model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384)
    )
    server, base = _serve(db)
    try:
        status, response = _http_error(f"{base}/v1/embeddings", payload)
        assert status == 400
        assert message in response["error"]
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_local_server_vector_only_text_requests_return_bad_request(tmp_path):
    """Text routes return 400 instead of dropping vector-only connections."""

    db = LodeDB.open_vector_store(tmp_path, vector_dim=384)
    server, base = _serve(db)
    try:
        for path, payload in (
            ("/v1/embeddings", {"model": "client-model", "input": "text"}),
            ("/add", {"text": "text"}),
        ):
            status, response = _http_error(f"{base}{path}", payload)
            assert status == 400
            assert "this index is vector-only" in response["error"]
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_local_server_openai_embeddings_reports_missing_runtime(tmp_path, monkeypatch):
    """The OpenAI embedding endpoint exposes the standard missing-runtime install hint."""

    backend = HashEmbeddingBackend(native_dim=384)
    db = LodeDB(path=tmp_path, model="minilm", _embedding_backend=backend)
    install_hint = (
        "text embedding needs an embedding runtime, which is not installed: "
        "pip install 'lodedb[embeddings]'"
    )

    def _missing_runtime(texts):
        raise ModuleNotFoundError(install_hint)

    monkeypatch.setattr(backend, "embed_documents", _missing_runtime)
    server, base = _serve(db)
    try:
        status, response = _http_error(
            f"{base}/embeddings", {"model": "client-model", "input": ["ok"]}
        )
        assert status == 501
        assert response["error"] == install_hint
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_cli_index_default_get_and_no_store_text_round_trip(tmp_path, monkeypatch):
    """`lodedb index` retains text by default; `--no-store-text` opts out.

    The CLI builds its own LodeDB, so we inject the deterministic hash backend by
    patching the symbol the CLI uses. This keeps the test model-free like the rest
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
