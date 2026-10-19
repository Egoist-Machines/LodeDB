"""HTTP client regression checks for the OpenKnowledge embedding benchmark."""

import http.client
import json
import sys
from pathlib import Path

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from bench_core import (  # noqa: E402
    DOCUMENT_TIMEOUT_SECONDS,
    QUERY_TIMEOUT_SECONDS,
    OpenAICompatibleClient,
    ProviderConfig,
)
from okchunk import chunk_document  # noqa: E402


class _FakeSocket:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def fileno(self) -> int:
        return 1

    def settimeout(self, timeout_seconds: float) -> None:
        self.timeouts.append(timeout_seconds)


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.status = 200
        self._body = body
        self.was_read = False

    def read(self) -> bytes:
        self.was_read = True
        return self._body


class _FakeConnection:
    def __init__(
        self,
        responses: list[_FakeResponse],
        errors: list[BaseException] | None = None,
    ) -> None:
        self.sock: _FakeSocket | None = _FakeSocket()
        self.timeout: float | None = None
        self.requests: list[tuple[str, str, bytes, dict[str, str]]] = []
        self.responses = responses
        self.errors = [] if errors is None else errors
        self.closed = False

    def request(
        self,
        method: str,
        target: str,
        *,
        body: bytes,
        headers: dict[str, str],
    ) -> None:
        self.requests.append((method, target, body, headers))
        if self.errors:
            raise self.errors.pop(0)

    def getresponse(self) -> _FakeResponse:
        return self.responses.pop(0)

    def close(self) -> None:
        self.closed = True
        self.sock = None


def _response() -> _FakeResponse:
    return _FakeResponse(
        json.dumps(
            {
                "data": [{"index": 0, "embedding": [1.0]}],
                "usage": {"total_tokens": 1},
            }
        ).encode("utf-8")
    )


def _client_with_connections(
    connections: list[_FakeConnection],
) -> tuple[OpenAICompatibleClient, list[float]]:
    client = OpenAICompatibleClient(
        ProviderConfig(
            name="lodedb",
            base_url="http://127.0.0.1:8099/v1",
            model="minilm",
            requested_dimensions=None,
            expected_dimensions=1,
            api_key="test-key",
            token_usage_label="test",
        )
    )
    connection_timeouts: list[float] = []

    def new_connection(timeout_seconds: float) -> _FakeConnection:
        connection_timeouts.append(timeout_seconds)
        connection = connections.pop(0)
        connection.timeout = timeout_seconds
        return connection

    client._new_connection = new_connection  # type: ignore[method-assign]
    return client, connection_timeouts


def test_client_reuses_connection_updates_timeout_and_sanitizes_document_input() -> None:
    document_response = _response()
    query_response = _response()
    connection = _FakeConnection([document_response, query_response])
    client, connection_timeouts = _client_with_connections([connection])
    lone_surrogate_chunk = chunk_document("a" * 7_999 + "\U0001f600")[0]

    document_result = client.embed_batch([lone_surrogate_chunk], "document")
    query_result = client.embed_batch(["cluster networking"], "query")

    assert document_result.retry_count == 0
    assert document_result.reconnects == 0
    assert query_result.retry_count == 0
    assert query_result.reconnects == 0
    assert connection_timeouts == [DOCUMENT_TIMEOUT_SECONDS]
    assert connection.sock is not None
    assert connection.sock.timeouts == [QUERY_TIMEOUT_SECONDS]
    assert len(connection.requests) == 2
    assert document_response.was_read
    assert query_response.was_read
    document_body = json.loads(connection.requests[0][2].decode("utf-8"))
    assert document_body["input"] == ["a" * 7_999 + "\ufffd"]


def test_remote_disconnect_reconnects_without_consuming_retry_budget() -> None:
    failed_connection = _FakeConnection([], [http.client.RemoteDisconnected("closed")])
    reconnected_response = _response()
    replacement_connection = _FakeConnection([reconnected_response])
    client, connection_timeouts = _client_with_connections(
        [failed_connection, replacement_connection]
    )

    result = client.embed_batch(["pod scheduling"], "document")

    assert result.retry_count == 0
    assert result.reconnects == 1
    assert client.reconnects == 1
    assert connection_timeouts == [DOCUMENT_TIMEOUT_SECONDS, DOCUMENT_TIMEOUT_SECONDS]
    assert failed_connection.closed
    assert len(failed_connection.requests) == 1
    assert len(replacement_connection.requests) == 1
    assert reconnected_response.was_read


def test_closed_keep_alive_socket_reconnects_without_consuming_retry_budget() -> None:
    initial_connection = _FakeConnection([_response()])
    replacement_connection = _FakeConnection([_response()])
    client, connection_timeouts = _client_with_connections(
        [initial_connection, replacement_connection]
    )

    client.embed_batch(["service discovery"], "document")
    initial_connection.sock = None
    result = client.embed_batch(["persistent storage"], "document")

    assert result.retry_count == 0
    assert result.reconnects == 1
    assert client.reconnects == 1
    assert connection_timeouts == [DOCUMENT_TIMEOUT_SECONDS, DOCUMENT_TIMEOUT_SECONDS]
    assert initial_connection.closed
    assert len(replacement_connection.requests) == 1
