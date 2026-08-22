"""Shared OpenKnowledge-compatible corpus, batching, and HTTP benchmark logic.

This module deliberately has no Modal import so its local smoke path can validate
the same batcher and OpenAI-compatible client against a local LodeDB server.
"""

from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import random
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from array import array
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    from benchmarks.openknowledge_embeddings.okchunk import chunk_document, javascript_length
except ModuleNotFoundError:  # Supports `python benchmarks/.../bench_core.py --local-smoke`.
    from okchunk import chunk_document, javascript_length

DEFAULT_MAX_BATCH_SIZE = 96
DEFAULT_MAX_BATCH_CHARS = 96_000
DOCUMENT_TIMEOUT_SECONDS = 30.0
QUERY_TIMEOUT_SECONDS = 8.0
MAX_RETRIES = 4
BACKOFF_BASE_SECONDS = 0.5
RETRYABLE_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})
LODEDB_DIMENSIONS = 384
OPENAI_DIMENSIONS = 1_536
# Source: OpenAI API pricing, text-embedding-3-small, accessed 2026-07-09.
OPENAI_PRICE_USD_PER_MILLION_TOKENS = 0.020

EmbeddingRole = Literal["document", "query"]
ProviderName = Literal["lodedb", "openai"]


@dataclass(frozen=True)
class ProviderConfig:
    """The request and response settings OpenKnowledge applies to one provider."""

    name: ProviderName
    base_url: str
    model: str
    requested_dimensions: int | None
    expected_dimensions: int
    api_key: str
    token_usage_label: str


@dataclass(frozen=True)
class TrafficSnapshot:
    request_body_bytes: int
    response_body_bytes: int

    @property
    def request_response_body_bytes(self) -> int:
        return self.request_body_bytes + self.response_body_bytes


@dataclass
class HttpBodyTraffic:
    """Body traffic from every HTTP attempt, including retries."""

    request_body_bytes: int = 0
    response_body_bytes: int = 0

    def snapshot(self) -> TrafficSnapshot:
        return TrafficSnapshot(self.request_body_bytes, self.response_body_bytes)


@dataclass(frozen=True)
class BatchResult:
    duration_seconds: float
    retry_count: int
    reconnects: int
    reported_total_tokens: int | float


@dataclass(frozen=True)
class Corpus:
    """The selected wiki documents reduced to the data the workload needs."""

    chunks: tuple[str, ...]
    query_candidates: tuple[str, ...]
    stats: dict[str, Any]


@dataclass(frozen=True)
class LocalLodeDBServer:
    base_url: str
    cold_start_seconds: float


class EmbeddingRequestError(RuntimeError):
    """A request failed after its OpenKnowledge-compatible retry policy."""


class EmbeddingResponseError(RuntimeError):
    """A successful HTTP response did not match OpenKnowledge's expectations."""


class _RetryableRequestError(RuntimeError):
    """Internal classification for an HTTP status eligible for retry."""


def batch_inputs(
    texts: Sequence[str],
    *,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
    max_batch_chars: int = DEFAULT_MAX_BATCH_CHARS,
) -> list[list[str]]:
    """Ports OpenKnowledge's greedy sequential count and character batcher."""

    batches: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for text in texts:
        if current and (
            len(current) >= max_batch_size or chars + javascript_length(text) > max_batch_chars
        ):
            batches.append(current)
            current = []
            chars = 0
        current.append(text)
        chars += javascript_length(text)
    if current:
        batches.append(current)
    return batches


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _latency_percentiles_ms(values_seconds: Sequence[float]) -> dict[str, float | None]:
    values_ms = [value * 1_000.0 for value in values_seconds]
    return {
        "p50": _round(_percentile(values_ms, 0.50)),
        "p95": _round(_percentile(values_ms, 0.95)),
        "max": _round(max(values_ms) if values_ms else None),
    }


def _chunk_char_percentiles(chunks: Sequence[str]) -> dict[str, float | int | None]:
    lengths = [javascript_length(chunk) for chunk in chunks]
    maximum = max(lengths) if lengths else None
    return {
        "p50": _round(_percentile(lengths, 0.50)),
        "p95": _round(_percentile(lengths, 0.95)),
        "max": maximum,
    }


def _clean_query_candidate(value: str) -> str:
    normalized = " ".join(value.replace("`", "").split())
    return normalized[:160].strip()


def _document_query_candidates(path: Path, text: str) -> list[str]:
    """Produces short deterministic queries from a document's title and headings."""

    candidates: list[str] = []
    frontmatter_title = re.search(r"(?m)^title:\s*[\"']?(.+?)[\"']?\s*$", text)
    if frontmatter_title:
        candidates.append(_clean_query_candidate(frontmatter_title.group(1)))
    for heading in re.findall(r"(?m)^\s{0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*$", text):
        candidates.append(_clean_query_candidate(heading))
        if len(candidates) >= 3:
            break
    if not candidates:
        candidates.append(_clean_query_candidate(path.stem.replace("-", " ")))
    return [candidate for candidate in candidates if candidate]


def load_kubernetes_corpus(corpus_root: Path, docs: int = 0) -> Corpus:
    """Loads sorted ``content/en/**/*.md`` files and applies OpenKnowledge chunking."""

    if docs < 0:
        raise ValueError("--docs must be zero (all documents) or a positive integer")
    markdown_root = corpus_root / "content" / "en"
    paths = sorted(markdown_root.rglob("*.md"), key=lambda path: path.as_posix())
    if docs:
        paths = paths[:docs]

    chunks: list[str] = []
    query_candidates: list[str] = []
    total_chars = 0
    docs_with_multiple_chunks = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        document_chunks = chunk_document(text)
        chunks.extend(document_chunks)
        query_candidates.extend(_document_query_candidates(path, text))
        total_chars += javascript_length(text)
        if len(document_chunks) > 1:
            docs_with_multiple_chunks += 1

    stats = {
        "docs": len(paths),
        "total_chars": total_chars,
        "chunks": len(chunks),
        "chunk_chars": _chunk_char_percentiles(chunks),
        "docs_with_more_than_one_chunk": docs_with_multiple_chunks,
    }
    return Corpus(tuple(chunks), tuple(query_candidates), stats)


def generate_queries(corpus: Corpus, query_count: int) -> list[str]:
    """Cycles deterministic title and heading candidates to make realistic short queries."""

    if query_count < 0:
        raise ValueError("--queries must be zero or a positive integer")
    if query_count == 0:
        return []
    if not corpus.query_candidates:
        raise ValueError("the selected corpus has no usable document title or heading for queries")
    return [
        corpus.query_candidates[index % len(corpus.query_candidates)]
        for index in range(query_count)
    ]


def _normalize_in_place(vector: array[float]) -> None:
    """Mirrors OpenKnowledge's Float32Array L2 normalization before vectors are returned."""

    norm = 0.0
    for value in vector:
        norm += value * value
    norm = math.sqrt(norm)
    if norm > 0.0:
        for index, value in enumerate(vector):
            vector[index] = value / norm


def _reported_total_tokens(payload: dict[str, Any]) -> int | float:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return 0
    total_tokens = usage.get("total_tokens", 0)
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, (int, float)):
        return 0
    return total_tokens


def _consume_embedding_response(
    payload: Any,
    *,
    expected_count: int,
    expected_dimensions: int,
) -> int | float:
    """Validates, orders, Float32-casts, and normalizes as OpenKnowledge does."""

    if not isinstance(payload, dict):
        raise EmbeddingResponseError("embeddings response was not a JSON object")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        got = len(data) if isinstance(data, list) else 0
        raise EmbeddingResponseError(
            f"embeddings response had {got} vectors, expected {expected_count}"
        )
    if not all(isinstance(item, dict) for item in data):
        raise EmbeddingResponseError("embeddings response contained a non-object vector item")

    ordered = sorted(data, key=lambda item: item.get("index", 0))
    for item in ordered:
        embedding = item.get("embedding")
        if not isinstance(embedding, list):
            raise EmbeddingResponseError("embeddings response contained a non-array embedding")
        if len(embedding) != expected_dimensions:
            raise EmbeddingResponseError(
                "embeddings response had "
                f"{len(embedding)} dimensions, expected {expected_dimensions}"
            )
        try:
            vector = array("f", embedding)
        except (TypeError, ValueError, OverflowError) as exc:
            raise EmbeddingResponseError(
                "embeddings response contained a non-numeric embedding"
            ) from exc
        _normalize_in_place(vector)
    return _reported_total_tokens(payload)


def _sleep_duration_for_retry(retry_number: int) -> float:
    """Matches the source embedder's jittered 500 ms exponential backoff."""

    ceiling_ms = BACKOFF_BASE_SECONDS * 1_000.0 * 2 ** (retry_number - 1)
    delay_ms = ceiling_ms / 2.0 + random.random() * (ceiling_ms / 2.0)
    return math.floor(delay_ms + 0.5) / 1_000.0


def _sanitize_lone_surrogates(text: str) -> str:
    """Replaces unpaired UTF-16 surrogate code units before UTF-8 encoding."""

    if not any(0xD800 <= ord(character) <= 0xDFFF for character in text):
        return text

    sanitized: list[str] = []
    index = 0
    while index < len(text):
        code_point = ord(text[index])
        if 0xD800 <= code_point <= 0xDBFF:
            if index + 1 < len(text) and 0xDC00 <= ord(text[index + 1]) <= 0xDFFF:
                low_surrogate = ord(text[index + 1])
                astral_code_point = (
                    0x10000 + (code_point - 0xD800) * 0x400 + low_surrogate - 0xDC00
                )
                sanitized.append(chr(astral_code_point))
                index += 2
                continue
            sanitized.append("\ufffd")
        elif 0xDC00 <= code_point <= 0xDFFF:
            sanitized.append("\ufffd")
        else:
            sanitized.append(text[index])
        index += 1
    return "".join(sanitized)


class OpenAICompatibleClient:
    """A standard-library HTTP port of OpenKnowledge's OpenAI-compatible embedder."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.endpoint = f"{config.base_url.rstrip('/')}/embeddings"
        endpoint = urllib.parse.urlsplit(self.endpoint)
        if endpoint.scheme not in {"http", "https"} or endpoint.hostname is None:
            raise ValueError("provider base_url must be an absolute HTTP or HTTPS URL")
        self._connection_type = (
            http.client.HTTPSConnection
            if endpoint.scheme == "https"
            else http.client.HTTPConnection
        )
        self._host = endpoint.hostname
        self._port = endpoint.port
        self._request_target = endpoint.path or "/"
        if endpoint.query:
            self._request_target = f"{self._request_target}?{endpoint.query}"
        self._connection: http.client.HTTPConnection | None = None
        self._connection_was_used = False
        self.reconnects = 0
        self.traffic = HttpBodyTraffic()

    def close(self) -> None:
        """Closes the provider connection once its workload is complete."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._connection_was_used = False

    def _new_connection(self, timeout_seconds: float) -> http.client.HTTPConnection:
        return self._connection_type(self._host, self._port, timeout=timeout_seconds)

    @staticmethod
    def _socket_is_closed(connection: http.client.HTTPConnection) -> bool:
        socket = connection.sock
        if socket is None:
            return True
        try:
            return socket.fileno() < 0
        except OSError:
            return True

    @staticmethod
    def _set_timeout(connection: http.client.HTTPConnection, timeout_seconds: float) -> None:
        connection.timeout = timeout_seconds
        if connection.sock is not None:
            # Documents run before queries, so retain the socket and update its timeout in place.
            connection.sock.settimeout(timeout_seconds)

    def _connection_for_request(
        self, timeout_seconds: float
    ) -> tuple[http.client.HTTPConnection, bool]:
        if self._connection is None:
            self._connection = self._new_connection(timeout_seconds)
            return self._connection, False

        if self._connection_was_used and self._socket_is_closed(self._connection):
            self.close()
            self._connection = self._new_connection(timeout_seconds)
            return self._connection, True

        self._set_timeout(self._connection, timeout_seconds)
        return self._connection, False

    def _reconnect(self, timeout_seconds: float) -> None:
        self.close()
        self._connection = self._new_connection(timeout_seconds)
        self.reconnects += 1

    def _post_once(self, body: bytes, timeout_seconds: float) -> bytes:
        reconnect_attempted = False
        while True:
            connection, needs_reconnect = self._connection_for_request(timeout_seconds)
            if needs_reconnect:
                self.reconnects += 1
                reconnect_attempted = True

            self.traffic.request_body_bytes += len(body)
            try:
                connection.request(
                    "POST",
                    self._request_target,
                    body=body,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.config.api_key}",
                    },
                )
                self._connection_was_used = True
                response = connection.getresponse()
                response_body = response.read()
                self.traffic.response_body_bytes += len(response_body)
                status = response.status
            except (
                http.client.RemoteDisconnected,
                http.client.BadStatusLine,
                http.client.CannotSendRequest,
                http.client.ResponseNotReady,
                BrokenPipeError,
                ConnectionResetError,
            ):
                if reconnect_attempted:
                    raise
                self._reconnect(timeout_seconds)
                reconnect_attempted = True
                continue
            except OSError:
                if not self._socket_is_closed(connection) or reconnect_attempted:
                    raise
                self._reconnect(timeout_seconds)
                reconnect_attempted = True
                continue

            if not 200 <= status < 300:
                if status in RETRYABLE_STATUSES:
                    raise _RetryableRequestError(f"embeddings request failed: HTTP {status}")
                raise EmbeddingRequestError(f"embeddings request failed: HTTP {status}")
            return response_body

    def embed_batch(self, texts: Sequence[str], role: EmbeddingRole) -> BatchResult:
        """Issues one request with the source embedder's body, timeout, and retry policy."""

        body_dict: dict[str, Any] = {
            "model": self.config.model,
            "input": [_sanitize_lone_surrogates(text) for text in texts],
            "encoding_format": "float",
        }
        if self.config.requested_dimensions is not None:
            body_dict["dimensions"] = self.config.requested_dimensions
        body = json.dumps(body_dict, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        timeout_seconds = DOCUMENT_TIMEOUT_SECONDS if role == "document" else QUERY_TIMEOUT_SECONDS
        started = time.monotonic()
        attempt = 0
        retries = 0
        reconnects_before = self.reconnects
        while True:
            try:
                response_body = self._post_once(body, timeout_seconds)
                payload = json.loads(response_body.decode("utf-8"))
                reported_total_tokens = _consume_embedding_response(
                    payload,
                    expected_count=len(texts),
                    expected_dimensions=self.config.expected_dimensions,
                )
                return BatchResult(
                    time.monotonic() - started,
                    retries,
                    self.reconnects - reconnects_before,
                    reported_total_tokens,
                )
            except _RetryableRequestError as exc:
                failure: BaseException = exc
            except (
                http.client.HTTPException,
                urllib.error.URLError,
                TimeoutError,
                OSError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as exc:
                failure = exc
            except EmbeddingResponseError as exc:
                raise EmbeddingRequestError("embeddings response was malformed") from exc

            if attempt >= MAX_RETRIES:
                raise EmbeddingRequestError("embeddings request failed after retries") from failure
            attempt += 1
            retries += 1
            time.sleep(_sleep_duration_for_retry(attempt))


def _traffic_delta(before: TrafficSnapshot, after: TrafficSnapshot) -> TrafficSnapshot:
    return TrafficSnapshot(
        request_body_bytes=after.request_body_bytes - before.request_body_bytes,
        response_body_bytes=after.response_body_bytes - before.response_body_bytes,
    )


def _is_loopback_url(url: str) -> bool:
    hostname = urllib.parse.urlparse(url).hostname
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _traffic_report(traffic: TrafficSnapshot, *, loopback: bool) -> dict[str, int]:
    total = traffic.request_response_body_bytes
    return {
        "request_body_bytes": traffic.request_body_bytes,
        "response_body_bytes": traffic.response_body_bytes,
        "request_response_body_bytes": total,
        "external_request_response_body_bytes": 0 if loopback else total,
        "loopback_request_response_body_bytes": total if loopback else 0,
    }


def _rate(count: int, elapsed_seconds: float) -> float | None:
    return None if elapsed_seconds <= 0.0 else _round(count / elapsed_seconds)


def measure_document_embeddings(
    client: OpenAICompatibleClient,
    batches: Sequence[Sequence[str]],
    *,
    document_count: int,
) -> dict[str, Any]:
    """Embeds every document chunk sequentially, exactly as OpenKnowledge's embed() does."""

    started = time.monotonic()
    latencies: list[float] = []
    retries = 0
    reconnects = 0
    reported_total_tokens: int | float = 0
    chunk_count = 0
    for batch in batches:
        result = client.embed_batch(batch, "document")
        latencies.append(result.duration_seconds)
        retries += result.retry_count
        reconnects += result.reconnects
        reported_total_tokens += result.reported_total_tokens
        chunk_count += len(batch)
    elapsed = time.monotonic() - started
    return {
        "wall_seconds": _round(elapsed),
        "chunks": chunk_count,
        "documents": document_count,
        "chunks_per_second": _rate(chunk_count, elapsed),
        "documents_per_second": _rate(document_count, elapsed),
        "batch_count": len(batches),
        "batch_latency_ms": _latency_percentiles_ms(latencies),
        "retry_count": retries,
        "reconnects": reconnects,
        "reported_usage_total_tokens": reported_total_tokens,
    }


def measure_query_embeddings(
    client: OpenAICompatibleClient,
    queries: Sequence[str],
) -> dict[str, Any]:
    """Embeds one query per sequential request, matching OpenKnowledge search behavior."""

    started = time.monotonic()
    latencies: list[float] = []
    retries = 0
    reconnects = 0
    reported_total_tokens: int | float = 0
    for query in queries:
        result = client.embed_batch([query], "query")
        latencies.append(result.duration_seconds)
        retries += result.retry_count
        reconnects += result.reconnects
        reported_total_tokens += result.reported_total_tokens
    elapsed = time.monotonic() - started
    return {
        "wall_seconds": _round(elapsed),
        "query_count": len(queries),
        "latency_ms": _latency_percentiles_ms(latencies),
        "retry_count": retries,
        "reconnects": reconnects,
        "reported_usage_total_tokens": reported_total_tokens,
    }


def _warmup_document_batch(
    client: OpenAICompatibleClient,
    batches: Sequence[Sequence[str]],
) -> dict[str, Any]:
    if not batches:
        return {"performed": False}
    result = client.embed_batch(batches[0], "document")
    return {
        "performed": True,
        "input_count": len(batches[0]),
        "latency_ms": _round(result.duration_seconds * 1_000.0),
        "retry_count": result.retry_count,
        "reconnects": result.reconnects,
        "reported_usage_total_tokens": result.reported_total_tokens,
    }


def run_provider_workload(
    config: ProviderConfig,
    *,
    document_batches: Sequence[Sequence[str]],
    document_count: int,
    queries: Sequence[str],
    cold_start_seconds: float | None = None,
) -> dict[str, Any]:
    """Runs a provider's warmup, document workload, and sequential query workload."""

    client = OpenAICompatibleClient(config)
    loopback = _is_loopback_url(config.base_url)

    warmup_before = client.traffic.snapshot()
    warmup = _warmup_document_batch(client, document_batches)
    warmup_after = client.traffic.snapshot()

    workload_before = client.traffic.snapshot()
    documents = measure_document_embeddings(
        client,
        document_batches,
        document_count=document_count,
    )
    query_metrics = measure_query_embeddings(client, queries)
    workload_after = client.traffic.snapshot()

    warmup["http_body_traffic"] = _traffic_report(
        _traffic_delta(warmup_before, warmup_after),
        loopback=loopback,
    )
    workload_traffic = _traffic_report(
        _traffic_delta(workload_before, workload_after),
        loopback=loopback,
    )
    reported_workload_tokens = (
        documents["reported_usage_total_tokens"] + query_metrics["reported_usage_total_tokens"]
    )
    cost_usd = (
        reported_workload_tokens / 1_000_000.0 * OPENAI_PRICE_USD_PER_MILLION_TOKENS
        if config.name == "openai"
        else 0.0
    )
    return {
        "model": config.model,
        "dimensions": config.expected_dimensions,
        "requested_dimensions": config.requested_dimensions,
        "cold_start_seconds": _round(cold_start_seconds),
        "warmup": warmup,
        "documents": documents,
        "queries": query_metrics,
        "token_usage": {
            "document_reported_usage_total_tokens": documents["reported_usage_total_tokens"],
            "query_reported_usage_total_tokens": query_metrics["reported_usage_total_tokens"],
            "workload_reported_usage_total_tokens": reported_workload_tokens,
            "label": config.token_usage_label,
        },
        "egress": workload_traffic,
        "cost_usd": _round(cost_usd, 12),
        "cost_scope": "document and query workload only; warmup is excluded",
    }


def _server_failure_message(process: subprocess.Popen[str], log: Any) -> str:
    log.seek(0)
    output = log.read()
    tail = output[-4_000:]
    return f"lodedb serve exited with code {process.returncode}: {tail}"


@contextmanager
def running_lodedb_server(
    *,
    path: Path = Path("/root/okbench-store"),
    port: int = 8_099,
    startup_timeout_seconds: float = 900.0,
) -> Iterator[LocalLodeDBServer]:
    """Starts the benchmark's CPU ONNX endpoint and measures readiness from Popen."""

    log = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    command = [
        "lodedb",
        "serve",
        "--model",
        "minilm",
        "--port",
        str(port),
        "--path",
        str(path),
        "--device",
        "cpu",
        "--runtime",
        "onnx",
    ]
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    health_url = f"http://127.0.0.1:{port}/healthz"
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError(_server_failure_message(process, log))
            try:
                with urllib.request.urlopen(health_url, timeout=1.0) as response:
                    if response.status == 200:
                        yield LocalLodeDBServer(
                            base_url=f"http://127.0.0.1:{port}/v1",
                            cold_start_seconds=time.monotonic() - started,
                        )
                        break
            except (urllib.error.URLError, TimeoutError, OSError):
                pass
            if time.monotonic() - started >= startup_timeout_seconds:
                raise TimeoutError("lodedb serve did not become healthy before the startup timeout")
            time.sleep(0.25)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15.0)
        log.close()


def run_benchmark(
    *,
    corpus_root: Path,
    docs: int,
    query_count: int,
    provider: str,
    openai_api_key: str | None,
    lodedb_store_path: Path = Path("/root/okbench-store"),
    cpu_count: int | None = None,
    corpus_revision: str | None = None,
) -> dict[str, Any]:
    """Runs requested providers in one process, always placing LodeDB before OpenAI."""

    if provider not in {"lodedb", "openai", "both"}:
        raise ValueError("--provider must be one of: lodedb, openai, both")
    corpus = load_kubernetes_corpus(corpus_root, docs)
    queries = generate_queries(corpus, query_count)
    document_batches = batch_inputs(corpus.chunks)
    results: dict[str, Any] = {
        "config": {
            "docs_flag": docs,
            "docs_selected": corpus.stats["docs"],
            "queries_flag": query_count,
            "queries_selected": len(queries),
            "provider_flag": provider,
            "cpu_count": cpu_count if cpu_count is not None else os.cpu_count(),
            "document_batch_count": len(document_batches),
            "document_batch_size": DEFAULT_MAX_BATCH_SIZE,
            "document_batch_chars": DEFAULT_MAX_BATCH_CHARS,
            "corpus_revision": corpus_revision,
        },
        "corpus": corpus.stats,
        "providers": {},
    }

    # Fairness: both providers use this chunker, batcher, retry policy, and HTTP client.
    if provider in {"lodedb", "both"}:
        with running_lodedb_server(path=lodedb_store_path) as server:
            results["providers"]["lodedb"] = run_provider_workload(
                ProviderConfig(
                    name="lodedb",
                    base_url=server.base_url,
                    model="minilm",
                    requested_dimensions=LODEDB_DIMENSIONS,
                    expected_dimensions=LODEDB_DIMENSIONS,
                    api_key="local-benchmark-placeholder",
                    token_usage_label="LodeDB server estimate from character count",
                ),
                document_batches=document_batches,
                document_count=corpus.stats["docs"],
                queries=queries,
                cold_start_seconds=server.cold_start_seconds,
            )

    if provider in {"openai", "both"}:
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the openai provider")
        # Remote OpenAI is measured from a datacenter, which is favorable to its network latency.
        # OpenKnowledge's config must match model dimensions, so real users receive LodeDB's
        # 384 dimensions and OpenAI's native 1536 dimensions rather than an artificial match.
        results["providers"]["openai"] = run_provider_workload(
            ProviderConfig(
                name="openai",
                base_url="https://api.openai.com/v1",
                model="text-embedding-3-small",
                requested_dimensions=None,
                expected_dimensions=OPENAI_DIMENSIONS,
                api_key=openai_api_key,
                token_usage_label="OpenAI provider-reported token total",
            ),
            document_batches=document_batches,
            document_count=corpus.stats["docs"],
            queries=queries,
        )
    return results


def _local_smoke_documents(count: int) -> list[str]:
    """Creates non-empty representative Markdown inputs without requiring the wiki clone."""

    topics = (
        "deployment rollouts",
        "service discovery",
        "cluster networking",
        "pod scheduling",
        "persistent storage",
    )
    return [
        f"# Kubernetes {topics[index % len(topics)]}\n\n"
        f"Local embedding smoke document {index}. "
        "This verifies OpenKnowledge-compatible document batching against LodeDB."
        for index in range(count)
    ]


def run_local_smoke(*, base_url: str, docs: int = 20) -> dict[str, Any]:
    """Drives roughly 20 short documents through the exact document embedding path."""

    if docs <= 0:
        raise ValueError("--docs must be a positive integer for --local-smoke")
    chunks = tuple(
        chunk for document in _local_smoke_documents(docs) for chunk in chunk_document(document)
    )
    batches = batch_inputs(chunks)
    client = OpenAICompatibleClient(
        ProviderConfig(
            name="lodedb",
            base_url=base_url,
            model="minilm",
            requested_dimensions=LODEDB_DIMENSIONS,
            expected_dimensions=LODEDB_DIMENSIONS,
            api_key="local-benchmark-placeholder",
            token_usage_label="LodeDB server estimate from character count",
        )
    )
    before = client.traffic.snapshot()
    documents = measure_document_embeddings(client, batches, document_count=docs)
    after = client.traffic.snapshot()
    return {
        "documents": documents,
        "http_body_traffic": _traffic_report(_traffic_delta(before, after), loopback=True),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="OpenKnowledge embedding benchmark local smoke")
    parser.add_argument(
        "--local-smoke", action="store_true", help="run against an existing local LodeDB server"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    parser.add_argument("--docs", type=int, default=20)
    args = parser.parse_args()
    if not args.local_smoke:
        parser.error("pass --local-smoke to run the local client smoke test")
    print(
        json.dumps(
            run_local_smoke(base_url=args.base_url, docs=args.docs), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    _main()
