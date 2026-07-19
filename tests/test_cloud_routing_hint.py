"""The `X-Ore-Store` routing hint: data-plane calls stamp org/env/store so a
store-sticky ingress can hash-route them for cache locality. Pure transport
concern — the server never reads the header, and calls without it stay
correct — so the tests assert wire shape only."""

from __future__ import annotations

import json

import pytest

# Collection must skip, not error, without the [cloud] extra installed.
pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
pytest.importorskip("nacl", reason="needs the [cloud] extra's dependencies")

import httpx  # noqa: E402

from lodedb.cloud.transfer import CloudClient  # noqa: E402


@pytest.fixture
def capture():
    """(client, seen requests) over a mock transport answering 200 {}."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    client = CloudClient(
        "http://testserver", "ore_sk_test", transport=httpx.MockTransport(handler)
    )
    try:
        yield client, seen
    finally:
        client.close()


def test_data_plane_verbs_carry_the_store_hint(capture):
    client, seen = capture
    client.search("acme", "prod", {"store": "user-42", "query": "q", "k": 3})
    client.add_documents("acme", "prod", {"store": "user-42", "documents": [{"text": "t"}]})
    client.write_status("acme", "prod", "user-42", "w-1")
    client.recall("acme", "prod", {"store": "user-42", "query": "q"})
    assert [r.headers.get("x-ore-store") for r in seen] == ["acme/prod/user-42"] * 4


def test_store_less_payloads_send_no_hint(capture):
    client, seen = capture
    client.search("acme", "prod", {"query": "q", "k": 3})  # server will 422; wire-only here
    client.me()
    assert [r.headers.get("x-ore-store") for r in seen] == [None, None]


def test_hint_rides_alongside_auth_and_body(capture):
    client, seen = capture
    client.search("acme", "prod", {"store": "user-42", "query": "q", "k": 3})
    request = seen[0]
    assert request.headers["authorization"] == "Bearer ore_sk_test"
    assert json.loads(request.content)["store"] == "user-42"
