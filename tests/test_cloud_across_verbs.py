"""CloudClient cross-store verbs against a stub transport: the path each verb
hits, the payload it forwards unchanged, and the absence of a store routing
hint (there is no single store to pin). The server contract itself lives in
`server/tests`."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
import httpx  # noqa: E402

from lodedb.cloud.transfer import CloudClient  # noqa: E402


def _client(seen: list) -> CloudClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"results": []})

    return CloudClient(
        "https://plane.example", "ore_sk_test", transport=httpx.MockTransport(handler)
    )


def test_browse_across_posts_the_payload_to_the_across_path():
    seen: list = []
    payload = {
        "stores": [{"store": "a"}, {"store": "b", "min_seq": 3}],
        "ids": ["m1"],
        "include_text": True,
    }
    with _client(seen) as client:
        assert client.browse_across("acme", "prod", payload) == {"results": []}
    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/v1/data/orgs/acme/environments/prod/stores/browse-across"
    assert json.loads(request.content) == payload
    assert "x-ore-store" not in request.headers
    assert request.headers["authorization"] == "Bearer ore_sk_test"


def test_search_across_posts_the_payload_to_the_across_path():
    seen: list = []
    payload = {"stores": [{"store": "a"}], "query": "brown fox", "k": 5}
    with _client(seen) as client:
        client.search_across("acme", "prod", payload)
    (request,) = seen
    assert request.url.path == "/v1/data/orgs/acme/environments/prod/stores/search-across"
    assert json.loads(request.content) == payload
    assert "x-ore-store" not in request.headers
