"""Bounded client-side retry of the 409s the server codes retryable.

Two remedies, both re-running the whole failed operation rather than one leg
of it: `segment_object_missing` on a write (the uploaded segment bytes left
object storage before registration acknowledged them, so the fix is to
re-upload AND re-register — this client's write requests carry the bytes, so
a full resend is exactly that) and `empty_head_artifact_missing` on store
creation (the whole creation re-runs, challenge fetch and sealing included).
Every other refusal, coded or not, surfaces unchanged on the first answer.
"""

from __future__ import annotations

import json

import pytest

# Collection must skip, not error, without the [cloud] extra installed
# (the modules below import httpx / pynacl at module level).
pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
pytest.importorskip("nacl", reason="needs the [cloud] extra's dependencies")

import httpx  # noqa: E402

from lodedb.cloud.client import Client  # noqa: E402
from lodedb.cloud.serving import CloudStore  # noqa: E402
from lodedb.cloud.transfer import CloudClient, CloudError  # noqa: E402

SEGMENT_MISSING = (
    "segment_object_missing: the uploaded segment is no longer in object "
    "storage, so this write cannot be acknowledged — re-upload it and "
    "register again"
)
EMPTY_HEAD_MISSING = (
    "empty_head_artifact_missing: an artifact for this store's empty head "
    "(abcdef123456) left object storage mid-publication, so the head was "
    "not published — the next write republishes it"
)


def _store_over(handler) -> CloudStore:
    client = CloudClient("http://testserver", "tok", transport=httpx.MockTransport(handler))
    return CloudStore(client, "acme", "prod", "user-42", owns_client=True)


def test_cloud_error_exposes_the_refusal_code():
    """OreCloud names each deliberately actionable refusal as the first
    token of detail; uncoded prose must never parse as a code."""
    assert CloudError(409, SEGMENT_MISSING).code == "segment_object_missing"
    assert CloudError(409, EMPTY_HEAD_MISSING).code == "empty_head_artifact_missing"
    assert CloudError(409, "sync refused: local and remote are diverged").code is None
    assert CloudError(409, "a concurrent writer holds this store").code is None
    assert CloudError(404, "no such store").code is None


def test_write_reuploads_and_reregisters_on_segment_object_missing(monkeypatch):
    """The coded 409's remedy is re-upload THEN re-register: the retry must
    resend the whole write request (the segment bytes ride in it), under the
    same idempotency key, not repeat a bytes-free registration."""
    monkeypatch.setattr("lodedb.cloud.serving.time.sleep", lambda s: None)
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/stores/documents")
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(409, json={"detail": SEGMENT_MISSING})
        return httpx.Response(200, json={"ids": ["m1"], "write_id": "w-1", "seq": 3})

    with _store_over(handler) as store:
        doc_id = store.add("remember the segment bytes")

    assert doc_id == "m1"
    assert len(bodies) == 2
    # The re-upload actually happened: the retry carried the identical full
    # payload, documents (the segment bytes) included, not a stripped
    # re-registration.
    assert bodies[1] == bodies[0]
    assert bodies[1]["documents"][0]["text"] == "remember the segment bytes"
    assert bodies[1]["idempotency_key"] == bodies[0]["idempotency_key"]


def test_write_conflict_retry_is_bounded(monkeypatch):
    """A segment that keeps vanishing is a real outage, not a race: the
    retry stops after the bounded budget and surfaces the coded 409."""
    monkeypatch.setattr("lodedb.cloud.serving.time.sleep", lambda s: None)
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(409, json={"detail": SEGMENT_MISSING})

    with _store_over(handler) as store:
        with pytest.raises(CloudError, match="segment_object_missing"):
            store.add("never lands")

    assert calls == 1 + CloudStore._WRITE_CONFLICT_RETRIES


def test_other_409s_surface_on_the_first_answer():
    """Only the one coded remedy retries. A different coded 409 (re-embed in
    progress: polling is the caller's decision, not a blind resend) and an
    uncoded conflict both surface immediately."""
    for detail in (
        "store_reembedding: writes are on hold while the store re-embeds",
        "a concurrent writer holds this store",
    ):
        calls = 0

        def handler(request: httpx.Request, _detail=detail) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(409, json={"detail": _detail})

        with _store_over(handler) as store:
            with pytest.raises(CloudError):
                store.add("refused outright")
        assert calls == 1, detail


def test_create_store_retries_empty_head_artifact_missing(monkeypatch):
    """Creation publishes the store's empty head; its one coded-retryable
    409 rolls the whole creation back, so the client re-runs the whole
    creation, bounded."""
    monkeypatch.setattr("lodedb.cloud.client.time.sleep", lambda s: None)
    creates = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal creates
        assert request.method == "POST" and request.url.path.endswith("/stores")
        creates += 1
        if creates == 1:
            return httpx.Response(409, json={"detail": EMPTY_HEAD_MISSING})
        return httpx.Response(201, json={"store": "user-42", "mode": "local_push"})

    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.create_store("user-42")

    assert result["store"] == "user-42"
    assert creates == 2


def test_create_store_retry_is_bounded_and_other_409s_stay_loud(monkeypatch):
    monkeypatch.setattr("lodedb.cloud.client.time.sleep", lambda s: None)
    from lodedb.cloud.client import _CREATE_CONFLICT_RETRIES

    calls = 0

    def always_missing(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(409, json={"detail": EMPTY_HEAD_MISSING})

    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(always_missing),
    ) as client:
        with pytest.raises(CloudError, match="empty_head_artifact_missing"):
            client.create_store("user-42")
    assert calls == 1 + _CREATE_CONFLICT_RETRIES

    calls = 0

    def already_registered(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(409, json={"detail": "store already registered"})

    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(already_registered),
    ) as client:
        with pytest.raises(CloudError, match="already registered"):
            client.create_store("user-42")
    assert calls == 1


def test_encrypted_create_retry_redoes_the_whole_operation(monkeypatch):
    """For a sealed store the whole creation includes the challenge fetch and
    the sealing: the retry must fetch a fresh recipient and re-seal, never
    replay a stale sealed blob against a rotated recipient."""
    pytest.importorskip("cryptography", reason="needs the [cloud-sealed] extra")
    import base64

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import x25519

    monkeypatch.setattr("lodedb.cloud.client.time.sleep", lambda s: None)
    recipient = base64.b64encode(
        x25519.X25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    ).decode()
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        if request.method == "GET":
            assert request.url.path.endswith("/stores/create-challenge")
            return httpx.Response(200, json={"recipient_public_key": recipient})
        if methods.count("POST") == 1:
            return httpx.Response(409, json={"detail": EMPTY_HEAD_MISSING})
        return httpx.Response(201, json={"store": "user-42", "encrypted": True})

    with Client(
        token="ore_sk_test",
        host="http://testserver",
        org="acme",
        environment="prod",
        transport=httpx.MockTransport(handler),
    ) as client:
        result = client.create_store("user-42", encrypted=True, key_material=b"m" * 32)

    assert result["encrypted"] is True
    assert methods == ["GET", "POST", "GET", "POST"]
