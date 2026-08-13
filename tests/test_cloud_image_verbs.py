"""The cloud image verbs (share-file-types Track A): payload shapes, the
20 MB client cap, input coercion, and the plane-embeds/never-retains
contract expressed as transport routing (images ride dedicated routes the
plane can gate per store preset)."""

from __future__ import annotations

import base64

import pytest

pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
pytest.importorskip("nacl", reason="needs the [cloud] extra's dependencies")

from lodedb.cloud.serving import CloudStore  # noqa: E402


class _FakeClient:
    def __init__(self):
        self.calls = []

    def add_images(self, org, environment, payload):
        self.calls.append(("add_images", org, environment, payload))
        return {
            "accepted": True,
            "write_id": "w-1",
            "seq": 1,
            "ids": [doc.get("id") or f"gen-{i}" for i, doc in enumerate(payload["documents"])],
        }

    def search_image(self, org, environment, payload):
        self.calls.append(("search_image", org, environment, payload))
        return {"hits": [{"id": "doc-1", "score": 0.9, "metadata": {"kind": "photo"}}]}

    def close(self):
        pass


def _store(client):
    return CloudStore(client, "org-1", "env-1", "user__abc__clip__preference", owns_client=False)


def test_add_image_encodes_bytes_and_routes_to_the_image_write(tmp_path):
    client = _FakeClient()
    store = _store(client)
    doc_id = store.add_image(b"pixels", id="img-1", text="a red bicycle", metadata={"page": 1})
    assert doc_id == "img-1"
    name, org, environment, payload = client.calls[0]
    assert (name, org, environment) == ("add_images", "org-1", "env-1")
    document = payload["documents"][0]
    assert base64.b64decode(document["image_b64"]) == b"pixels"
    assert document["text"] == "a red bicycle"
    assert document["metadata"] == {"page": "1"}, "metadata stringifies like every write verb"
    assert payload["idempotency_key"], "writes ride the idempotency contract"


def test_add_image_accepts_a_path(tmp_path):
    source = tmp_path / "photo.png"
    source.write_bytes(b"png-bytes")
    client = _FakeClient()
    _store(client).add_image(source)
    document = client.calls[0][3]["documents"][0]
    assert base64.b64decode(document["image_b64"]) == b"png-bytes"


def test_image_cap_and_empty_are_refused_client_side():
    client = _FakeClient()
    store = _store(client)
    with pytest.raises(ValueError, match="20 MB"):
        store.add_image(b"x" * (20 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="empty"):
        store.add_image(b"")
    assert client.calls == [], "a refused image never reaches the wire"


def test_search_by_image_routes_to_the_image_query():
    client = _FakeClient()
    hits = _store(client).search_by_image(
        b"query-pixels", k=3, filter={"kind": "photo"}, include_text=True
    )
    name, _, _, payload = client.calls[0]
    assert name == "search_image"
    assert base64.b64decode(payload["query_image_b64"]) == b"query-pixels"
    assert payload["k"] == 3
    assert payload["filter"] == {"kind": "photo"}
    assert payload["include_text"] is True
    assert hits[0].id == "doc-1"
