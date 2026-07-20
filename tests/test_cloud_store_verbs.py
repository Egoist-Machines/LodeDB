"""CloudStore verb wiring against a stub transport: what payload each verb
puts on the wire and how it folds the acceptance back into session state
(read-your-writes floor, `last_write_id`). No server involved — the accepted
write contract itself is covered end-to-end in `server/tests`."""

from __future__ import annotations

import pytest

# Collection must skip, not error, without the [cloud] extra installed
# (the modules below import httpx / pynacl at module level).
pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
pytest.importorskip("nacl", reason="needs the [cloud] extra's dependencies")

from lodedb.cloud.serving import CloudStore  # noqa: E402


class _StubClient:
    """Duck-types the one CloudClient call `remove_many` uses, recording the
    payload and answering a canned acceptance."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def remove_documents(self, org: str, environment: str, payload: dict) -> dict:
        self.calls.append((org, environment, payload))
        return {"write_id": "w-1", "seq": 7}


def _store(client: _StubClient) -> CloudStore:
    """A CloudStore over the stub, addressed like a real handle."""
    return CloudStore(client, "acme", "prod", "user-42", owns_client=False)


def test_remove_many_sends_one_batch_and_records_the_acceptance():
    client = _StubClient()
    store = _store(client)

    write_id = store.remove_many(["a", "b"])

    assert write_id == "w-1"
    assert store.last_write_id == "w-1"
    (org, environment, payload) = client.calls[0]
    assert (org, environment) == ("acme", "prod")
    assert payload["ids"] == ["a", "b"]
    assert payload["store"] == "user-42"
    # Every write carries an idempotency key so a transport retry replays the
    # original acceptance instead of registering a duplicate segment.
    assert payload["idempotency_key"]


def test_remove_delegates_to_the_batch_verb():
    client = _StubClient()
    store = _store(client)

    assert store.remove("a") == "w-1"
    assert client.calls[0][2]["ids"] == ["a"]


def test_remove_many_rejects_an_empty_batch():
    store = _store(_StubClient())
    with pytest.raises(ValueError, match="nothing to remove"):
        store.remove_many([])


class _UnprovisionedClient:
    """Answers every read with the 404 a not-yet-provisioned store gives."""

    def _refuse(self) -> dict:
        from lodedb.cloud.transfer import CloudError

        raise CloudError(404, "no such store")

    def search_many(self, org: str, environment: str, payload: dict) -> dict:
        return self._refuse()

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        return self._refuse()


def test_unprovisioned_batch_verbs_keep_query_cardinality():
    """Both batched search verbs answer an unprovisioned store with one empty
    hit list PER query — callers zip queries to results."""
    store = CloudStore(_UnprovisionedClient(), "acme", "prod", "user-42", owns_client=False)
    assert store.search_many(["a", "b", "c"]) == [[], [], []]
    assert store.search_many_by_vector([[0.1, 0.2], [0.3, 0.4]]) == [[], []]


def test_unprovisioned_browse_answers_empty():
    """Enumerating a user who hasn't written yet is the normal zero-setup
    flow, not an error."""
    store = CloudStore(_UnprovisionedClient(), "acme", "prod", "user-42", owns_client=False)
    assert store.browse() == []


class _BrowseClient:
    """Duck-types the add + browse calls, recording browse payloads and
    optionally answering one 425 (fold not caught up) before succeeding."""

    def __init__(self, too_early_first: bool = False) -> None:
        self.calls: list[dict] = []
        self._too_early = too_early_first

    def add_documents(self, org: str, environment: str, payload: dict) -> dict:
        return {"ids": ["m1"], "write_id": "w-9", "seq": 41}

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        self.calls.append(payload)
        if self._too_early:
            self._too_early = False
            from lodedb.cloud.transfer import CloudError

            raise CloudError(425, "not folded through seq 41 yet")
        return {"documents": [{"id": "m1", "metadata": {}, "chunk_count": 1}]}


def test_browse_carries_the_session_floor_and_retries_425():
    """Browse is a read like search: after a write on this handle it sends
    the session's read-your-writes floor as min_seq and briefly retries a
    425 instead of surfacing it — the write is durable, only its visibility
    trails by a fold cycle."""
    client = _BrowseClient(too_early_first=True)
    store = _store(client)
    store.add("first memory")  # acks with seq 41

    docs = store.browse()

    assert [doc["id"] for doc in docs] == ["m1"]
    assert [call.get("min_seq") for call in client.calls] == [41, 41]


class _AddClient:
    """Duck-types add_documents, recording each accepted payload."""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def add_documents(self, org: str, environment: str, payload: dict) -> dict:
        self.payloads.append(payload)
        return {
            "ids": [doc.get("id") or "assigned" for doc in payload["documents"]],
            "write_id": "w-1",
            "seq": 1,
        }


def test_add_coerces_metadata_like_the_local_handle():
    """Code written against the local `db.add` ergonomics may pass int/
    float/bool metadata values; the wire contract is strict str->str, so the
    handle stringifies exactly like the local `_coerce_metadata` — and
    refuses the value types the local handle refuses. Absent metadata stays
    None on the wire, not an empty map."""
    client = _AddClient()
    store = _store(client)

    store.add("hello", metadata={"year": 2020, "vip": True, "score": 1.5, "note": "plain"})
    meta = client.payloads[0]["documents"][0]["metadata"]
    assert meta == {"year": "2020", "vip": "true", "score": "1.5", "note": "plain"}

    with pytest.raises(ValueError, match="metadata value"):
        store.add("hello", metadata={"bad": ["a", "list"]})

    store.add("hello")
    assert client.payloads[-1]["documents"][0]["metadata"] is None


class _TextClient:
    """Duck-types browse_documents for get_texts: answers the asked-for ids
    (minus 'missing') plus one stray document a plain-page server might
    return."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        self.calls.append(payload)
        docs = [
            {"id": id, "metadata": {}, "chunk_count": 1, "text": f"text-{id}"}
            for id in payload["ids"]
            if id != "missing"
        ]
        docs.append({"id": "not-asked", "metadata": {}, "chunk_count": 1, "text": "stray"})
        return {"documents": docs}


def test_get_texts_batches_over_the_by_id_browse():
    """get_texts costs one by-id browse per hundred ids (not one request per
    id), omits missing ids, and keeps only requested ids — a server that
    answered a plain page could only under-answer, never mis-answer."""
    client = _TextClient()
    store = _store(client)

    ids = [f"doc-{i}" for i in range(150)] + ["missing"]
    texts = store.get_texts(ids)

    assert [len(call["ids"]) for call in client.calls] == [100, 51]
    assert all(call["include_text"] for call in client.calls)
    assert texts["doc-0"] == "text-doc-0" and texts["doc-149"] == "text-doc-149"
    assert "missing" not in texts and "not-asked" not in texts
    assert store.get_texts([]) == {}
    assert len(client.calls) == 2  # the empty batch made no request


class _TextOnlyClient:
    """A least-privilege `read:text` credential's view: browse (search-
    scoped) refuses with 403, the per-id text endpoint answers."""

    def __init__(self) -> None:
        self.browse_calls = 0
        self.text_calls: list[str] = []

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        self.browse_calls += 1
        from lodedb.cloud.transfer import CloudError

        raise CloudError(403, "returning stored text requires the 'read:search' scope")

    def store_text(self, org: str, environment: str, store: str, id: str, key=None) -> dict:
        self.text_calls.append(id)
        found = id != "missing"
        return {"id": id, "found": found, "text": f"text-{id}" if found else None}


def test_get_texts_falls_back_to_the_text_endpoint_for_text_only_keys():
    """A key holding only read:text cannot browse; get_texts then answers
    through the single-id text endpoint (the pre-batching shape) instead of
    failing a previously valid least-privilege call."""
    client = _TextOnlyClient()
    store = _store(client)

    texts = store.get_texts(["a", "missing", "b"])

    assert texts == {"a": "text-a", "b": "text-b"}
    assert client.browse_calls == 1
    assert client.text_calls == ["a", "missing", "b"]


def test_delete_store_erase_rides_the_query_string():
    """erase=True must reach the wire as ?erase=true — a silently dropped
    flag would downgrade a data-subject erasure to a grace-window delete."""
    import httpx

    from lodedb.cloud.transfer import CloudClient

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        stamp = "2026-07-20T00:00:00Z"
        return httpx.Response(
            200, json={"slug": "user-42-del-1", "deleted_at": stamp, "purge_after": stamp}
        )

    with CloudClient(
        "https://cloud.test", "tok", transport=httpx.MockTransport(handler)
    ) as client:
        client.delete_store("acme", "prod", "user-42")
        client.delete_store("acme", "prod", "user-42", erase=True)
    assert seen[0].endswith("/stores/user-42")
    assert seen[1].endswith("/stores/user-42?erase=true")


def test_browse_passes_ids_and_order_on_the_wire():
    """The by-id fetch and the recency order ride the wire only when asked
    for, so an untouched browse keeps its old payload shape."""
    client = _BrowseClient()
    store = _store(client)

    store.browse(ids=["a", "b"])
    store.browse(order="recent", after="m0", limit=2)
    store.browse()

    assert client.calls[0]["ids"] == ["a", "b"]
    assert "order" not in client.calls[0]
    assert (client.calls[1]["order"], client.calls[1]["after"], client.calls[1]["limit"]) == (
        "recent",
        "m0",
        2,
    )
    assert "ids" not in client.calls[1]
    assert "ids" not in client.calls[2] and "order" not in client.calls[2]
