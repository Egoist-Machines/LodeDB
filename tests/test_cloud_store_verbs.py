"""CloudStore verb wiring against a stub transport: what payload each verb
puts on the wire and how it folds the acceptance back into session state
(read-your-writes floor, `last_write_id`). No server involved; the accepted
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
    hit list PER query, so callers zip queries to results."""
    store = CloudStore(_UnprovisionedClient(), "acme", "prod", "user-42", owns_client=False)
    assert store.search_many(["a", "b", "c"]) == [[], [], []]
    assert store.search_many_by_vector([[0.1, 0.2], [0.3, 0.4]]) == [[], []]


def test_unprovisioned_browse_answers_empty():
    """Enumerating a user who hasn't written yet is the normal zero-setup
    flow, not an error."""
    store = CloudStore(_UnprovisionedClient(), "acme", "prod", "user-42", owns_client=False)
    assert store.browse() == []


class _NeverPushedClient:
    """Answers reads with the 404 a created-but-never-written store gives
    (the store row exists; no snapshot has published)."""

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        from lodedb.cloud.transfer import CloudError

        raise CloudError(404, "nothing has been pushed to this store yet — push a snapshot first")


def test_created_but_never_written_store_reads_empty():
    """A store created ahead of its first write must read as empty, exactly
    like one that does not exist yet; only genuinely foreign 404s stay loud."""
    store = CloudStore(_NeverPushedClient(), "acme", "prod", "user-42", owns_client=False)
    assert store.browse() == []
    assert store.list_documents() == []


class _AlwaysTooEarlyClient:
    """Every browse answers 425, like a store whose fold never catches up."""

    def add_documents(self, org: str, environment: str, payload: dict) -> dict:
        return {"ids": ["m1"], "write_id": "w-9", "seq": 41}

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        from lodedb.cloud.transfer import CloudError

        raise CloudError(425, "not folded through seq 41 yet", retry_after=0.01)


def test_list_documents_budget_bounds_the_visibility_wait():
    """A page's 425 retry stops at the walk's own deadline, not the handle's
    30s visibility budget, and surfaces as the walk's TimeoutError: the
    caller set one bound and must get one failure mode for breaching it."""
    import time

    store = _store(_AlwaysTooEarlyClient())
    store.add("first memory")  # arms the session floor

    started = time.monotonic()
    with pytest.raises(TimeoutError, match="did not finish"):
        store.list_documents(timeout=0.05)
    assert time.monotonic() - started < 5.0


class _BrowseClient:
    """Duck-types the add + browse calls, recording browse payloads and
    optionally answering one 425 (fold not caught up) before succeeding.
    When `retry_after` is given the refusal carries it, like a server that
    sends the Retry-After header."""

    def __init__(self, too_early_first: bool = False, retry_after: float | None = None) -> None:
        self.calls: list[dict] = []
        self._too_early = too_early_first
        self._retry_after = retry_after

    def add_documents(self, org: str, environment: str, payload: dict) -> dict:
        return {"ids": ["m1"], "write_id": "w-9", "seq": 41}

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        self.calls.append(payload)
        if self._too_early:
            self._too_early = False
            from lodedb.cloud.transfer import CloudError

            raise CloudError(425, "not folded through seq 41 yet", retry_after=self._retry_after)
        return {"documents": [{"id": "m1", "metadata": {}, "chunk_count": 1}]}


def test_browse_carries_the_session_floor_and_retries_425():
    """Browse is a read like search: after a write on this handle it sends
    the session's read-your-writes floor as min_seq and briefly retries a
    425 instead of surfacing it; the write is durable, only its visibility
    trails by a fold cycle."""
    client = _BrowseClient(too_early_first=True)
    store = _store(client)
    store.add("first memory")  # acks with seq 41

    docs = store.browse()

    assert [doc["id"] for doc in docs] == ["m1"]
    assert [call.get("min_seq") for call in client.calls] == [41, 41]


def test_read_retry_paces_itself_by_the_server_retry_after(monkeypatch):
    """A 425 carrying Retry-After paces the retry at the server's ask
    (polling faster just burns the search rate limit); without one the
    pause stays the 250ms default, and the server's ask never sleeps past
    the handle's own visibility budget."""
    pauses: list[float] = []
    monkeypatch.setattr("lodedb.cloud.serving.time.sleep", lambda s: pauses.append(s))

    store = _store(_BrowseClient(too_early_first=True, retry_after=1.0))
    store.add("first memory")
    store.browse()
    assert pauses == [1.0]

    pauses.clear()
    store = _store(_BrowseClient(too_early_first=True))
    store.add("first memory")
    store.browse()
    assert pauses == [0.25]

    pauses.clear()
    client = _BrowseClient(too_early_first=True, retry_after=60.0)
    store = CloudStore(
        client, "acme", "prod", "user-42", owns_client=False, write_visibility_timeout=0.5
    )
    store.add("first memory")
    store.browse()
    assert pauses and pauses[0] <= 0.5


def test_cloud_error_parses_the_retry_after_header():
    """The transport keeps the server's Retry-After on the refusal it
    raises; absent or malformed headers become None, never a crash."""
    import httpx

    from lodedb.cloud.transfer import CloudError, _raise_for

    def refusal(headers: dict[str, str]) -> CloudError:
        response = httpx.Response(425, json={"detail": "not folded yet"}, headers=headers)
        with pytest.raises(CloudError) as caught:
            _raise_for(response)
        return caught.value

    assert refusal({"Retry-After": "1"}).retry_after == 1.0
    assert refusal({"Retry-After": "2.5"}).retry_after == 2.5
    assert refusal({}).retry_after is None
    assert refusal({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}).retry_after is None
    assert refusal({"Retry-After": "-3"}).retry_after is None


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
    handle stringifies exactly like the local `_coerce_metadata`, and
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
    """Duck-types browse_documents + store_text for get_texts: the by-id
    browse answers every asked-for id except 'missing'; the text endpoint
    records which ids needed a per-id confirmation."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.text_calls: list[str] = []

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        self.calls.append(payload)
        return {
            "documents": [
                {"id": id, "metadata": {}, "chunk_count": 1, "text": f"text-{id}"}
                for id in payload["ids"]
                if id != "missing"
            ]
        }

    def store_text(self, org: str, environment: str, store: str, id: str, key=None) -> dict:
        self.text_calls.append(id)
        return {"id": id, "found": False, "text": None}


def test_get_texts_batches_over_the_by_id_browse():
    """get_texts does its bulk work as one by-id browse per hundred ids (not
    one request per id); an id absent from its page is confirmed through the
    text endpoint so the answer keeps exact per-id semantics, and an empty
    request makes no request at all."""
    client = _TextClient()
    store = _store(client)

    ids = [f"doc-{i}" for i in range(150)] + ["missing"]
    texts = store.get_texts(ids)

    assert [len(call["ids"]) for call in client.calls] == [100, 51]
    assert all(call["include_text"] for call in client.calls)
    assert texts["doc-0"] == "text-doc-0" and texts["doc-149"] == "text-doc-149"
    assert "missing" not in texts
    assert client.text_calls == ["missing"]  # only the absent id re-confirms
    assert store.get_texts([]) == {}
    assert len(client.calls) == 2  # the empty request made no HTTP call


def test_get_texts_distrusts_a_plain_page_answer():
    """A document nobody asked for means the control plane ignored the by-id
    fields (an older server answering a plain page). Every requested id then
    goes through the text endpoint, so the result stays exact instead of
    silently partial."""

    class _PlainPageClient:
        def __init__(self) -> None:
            self.text_calls: list[str] = []

        def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
            return {
                "documents": [
                    {"id": "stray", "metadata": {}, "chunk_count": 1, "text": "stray-text"},
                    {
                        "id": payload["ids"][0],
                        "metadata": {},
                        "chunk_count": 1,
                        "text": f"text-{payload['ids'][0]}",
                    },
                ]
            }

        def store_text(self, org: str, environment: str, store: str, id: str, key=None) -> dict:
            self.text_calls.append(id)
            return {"id": id, "found": True, "text": f"endpoint-{id}"}

    client = _PlainPageClient()
    store = _store(client)

    texts = store.get_texts(["a", "b"])

    assert texts == {"a": "endpoint-a", "b": "endpoint-b"}
    assert client.text_calls == ["a", "b"]


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
    """erase=True must reach the wire as ?erase=true. A silently dropped
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


class _PagingBrowseClient:
    """Answers filtered browse pages from a fixed id-ordered corpus, honoring
    after/limit and recording each payload. Enough server for the
    list_documents enumeration loop."""

    def __init__(self, count: int) -> None:
        self.calls: list[dict] = []
        self._ids = [f"doc-{index:04d}" for index in range(count)]

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        self.calls.append(payload)
        ids = self._ids
        if payload.get("after"):
            ids = [id for id in ids if id > payload["after"]]
        page = ids[: payload["limit"]]
        return {
            "documents": [
                {"id": id, "metadata": {"namespace": "default"}, "chunk_count": 1}
                for id in page
            ],
            "count": len(self._ids),
            "snapshot_id": "snap-1",
            "generation": 1,
        }


def test_list_documents_walks_every_page_in_local_record_shape():
    """limit=None enumerates the whole match set in 100-document pages and
    answers local-handle-shaped records."""
    client = _PagingBrowseClient(250)
    store = CloudStore(client, "acme", "prod", "user-42", owns_client=False)

    records = store.list_documents()

    assert len(records) == 250
    assert records[0] == {
        "id": "doc-0000",
        "metadata": {"namespace": "default"},
        "chunk_count": 1,
    }
    assert [call["limit"] for call in client.calls] == [100, 100, 100]
    assert client.calls[1]["after"] == "doc-0099"
    assert client.calls[2]["after"] == "doc-0199"


def test_list_documents_forwards_filter_and_pages_like_the_local_cursor():
    client = _PagingBrowseClient(5)
    store = CloudStore(client, "acme", "prod", "user-42", owns_client=False)

    records = store.list_documents(
        filter={"namespace": "default"}, after="doc-0001", limit=2
    )

    assert [record["id"] for record in records] == ["doc-0002", "doc-0003"]
    assert client.calls[0]["filter"] == {"namespace": "default"}
    assert client.calls[0]["after"] == "doc-0001"
    assert client.calls[0]["limit"] == 2


def test_list_documents_bounds_the_walk():
    """The keyword-only bounds fail closed before another page is fetched: a
    match set past max_documents raises ValueError, an outlived timeout raises
    TimeoutError. Enumeration never mutates, so both are safe to retry with
    a narrower filter or explicit paging."""
    client = _PagingBrowseClient(250)
    store = CloudStore(client, "acme", "prod", "user-42", owns_client=False)

    with pytest.raises(ValueError, match="more than 100 documents"):
        store.list_documents(max_documents=100)
    with pytest.raises(TimeoutError, match="did not finish"):
        store.list_documents(timeout=0.0)


class _SlowFinalPageClient:
    """One short (final) browse page whose fetch alone spends the caller's
    whole time budget, ticking the injected clock as a slow server would."""

    def __init__(self, clock: dict) -> None:
        self._clock = clock

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        self._clock["now"] += 5.0
        return {"documents": [{"id": "d1", "metadata": {}, "chunk_count": 1}]}


def test_list_documents_fails_closed_when_the_final_page_outlives_the_budget(monkeypatch):
    """The timeout is enforced around each fetch, not just before it: a
    single slow final page must raise, never turn a spent budget into a
    quiet success; callers treat the bound as a refusal mechanism."""
    clock = {"now": 0.0}
    monkeypatch.setattr("lodedb.cloud.serving.time.monotonic", lambda: clock["now"])
    store = CloudStore(
        _SlowFinalPageClient(clock), "acme", "prod", "user-42", owns_client=False
    )

    with pytest.raises(TimeoutError, match="did not finish"):
        store.list_documents(timeout=1.0)


def test_unprovisioned_list_documents_answers_empty():
    """Enumerating a user who hasn't written yet is the normal zero-setup
    flow, not an error (same rule as browse)."""
    store = CloudStore(_UnprovisionedClient(), "acme", "prod", "user-42", owns_client=False)
    assert store.list_documents() == []
