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
    """Answers every search with the 404 a not-yet-provisioned store gives."""

    def search_many(self, org: str, environment: str, payload: dict) -> dict:
        from lodedb.cloud.transfer import CloudError

        raise CloudError(404, "no such store")


def test_unprovisioned_batch_verbs_keep_query_cardinality():
    """Both batched search verbs answer an unprovisioned store with one empty
    hit list PER query — callers zip queries to results."""
    store = CloudStore(_UnprovisionedClient(), "acme", "prod", "user-42", owns_client=False)
    assert store.search_many(["a", "b", "c"]) == [[], [], []]
    assert store.search_many_by_vector([[0.1, 0.2], [0.3, 0.4]]) == [[], []]
