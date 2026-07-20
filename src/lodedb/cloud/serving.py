"""`CloudStore` — the cloud handle that duck-types a local LodeDB.

    from lodedb.cloud import Client

    client = Client()                   # tenancy from the credential
    memories = client.store("user-42")
    memories.add("prefers email over phone")
    hits = memories.recall("how should I contact them about the invoice?")

A store is one end user's own LodeDB instance (the user's id in the
agentic-memory product), auto-provisioned by its first write — open a user
that doesn't exist yet and reads answer empty until the first `add`.
Isolation is physical: this handle cannot reach any other user's instance.

`lodedb.cloud.connect("org/environment/store")` remains as path-string sugar
over the same handle for one-off scripts and console copy-paste.

The returned :class:`CloudStore` implements the read subset of the local
`lodedb.LodeDB` handle — `search` / `search_many` / `get` / `get_texts` /
`stats` / `count`, with hits shaped exactly like the local `LodeSearchHit`
(`hit.score` / `hit.id` / `hit.metadata`, and tuple unpacking) — so RAG
adapters and MCP tool bodies written against a local handle work unmodified
against the cloud. On top of that come the memory verbs: `add` (with TTL),
`recall`, `context_block`, `browse`, and `delete_memories`.

Credentials resolve like the CLI's: explicit ``token=``/``host=`` arguments
win, then the ``ORECLOUD_TOKEN`` environment variable, then the ``lodedb
cloud login`` credentials file; the host defaults to the hosted control
plane. Server-side, queries are embedded with the same preset that indexed
the data, so scores are the engine's own.

Connecting pre-warms the store on the serving tier by default (the first
query then skips the hydration cold start); pass ``warm=False`` to skip.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from lodedb.cloud.transfer import CloudClient, CloudError, ManagedRemote


class CloudSearchHit:
    """One scored hit — attribute access and tuple unpacking, mirroring the
    local `LodeSearchHit`. `text` is set when the search asked for it;
    `matched` (recall only) names the sub-queries that surfaced the hit."""

    __slots__ = ("score", "id", "metadata", "text", "matched")

    def __init__(
        self,
        *,
        score: float,
        id: str,
        metadata: dict[str, Any],
        text: str | None = None,
    ) -> None:
        self.score = float(score)
        self.id = str(id)
        self.metadata = dict(metadata)
        self.text = text
        self.matched: list[str] = []

    def __iter__(self):
        yield self.score
        yield self.id
        yield self.metadata

    def __repr__(self) -> str:
        return f"CloudSearchHit(score={self.score:.4f}, id={self.id!r}, metadata={self.metadata!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CloudSearchHit):
            return (self.score, self.id, self.metadata) == (
                other.score,
                other.id,
                other.metadata,
            )
        if isinstance(other, tuple):
            return tuple(self) == other
        return NotImplemented


def _hit(row: dict) -> CloudSearchHit:
    return CloudSearchHit(
        score=row["score"], id=row["id"], metadata=row["metadata"], text=row.get("text")
    )


def _coerced_metadata(metadata: dict[str, Any] | None) -> dict[str, str] | None:
    """The local handle's metadata coercion (ints/floats/bools stringified,
    other value types refused with the same error), applied before the wire.
    The server's contract is strict str->str; code written against the local
    ``db.add`` ergonomics must not 422 on ``{"year": 2020}``. ``None`` stays
    ``None`` — absent metadata, not an empty map."""
    if metadata is None:
        return None
    from lodedb.local.db import _coerce_metadata

    return _coerce_metadata(metadata)


class CloudStore:
    """A read handle over one managed store, duck-typing the local LodeDB
    read surface. Create via :func:`connect`."""

    def __init__(
        self,
        client: CloudClient,
        org: str,
        environment: str,
        store: str,
        key: str | None = None,
        *,
        read_your_writes: bool = True,
        write_visibility_timeout: float = 30.0,
        owns_client: bool = True,
    ) -> None:
        # Handles from `Client.store()` share the Client's connection pool
        # (owns_client=False), so closing one user's handle must not sever
        # every other user's; a standalone `connect()` handle owns its pool.
        self._client = client
        self._owns_client = bool(owns_client)
        self.org = org
        self.environment = environment
        self.store = store
        self.key = key
        # Session read-your-writes: the highest `seq` a write on THIS handle
        # acked with. Searches pass it as `min_seq` and briefly retry on 425
        # until the fold covers it (opt out with read_your_writes=False).
        self._read_your_writes = bool(read_your_writes)
        self._write_visibility_timeout = float(write_visibility_timeout)
        self._last_seq = 0
        # The most recent accepted write's id — the `wait_for` handle.
        self.last_write_id: str | None = None

    # ------------------------------------------------------------- queries

    def _empty_if_unprovisioned(self, call, empty):
        """Runs one read, answering `empty` when this user's store simply
        doesn't exist yet — a store is one end user and materializes on its
        first write, so reading a fresh user before their first memory is
        the normal zero-setup flow (the hosted MCP tools behave the same).
        Every other error stays loud."""
        try:
            return call()
        except CloudError as error:
            if error.status_code == 404 and "no such store" in error.detail:
                return empty
            raise

    def _searched(self, call, payload: dict[str, Any]) -> dict:
        """Runs one search call with session read-your-writes: `min_seq` is
        this handle's last acked write, and a 425 (fold not caught up yet) is
        retried briefly instead of surfacing — the write is durable, only its
        visibility is trailing by a fold cycle."""
        if self._read_your_writes and self._last_seq > 0:
            payload["min_seq"] = self._last_seq
        deadline = time.monotonic() + self._write_visibility_timeout
        while True:
            try:
                return call(self.org, self.environment, payload)
            except CloudError as error:
                if error.status_code != 425 or time.monotonic() >= deadline:
                    raise
                time.sleep(0.25)

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        mode: str | None = None,
        include_text: bool = False,
    ) -> list[CloudSearchHit]:
        """Top-`k` hits, engine-scored. `include_text=True` returns each
        hit's stored text inline (requires a `read:text`-scoped key and the
        store's `expose_text` flag). After a write on this handle, the search
        waits (briefly) for the write to become visible — session
        read-your-writes; disable with `connect(..., read_your_writes=False)`."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "query": query,
            "k": k,
            "filter": filter,
            "mode": mode,
            "include_text": include_text,
        }
        result = self._empty_if_unprovisioned(
            lambda: self._searched(self._client.search, payload), {"hits": []}
        )
        return [_hit(row) for row in result["hits"]]

    def search_many(
        self,
        queries: list[str],
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        mode: str | None = None,
        include_text: bool = False,
    ) -> list[list[CloudSearchHit]]:
        """Top-`k` hits per query, order-preserving — the batched search."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "queries": queries,
            "k": k,
            "filter": filter,
            "mode": mode,
            "include_text": include_text,
        }
        result = self._empty_if_unprovisioned(
            lambda: self._searched(self._client.search_many, payload),
            {"results": [[] for _ in queries]},
        )
        return [[_hit(row) for row in rows] for rows in result["results"]]

    # -------------------------------------------------------------- writes

    # Transport-level retries per write. Safe only because every write
    # carries an idempotency key: the server either registers the request
    # once or replays the original acceptance.
    _WRITE_TRANSPORT_RETRIES = 2

    def _accepted(self, result: dict) -> dict:
        """Records an accepted write's `seq` as this session's
        read-your-writes floor and remembers its `write_id`."""
        seq = int(result.get("seq", 0) or 0)
        if seq > self._last_seq:
            self._last_seq = seq
        self.last_write_id = str(result["write_id"])
        return result

    def _written(self, call, payload: dict[str, Any]) -> dict:
        """One accepted write under an auto-generated idempotency key.

        The failure this exists for: the server accepts the write but the
        response is lost (timeout, dropped connection). A naive resend would
        register a second segment — duplicate documents under fresh ids. The
        key pins the request, so the retry (same key, byte-identical body)
        gets the original acceptance replayed instead. Only transport-level
        failures are retried; an HTTP error is a real answer.
        """
        payload["idempotency_key"] = uuid.uuid4().hex
        attempt = 0
        while True:
            try:
                return self._accepted(call(self.org, self.environment, payload))
            except httpx.TransportError:
                attempt += 1
                if attempt > self._WRITE_TRANSPORT_RETRIES:
                    raise
                time.sleep(0.25 * attempt)

    def add(
        self,
        text: str,
        *,
        id: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Add (or replace) one document — the cloud `db.add`. The text is
        embedded server-side and the write is ACCEPTED (durable + ordered)
        when this returns; visibility follows within seconds, and a search on
        this handle waits for it (session read-your-writes). The first write
        to a store that doesn't exist yet provisions it (a store is one end
        user). `ttl_seconds` hides the memory from reads once it lapses
        (hide-not-delete); `agent_id`/`run_id` stamp provenance that reads
        can narrow on. Requires a `write`-scoped key. Returns the document
        id."""
        (doc_id,) = self.add_many(
            [{"text": text, "id": id, "metadata": metadata}],
            ttl_seconds=ttl_seconds,
            agent_id=agent_id,
            run_id=run_id,
        )
        return doc_id

    def add_many(
        self,
        documents: list[dict[str, Any]],
        *,
        ttl_seconds: int | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        """Add a batch of ``{"text", "id"?, "metadata"?}`` documents as one
        accepted write (one segment; the fold batches concurrent writes into
        one commit). Metadata values are stringified exactly like the local
        handle's (the wire contract is strict str->str). Returns the ids, in
        order — assigned at acceptance."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "documents": [
                {
                    "text": doc["text"],
                    "id": doc.get("id"),
                    "metadata": _coerced_metadata(doc.get("metadata")),
                }
                for doc in documents
            ],
        }
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if run_id is not None:
            payload["run_id"] = run_id
        result = self._written(self._client.add_documents, payload)
        return list(result["ids"])

    def add_vectors(
        self,
        vector: Sequence[float],
        *,
        id: str | None = None,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        ttl_seconds: int | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """Add (or replace) one pre-embedded document — the cloud
        `db.add_vectors`, for stores created with `vector_dim` (the server
        never embeds; the vector must be exactly the store's dims and is
        unit-normalized server-side, the local default). `text` is optional
        retained payload for `get()`. Returns the document id."""
        (doc_id,) = self.add_vectors_many(
            [{"vector": list(vector), "id": id, "text": text, "metadata": metadata}],
            ttl_seconds=ttl_seconds,
            agent_id=agent_id,
            run_id=run_id,
        )
        return doc_id

    def add_vectors_many(
        self,
        documents: list[dict[str, Any]],
        *,
        ttl_seconds: int | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[str]:
        """Add a batch of ``{"vector", "id"?, "text"?, "metadata"?}``
        documents as one accepted write. Vector-store counterpart of
        `add_many` (metadata stringified the same way); returns the ids in
        order."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "documents": [
                {
                    "vector": list(doc["vector"]),
                    "text": doc.get("text"),
                    "id": doc.get("id"),
                    "metadata": _coerced_metadata(doc.get("metadata")),
                }
                for doc in documents
            ],
        }
        if ttl_seconds is not None:
            payload["ttl_seconds"] = ttl_seconds
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if run_id is not None:
            payload["run_id"] = run_id
        result = self._written(self._client.add_documents, payload)
        return list(result["ids"])

    def search_by_vector(
        self,
        vector: Sequence[float],
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_text: bool = False,
    ) -> list[CloudSearchHit]:
        """Top-`k` hits for a pre-embedded query — the cloud
        `db.search_by_vector`, for vector stores (which have no server-side
        embedder; the vector must be exactly the store's dims)."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "query_vector": list(vector),
            "k": k,
            "filter": filter,
            "include_text": include_text,
        }
        result = self._empty_if_unprovisioned(
            lambda: self._searched(self._client.search, payload), {"hits": []}
        )
        return [_hit(row) for row in result["hits"]]

    def search_many_by_vector(
        self,
        vectors: list[Sequence[float]],
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_text: bool = False,
    ) -> list[list[CloudSearchHit]]:
        """One engine batch of pre-embedded queries — the cloud
        `db.search_many_by_vector`."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "query_vectors": [list(vector) for vector in vectors],
            "k": k,
            "filter": filter,
            "include_text": include_text,
        }
        result = self._empty_if_unprovisioned(
            lambda: self._searched(self._client.search_many, payload),
            # One empty hit list PER query (mirroring search_many): callers
            # zip queries to results, so the unprovisioned answer must keep
            # the cardinality.
            {"results": [[] for _ in vectors]},
        )
        return [[_hit(row) for row in hits] for hits in result["results"]]

    def remove(self, id: str) -> str:
        """Remove one document by id — the cloud `db.remove`, async-first:
        returns the accepted write's id once the removal is durably queued.
        Whether the document existed is decided when the fold applies the
        delete — ``wait_for(write_id)["result"]["removed"][0]`` answers it."""
        return self.remove_many([id])

    def remove_many(self, ids: Sequence[str]) -> str:
        """Remove a batch of documents by id as one accepted write — the
        cloud `db.remove_many`, async-first like :meth:`remove`: returns the
        write's id once the removals are durably queued (one segment, one
        fold). Per-id outcomes are decided when the fold applies the deletes
        — ``wait_for(write_id)["result"]["removed"]`` is the parallel bool
        list. An empty batch raises: with no accepted write there is no id to
        return (the local handle's ``remove_many([])`` no-op returns 0
        instead)."""
        document_ids = list(ids)
        if not document_ids:
            raise ValueError("ids is empty — nothing to remove")
        payload: dict[str, Any] = {"store": self.store, "key": self.key, "ids": document_ids}
        result = self._written(self._client.remove_documents, payload)
        return str(result["write_id"])

    def wait_for(self, write_id: str, *, timeout: float = 30.0) -> dict[str, Any]:
        """Blocks until an accepted write folds; returns its final status
        (`state`, covering `snapshot_id`/`generation`, `result`). Raises
        :class:`CloudError` (502) when the write was condemned, and
        :class:`TimeoutError` when `timeout` elapses first (the write stays
        queued and will still fold)."""
        deadline = time.monotonic() + float(timeout)
        while True:
            status = self._client.write_status(
                self.org, self.environment, self.store, write_id, key=self.key
            )
            if status["state"] == "folded":
                return status
            if status["state"] == "condemned":
                raise CloudError(
                    502,
                    "this write could not be applied (segment condemned: "
                    f"{status.get('error')}) — resubmit the documents",
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"write {write_id} did not fold within {timeout}s (it stays "
                    "queued and will still be applied)"
                )
            time.sleep(0.25)

    # ---------------------------------------------------------------- text

    def get(self, id: str) -> str | None:
        """One document's stored raw text by id (None when absent) — the
        cloud `db.get(id)`. Requires `read:text` and the store's
        `expose_text` flag."""
        result = self._empty_if_unprovisioned(
            lambda: self._client.store_text(
                self.org, self.environment, self.store, id, key=self.key
            ),
            {"found": False, "text": None},
        )
        return result["text"] if result["found"] else None

    # The local handle's alias.
    get_text = get

    def get_texts(self, ids: list[str]) -> dict[str, str]:
        """Stored text for several ids (missing ids are omitted), fetched as
        by-id browse pages of 100 (the by-id bound) — one request per
        hundred ids, not one per id. Browse is a search-scoped read, so a
        least-privilege key holding only `read:text` falls back to the
        single-id text endpoint (one request per id, the pre-batching
        behavior) instead of failing a previously valid call. Answers are
        filtered to the requested ids, so an older control plane (which
        ignores the by-id fields and answers a plain page) can only
        under-answer, never mis-answer."""
        texts: dict[str, str] = {}
        for start in range(0, len(ids), 100):
            batch = [str(id) for id in ids[start : start + 100]]
            wanted = set(batch)
            try:
                page = self.browse(ids=batch, include_text=True)
            except CloudError as error:
                if error.status_code != 403:
                    raise
                # The key can't browse (no search scope). The per-id text
                # endpoint is exactly what `read:text` grants — and if the
                # 403 was about text access itself, the fallback's first
                # request surfaces the same actionable refusal.
                return self._get_texts_by_id(ids)
            for doc in page:
                text = doc.get("text")
                if doc.get("id") in wanted and text is not None:
                    texts[doc["id"]] = text
        return texts

    def _get_texts_by_id(self, ids: list[str]) -> dict[str, str]:
        """The pre-batching shape: one text-endpoint request per id."""
        texts: dict[str, str] = {}
        for id in ids:
            text = self.get(id)
            if text is not None:
                texts[id] = text
        return texts

    # --------------------------------------------------------------- stats

    def stats(self, *, warm: bool = False) -> dict[str, Any]:
        """Metrics-only serving stats (counts, snapshot identity, payload
        flags) — the cloud `db.stats()` subset."""
        return self._client.serving_stats(
            self.org, self.environment, self.store, self.key, warm=warm
        )

    def count(self) -> int:
        stats = self._empty_if_unprovisioned(self.stats, {})
        return int(stats.get("document_count", 0) or 0)

    # -------------------------------------------------------- memory verbs

    def recall(
        self,
        text: str,
        *,
        k: int = 10,
        filter: dict[str, Any] | None = None,
        include_text: bool = False,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[CloudSearchHit]:
        """Non-exact retrieval from RAW text — pass a whole user message;
        the server derives sub-queries and fuses the rankings. Each hit's
        `matched` attribute (set on the returned objects) names the
        sub-queries that surfaced it. `agent_id`/`run_id` narrow to one
        agent's or one session's memories."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "text": text,
            "k": k,
            "filter": filter,
            "include_text": include_text,
        }
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if run_id is not None:
            payload["run_id"] = run_id
        result = self._empty_if_unprovisioned(
            lambda: self._searched(self._client.recall, payload), {"hits": []}
        )
        hits = []
        for row in result["hits"]:
            hit = _hit(row)
            hit.matched = row.get("matched", [])  # provenance, recall-only
            hits.append(hit)
        return hits

    def context_block(
        self,
        text: str | None = None,
        *,
        max_chars: int = 4_000,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> str:
        """A prompt-ready context block of this user's memories: the most
        recent ones plus (when `text` is given) the ones relevant to it.
        Requires text access (`read:text` scope + the store's `expose_text`
        flag)."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "max_chars": max_chars,
        }
        if text is not None:
            payload["text"] = text
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if run_id is not None:
            payload["run_id"] = run_id
        result = self._empty_if_unprovisioned(
            lambda: self._searched(self._client.context_block, payload),
            {"block": f"# Context: user {self.store}"},
        )
        return result["block"]

    def browse(
        self,
        *,
        after: str | None = None,
        limit: int = 25,
        include_text: bool = False,
        filter: dict[str, Any] | None = None,
        ids: Sequence[str] | None = None,
        order: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """This store's memories (ids + metadata, text when asked and
        allowed), in one of three shapes: keyset pages in the engine's
        stable id order (the default), most-recent-first pages
        (`order="recent"` — same last-id cursor, but it only holds within
        one served snapshot; a 422 asks the caller to restart when the
        store changed under the enumeration, and a match set past the
        server's scan cap also 422s — narrow the filter or use id order),
        or a by-id fetch (`ids=[...]` — exactly the named documents that
        exist, no paging; the server refuses `after`/`order` beside it).
        Like search, the enumeration honors session read-your-writes: after
        a write on this handle it waits briefly for that write's fold.
        `ids`/`order`/the read-your-writes token need a control plane that
        knows them — an older server ignores unknown browse fields and
        answers a plain id-ordered page."""
        payload: dict[str, Any] = {
            "store": self.store,
            "key": self.key,
            "after": after,
            "limit": limit,
            "include_text": include_text,
            "filter": filter,
        }
        if ids is not None:
            payload["ids"] = list(ids)
        if order is not None:
            payload["order"] = order
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if run_id is not None:
            payload["run_id"] = run_id
        result = self._empty_if_unprovisioned(
            lambda: self._searched(self._client.browse_documents, payload),
            {"documents": []},
        )
        return result["documents"]

    def delete_memories(
        self, *, agent_id: str | None = None, run_id: str | None = None
    ) -> dict[str, Any]:
        """Delete this store's memories in place (expired ones included),
        narrowable to one agent/run. The store stays registered — to forget
        the user entirely (and free their entitlement slot), delete the
        store itself (`CloudClient.delete_store` / `lodedb cloud store delete`).
        Returns the acceptance (`write_ids`, `document_count`, `max_seq`);
        pass a write id to ``wait_for`` to block until the removal folds. A
        search on this handle waits for the deletion (session
        read-your-writes), same as any other write."""
        payload: dict[str, Any] = {"store": self.store, "key": self.key}
        if agent_id is not None:
            payload["agent_id"] = agent_id
        if run_id is not None:
            payload["run_id"] = run_id
        result = self._client.delete_memories(self.org, self.environment, payload)
        for write_id in result.get("write_ids", []):
            self.last_write_id = str(write_id)
        # The deletion is a write like any other: raise the session's
        # read-your-writes floor so a search on this handle waits for it —
        # otherwise deleted memories can resurface until the fold lands.
        max_seq = result.get("max_seq")
        if max_seq is not None and int(max_seq) > self._last_seq:
            self._last_seq = int(max_seq)
        return result

    # ----------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Closes the underlying HTTP pool when this handle owns it; a
        handle borrowed from a `Client` leaves the shared pool alone."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> CloudStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        where = f"{self.org}/{self.environment}/{self.store}"
        return f"CloudStore({where!r})"


@dataclass(frozen=True)
class _BareStore:
    """A single-segment target: just the store id. The org/environment half
    comes from the credential (`resolve_tenancy`), the same way
    `Client().store(...)` resolves it — so a user never retypes what their
    environment-scoped token already pins down."""

    store: str


def _parse_target(target: str) -> ManagedRemote | _BareStore:
    """Accepts a bare store id (`user-42` — org/environment resolve from the
    credential), `org/environment/store`, and the explicit `orecloud://`
    spellings of all of these (the URL form also allows `org/environment`,
    defaulting the store). The store segment is the end-user id in the
    agentic-memory product — a store auto-provisions on its first write, so
    nothing needs creating first."""
    body = target
    is_url = target.startswith(ManagedRemote.SCHEME)
    if is_url:
        body = target[len(ManagedRemote.SCHEME) :]
    # Empty segments are malformed, never silently collapsed: filtering them
    # would reinterpret `orecloud://org//store` as the two-segment
    # org/environment form and aim the handle at the wrong tenancy.
    segments = body.strip("/").split("/")
    if len(segments) == 1 and segments[0]:
        return _BareStore(segments[0])
    if (len(segments) == 3 or (len(segments) == 2 and is_url)) and all(segments):
        return ManagedRemote(*segments)
    raise CloudError(
        422,
        f"malformed target {target!r}; expected a bare store id ('user-42' — "
        "org/environment come from the credential), 'org/environment/store', "
        "or an orecloud:// URL",
    )


def connect(
    target: str,
    *,
    token: str | None = None,
    host: str | None = None,
    key: str | None = None,
    warm: bool = True,
    timeout: float = 30.0,
    read_your_writes: bool = True,
    transport: httpx.BaseTransport | None = None,
) -> CloudStore:
    """Open a read handle over a managed store, addressed by path string.

    Prefer :class:`lodedb.cloud.Client`: the org/environment half of `target`
    repeats what an environment-scoped token already pins down, and
    ``Client().store("user-42")`` resolves it from the credential instead.
    This stays as sugar for one-off scripts and console copy-paste. It is
    also the seam behind ``lodedb``'s constructor front doors —
    ``LodeDB.cloud("user-42")`` and the ``LodeDB("orecloud://…")``
    config-string dispatch both land here (lodedb releases that ship the
    `[cloud]` extra), so the two must keep accepting the same keywords.

    `target` is a bare store id (`"user-42"` — the org/environment half
    resolves from the credential via `resolve_tenancy`, exactly like
    `Client().store()`), `"org/environment/store"`, or an `orecloud://` URL
    of either (the URL form also allows `org/environment`, defaulting the
    store). Credentials: explicit arguments, else the `ORECLOUD_TOKEN`
    environment variable, else the credentials file `lodedb cloud login`
    wrote (the host defaults to the hosted control plane). `warm=True`
    (default) asks the serving tier
    to hydrate and open the store now, so the first query is warm; it also
    verifies the target exists and the credential can read it. `key` names
    the index key when the store holds more than one (rare — a pushed LodeDB
    directory can carry several). `read_your_writes=True` (default) makes a
    search after a write on this handle wait briefly for that write's fold,
    so the session always sees its own writes.
    """
    from lodedb.cloud.client import resolve_credentials, resolve_tenancy

    remote = _parse_target(target)
    host, token = resolve_credentials(token, host)
    client = CloudClient(host, token, transport=transport, timeout=timeout)
    if isinstance(remote, _BareStore):
        # A bare store id carries no org/environment: resolve them from the
        # credential (token binding, else the account's only choices), and
        # never leak the pool when that resolution refuses.
        try:
            org, environment = resolve_tenancy(client, None, None)
        except BaseException:
            client.close()
            raise
        remote = ManagedRemote(org, environment, remote.store)
    store = CloudStore(
        client,
        remote.org,
        remote.environment,
        remote.store,
        key,
        read_your_writes=read_your_writes,
    )
    if warm:
        try:
            store.stats(warm=True)
        except CloudError as error:
            # Two fine-to-connect 404s: a store that exists but holds
            # nothing yet (first `add()` creates its first snapshot), and a
            # store that doesn't exist at all — a store is one end user,
            # and users materialize on their first write, so connecting to
            # a new user before their first memory is the normal flow. A
            # bad org/environment still fails loudly (different detail).
            connectable = error.status_code == 404 and (
                "nothing has been pushed" in error.detail
                or "no such store" in error.detail
            )
            if not connectable:
                client.close()
                raise
        except BaseException:
            # Transport failures (DNS, refused connection, timeout) must not
            # leak the freshly opened pool either.
            client.close()
            raise
    return store


# Back-compat alias from the index->store rename: existing imports of
# CloudIndex keep working; new code should use CloudStore.
CloudIndex = CloudStore
