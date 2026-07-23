"""The HTTP-speaking half of the client: a thin typed wrapper over the /v1
control-plane API, the sealed-box login handoff, and the managed
(`orecloud://`) transfer verbs.

Synchronous httpx by design. The CLI is a sequential tool, and the SDK's
cloud methods wrap this client in executors where needed.
`transport` is injectable so tests drive the real client against an
in-process ASGI app without a socket.

The managed transfer functions compose two layers with a deliberate seam:
this module moves bytes over HTTP (begin/commit sessions, presigned or
proxied blob transfers), while everything that touches the commit format
(identities, inventories, classification, the pointer document, sidecar
trust, the verified restore) happens in the Rust core via
``lodedb._turbovec.cloud.managed_*``. Python never interprets a manifest: a
head body is parsed only as opaque JSON and re-serialised for the Rust core,
which recomputes every identity through the engine's own canonical writer
(key-order- and formatting-insensitive), so no digest ever depends on
Python's serialisation.
"""

from __future__ import annotations

import base64
import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

import httpx
from nacl.public import PrivateKey, SealedBox

from lodedb._turbovec import cloud as _core


class CloudError(RuntimeError):
    """A control-plane refusal, carrying the HTTP status and its detail.
    `retry_after` is the response's Retry-After header in seconds, the pause
    a retrying caller should honor; None when the server sent none."""

    def __init__(self, status_code: int, detail: str, *, retry_after: float | None = None):
        super().__init__(f"{detail} (HTTP {status_code})")
        self.status_code = status_code
        self.detail = detail
        self.retry_after = retry_after


def _store_hint(org: str, environment: str, store: object) -> str | None:
    """The `X-Ore-Store` value for a data-plane call, or None when the store
    isn't identifiable in the payload (the ingress then falls back to plain
    balancing, which is always correct; stickiness is cache locality only).
    Percent-encoded: store names are end-user identifiers, and a non-ASCII
    (or control-character) id must never turn a valid request into a header
    encoding error. Quoting is deterministic, so the same store always maps
    to the same hash bucket."""
    if not isinstance(store, str) or not store:
        return None
    return quote(f"{org}/{environment}/{store}", safe="/")


def _retry_after(response: httpx.Response) -> float | None:
    """Seconds from the response's Retry-After header, None when absent,
    negative, or not the plain-seconds form (the HTTP-date form is rare
    enough on API responses not to be worth parsing here)."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _raise_for(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise CloudError(response.status_code, str(detail), retry_after=_retry_after(response))


class CloudClient:
    def __init__(
        self,
        host: str,
        token: str | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 30.0,
    ):
        headers = {}
        if token:
            headers["authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            base_url=host.rstrip("/"), headers=headers, transport=transport, timeout=timeout
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> CloudClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _request(self, method: str, path: str, *, store_hint: str | None = None, **kwargs) -> Any:
        """One API call. `store_hint` stamps `X-Ore-Store` (org/env/store) on
        data-plane requests so a store-sticky ingress can hash-route them to
        the pod holding that store warm; the server never reads it, and any
        pod answers correctly without it. The header is a routing hint, not
        a contract."""
        if store_hint is not None:
            headers = dict(kwargs.pop("headers", None) or {})
            headers["x-ore-store"] = store_hint
            kwargs["headers"] = headers
        response = self._http.request(method, path, **kwargs)
        _raise_for(response)
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    # ------------------------------------------------------------ identity

    def me(self) -> dict:
        return self._request("GET", "/v1/auth/me")

    def token_self(self) -> dict:
        """The presented token's own identity: kind, scopes, and (for
        environment tokens) the org/environment slugs it is bound to
        (both None for personal tokens)."""
        return self._request("GET", "/v1/tokens/self")

    # ------------------------------------------------------------ tokens

    def list_tokens(self) -> list[dict]:
        return self._request("GET", "/v1/tokens")["tokens"]

    def mint_token(
        self,
        kind: str,
        scopes: list[str],
        *,
        name: str = "",
        org: str | None = None,
        environment: str | None = None,
        expires_in_days: int | None = None,
    ) -> dict:
        body: dict[str, Any] = {"kind": kind, "scopes": scopes, "name": name}
        if org:
            body["org"] = org
        if environment:
            body["environment"] = environment
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return self._request("POST", "/v1/tokens", json=body)

    def revoke_token(self, token_id: str) -> dict:
        return self._request("POST", f"/v1/tokens/{token_id}/revoke")

    # ------------------------------------------------------------ tenancy

    def list_environments(self, org: str) -> list[dict]:
        return self._request("GET", f"/v1/orgs/{org}/environments")["environments"]

    def list_stores(
        self,
        org: str,
        environment: str,
        *,
        after: str | None = None,
        limit: int = 50,
        store: str | None = None,
        prefix: str | None = None,
        order: str = "name",
        after_last_write: str | None = None,
    ) -> dict:
        """One keyset page of an environment's stores (a store is one end
        user): {stores, count}. `store` narrows to one exact name (key
        resolution, not paging); `prefix` to a name prefix. `order="name"`
        pages with `after` (the previous page's last name); `order="recent"`
        lists most-recently-written-first and pages with `after` AND
        `after_last_write` (the previous page's last row's name and
        ISO `last_write_at`) together."""
        params: dict[str, str] = {"limit": str(limit)}
        if after is not None:
            params["after"] = after
        if store is not None:
            params["store"] = store
        if prefix is not None:
            params["prefix"] = prefix
        if order != "name":
            params["order"] = order
        if after_last_write is not None:
            params["after_last_write"] = after_last_write
        return self._request(
            "GET", f"/v1/orgs/{org}/environments/{environment}/stores", params=params
        )

    def store_stats(self, org: str, environment: str) -> dict:
        """Fleet-level counts over an environment's live stores (store =
        one end user): {stores, active_24h, active_7d, new_7d}."""
        return self._request(
            "GET", f"/v1/orgs/{org}/environments/{environment}/stores/stats"
        )

    def create_store(
        self,
        org: str,
        environment: str,
        store: str,
        key: str | None = None,
        *,
        mode: str = "local_push",
        expose_text: bool = False,
        preset: str | None = None,
        vector_dim: int | None = None,
        encrypted: bool = False,
        sealed_material: str | None = None,
    ) -> dict:
        """Register a store, optionally carrying delegated sealed material."""

        body: dict[str, Any] = {"store": store, "mode": mode, "expose_text": expose_text}
        if key is not None:
            body["key"] = key
        if preset is not None:
            body["preset"] = preset
        if vector_dim is not None:
            body["vector_dim"] = vector_dim
        if encrypted:
            body["encrypted"] = True
        if sealed_material is not None:
            body["sealed_material"] = sealed_material
        return self._request(
            "POST", f"/v1/orgs/{org}/environments/{environment}/stores", json=body
        )

    def store_create_challenge(self, org: str, environment: str) -> dict:
        """Fetch the HPKE recipient used to create an encrypted store."""

        return self._request(
            "GET", f"/v1/orgs/{org}/environments/{environment}/stores/create-challenge"
        )

    def store_unseal_challenge(self, org: str, environment: str, store: str) -> dict:
        """Fetch a single-use HPKE challenge for unsealing or key rotation."""

        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/"
            f"{quote(store, safe='')}/unseal/challenge",
        )

    def unseal_store(self, org: str, environment: str, store: str, payload: dict) -> dict:
        """Submit a sealed material response and request a live unseal grant."""

        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/{quote(store, safe='')}/unseal",
            json=payload,
        )

    def reseal_store(self, org: str, environment: str, store: str) -> dict:
        """Remove a store's live unseal grant and evict its decrypted state."""

        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/{quote(store, safe='')}/reseal",
        )

    def rotate_store_key(self, org: str, environment: str, store: str, payload: dict) -> dict:
        """Re-wrap a currently live store key under freshly sealed material."""

        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/"
            f"{quote(store, safe='')}/key/rotate",
            json=payload,
        )

    # ------------------------------------------------------------ lifecycle

    def delete_org(self, org: str) -> dict:
        """Soft-delete an org. Returns the parked slug (restore takes it)
        and the purge deadline."""
        return self._request("DELETE", f"/v1/orgs/{org}")

    def restore_org(self, parked_slug: str) -> dict:
        return self._request("POST", f"/v1/orgs/{parked_slug}/restore")

    def delete_environment(self, org: str, environment: str) -> dict:
        return self._request("DELETE", f"/v1/orgs/{org}/environments/{environment}")

    def restore_environment(self, org: str, parked_slug: str) -> dict:
        return self._request("POST", f"/v1/orgs/{org}/environments/{parked_slug}/restore")

    def delete_store(
        self, org: str, environment: str, store: str, *, erase: bool = False
    ) -> dict:
        """Soft-delete a whole store (every index key in it hides with it).
        `erase=True` is data-subject erasure: the grace window is skipped,
        restore refuses from that moment, and the next lifecycle sweep
        hard-deletes the store's rows and objects.

        Store names are end-user ids (free-form up to '/'), so the path
        segment is percent-encoded. A raw `?` or `#` would truncate the URL
        and address a DIFFERENT store."""
        return self._request(
            "DELETE",
            f"/v1/orgs/{org}/environments/{environment}/stores/{quote(store, safe='')}"
            + ("?erase=true" if erase else ""),
        )

    def restore_store(self, org: str, environment: str, parked_store: str) -> dict:
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/"
            f"{quote(parked_store, safe='')}/restore",
        )

    def delete_store_key(self, org: str, environment: str, store: str, key: str) -> dict:
        """Soft-delete ONE index key inside a store (the advanced multi-key
        path)."""
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/delete",
            json={"store": store, "key": key},
        )

    def restore_store_key(self, org: str, environment: str, store: str, parked_key: str) -> dict:
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/restore",
            json={"store": store, "key": parked_key},
        )

    def list_trash(self, org: str) -> list[dict]:
        """Soft-deleted resources under a live org, restorable until their
        `purge_after`."""
        return self._request("GET", f"/v1/orgs/{org}/trash")["items"]

    def export_org(self, org: str) -> dict:
        """The offboarding manifest: every live environment and store with its
        head snapshot identity (metadata only; bytes move via pull)."""
        return self._request("GET", f"/v1/orgs/{org}/export")

    # ------------------------------------------------------------ transfer

    def store_head(self, org: str, environment: str, store: str, key: str) -> dict:
        return self._request(
            "GET",
            f"/v1/orgs/{org}/environments/{environment}/stores/head",
            params={"store": store, "key": key},
        )

    def pull_plan(self, org: str, environment: str, store: str, key: str) -> dict:
        return self._request(
            "GET",
            f"/v1/orgs/{org}/environments/{environment}/stores/pull-plan",
            params={"store": store, "key": key},
        )

    def store_history(self, org: str, environment: str, store: str, key: str) -> list[dict]:
        """The store's restore window: every retained snapshot, newest first."""
        return self._request(
            "GET",
            f"/v1/orgs/{org}/environments/{environment}/stores/head-history",
            params={"store": store, "key": key},
        )["snapshots"]

    def rollback_store(
        self, org: str, environment: str, store: str, snapshot_id: str, key: str | None = None
    ) -> dict:
        """Moves the branch head back to a retained snapshot (reversible;
        the displaced head stays in the window for the retention period)."""
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/rollback",
            json={"store": store, "key": key, "snapshot_id": snapshot_id},
        )

    def begin_push(self, org: str, environment: str, payload: dict) -> dict:
        return self._request(
            "POST", f"/v1/orgs/{org}/environments/{environment}/push/begin", json=payload
        )

    def commit_push(self, org: str, environment: str, session_id: str, payload: dict) -> dict:
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/push/{session_id}/commit",
            json=payload,
        )

    def heartbeat_push(self, org: str, environment: str, session_id: str) -> dict:
        return self._request(
            "POST", f"/v1/orgs/{org}/environments/{environment}/push/{session_id}/heartbeat"
        )

    def abort_push(self, org: str, environment: str, session_id: str) -> dict:
        return self._request(
            "POST", f"/v1/orgs/{org}/environments/{environment}/push/{session_id}/abort"
        )

    def update_store(
        self,
        org: str,
        environment: str,
        store: str,
        key: str,
        *,
        expose_text: bool | None = None,
        mode: str | None = None,
    ) -> dict:
        body: dict[str, Any] = {"store": store, "key": key}
        if expose_text is not None:
            body["expose_text"] = expose_text
        if mode is not None:
            body["mode"] = mode
        return self._request(
            "PATCH", f"/v1/orgs/{org}/environments/{environment}/stores", json=body
        )

    # ------------------------------------------------------------ serving

    def search(self, org: str, environment: str, payload: dict) -> dict:
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/search",
            json=payload,
            store_hint=_store_hint(org, environment, payload.get("store")),
        )

    def search_many(self, org: str, environment: str, payload: dict) -> dict:
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/search-many",
            json=payload,
            store_hint=_store_hint(org, environment, payload.get("store")),
        )

    def store_text(
        self, org: str, environment: str, store: str, id: str, key: str | None = None
    ) -> dict:
        params: dict[str, str] = {"store": store, "id": id}
        if key:
            params["key"] = key
        return self._request(
            "GET",
            f"/v1/orgs/{org}/environments/{environment}/stores/text",
            params=params,
            store_hint=_store_hint(org, environment, params.get("store")),
        )

    def add_documents(self, org: str, environment: str, payload: dict) -> dict:
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/documents",
            json=payload,
            store_hint=_store_hint(org, environment, payload.get("store")),
        )

    def remove_documents(self, org: str, environment: str, payload: dict) -> dict:
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/documents/remove",
            json=payload,
            store_hint=_store_hint(org, environment, payload.get("store")),
        )

    def browse_documents(self, org: str, environment: str, payload: dict) -> dict:
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/documents/browse",
            json=payload,
            store_hint=_store_hint(org, environment, payload.get("store")),
        )

    # --------------------------------------------------------- memory verbs

    def recall(self, org: str, environment: str, payload: dict) -> dict:
        """Non-exact retrieval from raw text (server-side sub-queries + RRF)."""
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/recall",
            json=payload,
            store_hint=_store_hint(org, environment, payload.get("store")),
        )

    def context_block(self, org: str, environment: str, payload: dict) -> dict:
        """A prompt-ready context block from one user's store."""
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/context-block",
            json=payload,
            store_hint=_store_hint(org, environment, payload.get("store")),
        )

    def delete_memories(self, org: str, environment: str, payload: dict) -> dict:
        """Delete a store's memories in place (async, remove segments).
        The store row stays; `delete_store` forgets the user entirely."""
        return self._request(
            "POST",
            f"/v1/orgs/{org}/environments/{environment}/stores/memories/delete",
            json=payload,
            store_hint=_store_hint(org, environment, payload.get("store")),
        )

    def write_status(
        self, org: str, environment: str, store: str, write_id: str, key: str | None = None
    ) -> dict:
        """One accepted write's lifecycle: registered → folded/condemned."""
        params: dict[str, str] = {"store": store}
        if key:
            params["key"] = key
        return self._request(
            "GET",
            f"/v1/orgs/{org}/environments/{environment}/stores/writes/{write_id}",
            params=params,
            store_hint=_store_hint(org, environment, store),
        )

    def serving_stats(
        self,
        org: str,
        environment: str,
        store: str,
        key: str | None = None,
        *,
        warm: bool = False,
    ) -> dict:
        params: dict[str, str] = {"store": store}
        if key:
            params["key"] = key
        if warm:
            params["warm"] = "true"
        # The hint matters MOST here: `warm=True` is the pre-hydration call,
        # and it must land on the same pod the hinted queries will.
        return self._request(
            "GET",
            f"/v1/orgs/{org}/environments/{environment}/stores/serving-stats",
            params=params,
            store_hint=_store_hint(org, environment, store),
        )

    def upload_blob_proxy(self, proxy_path: str, handle: BinaryIO) -> None:
        """Streams one blob through the control plane's authenticated proxy."""
        response = self._http.put(proxy_path, content=handle, timeout=None)
        _raise_for(response)

    def download_blob_proxy(self, proxy_path: str, dest: Path) -> None:
        with self._http.stream("GET", proxy_path, timeout=None) as response:
            if response.status_code >= 400:
                response.read()  # buffer the error body so _raise_for can parse it
            _raise_for(response)
            with open(dest, "wb") as out:
                for chunk in response.iter_bytes():
                    out.write(chunk)


# ---------------------------------------------------------------- login

@dataclass(frozen=True)
class LoginStart:
    session_id: str
    user_code: str
    verification_url: str
    poll_interval_seconds: int


class LoginHandoff:
    """One CLI login attempt: keypair, session, poll loop, unseal.

    The private key lives only in this object (memory, this process); the
    server ever sees the public half, and the minted token comes back as a
    sealed box only this keypair opens.
    """

    def __init__(self, client: CloudClient, client_label: str):
        self._client = client
        self._key = PrivateKey.generate()
        info = client._request(
            "POST",
            "/v1/cli/sessions",
            json={
                "public_key": bytes(self._key.public_key).hex(),
                "client_label": client_label,
            },
        )
        self.start = LoginStart(
            session_id=info["id"],
            user_code=info["user_code"],
            verification_url=info["verification_url"],
            poll_interval_seconds=info["poll_interval_seconds"],
        )

    def poll_once(self) -> tuple[str, str | None]:
        """One poll: (state, token). The token appears on the claiming poll."""
        result = self._client._request("GET", f"/v1/cli/sessions/{self.start.session_id}")
        sealed = result.get("sealed_token")
        if sealed is None:
            return result["state"], None
        token = SealedBox(self._key).decrypt(base64.b64decode(sealed)).decode("utf-8")
        return result["state"], token

    def wait(self, *, timeout_seconds: float = 900.0, sleep=time.sleep) -> str:
        """Polls until approved and returns the token; raises CloudError on
        denial/expiry/timeout."""
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            state, token = self.poll_once()
            if token is not None:
                return token
            if state in ("denied", "expired", "claimed"):
                raise CloudError(409, f"login {state}")
            sleep(self.start.poll_interval_seconds)
        raise CloudError(408, "login timed out waiting for approval")


# ---------------------------------------------------------------- managed

# Engine store kinds → the wire contract's blob kinds. `tvann` (persisted ANN
# clusters) and `tvvf` (the rescore original-vector sidecar) are vector-derived,
# payload-free like `tvim`. Deliberately no default: an engine kind this table
# does not know must fail loudly at `_wire_kind` rather than ship mislabelled.
# The Rust inventory fails closed on unknown sub-manifests for the same reason.
_ENGINE_KIND_TO_WIRE = {
    "json": "state",
    "tvim": "vector",
    "tvmv": "vector",
    "tvtext": "text",
    "tvlex": "lexical",
    "tvann": "vector",
    "tvvf": "vector",
}


def _wire_kind(engine_kind: str) -> str:
    try:
        return _ENGINE_KIND_TO_WIRE[engine_kind]
    except KeyError:
        raise CloudError(
            422,
            f"this client cannot classify engine store kind {engine_kind!r} for the "
            "wire contract — upgrade lodedb before pushing this generation",
        ) from None


class SyncConflictError(CloudError):
    """A managed sync refused to transfer, or lost a race with a concurrent
    writer (diverged/unknown lineage needing an explicit force, a commit-time
    pointer conflict, or a head that moved mid-sync). The single conflict
    surface: callers retry the sync or resolve with --force-push/--force-pull;
    every other CloudError is a genuine refusal."""


@dataclass(frozen=True)
class ManagedRemote:
    """A parsed `orecloud://org/environment[/store]` target."""

    org: str
    environment: str
    store: str = "memory"

    SCHEME = "orecloud://"

    @classmethod
    def parse(cls, target: str) -> ManagedRemote | None:
        """The parsed remote, or None when `target` is not an orecloud URL.
        A malformed orecloud URL raises (never silently falls through to the
        dumb-target path)."""
        if not target.startswith(cls.SCHEME):
            return None
        segments = target[len(cls.SCHEME) :].strip("/").split("/")
        if len(segments) not in (2, 3) or not all(segments):
            raise CloudError(
                422,
                f"malformed managed target {target!r}; expected "
                "orecloud://org/environment[/store]",
            )
        return cls(*segments)

    def identity(self, host: str) -> str:
        """The sidecar remote-identity string. Includes the control-plane
        host; the same org/environment on two deployments is two remotes."""
        return (
            f"{self.SCHEME}{self.org}/{self.environment}/{self.store}"
            f"#host={host.rstrip('/')}"
        )


def _plan(client: CloudClient, dir: str, key: str, remote: ManagedRemote,
          host: str, *, include_text: bool, include_lexical: bool) -> tuple[dict, dict]:
    """One head fetch + one Rust plan: (head response, plan dict)."""
    head = client.store_head(remote.org, remote.environment, remote.store, key)
    body = head.get("body")
    plan = _core.managed_plan(
        dir,
        key,
        remote.identity(host),
        json.dumps(body) if body is not None else None,
        include_text=include_text,
        include_lexical=include_lexical,
    )
    return head, plan


def _upload_blobs(client: CloudClient, dir: str, plan_local: dict, need_upload: list[dict]) -> int:
    """Moves the need-upload set: presigned PUT when offered (falling back to
    the proxy on transport failure), else the proxy. Returns bytes sent."""
    name_by_sha: dict[str, str] = {}
    for artifact in plan_local["artifacts"]:
        name_by_sha.setdefault(artifact["sha256"], artifact["name"])
    sent = 0
    for item in need_upload:
        name = name_by_sha.get(item["sha256"])
        if name is None:
            raise CloudError(
                409,
                f"server asked for blob {item['sha256']} this push never declared",
            )
        path = Path(dir) / name
        uploaded = False
        if item.get("put_url"):
            try:
                with open(path, "rb") as handle:
                    response = httpx.put(
                        item["put_url"],
                        content=handle,
                        headers=item.get("put_headers") or {},
                        timeout=None,
                    )
                response.raise_for_status()
                uploaded = True
            except httpx.HTTPError:
                uploaded = False  # unreachable endpoint or refusal: use the proxy
        if not uploaded:
            with open(path, "rb") as handle:
                client.upload_blob_proxy(item["proxy_path"], handle)
        sent += item["size_bytes"]
    return sent


def _push_with_plan(
    client: CloudClient,
    dir: str,
    key: str,
    remote: ManagedRemote,
    host: str,
    head: dict,
    plan: dict,
) -> dict:
    """The push protocol against an already-classified head: begin (CAS-armed
    with that head), upload, commit, record the sidecar base."""
    local = plan["local"]
    if local is None:
        raise CloudError(404, f"no committed generation to push for index key {key!r}")
    identity = remote.identity(host)
    expected_head = head["head"]["snapshot_id"] if head.get("head") else None

    if expected_head == local["snapshot_id"]:
        # The remote already holds exactly this snapshot: record the agreed
        # base and publish nothing.
        _core.managed_record_base(dir, key, identity, local["body_json"])
        return {
            "index_key": key,
            "generation": local["generation"],
            "artifacts_written": 0,
            "artifacts_skipped": len(local["artifacts"]),
            "bytes_written": 0,
            "pointer_published": False,
        }

    begin = client.begin_push(
        remote.org,
        remote.environment,
        {
            "store": remote.store,
            "key": key,
            "snapshot_id": local["snapshot_id"],
            "logical_id": local["logical_id"],
            "generation": local["generation"],
            "expected_head": expected_head,
            "artifacts": [
                {
                    "sha256": artifact["sha256"],
                    "size_bytes": artifact["size_bytes"],
                    "kind": _wire_kind(artifact["kind"]),
                }
                for artifact in local["artifacts"]
            ],
        },
    )
    try:
        bytes_written = _upload_blobs(client, dir, local, begin["need_upload"])
        base = plan.get("base")
        client.commit_push(
            remote.org,
            remote.environment,
            begin["session_id"],
            {
                "body": json.loads(local["body_json"]),
                "pointer_document": local["pointer_document"],
                "parent_snapshot_id": base["snapshot_id"] if base else None,
            },
        )
    except BaseException:
        # Best-effort: release the server-side session instead of leaving it
        # to expire on its own. The original failure is what matters. An
        # abort that itself fails (network already gone) must not mask it.
        try:
            client.abort_push(remote.org, remote.environment, begin["session_id"])
        except Exception:
            pass
        raise
    _core.managed_record_base(dir, key, identity, local["body_json"])
    return {
        "index_key": key,
        "generation": local["generation"],
        "artifacts_written": len(begin["need_upload"]),
        "artifacts_skipped": len(local["artifacts"]) - len(begin["need_upload"]),
        "bytes_written": bytes_written,
        "pointer_published": True,
    }


def managed_push(
    client: CloudClient,
    dir: str,
    remote: ManagedRemote,
    key: str,
    *,
    host: str,
    include_text: bool = False,
    include_lexical: bool = False,
) -> dict:
    """Publish the local committed generation, raced through the head CAS.
    The managed analogue of the dumb `push` verb (last writer wins; a
    concurrent advance surfaces as a 409, and divergence *protection* is
    `managed_sync`'s job)."""
    head, plan = _plan(
        client, dir, key, remote, host,
        include_text=include_text, include_lexical=include_lexical,
    )
    return _push_with_plan(client, dir, key, remote, host, head, plan)


def _pull_with_body(
    client: CloudClient,
    dir: str,
    key: str,
    remote: ManagedRemote,
    host: str,
    expected_snapshot_id: str | None,
    discard_pending_wal: bool = False,
    expected_local_snapshot_id: str | None = None,
) -> dict:
    """The pull protocol: plan, download the missing blob set into staging,
    then let the Rust core materialise + verify-open + record the sidecar.

    `expected_snapshot_id` pins the pull to a classified head: if the remote
    advanced between classification and the plan fetch, refuse (retry the
    sync) rather than restore a snapshot nothing classified."""
    plan_response = client.pull_plan(remote.org, remote.environment, remote.store, key)
    snapshot = plan_response["snapshot"]
    if expected_snapshot_id is not None and snapshot["snapshot_id"] != expected_snapshot_id:
        # Only sync pins the expected head, so this is always a sync-level
        # conflict (SyncConflictError subclasses CloudError; plain-pull
        # callers that catch CloudError are unaffected).
        raise SyncConflictError(
            409,
            "the remote head moved between classification and pull; re-run the sync",
        )
    body_json = json.dumps(plan_response["body"])
    needed = {
        artifact["sha256"]
        for artifact in _core.managed_pull_requirements(dir, key, body_json)
    }
    downloads = {
        blob["sha256"]: blob for blob in plan_response["blobs"] if blob["sha256"] in needed
    }
    missing = needed - set(downloads)
    if missing:
        raise CloudError(
            502, f"pull plan omits {len(missing)} blob(s) the body references"
        )

    with tempfile.TemporaryDirectory(prefix="orecloud-pull-") as staging:
        for sha, blob in sorted(downloads.items()):
            dest = Path(staging) / sha
            fetched = False
            if blob.get("get_url"):
                try:
                    with httpx.stream("GET", blob["get_url"], timeout=None) as response:
                        response.raise_for_status()
                        with open(dest, "wb") as out:
                            for chunk in response.iter_bytes():
                                out.write(chunk)
                    fetched = True
                except httpx.HTTPError:
                    fetched = False  # unreachable endpoint: use the proxy
            if not fetched:
                client.download_blob_proxy(blob["proxy_path"], dest)
        return _core.managed_materialize(
            dir,
            key,
            remote.identity(host),
            body_json,
            staging,
            discard_pending_wal=discard_pending_wal,
            expected_local_snapshot_id=expected_local_snapshot_id,
        )


def managed_pull(
    client: CloudClient,
    remote: ManagedRemote,
    dir: str,
    key: str,
    *,
    host: str,
) -> dict:
    """Restore the branch head into `dir` and verify it opens (the managed
    analogue of the dumb `pull` verb)."""
    return _pull_with_body(client, dir, key, remote, host, expected_snapshot_id=None)


def managed_status(
    client: CloudClient,
    dir: str,
    remote: ManagedRemote,
    key: str,
    *,
    host: str,
    include_text: bool = False,
    include_lexical: bool = False,
) -> dict:
    """The status report for a managed remote (same fields as the dumb
    `status` verb, lineage included)."""
    _head, plan = _plan(
        client, dir, key, remote, host,
        include_text=include_text, include_lexical=include_lexical,
    )
    return {
        field: value
        for field, value in plan.items()
        if field
        not in ("local", "remote", "base", "base_is_current", "local_raw_snapshot_id")
    }


def managed_sync(
    client: CloudClient,
    dir: str,
    remote: ManagedRemote,
    key: str,
    *,
    host: str,
    include_text: bool = False,
    include_lexical: bool = False,
    force_push: bool = False,
    force_pull: bool = False,
) -> dict:
    """Three-pointer sync against a managed remote: classify (local, sidecar
    base, branch head), then run at most one fast-forward. The decision
    table matches the Rust `sync` verb's; the head CAS closes the race
    window.
    """
    if force_push and force_pull:
        raise ValueError("force_push and force_pull are mutually exclusive")
    head, plan = _plan(
        client, dir, key, remote, host,
        include_text=include_text, include_lexical=include_lexical,
    )
    classification = plan["classification"]
    identity = remote.identity(host)

    def outcome(action: str, forced: bool, transfer: dict | None) -> dict:
        report = dict(transfer) if transfer else {"index_key": key}
        report["classification"] = classification
        report["action"] = action
        report["forced"] = forced
        report["sidecar_corrupt"] = plan["sidecar_corrupt"]
        return report

    def _push_translating_conflict() -> dict:
        """The commit-time CAS 409 (head moved since the push began) is a
        sync-level conflict, not a generic refusal. Re-running the sync
        re-classifies against the new head and usually resolves it."""
        try:
            return _push_with_plan(client, dir, key, remote, host, head, plan)
        except CloudError as error:
            if error.status_code == 409 and "pointer conflict" in error.detail:
                raise SyncConflictError(
                    409, f"{error.detail}; re-run the sync to reconcile"
                ) from error
            raise

    # The local state this classification saw, as the materialization pin:
    # a local commit landing after this point refuses instead of being
    # overwritten ("" pins to classified-as-absent).
    classified_local = plan.get("local_raw_snapshot_id") or ""

    def _refuse_pending_wal() -> None:
        """A pull-direction transfer must not run over a local WAL still
        holding acknowledged writes (replaying them onto the pulled lineage
        would corrupt it; dropping them silently loses acked data). Refusing
        HERE, before a single blob downloads, mirrors the Rust verbs; the
        materialize step re-checks authoritatively under the writer lock. The
        scan runs only on this pull branch, so push/status planning never
        pays for it."""
        ops = _core.local_wal_ops(dir, key)
        if ops:
            raise SyncConflictError(
                409,
                f"the local database holds {ops} uncheckpointed WAL operation(s); "
                "checkpoint them by opening the store once, or re-run with "
                "--force-pull to discard them along with the local lineage",
            )

    if force_push:
        return outcome("push", True, _push_translating_conflict())
    if force_pull:
        head_sha = head["head"]["snapshot_id"] if head.get("head") else None
        if head_sha is None:
            raise CloudError(404, f"no committed generation to pull for index key {key!r}")
        return outcome(
            "pull",
            True,
            _pull_with_body(
                client,
                dir,
                key,
                remote,
                host,
                head_sha,
                discard_pending_wal=True,
                expected_local_snapshot_id=classified_local,
            ),
        )

    if classification == "in_sync":
        # Mirror the Rust sync's stale-base repair: agreeing ends with a
        # missing/stale recorded base record the agreed state.
        if not plan["base_is_current"] and head.get("body") is not None:
            _core.managed_record_base(dir, key, identity, json.dumps(head["body"]))
        return outcome("none", False, None)
    if classification in ("local_ahead", "republish"):
        return outcome("push", False, _push_translating_conflict())
    if classification == "remote_ahead":
        head_sha = head["head"]["snapshot_id"] if head.get("head") else None
        if head_sha is None:
            raise CloudError(404, f"no committed generation to pull for index key {key!r}")
        _refuse_pending_wal()
        return outcome(
            "pull",
            False,
            _pull_with_body(
                client,
                dir,
                key,
                remote,
                host,
                head_sha,
                expected_local_snapshot_id=classified_local,
            ),
        )

    # diverged/unknown: refuse with the same wording as the Rust verb.
    hint = (
        "re-run with --force-push to keep the local copy or --force-pull to keep "
        "the remote copy"
    )
    if plan["sidecar_corrupt"]:
        hint += (
            " (note: the sync sidecar was present but corrupt and was ignored, so the "
            "recorded base could not be trusted)"
        )
    raise SyncConflictError(
        409, f"sync refused: local and remote are {classification}; {hint}"
    )
