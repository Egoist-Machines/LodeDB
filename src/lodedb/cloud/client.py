"""`lodedb.cloud.Client`: the developer-facing handle, bound to one tenancy.

    from lodedb.cloud import Client

    client = Client()                     # credentials + tenancy from the token
    store = client.store("user-42")       # one end user's LodeDB instance
    store.add("prefers email over phone")

The org and environment are properties of the *credential*, not of every
call site: `ore_sk_`/`ore_pk_` tokens are minted bound to one environment,
and the server rejects any other. So the client asks the control plane what
the token is (`GET /v1/tokens/self`) and binds itself accordingly. Code
that connects with a production key and code that connects with a testing
key are the same code. Personal tokens (`ore_pat_`) span environments, so
they resolve to the account's only org and default to the ``testing``
environment (production is the explicit go-live choice, never a guess);
passing ``org=``/``environment=`` always wins. Passing exactly one of the
pair is checked against a bound token rather than silently ignored; passing
both skips introspection entirely (zero HTTP calls) and relies on the
server rejecting a mismatched binding at the first request.

Credentials resolve like the CLI's: explicit ``token=``/``host=`` arguments,
then the ``ORECLOUD_TOKEN`` environment variable, then the ``lodedb cloud
login`` credentials file. The host defaults to the hosted control plane;
``host=`` / ``ORECLOUD_HOST`` override it for self-hosted deployments.

Every control-plane verb the SDK speaks hangs off this object with the
tenancy already applied; per-store reads and writes live on the
:class:`~lodedb.cloud.serving.CloudStore` handles that :meth:`Client.store`
returns. The legacy ``lodedb.cloud.connect("org/environment/store")`` remains
as sugar over this class.
"""

from __future__ import annotations

from typing import Any

import httpx

from lodedb.cloud import _config
from lodedb.cloud.transfer import CloudClient, CloudError, ManagedRemote


def resolve_credentials(token: str | None, host: str | None) -> tuple[str, str]:
    """(host, token) from explicit arguments, the environment, or the
    `lodedb cloud login` file. The host falls back to the hosted control
    plane (`DEFAULT_HOST`), so a bare token is always enough; only a missing
    token raises, with a CloudError naming the fix."""
    if token is None or host is None:
        stored = _config.load_credentials()
        if token is None:
            token = stored.token if stored else None
        if host is None:
            host = stored.host if stored else None
    if not token:
        raise CloudError(
            401,
            "no credential configured — pass token=, set ORECLOUD_TOKEN, or run "
            "`lodedb cloud login`",
        )
    return (host or _config.DEFAULT_HOST).rstrip("/"), token


def resolve_tenancy(
    client: CloudClient, org: str | None, environment: str | None
) -> tuple[str, str]:
    """The (org, environment) this credential addresses.

    Explicit values win. An environment-scoped token supplies its own
    binding and *refuses* a conflicting explicit value. Silently ignoring
    one would aim writes somewhere the caller didn't name. A personal token
    falls back to the account's only org, then to the org's only live
    environment or the seeded ``testing`` default; anything else raises
    with the actual choices.
    """
    if org and environment:
        return org, environment
    try:
        info = client.token_self()
    except CloudError as error:
        if error.status_code == 404:
            # A control plane too old to introspect tokens: don't guess what
            # the credential is; name both escape hatches.
            raise CloudError(
                404,
                "this control plane does not support token introspection — "
                "pass org= and environment= explicitly, or upgrade the server",
            ) from error
        raise
    if info["environment"]:
        bound_org, bound_environment = info["org"], info["environment"]
        if org and org != bound_org:
            raise CloudError(
                403, f"this token is bound to org {bound_org!r}, not {org!r}"
            )
        if environment and environment != bound_environment:
            raise CloudError(
                403,
                f"this token is bound to environment {bound_environment!r}, "
                f"not {environment!r}",
            )
        return bound_org, bound_environment
    if org is None:
        slugs = [row["slug"] for row in client.me()["orgs"]]
        if not slugs:
            raise CloudError(404, "this account belongs to no org")
        if len(slugs) > 1:
            raise CloudError(
                422,
                "this account belongs to several orgs — pass org= "
                f"(one of: {', '.join(slugs)})",
            )
        org = slugs[0]
    if environment is None:
        slugs = [row["slug"] for row in client.list_environments(org)]
        if len(slugs) == 1:
            environment = slugs[0]
        elif "testing" in slugs:
            # The fixed pair seeded at signup: testing is the default;
            # production is the explicit go-live choice, never a guess.
            environment = "testing"
        else:
            raise CloudError(
                422,
                f"org {org!r} has {len(slugs)} environments — pass environment= "
                + (f"(one of: {', '.join(slugs)})" if slugs else "(none live)"),
            )
    return org, environment


class Client:
    """One credential, one control plane, one bound (org, environment).

    Construction resolves credentials and tenancy (see the module
    docstring) and performs at most two HTTP calls, zero when both
    ``org=`` and ``environment=`` are passed. The client owns one HTTP
    connection pool that every handle it creates shares; close it with
    :meth:`close` or a ``with`` block when done.
    """

    def __init__(
        self,
        *,
        token: str | None = None,
        host: str | None = None,
        org: str | None = None,
        environment: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host, _token = resolve_credentials(token, host)
        self._client = CloudClient(self.host, _token, transport=transport, timeout=timeout)
        try:
            self.org, self.environment = resolve_tenancy(self._client, org, environment)
        except BaseException:
            # A half-constructed client must not leak its connection pool.
            self._client.close()
            raise

    # ------------------------------------------------------------- stores

    def store(
        self,
        store: str,
        *,
        key: str | None = None,
        warm: bool = False,
        read_your_writes: bool = True,
    ):
        """A read/write handle over one store. A store is one end user's
        LodeDB instance, auto-provisioned by its first write, so this
        makes no HTTP call by default. `warm=True` additionally asks the serving
        tier to hydrate it now (first query skips the cold start). `key`
        names the index key when the store holds more than one (rare).
        The handle shares this client's connection pool; closing the
        handle does not close the client."""
        from lodedb.cloud.serving import CloudStore

        handle = CloudStore(
            self._client,
            self.org,
            self.environment,
            store,
            key,
            read_your_writes=read_your_writes,
            owns_client=False,
        )
        if warm:
            handle.stats(warm=True)
        return handle

    def list_stores(self, **params: Any) -> dict:
        """One keyset page of this environment's stores (a store is one end
        user): {stores, count}. Paging/filter params are `CloudClient
        .list_stores`'s."""
        return self._client.list_stores(self.org, self.environment, **params)

    def store_stats(self) -> dict:
        """Fleet-level counts over this environment's live stores:
        {stores, active_24h, active_7d, new_7d}."""
        return self._client.store_stats(self.org, self.environment)

    def create_store(self, store: str, key: str | None = None, **options: Any) -> dict:
        """Register a store explicitly (first writes auto-provision, so this
        is for choosing `mode=`/`preset=`/`expose_text=` up front, or
        `vector_dim=` for a bring-your-own-vectors store, which accepts
        `add_vectors`/`search_by_vector` and never embeds server-side)."""
        return self._client.create_store(self.org, self.environment, store, key, **options)

    def update_store(self, store: str, key: str, **changes: Any) -> dict:
        """Flip a store's `expose_text`/`mode` flags."""
        return self._client.update_store(self.org, self.environment, store, key, **changes)

    def delete_store(self, store: str, *, erase: bool = False) -> dict:
        """Soft-delete a whole store and forget this end user (restorable
        for the grace period; the entitlement slot frees immediately).
        `erase=True` is data-subject erasure: no grace window, restore
        refuses immediately, and the next lifecycle sweep hard-deletes."""
        return self._client.delete_store(self.org, self.environment, store, erase=erase)

    def restore_store(self, parked_store: str) -> dict:
        return self._client.restore_store(self.org, self.environment, parked_store)

    def delete_store_key(self, store: str, key: str) -> dict:
        """Soft-delete ONE index key inside a store (the advanced multi-key
        path)."""
        return self._client.delete_store_key(self.org, self.environment, store, key)

    def restore_store_key(self, store: str, parked_key: str) -> dict:
        return self._client.restore_store_key(self.org, self.environment, store, parked_key)

    def store_history(self, store: str, key: str) -> list[dict]:
        """The store's restore window: every retained snapshot, newest
        first."""
        return self._client.store_history(self.org, self.environment, store, key)

    def rollback_store(self, store: str, snapshot_id: str, key: str | None = None) -> dict:
        """Move the store's head back to a retained snapshot (reversible
        within the retention window)."""
        return self._client.rollback_store(self.org, self.environment, store, snapshot_id, key=key)

    # ------------------------------------------------------- environments

    def list_environments(self) -> list[dict]:
        """The bound org's environments (an environment token sees only its
        own). The pair is fixed: production and testing, seeded at signup;
        there is no creation surface."""
        return self._client.list_environments(self.org)

    def delete_environment(self, slug: str | None = None) -> dict:
        """Soft-delete an environment (the bound one when `slug` is
        omitted) and everything in it."""
        return self._client.delete_environment(self.org, slug or self.environment)

    def restore_environment(self, parked_slug: str) -> dict:
        return self._client.restore_environment(self.org, parked_slug)

    # ---------------------------------------------------------------- org

    def list_trash(self) -> list[dict]:
        """Soft-deleted resources under the org, restorable until their
        `purge_after`."""
        return self._client.list_trash(self.org)

    def export_org(self) -> dict:
        """The offboarding manifest: every live environment and store with
        its head snapshot identity (metadata only)."""
        return self._client.export_org(self.org)

    def delete_org(self) -> dict:
        """Soft-delete the whole org. Returns the parked slug restore takes
        and the purge deadline."""
        return self._client.delete_org(self.org)

    def restore_org(self, parked_slug: str) -> dict:
        return self._client.restore_org(parked_slug)

    # ------------------------------------------------------------- tokens

    def me(self) -> dict:
        """The signed-in account (personal tokens only; an environment
        token is not a person and gets a 403)."""
        return self._client.me()

    def token_self(self) -> dict:
        """What the presented credential is: kind, scopes, and its
        org/environment binding (None/None for personal tokens)."""
        return self._client.token_self()

    def list_tokens(self) -> list[dict]:
        return self._client.list_tokens()

    def mint_token(self, kind: str, scopes: list[str], **options: Any) -> dict:
        """Mint a token; environment tokens (`secret`/`publishable`)
        default to this client's bound org/environment unless overridden."""
        if kind != "personal":
            options.setdefault("org", self.org)
            options.setdefault("environment", self.environment)
        return self._client.mint_token(kind, scopes, **options)

    def revoke_token(self, token_id: str) -> dict:
        return self._client.revoke_token(token_id)

    # ------------------------------------------------------------ transfer

    def remote(self, store: str = "memory") -> ManagedRemote:
        """This tenancy as a `ManagedRemote`, for the module-level transfer
        verbs (`managed_push`/`managed_pull`/`managed_sync`) that move a
        local LodeDB directory to and from the cloud."""
        return ManagedRemote(self.org, self.environment, store)

    # ----------------------------------------------------------- lifecycle

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Client({self.org!r}, {self.environment!r}, host={self.host!r})"
