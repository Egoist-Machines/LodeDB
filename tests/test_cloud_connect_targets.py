"""`connect()` target forms: the full `org/environment/store` triple (and its
`orecloud://` spelling) parses locally with zero HTTP, while a bare store id
(`user-42`, the form behind `LodeDB.cloud("user-42")`) resolves its
org/environment from the credential via token introspection, exactly like
`Client().store()`. HTTP is a `httpx.MockTransport`, so these run serverless;
the introspection contract itself is covered in `server/tests`."""

from __future__ import annotations

import pytest

# Collection must skip, not error, without the [cloud] extra installed
# (httpx and the modules below are the extra's dependencies).
pytest.importorskip("httpx", reason="needs the [cloud] extra's dependencies")
pytest.importorskip("nacl", reason="needs the [cloud] extra's dependencies")

import httpx  # noqa: E402

from lodedb.cloud.serving import _BareStore, _parse_target, connect  # noqa: E402
from lodedb.cloud.transfer import CloudError  # noqa: E402


def _transport(handled: dict[str, dict]) -> httpx.MockTransport:
    """Answers each path in `handled` with its JSON; anything else fails the
    test loudly (a target that should parse locally must make no HTTP call)."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = handled.get(request.url.path)
        assert body is not None, f"unexpected HTTP call: {request.url.path}"
        return httpx.Response(200, json=body)

    return httpx.MockTransport(handler)


# ------------------------------------------------------------ parsing only


def test_triple_and_url_forms_parse_locally():
    for target in ("acme/prod/user-42", "orecloud://acme/prod/user-42"):
        remote = _parse_target(target)
        assert (remote.org, remote.environment, remote.store) == ("acme", "prod", "user-42")


def test_two_segment_url_defaults_the_store():
    remote = _parse_target("orecloud://acme/prod")
    assert (remote.org, remote.environment, remote.store) == ("acme", "prod", "memory")


def test_single_segment_forms_are_bare_stores():
    for target in ("user-42", "orecloud://user-42"):
        assert _parse_target(target) == _BareStore("user-42")


def test_two_segment_bare_form_stays_malformed():
    """Without the scheme, two segments are ambiguous (org/environment vs
    environment/store) and refuse with the accepted forms named."""
    with pytest.raises(CloudError, match="bare store id"):
        _parse_target("prod/user-42")


# --------------------------------------------------------------- connect()


def test_bare_store_resolves_tenancy_from_the_credential():
    """`connect("user-42")` introspects the token once and lands on the bound
    org/environment, the seam `LodeDB.cloud("user-42")` rides."""
    transport = _transport(
        {
            "/v1/tokens/self": {
                "kind": "secret",
                "scopes": ["read:search", "write"],
                "org": "acme",
                "environment": "prod",
            }
        }
    )
    store = connect(
        "user-42", token="sk-test", host="https://api.test", warm=False, transport=transport
    )
    try:
        assert (store.org, store.environment, store.store) == ("acme", "prod", "user-42")
    finally:
        store.close()


def test_full_triple_makes_no_http_call():
    """A fully qualified target must not introspect. The transport rejects
    every request, so connecting cold (warm=False) proves zero HTTP."""
    store = connect(
        "acme/prod/user-42",
        token="sk-test",
        host="https://api.test",
        warm=False,
        transport=_transport({}),
    )
    try:
        assert (store.org, store.environment, store.store) == ("acme", "prod", "user-42")
    finally:
        store.close()


def test_bare_store_with_an_unbound_multi_org_token_refuses_with_choices():
    """A personal token spanning several orgs cannot pin a bare store id:
    connect surfaces resolve_tenancy's 422 naming the choices (and must not
    leak its freshly opened pool; closing after is a no-op either way)."""
    transport = _transport(
        {
            "/v1/tokens/self": {"kind": "personal", "scopes": [], "org": None, "environment": None},
            "/v1/auth/me": {"orgs": [{"slug": "acme"}, {"slug": "umbrella"}]},
        }
    )
    with pytest.raises(CloudError, match="several orgs"):
        connect(
            "user-42", token="pat", host="https://api.test", warm=False, transport=transport
        )


def test_empty_segments_are_malformed_not_reinterpreted():
    """`orecloud://org//store` must refuse: silently dropping the empty
    segment would reparse it as the two-segment org/environment form and aim
    the handle at the wrong tenancy."""
    for target in ("orecloud://acme//user-42", "acme//user-42"):
        with pytest.raises(CloudError):
            _parse_target(target)
