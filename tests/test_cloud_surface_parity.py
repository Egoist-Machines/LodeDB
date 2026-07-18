"""Executable audit of the LodeDB-surface contract: `CloudStore` duck-types
the local `lodedb.LodeDB` verb surface (lodedb's `LodeDB("orecloud://…")`
front door returns one), so every shared verb must keep a matching call
shape. Two kinds of drift are allowed and named here: cloud-only keywords
are additive (multi-tenant provenance, TTL, text exposure, serving warm-up),
and a few local keywords have no cloud meaning (the server always
unit-normalizes vectors; there is no server-side filtered count yet).
Anything outside those allowances fails this suite before it ships."""

from __future__ import annotations

import inspect

import pytest

from lodedb.cloud.serving import CloudStore

LodeDB = pytest.importorskip("lodedb").LodeDB

# Every verb the two handles share. `get_text` is the documented synonym of
# `get` on both sides; `remove`/`remove_many` are async-first on the cloud
# (they return a write id, not a bool/count) but take the same arguments.
SHARED_VERBS = [
    "add",
    "add_many",
    "add_vectors",
    "add_vectors_many",
    "search",
    "search_many",
    "search_by_vector",
    "search_many_by_vector",
    "get",
    "get_text",
    "get_texts",
    "remove",
    "remove_many",
    "count",
    "stats",
    "close",
]

# Keyword-only params only the cloud has: write provenance and retention
# (`ttl_seconds`/`agent_id`/`run_id`), inline text on hits (`include_text`),
# and serving warm-up (`warm`). All additive — code written against the
# shared shape runs unchanged.
CLOUD_ONLY_KEYWORDS = {"ttl_seconds", "agent_id", "run_id", "include_text", "warm"}

# Keyword-only params only the local handle has, per verb. `normalize` does
# not exist on the cloud because the server unit-normalizes every vector
# (the local default); `count(filter=)` needs a server-side filtered count
# the API does not offer yet.
LOCAL_ONLY_KEYWORDS = {
    "add_vectors": {"normalize"},
    "add_vectors_many": {"normalize"},
    "search_by_vector": {"normalize"},
    "search_many_by_vector": {"normalize"},
    "count": {"filter"},
}


def _shape(method) -> tuple[list[str], set[str]]:
    """A method's call shape: ordered positional-or-keyword names (sans self)
    and the set of keyword-only names."""
    parameters = inspect.signature(method).parameters.values()
    positional = [
        parameter.name
        for parameter in parameters
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        and parameter.name != "self"
    ]
    keyword_only = {
        parameter.name
        for parameter in parameters
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    return positional, keyword_only


@pytest.mark.parametrize("verb", SHARED_VERBS)
def test_shared_verb_keeps_the_local_call_shape(verb):
    """CloudStore's version of each shared verb accepts the local call shape:
    identical positional arguments, and keyword drift only within the named
    cloud-only / local-only allowances."""
    local_positional, local_keywords = _shape(getattr(LodeDB, verb))
    cloud_positional, cloud_keywords = _shape(getattr(CloudStore, verb))

    assert cloud_positional == local_positional, (
        f"{verb}: positional shape drifted — local {local_positional}, "
        f"cloud {cloud_positional}"
    )
    unexplained_cloud = cloud_keywords - local_keywords - CLOUD_ONLY_KEYWORDS
    assert not unexplained_cloud, (
        f"{verb}: cloud grew keywords outside the additive allowance: "
        f"{sorted(unexplained_cloud)}"
    )
    unexplained_local = (
        local_keywords - cloud_keywords - LOCAL_ONLY_KEYWORDS.get(verb, set())
    )
    assert not unexplained_local, (
        f"{verb}: cloud is missing local keywords with no recorded reason: "
        f"{sorted(unexplained_local)}"
    )
