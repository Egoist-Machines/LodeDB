"""Payload-free reporting helpers for the migration toolkit.

Every artifact the migrator emits (the plan, the ``migration.json`` manifest, any
log line) must keep LodeDB's privacy boundary: counts, bytes, timings, ids or id
hashes, dimensions, versions, and warnings only — never raw documents, queries,
vectors, embeddings, payloads, or credentials. The helpers here are the single
place that knows how to redact a connection string and how to fingerprint ids and
source locations without leaking their contents.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from typing import Any

# Connection-string shapes we redact in plans/manifests/logs. We never store a raw
# DSN; we store its scheme + a short hash so two runs against the same source are
# comparable without the credential or host ever touching disk.
_URL_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.\-]*)://")
# Hosts that are safe to connect to without an explicit remote override.
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})


def hash_id(value: str) -> str:
    """Returns a short, stable, non-reversible fingerprint of one id.

    Used wherever a report needs to reference a specific row (a skipped id, a
    sampled id) without writing the id itself. The hash is salted with a fixed
    domain tag so it cannot be confused with any other id hash in the codebase.
    """

    digest = hashlib.sha256(("lodedb-migrate-id\x00" + str(value)).encode("utf-8")).hexdigest()
    return digest[:16]


def hash_ids(values: Iterable[str]) -> list[str]:
    """Returns short id fingerprints for a collection of ids, order preserved."""

    return [hash_id(value) for value in values]


def fingerprint_text(value: str) -> str:
    """Returns a short content fingerprint of an opaque blob (e.g. a file path).

    The fingerprint lets a manifest record *which* source it read from (so a
    resume/validate can confirm it is the same source) without recording the
    path or its contents verbatim.
    """

    digest = hashlib.sha256(("lodedb-migrate-loc\x00" + str(value)).encode("utf-8")).hexdigest()
    return digest[:16]


def redact_connection_string(value: str) -> str:
    """Returns a credential-free, log-safe rendering of a connection string.

    A DSN such as ``postgresql://user:pw@db.example.com:5432/app`` becomes
    ``postgresql://<redacted>`` — enough to show the provider scheme without ever
    surfacing the username, password, host, port, or database name. A bare path or
    name (no ``scheme://``) is treated as non-secret and returned unchanged.
    """

    match = _URL_RE.match(value)
    if match is None:
        return value
    return f"{match.group('scheme')}://<redacted>"


def connection_host(value: str) -> str | None:
    """Extracts the host from a URL-style connection string, if present.

    Returns ``None`` for a non-URL value (a path/name) and for a URL with no host.
    Used only to decide local vs remote; the host itself is never written to a
    report.
    """

    match = _URL_RE.match(value)
    if match is None:
        return None
    rest = value[match.end() :]
    # Drop any path/query so we look only at the authority section.
    authority = re.split(r"[/?#]", rest, maxsplit=1)[0]
    # Strip credentials: everything up to and including the last '@'.
    if "@" in authority:
        authority = authority.rsplit("@", 1)[1]
    # An IPv6 literal is bracketed: [::1]:5432.
    if authority.startswith("["):
        end = authority.find("]")
        if end != -1:
            return authority[1:end]
    host = authority.split(":", 1)[0]
    return host or None


def is_local_source(value: str) -> bool:
    """Returns True when a source is local (a filesystem path or a loopback host).

    A non-URL value is always local (a path/collection name on this machine). A
    URL is local only when its host is loopback. This is the gate the direct
    pgvector path uses before connecting without ``--allow-remote-source``.
    """

    host = connection_host(value)
    if host is None:
        return True
    return host.lower() in _LOCAL_HOSTS


def assert_payload_free(report: Any, *, where: str = "report") -> None:
    """Best-effort guard that no obvious raw-text/credential key leaked into a report.

    Walks a JSON-ish structure and raises :class:`ValueError` if it finds a key
    that should never appear in a payload-free artifact (``text``, ``vector``,
    ``password``, …). It is a defensive backstop, not a substitute for building
    reports out of redacted fields in the first place; tests assert on it so a
    future field addition that leaks is caught.
    """

    banned_keys = {
        "text",
        "texts",
        "page_content",
        "content",
        "document",
        "documents",
        "vector",
        "vectors",
        "embedding",
        "embeddings",
        "payload",
        "payloads",
        "query",
        "queries",
        "password",
        "secret",
        "token",
        "api_key",
        "connection_string",
        "dsn",
    }

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if isinstance(key, str) and key.lower() in banned_keys:
                    raise ValueError(
                        f"{where} leaked a non-payload-free key {key!r} at {path or '<root>'}"
                    )
                _walk(child, f"{path}.{key}" if path else str(key))
        elif isinstance(node, (list, tuple)):
            for index, child in enumerate(node):
                _walk(child, f"{path}[{index}]")

    _walk(report, "")


def sample_indices(count: int, *, limit: int) -> list[int]:
    """Returns up to ``limit`` evenly spread indices over ``range(count)``.

    Used to pick a representative sample of rows to validate (ids/metadata/text
    parity) without materializing the whole corpus or biasing toward the head.
    """

    if count <= 0 or limit <= 0:
        return []
    if count <= limit:
        return list(range(count))
    step = count / float(limit)
    seen: list[int] = []
    used: set[int] = set()
    for k in range(limit):
        index = int(k * step)
        if index >= count:
            index = count - 1
        if index not in used:
            used.add(index)
            seen.append(index)
    return seen


def take_sample(values: Sequence[str], *, limit: int) -> list[str]:
    """Returns a representative sample of values using :func:`sample_indices`."""

    return [values[i] for i in sample_indices(len(values), limit=limit)]
