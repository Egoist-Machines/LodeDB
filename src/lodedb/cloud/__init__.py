"""``lodedb.cloud`` is the OreCloud managed-cloud client (the ``[cloud]`` extra).

First-party code, shipped in the lodedb wheel: :class:`Client` (one
credential, one bound org/environment, per-user store handles),
:func:`connect` / :class:`CloudStore` (one end user's store over HTTPS, the
handle ``LodeDB.cloud()`` returns), the transfer verbs (``push`` / ``pull`` /
``sync`` / ``status`` / ``verify`` / ``keys``) over the native transfer core
bundled as ``lodedb._turbovec.cloud``, and the ``lodedb cloud`` CLI. The
``[cloud]`` extra installs only the client's third-party dependencies
(``httpx``, ``pynacl``); there is no separate cloud distribution.

Every name resolves lazily (PEP 562): importing this module (or ``lodedb``
itself) touches neither httpx nor the network, guarded by
``tests/test_import_boundary.py``. Fetching an HTTP-backed name without the
extra's dependencies installed raises an ``ImportError`` carrying the
install hint; the transfer verbs ride the always-present native extension.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

# The hosted control plane every credential path defaults to (stdlib-only
# module, so this eager import keeps the plain-import boundary intact);
# --host / ORECLOUD_HOST / host= override it.
from lodedb.cloud._config import DEFAULT_HOST as DEFAULT_HOST

# Managed-cloud target scheme recognized by the LodeDB(...) config-string
# dispatch. LodeDB.cloud() additionally accepts scheme-less short forms
# ("user-42", "org/environment/store"); the plain constructor must not,
# because a bare store id is indistinguishable from a relative local path.
CLOUD_TARGET_SCHEME = "orecloud://"

_INSTALL_HINT = (
    'the cloud client dependencies are not installed — run: pip install "lodedb[cloud]"'
)

# The six local↔remote transfer verbs live on the bundled native extension
# (no third-party dependency), so they resolve even without the extra.
_NATIVE_EXPORTS = frozenset({"keys", "pull", "push", "status", "sync", "verify"})

# HTTP-backed exports and the submodule each lives in. These modules import
# httpx (and, for login, pynacl) at module level, so they are reached only
# through this lazy table, never on a plain import.
_HTTP_EXPORTS = {
    "Client": "client",
    "connect": "serving",
    "CloudStore": "serving",
    "CloudIndex": "serving",  # back-compat alias for CloudStore
    "CloudSearchHit": "serving",
    "CloudClient": "transfer",
    "CloudError": "transfer",
    "SyncConflictError": "transfer",
}

__all__ = sorted(
    {"CLOUD_TARGET_SCHEME", "DEFAULT_HOST", "open_cloud_target", *_NATIVE_EXPORTS, *_HTTP_EXPORTS}
)


def open_cloud_target(target: str, options: dict[str, Any]) -> Any:
    """Opens a managed-cloud target via :func:`connect` and returns its store
    handle, the one funnel behind both constructor front doors,
    ``LodeDB.cloud("user-42")`` and the ``LodeDB("orecloud://…")``
    config-string dispatch.

    The client is imported lazily, only here, so a plain ``import lodedb``
    stays network-free; without the ``[cloud]`` extra's dependencies this
    raises an ``ImportError`` carrying the install hint instead of a bare
    traceback. ``options`` must be keywords of :func:`connect` (``token``,
    ``host``, ...). Local-only construction options (``model=``,
    ``read_only=``, ...) are rejected up front, because embedding and
    storage for a cloud store are configured server-side, not per handle.
    """
    try:
        from lodedb.cloud.serving import connect
    except ImportError:
        raise ImportError(
            f"{target!r} is a managed-cloud target, but {_INSTALL_HINT}"
        ) from None
    # Derive the accepted option set from connect's own signature so the two
    # never drift; anything else is a local-only option that has no cloud
    # meaning and deserves a targeted error, not a confusing TypeError from
    # deep inside the client.
    allowed = set(inspect.signature(connect).parameters) - {"target"}
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise TypeError(
            f"cloud targets do not accept {', '.join(unknown)}; available "
            f"options: {', '.join(sorted(allowed))} (embedding and storage "
            "are configured server-side, per store)"
        )
    return connect(target, **options)


def __getattr__(name: str) -> Any:
    """PEP 562: resolves the public surface lazily. Transfer verbs come from
    the bundled native extension, HTTP-backed names from their submodule (with
    the install hint when the extra's dependencies are absent)."""
    if name in _NATIVE_EXPORTS:
        from lodedb import _turbovec

        return getattr(_turbovec.cloud, name)
    submodule = _HTTP_EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"lodedb.cloud.{submodule}")
    except ImportError:
        raise ImportError(f"lodedb.cloud.{name} is unavailable — {_INSTALL_HINT}") from None
    return getattr(module, name)
