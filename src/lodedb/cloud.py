"""The ``lodedb.cloud`` import root: a lazy proxy over the optional OreCloud
managed-cloud companion (the ``lodedb[cloud]`` extra).

``from lodedb.cloud import Client`` (or ``connect``, ``CloudStore``, …)
forwards to the ``orecloud`` package so lodedb is the single documented
import root; ``from orecloud import Client`` keeps working unchanged. The
companion is imported only when a name is actually fetched — importing this
module (or ``lodedb`` itself) never loads it, guarded by
``tests/test_import_boundary.py``. Without the extra installed, fetching any
name raises an ``ImportError`` carrying the install hint.

This module also hosts :func:`open_cloud_target`, the one funnel behind both
constructor front doors: ``LodeDB.cloud("user-42")`` (the human-facing form)
and the ``LodeDB("orecloud://…")`` config-string dispatch.
"""

from __future__ import annotations

import inspect
from typing import Any

# Managed-cloud target scheme recognized by the LodeDB(...) config-string
# dispatch. LodeDB.cloud() additionally accepts scheme-less short forms
# ("user-42", "org/environment/store"); the plain constructor must not,
# because a bare store id is indistinguishable from a relative local path.
CLOUD_TARGET_SCHEME = "orecloud://"

_INSTALL_HINT = 'the cloud companion is not installed — run: pip install "lodedb[cloud]"'


def open_cloud_target(target: str, options: dict[str, Any]) -> Any:
    """Opens a managed-cloud target via the companion's ``connect`` and
    returns its store handle.

    The companion is imported lazily, only here, so a plain ``import
    lodedb`` never loads it; when it is absent this raises an
    ``ImportError`` carrying the install hint instead of a bare traceback.
    ``options`` must be keywords of ``orecloud.connect`` (``token``,
    ``host``, ...) — local-only construction options (``model=``,
    ``read_only=``, ...) are rejected up front, because embedding and
    storage for a cloud store are configured server-side, not per handle.
    """
    try:
        from orecloud import connect
    except ImportError:
        raise ImportError(
            f"{target!r} is a managed-cloud target, but {_INSTALL_HINT}"
        ) from None
    # Derive the accepted option set from connect's own signature so the two
    # never drift; anything else is a local-only option that has no cloud
    # meaning and deserves a targeted error, not a confusing TypeError from
    # deep inside the companion.
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
    """PEP 562: forwards public names (``Client``, ``connect``, …) to the
    lazily imported companion. Underscore names stay local so attribute
    probes (``__path__``, pickling machinery) never trigger the import."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import orecloud
    except ImportError:
        raise ImportError(f"lodedb.cloud.{name} is unavailable — {_INSTALL_HINT}") from None
    return getattr(orecloud, name)
