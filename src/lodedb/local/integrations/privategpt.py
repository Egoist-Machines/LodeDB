"""PrivateGPT vector-store provider for LodeDB.

PrivateGPT (zylon-ai/private-gpt) does not own a vector-store interface of its own: its store
layer is built on **LlamaIndex's** ``BasePydanticVectorStore``, selected at startup by the
``vectorstore.database`` key in ``settings.yaml``. Because LodeDB already ships a LlamaIndex
``BasePydanticVectorStore`` (:class:`lodedb.local.integrations.llama_index.LodeDBVectorStore`,
the ``lodedb[llama-index]`` extra), wiring PrivateGPT up is **not a new adapter**; it is a thin
provider shim plus one line of registration.

PrivateGPT's selection mechanism (``private_gpt/components/vector_store/``) is a small factory
registry:

- ``VectorStoreFactory`` is an ABC with ``vector_store(collection) -> BasePydanticVectorStore``,
  constructed as ``provider(settings, embed_dim)``.
- ``register_vector_store(database, provider)`` adds a provider to a process-local ``_PROVIDERS``
  dict, and ``VectorStoreComponent`` resolves ``settings.vectorstore.database`` against it. The
  ``database`` field is a free-form ``str`` (no ``Literal`` allow-list), so ``database: lodedb``
  is accepted as soon as the provider is registered.

This module supplies that provider:

- :class:`LodeDBVectorStoreFactory`: a ``VectorStoreFactory`` that reads PrivateGPT's settings
  (the embedding ``embed_dim``, the requested ``collection``, and an optional ``lodedb:`` block)
  and builds a :class:`LodeDBVectorStore`, one local on-disk index per collection.
- :func:`register_lodedb_provider`: registers the factory under ``"lodedb"`` so PrivateGPT can
  select it. Call it once before PrivateGPT builds its ``VectorStoreComponent``.

**Text-path, like the LlamaIndex adapter.** LodeDB embeds text internally with the model picked
in the ``lodedb:`` settings block, so PrivateGPT's own embedding model is bypassed for storage
and query (``is_embedding_query=False``); ``vectorstore.embed_dim`` is therefore informational
here. Keep PrivateGPT pointed at a cheap/mock embedding (it still computes vectors that LodeDB
discards) to avoid a redundant remote embedding call, exactly as documented for the LlamaIndex
``VectorStoreIndex`` path.

**What lives where.** Everything LodeDB-specific (this shim, the settings keys it reads, the
registration call) lives here. The only thing that must happen inside PrivateGPT's own process
is *triggering* the registration (``_PROVIDERS`` is process-local with no entry-point
auto-discovery), which is a one-line import (or a two-line custom launcher); see
``examples/privategpt_provider.py`` and ``docs/integrations.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from lodedb.local.db import LodeDB
from lodedb.local.integrations.llama_index import LodeDBVectorStore

if TYPE_CHECKING:  # pragma: no cover - typing only; PrivateGPT is an app, not a runtime dep
    from llama_index.core.vector_stores.types import BasePydanticVectorStore


def _load_private_gpt_factory_base() -> tuple[type, Any]:
    """Imports PrivateGPT's ``VectorStoreFactory`` ABC and ``register_vector_store``.

    Raises a clear, actionable error if PrivateGPT is not importable. PrivateGPT is an
    application rather than a published library, so it is intentionally **not** a LodeDB
    dependency; this shim runs inside an existing PrivateGPT checkout/virtualenv.
    """

    try:
        from private_gpt.components.vector_store.factory import (  # type: ignore[import-not-found]
            VectorStoreFactory,
            register_vector_store,
        )
    except ImportError as exc:  # pragma: no cover - exercised via stub in tests
        raise ImportError(
            "the LodeDB PrivateGPT provider must run inside a PrivateGPT environment "
            "(zylon-ai/private-gpt) that exposes "
            "private_gpt.components.vector_store.factory; it could not be imported. "
            "Install LodeDB's LlamaIndex extra into PrivateGPT's environment "
            "(pip install 'lodedb[llama-index]') and run this from there. "
            "If you are on a PrivateGPT version without the vector-store factory registry, "
            "wire LodeDBVectorStore in by hand instead (see docs/integrations.md)."
        ) from exc
    return VectorStoreFactory, register_vector_store


def _build_factory_class() -> type:
    """Builds the ``LodeDBVectorStoreFactory`` class bound to PrivateGPT's ABC.

    The base class lives in PrivateGPT (an app, not a dependency), so the subclass is created at
    call time once PrivateGPT is importable, keeping this module import-clean without it.
    """

    base, _ = _load_private_gpt_factory_base()

    class LodeDBVectorStoreFactory(base):  # type: ignore[valid-type, misc]
        """PrivateGPT ``VectorStoreFactory`` that yields a LodeDB-backed store per collection.

        One :class:`LodeDBVectorStore` is created per ``collection``, each its own on-disk LodeDB
        index under the configured ``path`` (``<path>/<collection>``), so PrivateGPT's
        collection-based multitenancy maps to separate local indexes. Construction options come
        from an optional ``lodedb:`` block in ``settings.yaml`` (see :func:`_lodedb_settings`):
        ``path`` (default ``local_data/lodedb``), ``model`` (default ``minilm``), ``device``
        (default ``auto``), ``store_text`` (default ``True``; keep it on for hybrid/lexical), and
        ``index_text`` (default follows ``store_text``, so ``True`` unless ``store_text`` is off).
        """

        def __init__(self, settings: Any, embed_dim: int | None = None) -> None:
            super().__init__(settings, embed_dim)
            self._stores: dict[str, LodeDBVectorStore] = {}
            opts = _lodedb_settings(settings)
            self._root = Path(opts["path"]).expanduser()
            self._model = str(opts["model"])
            self._device = str(opts["device"])
            self._store_text = bool(opts["store_text"])
            # None means "follow store_text" (LodeDB resolves it); preserve it.
            self._index_text = None if opts["index_text"] is None else bool(opts["index_text"])

        def vector_store(self, collection: str) -> BasePydanticVectorStore:
            """Returns (caching per collection) a LodeDB-backed store for ``collection``."""

            store = self._stores.get(collection)
            if store is None:
                db = LodeDB(
                    path=self._root / collection,
                    model=self._model,
                    device=self._device,
                    store_text=self._store_text,
                    index_text=self._index_text,
                )
                store = LodeDBVectorStore(db)
                self._stores[collection] = store
            return store

        def close(self) -> None:
            """Closes every open per-collection LodeDB handle (called by PrivateGPT teardown)."""

            for store in self._stores.values():
                store.client.close()
            self._stores.clear()

    LodeDBVectorStoreFactory.__qualname__ = "LodeDBVectorStoreFactory"
    return LodeDBVectorStoreFactory


# Default LodeDB construction options, overridable from a ``lodedb:`` block in settings.yaml.
_DEFAULT_LODEDB_OPTIONS: dict[str, Any] = {
    "path": "local_data/lodedb",
    "model": "minilm",
    "device": "auto",
    "store_text": True,
    "index_text": None,
}


def _lodedb_settings(settings: Any) -> dict[str, Any]:
    """Reads the optional ``lodedb`` block from PrivateGPT settings, filling defaults.

    Accepts either a Pydantic-style ``settings`` object with a ``lodedb`` attribute or a plain
    mapping; an absent block (or absent keys) falls back to :data:`_DEFAULT_LODEDB_OPTIONS`.
    Recognized keys: ``path``, ``model``, ``device``, ``store_text``, ``index_text``.
    """

    raw: Any = None
    if settings is not None:
        raw = getattr(settings, "lodedb", None)
        if raw is None and isinstance(settings, dict):
            raw = settings.get("lodedb")

    block: dict[str, Any] = {}
    if raw is not None:
        if hasattr(raw, "model_dump"):  # pydantic v2 settings model
            block = dict(raw.model_dump())
        elif isinstance(raw, dict):
            block = dict(raw)
        else:  # arbitrary settings object: pull the known fields off attributes
            block = {key: getattr(raw, key) for key in _DEFAULT_LODEDB_OPTIONS if hasattr(raw, key)}

    options = dict(_DEFAULT_LODEDB_OPTIONS)
    for key in _DEFAULT_LODEDB_OPTIONS:
        value = block.get(key)
        if value is not None:
            options[key] = value
    return options


def register_lodedb_provider(database: str = "lodedb") -> type:
    """Registers the LodeDB provider with PrivateGPT's vector-store factory registry.

    Call this once inside a PrivateGPT process before its ``VectorStoreComponent`` is built
    (PrivateGPT resolves ``settings.vectorstore.database`` against ``_PROVIDERS`` at startup).
    After registering, set ``vectorstore.database: lodedb`` in ``settings.yaml`` (or
    ``PGPT_VECTORSTORE=lodedb``) to select it. Returns the registered factory class.

    Idempotent: registering the same ``database`` name again simply rebinds the provider.
    """

    factory_cls = _build_factory_class()
    _, register_vector_store = _load_private_gpt_factory_base()
    register_vector_store(str(database), factory_cls)
    return factory_cls


__all__ = ["register_lodedb_provider"]
