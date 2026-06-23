"""LodeDB — the local-first, embedded, no-auth vector database SDK.

``LodeDB`` runs the engine in-process in a local profile that:

- binds to loopback only (never a network address);
- requires no authentication — the user never supplies or sees any credential;
- keeps telemetry metrics-only (counts / bytes / latency, never payloads);
- stores vectors in the compact TurboVec format and commits every change
  atomically via a per-index root manifest (``<key>.commit.json``) over
  generation-addressed artifacts under ``<key>.gen/`` (``.json``/``.jsd``
  state, ``.tvim``/``.tvd`` vectors, and the opt-in ``.tvtext``/``.txd`` raw
  text), so a crash rolls back to the last committed generation and readers see
  a consistent snapshot.

On CUDA hosts, eligible batched queries can use the optional GPU-resident
TurboVec scan; otherwise the compact CPU kernel is the source of truth and
fallback. Embedding can run on MPS/CUDA/CPU depending on the requested device.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durability_from_env, normalize_durability
from lodedb.engine._predicate import coerce_sdk_filter
from lodedb.engine.core import (
    EngineDocument,
    EngineSecurityConfig,
    LodeEngine,
)
from lodedb.engine.index import EngineError, LodeIndex
from lodedb.engine.route_registry import default_route_registry, load_route_registry
from lodedb.local.backends import (
    LocalEmbeddingResolution,
    build_local_embedding_backend,
)
from lodedb.local.presets import LocalModelPreset, resolve_preset

# Fixed local identifier for the single-process client context. It never leaves
# the process and has no security meaning — the local engine is auth-free.
_LOCAL_CLIENT_ID = "lodedb-local"
# The local DB is a single, stable index. Pinning it to the engine's
# DEFAULT_INDEX_ID makes the persisted snapshot key the bare client hash (the
# legacy default snapshot name), so reopening the same path binds to the same
# on-disk state instead of minting a fresh random index id each open.
_LOCAL_INDEX_ID = "default"
_LOCAL_BIND_HOST = "127.0.0.1"
# Retrieval modes accepted by ``search``/``search_many``. ``vector`` is the
# default; ``hybrid`` (BM25 + RRF) and ``lexical`` (BM25 only) need a lexical
# source: a persisted postings store (``index_text=True``) or retained raw text
# (``store_text=True``) the BM25 index is rebuilt from.
_LEXICAL_SEARCH_MODES = frozenset({"hybrid", "lexical"})
_SEARCH_MODES = frozenset({"vector"}) | _LEXICAL_SEARCH_MODES


class LodeSearchHit:
    """One redacted local search result: ``(score, id, metadata)``.

    Supports tuple-style unpacking (``score, id, metadata``) to match the
    documented ``search`` return shape, plus attribute access.
    """

    __slots__ = ("score", "id", "metadata")

    def __init__(self, *, score: float, id: str, metadata: dict[str, Any]) -> None:
        """Stores the score, document id, and redacted metadata for one hit."""

        self.score = float(score)
        self.id = str(id)
        self.metadata = dict(metadata)

    def __iter__(self):
        """Yields ``(score, id, metadata)`` so hits unpack like the spec tuple."""

        yield self.score
        yield self.id
        yield self.metadata

    def __repr__(self) -> str:
        """Returns a compact, payload-free representation of the hit."""

        return f"LodeSearchHit(score={self.score:.4f}, id={self.id!r}, metadata={self.metadata!r})"

    def __eq__(self, other: object) -> bool:
        """Compares hits structurally (and to plain ``(score, id, metadata)`` tuples)."""

        if isinstance(other, LodeSearchHit):
            return (self.score, self.id, self.metadata) == (other.score, other.id, other.metadata)
        if isinstance(other, tuple) and len(other) == 3:
            return (self.score, self.id, self.metadata) == other
        return NotImplemented


class ReadOnlyError(RuntimeError):
    """Raised when a mutating call is made on a ``read_only=True`` LodeDB handle."""


class LodeDB:
    """Embedded, local-first vector database. Data stays on your machine.

    Example::

        db = LodeDB(path="./data", model="minilm", device="auto")
        doc_id = db.add("the quick brown fox", metadata={"topic": "animals"})
        for score, hit_id, meta in db.search("fox", k=5):
            ...
        db.persist()

    All documents, embeddings, and queries stay local: nothing is sent to a
    server, and no auth is required. Each change commits atomically via the
    engine's per-index root manifest, so a crash rolls back to the last
    committed generation and reopening the same path replays it safely.

    Raw document text is retained by default in a dedicated on-disk store (a
    ``.tvtext`` base + ``.txd`` delta journal, committed O(changed) per write),
    so the original text is retrievable by id (``get``/``get_text``/``get_texts``).
    Telemetry, audit, and the redacted ``.json``/``.jsd`` snapshot stay
    payload-free regardless; pass ``store_text=False`` to opt out of retaining
    text entirely::

        db = LodeDB(path="./data")            # store_text=True by default
        fox = db.add("the quick brown fox")
        db.get(fox)                   # -> "the quick brown fox"
        db.get_texts([fox])           # -> {fox: "the quick brown fox"}
    """

    def __init__(
        self,
        path: str | Path,
        *,
        model: str = "minilm",
        device: str = "auto",
        batch_size: int = 32,
        max_seq_length: int | None = None,
        chunk_character_limit: int = 900,
        store_text: bool = True,
        index_text: bool = False,
        read_only: bool = False,
        durability: str | None = None,
        route_registry_path: str | Path | None = None,
        _embedding_backend: Any | None = None,
    ) -> None:
        """Opens (or creates) an on-disk local index, loading any persisted state.

        ``model`` is a preset (``"minilm"`` fast default, ``"bge"`` quality).
        ``device`` is ``"auto"``/``"cpu"``/``"mps"``/``"cuda"`` (embedding only).
        ``read_only=True`` opens a non-mutating snapshot handle that takes **no**
        writer lock, so it can read a path while another process holds the
        single-writer lock (e.g. query while ``lodedb serve`` runs); mutating
        calls raise :class:`ReadOnlyError` and the path must already exist. See
        :meth:`open_readonly`.
        ``durability`` is ``"fast"`` (default: atomic but not power-loss durable)
        or ``"fsync"`` (fsync each file + its directory on commit, trading commit
        throughput for power-loss durability). Unset reads ``LODEDB_DURABILITY``.
        ``store_text`` controls durable raw-text retention and defaults to
        ``True``: the original text passed to ``add``/``add_many`` is kept in a
        dedicated on-disk sidecar so ``get``/``get_text``/``get_texts`` can return
        it (across reopens too). Pass ``store_text=False`` to opt out of retaining
        text at all — telemetry, audit, and the redacted snapshot never carry text
        regardless of this flag. Reopen the same path with the same ``store_text``
        value you wrote with.
        ``index_text`` controls durable lexical-index persistence and defaults to
        ``False``: when ``True``, the per-chunk tokens of each added document are
        kept in a dedicated ``.tvlex`` sidecar (base + ``.lxd`` journal, committed
        O(changed) per write), so ``mode="hybrid"``/``"lexical"`` survive a reopen
        without rebuilding from raw text and without requiring ``store_text=True``.
        The sidecar holds payload-derived terms only and, like the raw-text
        sidecar, never reaches telemetry, audit, or the redacted snapshot. The
        default leaves the on-disk layout byte-for-byte unchanged. Reopen the same
        path with the same ``index_text`` value you wrote with.
        ``_embedding_backend`` is an internal hook for tests/fixtures.
        """

        self.path = Path(path)
        self.store_text = bool(store_text)
        self.index_text = bool(index_text)
        self.read_only = bool(read_only)
        # "fast" (atomic rename only) vs "fsync" (power-loss durable). An
        # explicit arg wins; otherwise LODEDB_DURABILITY, else fast.
        fsync_on_commit = (
            durability_from_env() if durability is None else normalize_durability(durability)
        )
        self.preset: LocalModelPreset = resolve_preset(model)
        seq_len = int(max_seq_length) if max_seq_length is not None else 256

        if _embedding_backend is not None:
            backend = _embedding_backend
            self.embedding_resolution = LocalEmbeddingResolution(
                requested_device=device,
                backend_name=getattr(backend, "name", "injected"),
                effective_device="injected",
                fallback_used=False,
                fallback_reason="",
            )
        else:
            backend, self.embedding_resolution = build_local_embedding_backend(
                self.preset,
                device=device,
                batch_size=batch_size,
                max_seq_length=seq_len,
            )
        self._embedding_backend = backend

        if self.read_only:
            # A read-only handle reads an existing store; it never creates one,
            # so a missing path is a clear error rather than a silent empty DB.
            if not self.path.is_dir():
                raise FileNotFoundError(
                    f"LodeDB(read_only=True) requires an existing directory: {self.path}"
                )
        else:
            self.path.mkdir(parents=True, exist_ok=True)
        security = EngineSecurityConfig(
            bind_host=_LOCAL_BIND_HOST,
            route_profile=self.preset.route_profile,
            telemetry_mode="metrics_only",
            allow_raw_result_text=self.store_text,
            persist_lexical_index=self.index_text,
        )
        self._engine = LodeEngine(
            security=security,
            route_registry=(
                load_route_registry(str(route_registry_path))
                if route_registry_path is not None
                else default_route_registry()
            ),
            chunk_character_limit=int(chunk_character_limit),
            persistence_dir=self.path,
            read_only=self.read_only,
            fsync_on_commit=fsync_on_commit,
            embedding_backend=backend,
            route_policy=self.preset.route_policy,
        )
        self._index = LodeIndex(
            self._engine,
            client_id=_LOCAL_CLIENT_ID,
            index_id=_LOCAL_INDEX_ID,
        )
        # Ensure the (single) local index exists; load-on-open already restored
        # any persisted state for this client hash via the engine constructor.
        self._ensure_index()
        self._auto_id_counter = 0

    # -- public API ---------------------------------------------------------

    @classmethod
    def open_readonly(cls, path: str | Path, **kwargs: Any) -> LodeDB:
        """Opens an existing store as a non-mutating, lock-free reader.

        Sugar for ``LodeDB(path, read_only=True, **kwargs)``: it takes no writer
        lock (so it can read a path while a writer holds it), serves
        ``search``/``get``/``stats``, and raises :class:`ReadOnlyError` on any
        mutating call. The path must already exist.
        """

        return cls(path, read_only=True, **kwargs)

    def add(
        self,
        text: str,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Adds (or replaces) one document and returns its id.

        A missing ``id`` is auto-generated. Reusing an id upserts that document.
        The engine commits this mutation atomically before returning, so a crash
        after ``add`` rolls back cleanly to the last committed state on reopen.
        """

        self._require_writable()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        document_id = str(id) if id is not None else self._next_auto_id()
        document = EngineDocument(
            document_id=document_id,
            text=text,
            metadata=_coerce_metadata(metadata),
        )
        self._index.upsert_batch((document,))
        return document_id

    def add_many(
        self,
        documents: list[Mapping[str, Any]],
    ) -> list[str]:
        """Adds a batch of ``{"text", "id"?, "metadata"?}`` docs; returns the ids.

        Batched embedding is more efficient than repeated ``add`` calls.
        """

        self._require_writable()
        payload: list[EngineDocument] = []
        ids: list[str] = []
        for item in documents:
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("each document needs a non-empty 'text'")
            document_id = str(item["id"]) if item.get("id") is not None else self._next_auto_id()
            ids.append(document_id)
            payload.append(
                EngineDocument(
                    document_id=document_id,
                    text=text,
                    metadata=_coerce_metadata(item.get("metadata")),
                )
            )
        if payload:
            self._index.upsert_batch(tuple(payload))
        return ids

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        mode: str = "vector",
    ) -> list[LodeSearchHit]:
        """Returns the top-``k`` hits as ``(score, id, metadata)``-style rows.

        ``mode`` selects the retrieval strategy and defaults to ``"vector"``
        (pure vector search, behavior unchanged):

        - ``"vector"`` — embedding cosine similarity only.
        - ``"hybrid"`` — runs a lexical BM25 ranker alongside the vector scan and
          fuses the two ranked lists with Reciprocal Rank Fusion, so exact tokens
          that the embedding misses (error codes like ``E1234``, serials like
          ``ABC-123``, dates like ``2024-01-15``) are surfaced when they appear in
          the document body. Recommended for local RAG, where a missed exact match
          is the difference between a usable and a useless answer.
        - ``"lexical"`` — the BM25 ranking alone (no vector scan).

        ``"hybrid"`` and ``"lexical"`` build an in-memory BM25 index from a
        lexical source, so they require opening LodeDB with either
        ``index_text=True`` (a durable postings store that survives reopens
        without raw text) or ``store_text=True`` (the index rebuilt from the
        retained raw text, the default); requesting them with neither raises
        :class:`ValueError`. The serving index lives in memory, is maintained
        incrementally across mutations (a small change folds in only the changed
        chunks), and never changes the on-disk format.

        ``filter`` narrows results by metadata and is pushed into the TurboVec
        allowlist by the engine, so ``k`` still returns the true top-``k`` of the
        matching subset (not a post-filtered slice of an unfiltered top-``k``); in
        ``"hybrid"``/``"lexical"`` the same allowlist constrains both rankers. It
        accepts either a flat ``{field: value}`` exact-match map or a Mongo-style
        predicate:

        - comparison ``$eq`` ``$ne`` ``$gt`` ``$gte`` ``$lt`` ``$lte`` ``$in``
          ``$nin`` ``$exists`` — e.g. ``{"year": {"$gte": 2020, "$lt": 2025}}``;
        - composition ``$and`` / ``$or`` / ``$not`` (nestable) — e.g.
          ``{"$or": [{"topic": "ml"}, {"year": {"$gte": 2023}}]}``.

        A bare scalar is exact-match sugar for ``$eq``, so existing filters are
        unchanged. Metadata is stored as strings, so the ordered operators
        (``$gt``/``$gte``/``$lt``/``$lte``) compare numerically only when both the
        stored value and the operand parse as finite numbers (otherwise
        lexicographically), whereas ``$eq``/``$ne``/``$in``/``$nin`` always compare
        as strings — so ``{"price": {"$eq": 9.9}}`` does not match a stored ``9.90``
        while ``{"price": {"$gte": 9.9, "$lte": 9.9}}`` does. ``$ne``/``$nin`` and
        ``$exists: False`` also match documents missing the field. Raw query text
        never leaves the process and never appears in telemetry.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        resolved_mode = self._resolve_mode(mode)
        response = self._index.query(
            query,
            top_k=int(k),
            filter=_normalize_filter(filter),
            mode=resolved_mode,
        )
        return self._hits_from_result_rows(response.get("results", []))

    def search_many(
        self,
        queries: list[str],
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        mode: str = "vector",
    ) -> list[list[LodeSearchHit]]:
        """Returns top-``k`` hits for each query, preserving query order.

        Batched search is the public SDK path that lets CUDA hosts use the
        optional GPU-resident TurboVec scan for eligible query batches. Single
        queries and unavailable GPU dependencies fall back to the compact CPU
        kernel; raw query text still never appears in telemetry. ``filter`` takes
        the same exact-match-or-predicate grammar as :meth:`search` and is applied
        identically to every query in the batch.

        ``mode`` matches :meth:`search` (``"vector"`` default, ``"hybrid"``,
        ``"lexical"``) and applies to every query in the batch;
        ``search_many(mode="hybrid")`` returns the same result as the
        corresponding repeated single :meth:`search` call. ``"hybrid"`` and
        ``"lexical"`` require a lexical source (``index_text=True`` or
        ``store_text=True``). A batch of hybrid queries batches its vector half on
        the shared scan (the GPU serves it where available) and fuses each query's
        BM25 ranking on the CPU; lexical queries run BM25 on the CPU.
        """

        if not isinstance(queries, list) or not queries:
            raise ValueError("queries must be a non-empty list of strings")
        for query in queries:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("each query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        resolved_mode = self._resolve_mode(mode)
        normalized_filter = _normalize_filter(filter)
        batches = self._index.query_batch(
            [
                {
                    "query": query,
                    "top_k": int(k),
                    "filter": normalized_filter,
                    "mode": resolved_mode,
                }
                for query in queries
            ]
        ).get("queries", [])
        if not isinstance(batches, list):
            raise RuntimeError("invalid engine response: queries must be a list")
        out: list[list[LodeSearchHit]] = []
        for item in batches:
            if not isinstance(item, Mapping):
                raise RuntimeError("invalid engine response: query item must be an object")
            out.append(self._hits_from_result_rows(item.get("results", [])))
        return out

    def remove(self, id: str) -> bool:
        """Removes one document by id. Returns True if a document was deleted."""

        self._require_writable()
        if not isinstance(id, str) or not id.strip():
            raise ValueError("id must be a non-empty string")
        # `delete_documents` reports `document_count` as the number of unique
        # ids *requested* (not necessarily existing); `deleted_chunks` counts
        # chunks actually removed, so a positive value means the doc existed.
        response = self._index.delete_batch((id,))
        return int(response.get("deleted_chunks", 0) or 0) > 0

    def get(self, id: str) -> str | None:
        """Returns the stored raw text for a document id, or ``None`` if absent.

        This is the primary retrieval verb; :meth:`get_text` is a synonym. As a
        deliberate counterpart to ``add``, ``get(hit.id)`` is how an application
        recovers the original text for a document a search selected — search hits
        themselves stay payload-free. Available unless the DB was opened with
        ``store_text=False`` (see :meth:`get_text` for the exact semantics).
        """

        return self.get_text(id)

    def get_text(self, id: str) -> str | None:
        """Returns the stored raw text for a document id, or ``None`` if absent.

        Text retention is on by default; if the DB was opened with
        ``store_text=False`` this raises :class:`ValueError` so opting out is
        explicit. When text is retained, an unknown id (or one whose text was not
        stored) returns ``None`` rather than raising, so callers can probe ids
        ergonomically.
        Retrieved text is returned only to the caller and never logged or sent to
        telemetry.
        """

        if not isinstance(id, str) or not id.strip():
            raise ValueError("id must be a non-empty string")
        if not self.store_text:
            raise ValueError("raw text retrieval requires opening LodeDB with store_text=True")
        try:
            return self._index.get_document_text(id)
        except EngineError as exc:
            if exc.status_code == 404:
                return None
            raise

    def get_texts(self, ids: list[str]) -> dict[str, str]:
        """Returns a ``{id: text}`` map for the stored ids that have text.

        Requires ``store_text=True``. Ids that are unknown or whose text was not
        stored are omitted from the returned mapping, so the batch never fails
        because of one missing id.
        """

        if not isinstance(ids, list):
            raise ValueError("ids must be a list of strings")
        for value in ids:
            if not isinstance(value, str) or not value.strip():
                raise ValueError("each id must be a non-empty string")
        if not self.store_text:
            raise ValueError("raw text retrieval requires opening LodeDB with store_text=True")
        if not ids:
            return {}
        return self._index.get_document_texts(ids)

    def persist(self) -> dict[str, Any]:
        """Flushes durable on-disk state and returns redacted storage stats.

        The engine already persists on every mutation; this is an explicit
        durability + stats checkpoint. State is reloaded automatically on the
        next ``LodeDB(path=...)`` open.
        """

        # A no-op upsert path is avoided; instead surface the engine's current
        # redacted stats, which include persisted byte accounting.
        return self._index.stats()

    def count(self) -> int:
        """Returns the number of documents currently stored."""

        return int(self._index.stats().get("document_count", 0) or 0)

    def stats(self) -> dict[str, Any]:
        """Returns redacted engine stats (counts, storage bytes, telemetry)."""

        return self._index.stats()

    def close(self) -> None:
        """Releases the single-writer lock and engine references; state stays on disk."""

        if self._engine is not None:
            self._engine.close()
        self._index = None  # type: ignore[assignment]
        self._engine = None  # type: ignore[assignment]

    def __enter__(self) -> LodeDB:
        """Enters a context manager; state is already loaded on open."""

        return self

    def __exit__(self, *exc: object) -> None:
        """Exits the context manager (state is durable on disk already)."""

        self.close()

    # -- internals ----------------------------------------------------------

    def _require_writable(self) -> None:
        """Raises :class:`ReadOnlyError` if this handle was opened read-only."""

        if self.read_only:
            raise ReadOnlyError(
                "this LodeDB handle is read-only; open without read_only=True to modify it"
            )

    def _resolve_mode(self, mode: str) -> str:
        """Validates a search mode and enforces the lexical-source requirement.

        Returns the canonical lowercase mode. Raises :class:`ValueError` for an
        unknown mode, or when a lexical/hybrid mode is requested on a handle that
        has no lexical source — neither ``index_text=True`` (a persisted BM25
        postings store) nor ``store_text=True`` (the BM25 index rebuilt from the
        retained raw text), so there is nothing to build the index from.
        """

        if not isinstance(mode, str):
            raise ValueError("mode must be a string")
        value = mode.strip().lower() or "vector"
        if value not in _SEARCH_MODES:
            allowed = ", ".join(sorted(_SEARCH_MODES))
            raise ValueError(f"mode must be one of: {allowed}")
        if value in _LEXICAL_SEARCH_MODES and not (self.index_text or self.store_text):
            raise ValueError(
                f"mode={value!r} requires a lexical source: open LodeDB with "
                "index_text=True (persists the BM25 postings, durable across reopens) "
                "or store_text=True (the BM25 index is rebuilt from the retained raw text)"
            )
        return value

    def _ensure_index(self) -> None:
        """Binds the single local index, creating it unless this handle is read-only."""

        try:
            self._index.get_index()
        except Exception:  # noqa: BLE001 - missing index
            if self.read_only:
                # A read-only handle never creates state: an absent index just
                # means an empty or not-yet-written store, so reads return nothing.
                return
            self._index.create(name="lodedb-local")

    def _next_auto_id(self) -> str:
        """Returns a unique, collision-resistant auto id for an added document."""

        self._auto_id_counter += 1
        return f"doc-{secrets.token_hex(8)}-{self._auto_id_counter}"

    def _metadata_for_document(self, document_id: str) -> dict[str, Any]:
        """Returns redacted metadata for a document id (empty when unavailable)."""

        try:
            record = self._index.get_document(document_id)
        except Exception:  # noqa: BLE001 - tolerate races / missing metadata
            return {}
        metadata = record.get("metadata", {})
        return dict(metadata) if isinstance(metadata, Mapping) else {}

    def _hits_from_result_rows(self, rows: Any) -> list[LodeSearchHit]:
        """Hydrates engine result rows into public payload-free hit objects."""

        if not isinstance(rows, list):
            raise RuntimeError("invalid engine response: results must be a list")
        hits: list[LodeSearchHit] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("invalid engine response: result row must be an object")
            document_id = str(row["document_id"])
            hits.append(
                LodeSearchHit(
                    score=float(row["score"]),
                    id=document_id,
                    metadata=self._metadata_for_document(document_id),
                )
            )
        return hits


def _normalize_filter(filter: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Normalizes the ergonomic local filter into the engine filter schema.

    Accepts either the structured engine form
    (``{"metadata": {...}, "document_ids": [...]}``) or a flat metadata dict
    (``{"topic": "physics"}``), which is wrapped as ``{"metadata": {...}}``.
    Metadata values are stringified to match the engine's string metadata
    model so a flat ``{"year": 2020}`` filter matches a stored ``"2020"``.
    The metadata may use predicate operators (``$eq``/``$ne``/``$gt``/``$gte``/
    ``$lt``/``$lte``/``$in``/``$nin``/``$exists`` and ``$and``/``$or``/``$not``);
    a bare scalar stays exact-match. See ``lodedb.engine._predicate``.
    """

    if filter is None:
        return None
    if not isinstance(filter, Mapping):
        raise ValueError("filter must be a mapping")
    structured_keys = {"metadata", "document_ids"}
    if set(filter) <= structured_keys and any(key in filter for key in structured_keys):
        out: dict[str, Any] = {}
        if "metadata" in filter:
            out["metadata"] = coerce_sdk_filter(filter["metadata"])
        if "document_ids" in filter:
            out["document_ids"] = list(filter["document_ids"])
        return out
    # Flat metadata form (also where a top-level $and/$or/$not lands).
    return {"metadata": coerce_sdk_filter(filter)}


def _coerce_metadata(metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """Coerces metadata values to strings to match the engine's metadata model.

    The engine validates metadata as a string->string map; we stringify
    scalars so a local caller can pass ints/bools/floats ergonomically.
    """

    if metadata is None:
        return {}
    coerced: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValueError("metadata keys must be strings")
        if isinstance(value, bool):
            coerced[key] = "true" if value else "false"
        elif isinstance(value, (str, int, float)):
            coerced[key] = str(value)
        else:
            raise ValueError(
                f"metadata value for {key!r} must be a string, number, or bool"
            )
    return coerced
