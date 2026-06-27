"""LodeDB — the local-first, embedded, no-auth vector database SDK.

``LodeDB`` runs the engine in-process in a local profile that:

- binds the embedded SDK to loopback, while `lodedb serve` may intentionally bind a
  private-network address for trusted-LAN demos;
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

import math
import os
import secrets
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lodedb.engine._atomic_io import durability_from_env, normalize_durability
from lodedb.engine._predicate import coerce_sdk_filter
from lodedb.engine.core import (
    EngineDocument,
    EngineSecurityConfig,
    EngineVectorDocument,
    LodeEngine,
    index_state_key_for_client_hash,
    sha256_text,
)
from lodedb.engine.embedding_backends import EngineEmbeddingBackend
from lodedb.engine.index import EngineError, LodeIndex
from lodedb.engine.native_adapter import NativeCoreAdapter, NativeCoreEngineHandle
from lodedb.engine.route_registry import default_route_registry, load_route_registry
from lodedb.engine.runtime_policy import (
    NativeCoreMode,
    commit_mode_from_env,
    native_core_mode_from_env,
    native_core_strict_parity_from_env,
    native_core_write_mode_from_env,
    parse_commit_mode,
)
from lodedb.local.backends import (
    LocalEmbeddingResolution,
    build_local_embedding_backend,
)
from lodedb.local.presets import (
    LocalModelPreset,
    custom_embedder_route_policy,
    resolve_preset,
    vector_only_route_policy,
)

# Fixed local identifier for the single-process client context. It never leaves
# the process and has no security meaning — the local engine is auth-free.
_LOCAL_CLIENT_ID = "lodedb-local"
_LOCAL_CLIENT_ID_HASH = sha256_text(_LOCAL_CLIENT_ID)
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


class VectorOnlyIndexError(RuntimeError):
    """Raised when a text-in call (``add``/``search``) is made on a vector-only handle.

    A vector-only index (opened with ``vector_dim=`` / :meth:`open_vector_store`)
    has no embedding model, so text cannot be embedded; use ``add_vectors`` /
    ``search_by_vector`` with precomputed vectors instead.
    """


class ImageEmbeddingUnsupportedError(RuntimeError):
    """Raised when ``add_image``/``search_by_image`` is used on a non-multimodal index.

    Image verbs need a backend that embeds images: open with ``model="clip"`` (the
    shared image/text preset), or pass a custom ``embedder=`` that exposes an
    ``embed_images`` method. A text-only preset (``"minilm"``/``"bge"``) or a
    vector-only index cannot embed images; for the latter, embed the image with
    your own model and use ``add_vectors`` / ``search_by_vector``.
    """


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
        vector_dim: int | None = None,
        embedder: EngineEmbeddingBackend | None = None,
        bit_width: int | None = None,
        device: str = "auto",
        embedding_runtime: str = "auto",
        batch_size: int = 32,
        max_seq_length: int | None = None,
        chunk_character_limit: int = 900,
        store_text: bool = True,
        index_text: bool = False,
        read_only: bool = False,
        durability: str | None = None,
        commit_mode: str | None = None,
        route_registry_path: str | Path | None = None,
        _embedding_backend: Any | None = None,
    ) -> None:
        """Opens (or creates) an on-disk local index, loading any persisted state.

        ``model`` is a preset (``"minilm"`` fast default, ``"bge"`` quality). Pass
        ``embedder=`` instead to drive the index with your own
        :class:`~lodedb.engine.embedding_backends.EngineEmbeddingBackend` at any
        embedding dimension; ``model`` is then ignored and the index shape is taken
        from the backend's ``native_dim`` (and its ``required_model_name``, when it
        declares one, is pinned into the snapshot header and re-enforced on reopen,
        so reopening with a mismatching embedder is rejected rather than scored).
        Use a non-secret public identifier for ``required_model_name``: it is written
        to the on-disk header, so it must not carry credentials or API keys.
        ``embedder`` is mutually exclusive with ``vector_dim`` (a vector-only,
        no-embedder index opened via :meth:`open_vector_store`).
        ``device`` is ``"auto"``/``"cpu"``/``"mps"``/``"cuda"`` (embedding only).
        ``embedding_runtime`` selects the embedding runtime: ``"auto"`` (default;
        prefer ONNX Runtime, fall back to PyTorch sentence-transformers when ONNX
        cannot be set up), ``"onnx"`` (force ONNX Runtime), or ``"torch"`` (force
        sentence-transformers). ``embedding_resolution`` reports which was used.
        ``read_only=True`` opens a non-mutating snapshot handle that takes **no**
        writer lock, so it can read a path while another process holds the
        single-writer lock (e.g. query while ``lodedb serve`` runs); mutating
        calls raise :class:`ReadOnlyError` and the path must already exist. See
        :meth:`open_readonly`.
        ``durability`` is ``"fast"`` (default: atomic but not power-loss durable)
        or ``"fsync"`` (fsync each file + its directory on commit, trading commit
        throughput for power-loss durability). Unset reads ``LODEDB_DURABILITY``.
        ``commit_mode`` selects how each mutation is committed and defaults to
        ``"wal"``: each ``add``/``remove`` appends one framed record to a ``<key>.wal`` file
        (a single ``write`` plus, in ``durability="fsync"``, one ``fsync``) and a
        full generation is checkpointed only periodically, which makes the common
        single-add commit far cheaper. The WAL is replayed crash-atomically on the
        next open (a half-written trailing record is discarded), and a clean
        ``close``/``persist`` folds it into a generation. WAL mode drops the
        lock-free concurrent-reader snapshot that ``open_readonly`` relies on for
        *uncheckpointed* writes — it is meant for single-process deployments —
        but the on-disk generation a reader sees is always a consistent committed
        one. Pass ``"generation"`` for the historical path where every change
        publishes a new crash-atomic, lock-free MVCC-readable generation. Unset
        reads ``LODEDB_COMMIT_MODE``.
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
        self._native_core_mode = native_core_mode_from_env()
        self._native_core_write_mode = native_core_write_mode_from_env()
        self._native_core_strict_parity = native_core_strict_parity_from_env()
        self._native_core_fail_closed = (
            self._native_core_mode == NativeCoreMode.ON and "LODEDB_NATIVE_CORE" in os.environ
        ) or (
            self._native_core_write_mode == NativeCoreMode.ON
            and "LODEDB_NATIVE_CORE_WRITE" in os.environ
        )
        self._native_vector_engine: NativeCoreEngineHandle | None = None
        # The native CoreEngine is an unsendable PyO3 object: it may only be
        # touched from the thread that created it. Records that thread so the
        # native path can fall back to the thread-safe Python oracle when a
        # shared handle is used from worker threads (see _native_thread_local).
        self._native_engine_thread_id: int | None = None
        self._native_vector_mutable = False
        self._native_vector_covered = False
        self._native_core_fallback_reason = ""
        self._native_core_version = ""
        self._native_core_abi_version = 0
        self._native_write_shadow_dir: tempfile.TemporaryDirectory[str] | None = None
        self._native_write_through_enabled = False
        self._native_shadow_persist_count = 0
        self._native_shadow_persist_verified = False
        self._native_text_shadow_enabled = False
        self._chunk_character_limit = int(chunk_character_limit)
        # bit_width is only meaningful for vector-only / custom-embedder indexes (a
        # preset's width is fixed by its route). TurboVec stores 2- or 4-bit codes, so
        # reject any other width at the boundary; ``None`` means "use the index default".
        if bit_width is not None and int(bit_width) not in {2, 4}:
            raise ValueError(f"bit_width must be 2 or 4, got {bit_width!r}")
        resolved_bit_width = 4 if bit_width is None else int(bit_width)
        # "fast" (atomic rename only) vs "fsync" (power-loss durable). An
        # explicit arg wins; otherwise LODEDB_DURABILITY, else fast.
        fsync_on_commit = (
            durability_from_env() if durability is None else normalize_durability(durability)
        )
        # "wal" (append + periodic checkpoint, default) vs "generation"
        # (per-mutation MVCC publish). Explicit arg wins; otherwise LODEDB_COMMIT_MODE.
        resolved_commit_mode = (
            commit_mode_from_env() if commit_mode is None else parse_commit_mode(commit_mode)
        )
        seq_len = int(max_seq_length) if max_seq_length is not None else 256
        # vector_dim set => a vector-only index (no embedding model): only the
        # vector-in verbs work, at the caller's chosen dim. Otherwise a preset
        # index that embeds text internally.
        self.vector_only = vector_dim is not None
        self.commit_mode = resolved_commit_mode
        self.preset: LocalModelPreset | None
        if embedder is not None and self.vector_only:
            raise ValueError("embedder and vector_dim are mutually exclusive")
        if embedder is not None and _embedding_backend is not None:
            raise ValueError("embedder and _embedding_backend are mutually exclusive")

        if self.vector_only:
            if _embedding_backend is not None:
                raise ValueError("vector_dim and _embedding_backend are mutually exclusive")
            dim = int(vector_dim)  # type: ignore[arg-type]
            if not 1 <= dim <= 65536:
                raise ValueError("vector_dim must be between 1 and 65536")
            self.preset = None
            self._vector_dim_value: int | None = dim
            backend = None
            self._embedding_backend = None
            self.embedding_resolution = LocalEmbeddingResolution(
                requested_device=device,
                backend_name="none",
                effective_device="none",
                fallback_used=False,
                fallback_reason="vector-only index (no embedding model)",
            )
            route_policy = vector_only_route_policy(dim, bit_width=resolved_bit_width)
            route_profile = route_policy.profile
        elif embedder is not None:
            # A caller-supplied embedding backend: the index is text-capable (the
            # backend embeds in/out), but its shape comes from the backend, not a
            # preset, so it can run at any dimension.
            native_dim = int(getattr(embedder, "native_dim", 0))
            if native_dim <= 0:
                raise ValueError("embedder must expose a positive native_dim")
            model_identity = getattr(embedder, "required_model_name", None)
            if not model_identity:
                raise ValueError(
                    "embedder must set a non-empty required_model_name: a public, "
                    "non-secret identifier for the model that produced the vectors. It is "
                    "pinned into the index header and re-enforced on reopen, so a "
                    "same-dimension different-model backend is rejected rather than scored. "
                    "Identity-free fixtures belong on the internal _embedding_backend hook."
                )
            self.preset = None
            self._vector_dim_value = native_dim
            backend = embedder
            self._embedding_backend = backend
            self.embedding_resolution = LocalEmbeddingResolution(
                requested_device=device,
                backend_name=getattr(backend, "name", "custom"),
                effective_device="injected",
                fallback_used=False,
                fallback_reason="",
            )
            route_policy = custom_embedder_route_policy(
                native_dim, bit_width=resolved_bit_width, model_identity=model_identity
            )
            route_profile = route_policy.profile
        else:
            self.preset = resolve_preset(model)
            # A preset's bit width is fixed by its route; an explicit, conflicting
            # bit_width would otherwise be silently ignored (the store stays the
            # preset's width). Reject it so the requested width is never a lie.
            if bit_width is not None and int(bit_width) != self.preset.turbovec_bit_width:
                raise ValueError(
                    f"model {model!r} is a {self.preset.turbovec_bit_width}-bit preset; "
                    "bit_width is only configurable for a vector_dim= or embedder= index"
                )
            self._vector_dim_value = None
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
                    embedding_runtime=embedding_runtime,
                )
            self._embedding_backend = backend
            route_policy = self.preset.route_policy
            route_profile = self.preset.route_profile

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
            route_profile=route_profile,
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
            chunk_character_limit=self._chunk_character_limit,
            persistence_dir=self.path,
            read_only=self.read_only,
            fsync_on_commit=fsync_on_commit,
            commit_mode=resolved_commit_mode,
            embedding_backend=backend,
            route_policy=route_policy,
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
        native_durability = "fsync" if fsync_on_commit else "relaxed"
        self._maybe_init_native_vector_engine(
            resolved_bit_width,
            durability=native_durability,
            commit_mode=resolved_commit_mode.value,
            route_profile=route_profile,
            storage_profile=route_policy.index_backend,
            model=route_policy.model,
            provider=route_policy.provider,
            task=route_policy.task,
        )
        if self._native_vector_engine is not None:
            self._native_engine_thread_id = threading.get_ident()
        # Redacted, per-handle image-embedding counters (no paths/captions), surfaced
        # under stats()["image_embedding"] so operators can see CLIP encode cost.
        # Split by phase: "ingest" (add_image/add_images) vs "query" (search_by_image).
        self._image_metrics: dict[str, dict[str, Any]] = {
            phase: {
                "images_embedded": 0,
                "encode_calls": 0,
                "encode_seconds": 0.0,
                "encode_failures": 0,
            }
            for phase in ("ingest", "query")
        }

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

    @classmethod
    def open_vector_store(
        cls,
        path: str | Path,
        *,
        vector_dim: int,
        bit_width: int = 4,
        **kwargs: Any,
    ) -> LodeDB:
        """Opens (or creates) a bring-your-own-vectors index at a chosen dimension.

        Sugar for ``LodeDB(path, vector_dim=vector_dim, bit_width=bit_width, ...)``.
        The index has **no internal embedding model**: only ``add_vectors`` /
        ``add_vectors_many`` / ``search_by_vector`` / ``search_many_by_vector``
        work, and the text-in verbs (``add``/``search``) raise
        :class:`VectorOnlyIndexError`. Vectors must have dimension ``vector_dim``
        (any value your own embedder produces, e.g. 1536 or 3072), so this is the
        path for plugging LodeDB in as the vector backend behind a system that owns
        its embedder. The dimension and a redacted ``model="external"`` identity are
        persisted and re-enforced on reopen. Install without ``sentence-transformers``
        is fine for a vector-only workload (the embedder is imported lazily).
        """

        return cls(path, vector_dim=vector_dim, bit_width=bit_width, **kwargs)

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
        self._require_text_capable()
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        document_id = str(id) if id is not None else self._next_auto_id()
        document = EngineDocument(
            document_id=document_id,
            text=text,
            metadata=_coerce_metadata(metadata),
        )
        self._index.upsert_batch((document,))
        self._native_upsert_text_documents((document,))
        return document_id

    def add_many(
        self,
        documents: list[Mapping[str, Any]],
    ) -> list[str]:
        """Adds a batch of ``{"text", "id"?, "metadata"?}`` docs; returns the ids.

        Batched embedding is more efficient than repeated ``add`` calls.
        """

        self._require_writable()
        self._require_text_capable()
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
            self._native_upsert_text_documents(tuple(payload))
        return ids

    def add_vectors(
        self,
        vector: Sequence[float],
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        text: str | None = None,
        normalize: bool = True,
    ) -> str:
        """Adds (or replaces) one document from a precomputed embedding vector.

        The *vector-in* counterpart to :meth:`add`: the caller supplies the
        embedding directly (e.g. from their own model), and LodeDB stores it
        verbatim without embedding or chunking text — so this is how an
        external system (or a graph layer that embeds once) reuses its own
        vectors. The vector must have the index's embedding dimension
        (``minilm`` -> 384, ``bge`` -> 768; see :attr:`preset`). It is
        L2-normalized by default so cosine scores stay comparable with the text
        path and with self-embedded documents in the same index; pass
        ``normalize=False`` if your vectors are already unit-norm.

        ``text`` is optional retained payload text for integrations that own
        embeddings but still need durable payload reconstruction. It is stored
        only in the dedicated raw-text sidecar when ``store_text=True``; it is
        never embedded, chunked, or written into redacted metadata. Without
        ``text``, :meth:`get` returns ``None`` for the vector-in document. Reusing
        an ``id`` upserts; an identical vector is a no-op re-encode. The mutation
        commits atomically before returning.

        Note: vectors added here are compared by similarity against everything
        else in the index, so only mix vectors produced by the *same* embedding
        model (mixing models in one index makes scores meaningless).
        """

        self._require_writable()
        document_id = str(id) if id is not None else self._next_auto_id()
        prepared = _prepare_vector(vector, self._vector_dim, normalize=normalize)
        self._index.upsert_vectors_batch(
            (
                EngineVectorDocument(
                    document_id=document_id,
                    vector=prepared,
                    metadata=_coerce_metadata(metadata),
                    text=_coerce_optional_text(text),
                ),
            )
        )
        self._native_upsert_vectors(
            (
                EngineVectorDocument(
                    document_id=document_id,
                    vector=prepared,
                    metadata=_coerce_metadata(metadata),
                    text=_coerce_optional_text(text),
                ),
            )
        )
        return document_id

    def add_vectors_many(
        self,
        documents: list[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> list[str]:
        """Adds a batch of ``{"vector", "id"?, "metadata"?}`` precomputed-vector docs.

        Vector-in counterpart to :meth:`add_many`. Each ``vector`` must match the
        index embedding dimension and is L2-normalized by default (see
        :meth:`add_vectors`). Returns the ids in input order.
        """

        self._require_writable()
        payload: list[EngineVectorDocument] = []
        ids: list[str] = []
        for item in documents:
            vector = item.get("vector")
            if vector is None:
                raise ValueError("each document needs a 'vector'")
            document_id = str(item["id"]) if item.get("id") is not None else self._next_auto_id()
            ids.append(document_id)
            payload.append(
                EngineVectorDocument(
                    document_id=document_id,
                    vector=_prepare_vector(vector, self._vector_dim, normalize=normalize),
                    metadata=_coerce_metadata(item.get("metadata")),
                    text=_coerce_optional_text(item.get("text")),
                )
            )
        if payload:
            self._index.upsert_vectors_batch(tuple(payload))
            self._native_upsert_vectors(tuple(payload))
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

        self._require_text_capable()
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        resolved_mode = self._resolve_mode(mode)
        normalized_filter = _normalize_filter(filter)
        # Text/lexical/hybrid search keeps the Python oracle + parity cross-check
        # even in native-on: the native lexical/hybrid path still has divergences
        # from Python (incremental lexical rebuilds, vector-caption lexical, commit
        # drift), so Python stays authoritative here and native results are only
        # returned when they match. The authoritative native fast path is applied
        # to the vector-in search paths, where native is parity-clean.
        response = self._index.query(
            query,
            top_k=int(k),
            filter=normalized_filter,
            mode=resolved_mode,
            include=("metadata",),
        )
        python_hits = self._hits_from_result_rows(response.get("results", []))
        native_hits = self._native_search_text(
            query,
            k=int(k),
            filter=normalized_filter,
            mode=resolved_mode,
        )
        if native_hits is not None:
            self._check_native_hit_parity(
                python_hits,
                native_hits,
                reason="native_core_text_parity_mismatch",
            )
            if self._native_vector_covered and self._native_core_mode == NativeCoreMode.ON:
                return native_hits
        return python_hits

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

        self._require_text_capable()
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
                    "include": ("metadata",),
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

    def search_by_vector(
        self,
        vector: Sequence[float],
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        normalize: bool = True,
    ) -> list[LodeSearchHit]:
        """Returns the top-``k`` hits for a precomputed query embedding vector.

        The *vector-in* counterpart to :meth:`search`: the caller supplies the
        query embedding directly, so no text is embedded. The vector must have
        the index embedding dimension and is L2-normalized by default to match
        how stored vectors are normalized. ``filter`` takes the same
        exact-match-or-predicate grammar as :meth:`search` and is pushed into
        the TurboVec allowlist identically.
        """

        if k <= 0:
            raise ValueError("k must be positive")
        prepared = _prepare_vector(vector, self._vector_dim, normalize=normalize)
        normalized_filter = _normalize_filter(filter)
        if self._native_query_authoritative() and self._native_filter_shortcut_safe(
            normalized_filter
        ):
            # Explicit native-on with no parity/shadow validation: the native
            # result is authoritative, so skip the redundant Python oracle query
            # on the hot path. Fall back to Python only when this handle is not
            # natively covered.
            native_hits = self._native_search_by_vector(
                prepared, k=int(k), filter=normalized_filter
            )
            if native_hits is not None:
                return native_hits
            response = self._index.query_vector(
                prepared, top_k=int(k), filter=normalized_filter, include=("metadata",)
            )
            return self._hits_from_result_rows(response.get("results", []))
        response = self._index.query_vector(
            prepared,
            top_k=int(k),
            filter=normalized_filter,
            include=("metadata",),
        )
        python_hits = self._hits_from_result_rows(response.get("results", []))
        native_hits = self._native_search_by_vector(
            prepared,
            k=int(k),
            filter=normalized_filter,
        )
        if native_hits is None:
            return python_hits
        self._check_native_vector_parity(python_hits, native_hits)
        if not self._native_vector_covered:
            return python_hits
        if self._native_core_mode == NativeCoreMode.ON:
            return native_hits
        return python_hits

    def search_many_by_vector(
        self,
        vectors: list[Sequence[float]],
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        normalize: bool = True,
    ) -> list[list[LodeSearchHit]]:
        """Returns top-``k`` hits for each precomputed query vector, preserving order.

        Batched vector-in search; like :meth:`search_many` it is the path that
        lets CUDA hosts use the GPU-resident scan for eligible batches.
        """

        if not isinstance(vectors, list) or not vectors:
            raise ValueError("vectors must be a non-empty list")
        if k <= 0:
            raise ValueError("k must be positive")
        normalized_filter = _normalize_filter(filter)
        prepared_vectors = [
            _prepare_vector(vector, self._vector_dim, normalize=normalize) for vector in vectors
        ]
        if self._native_query_authoritative() and self._native_filter_shortcut_safe(
            normalized_filter
        ):
            native_batches = self._native_search_many_by_vector(
                prepared_vectors, k=int(k), filter=normalized_filter
            )
            if native_batches is not None:
                return native_batches
            # Not natively covered: fall back to the Python oracle batch query.
        items = [
            {
                "vector": prepared,
                "top_k": int(k),
                "filter": normalized_filter,
                "include": ("metadata",),
            }
            for prepared in prepared_vectors
        ]
        batches = self._index.query_vectors_batch(items).get("queries", [])
        if not isinstance(batches, list):
            raise RuntimeError("invalid engine response: queries must be a list")
        out: list[list[LodeSearchHit]] = []
        for item in batches:
            if not isinstance(item, Mapping):
                raise RuntimeError("invalid engine response: query item must be an object")
            out.append(self._hits_from_result_rows(item.get("results", [])))
        if self._native_query_authoritative():
            # Already tried native above and it was not covered; serve Python.
            return out
        native_batches = self._native_search_many_by_vector(
            prepared_vectors,
            k=int(k),
            filter=normalized_filter,
        )
        if native_batches is None:
            return out
        for python_hits, native_hits in zip(out, native_batches, strict=True):
            self._check_native_vector_parity(python_hits, native_hits)
        if not self._native_vector_covered:
            return out
        if self._native_core_mode == NativeCoreMode.ON:
            return native_batches
        return out

    def add_image(
        self,
        image: Any,
        *,
        id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        text: str | None = None,
    ) -> str:
        """Embeds one image with the index's multimodal model and stores it; returns its id.

        Requires a multimodal index (``model="clip"``, or a custom ``embedder=``
        exposing ``embed_images``); otherwise raises
        :class:`ImageEmbeddingUnsupportedError`. ``image`` may be a filesystem path
        (``str`` / :class:`~pathlib.Path`), raw ``bytes``, or a PIL ``Image``. The
        image is embedded into the model's shared image/text space and stored as a
        single vector via the same atomic-commit + TurboVec path as
        :meth:`add_vectors`.

        The raw image bytes are **not** stored: keep the file on disk (or in object
        storage) and put its path/URI in ``metadata`` so a hit can be resolved back
        to the image; ``text`` is optional retained payload (e.g. a caption), kept
        only in the raw-text sidecar when ``store_text=True``. Because the CLIP
        space is shared, a stored image is retrievable cross-modally by
        :meth:`search` (a text query) and by :meth:`search_by_image`.
        """

        self._require_writable()
        backend = self._image_backend()
        # Validate the cheap payload before the expensive encode, so a bad request
        # wastes no CLIP encode and does not skew ingest metrics (mirrors add_images).
        _coerce_metadata(metadata)
        _coerce_optional_text(text)
        vector = self._embed_images_tracked(backend, (image,), phase="ingest")[0]
        return self.add_vectors(vector, id=id, metadata=metadata, text=text)

    def add_images(
        self,
        documents: list[Mapping[str, Any]],
        *,
        normalize: bool = True,
    ) -> list[str]:
        """Embeds and stores a batch of images, then commits once.

        Batched counterpart to :meth:`add_image`: each item is a mapping with an
        ``"image"`` (path / ``bytes`` / PIL image) plus optional ``"id"``,
        ``"metadata"``, and ``"text"`` (caption). Images are decoded and encoded in
        backend-sized batches, so peak *decoded-image* memory is bounded by the batch
        rather than the whole gallery; the resulting vectors do accumulate (one
        ``EngineVectorDocument`` per image) for a single atomic commit, so this is far
        cheaper than repeated :meth:`add_image`.

        This is an **atomic batch, not a streaming bulk-ingest API**: the whole call
        commits once, so memory grows with the call size and a late failure loses the
        whole call. For a large gallery, drive it in chunks yourself, one atomic commit
        per chunk, which also gives natural progress and resume points::

            CHUNK = 512
            for start in range(0, len(items), CHUNK):
                db.add_images(items[start : start + CHUNK])   # one commit per chunk
                # ...record progress (e.g. `start`) here to resume after a failure

        Returns the ids in input order. Requires a multimodal index (``model="clip"``
        or a custom ``embedder=`` exposing ``embed_images``); the per-image storage
        contract matches :meth:`add_image` (the raw image bytes are never stored).
        """

        self._require_writable()
        backend = self._image_backend()
        if not isinstance(documents, list):
            raise ValueError("documents must be a list")
        # Validate and coerce every item up front (image present, metadata/text valid)
        # so a bad item fails before any image is embedded, not after a wasted batch.
        images: list[Any] = []
        ids: list[str] = []
        metadatas: list[dict[str, str]] = []
        texts: list[str | None] = []
        for item in documents:
            image = item.get("image")
            if image is None:
                raise ValueError("each document needs an 'image'")
            images.append(image)
            ids.append(str(item["id"]) if item.get("id") is not None else self._next_auto_id())
            metadatas.append(_coerce_metadata(item.get("metadata")))
            texts.append(_coerce_optional_text(item.get("text")))
        if not images:
            return []
        # Decode + encode in backend-sized batches so peak decoded-image memory is
        # bounded by the batch, then commit all vectors at once.
        batch_size = max(1, int(getattr(backend, "batch_size", 16) or 16))
        vectors: list[Any] = []
        for start in range(0, len(images), batch_size):
            vectors.extend(
                self._embed_images_tracked(
                    backend, images[start : start + batch_size], phase="ingest"
                )
            )
        payload = [
            EngineVectorDocument(
                document_id=ids[index],
                vector=_prepare_vector(vectors[index], self._vector_dim, normalize=normalize),
                metadata=metadatas[index],
                text=texts[index],
            )
            for index in range(len(images))
        ]
        self._index.upsert_vectors_batch(tuple(payload))
        self._native_upsert_vectors(tuple(payload))
        return ids

    def search_by_image(
        self,
        image: Any,
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
    ) -> list[LodeSearchHit]:
        """Returns the top-``k`` hits for an image query, cross-modal over the shared space.

        Requires a multimodal index (``model="clip"``, or a custom ``embedder=``
        exposing ``embed_images``); otherwise raises
        :class:`ImageEmbeddingUnsupportedError`. The ``image`` (path / bytes / PIL
        image) is embedded into the shared image/text space and searched with the
        same vector-in path as :meth:`search_by_vector`, so it matches both stored
        images and stored text. ``filter`` takes the same grammar as
        :meth:`search`.
        """

        if k <= 0:
            raise ValueError("k must be positive")
        backend = self._image_backend()
        vector = self._embed_images_tracked(backend, (image,), phase="query")[0]
        return self.search_by_vector(vector, k=k, filter=filter)

    def remove(self, id: str) -> bool:
        """Removes one document by id. Returns True if a document was deleted."""

        self._require_writable()
        if not isinstance(id, str) or not id.strip():
            raise ValueError("id must be a non-empty string")
        # `delete_documents` reports `document_count` as the number of unique
        # ids *requested* (not necessarily existing); `deleted_chunks` counts
        # chunks actually removed, so a positive value means the doc existed.
        response = self._index.delete_batch((id,))
        deleted = int(response.get("deleted_chunks", 0) or 0) > 0
        self._native_delete_documents((id,))
        return deleted

    def _update_document_payload(
        self,
        id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        text: str | None = None,
        clear_text: bool = False,
    ) -> None:
        """Internal adapter hook for metadata/raw-text updates without re-embedding."""

        self._require_writable()
        if not isinstance(id, str) or not id.strip():
            raise ValueError("id must be a non-empty string")
        self._index.update_document_payload(
            id,
            metadata=_coerce_metadata(metadata) if metadata is not None else None,
            text=text,
            clear_text=clear_text,
        )
        # Python is the durable authority for payload-only updates; the native
        # core has no mirrored update path here. Invalidate native coverage so
        # subsequent native reads (get/get_document/list/count/search) do not
        # return the pre-update metadata or text. The Python oracle, which just
        # applied the update, serves these reads, and a later reopen re-seeds
        # native from the updated on-disk store.
        if self._native_vector_engine is not None and self._native_vector_covered:
            self._native_vector_covered = False
            self._native_text_shadow_enabled = False
            self._native_core_fallback_reason = "native_core_payload_update_unmirrored"

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
        native_handled, native_text = self._native_get_text(id)
        # Return a native HIT directly, but fall through to the Python oracle on a
        # native miss: an existing store seeded under an older raw-text sidecar
        # layout the native reader does not understand counts its documents (so
        # coverage is claimed) yet serves no text. Python loaded the same store and
        # is authoritative; a fresh native store always has the text it wrote, so
        # this adds no Python call on the covered fast path.
        if (
            native_handled
            and native_text is not None
            and self._native_core_mode == NativeCoreMode.ON
        ):
            return native_text
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
        native_handled, native_texts = self._native_get_texts(ids)
        if native_handled and self._native_core_mode == NativeCoreMode.ON:
            # Fall back to the Python oracle for any requested id native did not
            # return (e.g. an existing store seeded under an older raw-text sidecar
            # layout the native reader cannot serve). Native hits are preferred;
            # only the misses consult Python.
            missing = [value for value in ids if value not in native_texts]
            if not missing:
                return native_texts
            merged = dict(native_texts)
            merged.update(self._index.get_document_texts(missing))
            return merged
        return self._index.get_document_texts(ids)

    def get_document(self, id: str) -> dict[str, Any] | None:
        """Returns one document's redacted record by id, or ``None`` if absent.

        The record is payload-free — ``{"id", "metadata", "chunk_count",
        "content_hash"}`` — with **no** text and **no** vectors. This is the
        by-id metadata read a graph / knowledge-graph layer uses to resolve an
        edge's endpoints (or a node's attributes) without issuing a similarity
        search; use :meth:`get`/:meth:`get_text` to recover the raw text.
        """

        if not isinstance(id, str) or not id.strip():
            raise ValueError("id must be a non-empty string")
        native_handled, native_record = self._native_get_document(id)
        if native_handled and self._native_core_mode == NativeCoreMode.ON:
            return None if native_record is None else _public_document_record(native_record)
        try:
            record = self._index.get_document(id)
        except EngineError as exc:
            if exc.status_code == 404:
                return None
            raise
        return _public_document_record(record)

    def list_documents(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns the **complete** set of document records, optionally filtered.

        Unlike :meth:`search`, this is enumeration, not ranking: it returns
        *every* matching document — no ``k`` cap, no query vector, no scoring —
        which is the primitive a graph / knowledge-graph layer needs for
        deterministic traversal ("all edges whose ``src`` is X", "all nodes of
        ``type`` Person"). Each record is the payload-free
        ``{"id", "metadata", "chunk_count", "content_hash"}``.

        ``filter`` takes the same exact-match-or-predicate grammar as
        :meth:`search` (``$eq``/``$in``/``$gte``/``$and``/``$or``/… plus a
        ``document_ids`` allowlist). It is resolved engine-side through the
        per-field planner in O(matches), not by scanning the corpus, so this stays
        flat as the corpus grows while the match set stays small.
        """

        normalized_filter = _normalize_filter(filter)
        native_handled, native_records = self._native_list_documents(normalized_filter)
        if native_handled and self._native_core_mode == NativeCoreMode.ON:
            return [_public_document_record(record) for record in native_records]
        try:
            raw = self._index.list_documents(filter=normalized_filter)
        except EngineError as exc:
            # A read-only handle on a not-yet-written path has no index yet;
            # that is an empty store, not an error.
            if exc.status_code == 404:
                return []
            raise
        return [_public_document_record(record) for record in raw]

    def persist(self) -> dict[str, Any]:
        """Flushes durable on-disk state and returns redacted storage stats.

        The engine already persists on every mutation; this is an explicit
        durability + stats checkpoint. In the default ``commit_mode="wal"`` it folds the
        outstanding write-ahead log into a fresh committed generation (so the
        on-disk base is fully up to date and the WAL is empty); in
        ``commit_mode="generation"`` there is nothing buffered, so it only reports stats.
        State is reloaded automatically on the next ``LodeDB(path=...)`` open.
        """

        if not self.read_only and self._engine is not None:
            # Fold any WAL backlog into a generation; a no-op in generation mode.
            self._engine.checkpoint()
            self._native_persist()
        # Surface the engine's current redacted stats, which include persisted
        # byte accounting.
        return self._index.stats()

    def count(self, *, filter: Mapping[str, Any] | None = None) -> int:
        """Returns the number of documents stored, optionally matching a filter.

        With ``filter`` (the same grammar as :meth:`search` / :meth:`list_documents`)
        returns the count of matching documents, resolved engine-side through the
        per-field planner in O(matches) without materializing any record.
        """

        if filter is None:
            stats = self._native_stats()
            if stats is not None and self._native_core_mode == NativeCoreMode.ON:
                return int(stats.get("document_count", 0) or 0)
            return int(self._index.stats().get("document_count", 0) or 0)
        return self._index.count_documents(filter=_normalize_filter(filter))

    def stats(self) -> dict[str, Any]:
        """Returns redacted engine stats (counts, storage bytes, telemetry).

        Includes an ``"image_embedding"`` block with this handle's redacted CLIP
        encode counters split by phase (``"ingest"`` for add_image(s), ``"query"`` for
        search_by_image): images embedded, encode calls, cumulative encode seconds, and
        failures. So image-embedding cost is visible separately from storage commit
        cost, and ingest separately from query. The counters carry no paths, captions,
        or pixels, and are **scoped to this handle**: they start at zero on each open
        (they are not persisted) and are not aggregated across a collection's spaces or
        across processes. For fleet-wide image-encode visibility, read them per handle
        and aggregate in your own metrics pipeline.
        """

        stats = self._index.stats()
        if isinstance(stats, dict):
            stats["image_embedding"] = {
                phase: dict(counters) for phase, counters in self._image_metrics.items()
            }
            stats["native_core"] = self._native_core_stats()
        return stats

    def close(self) -> None:
        """Releases the single-writer lock and engine references; state stays on disk."""

        if self._engine is not None:
            self._engine.close()
        if self._native_vector_engine is not None:
            # Only close the unsendable native engine from its owning thread; a
            # cross-thread close would panic (PanicException escapes except
            # Exception). Dropping the reference suffices on a foreign thread:
            # Python owns the single-writer lock and the OS reclaims the rest at
            # process exit.
            owns_thread = (
                self._native_engine_thread_id is None
                or threading.get_ident() == self._native_engine_thread_id
            )
            if owns_thread:
                try:
                    self._native_vector_engine.close()
                except Exception:
                    pass
        if self._native_write_shadow_dir is not None:
            self._native_write_shadow_dir.cleanup()
            self._native_write_shadow_dir = None
        self._native_vector_engine = None
        self._native_vector_mutable = False
        self._native_vector_covered = False
        self._native_write_through_enabled = False
        self._index = None  # type: ignore[assignment]
        self._engine = None  # type: ignore[assignment]

    def __enter__(self) -> LodeDB:
        """Enters a context manager; state is already loaded on open."""

        return self

    def __exit__(self, *exc: object) -> None:
        """Exits the context manager (state is durable on disk already)."""

        self.close()

    # -- internals ----------------------------------------------------------

    def _maybe_init_native_vector_engine(
        self,
        bit_width: int,
        *,
        durability: str,
        commit_mode: str,
        route_profile: str,
        storage_profile: str,
        model: str,
        provider: str,
        task: str,
    ) -> None:
        """Initializes the native vector engine when the rollout policy allows it."""

        if self._native_core_mode == NativeCoreMode.OFF:
            return
        write_through = self._native_core_write_mode == NativeCoreMode.ON
        text_native = not self.vector_only and (
            self._native_core_mode != NativeCoreMode.OFF or write_through
        )
        if not self.vector_only and not text_native:
            return
        adapter = NativeCoreAdapter()
        if not adapter.available:
            self._native_core_fallback_reason = "native_core_extension_unavailable"
            if self._native_core_fail_closed:
                raise RuntimeError("LODEDB_NATIVE_CORE=on requires lodedb._native_core")
            return
        self._native_core_version = adapter.version
        self._native_core_abi_version = adapter.abi_version
        if self.read_only:
            try:
                native_engine = adapter.open_readonly_engine(
                    self.path,
                    durability=durability,
                    commit_mode=commit_mode,
                    store_text=self.store_text,
                    index_text=self.index_text,
                    chunk_character_limit=self._chunk_character_limit,
                )
                native_engine.stats(_LOCAL_INDEX_ID)
            except Exception as exc:
                self._native_core_fallback_reason = "read_only_native_vector_seed_unavailable"
                if self._native_core_fail_closed:
                    raise RuntimeError("failed to initialize read-only native core") from exc
                return
            self._native_vector_engine = native_engine
            self._native_vector_mutable = False
            self._native_vector_covered = True
            return
        document_count = int(self._index.stats().get("document_count", 0) or 0)
        can_seed_existing = self.vector_only or self.index_text or self.store_text
        if not write_through and document_count != 0 and can_seed_existing:
            try:
                native_engine = adapter.open_readonly_engine(
                    self.path,
                    durability=durability,
                    commit_mode=commit_mode,
                    store_text=self.store_text,
                    index_text=self.index_text,
                    chunk_character_limit=self._chunk_character_limit,
                )
                native_stats = native_engine.stats(_LOCAL_INDEX_ID)
            except Exception:
                self._native_core_fallback_reason = "native_core_existing_store_seed_unavailable"
                return
            if int(native_stats.get("document_count", 0) or 0) == document_count:
                self._native_vector_engine = native_engine
                self._native_vector_mutable = False
                self._native_vector_covered = True
                self._native_text_shadow_enabled = text_native and not self.vector_only
                return
            self._native_core_fallback_reason = "native_core_existing_store_seed_mismatch"
            return
        if not write_through and document_count != 0 and not self.vector_only:
            self._native_core_fallback_reason = "native_core_existing_text_seed_requires_text"
            return
        if write_through and document_count != 0 and not can_seed_existing:
            self._native_core_fallback_reason = "native_core_write_on_existing_store_unavailable"
            raise RuntimeError(
                "LODEDB_NATIVE_CORE_WRITE=on for existing text stores requires retained text"
            )
        try:
            if write_through:
                native_engine = adapter.open_engine(
                    path=self.path,
                    read_only=False,
                    durability=durability,
                    commit_mode=commit_mode,
                    store_text=self.store_text,
                    index_text=self.index_text,
                    chunk_character_limit=self._chunk_character_limit,
                )
            elif self._native_core_write_mode == NativeCoreMode.SHADOW:
                shadow_dir = tempfile.TemporaryDirectory(prefix="lodedb-native-shadow-")
                try:
                    native_engine = adapter.open_engine(
                        path=shadow_dir.name,
                        read_only=False,
                        durability=durability,
                        commit_mode="generation",
                        store_text=self.store_text,
                        index_text=self.index_text,
                        chunk_character_limit=self._chunk_character_limit,
                    )
                except Exception:
                    shadow_dir.cleanup()
                    raise
                self._native_write_shadow_dir = shadow_dir
            else:
                native_engine = adapter.new_engine()
            try:
                native_engine.stats(_LOCAL_INDEX_ID)
            except Exception:
                native_engine.create_index_with_options(
                    _native_vector_index_options(
                        index_id=_LOCAL_INDEX_ID,
                        index_key=index_state_key_for_client_hash(
                            _LOCAL_CLIENT_ID_HASH, _LOCAL_INDEX_ID
                        ),
                        client_id_hash=_LOCAL_CLIENT_ID_HASH,
                        name="lodedb-local",
                        model=model,
                        provider=provider,
                        task=task,
                        route_profile=route_profile,
                        storage_profile=storage_profile,
                        vector_dim=self._vector_dim,
                        bit_width=bit_width,
                    )
                )
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_init_failed"
            if self._native_core_fail_closed:
                raise RuntimeError("failed to initialize native core") from exc
            return
        native_document_count = int(
            native_engine.stats(_LOCAL_INDEX_ID).get("document_count", 0) or 0
        )
        if native_document_count != document_count:
            self._native_core_fallback_reason = "native_core_existing_store_seed_mismatch"
            if write_through:
                raise RuntimeError("native core existing-store seed mismatch")
            return
        self._native_vector_engine = native_engine
        self._native_vector_mutable = True
        self._native_vector_covered = True
        self._native_write_through_enabled = write_through and self._native_vector_covered
        self._native_text_shadow_enabled = text_native and self._native_vector_covered

    def _native_query_authoritative(self) -> bool:
        """True when explicit native-on should serve reads without the Python oracle.

        In ``NativeCoreMode.ON`` the native result is authoritative, so running the
        Python query first is redundant work on the hot path (the original cause of
        native-on being slower than native-off for the public API). The Python
        oracle is still run for the cross-check when strict parity or shadow-write
        validation is requested.
        """

        return (
            self._native_core_mode == NativeCoreMode.ON
            and not self._native_core_strict_parity
            and self._native_core_write_mode != NativeCoreMode.SHADOW
        )

    @staticmethod
    def _native_filter_shortcut_safe(normalized_filter: Mapping[str, Any] | None) -> bool:
        """False when a ``document_ids`` filter would fail engine validation.

        The native-authoritative read path skips the Python engine, which is where
        ``document_ids`` is validated (nonempty list of nonblank strings). When the
        filter would be rejected, return False so the caller falls back to the
        Python path and raises the same error native-off raises, instead of
        silently returning empty native results. The native path is not disabled.
        """

        if not normalized_filter:
            return True
        document_ids = normalized_filter.get("document_ids")
        if document_ids is None:
            return True
        if not isinstance(document_ids, list) or not document_ids:
            return False
        return all(isinstance(item, str) and item.strip() for item in document_ids)

    def _native_thread_local(self) -> bool:
        """True when the native engine exists and the caller owns its thread.

        ``PyCoreEngine`` is an unsendable PyO3 object: touching it from a thread
        other than the one that created it raises a ``PanicException`` (which is
        not an ``Exception`` and so escapes the per-call fallback guards). The
        Python engine is the thread-safe oracle, so the first cross-thread access
        permanently disables native coverage for this handle and every thread
        falls back to Python. Returns ``False`` (and trips that fallback) when the
        engine is absent or the caller is on a different thread.
        """

        if self._native_vector_engine is None:
            return False
        if (
            self._native_engine_thread_id is None
            or threading.get_ident() == self._native_engine_thread_id
        ):
            return True
        self._native_vector_covered = False
        self._native_text_shadow_enabled = False
        self._native_write_through_enabled = False
        self._native_core_fallback_reason = "native_core_cross_thread_access"
        return False

    def _native_upsert_vectors(self, documents: tuple[EngineVectorDocument, ...]) -> None:
        """Mirrors vector mutations into native core while Python remains durable."""

        if not documents or not self._native_thread_local() or not self._native_vector_covered:
            return
        if not self._native_vector_mutable:
            self._native_vector_covered = False
            self._native_core_fallback_reason = "native_core_readonly_seed_invalidated"
            return
        try:
            self._native_vector_engine.upsert_vectors(_LOCAL_INDEX_ID, documents)
            if self._native_should_persist_after_mutation():
                self._native_vector_engine.persist()
        except Exception as exc:
            self._native_vector_covered = False
            self._native_write_through_enabled = False
            self._native_core_fallback_reason = "native_core_upsert_failed"
            if self._native_core_fail_closed:
                raise RuntimeError("native core vector upsert failed") from exc
            if self._native_core_strict_parity:
                raise

    def _native_upsert_text_documents(self, documents: tuple[EngineDocument, ...]) -> None:
        """Mirrors text mutations into native core while Python remains durable."""

        if (
            not documents
            or not self._native_text_shadow_enabled
            or not self._native_thread_local()
            or not self._native_vector_covered
            or self._embedding_backend is None
        ):
            return
        if not self._native_vector_mutable:
            self._native_vector_covered = False
            self._native_text_shadow_enabled = False
            self._native_core_fallback_reason = "native_core_readonly_seed_invalidated"
            return
        try:
            plan = self._native_vector_engine.prepare_text_upsert(
                _LOCAL_INDEX_ID,
                documents,
                store_text=self.store_text,
                index_text=self.index_text,
                chunk_character_limit=self._chunk_character_limit,
            )
            chunks_to_embed = tuple(
                str(chunk.get("text", "")) for chunk in plan.get("chunks_to_embed", [])
            )
            started = time.perf_counter()
            embeddings = (
                self._embedding_backend.embed_documents(chunks_to_embed)
                if chunks_to_embed
                else ()
            )
            embedding_time_ms = (time.perf_counter() - started) * 1000.0
            self._native_vector_engine.apply_text_upsert(
                plan,
                embeddings,
                embedding_time_ms=embedding_time_ms,
            )
            if self._native_should_persist_after_mutation():
                self._native_vector_engine.persist()
        except Exception as exc:
            self._native_vector_covered = False
            self._native_text_shadow_enabled = False
            self._native_write_through_enabled = False
            self._native_core_fallback_reason = "native_core_text_upsert_failed"
            if self._native_core_fail_closed:
                raise RuntimeError("native core text upsert failed") from exc
            if self._native_core_strict_parity:
                raise

    def _native_delete_documents(self, document_ids: tuple[str, ...]) -> None:
        """Mirrors vector deletes into native core while Python remains durable."""

        if (
            not document_ids
            or not self._native_thread_local()
            or not self._native_vector_covered
        ):
            return
        if not self._native_vector_mutable:
            self._native_vector_covered = False
            self._native_core_fallback_reason = "native_core_readonly_seed_invalidated"
            return
        try:
            self._native_vector_engine.delete_documents(_LOCAL_INDEX_ID, document_ids)
            if self._native_should_persist_after_mutation():
                self._native_vector_engine.persist()
        except Exception as exc:
            self._native_vector_covered = False
            self._native_write_through_enabled = False
            self._native_core_fallback_reason = "native_core_delete_failed"
            if self._native_core_fail_closed:
                raise RuntimeError("native core vector delete failed") from exc
            if self._native_core_strict_parity:
                raise

    def _native_search_by_vector(
        self,
        vector: Sequence[float],
        *,
        k: int,
        filter: Mapping[str, Any] | None,
    ) -> list[LodeSearchHit] | None:
        """Returns native vector hits when this handle is covered by native state."""

        if not self._native_thread_local() or not self._native_vector_covered:
            return None
        try:
            payload = self._native_vector_engine.query_vector(
                _LOCAL_INDEX_ID,
                vector,
                top_k=k,
                filter=filter,
            )
        except Exception as exc:
            self._native_vector_covered = False
            self._native_core_fallback_reason = "native_core_query_failed"
            if self._native_core_fail_closed:
                raise RuntimeError("native core vector query failed") from exc
            if self._native_core_strict_parity:
                raise
            return None
        return self._hits_from_native_rows(payload.get("hits", []))

    def _native_search_many_by_vector(
        self,
        vectors: list[Sequence[float]],
        *,
        k: int,
        filter: Mapping[str, Any] | None,
    ) -> list[list[LodeSearchHit]] | None:
        """Returns native vector batch hits when this handle is covered by native state."""

        if not self._native_thread_local() or not self._native_vector_covered:
            return None
        try:
            batches = self._native_vector_engine.query_vectors_batch(
                _LOCAL_INDEX_ID,
                vectors,
                top_k=k,
                filter=filter,
            )
        except Exception as exc:
            self._native_vector_covered = False
            self._native_core_fallback_reason = "native_core_batch_query_failed"
            if self._native_core_fail_closed:
                raise RuntimeError("native core vector batch query failed") from exc
            if self._native_core_strict_parity:
                raise
            return None
        return [self._hits_from_native_rows(batch.get("hits", [])) for batch in batches]

    def _native_search_text(
        self,
        query: str,
        *,
        k: int,
        filter: Mapping[str, Any] | None,
        mode: str,
    ) -> list[LodeSearchHit] | None:
        """Returns native text hits when explicit shadow text coverage is active."""

        if (
            not self._native_text_shadow_enabled
            or not self._native_thread_local()
            or not self._native_vector_covered
        ):
            return None
        try:
            query_embedding = None
            if mode in {"vector", "hybrid"}:
                if self._embedding_backend is None:
                    raise RuntimeError("native text query requires an embedding backend")
                query_embedding = self._embedding_backend.embed_query(query)
            if hasattr(self._native_vector_engine, "search_text"):
                payload = self._native_vector_engine.search_text(
                    _LOCAL_INDEX_ID,
                    query,
                    mode,
                    query_embedding,
                    top_k=k,
                    filter=filter,
                )
            else:
                query_plan = self._native_vector_engine.prepare_query_text(query, mode)
                payload = self._native_vector_engine.search_embedded_text(
                    _LOCAL_INDEX_ID,
                    query_plan,
                    query_embedding,
                    top_k=k,
                    filter=filter,
                )
        except Exception as exc:
            self._native_vector_covered = False
            self._native_text_shadow_enabled = False
            self._native_core_fallback_reason = "native_core_text_query_failed"
            if self._native_core_fail_closed:
                raise RuntimeError("native core text query failed") from exc
            if self._native_core_strict_parity:
                raise
            return None
        return self._hits_from_native_rows(payload.get("hits", []))

    def _native_get_text(self, document_id: str) -> tuple[bool, str | None]:
        """Returns one native raw-text value when this handle is covered."""

        if not self._native_thread_local() or not self._native_vector_covered:
            return False, None
        try:
            return True, self._native_vector_engine.get_document_text(
                _LOCAL_INDEX_ID,
                document_id,
            )
        except Exception as exc:
            self._mark_native_read_failed("native_core_get_text_failed", exc)
            return False, None

    def _native_get_texts(self, document_ids: list[str]) -> tuple[bool, dict[str, str]]:
        """Returns native raw-text values when this handle is covered."""

        if not self._native_thread_local() or not self._native_vector_covered:
            return False, {}
        try:
            return True, self._native_vector_engine.get_document_texts(
                _LOCAL_INDEX_ID,
                document_ids,
            )
        except Exception as exc:
            self._mark_native_read_failed("native_core_get_texts_failed", exc)
            return False, {}

    def _native_get_document(self, document_id: str) -> tuple[bool, dict[str, Any] | None]:
        """Returns one native payload-free document record when covered."""

        if not self._native_thread_local() or not self._native_vector_covered:
            return False, None
        try:
            return True, self._native_vector_engine.get_document(
                _LOCAL_INDEX_ID,
                document_id,
            )
        except Exception as exc:
            self._mark_native_read_failed("native_core_get_document_failed", exc)
            return False, None

    def _native_list_documents(
        self,
        filter: Mapping[str, Any] | None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Returns native payload-free document records when covered."""

        if not self._native_thread_local() or not self._native_vector_covered:
            return False, []
        try:
            return True, self._native_vector_engine.list_documents(
                _LOCAL_INDEX_ID,
                filter=filter,
            )
        except Exception as exc:
            self._mark_native_read_failed("native_core_list_documents_failed", exc)
            return False, []

    def _native_stats(self) -> dict[str, Any] | None:
        """Returns native stats only when this handle is covered by native state."""

        if not self._native_thread_local() or not self._native_vector_covered:
            return None
        try:
            return self._native_vector_engine.stats(_LOCAL_INDEX_ID)
        except Exception:
            self._native_vector_covered = False
            self._native_core_fallback_reason = "native_core_stats_failed"
            return None

    def _native_core_stats(self) -> dict[str, Any]:
        """Returns redacted native-core rollout status for this handle."""

        stats = self._native_stats() or {}
        return {
            "mode": self._native_core_mode.value,
            "write_mode": self._native_core_write_mode.value,
            "version": self._native_core_version,
            "abi_version": self._native_core_abi_version,
            "enabled": self._native_vector_engine is not None,
            "covered": self._native_vector_covered,
            "fallback_reason": self._native_core_fallback_reason,
            "document_count": int(stats.get("document_count", 0) or 0),
            "write_through": self._native_write_through_enabled,
            "shadow_persist_count": self._native_shadow_persist_count,
            "shadow_persist_verified": self._native_shadow_persist_verified,
        }

    def _native_persist(self) -> None:
        """Persists the native shadow writer when write rollout requests it."""

        if (
            self._native_core_write_mode == NativeCoreMode.OFF
            or not self._native_thread_local()
            or not self._native_vector_covered
            or not self._native_vector_mutable
        ):
            return
        try:
            self._native_vector_engine.persist()
            if self._native_core_write_mode == NativeCoreMode.SHADOW:
                self._verify_native_shadow_persist()
        except Exception as exc:
            self._native_vector_covered = False
            self._native_write_through_enabled = False
            self._native_vector_mutable = False
            self._native_shadow_persist_verified = False
            self._native_core_fallback_reason = (
                "native_core_shadow_persist_failed"
                if self._native_core_write_mode == NativeCoreMode.SHADOW
                else "native_core_write_persist_failed"
            )
            if self._native_core_fail_closed:
                raise RuntimeError("native core persist failed") from exc
            if self._native_core_strict_parity:
                raise

    def _native_should_persist_after_mutation(self) -> bool:
        """Returns whether native writes should publish a generation immediately."""

        return self._native_write_through_enabled and self.commit_mode.value == "generation"

    def _mark_native_read_failed(self, reason: str, exc: Exception) -> None:
        """Records a native read fallback and applies rollout failure policy."""

        self._native_vector_covered = False
        self._native_text_shadow_enabled = False
        self._native_write_through_enabled = False
        self._native_vector_mutable = False
        self._native_core_fallback_reason = reason
        if self._native_core_fail_closed:
            raise RuntimeError("native core document read failed") from exc
        if self._native_core_strict_parity:
            raise RuntimeError("native core document read failed") from exc

    def _verify_native_shadow_persist(self) -> None:
        """Compares redacted native shadow counts to the Python authoritative store."""

        native_stats = self._native_stats()
        if native_stats is None:
            raise RuntimeError("native core shadow stats unavailable")
        python_stats = self._index.stats()
        for key in ("document_count", "chunk_count"):
            if key in native_stats and int(native_stats[key]) != int(python_stats.get(key, 0) or 0):
                raise RuntimeError(f"native core shadow {key} mismatch")
        self._native_shadow_persist_count += 1
        self._native_shadow_persist_verified = True

    def _check_native_vector_parity(
        self,
        python_hits: list[LodeSearchHit],
        native_hits: list[LodeSearchHit],
    ) -> None:
        """Raises on native/Python hit mismatch when strict parity is active."""

        self._check_native_hit_parity(
            python_hits,
            native_hits,
            reason="native_core_vector_parity_mismatch",
        )

    def _check_native_hit_parity(
        self,
        python_hits: list[LodeSearchHit],
        native_hits: list[LodeSearchHit],
        *,
        reason: str,
    ) -> None:
        """Raises on native/Python hit-order mismatch when strict parity is active."""

        python_ids = [hit.id for hit in python_hits]
        native_ids = [hit.id for hit in native_hits]
        if python_ids == native_ids:
            return
        # Equal-score ties are a valid ambiguity: with duplicate or exactly-tied
        # vectors, Python and native may order tied hits differently, and at the
        # top-k boundary may even select different members of a tie group. Both
        # are correct top-k results. Treat the results as matching when they are
        # the same length and every position's score agrees within tolerance; a
        # genuinely wrong ranking surfaces as a score mismatch, while a
        # differently-resolved tie does not.
        if len(python_hits) == len(native_hits) and all(
            math.isclose(p.score, n.score, rel_tol=1e-5, abs_tol=1e-6)
            for p, n in zip(python_hits, native_hits, strict=True)
        ):
            return
        self._native_vector_covered = False
        self._native_text_shadow_enabled = False
        self._native_core_fallback_reason = reason
        if self._native_core_fail_closed or self._native_core_strict_parity:
            raise RuntimeError(
                f"native core vector parity mismatch: python={python_ids!r} native={native_ids!r}"
            )

    @staticmethod
    def _hits_from_native_rows(rows: Any) -> list[LodeSearchHit]:
        """Hydrates native search rows into public payload-free hit objects."""

        if not isinstance(rows, list):
            raise RuntimeError("invalid native response: hits must be a list")
        hits: list[LodeSearchHit] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise RuntimeError("invalid native response: hit row must be an object")
            hits.append(
                LodeSearchHit(
                    score=float(row["score"]),
                    id=str(row["document_id"]),
                    metadata=dict(row.get("metadata", {})),
                )
            )
        return hits

    @property
    def _vector_dim(self) -> int:
        """The embedding dimension this index accepts.

        The preset's native dim in normal mode, or the caller's ``vector_dim`` in
        a vector-only index.
        """

        if self._vector_dim_value is not None:
            return self._vector_dim_value
        assert self.preset is not None  # preset mode always has a preset
        return self.preset.native_dim

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

    def _require_text_capable(self) -> None:
        """Raises :class:`VectorOnlyIndexError` on a vector-only (no-embedder) handle."""

        if self.vector_only:
            raise VectorOnlyIndexError(
                "this index is vector-only (no embedding model); use add_vectors / "
                "add_vectors_many / search_by_vector / search_many_by_vector"
            )

    def _image_backend(self) -> Any:
        """Returns the embedding backend if it can embed images, else raises.

        Image verbs are duck-typed on an ``embed_images`` method, which the
        ``"clip"`` preset's backend (and any custom multimodal ``embedder=``)
        provides and the text-only presets do not.
        """

        backend = self._embedding_backend
        if backend is None or not callable(getattr(backend, "embed_images", None)):
            raise ImageEmbeddingUnsupportedError(
                "this index cannot embed images; open LodeDB with model='clip' "
                "(install the extra: pip install 'lodedb[image]'), or pass a custom "
                "embedder= that exposes embed_images"
            )
        return backend

    def _embed_images_tracked(
        self, backend: Any, images: Sequence[Any], *, phase: str
    ) -> tuple[Any, ...]:
        """Embeds images via the backend, recording redacted encode metrics.

        Counts images, cumulative encode time, and failures into
        ``self._image_metrics[phase]`` (``"ingest"`` for add_image(s), ``"query"`` for
        search_by_image; surfaced via :meth:`stats`), so operators can see CLIP encode
        cost separately from storage-commit cost, and ingest separately from query. No
        paths or pixels are recorded.
        """

        metrics = self._image_metrics[phase]
        started = time.perf_counter()
        try:
            vectors = tuple(backend.embed_images(images))
        except Exception:
            metrics["encode_failures"] += 1
            raise
        metrics["encode_calls"] += 1
        metrics["images_embedded"] += len(images)
        metrics["encode_seconds"] += time.perf_counter() - started
        return vectors

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
            # The engine inlines redacted metadata when the query opts in via
            # include=("metadata",), which the search verbs now do. Fall back to a
            # by-id read only when a row lacks it, so any other caller stays correct.
            row_metadata = row.get("metadata")
            metadata = (
                dict(row_metadata)
                if isinstance(row_metadata, Mapping)
                else self._metadata_for_document(document_id)
            )
            hits.append(
                LodeSearchHit(
                    score=float(row["score"]),
                    id=document_id,
                    metadata=metadata,
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


def _public_document_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Projects an engine document record into the public, payload-free shape.

    The engine returns ``document_id``; the public SDK uses ``id`` everywhere
    (``add(id=...)``, ``LodeSearchHit.id``, ``get(id)``), so the key is renamed
    for consistency. Only redacted fields are surfaced — never text or vectors.
    """

    metadata = record.get("metadata", {})
    return {
        "id": str(record.get("document_id", "")),
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
        "chunk_count": int(record.get("chunk_count", 0) or 0),
        "content_hash": str(record.get("content_hash", "")),
    }


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


def _native_vector_index_options(
    *,
    index_id: str,
    index_key: str,
    client_id_hash: str,
    name: str,
    model: str,
    provider: str,
    task: str,
    route_profile: str,
    storage_profile: str,
    vector_dim: int,
    bit_width: int,
) -> dict[str, Any]:
    """Builds the native-core index creation payload for the local vector store."""

    return {
        "index_id": str(index_id),
        "index_key": str(index_key),
        "client_id_hash": str(client_id_hash),
        "name": str(name),
        "model": str(model),
        "provider": str(provider),
        "task": str(task),
        "route_profile": str(route_profile),
        "storage_profile": str(storage_profile),
        "vector_dim": int(vector_dim),
        "bit_width": int(bit_width),
    }


def _coerce_optional_text(value: Any) -> str | None:
    """Validates optional retained text on vector-in document mappings."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("vector document 'text' must be a string when provided")
    return value


def _prepare_vector(
    vector: Sequence[float],
    dim: int,
    *,
    normalize: bool,
) -> tuple[float, ...]:
    """Validates a precomputed embedding and (optionally) L2-normalizes it.

    Enforces the index dimension and finiteness at the SDK boundary so callers
    get a clean ``ValueError`` instead of a deep engine/kernel error. When
    ``normalize`` is set, the vector is scaled to unit norm so cosine scores
    match the text path (which normalizes embeddings on write and query).
    """

    try:
        values = [float(component) for component in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError("vector must be a sequence of numbers") from exc
    if len(values) != dim:
        raise ValueError(f"vector must have dimension {dim}, got {len(values)}")
    if not all(math.isfinite(component) for component in values):
        raise ValueError("vector must contain only finite values")
    if normalize:
        norm = math.sqrt(sum(component * component for component in values))
        if norm == 0.0:
            raise ValueError(
                "cannot normalize a zero vector; pass a non-zero vector or normalize=False"
            )
        values = [component / norm for component in values]
    return tuple(values)
