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

import atexit
import json
import math
import secrets
import sys
import threading
import time
import weakref
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lodedb.engine._atomic_io import durability_from_env, normalize_durability
from lodedb.engine._commit_manifest import (
    COMMIT_MANIFEST_SUFFIX,
    base_json_path,
    read_commit_manifest,
)
from lodedb.engine._filelock import ConcurrentWriterError
from lodedb.engine._predicate import coerce_sdk_filter
from lodedb.engine.core import (
    EngineDocument,
    EngineVectorDocument,
    _state_from_payload,
    chunk_text,
    index_state_key_for_client_hash,
    sha256_text,
)
from lodedb.engine.embedding_backends import EngineEmbeddingBackend
from lodedb.engine.native_adapter import NativeCoreAdapter, NativeCoreEngineHandle
from lodedb.engine.runtime_policy import (
    commit_mode_from_env,
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


class _ThreadConfinedNativeEngine:
    """Routes every native call onto the engine's single home worker thread.

    The native :class:`~lodedb.engine.native_adapter.NativeCoreEngineHandle` wraps
    an ``unsendable`` PyO3 ``PyCoreEngine``: touching it from a thread other than
    the one that created it raises a ``PanicException``. This proxy makes a shared
    handle usable from any thread by submitting each call to the dedicated
    single-worker executor that opened the engine; a call already on that worker
    runs inline to avoid a self-deadlock. Only the engine object itself is
    unsendable, and it is only ever touched inside :meth:`_run` on the worker, so
    the numpy arrays and JSON-able arguments callers pass cross threads fine. The
    engine is reached through a one-element ``holder`` list (shared with the GC
    finalizer) so teardown can drop it on the worker, never on a collecting thread.
    """

    __slots__ = ("_executor", "_home_thread_id", "_holder")

    def __init__(
        self,
        executor: ThreadPoolExecutor,
        home_thread_id: int,
        holder: list[NativeCoreEngineHandle],
    ) -> None:
        """Stores the worker executor, its thread id, and the engine holder list."""

        self._executor = executor
        self._home_thread_id = home_thread_id
        self._holder = holder

    def _run(self, fn):
        """Runs ``fn`` on the home worker thread (inline if already on it)."""

        if threading.get_ident() == self._home_thread_id:
            return fn()
        return self._executor.submit(fn).result()

    def __getattr__(self, name: str):
        """Proxies an attribute, wrapping callables to run on the worker thread."""

        attr = getattr(self._holder[0], name)
        if not callable(attr):
            return attr
        return lambda *args, **kwargs: self._run(lambda: attr(*args, **kwargs))


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
        embedding_dtype: str = "float32",
        batch_size: int = 32,
        max_seq_length: int | None = None,
        chunk_character_limit: int = 900,
        store_text: bool = True,
        index_text: bool | None = None,
        ann: str | AnnOptions | None = None,
        ann_clusters: int | None = None,
        ann_nprobe: int | None = None,
        compression: bool = True,
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
        sentence-transformers). ``embedding_resolution`` reports which was used,
        including whether a CUDA request fell back to the CPU (the default
        ``onnxruntime`` wheel is CPU-only; install ``onnxruntime-gpu`` for the GPU,
        and see ``docs/deployment-and-performance.md``). A fourth runtime,
        ``"torch-compile"``, is an opt-in ``torch.compile``d encoder for low
        single-query GPU-serving latency (text presets only).
        ``embedding_dtype`` (``"float32"`` default, ``"float16"``/``"bfloat16"``) is
        honored only by ``"torch-compile"`` and, on CUDA, halves the weight bytes
        streamed per forward. Half-precision embeddings are not bit-identical to fp32
        (measured cosine ~0.999 on MiniLM) but are recall-preserving, and every
        runtime returns L2-normalized fp32 vectors, so a store built at one dtype
        stays searchable from another.
        ``batch_size`` (default ``32``) is how many texts are embedded per forward
        pass; a larger batch raises embedding throughput on a GPU (and, less so, on
        the CPU) at some memory cost. ``search``/``search_many`` embed their query
        set in one batch, so a large batched query benefits from a bigger
        ``batch_size`` or a GPU. ``max_seq_length`` (default ``256`` tokens) is the
        per-document token budget before truncation: raise it for long documents
        whose tail carries meaning, lower it to embed faster.
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
        ``index_text`` controls durable lexical-index persistence and **defaults to
        match** ``store_text`` (so ``True`` by default, and ``False`` when you disable
        ``store_text``): the per-chunk tokens of each added document are kept in a
        dedicated ``.tvlex`` sidecar (base + ``.lxd`` journal, committed O(changed) per
        write), so ``mode="hybrid"``/``"lexical"`` are ready out of the box and survive
        a reopen without rebuilding from raw text. Tying the default to ``store_text``
        keeps the ``store_text=False`` "retain no text at all" promise: opting out of
        raw text also opts out of persisting tokens. The sidecar holds payload-derived
        terms only and, like the raw-text sidecar, never reaches telemetry, audit, or
        the redacted snapshot. Pass an explicit bool to decouple the two: ``True`` on a
        ``store_text=False`` store persists a lexical index without retaining raw text;
        ``False`` on a ``store_text=True`` store skips the ``.tvlex`` and rebuilds the
        lexical index from retained raw text on open instead. Reopen the same path with
        the same effective ``index_text`` value you wrote with.
        ``ann`` opts into approximate nearest-neighbor acceleration for the vector
        scan and defaults to ``None`` (exact scan, full recall). Pass
        ``ann="cluster"`` to enable IVF-style cluster pruning: the query scores
        cluster centroids, scans only the nearest clusters, and the exact TurboVec
        scan re-scores those candidates, so returned scores are exact but the result
        set is approximate (a true neighbor in an unprobed cluster can be missed, so
        recall is below 100%). Exact scan stays the default and the authority. Tune
        with ``ann_clusters`` (partition count, defaults to about ``sqrt(n)``) and
        ``ann_nprobe`` (clusters probed per query, defaults to about
        ``sqrt(clusters)``); probing every cluster reproduces the exact result.
        Equivalently, pass a structured :class:`AnnOptions` as ``ann=`` (e.g.
        ``ann=AnnOptions(clusters=256, nprobe=16)``) instead of the loose tuning
        keywords. ANN is worthwhile for large corpora where the full scan is the
        bottleneck; small and mid-size corpora should keep the exact default. Like ``compression``,
        ``ann`` is a create-time choice: on reopen of an existing store the
        persisted config wins and these arguments are ignored, so an exact store
        stays exact and an ANN store keeps its clustering (and tuning) regardless
        of what a reopen passes.
        ``compression`` controls whether the retained raw-text store (the
        ``.tvtext`` base and ``.txd`` segments) is zstd-compressed and defaults to
        ``True``; it has no effect when ``store_text=False``. The setting is
        persisted in the text-store manifest and the persisted value wins on
        reopen, so a store keeps the compression it was created with and the
        passed value only seeds a freshly created store. Reads are unaffected (the
        reader detects compression on disk), so a store written either way always
        reads back, and an existing store created before this option keeps loading.
        On reopen the persisted index identity is re-enforced: the embedding model,
        dimension, provider, task, storage profile, and bit width must match what
        the path was written with, and so must the effective ``store_text`` /
        ``index_text`` flags. Reopening a path with a different ``model`` (or
        ``embedder`` / ``vector_dim`` / ``bit_width``) raises rather than silently
        rescoring, so changing any of these means a fresh path and a reindex, not an
        in-place conversion (a vector-only path, for one, cannot be reopened as a
        text-in preset index).
        ``_embedding_backend`` is an internal hook for tests/fixtures.
        """

        self.path = Path(path)
        self.store_text = bool(store_text)
        # index_text defaults to match store_text: retaining text also persists its
        # lexical index (so hybrid/lexical are ready out of the box), and opting out
        # of text retention also opts out of persisting tokens (store_text=False keeps
        # its "no text at all" promise). Pass an explicit bool to decouple them.
        self.index_text = self.store_text if index_text is None else bool(index_text)
        # Opt-in ANN config (None => exact scan). Validated here for a friendly
        # error; the native core is the authority and re-validates the algorithm.
        self.ann = _resolve_ann_options(ann, ann_clusters, ann_nprobe)
        self.compression = bool(compression)
        self.read_only = bool(read_only)
        # The native core is the sole reader/writer for this handle. It is an
        # unsendable PyO3 object, so it is opened on (and only ever touched from) a
        # single dedicated worker thread; _ThreadConfinedNativeEngine routes every
        # call there so a shared handle works from any caller thread. _op_lock
        # serializes whole operations across concurrent callers so prepare/apply
        # spans, lazy embedding-model init, and the columnar index a query reads
        # never interleave.
        self._native_executor: ThreadPoolExecutor | None = None
        self._op_lock = threading.RLock()
        self._native_vector_engine: _ThreadConfinedNativeEngine | None = None
        self._native_engine_thread_id: int | None = None
        self._native_vector_mutable = False
        self._native_vector_covered = False
        self._native_core_fallback_reason = ""
        self._native_core_version = ""
        self._native_core_abi_version = 0
        self._native_write_through_enabled = False
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
                    embedding_dtype=embedding_dtype,
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
            ann=self.ann,
        )
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
        with self._op_lock:
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
            with self._op_lock:
                self._native_upsert_text_documents(tuple(payload))
        return ids

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embeds a batch of texts with the index's document embedding model.

        Uses the same document-side embedding path as :meth:`add` without
        storing anything. Texts longer than the index chunk limit are embedded
        as chunks, mean-pooled, and L2-normalized. Returns one vector per input
        text, in input order.
        """

        self._require_text_capable()
        if not isinstance(texts, list) or not texts:
            raise ValueError("texts must be a non-empty list of strings")
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("each text must be a non-empty string")
        backend = self._embedding_backend
        if backend is None:
            raise RuntimeError("text embedding requires an embedding backend")

        chunks_by_text = [chunk_text(text, self._chunk_character_limit) for text in texts]
        chunks: list[str] = []
        offsets: list[tuple[int, int]] = []
        for text, text_chunks in zip(texts, chunks_by_text, strict=True):
            start = len(chunks)
            # Preserve the existing one-text embedding behavior exactly. chunk_text
            # strips its input, while a one-chunk endpoint request previously reached
            # the backend unchanged.
            chunks.extend((text,) if len(text_chunks) == 1 else text_chunks)
            offsets.append((start, len(chunks)))

        with self._op_lock:
            vectors = backend.embed_documents(tuple(chunks))

        embeddings: list[list[float]] = []
        for start, end in offsets:
            if end - start == 1:
                embeddings.append(list(vectors[start]))
                continue
            pooled = np.asarray(vectors[start:end], dtype=np.float64).mean(axis=0)
            norm = float(np.linalg.norm(pooled))
            if norm == 0.0:
                raise ValueError("mean-pooled embedding has zero norm")
            embeddings.append((pooled / norm).tolist())
        return embeddings

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
        # Shares the vectorized prepare with add_vectors_many so a single add and a
        # batched add normalize a vector identically; tolist() keeps the stored
        # vector a plain float tuple.
        prepared = tuple(
            _prepare_vector_matrix([vector], self._vector_dim, normalize=normalize)[0].tolist()
        )
        document = EngineVectorDocument(
            document_id=document_id,
            vector=prepared,
            metadata=_coerce_metadata(metadata),
            text=_coerce_optional_text(text),
        )
        with self._op_lock:
            self._native_upsert_vectors((document,))
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
        raw_vectors: list[Any] = []
        ids: list[str] = []
        metadatas: list[Mapping[str, str]] = []
        texts: list[str | None] = []
        for item in documents:
            vector = item.get("vector")
            if vector is None:
                raise ValueError("each document needs a 'vector'")
            raw_vectors.append(vector)
            document_id = str(item["id"]) if item.get("id") is not None else self._next_auto_id()
            ids.append(document_id)
            metadatas.append(_coerce_metadata(item.get("metadata")))
            texts.append(_coerce_optional_text(item.get("text")))
        if not raw_vectors:
            return ids
        # Validate + normalize the whole batch as one matrix and hand it to the
        # native array upsert as-is: the vectors never become per-row Python tuples,
        # and normalize identically to a single add_vectors.
        matrix = _prepare_vector_matrix(raw_vectors, self._vector_dim, normalize=normalize)
        sidecar = [
            {"document_id": document_id, "metadata": metadata, "text": text}
            for document_id, metadata, text in zip(ids, metadatas, texts, strict=True)
        ]
        with self._op_lock:
            self._native_upsert_vectors_matrix(matrix, sidecar)
        return ids

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        mode: str | None = None,
    ) -> list[LodeSearchHit]:
        """Returns the top-``k`` hits as ``(score, id, metadata)``-style rows.

        ``mode`` selects the retrieval strategy. Left unset (the default) it
        resolves to ``"hybrid"`` when a lexical source is available
        (``store_text=True`` or ``index_text=True``, both on by default) and to
        ``"vector"`` otherwise, so the recommended fused ranking is the default
        wherever it can run while a vector-only store still searches without
        raising. Pass an explicit mode to override:

        - ``"vector"`` — embedding cosine similarity only.
        - ``"hybrid"`` — runs a lexical BM25 ranker alongside the vector scan and
          fuses the two ranked lists with Reciprocal Rank Fusion, so exact tokens
          that the embedding misses (error codes like ``E1234``, serials like
          ``ABC-123``, dates like ``2024-01-15``) are surfaced when they appear in
          the document body. The default for local RAG, where a missed exact match
          is the difference between a usable and a useless answer.
        - ``"lexical"`` — the BM25 ranking alone (no vector scan).

        ``"hybrid"`` and ``"lexical"`` build an in-memory BM25 index from a
        lexical source, so they require opening LodeDB with either
        ``index_text=True`` (a durable postings store that survives reopens
        without raw text) or ``store_text=True`` (the index rebuilt from the
        retained raw text, the default); requesting either *explicitly* with
        neither source raises :class:`ValueError`, whereas the unset default
        falls back to ``"vector"`` instead of raising. The serving index lives in
        memory, is maintained
        incrementally across mutations (a small change folds in only the changed
        chunks), and never changes the on-disk format.

        A document longer than ``chunk_character_limit`` (the ``LodeDB(...)``
        default is 900 characters) is split into chunks, and every mode scores
        chunks, so one long document can appear in the results more than once: each
        such hit carries the **same** ``id`` (the document id) with a different
        per-chunk ``score``. For one row per document, dedupe by ``hit.id`` keeping
        the first (best-scoring) occurrence. ``k`` counts chunk hits, so request a
        larger ``k`` when long documents are expected. :meth:`get` / :meth:`get_texts`
        return the reassembled full document text for an id regardless of chunking.

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
        with self._op_lock:
            return self._native_search_text(
                query,
                k=int(k),
                filter=normalized_filter,
                mode=resolved_mode,
            )

    def search_many(
        self,
        queries: list[str],
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        mode: str | None = None,
    ) -> list[list[LodeSearchHit]]:
        """Returns top-``k`` hits for each query, preserving query order.

        Batched search is the public SDK path that lets CUDA hosts use the
        optional GPU-resident TurboVec scan for eligible query batches. Single
        queries and unavailable GPU dependencies fall back to the compact CPU
        kernel; raw query text still never appears in telemetry. ``filter`` takes
        the same exact-match-or-predicate grammar as :meth:`search` and is applied
        identically to every query in the batch.

        ``mode`` matches :meth:`search` (unset resolves to ``"hybrid"`` when a
        lexical source is available, else ``"vector"``) and applies to every
        query in the batch; ``search_many(mode="hybrid")`` returns the same result
        as the corresponding repeated single :meth:`search` call. ``"hybrid"`` and
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
        with self._op_lock:
            return self._native_search_text_batch(
                queries, k=int(k), filter=normalized_filter, mode=resolved_mode
            )

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
        # Shares the vectorized prepare with search_many_by_vector so a single query
        # and a batched query normalize a vector identically.
        prepared = _prepare_vector_matrix([vector], self._vector_dim, normalize=normalize)[0]
        normalized_filter = _normalize_filter(filter)
        with self._op_lock:
            return self._native_search_by_vector(
                prepared, k=int(k), filter=normalized_filter
            )

    def search_many_by_vector(
        self,
        vectors: list[Sequence[float]] | np.ndarray,
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        normalize: bool = True,
    ) -> list[list[LodeSearchHit]]:
        """Returns top-``k`` hits for each precomputed query vector, preserving order.

        Batched vector-in search; like :meth:`search_many` it is the path that
        lets CUDA hosts use the GPU-resident scan for eligible batches. ``vectors``
        may be a list of per-query vectors or a ``(nq, dim)`` float array; passing a
        contiguous ``float32`` array is fastest (the batch crosses to the native
        core with no per-query Python). For raw scores and ids without building hit
        objects, see :meth:`search_many_by_vector_arrays`.
        """

        if k <= 0:
            raise ValueError("k must be positive")
        matrix = _prepare_vector_matrix(vectors, self._vector_dim, normalize=normalize)
        normalized_filter = _normalize_filter(filter)
        with self._op_lock:
            return self._native_search_many_by_vector(
                matrix, k=int(k), filter=normalized_filter
            )

    def search_many_by_vector_arrays(
        self,
        vectors: list[Sequence[float]] | np.ndarray,
        *,
        k: int = 10,
        filter: Mapping[str, Any] | None = None,
        normalize: bool = True,
        include_metadata: bool = False,
    ) -> tuple[np.ndarray, list[list[str]], int]:
        """Batched vector search returning flat arrays instead of hit objects.

        The throughput-oriented counterpart of :meth:`search_many_by_vector`:
        returns ``(scores, ids, k)`` where ``scores`` is a ``(nq, k)`` ``float32``
        array and ``ids`` is an ``nq``-long list of ``k`` document-id lists, without
        constructing a :class:`LodeSearchHit` per hit. ``k`` is the served width
        (``min(k, corpus)``, and ``min(k, allowlist)`` under a filter). When
        ``include_metadata`` is set, each hit's redacted metadata is returned too,
        as a fourth ``(nq, k)`` nested list. Use this when driving many queries and
        you only need scores and ids.
        """

        if k <= 0:
            raise ValueError("k must be positive")
        matrix = _prepare_vector_matrix(vectors, self._vector_dim, normalize=normalize)
        nq = int(matrix.shape[0])
        normalized_filter = _normalize_filter(filter)
        with self._op_lock:
            scores, ids_flat, metadata_flat, served_k = self._native_search_many_by_vector_arrays(
                matrix, k=int(k), filter=normalized_filter, want_metadata=include_metadata
            )
        served_k = int(served_k)
        scores2d = np.asarray(scores, dtype=np.float32).reshape(nq, served_k)

        def _nest(flat: list[Any]) -> list[list[Any]]:
            if not served_k:
                return [[] for _ in range(nq)]
            return [flat[base : base + served_k] for base in range(0, nq * served_k, served_k)]

        ids = _nest(ids_flat)
        if include_metadata:
            return scores2d, ids, _nest(metadata_flat), served_k  # type: ignore[return-value]
        return scores2d, ids, served_k

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
        with self._op_lock:
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

        return self.remove_many((id,)) > 0

    def remove_many(self, ids: Sequence[str]) -> int:
        """Removes documents by id in one mutation and returns the number deleted.

        The ids are sent to the native core as one batch, so an auto-persisting
        handle publishes one commit regardless of batch size. An empty batch is
        a no-op.
        """

        self._require_writable()
        document_ids = tuple(ids)
        if any(
            not isinstance(document_id, str) or not document_id.strip()
            for document_id in document_ids
        ):
            raise ValueError("ids must contain only non-empty strings")
        if not document_ids:
            return 0
        with self._op_lock:
            native_response = self._native_delete_documents(document_ids)
            return int(native_response.get("documents_deleted", 0) or 0)

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
        with self._op_lock:
            self._native_update_document_payload(
                id, metadata=metadata, text=text, clear_text=clear_text
            )

    def _native_update_document_payload(
        self,
        document_id: str,
        *,
        metadata: Mapping[str, Any] | None = None,
        text: str | None = None,
        clear_text: bool = False,
    ) -> None:
        """Applies a metadata / raw-text update to the native core (fail closed)."""

        try:
            self._native_vector_engine.update_document_payload(
                _LOCAL_INDEX_ID,
                document_id,
                metadata=_coerce_metadata(metadata) if metadata is not None else None,
                text=text,
                clear_text=clear_text,
            )
            if self._native_should_persist_after_mutation():
                self._native_vector_engine.persist()
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_payload_update_failed"
            raise RuntimeError("native core payload update failed") from exc

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
        with self._op_lock:
            return self._native_get_text(id)

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
        with self._op_lock:
            return self._native_get_texts(ids)

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
        with self._op_lock:
            native_record = self._native_get_document(id)
        return None if native_record is None else _public_document_record(native_record)

    def list_documents(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
        after: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Returns document records, optionally filtered and paged by a keyset cursor.

        Unlike :meth:`search`, this is enumeration, not ranking: it returns matching
        documents with no query vector and no scoring — the primitive a graph /
        knowledge-graph layer needs for deterministic traversal ("all edges whose
        ``src`` is X", "all nodes of ``type`` Person"). Each record is the
        payload-free ``{"id", "metadata", "chunk_count", "content_hash"}``.

        ``filter`` takes the same exact-match-or-predicate grammar as :meth:`search`
        (``$eq``/``$in``/``$gte``/``$and``/``$or``/… plus a ``document_ids``
        allowlist) and is resolved engine-side through the per-field planner in
        O(matches), not by scanning the corpus. ``after`` and ``limit`` page the
        (stable id-ordered) result set: pass the last id of a page as ``after`` and a
        page size as ``limit`` to stream large match sets without materializing them
        all at once. With neither, the complete match set is returned (no ``k`` cap).
        """

        normalized_filter = _normalize_filter(filter)
        with self._op_lock:
            records = self._native_list_documents(
                normalized_filter, after=after, limit=limit
            )
        return [_public_document_record(record) for record in records]

    def persist(self) -> dict[str, Any]:
        """Flushes durable on-disk state and returns redacted storage stats.

        The native core commits each mutation; this is an explicit durability +
        stats checkpoint. In the default ``commit_mode="wal"`` it folds the
        outstanding write-ahead log into a fresh committed generation (so the
        on-disk base is fully up to date and the WAL is empty); in
        ``commit_mode="generation"`` there is nothing buffered, so it only reports
        stats. State is reloaded automatically on the next ``LodeDB(path=...)`` open.
        """

        with self._op_lock:
            if not self.read_only:
                self._native_persist()
            return self._native_stats()

    def refresh(self) -> None:
        """Overlays the current write-ahead log tail into this handle's in-memory
        view without checkpointing.

        A ``read_only=True`` reader loads the last committed generation on open and
        is otherwise a stable snapshot; call this to fold in records other processes
        appended since (e.g. via :class:`~lodedb.Appender`), and to reach
        read-your-writes for an appended LSN (see :meth:`applied_lsn`). A no-op on a
        writable handle, which already folds the WAL when it opens.
        """

        with self._op_lock:
            self._native_vector_engine.refresh()

    def applied_lsn(self) -> int:
        """Returns the highest log sequence number reflected in this handle's view.

        Compare it to the LSN an :class:`~lodedb.Appender` returned for
        read-your-writes: the appended record is visible here once
        ``applied_lsn() >= that_lsn``. On a read-only handle call :meth:`refresh`
        first to fold the current WAL tail into the view.
        """

        with self._op_lock:
            return int(self._native_vector_engine.applied_lsn(_LOCAL_INDEX_ID))

    def count(self, *, filter: Mapping[str, Any] | None = None) -> int:
        """Returns the number of documents stored, optionally matching a filter.

        With ``filter`` (the same grammar as :meth:`search` / :meth:`list_documents`)
        returns the count of matching documents, resolved engine-side through the
        per-field planner in O(matches) without materializing any record.
        """

        with self._op_lock:
            if filter is None:
                return int(self._native_stats().get("document_count", 0) or 0)
            normalized_filter = _normalize_filter(filter)
            return len(self._native_list_documents(normalized_filter))

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

        with self._op_lock:
            stats = dict(self._native_stats())
        stats["image_embedding"] = {
            phase: dict(counters) for phase, counters in self._image_metrics.items()
        }
        stats["native_core"] = self._native_core_stats()
        stats["document_count"] = int(stats.get("document_count", 0) or 0)
        return stats

    def close(self) -> None:
        """Closes the native core (folding writes durably); state stays on disk."""

        self._shutdown_native(persist=True)

    def discard(self) -> None:
        """Closes the handle WITHOUT persisting; the writer lock is released.

        Un-persisted in-memory state is dropped and the store stays at its last
        committed state on disk. This is the abort path for a writable handle
        whose in-memory batch failed mid-apply (e.g. a partially applied fold
        segment): a graceful :meth:`close` would persist the poisoned state.
        WAL-mode writes are unaffected -- each was already durably logged at
        write time and replays on the next open. Idempotent, and equivalent to
        :meth:`close` for read-only handles.
        """

        self._shutdown_native(persist=False)

    def _shutdown_native(self, *, persist: bool) -> None:
        """Tears down the native engine on its home worker thread and stops the
        executor. ``persist=True`` closes the store (folding writes durably);
        ``persist=False`` discards un-persisted state. Idempotent."""

        with self._op_lock:
            executor = self._native_executor
            if executor is None:
                return
            # The native engine is unsendable: it must be closed AND dropped (final
            # decref) on its home worker thread. Detach the GC finalizer so it does
            # not double-close, drop the proxy's reference, then drain the holder on
            # the worker so the engine's last decref lands there. A close failure is
            # surfaced (after the worker is stopped) since it can mean durable writes
            # were not flushed.
            finalizer = getattr(self, "_native_finalizer", None)
            if finalizer is not None:
                finalizer.detach()
            holder = self._native_engine_holder
            home_thread_id = self._native_engine_thread_id
            self._native_vector_engine = None
            self._native_engine_holder = []  # type: ignore[assignment]
            self._native_executor = None
            self._native_vector_mutable = False
            self._native_vector_covered = False
            self._native_write_through_enabled = False
            on_worker = home_thread_id is not None and threading.get_ident() == home_thread_id

            def _close_on_worker() -> None:
                # Runs on the home worker: pop and close (or discard) the engine so
                # its final decref lands here, on the thread that created it.
                if holder:
                    engine = holder.pop()
                    try:
                        if persist:
                            engine.close()
                        else:
                            engine.discard()
                    finally:
                        # Drop the unsendable native handle here on the home
                        # worker even if close() raised: otherwise it stays
                        # reachable through the raised exception's traceback and
                        # is decref'd on the calling thread, which panics (the
                        # handle is thread-confined).
                        engine._engine = None

            native_close_error: Exception | None = None
            try:
                if on_worker:
                    _close_on_worker()
                else:
                    executor.submit(_close_on_worker).result()
            except Exception as exc:  # noqa: BLE001 - surfaced below
                native_close_error = exc
            executor.shutdown(wait=not on_worker)
        if native_close_error is not None:
            action = "close" if persist else "discard"
            raise RuntimeError(f"native core {action} failed") from native_close_error

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
        ann: dict[str, Any] | None = None,
    ) -> None:
        """Opens the native core engine on a dedicated worker thread.

        The native engine is the sole reader/writer for this handle, so it is
        always opened (the bundled extension is always present): a writable handle
        opens a write-through mutable engine and creates the local index if it does
        not already exist; a read-only handle opens a lock-free read-only engine.
        It is unsendable, so it is created on a single-worker executor and wrapped
        in :class:`_ThreadConfinedNativeEngine`, which routes every later call back
        to that worker. A failure raises (there is no Python fallback).
        """

        adapter = NativeCoreAdapter()
        if not adapter.available:
            self._native_core_fallback_reason = "native_core_extension_unavailable"
            raise RuntimeError("LodeDB requires the bundled native core (lodedb._native_core)")
        self._native_core_version = adapter.version
        self._native_core_abi_version = adapter.abi_version
        index_options = _native_vector_index_options(
            index_id=_LOCAL_INDEX_ID,
            index_key=index_state_key_for_client_hash(_LOCAL_CLIENT_ID_HASH, _LOCAL_INDEX_ID),
            client_id_hash=_LOCAL_CLIENT_ID_HASH,
            name="lodedb-local",
            model=model,
            provider=provider,
            task=task,
            route_profile=route_profile,
            storage_profile=storage_profile,
            vector_dim=self._vector_dim,
            bit_width=bit_width,
            ann=ann,
        )
        read_only = self.read_only

        def _finalize_open(handle: NativeCoreEngineHandle) -> NativeCoreEngineHandle:
            # Validates the persisted identity (and creates the index on a fresh
            # writable store). On any failure here the engine is already open, so
            # close + drop it on this worker thread before propagating: that
            # releases the writer lock and keeps the unsendable object from being
            # decref'd on the foreign thread that observes the error.
            try:
                if read_only:
                    # A read-only handle never creates state; reject only when the
                    # committed store on disk has an identity contradicting this
                    # handle (a missing index is an empty store and serves nothing).
                    _validate_native_index_identity(self.path, index_options)
                    return handle
                try:
                    handle.stats(_LOCAL_INDEX_ID)
                except Exception:
                    # Fresh store: create the index and commit an initial generation
                    # so a root manifest exists on disk. In WAL mode the native
                    # recovery path only replays a `<key>.wal` that sits alongside a
                    # committed `<key>.commit.json`; without this seed commit, a
                    # crash before the first checkpoint would leave an orphan WAL the
                    # next open could not discover.
                    handle.create_index_with_options(index_options)
                    handle.persist()
                else:
                    # The store already holds this index: enforce its full persisted
                    # identity against this handle's route before serving or
                    # mutating, so a reopen at a different model / dim / bit width
                    # fails fast.
                    _validate_native_index_identity(self.path, index_options)
                return handle
            except Exception:
                # The engine is open on this worker. Close it (releasing the writer
                # lock) and drop the unsendable PyCoreEngine HERE, on its home
                # thread, before the error propagates: a later decref of an object
                # captured by the exception traceback on the observing thread would
                # panic. Nulling the wrapper's inner engine forces that final
                # decref to happen now, on the worker.
                try:
                    handle.close()
                except Exception:
                    pass
                handle._engine = None  # type: ignore[assignment]
                del handle
                raise

        def _open() -> NativeCoreEngineHandle:
            # Runs on the worker thread: records its id as the engine's home thread
            # and opens (and, for a fresh writable store, creates the index in) the
            # native engine here so the unsendable handle is only ever touched here.
            self._native_engine_thread_id = threading.get_ident()
            if read_only:
                handle = adapter.open_readonly_engine(
                    self.path,
                    durability=durability,
                    commit_mode=commit_mode,
                    store_text=self.store_text,
                    index_text=self.index_text,
                    chunk_character_limit=self._chunk_character_limit,
                    compression=self.compression,
                )
                return _finalize_open(handle)
            if commit_mode != "wal" and _has_leftover_wal(self.path):
                # A crash in WAL mode can leave a `<key>.wal` with no fresh
                # checkpoint. The native recovery path only replays a WAL during a
                # WAL-mode open, so a reopen in generation mode would otherwise drop
                # those durably-logged writes. Fold the leftover WAL into a committed
                # generation with a transient WAL-mode open (replay + persist on
                # close) before opening in the requested mode.
                recovery = adapter.open_engine(
                    path=self.path,
                    read_only=False,
                    durability=durability,
                    commit_mode="wal",
                    store_text=self.store_text,
                    index_text=self.index_text,
                    chunk_character_limit=self._chunk_character_limit,
                    compression=self.compression,
                    acquire_writer_lock=True,
                )
                try:
                    recovery.close()  # persists the replayed WAL into a generation
                finally:
                    recovery._engine = None  # type: ignore[assignment]
                    del recovery
            # The native engine is the sole writer for this handle, so it takes
            # the shared <dir>/.lodedb.lock single-writer lock itself.
            handle = adapter.open_engine(
                path=self.path,
                read_only=False,
                durability=durability,
                commit_mode=commit_mode,
                store_text=self.store_text,
                index_text=self.index_text,
                chunk_character_limit=self._chunk_character_limit,
                compression=self.compression,
                acquire_writer_lock=True,
            )
            return _finalize_open(handle)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lodedb-native")
        try:
            real_handle = executor.submit(_open).result()
        except Exception as exc:
            executor.shutdown(wait=True)
            if _is_writer_lock_contention(exc):
                # Another process holds the native single-writer lock. Surface the
                # SDK's stable single-writer error (LodeDB is single-writer per
                # path) rather than the generic init failure.
                self._native_core_fallback_reason = "native_core_writer_lock_contended"
                raise ConcurrentWriterError(
                    f"LodeDB at {self.path} is already open by another process "
                    "(LodeDB is single-writer per path); close the other handle, or "
                    "raise LODEDB_PERSIST_LOCK_TIMEOUT to wait longer."
                ) from exc
            self._native_core_fallback_reason = "native_core_init_failed"
            raise RuntimeError(f"failed to initialize native core: {exc!r}") from exc
        self._native_executor = executor
        # The unsendable engine must be closed AND dropped (final decref) on its
        # home worker; a decref on any other thread panics. Both the proxy and the
        # GC finalizer reach the engine only through this one-element holder, so the
        # worker-side teardown can pop and drop it there, leaving the holder empty
        # before any other reference (proxy slot, finalizer arg) is released on the
        # collecting thread. _ThreadConfinedNativeEngine reads holder[0] per call.
        self._native_engine_holder: list[NativeCoreEngineHandle] = [real_handle]
        self._native_vector_engine = _ThreadConfinedNativeEngine(
            executor, self._native_engine_thread_id, self._native_engine_holder
        )
        self._native_vector_mutable = not read_only
        self._native_vector_covered = True
        self._native_write_through_enabled = not read_only
        # Drop the native engine on its home worker when this handle is garbage
        # collected without close(); the unsendable engine must be dropped there.
        self._native_finalizer = weakref.finalize(
            self,
            _shutdown_native_engine,
            executor,
            self._native_engine_holder,
            self._native_engine_thread_id,
        )
        # Also close on its home worker at interpreter exit if the caller never
        # does: the GC finalizer above runs too late (after the worker is joined),
        # so without this an un-closed handle drops on the finalizing thread and
        # PyO3 writes a spurious "dropped on another thread" unraisable.
        _OPEN_NATIVE_DBS.add(self)
        _register_early_native_atexit()

    def _native_upsert_vectors(self, documents: tuple[EngineVectorDocument, ...]) -> None:
        """Upserts vector documents into the native core (fail closed)."""

        if not documents:
            return
        try:
            self._native_vector_engine.upsert_vectors(_LOCAL_INDEX_ID, documents)
            if self._native_should_persist_after_mutation():
                self._native_vector_engine.persist()
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_upsert_failed"
            raise RuntimeError("native core vector upsert failed") from exc

    def _native_upsert_vectors_matrix(
        self,
        matrix: np.ndarray,
        sidecar: list[Mapping[str, Any]],
    ) -> None:
        """Upserts a prepared f32 matrix plus a per-doc sidecar (fail closed).

        The zero-round-trip counterpart of :meth:`_native_upsert_vectors`: the
        normalized ``(n, dim)`` matrix reaches the native array upsert without
        becoming per-vector Python tuples. ``sidecar`` is the row-ordered
        ``[{document_id, metadata, text}]`` list.
        """

        if not sidecar:
            return
        try:
            self._native_vector_engine.upsert_vectors_matrix(_LOCAL_INDEX_ID, matrix, sidecar)
            if self._native_should_persist_after_mutation():
                self._native_vector_engine.persist()
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_upsert_failed"
            raise RuntimeError("native core vector upsert failed") from exc

    def _native_upsert_text_documents(self, documents: tuple[EngineDocument, ...]) -> None:
        """Embeds and upserts text documents through the native core (fail closed).

        The native core plans the chunking, Python embeds the chunks it asks for,
        and the native core applies the embedded plan and (in generation mode)
        publishes the durable generation.
        """

        if not documents:
            return
        if self._embedding_backend is None:
            raise RuntimeError("native text upsert requires an embedding backend")
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
            self._native_core_fallback_reason = "native_core_text_upsert_failed"
            raise RuntimeError("native core text upsert failed") from exc

    def _native_delete_documents(self, document_ids: tuple[str, ...]) -> dict[str, Any]:
        """Deletes documents in the native core and returns its response (fail closed)."""

        if not document_ids:
            return {"documents_deleted": 0}
        try:
            response = self._native_vector_engine.delete_documents(_LOCAL_INDEX_ID, document_ids)
            if self._native_should_persist_after_mutation():
                self._native_vector_engine.persist()
            return response
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_delete_failed"
            raise RuntimeError("native core vector delete failed") from exc

    def _native_search_by_vector(
        self,
        vector: Sequence[float],
        *,
        k: int,
        filter: Mapping[str, Any] | None,
    ) -> list[LodeSearchHit]:
        """Returns native vector hits for a precomputed query vector (fail closed)."""

        try:
            payload = self._native_vector_engine.query_vector(
                _LOCAL_INDEX_ID,
                vector,
                top_k=k,
                filter=filter,
            )
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_query_failed"
            raise RuntimeError("native core vector query failed") from exc
        return self._hits_from_native_rows(payload.get("hits", []))

    def _native_search_many_by_vector(
        self,
        vectors: list[Sequence[float]],
        *,
        k: int,
        filter: Mapping[str, Any] | None,
    ) -> list[list[LodeSearchHit]]:
        """Returns native vector batch hits for a query batch (fail closed)."""

        try:
            # Near-zero-copy path: flat query matrix in, arrays out (scores, ids,
            # batched metadata) with no per-hit JSON. Optional on the engine, so it
            # falls back to the JSON batch query for older extensions and minimal
            # engine stubs that only implement query_vectors_batch.
            arrays_query = getattr(
                self._native_vector_engine, "query_vectors_batch_arrays", None
            )
            if callable(arrays_query):
                arrays = arrays_query(_LOCAL_INDEX_ID, vectors, top_k=k, filter=filter)
                if arrays is not None:
                    return self._hits_from_native_arrays(*arrays, query_count=len(vectors))
            batches = self._native_vector_engine.query_vectors_batch(
                _LOCAL_INDEX_ID,
                vectors,
                top_k=k,
                filter=filter,
            )
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_batch_query_failed"
            raise RuntimeError("native core vector batch query failed") from exc
        return [self._hits_from_native_rows(batch.get("hits", [])) for batch in batches]

    def _native_search_many_by_vector_arrays(
        self,
        matrix: np.ndarray,
        *,
        k: int,
        filter: Mapping[str, Any] | None,
        want_metadata: bool,
    ) -> tuple[Any, list[str], list[Mapping[str, Any]], int]:
        """Returns flat native arrays ``(scores, ids, metadata, k)`` for a query batch.

        The un-hydrated counterpart of :meth:`_native_search_many_by_vector`: the
        near-zero-copy arrays cross the boundary and are returned as-is, so the
        caller can hand back scores and ids without a per-hit object. Requires the
        native array-out path (present in the bundled extension); fails closed.
        """

        arrays_query = getattr(self._native_vector_engine, "query_vectors_batch_arrays", None)
        if not callable(arrays_query):
            raise RuntimeError("native core lacks the arrays vector-batch query")
        try:
            arrays = arrays_query(
                _LOCAL_INDEX_ID, matrix, top_k=k, filter=filter, want_metadata=want_metadata
            )
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_batch_query_failed"
            raise RuntimeError("native core vector batch query failed") from exc
        if arrays is None:
            raise RuntimeError("native core arrays vector-batch query is unavailable")
        scores, document_ids, metadata, served_k = arrays
        return scores, document_ids, metadata, served_k

    def _native_search_text(
        self,
        query: str,
        *,
        k: int,
        filter: Mapping[str, Any] | None,
        mode: str,
    ) -> list[LodeSearchHit]:
        """Returns native hits for a text/hybrid/lexical query (fail closed)."""

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
            self._native_core_fallback_reason = "native_core_text_query_failed"
            raise RuntimeError("native core text query failed") from exc
        return self._hits_from_native_rows(payload.get("hits", []))

    def _native_search_text_batch(
        self,
        queries: list[str],
        *,
        k: int,
        filter: Mapping[str, Any] | None,
        mode: str,
    ) -> list[list[LodeSearchHit]]:
        """Returns native hits for a batch of text queries (fail closed).

        Mirrors :meth:`_native_search_text` but scores the whole batch through one
        shared native scan (GPU-eligible), embedding each query in Python first.
        Used by :meth:`search_many`.
        """

        try:
            query_embeddings = None
            if mode in {"vector", "hybrid"}:
                if self._embedding_backend is None:
                    raise RuntimeError("native text query requires an embedding backend")
                query_embeddings = [
                    self._embedding_backend.embed_query(query) for query in queries
                ]
            payloads = self._native_vector_engine.search_text_batch(
                _LOCAL_INDEX_ID,
                queries,
                mode,
                query_embeddings,
                top_k=k,
                filter=filter,
            )
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_text_batch_query_failed"
            raise RuntimeError("native core text batch query failed") from exc
        return [self._hits_from_native_rows(payload.get("hits", [])) for payload in payloads]

    def _native_get_text(self, document_id: str) -> str | None:
        """Returns one native raw-text value, or ``None`` if absent (fail closed)."""

        try:
            return self._native_vector_engine.get_document_text(_LOCAL_INDEX_ID, document_id)
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_get_text_failed"
            raise RuntimeError("native core document read failed") from exc

    def _native_get_texts(self, document_ids: list[str]) -> dict[str, str]:
        """Returns a ``{id: text}`` map for the stored ids that have text (fail closed)."""

        try:
            return self._native_vector_engine.get_document_texts(_LOCAL_INDEX_ID, document_ids)
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_get_texts_failed"
            raise RuntimeError("native core document read failed") from exc

    def _native_get_document(self, document_id: str) -> dict[str, Any] | None:
        """Returns one native payload-free document record, or ``None`` (fail closed)."""

        try:
            return self._native_vector_engine.get_document(_LOCAL_INDEX_ID, document_id)
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_get_document_failed"
            raise RuntimeError("native core document read failed") from exc

    def _native_list_documents(
        self,
        filter: Mapping[str, Any] | None,
        *,
        after: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Returns native payload-free document records, optionally paged (fail closed)."""

        try:
            return self._native_vector_engine.list_documents(
                _LOCAL_INDEX_ID,
                filter=filter,
                after=after,
                limit=limit,
            )
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_list_documents_failed"
            raise RuntimeError("native core document read failed") from exc

    def _native_stats(self) -> dict[str, Any]:
        """Returns native index stats (fail closed)."""

        try:
            return self._native_vector_engine.stats(_LOCAL_INDEX_ID)
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_stats_failed"
            raise RuntimeError("native core stats failed") from exc

    def _native_core_stats(self) -> dict[str, Any]:
        """Returns redacted native-core status for this handle."""

        try:
            stats = self._native_stats()
        except Exception:  # noqa: BLE001 - status report must never raise
            stats = {}
        return {
            "mode": "on",
            "write_mode": "on" if self._native_write_through_enabled else "off",
            "version": self._native_core_version,
            "abi_version": self._native_core_abi_version,
            "enabled": self._native_vector_engine is not None,
            "covered": self._native_vector_covered,
            "fallback_reason": self._native_core_fallback_reason,
            "document_count": int(stats.get("document_count", 0) or 0),
            "write_through": self._native_write_through_enabled,
        }

    def _native_persist(self) -> None:
        """Folds the native core's outstanding writes into a durable generation."""

        if not self._native_vector_mutable:
            return
        try:
            self._native_vector_engine.persist()
        except Exception as exc:
            self._native_core_fallback_reason = "native_core_write_persist_failed"
            raise RuntimeError("native core persist failed") from exc

    def _native_should_persist_after_mutation(self) -> bool:
        """Whether native must publish a generation immediately after a mutation.

        Generation mode only: there a per-add generation publish IS the durability.
        In WAL mode the native upsert already appended a durable WAL record, so a
        per-add ``persist()`` would only republish a full generation redundantly
        (the real cost behind native's slow per-add WAL durability); the generation
        publish is deferred to checkpoint/close instead.
        """

        return self._native_write_through_enabled and self.commit_mode.value == "generation"

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

    @staticmethod
    def _hits_from_native_arrays(
        scores: Any,
        document_ids: list[str],
        metadata: list[Mapping[str, Any]],
        k: int,
        *,
        query_count: int,
    ) -> list[list[LodeSearchHit]]:
        """Builds hit rows from flat ``[query_count * k]`` native arrays.

        The near-zero-copy counterpart of :meth:`_hits_from_native_rows`: scores
        arrive as a numpy array, ids as a string list, and metadata as a dict list,
        all flat and row-major by query, so no per-hit JSON object is parsed.
        """

        if k == 0:
            return [[] for _ in range(query_count)]
        scores_list = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        metadata_len = len(metadata)
        rows: list[list[LodeSearchHit]] = []
        for query_index in range(query_count):
            base = query_index * k
            row = [
                LodeSearchHit(
                    score=float(scores_list[base + offset]),
                    id=str(document_ids[base + offset]),
                    metadata=dict(metadata[base + offset])
                    if base + offset < metadata_len
                    else {},
                )
                for offset in range(k)
            ]
            rows.append(row)
        return rows

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

    def _resolve_mode(self, mode: str | None) -> str:
        """Validates a search mode and enforces the lexical-source requirement.

        ``None`` (the search default) resolves to ``"hybrid"`` when a lexical
        source is available (``index_text=True`` or ``store_text=True``) and to
        ``"vector"`` otherwise, so the fused ranking is the default wherever it
        can run and a vector-only store never raises on an unset mode.

        Returns the canonical lowercase mode. Raises :class:`ValueError` for an
        unknown mode, or when a lexical/hybrid mode is requested *explicitly* on a
        handle that has no lexical source — neither ``index_text=True`` (a
        persisted BM25 postings store) nor ``store_text=True`` (the BM25 index
        rebuilt from the retained raw text), so there is nothing to build the
        index from.
        """

        if mode is None:
            return "hybrid" if (self.index_text or self.store_text) else "vector"
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

    def _next_auto_id(self) -> str:
        """Returns a unique, collision-resistant auto id for an added document."""

        self._auto_id_counter += 1
        return f"doc-{secrets.token_hex(8)}-{self._auto_id_counter}"


# The native lock contention errors, raised by the Rust core when a hold on
# <dir>/.lodedb.lock cannot be acquired before the timeout elapses: an exclusive
# acquire reports "another writer holds the lodedb lock", a shared (appender)
# acquire reports "an exclusive writer holds the lodedb lock". The native binding
# surfaces both as ValueError; the shared suffix matches either.
_NATIVE_WRITER_LOCK_CONTENTION_MARKER = "holds the lodedb lock"


def _is_writer_lock_contention(error: BaseException) -> bool:
    """Returns whether a native open failed because another holder has the lock."""

    return _NATIVE_WRITER_LOCK_CONTENTION_MARKER in str(error)


# Engines we could not drop on their home worker (interpreter teardown, or a
# worker that did not answer in time) are parked here so their final decref never
# lands on a foreign thread (the unsendable PyCoreEngine panics if it does). The
# OS reclaims them at process exit; this only prevents a spurious teardown panic.
_LEAKED_NATIVE_ENGINES: list[NativeCoreEngineHandle] = []


# Open handles whose unsendable native engine still needs dropping on its home
# worker. The per-handle GC finalizer (_shutdown_native_engine) runs from weakref's
# atexit callback, which fires *after* concurrent.futures has already joined the
# worker threads; by then the worker is gone and the engine's final decref lands on
# the finalizing thread, which makes PyO3 write an "is being dropped on another
# thread" unraisable (and skip the drop). Closing them from a hook that runs *before*
# that worker join gives each engine a clean worker-side drop and no teardown noise.
_OPEN_NATIVE_DBS: weakref.WeakSet[LodeDB] = weakref.WeakSet()
_EARLY_ATEXIT_REGISTERED = False


def _close_open_native_dbs_at_exit() -> None:
    """Closes every still-open native engine on its home worker before teardown.

    Runs before ``concurrent.futures`` joins its worker threads, so each engine's
    final decref lands on the worker that created it. ``close()`` is idempotent and
    detaches the GC finalizer, so a handle the caller already closed is a no-op.
    """

    for db in list(_OPEN_NATIVE_DBS):
        try:
            db.close()
        except Exception:  # noqa: BLE001 - teardown is best effort; never raise at exit
            pass


def _register_early_native_atexit() -> None:
    """Arms the pre-worker-join teardown hook once, on the first native open.

    Registered lazily (after the ``concurrent.futures`` import that arms its own
    worker-join hook) so the LIFO atexit order runs ours first. Prefers
    ``threading._register_atexit`` (runs before the worker join, where a clean drop
    is still possible); falls back to ``atexit.register`` on the rare interpreter
    without it (there the worker is already gone by teardown, so the per-handle
    finalizer's park is the safety net instead).
    """

    global _EARLY_ATEXIT_REGISTERED
    if _EARLY_ATEXIT_REGISTERED:
        return
    _EARLY_ATEXIT_REGISTERED = True
    register = getattr(threading, "_register_atexit", None)
    if register is not None:
        try:
            register(_close_open_native_dbs_at_exit)
            return
        except RuntimeError:
            # Interpreter already finalizing; fall through to a best-effort atexit.
            pass
    atexit.register(_close_open_native_dbs_at_exit)


def _shutdown_native_engine(
    executor: ThreadPoolExecutor,
    holder: list[NativeCoreEngineHandle],
    home_thread_id: int | None,
) -> None:
    """Drops the unsendable native engine on its home worker thread (GC fallback).

    The GC finalizer for a handle the caller never closed. It drops the engine
    without a clean ``close()`` (no WAL fold): each mutation is already durable on
    its own (a generation publish in generation mode, a WAL record in WAL mode), so
    a leftover WAL is simply replayed on the next open. The engine is reached
    through ``holder`` (also captured by the finalizer): the worker task pops it out
    and drops it there, so when this function returns the holder is empty and no
    unsendable object remains for the collecting thread to drop.

    During interpreter shutdown the executor's own atexit hook is joining the worker
    threads, so submitting work can hang or has already been refused; we then park
    the engine in :data:`_LEAKED_NATIVE_ENGINES` rather than risk a wrong-thread
    drop (the process is exiting, so the OS reclaims it). A normal-GC worker drop
    uses a bounded wait and falls back to the same park on timeout.
    """

    if not holder:
        return
    on_worker = home_thread_id is not None and threading.get_ident() == home_thread_id

    def _drop_on_worker() -> None:
        # Runs on the home worker: take the engine out of the shared holder so its
        # final decref lands here, on the thread that created it.
        if holder:
            holder.pop()

    if on_worker:
        _drop_on_worker()
        executor.shutdown(wait=False)
        return
    if home_thread_id is None:
        _LEAKED_NATIVE_ENGINES.extend(holder)
        holder.clear()
        return
    if sys.is_finalizing():
        # Interpreter teardown: the executor's own atexit hook is joining the worker,
        # so a blocking wait can hang. Hand the drop to the worker fire-and-forget
        # (it usually drains before being joined, dropping on the right thread); if
        # the worker is already gone, submit raises and we park the engine so it is
        # never decref'd on this foreign thread.
        try:
            executor.submit(_drop_on_worker)
        except Exception:  # noqa: BLE001 - worker already shut down
            _LEAKED_NATIVE_ENGINES.extend(holder)
            holder.clear()
        return
    try:
        executor.submit(_drop_on_worker).result(timeout=10.0)
    except Exception:  # noqa: BLE001 - worker dead/stuck; park instead of dropping here
        _LEAKED_NATIVE_ENGINES.extend(holder)
        holder.clear()
    executor.shutdown(wait=False)


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


@dataclass(frozen=True)
class AnnOptions:
    """Structured ANN tuning, mirroring the native core's ``CoreAnnOptions``.

    Pass to ``LodeDB(ann=...)`` (or ``open_vector_store(ann=...)``) as an
    alternative to the loose ``ann=``/``ann_clusters=``/``ann_nprobe=`` keyword
    arguments::

        LodeDB(path, ann=AnnOptions(clusters=256, nprobe=16))

    The native core is the authority on which algorithms and knobs exist and
    validates them, so this type just carries the shape (unset knobs fall back to
    the core's corpus-derived defaults). Keeping it in one place means a new knob
    is added here once rather than as another loose keyword argument re-validated
    in Python.
    """

    algorithm: str = "cluster"
    clusters: int | None = None
    nprobe: int | None = None

    def to_core_dict(self) -> dict[str, Any]:
        """Renders the native-core ``ann`` option dict, omitting unset knobs."""

        options: dict[str, Any] = {"algorithm": str(self.algorithm)}
        if self.clusters is not None:
            options["clusters"] = int(self.clusters)
        if self.nprobe is not None:
            options["nprobe"] = int(self.nprobe)
        return options


def _resolve_ann_options(
    ann: str | AnnOptions | None,
    clusters: int | None,
    nprobe: int | None,
) -> dict[str, Any] | None:
    """Builds the native-core ``ann`` option dict, or ``None`` for exact scan.

    ``ann`` opts into ANN: an :class:`AnnOptions` (the structured form) or the
    algorithm string (``"cluster"`` today), with ``clusters``/``nprobe`` as loose
    tuning for the string form. The native core is the authority on which
    algorithms and knobs exist and validates their values; the structured form
    defers to it, and only the loose-keyword shape (tuning without ``ann=``,
    non-positive counts) is still checked here for a friendly pre-FFI error.
    """

    if isinstance(ann, AnnOptions):
        # The structured form carries its own tuning, so the loose knobs must not
        # also be set (which would be ambiguous). Values are the core's to validate.
        if clusters is not None or nprobe is not None:
            raise ValueError(
                "pass ANN tuning via AnnOptions(...) or ann_clusters/ann_nprobe, not both"
            )
        return ann.to_core_dict()
    if ann is None:
        if clusters is not None or nprobe is not None:
            raise ValueError("ann_clusters/ann_nprobe require ann= to be set")
        return None
    options: dict[str, Any] = {"algorithm": str(ann)}
    if clusters is not None:
        if int(clusters) < 1:
            raise ValueError("ann_clusters must be a positive integer")
        options["clusters"] = int(clusters)
    if nprobe is not None:
        if int(nprobe) < 1:
            raise ValueError("ann_nprobe must be a positive integer")
        options["nprobe"] = int(nprobe)
    return options


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
    ann: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds the native-core index creation payload for the local vector store."""

    options: dict[str, Any] = {
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
    # Omit the key entirely when exact, so the create payload (and the persisted
    # state header) stays byte-for-byte unchanged for non-ANN indexes.
    if ann is not None:
        options["ann"] = ann
    return options


def _has_leftover_wal(path: Path) -> bool:
    """Returns whether a non-empty WAL log is present (a crash before checkpoint)."""

    return any(p.is_file() and p.stat().st_size > 0 for p in Path(path).glob("*.wal"))


def _persisted_index_identity(path: Path) -> dict[str, Any] | None:
    """Reads the committed index identity (model/dim/etc.) for the local store.

    Returns the persisted route identity from the consistent generation named by
    the ``<key>.commit.json`` root manifest (the same on-disk source the snapshot
    auditor reads), or ``None`` when the store has no committed index yet. Only the
    redacted state header is read; journal deltas never change the identity.
    """

    commits = sorted(Path(path).glob(f"*{COMMIT_MANIFEST_SUFFIX}"))
    for commit_path in commits:
        manifest = read_commit_manifest(commit_path)
        if manifest is None:
            continue
        key = commit_path.name[: -len(COMMIT_MANIFEST_SUFFIX)]
        base = base_json_path(Path(path), key, int(manifest["base_epoch"]))
        try:
            payload = json.loads(base.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        state = _state_from_payload(payload)
        return {
            "native_dim": int(state.native_dim),
            "model": str(state.model),
            "provider": str(state.provider),
            "task": str(state.task),
            "storage_profile": str(state.storage_profile),
            "turbovec_bit_width": int(state.turbovec_bit_width),
        }
    return None


def _validate_native_index_identity(path: Path, options: Mapping[str, Any]) -> None:
    """Rejects a reopen whose persisted index identity contradicts this handle.

    Enforces the full persisted route identity (dimension, model, provider, task,
    storage profile, and TurboVec bit width) against the identity this handle was
    opened with. Checking only model/dim is not enough: a vector-only store and a
    custom-embedder index can share model/provider/dim yet differ in task, and a
    reopen at a different bit width must fail rather than silently keep the stored
    width. A store with no committed generation yet has no identity to contradict.
    """

    persisted = _persisted_index_identity(path)
    if persisted is None:
        return
    for label, persisted_value, expected_value in (
        ("dimension", persisted["native_dim"], int(options["vector_dim"])),
        ("model", persisted["model"], str(options["model"])),
        ("provider", persisted["provider"], str(options["provider"])),
        ("task", persisted["task"], str(options["task"])),
        ("storage profile", persisted["storage_profile"], str(options["storage_profile"])),
        ("bit_width", persisted["turbovec_bit_width"], int(options["bit_width"])),
    ):
        if persisted_value != expected_value:
            raise RuntimeError(
                f"persisted index {label} {persisted_value!r} does not match the opened "
                f"index {label} {expected_value!r}; reopen with the same model / embedder "
                "/ vector_dim / bit_width it was written with"
            )


def _coerce_optional_text(value: Any) -> str | None:
    """Validates optional retained text on vector-in document mappings."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("vector document 'text' must be a string when provided")
    return value


def _finite_float_vector(
    vector: Sequence[float],
    *,
    normalize: bool,
    dim: int | None = None,
) -> list[float]:
    """Validates a precomputed embedding and (optionally) L2-normalizes it.

    The shared vector-boundary check for both :meth:`LodeDB.add_vectors` and
    :class:`~lodedb.local.appender.Appender`: coerce to Python ``float`` (float64),
    reject non-finite values, and -- when ``normalize`` is set -- scale to unit
    norm so cosine scores match the text path (which normalizes on write and
    query). ``math.hypot`` scales internally, so a large-but-finite component
    (whose square would overflow float64) cannot push the norm to inf and silently
    zero the vector. When ``dim`` is given the length must match it; otherwise the
    vector must merely be non-empty (the appender validates the dimension in the
    native core against the store's shape).
    """

    try:
        values = [float(component) for component in vector]
    except (TypeError, ValueError) as exc:
        raise ValueError("vector must be a sequence of numbers") from exc
    if dim is not None:
        if len(values) != dim:
            raise ValueError(f"vector must have dimension {dim}, got {len(values)}")
    elif not values:
        raise ValueError("vector must be non-empty")
    if not all(math.isfinite(component) for component in values):
        raise ValueError("vector must contain only finite values")
    if normalize:
        norm = math.hypot(*values)
        if norm == 0.0:
            raise ValueError(
                "cannot normalize a zero vector; pass normalize=False to store it as-is"
            )
        if not math.isfinite(norm):
            raise ValueError("vector norm overflows; scale the vector down first")
        values = [component / norm for component in values]
    return values


def _prepare_vector(
    vector: Sequence[float],
    dim: int,
    *,
    normalize: bool,
) -> tuple[float, ...]:
    """Validates a precomputed embedding, (optionally) L2-normalizes it, and
    enforces the index dimension at the SDK boundary. See
    :func:`_finite_float_vector`."""

    return tuple(_finite_float_vector(vector, normalize=normalize, dim=dim))


def _prepare_vector_matrix(
    vectors: Any,
    dim: int,
    *,
    normalize: bool,
) -> np.ndarray:
    """Validates and (optionally) L2-normalizes a batch of vectors as one f32 matrix.

    The vectorized boundary for both the batched vector-in search path and the
    vector-in write path: accepts a ``(n, dim)`` float array or any sequence of
    per-row vectors, and returns a C-contiguous ``float32`` ``(n, dim)`` matrix with
    no per-row Python. The common case stays in float32; the norm is accumulated in
    float64 (never overflows for float32-range inputs) so normalization matches the
    scalar path within ~1 ulp -- ranking-neutral for queries, and single- and
    batch-vector writes share this path so a store stays internally consistent. A
    component beyond float32 range (finite only in float64) drops to a float64 path
    that scales by max-abs before squaring, the same guard :func:`_finite_float_vector`
    gives per vector. The native core re-validates finiteness, stricter.
    """

    try:
        # A component beyond float32 range overflows to inf here; that is expected
        # and handled by the finiteness fallback below, so silence the warning.
        with np.errstate(over="ignore"):
            matrix = np.ascontiguousarray(vectors, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("vectors must be sequences of numbers") from exc
    if matrix.size == 0:
        raise ValueError("vectors must be a non-empty batch")
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise ValueError(f"each vector must have dimension {dim}")
    if not np.isfinite(matrix).all():
        # A value is non-finite, or was finite but exceeded float32 range and became
        # inf on the cast; re-read in float64 to tell them apart.
        wide = np.asarray(vectors, dtype=np.float64)
        if not np.isfinite(wide).all():
            raise ValueError("vector must contain only finite values")
        # Large-but-finite: normalize in float64, or (no normalize) let the native
        # core reject the out-of-range value exactly as the scalar path did.
        if normalize:
            return _l2_normalize_wide(wide)
        return np.ascontiguousarray(wide, dtype=np.float32)
    if normalize:
        norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix, dtype=np.float64))
        if (norms == 0.0).any():
            # A row is zero in float32: either a genuine zero vector or a tiny row
            # (e.g. ~1e-300) that underflowed on the cast. Re-normalize the original
            # in float64, which preserves tiny rows via max-abs scaling and raises
            # only for a true zero vector.
            return _l2_normalize_wide(np.asarray(vectors, dtype=np.float64))
        return np.ascontiguousarray(matrix / norms[:, None], dtype=np.float32)
    return matrix


def _l2_normalize_wide(wide: np.ndarray) -> np.ndarray:
    """L2-normalizes a float64 matrix, scaling each row by its max abs before
    squaring so a large-but-finite component cannot overflow the norm to inf.
    Returns a C-contiguous float32 matrix. See :func:`_prepare_vector_matrix`."""

    scale = np.abs(wide).max(axis=1)
    if (scale == 0.0).any():
        raise ValueError(
            "cannot normalize a zero vector; pass normalize=False to store it as-is"
        )
    scaled = wide / scale[:, None]
    unit = scaled / np.sqrt(np.square(scaled).sum(axis=1))[:, None]
    return np.ascontiguousarray(unit, dtype=np.float32)
