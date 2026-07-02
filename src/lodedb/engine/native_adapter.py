"""Private Python compatibility adapter for the native core.

This module is intentionally not part of the public API. It keeps the current
Python engine dataclasses as the oracle-facing shape and translates them to the
native-core JSON contracts used by the hidden ``_native_core`` extension. The
extension is imported lazily so ``import lodedb`` remains dependency-light and a
missing native module is a rollout decision, not an import-time failure.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from os import PathLike
from typing import Any, Protocol

from lodedb.engine.core import EngineDocument, EngineQuery, EngineResponse, EngineVectorDocument


def _dumps(payload: Any) -> str:
    """Serializes a payload to the compact, key-sorted JSON the native core expects.

    Shared by the adapter and its engine/appender handles so every FFI call and
    every checksummed body encodes byte-identically (the native and Python engines
    verify each other's writes, so the encoding must match).
    """

    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class NativeCorePayload(dict[str, Any]):
    """Native JSON-backed dict for private adapter round-trips."""

    def __init__(self, payload: Mapping[str, Any], *, native_json: str) -> None:
        super().__init__(payload)
        self.native_json = native_json


class NativeCoreModule(Protocol):
    """Subset of the hidden native module used by the adapter."""

    def CoreEngine(self) -> Any: ...
    def CoreAppender(self) -> Any: ...
    def cuda_runtime_available(self) -> bool: ...
    def native_core_abi_version(self) -> int: ...
    def native_core_version(self) -> str: ...
    def round_trip_core_json(self, type_name: str, json_payload: str) -> str: ...


@dataclass(frozen=True)
class NativeCoreError:
    """Stable native error mapped into the endpoint-shaped Python response."""

    code: str
    message: str


class NativeCoreAdapter:
    """Maps Python engine dataclasses to hidden native-core JSON contracts."""

    def __init__(self, native_module: NativeCoreModule | None = None) -> None:
        self._native_module = native_module

    @property
    def available(self) -> bool:
        return self._module_or_none() is not None

    @property
    def version(self) -> str:
        module = self._module_or_none()
        if module is None:
            return ""
        return str(module.native_core_version())

    @property
    def abi_version(self) -> int:
        module = self._module_or_none()
        if module is None:
            return 0
        return int(module.native_core_abi_version())

    def cuda_runtime_available(self) -> bool:
        """Returns whether the bundled native core can run the GPU-resident scan.

        Probes the real CUDA driver the cudarc scan gates on (not torch or CuPy);
        returns False when the native extension is unavailable.
        """

        module = self._module_or_none()
        if module is None:
            return False
        return bool(module.cuda_runtime_available())

    def document_json(self, document: EngineDocument) -> str:
        return json.dumps(self.document_payload(document), sort_keys=True, separators=(",", ":"))

    def vector_document_json(self, document: EngineVectorDocument) -> str:
        return json.dumps(
            self.vector_document_payload(document), sort_keys=True, separators=(",", ":")
        )

    def query_json(self, query: EngineQuery) -> str:
        return json.dumps(self.query_payload(query), sort_keys=True, separators=(",", ":"))

    def round_trip(self, type_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        module = self._require_module()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return json.loads(module.round_trip_core_json(type_name, encoded))

    def new_engine(self) -> NativeCoreEngineHandle:
        """Creates an in-memory native engine handle through the hidden extension."""

        module = self._require_module()
        return NativeCoreEngineHandle(module.CoreEngine())

    def open_engine(
        self,
        *,
        path: str | PathLike[str],
        read_only: bool,
        durability: str,
        commit_mode: str,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: int,
        compression: bool = True,
        acquire_writer_lock: bool = True,
    ) -> NativeCoreEngineHandle:
        """Opens a persistent native engine handle through the hidden extension.

        The native engine is the sole writer for a LodeDB handle, so a writable
        open takes the shared ``<dir>/.lodedb.lock`` single-writer lock by
        default (``acquire_writer_lock=True``); pass ``False`` only when an outer
        caller already holds that lock for the process.
        """

        module = self._require_module()
        options = self.open_options_payload(
            path=path,
            read_only=read_only,
            durability=durability,
            commit_mode=commit_mode,
            store_text=store_text,
            index_text=index_text,
            chunk_character_limit=chunk_character_limit,
            compression=compression,
            acquire_writer_lock=acquire_writer_lock,
        )
        return NativeCoreEngineHandle(module.CoreEngine.open(_dumps(options)))

    def open_readonly_engine(
        self,
        path: str | PathLike[str],
        *,
        durability: str,
        commit_mode: str,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: int,
        compression: bool = True,
    ) -> NativeCoreEngineHandle:
        """Opens a lock-free read-only native engine snapshot."""

        module = self._require_module()
        options = self.open_options_payload(
            path=path,
            read_only=True,
            durability=durability,
            commit_mode=commit_mode,
            store_text=store_text,
            index_text=index_text,
            chunk_character_limit=chunk_character_limit,
            compression=compression,
        )
        return NativeCoreEngineHandle(
            module.CoreEngine.open_readonly(str(path), _dumps(options))
        )

    def open_appender(
        self,
        *,
        path: str | PathLike[str],
        durability: str = "buffered",
        store_text: bool = False,
        index_text: bool = False,
        acquire_writer_lock: bool = True,
    ) -> NativeCoreAppenderHandle:
        """Opens a shared-lock appender over the single index at ``path``.

        Many processes can each hold an appender and durably log vector-in records
        to the store's WAL concurrently; the next exclusive writer folds them into
        the index on open. The store must be in WAL commit mode and hold exactly
        one index. ``store_text``/``index_text`` mirror the store's writer: a
        document's caption is retained only under ``store_text`` and its lexical
        tokens are logged only under ``index_text``. Both default off (privacy: no
        raw text reaches ``<key>.wal``); enable them only for a store whose writer
        also does, or the writer drops the payload at checkpoint.
        Appenders take the shared ``<dir>/.lodedb.lock`` lock by default
        (``acquire_writer_lock=True``); pass ``False`` only when an outer caller
        already owns exclusion for the process.
        """

        module = self._require_module()
        options = self.open_options_payload(
            path=path,
            read_only=False,
            durability=durability,
            # Appends reach the index only through the WAL, which a generation-mode
            # writer never replays, so the appender requires WAL commit mode.
            commit_mode="wal",
            store_text=store_text,
            index_text=index_text,
            chunk_character_limit=8192,
            acquire_writer_lock=acquire_writer_lock,
        )
        return NativeCoreAppenderHandle(module.CoreAppender.open(_dumps(options)))

    @staticmethod
    def document_payload(document: EngineDocument) -> dict[str, Any]:
        return {
            "document_id": str(document.document_id),
            "text": str(document.text),
            "metadata": {str(key): str(value) for key, value in document.metadata.items()},
        }

    @staticmethod
    def vector_document_payload(document: EngineVectorDocument) -> dict[str, Any]:
        return {
            "document_id": str(document.document_id),
            "vector": [float(value) for value in document.vector],
            "metadata": {str(key): str(value) for key, value in document.metadata.items()},
            "text": None if document.text is None else str(document.text),
        }

    @staticmethod
    def query_payload(query: EngineQuery) -> dict[str, Any]:
        return {
            "text": str(query.text),
            "top_k": int(query.top_k),
            "filter": query.filter,
            "include": [str(value) for value in query.include],
            "mode": str(query.mode),
            "embedding": (
                None if query.embedding is None else [float(value) for value in query.embedding]
            ),
        }

    @staticmethod
    def open_options_payload(
        *,
        path: str | PathLike[str],
        read_only: bool,
        durability: str,
        commit_mode: str,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: int,
        compression: bool = True,
        acquire_writer_lock: bool = True,
    ) -> dict[str, Any]:
        return {
            "path": str(path),
            "read_only": bool(read_only),
            "durability": str(durability),
            "commit_mode": str(commit_mode),
            "store_text": bool(store_text),
            "index_text": bool(index_text),
            # Whether NEW writes to the retained document-text store are
            # zstd-compressed. The native core records the chosen value in the
            # text-store manifest and the persisted value wins on reopen, so this
            # only seeds a freshly created store.
            "compress_text": bool(compression),
            "chunk_character_limit": int(chunk_character_limit),
            # The native engine is the sole writer for a LodeDB handle, so it
            # takes the shared <dir>/.lodedb.lock single-writer lock itself. A
            # read-only open is always lock-free. The flag is ignored by the
            # read-only native path, which never locks.
            "acquire_writer_lock": bool(acquire_writer_lock) and not bool(read_only),
        }

    @staticmethod
    def index_create_options_payload(
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

    @staticmethod
    def response_from_native(status_code: int, payload: dict[str, Any]) -> EngineResponse:
        return EngineResponse(int(status_code), dict(payload))

    @staticmethod
    def response_from_error(error: NativeCoreError) -> EngineResponse:
        status_code = 404 if error.code == "NOT_FOUND" else 400
        if error.code in {"CORRUPT_STORE", "INTERNAL"}:
            status_code = 500
        return EngineResponse(
            status_code,
            {
                "status": "error",
                "error": error.message,
                "native_core_error": error.code,
            },
        )

    def _require_module(self) -> NativeCoreModule:
        module = self._module_or_none()
        if module is None:
            raise RuntimeError("native core extension is not available")
        return module

    def _module_or_none(self) -> NativeCoreModule | None:
        if self._native_module is not None:
            return self._native_module
        try:
            self._native_module = importlib.import_module("lodedb._native_core")
        except ImportError:
            return None
        return self._native_module


class NativeCoreEngineHandle:
    """Small JSON-backed wrapper over ``lodedb._native_core.CoreEngine``."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def create_index(self, index_id: str, *, vector_dim: int, bit_width: int = 4) -> None:
        self._engine.create_index(str(index_id), int(vector_dim), int(bit_width))

    def create_index_with_options(self, options: Mapping[str, Any]) -> None:
        self._engine.create_index_with_options(_dumps(dict(options)))

    def upsert_vectors(
        self,
        index_id: str,
        documents: Iterable[EngineVectorDocument],
    ) -> dict[str, Any]:
        documents = list(documents)
        array_upsert = getattr(self._engine, "upsert_vectors_array", None)
        if callable(array_upsert) and documents:
            import numpy as np

            # Ship the embedding matrix as one contiguous float32 array and only the
            # ids/metadata/text as a small batched JSON sidecar, instead of
            # JSON-encoding every float of every row (the bulk of a durable add).
            # Stack the vectors at the C level (np.asarray per row) rather than a
            # per-coordinate Python `float(...)` loop over every embedding.
            matrix = np.ascontiguousarray(
                [np.asarray(document.vector, dtype=np.float32) for document in documents],
                dtype=np.float32,
            )
            sidecar = [
                {
                    "document_id": str(document.document_id),
                    "metadata": {
                        str(key): str(value) for key, value in document.metadata.items()
                    },
                    "text": None if document.text is None else str(document.text),
                }
                for document in documents
            ]
            return self._loads(array_upsert(str(index_id), matrix, _dumps(sidecar)))
        payload = [NativeCoreAdapter.vector_document_payload(document) for document in documents]
        return self._loads(
            self._engine.upsert_vectors(
                str(index_id),
                _dumps(payload),
            )
        )

    def delete_documents(self, index_id: str, document_ids: Iterable[str]) -> dict[str, Any]:
        return self._loads(
            self._engine.delete_documents(
                str(index_id),
                _dumps([str(document_id) for document_id in document_ids]),
            )
        )

    def update_document_payload(
        self,
        index_id: str,
        document_id: str,
        *,
        metadata: Mapping[str, str] | None = None,
        text: str | None = None,
        clear_text: bool = False,
    ) -> dict[str, Any]:
        """Applies a metadata/raw-text update without re-embedding the vector.

        ``text_json`` is the JSON of an ``Option<Option<String>>``: ``"null"``
        clears the stored text, a JSON string sets it, and ``None`` (omitted)
        leaves it unchanged. ``metadata_json`` is ``None`` to leave metadata
        unchanged, else the full replacement string->string map.
        """

        metadata_json = (
            None
            if metadata is None
            else _dumps({str(key): str(value) for key, value in metadata.items()})
        )
        if clear_text:
            text_json: str | None = _dumps(None)
        elif text is not None:
            text_json = _dumps(text)
        else:
            text_json = None
        return self._loads(
            self._engine.update_document_payload(
                str(index_id), str(document_id), metadata_json, text_json
            )
        )

    def upsert_multivector(
        self,
        index_id: str,
        documents: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Upserts late-interaction documents (pooled vector + patch matrix).

        Each document is ``{document_id, vector, metadata, dtype, patch_count,
        patch_bytes}``. The pooled vectors ship as one contiguous float32 matrix
        and every encoded patch matrix is concatenated into one uint8 buffer
        partitioned by the per-document ``nbytes`` in the sidecar.
        """

        import numpy as np

        documents = list(documents)
        matrix = (
            np.ascontiguousarray(
                [np.asarray(document["vector"], dtype=np.float32) for document in documents],
                dtype=np.float32,
            )
            if documents
            else np.zeros((0, 0), dtype=np.float32)
        )
        blobs = [bytes(document["patch_bytes"]) for document in documents]
        joined = b"".join(blobs)
        patch_bytes = np.frombuffer(joined, dtype=np.uint8)
        sidecar = [
            {
                "document_id": str(document["document_id"]),
                "metadata": {
                    str(key): str(value)
                    for key, value in dict(document.get("metadata", {})).items()
                },
                "dtype": str(document["dtype"]),
                "patch_count": int(document["patch_count"]),
                "nbytes": len(blob),
            }
            for document, blob in zip(documents, blobs, strict=True)
        ]
        return self._loads(
            self._engine.upsert_multivector(
                str(index_id), matrix, patch_bytes, _dumps(sidecar)
            )
        )

    def query_multivector(
        self,
        index_id: str,
        query: Any,
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Late-interaction MaxSim query; ``query`` is an ``(n_query, dim)`` matrix."""

        import numpy as np

        matrix = np.ascontiguousarray(np.asarray(query, dtype=np.float32))
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        filter_json = None if filter is None else _dumps(dict(filter))
        return self._loads(
            self._engine.query_multivector(str(index_id), matrix, int(top_k), filter_json)
        )

    def query_vector(
        self,
        index_id: str,
        vector: Iterable[float],
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        filter_json = None if filter is None else _dumps(dict(filter))
        array_query = getattr(self._engine, "query_vector_array", None)
        if callable(array_query):
            import numpy as np

            # Passing the query as a contiguous float32 array skips the per-query
            # Python float list-build + json.dumps + Rust serde-parse, which is the
            # dominant cost of the single-query native path. The top-k result stays
            # JSON (small), so the return contract is unchanged. Convert the input
            # directly (no intermediate tuple), so an ndarray query is reused
            # in place and the borrowed buffer reaches the kernel without a copy.
            query = np.ascontiguousarray(vector, dtype=np.float32)
            return self._loads(array_query(str(index_id), query, int(top_k), filter_json))
        return self._loads(
            self._engine.query_vector(
                str(index_id),
                _dumps([float(value) for value in vector]),
                int(top_k),
                filter_json,
            )
        )

    def query_vectors_batch(
        self,
        index_id: str,
        vectors: Iterable[Iterable[float]],
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        filter_json = None if filter is None else _dumps(dict(filter))
        array_batch = getattr(self._engine, "query_vectors_batch_array", None)
        if callable(array_batch):
            import numpy as np

            rows = tuple(tuple(vector) for vector in vectors)
            if not rows:
                return []
            matrix = np.ascontiguousarray(rows, dtype=np.float32)
            raw = array_batch(str(index_id), matrix, int(top_k), filter_json)
        else:
            raw = self._engine.query_vectors_batch(
                str(index_id),
                _dumps([[float(value) for value in vector] for vector in vectors]),
                int(top_k),
                filter_json,
            )
        value = json.loads(raw)
        if not isinstance(value, list):
            raise RuntimeError("native core returned a non-list JSON payload")
        return [dict(item) for item in value]

    def query_vectors_batch_arrays(
        self,
        index_id: str,
        vectors: Any,
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> tuple[Any, list[str], list[dict[str, Any]], int] | None:
        """Near-zero-copy batch query.

        Returns ``(scores, document_ids, metadata, k)`` as flat ``[nq * k]`` buffers
        (scores a numpy array, ids a string list, metadata a dict list) when the
        native array-out path is available, else ``None`` so the caller uses the
        JSON path. The query matrix is built once via numpy rather than a Python
        tuple-of-tuples, and only metadata crosses as JSON.
        """

        array_out = getattr(self._engine, "query_vectors_batch_array_out", None)
        if not callable(array_out):
            return None
        import numpy as np

        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return None
        filter_json = None if filter is None else _dumps(dict(filter))
        scores, document_ids, metadata_json, k = array_out(
            str(index_id), matrix, int(top_k), filter_json
        )
        metadata = json.loads(metadata_json) if metadata_json else []
        return scores, list(document_ids), list(metadata), int(k)

    def prepare_text_upsert(
        self,
        index_id: str,
        documents: Iterable[EngineDocument],
        *,
        store_text: bool,
        index_text: bool,
        chunk_character_limit: int,
    ) -> dict[str, Any]:
        payload = [NativeCoreAdapter.document_payload(document) for document in documents]
        plan_json = self._engine.prepare_text_upsert(
            str(index_id),
            _dumps(payload),
            bool(store_text),
            bool(index_text),
            int(chunk_character_limit),
        )
        return self._loads_native_payload(plan_json)

    def apply_text_upsert(
        self,
        plan: Mapping[str, Any],
        embeddings: Iterable[Iterable[float]],
        *,
        embedding_time_ms: float,
    ) -> dict[str, Any]:
        plan_json = (
            plan.native_json
            if isinstance(plan, NativeCorePayload)
            else _dumps(dict(plan))
        )
        embedding_rows = tuple(embeddings)
        array_apply = getattr(self._engine, "apply_text_upsert_array", None)
        if callable(array_apply):
            import numpy as np

            embedding_array = (
                np.ascontiguousarray(embedding_rows, dtype=np.float32)
                if embedding_rows
                else np.empty((0, 0), dtype=np.float32)
            )
            return self._loads(
                array_apply(
                    plan_json,
                    embedding_array,
                    float(embedding_time_ms),
                )
            )
        embedding_payload = [[float(value) for value in row] for row in embedding_rows]
        return self._loads(
            self._engine.apply_text_upsert(
                plan_json,
                _dumps(embedding_payload),
                float(embedding_time_ms),
            )
        )

    def prepare_query_text(self, query: str, mode: str) -> dict[str, Any]:
        return self._loads_native_payload(self._engine.prepare_query_text(str(query), str(mode)))

    def search_embedded_text(
        self,
        index_id: str,
        query_plan: Mapping[str, Any],
        query_embedding: Iterable[float] | None,
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_plan_json = (
            query_plan.native_json
            if isinstance(query_plan, NativeCorePayload)
            else _dumps(dict(query_plan))
        )
        return self._search_embedded_text_json(
            index_id,
            query_plan_json,
            query_embedding,
            top_k=top_k,
            filter=filter,
        )

    def search_text(
        self,
        index_id: str,
        query: str,
        mode: str,
        query_embedding: Iterable[float] | None,
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        query_plan_json = _dumps(
            {
                "query": str(query),
                "mode": str(mode),
                "query_tokens": [],
                "requires_embedding": mode in {"vector", "hybrid"},
            }
        )
        return self._search_embedded_text_json(
            index_id,
            query_plan_json,
            query_embedding,
            top_k=top_k,
            filter=filter,
        )

    def search_text_batch(
        self,
        index_id: str,
        queries: Iterable[str],
        mode: str,
        query_embeddings: Any | None,
        *,
        top_k: int,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Batched text/hybrid/lexical search sharing one native vector scan.

        ``query_embeddings`` is an ``(nq, dim)`` float array for vector/hybrid (the
        SDK embeds in Python) and ``None`` for lexical. Returns one result payload
        per query, in input order.
        """

        requires_embedding = mode in {"vector", "hybrid"}
        plans = [
            {
                "query": str(query),
                "mode": str(mode),
                "query_tokens": [],
                "requires_embedding": requires_embedding,
            }
            for query in queries
        ]
        plans_json = _dumps(plans)
        filter_json = None if filter is None else _dumps(dict(filter))
        if query_embeddings is not None:
            import numpy as np

            matrix = np.ascontiguousarray(query_embeddings, dtype=np.float32)
            payload = self._engine.search_embedded_text_batch(
                index_id, plans_json, matrix, int(top_k), filter_json
            )
        else:
            payload = self._engine.search_embedded_text_batch(
                index_id, plans_json, None, int(top_k), filter_json
            )
        result = json.loads(payload)
        if not isinstance(result, list):
            raise RuntimeError("native core returned a non-list batch text payload")
        return [dict(item) for item in result]

    def _search_embedded_text_json(
        self,
        index_id: str,
        query_plan_json: str,
        query_embedding: Iterable[float] | None,
        *,
        top_k: int,
        filter: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        array_search = getattr(self._engine, "search_embedded_text_array", None)
        if query_embedding is not None and callable(array_search):
            import numpy as np

            query_array = np.ascontiguousarray(tuple(query_embedding), dtype=np.float32)
            return self._loads(
                array_search(
                    str(index_id),
                    query_plan_json,
                    query_array,
                    int(top_k),
                    None if filter is None else _dumps(dict(filter)),
                )
            )
        return self._loads(
            self._engine.search_embedded_text(
                str(index_id),
                query_plan_json,
                None
                if query_embedding is None
                else _dumps([float(value) for value in query_embedding]),
                int(top_k),
                None if filter is None else _dumps(dict(filter)),
            )
        )

    def stats(self, index_id: str) -> dict[str, Any]:
        return self._loads(self._engine.stats(str(index_id)))

    def get_document_text(self, index_id: str, document_id: str) -> str | None:
        value = json.loads(self._engine.get_document_text(str(index_id), str(document_id)))
        if value is None:
            return None
        if not isinstance(value, str):
            raise RuntimeError("native core returned a non-string document text payload")
        return value

    def get_document_texts(self, index_id: str, document_ids: Iterable[str]) -> dict[str, str]:
        value = json.loads(
            self._engine.get_document_texts(
                str(index_id),
                _dumps([str(document_id) for document_id in document_ids]),
            )
        )
        if not isinstance(value, dict):
            raise RuntimeError("native core returned a non-object text map payload")
        return {str(key): str(text) for key, text in value.items()}

    def get_document(self, index_id: str, document_id: str) -> dict[str, Any] | None:
        value = json.loads(self._engine.get_document(str(index_id), str(document_id)))
        if value is None:
            return None
        if not isinstance(value, dict):
            raise RuntimeError("native core returned a non-object document payload")
        return dict(value)

    def list_documents(
        self,
        index_id: str,
        *,
        filter: Mapping[str, Any] | None = None,
        after: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        value = json.loads(
            self._engine.list_documents(
                str(index_id),
                None if filter is None else _dumps(dict(filter)),
                None if after is None else str(after),
                None if limit is None else int(limit),
            )
        )
        if not isinstance(value, list):
            raise RuntimeError("native core returned a non-list document payload")
        return [dict(item) for item in value]

    def persist(self) -> None:
        self._engine.persist()

    def close(self) -> None:
        self._engine.close()

    @staticmethod
    def _loads(payload: str) -> dict[str, Any]:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise RuntimeError("native core returned a non-object JSON payload")
        return value

    @staticmethod
    def _loads_native_payload(payload: str) -> NativeCorePayload:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise RuntimeError("native core returned a non-object JSON payload")
        return NativeCorePayload(value, native_json=payload)


class NativeCoreAppenderHandle:
    """Small JSON-backed wrapper over ``lodedb._native_core.CoreAppender``.

    Each ``append_*`` durably logs one WAL record under the shared lock and
    returns the log sequence number (LSN) assigned to it.
    """

    def __init__(self, appender: Any) -> None:
        self._appender = appender

    def append_vectors(self, documents: Iterable[EngineVectorDocument]) -> int:
        payload = [NativeCoreAdapter.vector_document_payload(document) for document in documents]
        return int(self._appender.append_vectors(_dumps(payload)))

    def append_deletes(self, document_ids: Iterable[str]) -> int:
        return int(
            self._appender.append_deletes(
                _dumps([str(document_id) for document_id in document_ids])
            )
        )
