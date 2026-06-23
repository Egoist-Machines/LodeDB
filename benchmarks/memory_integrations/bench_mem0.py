"""mem0 suite: LodeDB vs mem0's own vector-store providers.

mem0 owns embeddings and drives its backends through ``VectorStoreBase``, so this
suite is a clean vector-in comparison: the same precomputed memory vectors and
payloads go into LodeDB's adapter and into mem0's default (Qdrant) plus its FAISS
and Chroma providers. The workflow is agent memory: insert scoped memories
(``user_id`` / ``agent_id`` / ``run_id`` / ``category``), search, filtered search
by user, update, accrue more memories one at a time, then reopen.

Backends whose library is missing are skipped (recorded), so this runs with
whatever is installed.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
from common import (
    CHROMA_BATCH,
    Embedded,
    StoreDriver,
    dir_bytes,
    exact_topk,
    latency_summary,
    memory_payload,
    recall_at_k,
    run_core_phases,
)

_NS = uuid.UUID("0de00b00-0000-4000-8000-000000000000")
_FILTER_USER = "u0"


def _mem_ids(n: int) -> list[str]:
    """Deterministic UUID ids (Qdrant requires UUID/int point ids)."""

    return [str(uuid.uuid5(_NS, f"mem{i}")) for i in range(n)]


class _Mem0Driver(StoreDriver):
    """Wraps a mem0 ``VectorStoreBase`` provider in the common driver protocol."""

    role = "baseline"
    embeds_on_ingest = False  # mem0 is vector-in: stores receive precomputed vectors
    incremental_is_delta = True

    def __init__(self, name: str, base_dir: Path, dim: int) -> None:
        self.name = name
        self.dim = dim
        self.dir = base_dir / f"mem0_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store = self._construct()

    def _construct(self) -> Any:  # provider-specific
        raise NotImplementedError

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        self.store.insert(vectors=vectors, payloads=metadatas, ids=ids)

    def query_one(self, qvec, k) -> list[str]:
        hits = self.store.search(query="", vectors=qvec, top_k=k)
        return [str(h.id) for h in hits]

    def filtered_query_one(self, qvec, k, filters) -> list[str]:
        hits = self.store.search(query="", vectors=qvec, top_k=k, filters=filters)
        return [str(h.id) for h in hits]

    def persist(self) -> None:
        pass  # mem0 providers persist on every write (Qdrant/Chroma/FAISS path-backed)

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir)

    def incremental_add(self, doc_id, text, vector, metadata) -> None:
        self.store.insert(vectors=[vector], payloads=[metadata], ids=[doc_id])

    def reopen(self) -> int:
        self._close_handle()
        self.store = self._construct()
        return self._count()

    def _count(self) -> int:
        try:
            info = self.store.col_info()
            if isinstance(info, dict) and "count" in info:
                return int(info["count"])
            # Chroma's col_info returns a Collection object exposing .count()
            counter = getattr(info, "count", None)
            if callable(counter):
                return int(counter())
        except Exception:
            pass
        try:
            rows = self.store.list(top_k=10**9)
            rows = rows[0] if rows and isinstance(rows[0], list) else rows
            return len(rows)
        except Exception:
            return -1

    def _close_handle(self) -> None:
        pass

    def close(self) -> None:
        self._close_handle()


class _LodeDBDriver(_Mem0Driver):
    role = "lodedb"

    def _construct(self) -> Any:
        from lodedb.local.integrations.mem0 import LodeDBVectorStore

        return LodeDBVectorStore(
            collection_name="mem0", path=str(self.dir), embedding_model_dims=self.dim
        )

    def _close_handle(self) -> None:
        try:
            self.store.close()
        except Exception:
            pass


class _QdrantDriver(_Mem0Driver):
    def _construct(self) -> Any:
        from mem0.vector_stores.qdrant import Qdrant

        return Qdrant(
            collection_name="mem0",
            embedding_model_dims=self.dim,
            path=str(self.dir),
            on_disk=True,
        )

    def _close_handle(self) -> None:
        try:
            self.store.client.close()
        except Exception:
            pass


class _FaissDriver(_Mem0Driver):
    def _construct(self) -> Any:
        from mem0.vector_stores.faiss import FAISS

        return FAISS(
            collection_name="mem0",
            path=str(self.dir),
            distance_strategy="cosine",
            embedding_model_dims=self.dim,
        )


class _ChromaDriver(_Mem0Driver):
    def _construct(self) -> Any:
        from mem0.vector_stores.chroma import ChromaDB

        return ChromaDB(collection_name="mem0", path=str(self.dir))

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        for i in range(0, len(ids), CHROMA_BATCH):  # chromadb caps single-add batch size
            sl = slice(i, i + CHROMA_BATCH)
            self.store.insert(vectors=vectors[sl], payloads=metadatas[sl], ids=ids[sl])


_PROVIDERS: list[tuple[str, type[_Mem0Driver]]] = [
    ("lodedb", _LodeDBDriver),
    ("qdrant", _QdrantDriver),
    ("faiss", _FaissDriver),
    ("chroma", _ChromaDriver),
]


def _make_extra_phases(
    embedded: Embedded,
    *,
    k: int,
    within_user_truth: list[set[str]],
    payloads: list[dict[str, Any]],
    ids: list[str],
):
    """Builds the mem0-specific filtered-search + update phases for a driver."""

    import time  # noqa: PLC0415

    update_targets = list(range(0, min(64, len(ids))))

    def extra(driver: _Mem0Driver, result: dict[str, Any]) -> None:
        # filtered search by user_id (mem0's most common scoped read)
        filters = {"user_id": _FILTER_USER}
        latencies: list[float] = []
        returned: list[list[str]] = []
        try:
            for row in embedded.query_vectors:
                qv = row.tolist()
                s = time.perf_counter()
                hits = driver.filtered_query_one(qv, k, filters)
                latencies.append((time.perf_counter() - s) * 1000.0)
                returned.append(hits)
            result["filtered_query"] = {
                **latency_summary(latencies),
                "filter": "user_id=u0",
                "recall_at_k": recall_at_k(returned, within_user_truth, k),
            }
        except Exception as exc:
            result["filtered_query"] = {"supported": False, "error": type(exc).__name__}

        # update existing memories (re-embed/re-scope an accrued memory)
        upd_latencies: list[float] = []
        try:
            for i in update_targets:
                new_payload = dict(payloads[i])
                new_payload["category"] = "updated"
                s = time.perf_counter()
                driver.store.update(
                    ids[i], vector=embedded.doc_vectors[i].tolist(), payload=new_payload
                )
                upd_latencies.append((time.perf_counter() - s) * 1000.0)
            result["update"] = latency_summary(upd_latencies)
        except Exception as exc:
            result["update"] = {"supported": False, "error": type(exc).__name__}

    return extra


def run_mem0_suite(
    embedded: Embedded,
    *,
    n_users: int,
    k: int,
    incremental_count: int,
    workdir: Path,
) -> dict[str, Any]:
    """Runs the mem0 agent-memory workflow across all available providers."""

    n = len(embedded.doc_vectors)
    ids = _mem_ids(n)
    inc_ids = [str(uuid.uuid5(_NS, f"inc{i}")) for i in range(incremental_count)]
    payloads = [memory_payload(i, n_users) for i in range(n)]

    truth = exact_topk(ids, embedded.doc_vectors, embedded.query_vectors, k)
    allowed = np.array([(i % n_users) == 0 for i in range(n)], dtype=bool)
    within_user_truth = exact_topk(
        ids, embedded.doc_vectors, embedded.query_vectors, k, allowed=allowed
    )

    backends: list[dict[str, Any]] = []
    for name, cls in _PROVIDERS:
        try:
            driver = cls(name, workdir, embedded.native_dim)
        except Exception as exc:
            backends.append(
                {"backend": name, "role": cls.role, "skipped": True, "error": _short(exc)}
            )
            continue
        try:
            extra = _make_extra_phases(
                embedded, k=k, within_user_truth=within_user_truth, payloads=payloads, ids=ids
            )
            metrics = run_core_phases(
                driver,
                embedded,
                ids,
                ["" for _ in ids],  # mem0 is vector-in; no text path
                payloads,
                truth,
                k=k,
                incremental_count=incremental_count,
                incremental_ids=inc_ids,
                extra_phases=extra,
            )
            backends.append(metrics)
        except Exception as exc:
            backends.append(
                {"backend": name, "role": cls.role, "failed": True, "error": _short(exc)}
            )
            try:
                driver.close()
            except Exception:
                pass

    return {
        "framework": "mem0",
        "interface": "mem0.vector_stores.base.VectorStoreBase",
        "workflow": "agent_memory",
        "document_count": n,
        "query_count": int(embedded.query_vectors.shape[0]),
        "n_users": n_users,
        "filter_user_share": round(float(allowed.mean()), 4),
        "backends": backends,
    }


def _short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:120]}"
