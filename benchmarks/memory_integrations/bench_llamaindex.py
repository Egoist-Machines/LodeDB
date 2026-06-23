"""LlamaIndex suite: LodeDB vs LlamaIndex's default and common vector stores.

Drives the ``BasePydanticVectorStore`` interface that the LodeDB adapter
implements, across LodeDB, the in-memory default (``SimpleVectorStore``), Faiss,
Chroma, and Qdrant. Each document becomes a ``TextNode`` carrying both its text
(for LodeDB's text-path adapter, which re-embeds) and its precomputed embedding
(for the vector-path baselines), so the comparison fixes the embedding model and
varies only the store. Queries run by precomputed query embedding; LodeDB uses
``search_by_vector`` for the same by-vector path. Workflow: bulk add, retrieve,
persist + footprint, incremental durable adds, reopen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common import (
    CHROMA_BATCH,
    Embedded,
    StoreDriver,
    dir_bytes,
    exact_topk,
    rag_metadata,
    run_core_phases,
    uuid_ids,
)

_DIM = 384


def _node(node_id: str, text: str, vector: list[float], metadata: dict[str, Any]):
    from llama_index.core.schema import TextNode  # noqa: PLC0415

    return TextNode(id_=node_id, text=text, embedding=vector, metadata=dict(metadata))


def _vsquery(qvec: list[float], k: int):
    from llama_index.core.vector_stores import VectorStoreQuery  # noqa: PLC0415

    return VectorStoreQuery(query_embedding=qvec, similarity_top_k=k)


class _LIDriver(StoreDriver):
    """Base LlamaIndex driver; baselines query by precomputed embedding."""

    role = "baseline"
    embeds_on_ingest = False
    incremental_is_delta = True
    supports_reopen = True

    def __init__(self, name: str, base_dir: Path, dim: int) -> None:
        self.name = name
        self.dim = dim
        self.dir = base_dir / f"li_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.store: Any = None

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        nodes = [_node(ids[i], texts[i], vectors[i], metadatas[i]) for i in range(len(ids))]
        self.store.add(nodes)

    def query_one(self, qvec, k) -> list[str]:
        return list(self.store.query(_vsquery(qvec, k)).ids or [])

    def incremental_add(self, doc_id, text, vector, metadata) -> None:
        self.store.add([_node(doc_id, text, vector, metadata)])
        self._persist_incremental()

    def _persist_incremental(self) -> None:
        pass

    def persist(self) -> None:
        pass

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir)

    def close(self) -> None:
        pass


class _SimpleDriver(_LIDriver):
    # LlamaIndex's default store: in-RAM, durability is a full JSON rewrite.
    incremental_is_delta = False

    def __init__(self, name, base_dir, dim) -> None:
        super().__init__(name, base_dir, dim)
        self.file = self.dir / "simple_store.json"

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        from llama_index.core.vector_stores import SimpleVectorStore  # noqa: PLC0415

        self.store = SimpleVectorStore()
        super().ingest(ids, texts, vectors, metadatas)

    def persist(self) -> None:
        self.store.persist(persist_path=str(self.file))

    def _persist_incremental(self) -> None:
        self.store.persist(persist_path=str(self.file))  # O(corpus) per added node

    def reopen(self) -> int:
        from llama_index.core.vector_stores import SimpleVectorStore  # noqa: PLC0415

        self.store = SimpleVectorStore.from_persist_path(str(self.file))
        return len(self.store.data.embedding_dict)


class _FaissDriver(_LIDriver):
    incremental_is_delta = False  # the faiss index is rewritten whole on persist

    # FaissVectorStore keeps no docstore: used standalone it returns the faiss
    # positional index, not the node id. We map positions back through insertion
    # order so recall is correct, and note the limitation (no payload round-trip).
    _order: list[str] = []

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        import faiss  # noqa: PLC0415
        from llama_index.vector_stores.faiss import FaissVectorStore  # noqa: PLC0415

        self.store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(self.dim))
        self._order = list(ids)
        super().ingest(ids, texts, vectors, metadatas)

    def query_one(self, qvec, k) -> list[str]:
        out: list[str] = []
        for rid in self.store.query(_vsquery(qvec, k)).ids or []:
            try:
                out.append(self._order[int(rid)])
            except (ValueError, IndexError):
                out.append(str(rid))
        return out

    def incremental_add(self, doc_id, text, vector, metadata) -> None:
        self.store.add([_node(doc_id, text, vector, metadata)])
        self._order.append(doc_id)
        self._persist_incremental()

    def persist(self) -> None:
        self.store.persist(persist_path=str(self.dir / "faiss.index"))

    def _persist_incremental(self) -> None:
        self.store.persist(persist_path=str(self.dir / "faiss.index"))

    def reopen(self) -> int:
        from llama_index.vector_stores.faiss import FaissVectorStore  # noqa: PLC0415

        self.store = FaissVectorStore.from_persist_path(str(self.dir / "faiss.index"))
        return int(self.store._faiss_index.ntotal)


class _ChromaDriver(_LIDriver):
    def _collection(self):
        import chromadb  # noqa: PLC0415

        client = chromadb.PersistentClient(path=str(self.dir))
        return client.get_or_create_collection("rag")

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: PLC0415

        self.store = ChromaVectorStore(chroma_collection=self._collection())
        nodes = [_node(ids[i], texts[i], vectors[i], metadatas[i]) for i in range(len(ids))]
        for i in range(0, len(nodes), CHROMA_BATCH):  # chromadb caps single-add batch size
            self.store.add(nodes[i : i + CHROMA_BATCH])

    def reopen(self) -> int:
        from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: PLC0415

        coll = self._collection()
        self.store = ChromaVectorStore(chroma_collection=coll)
        return int(coll.count())


class _QdrantDriver(_LIDriver):
    def _client(self):
        from qdrant_client import QdrantClient  # noqa: PLC0415

        return QdrantClient(path=str(self.dir))

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        from llama_index.vector_stores.qdrant import QdrantVectorStore  # noqa: PLC0415

        self._qclient = self._client()
        self.store = QdrantVectorStore(client=self._qclient, collection_name="rag")
        super().ingest(ids, texts, vectors, metadatas)

    def _close_client(self) -> None:
        try:
            self._qclient.close()
        except Exception:
            pass

    def reopen(self) -> int:
        from llama_index.vector_stores.qdrant import QdrantVectorStore  # noqa: PLC0415

        self._close_client()
        self._qclient = self._client()
        self.store = QdrantVectorStore(client=self._qclient, collection_name="rag")
        return int(self._qclient.count("rag").count)

    def close(self) -> None:
        self._close_client()


class _LodeDBDriver(_LIDriver):
    role = "lodedb"
    embeds_on_ingest = True

    def __init__(self, name, base_dir, dim, *, model: str, device: str) -> None:
        super().__init__(name, base_dir, dim)
        self.model = model
        self.device = device
        self._db: Any = None

    def _open(self) -> None:
        from lodedb import LodeDB  # noqa: PLC0415
        from lodedb.local.integrations.llama_index import LodeDBVectorStore  # noqa: PLC0415

        self._db = LodeDB(path=str(self.dir / "lode"), model=self.model, device=self.device)
        self.store = LodeDBVectorStore(self._db)

    def warmup(self) -> None:
        self._open()
        try:
            self._db._embedding_backend.embed_documents(("warmup",))
        except Exception:
            pass

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        nodes = [_node(ids[i], texts[i], vectors[i], metadatas[i]) for i in range(len(ids))]
        self.store.add(nodes)

    def query_one(self, qvec, k) -> list[str]:
        return [h.id for h in self._db.search_by_vector(qvec, k=k)]

    def persist(self) -> None:
        self._db.persist()

    def reopen(self) -> int:
        self._db.close()
        self._open()
        return int(self._db.stats().get("document_count", 0) or 0)

    def close(self) -> None:
        try:
            self._db.close()
        except Exception:
            pass


def run_llamaindex_suite(
    embedded: Embedded,
    texts: list[str],
    *,
    model: str,
    device: str,
    k: int,
    incremental_count: int,
    workdir: Path,
) -> dict[str, Any]:
    """Runs the RAG workflow across all available LlamaIndex vector stores."""

    n = len(texts)
    ids = uuid_ids(n, "llamaindex")
    inc_ids = uuid_ids(incremental_count, "llamaindex-inc")
    metadatas = [rag_metadata(i) for i in range(n)]
    truth = exact_topk(ids, embedded.doc_vectors, embedded.query_vectors, k)

    builders: list[tuple[str, Any]] = [
        ("lodedb", lambda d: _LodeDBDriver("lodedb", d, _DIM, model=model, device=device)),
        ("simple", lambda d: _SimpleDriver("simple", d, _DIM)),
        ("faiss", lambda d: _FaissDriver("faiss", d, _DIM)),
        ("chroma", lambda d: _ChromaDriver("chroma", d, _DIM)),
        ("qdrant", lambda d: _QdrantDriver("qdrant", d, _DIM)),
    ]

    backends: list[dict[str, Any]] = []
    for name, make in builders:
        try:
            driver = make(workdir)
        except Exception as exc:
            backends.append({"backend": name, "skipped": True, "error": _short(exc)})
            continue
        try:
            metrics = run_core_phases(
                driver,
                embedded,
                ids,
                texts,
                metadatas,
                truth,
                k=k,
                incremental_count=incremental_count,
                incremental_ids=inc_ids,
            )
            backends.append(metrics)
        except Exception as exc:
            backends.append({"backend": name, "failed": True, "error": _short(exc)})
            try:
                driver.close()
            except Exception:
                pass

    return {
        "framework": "llamaindex",
        "interface": "llama_index.core.vector_stores.types.BasePydanticVectorStore",
        "workflow": "rag",
        "document_count": n,
        "query_count": int(embedded.query_vectors.shape[0]),
        "default_backend": "simple",
        "backends": backends,
    }


def _short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:120]}"
