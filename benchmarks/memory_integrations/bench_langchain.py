"""LangChain suite: LodeDB vs LangChain's default and common vector stores.

Drives the ``langchain_core.vectorstores.VectorStore`` interface (the exact
contract the LodeDB adapter implements) across LodeDB, the in-memory default
(``InMemoryVectorStore``), FAISS, Chroma, and Qdrant. The workflow is RAG: bulk
ingest a corpus, retrieve, force durability and measure on-disk footprint, accrue
documents one at a time (durably), then reopen.

To keep this a store comparison rather than an embedder comparison, every backend
uses the same fixed embedding. Baselines receive precomputed vectors through a
caching ``Embeddings`` (so their ingest is store-only), and queries run through
each store's ``similarity_search_by_vector`` with precomputed query vectors (no
query embedding charged). LodeDB's adapter embeds text internally by design, so
its ingest is end-to-end and the runner reports a store-only figure by subtracting
the shared embedding time; its query path uses ``search_by_vector`` for the same
by-vector comparison.
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


def _make_cached_embeddings(cache: dict[str, list[float]], embedded: Embedded):
    """A LangChain ``Embeddings`` that serves precomputed doc vectors from a cache."""

    from langchain_core.embeddings import Embeddings  # noqa: PLC0415

    class CachedEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [cache.get(t) or embedded.embed_text(t) for t in texts]

        def embed_query(self, text: str) -> list[float]:
            return cache.get(text) or embedded.embed_text(text)

    return CachedEmbeddings()


def _doc_id(doc: Any) -> str:
    meta = getattr(doc, "metadata", None) or {}
    return str(meta.get("_id") or getattr(doc, "id", "") or "")


class _LCDriver(StoreDriver):
    """Base LangChain driver; baselines store precomputed vectors via the cache."""

    role = "baseline"
    embeds_on_ingest = False
    incremental_is_delta = True
    supports_reopen = True

    def __init__(self, name: str, base_dir: Path, emb: Any) -> None:
        self.name = name
        self.dir = base_dir / f"lc_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.emb = emb
        self.store: Any = None

    @staticmethod
    def _metas(ids, metadatas):
        return [{**m, "_id": ids[i]} for i, m in enumerate(metadatas)]

    def query_one(self, qvec, k) -> list[str]:
        docs = self.store.similarity_search_by_vector(qvec, k=k)
        return [_doc_id(d) for d in docs]

    def query_batch(self, qvecs, k) -> list[list[str]]:
        # LangChain's VectorStore retriever contract is single-query.
        return [self.query_one(qv, k) for qv in qvecs]

    def incremental_add(self, doc_id, text, vector, metadata) -> None:
        self.store.add_texts([text], metadatas=[{**metadata, "_id": doc_id}], ids=[doc_id])
        self._persist_incremental()

    def _persist_incremental(self) -> None:
        pass  # auto-persisting stores need nothing extra per add

    def persist(self) -> None:
        pass

    def footprint_bytes(self) -> int:
        return dir_bytes(self.dir)

    def close(self) -> None:
        pass


class _InMemoryDriver(_LCDriver):
    # The default store is pure RAM; durability is a full JSON dump every time.
    incremental_is_delta = False

    def __init__(self, name, base_dir, emb) -> None:
        super().__init__(name, base_dir, emb)
        self.file = self.dir / "store.json"

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        from langchain_core.vectorstores import InMemoryVectorStore  # noqa: PLC0415

        self.store = InMemoryVectorStore(self.emb)
        self.store.add_texts(texts, metadatas=self._metas(ids, metadatas), ids=ids)

    def persist(self) -> None:
        self.store.dump(str(self.file))

    def _persist_incremental(self) -> None:
        self.store.dump(str(self.file))  # O(corpus) rewrite per added memory

    def reopen(self) -> int:
        from langchain_core.vectorstores import InMemoryVectorStore  # noqa: PLC0415

        self.store = InMemoryVectorStore.load(str(self.file), self.emb)
        return len(self.store.store)


class _FaissDriver(_LCDriver):
    incremental_is_delta = False  # FAISS dumps the whole index to make a write durable

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        from langchain_community.vectorstores import FAISS  # noqa: PLC0415

        self.store = FAISS.from_texts(
            texts, self.emb, metadatas=self._metas(ids, metadatas), ids=ids
        )

    def persist(self) -> None:
        self.store.save_local(str(self.dir))

    def _persist_incremental(self) -> None:
        self.store.save_local(str(self.dir))

    def reopen(self) -> int:
        from langchain_community.vectorstores import FAISS  # noqa: PLC0415

        self.store = FAISS.load_local(str(self.dir), self.emb, allow_dangerous_deserialization=True)
        return int(self.store.index.ntotal)


class _ChromaDriver(_LCDriver):
    def ingest(self, ids, texts, vectors, metadatas) -> None:
        from langchain_chroma import Chroma  # noqa: PLC0415

        self.store = Chroma(
            collection_name="rag", embedding_function=self.emb, persist_directory=str(self.dir)
        )
        metas = self._metas(ids, metadatas)
        for i in range(0, len(ids), CHROMA_BATCH):  # chromadb caps single-add batch size
            sl = slice(i, i + CHROMA_BATCH)
            self.store.add_texts(texts[sl], metadatas=metas[sl], ids=ids[sl])

    def reopen(self) -> int:
        from langchain_chroma import Chroma  # noqa: PLC0415

        self.store = Chroma(
            collection_name="rag", embedding_function=self.emb, persist_directory=str(self.dir)
        )
        return int(self.store._collection.count())


class _QdrantDriver(_LCDriver):
    def ingest(self, ids, texts, vectors, metadatas) -> None:
        from langchain_qdrant import QdrantVectorStore  # noqa: PLC0415

        self.store = QdrantVectorStore.from_texts(
            texts,
            self.emb,
            metadatas=self._metas(ids, metadatas),
            ids=ids,
            path=str(self.dir),
            collection_name="rag",
        )

    def reopen(self) -> int:
        from langchain_qdrant import QdrantVectorStore  # noqa: PLC0415
        from qdrant_client import QdrantClient  # noqa: PLC0415

        self._close_client()
        client = QdrantClient(path=str(self.dir))
        self.store = QdrantVectorStore(client=client, collection_name="rag", embedding=self.emb)
        return int(client.count("rag").count)

    def _close_client(self) -> None:
        try:
            self.store.client.close()
        except Exception:
            pass

    def close(self) -> None:
        self._close_client()


class _LanceDBDriver(_LCDriver):
    # Embedded columnar store; appends fragments, no full rewrite to persist. The
    # adapter defaults to mode="overwrite" (each add replaces the table), so pin
    # mode="append" or incremental adds would wipe the corpus.
    def _open(self):
        from langchain_community.vectorstores import LanceDB  # noqa: PLC0415

        return LanceDB(uri=str(self.dir), table_name="rag", embedding=self.emb, mode="append")

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        self.store = self._open()
        self.store.add_texts(texts, metadatas=self._metas(ids, metadatas), ids=ids)

    def reopen(self) -> int:
        self.store = self._open()
        try:
            return int(self.store.get_table().count_rows())
        except Exception:
            return -1


class _SQLiteVecDriver(_LCDriver):
    # Single-file sqlite database with the sqlite-vec extension.
    def ingest(self, ids, texts, vectors, metadatas) -> None:
        from langchain_community.vectorstores import SQLiteVec  # noqa: PLC0415

        self._dbfile = str(self.dir / "vec.db")
        conn = SQLiteVec.create_connection(db_file=self._dbfile)
        self.store = SQLiteVec(table="rag", connection=conn, embedding=self.emb)
        self.store.add_texts(texts, metadatas=self._metas(ids, metadatas), ids=ids)

    def reopen(self) -> int:
        from langchain_community.vectorstores import SQLiteVec  # noqa: PLC0415

        conn = SQLiteVec.create_connection(db_file=self._dbfile)
        self.store = SQLiteVec(table="rag", connection=conn, embedding=self.emb)
        try:
            return int(self.store._connection.execute("SELECT count(*) FROM rag").fetchone()[0])
        except Exception:
            return -1


class _PgVectorDriver(_LCDriver):
    # Postgres + pgvector via an embedded server (pgserver bundles the binary +
    # the vector extension). No ANN index, so it is an exact seq scan, like LodeDB.
    def __init__(self, name, base_dir, emb) -> None:
        super().__init__(name, base_dir, emb)
        self._srv = None
        self._uri = None

    def warmup(self) -> None:
        import pgserver  # noqa: PLC0415

        self._srv = pgserver.get_server(str(self.dir / "pg"))
        self._srv.psql("CREATE EXTENSION IF NOT EXISTS vector;")
        self._uri = self._srv.get_uri()

    def _open(self):
        from langchain_postgres import PGVector  # noqa: PLC0415

        return PGVector(
            embeddings=self.emb,
            collection_name="rag",
            connection=self._uri.replace("postgresql://", "postgresql+psycopg://"),
            use_jsonb=True,
        )

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        self.store = self._open()
        metas = self._metas(ids, metadatas)
        # libpq caps a single statement at 65,535 bound params and PGVector inserts
        # one multi-row INSERT (~5 params/row), so chunk to stay under the limit.
        for i in range(0, len(ids), CHROMA_BATCH):
            sl = slice(i, i + CHROMA_BATCH)
            self.store.add_texts(texts[sl], metadatas=metas[sl], ids=ids[sl])

    def footprint_bytes(self) -> int:
        # The embedding table + indexes + toast (excludes server WAL/catalogs),
        # the fairest "this data's footprint" measure for a server-backed store.
        try:
            import psycopg  # noqa: PLC0415

            with psycopg.connect(self._uri) as c:
                return int(
                    c.execute("SELECT pg_total_relation_size('langchain_pg_embedding')").fetchone()[
                        0
                    ]
                )
        except Exception:
            return dir_bytes(self.dir)

    def reopen(self) -> int:
        self.store = self._open()
        try:
            import psycopg  # noqa: PLC0415

            with psycopg.connect(self._uri) as c:
                return int(c.execute("SELECT count(*) FROM langchain_pg_embedding").fetchone()[0])
        except Exception:
            return -1

    def close(self) -> None:
        try:
            self._srv.cleanup()
        except Exception:
            pass


class _LodeDBDriver(_LCDriver):
    role = "lodedb"
    # Fed the same precomputed vectors as every baseline (via LodeDB's vector-in
    # SDK), so no backend is charged for embedding and this is a store-vs-store
    # comparison. The LangChain adapter is text-path in production (it embeds
    # internally), but the embedding is shared work every backend's app pays; the
    # harness factors it out for all of them, LodeDB included.
    embeds_on_ingest = False
    incremental_is_delta = True
    batch_path = "search_many_by_vector (GPU-resident scan on CUDA)"

    def __init__(self, name, base_dir, emb, *, model: str, device: str) -> None:
        super().__init__(name, base_dir, emb)
        self.model = model
        self.device = device
        self._db: Any = None

    def _open(self) -> None:
        from lodedb import LodeDB  # noqa: PLC0415
        from lodedb.local.integrations.langchain import LodeDBVectorStore  # noqa: PLC0415

        self._db = LodeDB(path=str(self.dir / "lode"), model=self.model, device=self.device)
        self.store = LodeDBVectorStore(self._db)

    def warmup(self) -> None:
        self._open()  # vectors are precomputed, so no embedding model to warm

    def ingest(self, ids, texts, vectors, metadatas) -> None:
        self._db.add_vectors_many(
            [
                {"vector": vectors[i], "id": ids[i], "metadata": metadatas[i], "text": texts[i]}
                for i in range(len(ids))
            ],
            normalize=False,
        )

    def incremental_add(self, doc_id, text, vector, metadata) -> None:
        self._db.add_vectors(vector, id=doc_id, metadata=metadata, text=text, normalize=False)

    def query_one(self, qvec, k) -> list[str]:
        return [h.id for h in self._db.search_by_vector(qvec, k=k)]

    def query_batch(self, qvecs, k) -> list[list[str]]:
        # The batched path: on a CUDA host with cupy this runs the GPU-resident scan.
        return [[h.id for h in hits] for hits in self._db.search_many_by_vector(qvecs, k=k)]

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


def run_langchain_suite(
    embedded: Embedded,
    texts: list[str],
    *,
    model: str,
    device: str,
    k: int,
    incremental_count: int,
    batch_size: int,
    workdir: Path,
) -> dict[str, Any]:
    """Runs the RAG workflow across all available LangChain vector stores."""

    n = len(texts)
    ids = uuid_ids(n, "langchain")
    inc_ids = uuid_ids(incremental_count, "langchain-inc")
    metadatas = [rag_metadata(i) for i in range(n)]
    truth = exact_topk(ids, embedded.doc_vectors, embedded.query_vectors, k)

    cache = {text: embedded.doc_vectors[i].tolist() for i, text in enumerate(texts)}
    emb = _make_cached_embeddings(cache, embedded)

    builders: list[tuple[str, Any]] = [
        ("lodedb", lambda d: _LodeDBDriver("lodedb", d, emb, model=model, device=device)),
        ("inmemory", lambda d: _InMemoryDriver("inmemory", d, emb)),
        ("faiss", lambda d: _FaissDriver("faiss", d, emb)),
        ("chroma", lambda d: _ChromaDriver("chroma", d, emb)),
        ("qdrant", lambda d: _QdrantDriver("qdrant", d, emb)),
        ("lancedb", lambda d: _LanceDBDriver("lancedb", d, emb)),
        ("sqlite-vec", lambda d: _SQLiteVecDriver("sqlite-vec", d, emb)),
        ("pgvector", lambda d: _PgVectorDriver("pgvector", d, emb)),
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
                batch_size=batch_size,
            )
            backends.append(metrics)
        except Exception as exc:
            backends.append({"backend": name, "failed": True, "error": _short(exc)})
            try:
                driver.close()
            except Exception:
                pass

    return {
        "framework": "langchain",
        "interface": "langchain_core.vectorstores.VectorStore",
        "workflow": "rag",
        "document_count": n,
        "query_count": int(embedded.query_vectors.shape[0]),
        "default_backend": "inmemory",
        "backends": backends,
    }


def _short(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:120]}"
