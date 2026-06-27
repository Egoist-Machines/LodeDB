from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pytest


def _load_turbovec_extension_override() -> None:
    path = os.environ.get("LODEDB_NATIVE_CORE_EXTENSION_PATH")
    if not path:
        return
    import lodedb  # noqa: F401 - ensure the package parent exists

    extension_path = Path(path)
    spec = importlib.util.spec_from_file_location("lodedb._turbovec", extension_path)
    if spec is None or spec.loader is None:
        loader = importlib.machinery.ExtensionFileLoader(
            "lodedb._turbovec",
            str(extension_path),
        )
        spec = importlib.util.spec_from_file_location(
            "lodedb._turbovec",
            extension_path,
            loader=loader,
        )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load extension from {extension_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["lodedb._turbovec"] = module
    spec.loader.exec_module(module)


_load_turbovec_extension_override()

try:
    native_core = importlib.import_module("lodedb._native_core")
except ImportError as exc:
    pytest.skip(
        f"lodedb._native_core extension bridge is not available: {exc}",
        allow_module_level=True,
    )


def _onehot(axis: int, dim: int = 8) -> list[float]:
    vector = [0.0] * dim
    vector[axis] = 1.0
    return vector


def _loads(payload: str) -> dict:
    return json.loads(payload)


class _CountingEmbeddingBackend:
    """Hash backend that counts how many texts/queries it embeds."""

    required_model_name = None

    def __init__(self, native_dim: int = 384) -> None:
        from lodedb.engine.embedding_backends import HashEmbeddingBackend

        self.native_dim = native_dim
        self._inner = HashEmbeddingBackend(native_dim=native_dim)
        self.doc_texts_embedded = 0
        self.query_embeds = 0

    def embed_documents(self, texts):
        self.doc_texts_embedded += len(texts)
        return self._inner.embed_documents(texts)

    def embed_query(self, text):
        self.query_embeds += 1
        return self._inner.embed_query(text)


def _text_add_search_embed_counts(mode, write_mode, store_dir, monkeypatch):
    """Adds 8 text docs and runs one vector query, returning embed counts."""
    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", mode)
    if write_mode is None:
        monkeypatch.delenv("LODEDB_NATIVE_CORE_WRITE", raising=False)
    else:
        monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", write_mode)
    backend = _CountingEmbeddingBackend(native_dim=384)
    db = LodeDB(store_dir, _embedding_backend=backend)
    db.add_many(
        [
            {"id": f"doc-{i}", "text": f"launch code E-{1000 + i} report number {i}"}
            for i in range(8)
        ]
    )
    db.search("launch code", k=3, mode="vector")
    covered = db.stats()["native_core"]["covered"]
    db.close()
    return backend.doc_texts_embedded, backend.query_embeds, covered


def test_native_core_extension_executes_vector_store_flow() -> None:
    assert native_core.native_core_abi_version() == 1
    engine = native_core.CoreEngine()
    engine.create_index("default", 8, 4)
    mutation = _loads(
        engine.upsert_vectors(
            "default",
            json.dumps(
                [
                    {
                        "document_id": "a",
                        "vector": _onehot(0),
                        "metadata": {"topic": "ops"},
                        "text": None,
                    },
                    {
                        "document_id": "b",
                        "vector": _onehot(1),
                        "metadata": {"topic": "ml"},
                        "text": None,
                    },
                ]
            ),
        )
    )
    assert mutation["documents_upserted"] == 2

    hits = _loads(
        engine.query_vector(
            "default",
            json.dumps(_onehot(1)),
            2,
            json.dumps({"metadata": {"topic": "ml"}}),
        )
    )
    assert hits["hits"][0]["document_id"] == "b"
    assert hits["hits"][0]["metadata"] == {"topic": "ml"}

    stats = _loads(engine.stats("default"))
    assert stats["document_count"] == 2
    assert stats["raw_payload_text_present"] is False


@pytest.mark.skipif(
    os.name != "posix",
    reason="the native writer lock uses a BSD flock, implemented for unix only; "
    "non-unix native standalone writers are not a target and skip the OS lock",
)
def test_native_writer_lock_contends_with_python_writer(tmp_path) -> None:
    """A standalone native writer shares the single-writer lock with Python."""

    from lodedb import LodeDB

    store = tmp_path / "store"
    db = LodeDB.open_vector_store(store, vector_dim=8)
    options = json.dumps(
        {
            "path": str(store),
            "read_only": False,
            "durability": "relaxed",
            "commit_mode": "wal",
            "store_text": False,
            "index_text": False,
            "chunk_character_limit": 900,
            "acquire_writer_lock": True,
        }
    )
    previous = os.environ.get("LODEDB_PERSIST_LOCK_TIMEOUT")
    os.environ["LODEDB_PERSIST_LOCK_TIMEOUT"] = "0"
    try:
        # Python already holds <store>/.lodedb.lock; a native writer taking the
        # same shared lock must fail instead of opening a second writer. The
        # contended-lock CoreError maps to ValueError at the binding boundary.
        with pytest.raises(ValueError):
            native_core.CoreEngine.open(options)
    finally:
        if previous is None:
            os.environ.pop("LODEDB_PERSIST_LOCK_TIMEOUT", None)
        else:
            os.environ["LODEDB_PERSIST_LOCK_TIMEOUT"] = previous
        db.close()

    # After the Python writer closes, a native writer can take the lock.
    os.environ["LODEDB_PERSIST_LOCK_TIMEOUT"] = "0"
    try:
        native_engine = native_core.CoreEngine.open(options)
    finally:
        os.environ.pop("LODEDB_PERSIST_LOCK_TIMEOUT", None)
    del native_engine


def test_native_core_extension_apply_text_upsert_array_handles_empty_embeddings() -> None:
    """Re-applying text with no new chunks sends an empty array and must not panic."""

    engine = native_core.CoreEngine()
    engine.create_index("text", 8, 4)
    documents = json.dumps([{"document_id": "d", "text": "hello world", "metadata": {}}])
    plan = _loads(engine.prepare_text_upsert("text", documents, True, True, 900))
    engine.apply_text_upsert_array(
        json.dumps(plan), np.asarray([_onehot(0)], dtype=np.float32), 0.0
    )

    # The same id + content needs no re-embedding, so chunks_to_embed is empty
    # and the binding passes a (0, 0) float32 array; this must apply cleanly.
    plan_again = _loads(engine.prepare_text_upsert("text", documents, True, True, 900))
    assert plan_again["chunks_to_embed"] == []
    applied = _loads(
        engine.apply_text_upsert_array(
            json.dumps(plan_again), np.empty((0, 0), dtype=np.float32), 0.0
        )
    )
    assert applied["embedded_chunks"] == 0
    assert _loads(engine.stats("text"))["document_count"] == 1


def test_native_core_extension_array_vector_paths_match_json() -> None:
    """The array-input vector fast paths return the same hits as the JSON paths."""

    json_engine = native_core.CoreEngine()
    json_engine.create_index("default", 8, 4)
    array_engine = native_core.CoreEngine()
    array_engine.create_index("default", 8, 4)

    documents = [
        {"document_id": "a", "vector": _onehot(0), "metadata": {"topic": "ops"}, "text": None},
        {"document_id": "b", "vector": _onehot(1), "metadata": {"topic": "ml"}, "text": None},
        {"document_id": "c", "vector": _onehot(2), "metadata": {"topic": "ml"}, "text": None},
    ]
    json_mutation = _loads(json_engine.upsert_vectors("default", json.dumps(documents)))

    matrix = np.asarray([doc["vector"] for doc in documents], dtype=np.float32)
    sidecar = [
        {"document_id": doc["document_id"], "metadata": doc["metadata"], "text": doc["text"]}
        for doc in documents
    ]
    array_mutation = _loads(
        array_engine.upsert_vectors_array("default", matrix, json.dumps(sidecar))
    )
    assert array_mutation == json_mutation
    assert array_mutation["documents_upserted"] == 3

    query = np.asarray(_onehot(1), dtype=np.float32)
    json_hits = _loads(json_engine.query_vector("default", json.dumps(_onehot(1)), 2, None))
    array_hits = _loads(array_engine.query_vector_array("default", query, 2, None))
    assert array_hits == json_hits
    assert array_hits["hits"][0]["document_id"] == "b"

    # Filtered + batch array paths match the JSON paths byte for byte.
    filter_json = json.dumps({"metadata": {"topic": "ml"}})
    json_filtered = _loads(
        json_engine.query_vector("default", json.dumps(_onehot(2)), 3, filter_json)
    )
    array_filtered = _loads(
        array_engine.query_vector_array(
            "default", np.asarray(_onehot(2), dtype=np.float32), 3, filter_json
        )
    )
    assert array_filtered == json_filtered

    batch = np.asarray([_onehot(0), _onehot(1)], dtype=np.float32)
    json_batch = _loads(
        json_engine.query_vectors_batch(
            "default", json.dumps([_onehot(0), _onehot(1)]), 2, None
        )
    )
    array_batch = _loads(array_engine.query_vectors_batch_array("default", batch, 2, None))
    assert array_batch == json_batch


def test_native_core_extension_array_paths_handle_edge_inputs() -> None:
    """The hidden array PyO3 methods reject bad shapes and handle empties/dupes."""

    engine = native_core.CoreEngine()
    engine.create_index("default", 8, 4)

    # Empty batch query (0 rows, valid dim) returns no result rows.
    empty_batch = _loads(
        engine.query_vectors_batch_array(
            "default", np.empty((0, 8), dtype=np.float32), 3, None
        )
    )
    assert empty_batch == []

    # Mismatched sidecar length is rejected.
    with pytest.raises(ValueError):
        engine.upsert_vectors_array(
            "default",
            np.asarray([_onehot(0), _onehot(1)], dtype=np.float32),
            json.dumps([{"document_id": "only-one", "metadata": {}, "text": None}]),
        )

    # Non-contiguous input is rejected with a typed error.
    non_contiguous = np.zeros((2, 16), dtype=np.float32)[:, ::2]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    with pytest.raises(ValueError):
        engine.query_vectors_batch_array("default", non_contiguous, 1, None)

    # A duplicate document id within one array batch collapses to last-wins.
    mutation = _loads(
        engine.upsert_vectors_array(
            "default",
            np.asarray([_onehot(0), _onehot(1)], dtype=np.float32),
            json.dumps(
                [
                    {"document_id": "dup", "metadata": {"v": "0"}, "text": None},
                    {"document_id": "dup", "metadata": {"v": "1"}, "text": None},
                ]
            ),
        )
    )
    assert mutation["documents_upserted"] == 2
    assert _loads(engine.stats("default"))["document_count"] == 1
    hit = _loads(
        engine.query_vector_array("default", np.asarray(_onehot(1), dtype=np.float32), 1, None)
    )
    assert hit["hits"][0]["document_id"] == "dup"
    assert hit["hits"][0]["metadata"] == {"v": "1"}


def test_native_core_document_ids_filter_does_not_scale_with_corpus() -> None:
    """A small document_ids allowlist must not pay O(corpus) filter resolution.

    `resolve_filter` builds candidates straight from the requested ids rather than
    cloning the whole corpus, so a one-id filtered query costs the same as an
    unfiltered query (both pay only TurboVec's shared scan). The earlier
    clone-then-intersect made the filtered query several times slower than the
    unfiltered one at scale; this guards against that regression.
    """

    dim = 32
    n = 16000
    engine = native_core.CoreEngine()
    engine.create_index("default", dim, 4)
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    engine.upsert_vectors_array(
        "default",
        vectors,
        json.dumps([{"document_id": f"d{i}", "metadata": {}, "text": None} for i in range(n)]),
    )
    query = vectors[0]
    one_id_filter = json.dumps({"document_ids": ["d0"]})

    # Correctness: the one-id allowlist returns exactly that document.
    hit = _loads(engine.query_vector_array("default", query, 5, one_id_filter))
    assert [h["document_id"] for h in hit["hits"]] == ["d0"]

    def median_ms(filter_json) -> float:
        for _ in range(20):  # warm the quantized layout + caches
            engine.query_vector_array("default", query, 5, filter_json)
        samples = []
        for _ in range(80):
            start = time.perf_counter()
            engine.query_vector_array("default", query, 5, filter_json)
            samples.append((time.perf_counter() - start) * 1000.0)
        return statistics.median(samples)

    filtered = median_ms(one_id_filter)
    unfiltered = median_ms(None)
    # Generous bound: the fixed path is ~1x unfiltered; a corpus clone regression
    # was 5-8x at this size. 4x cleanly separates the two without CI flakiness.
    assert filtered <= unfiltered * 4.0 + 0.05, (
        f"one-id document_ids query {filtered:.4f} ms vs unfiltered {unfiltered:.4f} ms "
        "suggests O(corpus) filter resolution"
    )


def test_native_core_extension_accepts_index_create_options() -> None:
    engine = native_core.CoreEngine()
    index_key = "6f78dec251fa5e544784ac1af95b0ae6530cad714a2d34f8c4615740ecbf8205"
    engine.create_index_with_options(
        json.dumps(
            {
                "index_id": "default",
                "index_key": index_key,
                "client_id_hash": index_key,
                "name": "lodedb-local",
                "model": "external",
                "provider": "external",
                "task": "vector-only",
                "route_profile": "vector-only",
                "storage_profile": "turbovec_direct",
                "vector_dim": 8,
                "bit_width": 4,
            }
        )
    )
    stats = _loads(engine.stats("default"))
    assert stats["document_count"] == 0


def test_native_core_extension_executes_text_prepare_apply_flow() -> None:
    engine = native_core.CoreEngine()
    engine.create_index("text", 8, 4)
    plan = _loads(
        engine.prepare_text_upsert(
            "text",
            json.dumps(
                [
                    {
                        "document_id": "doc-alpha",
                        "text": "Alpha launch notes mention error code E-1001.",
                        "metadata": {"topic": "ops"},
                    }
                ]
            ),
            True,
            True,
            900,
        )
    )
    assert plan["chunks_to_embed"][0]["document_id"] == "doc-alpha"
    assert plan["chunks_to_embed"][0]["text"] == "Alpha launch notes mention error code E-1001."

    applied = _loads(
        engine.apply_text_upsert_array(
            json.dumps(plan),
            np.asarray([_onehot(3)], dtype=np.float32),
            1.25,
        )
    )
    assert applied["embedded_chunks"] == 1
    assert applied["embedding_time_ms"] == 1.25
    assert _loads(engine.get_document_text("text", "doc-alpha")) == (
        "Alpha launch notes mention error code E-1001."
    )
    assert _loads(engine.get_document_texts("text", json.dumps(["doc-alpha", "missing"]))) == {
        "doc-alpha": "Alpha launch notes mention error code E-1001."
    }
    record = _loads(engine.get_document("text", "doc-alpha"))
    assert record["document_id"] == "doc-alpha"
    assert record["metadata"] == {"topic": "ops"}
    assert record["chunk_count"] == 1
    assert "text" not in record
    assert _loads(
        engine.list_documents("text", json.dumps({"metadata": {"topic": "ops"}}))
    )[0]["document_id"] == "doc-alpha"

    query_plan = _loads(engine.prepare_query_text("E-1001", "vector"))
    assert query_plan["requires_embedding"] is True
    hits = _loads(
        engine.search_embedded_text_array(
            "text",
            json.dumps(query_plan),
            np.asarray(_onehot(3), dtype=np.float32),
            1,
            json.dumps({"metadata": {"topic": "ops"}}),
        )
    )
    assert hits["hits"][0]["document_id"] == "doc-alpha"

    lexical_plan = _loads(engine.prepare_query_text("E-1001", "lexical"))
    assert lexical_plan["requires_embedding"] is False
    lexical_hits = _loads(
        engine.search_embedded_text(
            "text",
            json.dumps(lexical_plan),
            None,
            1,
            json.dumps({"metadata": {"topic": "ops"}}),
        )
    )
    assert lexical_hits["hits"][0]["document_id"] == "doc-alpha"

    hybrid_plan = _loads(engine.prepare_query_text("E-1001", "hybrid"))
    assert hybrid_plan["requires_embedding"] is True
    hybrid_hits = _loads(
        engine.search_embedded_text(
            "text",
            json.dumps(hybrid_plan),
            json.dumps(_onehot(3)),
            1,
            json.dumps({"metadata": {"topic": "ops"}}),
        )
    )
    assert hybrid_hits["hits"][0]["document_id"] == "doc-alpha"


def test_native_core_extension_opens_readonly_persisted_vector_fixture(tmp_path) -> None:
    source = Path(__file__).resolve().parent / "fixtures" / "persisted" / "v0_4_store_text"
    store = tmp_path / "store"
    shutil.copytree(source, store)
    options = {
        "path": str(store),
        "read_only": True,
        "durability": "relaxed",
        "commit_mode": "generation",
        "store_text": True,
        "index_text": False,
    }

    engine = native_core.CoreEngine.open_readonly(str(store), json.dumps(options))
    stats = _loads(engine.stats("default"))
    assert stats["document_count"] == 3

    hits = _loads(
        engine.query_vector(
            "default",
            json.dumps(_onehot(0)),
            3,
            None,
        )
    )
    assert hits["hits"][0]["document_id"] == "vec-alpha"


def test_native_core_extension_written_vector_store_opens_in_python(tmp_path) -> None:
    from lodedb import LodeDB

    index_key = "6f78dec251fa5e544784ac1af95b0ae6530cad714a2d34f8c4615740ecbf8205"
    options = {
        "path": str(tmp_path),
        "read_only": False,
        "durability": "relaxed",
        "commit_mode": "generation",
        "store_text": True,
        "index_text": False,
    }
    engine = native_core.CoreEngine.open(json.dumps(options))
    engine.create_index_with_options(
        json.dumps(
            {
                "index_id": "default",
                "index_key": index_key,
                "client_id_hash": index_key,
                "name": "lodedb-local",
                "model": "external",
                "provider": "external",
                "task": "vector-only",
                "route_profile": "vector-only",
                "storage_profile": "turbovec_direct",
                "vector_dim": 8,
                "bit_width": 4,
            }
        )
    )
    mutation = _loads(
        engine.upsert_vectors(
            "default",
            json.dumps(
                [
                    {
                        "document_id": "native-a",
                        "vector": _onehot(0),
                        "metadata": {"kind": "native"},
                        "text": "Native retained text.",
                    },
                    {
                        "document_id": "native-b",
                        "vector": _onehot(1),
                        "metadata": {"kind": "native"},
                        "text": None,
                    },
                ]
            ),
        )
    )
    assert mutation["documents_upserted"] == 2
    engine.persist()
    engine.close()

    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    hit = db.search_by_vector(_onehot(1), k=1)[0]
    assert hit.id == "native-b"
    assert db.get("native-a") == "Native retained text."


def test_native_core_write_shadow_verifies_counts(tmp_path, monkeypatch) -> None:
    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "shadow")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "shadow")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    db.add_vectors(_onehot(0), id="shadow-a", metadata={"kind": "shadow"})
    db.add_vectors(_onehot(1), id="shadow-b", metadata={"kind": "shadow"})

    db.persist()
    stats = db.stats()["native_core"]

    assert stats["write_mode"] == "shadow"
    assert stats["version"]
    assert stats["abi_version"] == 1
    assert stats["shadow_persist_count"] == 1
    assert stats["shadow_persist_verified"] is True
    assert db.search_by_vector(_onehot(1), k=1)[0].id == "shadow-b"


def test_default_native_on_text_does_not_double_embed(tmp_path, monkeypatch) -> None:
    off_add, off_q, _ = _text_add_search_embed_counts("off", None, tmp_path / "off", monkeypatch)
    on_add, on_q, on_cov = _text_add_search_embed_counts("on", None, tmp_path / "on", monkeypatch)

    # Native still covers vectors (the mirror runs) so search_by_vector stays
    # native-authoritative, but the default read-on path must not pay for a
    # second model inference on either add or query while Python is the oracle.
    assert on_cov is True
    assert on_add == off_add
    assert on_q == off_q


def test_shadow_mode_runs_native_text_query_for_parity(tmp_path, monkeypatch) -> None:
    off_add, off_q, _ = _text_add_search_embed_counts("off", None, tmp_path / "off", monkeypatch)
    sh_add, sh_q, _ = _text_add_search_embed_counts(
        "shadow", None, tmp_path / "shadow", monkeypatch
    )

    # Validation modes intentionally pay for the cross-check: the native text
    # query runs (re-embedding the query) so parity can be verified, while the
    # write mirror still shares the writer's document embeddings.
    assert sh_add == off_add
    assert sh_q > off_q


def test_native_core_write_on_vector_store_persists_python_readable_store(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8, commit_mode="generation")
    db.add_vectors(_onehot(0), id="write-a", metadata={"kind": "write"}, text="Native write A")
    db.add_vectors(_onehot(1), id="write-b", metadata={"kind": "write"}, text="Native write B")
    stats = db.stats()["native_core"]

    assert stats["write_mode"] == "on"
    assert stats["write_through"] is True
    assert stats["covered"] is True
    assert db.search_by_vector(_onehot(1), k=1)[0].id == "write-b"
    db.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    reopened = LodeDB.open_vector_store(tmp_path, vector_dim=8, commit_mode="generation")
    assert reopened.search_by_vector(_onehot(1), k=1)[0].id == "write-b"
    assert reopened.get("write-a") == "Native write A"


def test_native_core_write_on_vector_store_wal_mode_persists_python_readable_store(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    db.add_vectors(_onehot(0), id="wal-a", metadata={"kind": "wal"}, text="Native WAL A")
    db.add_vectors(_onehot(1), id="wal-b", metadata={"kind": "wal"}, text="Native WAL B")
    stats = db.stats()["native_core"]

    assert stats["write_mode"] == "on"
    assert stats["write_through"] is True
    assert stats["covered"] is True
    assert db.search_by_vector(_onehot(1), k=1)[0].id == "wal-b"
    db.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    reopened = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    assert reopened.search_by_vector(_onehot(1), k=1)[0].id == "wal-b"
    assert reopened.get("wal-a") == "Native WAL A"


def test_native_write_through_generation_commits_are_o_changed(tmp_path, monkeypatch) -> None:
    """Native generation write-through must publish O(changed) deltas, not full bases.

    Native is the sole durable writer in write-through (Python defers), so each
    single-row add appends a generation delta onto native's own base. Per-add
    latency must stay flat as the corpus grows instead of the prior O(corpus)
    full-base rewrite (which grew to ~100 ms/add at ~1k rows).
    """
    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")
    dim = 16
    rng = np.random.default_rng(0)
    db = LodeDB.open_vector_store(tmp_path, vector_dim=dim, commit_mode="generation")

    def add_bucket(start, count):
        elapsed = []
        for i in range(start, start + count):
            vec = rng.standard_normal(dim).astype(np.float32).tolist()
            began = time.perf_counter()
            db.add_vectors(vec, id=f"d{i}", metadata={"u": str(i)}, normalize=False)
            elapsed.append((time.perf_counter() - began) * 1000.0)
        return statistics.median(elapsed)

    first = add_bucket(0, 200)
    for i in range(200, 1800):
        db.add_vectors(
            rng.standard_normal(dim).astype(np.float32).tolist(), id=f"d{i}", normalize=False
        )
    last = add_bucket(1800, 200)
    assert db.stats()["native_core"]["write_through"] is True
    db.close()

    # Flat within a generous bound: a full-base regression would be many times
    # slower at the tail than at the head, while O(changed) deltas stay level.
    assert last <= first * 4.0 + 1.0, (
        f"per-add latency grew: first={first:.3f} ms last={last:.3f} ms"
    )


def test_native_write_through_generation_churn_round_trips(tmp_path, monkeypatch) -> None:
    """Add/update/delete churn under native write-through reopens consistently.

    Exercises the generation delta path (state + tvim + removed-id tracking) and
    confirms the native-authored base+deltas reopen correctly under BOTH the native
    reader and the Python reader (cross-read), with no tvim/state row drift."""
    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")
    dim = 8
    rng = np.random.default_rng(7)

    def vec():
        return rng.standard_normal(dim).astype(np.float32).tolist()

    oracle: dict[str, list[float]] = {}
    db = LodeDB.open_vector_store(tmp_path, vector_dim=dim, commit_mode="generation")
    for _ in range(400):
        if rng.random() < 0.65 or not oracle:
            doc_id = f"d{int(rng.integers(0, 120))}"  # reuse ids -> updates
            v = vec()
            db.add_vectors(
                v, id=doc_id, metadata={"b": str(int(rng.integers(0, 8)))}, normalize=False
            )
            oracle[doc_id] = v
        else:
            doc_id = list(oracle)[int(rng.integers(0, len(oracle)))]
            db.remove(doc_id)
            oracle.pop(doc_id, None)
    db.close()

    for mode in ("on", "off"):
        monkeypatch.setenv("LODEDB_NATIVE_CORE", mode)
        ro = LodeDB.open_vector_store(tmp_path, vector_dim=dim, commit_mode="generation")
        assert ro.stats()["document_count"] == len(oracle), f"[{mode}] count mismatch"
        sample = list(oracle)[0]
        hits = ro.search_by_vector(oracle[sample], k=1)
        assert hits and hits[0].id == sample, f"[{mode}] search top != {sample}"
        ro.close()


def test_native_write_through_cross_thread_fallback_preserves_writes(tmp_path, monkeypatch) -> None:
    """A cross-thread write under native write-through must not be lost.

    Native is thread-confined and owns durability in generation write-through, so a
    write from a non-owner thread disables native. The Python engine must then
    reclaim durability and republish the current state; otherwise the acknowledged
    cross-thread mutation (applied to Python memory but with native's commit
    deferred) would vanish on reopen.
    """
    import threading

    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8, commit_mode="generation")
    db.add_vectors(_onehot(0), id="main", normalize=False)

    def add_from_other_thread() -> None:
        db.add_vectors(_onehot(1), id="thread", normalize=False)

    worker = threading.Thread(target=add_from_other_thread)
    worker.start()
    worker.join()
    db.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    reopened = LodeDB.open_vector_store(tmp_path, vector_dim=8, commit_mode="generation")
    assert reopened.stats()["document_count"] == 2
    assert reopened.search_by_vector(_onehot(0), k=1)[0].id == "main"
    assert reopened.search_by_vector(_onehot(1), k=1)[0].id == "thread"


def test_native_core_on_existing_vector_store_uses_readonly_seed(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    writer = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    writer.add_vectors(_onehot(0), id="seed-a", metadata={"kind": "seed"})
    writer.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    stats = db.stats()["native_core"]

    assert stats["enabled"] is True
    assert stats["covered"] is True
    assert db.search_by_vector(_onehot(0), k=1)[0].id == "seed-a"


def test_native_core_write_on_existing_vector_store_uses_writable_seed(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    writer = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    writer.add_vectors(_onehot(0), id="seed-a", metadata={"kind": "seed"})
    writer.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")

    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    db.add_vectors(_onehot(1), id="seed-b", metadata={"kind": "seed"})
    stats = db.stats()["native_core"]

    assert stats["enabled"] is True
    assert stats["covered"] is True
    assert stats["write_through"] is True
    assert db.search_by_vector(_onehot(1), k=1)[0].id == "seed-b"
    db.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    reopened = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    assert reopened.search_by_vector(_onehot(1), k=1)[0].id == "seed-b"


def test_native_core_on_existing_index_text_store_uses_readonly_seed(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB
    from lodedb.engine.embedding_backends import HashEmbeddingBackend

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    writer = LodeDB(
        tmp_path,
        index_text=True,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    writer.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    writer.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    db = LodeDB(
        tmp_path,
        index_text=True,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    stats = db.stats()["native_core"]

    assert stats["enabled"] is True
    assert stats["covered"] is True
    assert db.search("Alpha", k=1, mode="lexical")[0].id == "doc-alpha"


def test_native_core_write_on_existing_index_text_store_uses_writable_seed(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB
    from lodedb.engine.embedding_backends import HashEmbeddingBackend

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    writer = LodeDB(
        tmp_path,
        index_text=True,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    writer.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    writer.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")

    db = LodeDB(
        tmp_path,
        index_text=True,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    db.add("Beta launch notes mention error code E-2002.", id="doc-beta")
    stats = db.stats()["native_core"]

    assert stats["enabled"] is True
    assert stats["covered"] is True
    assert stats["write_through"] is True
    assert db.search("Beta", k=1, mode="lexical")[0].id == "doc-beta"
    db.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    reopened = LodeDB(
        tmp_path,
        index_text=True,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    assert reopened.search("Beta", k=1, mode="lexical")[0].id == "doc-beta"


def test_native_core_on_existing_raw_text_store_uses_readonly_seed(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB
    from lodedb.engine.embedding_backends import HashEmbeddingBackend

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    writer = LodeDB(
        tmp_path,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    writer.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    writer.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    db = LodeDB(
        tmp_path,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    stats = db.stats()["native_core"]

    assert stats["enabled"] is True
    assert stats["covered"] is True
    assert db.search("Alpha", k=1, mode="lexical")[0].id == "doc-alpha"


def test_native_core_on_text_store_uses_native_query_without_write_through(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB
    from lodedb.engine.embedding_backends import HashEmbeddingBackend

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.delenv("LODEDB_NATIVE_CORE_WRITE", raising=False)
    db = LodeDB(tmp_path, _embedding_backend=HashEmbeddingBackend(native_dim=384))
    db.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    stats = db.stats()["native_core"]

    assert stats["enabled"] is True
    assert stats["covered"] is True
    assert stats["write_through"] is False
    assert db.search("Alpha", k=1, mode="lexical")[0].id == "doc-alpha"


def test_native_core_write_on_text_store_persists_python_readable_store(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB
    from lodedb.engine.embedding_backends import HashEmbeddingBackend

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")
    backend = HashEmbeddingBackend(native_dim=384)
    db = LodeDB(tmp_path, _embedding_backend=backend, commit_mode="generation")
    db.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    stats = db.stats()["native_core"]

    assert stats["write_mode"] == "on"
    assert stats["write_through"] is True
    assert stats["covered"] is True
    assert db.search("Alpha", k=1, mode="lexical")[0].id == "doc-alpha"
    db.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    reopened = LodeDB(
        tmp_path,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
        commit_mode="generation",
    )
    assert reopened.get("doc-alpha") == "Alpha launch notes mention error code E-1001."
    assert reopened.search("Alpha", k=1, mode="lexical")[0].id == "doc-alpha"


def test_native_core_write_on_text_store_wal_mode_persists_python_readable_store(
    tmp_path, monkeypatch
) -> None:
    from lodedb import LodeDB
    from lodedb.engine.embedding_backends import HashEmbeddingBackend

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "on")
    monkeypatch.setenv("LODEDB_NATIVE_CORE_WRITE", "on")
    backend = HashEmbeddingBackend(native_dim=384)
    db = LodeDB(tmp_path, _embedding_backend=backend)
    db.add("Alpha launch notes mention error code E-1001.", id="doc-alpha")
    stats = db.stats()["native_core"]

    assert stats["write_mode"] == "on"
    assert stats["write_through"] is True
    assert stats["covered"] is True
    assert db.search("Alpha", k=1, mode="lexical")[0].id == "doc-alpha"
    db.close()

    monkeypatch.setenv("LODEDB_NATIVE_CORE", "off")
    reopened = LodeDB(
        tmp_path,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    assert reopened.get("doc-alpha") == "Alpha launch notes mention error code E-1001."
    assert reopened.search("Alpha", k=1, mode="lexical")[0].id == "doc-alpha"


def test_native_core_adapter_can_discover_extension_when_installed(monkeypatch) -> None:
    module = importlib.import_module("lodedb._native_core")
    monkeypatch.setattr("importlib.import_module", lambda name: module)

    from lodedb.engine.native_adapter import NativeCoreAdapter

    assert NativeCoreAdapter().available is True
