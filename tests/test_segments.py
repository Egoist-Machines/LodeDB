"""WAL segment files: store-free planning, encode/decode, and external folds.

`lodedb.local.segments` is the advanced building-block API for out-of-band
ingest (a cloud multi-writer pipeline): plan + embed + encode a segment with no
open store, then fold the bytes into a warm writable generation-mode handle and
publish one delta per fold batch. These tests drive the full flow through the
real engine with the deterministic hash embedding backend.
"""

from __future__ import annotations

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import LodeDB, ReadOnlyError
from lodedb.local.segments import (
    build_embedded_documents_record,
    decode_segment,
    delete_documents_record,
    encode_segment,
    fold_segment,
    plan_documents,
)

DIM = 384
SECRET = "TOPSECRETSEGMENTTEXT"


def _be() -> HashEmbeddingBackend:
    return HashEmbeddingBackend(native_dim=DIM)


def _writer(path) -> LodeDB:
    # External folds require a generation-mode handle: in WAL mode the engine
    # refuses apply (an external fold could strand unfolded local WAL records).
    return LodeDB(path=path, model="minilm", commit_mode="generation", _embedding_backend=_be())


def _add_segment(documents, *, store_text: bool = True, index_text: bool = True) -> bytes:
    """Plans, embeds (hash backend), and encodes one add segment."""

    plan = plan_documents(
        documents, store_text=store_text, index_text=index_text, chunk_character_limit=900
    )
    chunk_texts = tuple(str(chunk["text"]) for chunk in plan["chunks_to_embed"])
    embeddings = _be().embed_documents(chunk_texts)
    record = build_embedded_documents_record(plan, embeddings, vector_dim=DIM)
    return encode_segment([record])


def test_segment_fold_end_to_end(tmp_path):
    """plan -> embed -> encode -> fold -> persist -> search/get, watermark advanced."""

    # Create the target store, then reopen warm (the orchestrator's shape).
    _writer(tmp_path).close()
    segment = _add_segment(
        [
            {"text": "hello world from lodedb", "id": "a", "metadata": {"kind": "note"}},
            {"text": "a second segmented document", "id": "b"},
        ]
    )

    db = _writer(tmp_path)
    try:
        first_lsn = db.applied_lsn() + 1
        applied = fold_segment(db, segment, first_lsn=first_lsn)
        assert applied == 1  # one record covers the whole add batch
        assert db.applied_lsn() == first_lsn
        db.persist()
        hits = db.search("hello world from lodedb", k=2)
        assert any(hit.id == "a" for hit in hits)
        assert db.get("a") == "hello world from lodedb"
        assert db.count() == 2
    finally:
        db.close()

    # The fold survives a reopen (it was published by the persist).
    reopened = _writer(tmp_path)
    try:
        assert reopened.count() == 2
        assert reopened.get("b") == "a second segmented document"
    finally:
        reopened.close()


def test_refold_is_idempotent(tmp_path):
    _writer(tmp_path).close()
    segment = _add_segment([{"text": "idempotent fold text", "id": "a"}])

    db = _writer(tmp_path)
    try:
        first_lsn = db.applied_lsn() + 1
        assert fold_segment(db, segment, first_lsn=first_lsn) == 1
        db.persist()
        watermark = db.applied_lsn()
        # Same handle: everything at or below the watermark skips.
        assert fold_segment(db, segment, first_lsn=first_lsn) == 0
        assert db.applied_lsn() == watermark
        db.persist()  # watermark-safe second persist
    finally:
        db.close()

    reopened = _writer(tmp_path)
    try:
        # After reopen the refold is equally a no-op and nothing duplicates.
        assert fold_segment(reopened, segment, first_lsn=first_lsn) == 0
        reopened.persist()
        assert reopened.count() == 1
    finally:
        reopened.close()


def test_delete_segment_folds(tmp_path):
    _writer(tmp_path).close()
    add = _add_segment(
        [
            {"text": "keep this document", "id": "keep"},
            {"text": "drop this document", "id": "drop"},
        ]
    )
    remove = encode_segment([delete_documents_record("drop")])

    db = _writer(tmp_path)
    try:
        fold_segment(db, add, first_lsn=db.applied_lsn() + 1)
        db.persist()
        # The commit inflates the watermark past the last stamped LSN, so a later
        # batch must re-floor from the committed applied_lsn -- exactly what the
        # fold orchestrator does.
        assert fold_segment(db, remove, first_lsn=db.applied_lsn() + 1) == 1
        db.persist()
        assert db.get("drop") is None
        assert db.count() == 1
    finally:
        db.close()


def test_fold_failure_modes(tmp_path):
    _writer(tmp_path).close()
    segment = _add_segment([{"text": "failure mode fixture", "id": "a"}])

    # Truncated bytes fail the strict decode.
    db = _writer(tmp_path)
    try:
        with pytest.raises(Exception, match="torn|corrupt|header"):
            fold_segment(db, segment[:-3], first_lsn=db.applied_lsn() + 1)
        # A pre-stamped segment is refused (stamping is the fold's job).
        records = decode_segment(segment)
        stamped = [{**record, "lsn": 1} for record in records]
        with pytest.raises(Exception, match="must not carry|already carry"):
            encode_segment(stamped)
    finally:
        db.close()

    # Read-only handles must never fold.
    reader = LodeDB.open_readonly(tmp_path, model="minilm", _embedding_backend=_be())
    try:
        with pytest.raises(ReadOnlyError):
            fold_segment(reader, segment, first_lsn=1)
    finally:
        reader.close()

    # WAL-mode handles must fold their own tail, not external segments.
    wal_dir = tmp_path / "wal-mode"
    wal_db = LodeDB(path=wal_dir, model="minilm", commit_mode="wal", _embedding_backend=_be())
    try:
        assert wal_db.commit_mode == "wal"
        with pytest.raises(Exception, match="generation commit mode"):
            fold_segment(wal_db, segment, first_lsn=1)
    finally:
        wal_db.close()


def test_discard_abandons_a_partially_applied_fold(tmp_path):
    """A fold that raises mid-batch leaves partial state in memory only;
    ``discard()`` abandons it without persisting (a graceful ``close()`` would
    publish it) and releases the writer lock immediately."""

    _writer(tmp_path).close()
    plan = plan_documents([{"text": "partially applied survivor", "id": "partial"}])
    chunk_texts = tuple(str(chunk["text"]) for chunk in plan["chunks_to_embed"])
    good = build_embedded_documents_record(
        plan, _be().embed_documents(chunk_texts), vector_dim=DIM
    )
    # Record 1 applies, record 2 raises at apply time: a well-formed frame whose
    # payload fails validation inside the engine (missing `documents`).
    poison = encode_segment([good, {"op": "apply_embedded_documents", "payload": {}}])

    db = _writer(tmp_path)
    try:
        committed_lsn = db.applied_lsn()
        with pytest.raises(Exception, match="documents"):
            fold_segment(db, poison, first_lsn=committed_lsn + 1)
        # The poisoned in-memory state a graceful close() would persist.
        assert db.count() == 1
    finally:
        db.discard()

    # The discard released the writer lock (this open would fail otherwise) and
    # persisted nothing: the store is still at its committed (empty) state.
    reopened = _writer(tmp_path)
    try:
        assert reopened.count() == 0
        assert reopened.get("partial") is None
        assert reopened.applied_lsn() == committed_lsn
    finally:
        reopened.close()


def test_record_builder_failure_modes():
    plan = plan_documents([{"text": "validation fixture", "id": "a"}], store_text=True)
    chunk_count = len(plan["chunks_to_embed"])
    # Embedding count mismatch.
    with pytest.raises(Exception, match="count"):
        build_embedded_documents_record(plan, [], vector_dim=DIM)
    # Dimension mismatch.
    with pytest.raises(Exception, match="dimension"):
        build_embedded_documents_record(plan, [[0.5] * 8] * chunk_count, vector_dim=DIM)
    # Non-finite embedding.
    poisoned = [[float("nan")] * DIM] * chunk_count
    with pytest.raises(Exception, match="non-finite"):
        build_embedded_documents_record(plan, poisoned, vector_dim=DIM)
    # Non-native op fails at encode, before any upload could happen.
    with pytest.raises(Exception, match="does not support"):
        encode_segment([{"op": "upsert_documents", "payload": {"documents": []}}])
    # Input validation.
    with pytest.raises(ValueError):
        plan_documents([])
    with pytest.raises(ValueError):
        plan_documents([{"text": "no id"}])
    with pytest.raises(ValueError):
        plan_documents([{"text": "  ", "id": "a"}])
    with pytest.raises(ValueError):
        delete_documents_record([])
    with pytest.raises(ValueError):
        delete_documents_record([" "])


def test_store_text_false_keeps_raw_text_out_of_segment_bytes():
    segment = _add_segment(
        [{"text": f"{SECRET} body", "id": "a"}], store_text=False, index_text=False
    )
    assert SECRET.encode() not in segment
    # And the flag on keeps it (the retention contract, not a hashing accident).
    retained = _add_segment([{"text": f"{SECRET} body", "id": "a"}], store_text=True)
    assert SECRET.encode() in retained


def test_segment_record_matches_appender_wal_record(tmp_path):
    """The segment record is byte-for-byte the record Appender.append_text_many
    logs for the same input (one builder, zero drift)."""

    from lodedb.local.appender import Appender

    # Author the WAL record through the real appender (WAL-mode store).
    db = LodeDB(path=tmp_path, model="minilm", commit_mode="wal", _embedding_backend=_be())
    db.add("seed the store", id="seed")
    db.close()
    documents = [{"text": "parity check document", "id": "a", "metadata": {"kind": "note"}}]
    with Appender.open(tmp_path, embedder=_be()) as appender:
        appender.append_text_many(documents)
    wal_files = list(tmp_path.glob("*.wal"))
    assert len(wal_files) == 1
    # A WAL file and a segment share the frame format, so the strict decoder
    # reads the appended file directly.
    wal_records = decode_segment(wal_files[0].read_bytes())
    assert len(wal_records) == 1
    assert wal_records[0]["lsn"] is not None

    # Build the same record store-free with the appender's retention defaults.
    plan = plan_documents(documents, store_text=False, index_text=False)
    chunk_texts = tuple(str(chunk["text"]) for chunk in plan["chunks_to_embed"])
    record = build_embedded_documents_record(
        plan, _be().embed_documents(chunk_texts), vector_dim=DIM
    )
    assert record["op"] == wal_records[0]["op"]
    # Embeddings compare at f32 precision: reading the WAL text back through the
    # strict decoder re-parses each float with serde_json's fast (non-roundtrip)
    # parser, which can drift the widened f64 by one ulp. The byte-exact parity
    # of the encoded record against the appender's WAL file is pinned on the
    # Rust side (payload_builder_matches_appender_wal_record), where no text
    # re-parse is involved.
    import numpy as np

    def _split(payload):
        chunks = [dict(chunk) for chunk in payload["added_chunks"]]
        vectors = [chunk.pop("embedding") for chunk in chunks]
        rest = {key: value for key, value in payload.items() if key != "added_chunks"}
        return rest, chunks, vectors

    ours_rest, ours_chunks, ours_vectors = _split(record["payload"])
    wal_rest, wal_chunks, wal_vectors = _split(wal_records[0]["payload"])
    assert ours_rest == wal_rest
    assert ours_chunks == wal_chunks
    assert len(ours_vectors) == len(wal_vectors)
    for ours_vector, wal_vector in zip(ours_vectors, wal_vectors, strict=True):
        assert np.array_equal(
            np.asarray(ours_vector, dtype=np.float32),
            np.asarray(wal_vector, dtype=np.float32),
        )
