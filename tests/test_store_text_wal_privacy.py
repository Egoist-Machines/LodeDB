"""store_text=False must keep raw text off disk, including the write-ahead log.

WAL replay re-embeds a text document from its logged body, so the writer's WAL is
a place raw text could otherwise leak even when store_text=False. These tests
assert that no caption or document body ever reaches any on-disk file under
store_text=False, across the vector-in, image, and text-in paths.
"""

from __future__ import annotations

import gc

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend, hash_embedding

SECRET = b"TOPSECRETRAWTEXT"
DIM = 8


def _files_containing(root, needle: bytes) -> list[str]:
    """Returns the on-disk files under root whose bytes contain needle.

    Skips the single-writer lock sentinel (``.lodedb.lock``): it carries no payload,
    and on Windows the live writer holds it open with a byte-range lock, so reading
    it raises PermissionError (a portability quirk, not a leak). The WAL and every
    data sidecar are opened-and-closed per write, so they stay in scope; a
    PermissionError on anything else is tolerated defensively for the same reason.
    """

    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name == ".lodedb.lock":
            continue
        try:
            data = path.read_bytes()
        except PermissionError:
            continue
        if needle in data:
            hits.append(str(path.relative_to(root)))
    return hits


class _FakeClip(HashEmbeddingBackend):
    """A text + image fake backend (no model download) for the image path."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__(native_dim=dim)
        self.name = "clip"
        self.required_model_name = "fake-clip"  # public embedder= requires an identity

    def embed_images(self, images):
        return tuple(hash_embedding(str(image), self.native_dim) for image in images)


def test_add_vectors_caption_no_leak_under_wal(tmp_path):
    # A vector-only index keeps the WAL (no text-in path); the optional vector
    # caption must be dropped from the WAL payload under store_text=False.
    db = LodeDB.open_vector_store(tmp_path, vector_dim=DIM, store_text=False, commit_mode="wal")
    db.add_vectors([1, 0, 0, 0, 0, 0, 0, 0], id="a", text=SECRET.decode())
    assert db.commit_mode == "wal"  # stayed on the WAL, did not silently fall back
    assert _files_containing(tmp_path, SECRET) == []
    db.close()


def test_add_image_caption_no_leak(tmp_path):
    # An image add keeps the WAL; the caption is dropped from the vector WAL payload
    # when store_text is off, so no raw text reaches disk.
    db = LodeDB(path=tmp_path, embedder=_FakeClip(), store_text=False)
    db.add_image("photo-1", id="a", text=SECRET.decode())
    assert db.commit_mode == "wal"
    assert _files_containing(tmp_path, SECRET) == []
    db.close()


def test_add_text_in_no_leak(tmp_path):
    db = LodeDB(
        path=tmp_path,
        model="minilm",
        store_text=False,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    assert db.commit_mode == "wal"  # text-in + store_text=False keeps the WAL now
    db.add(SECRET.decode(), id="d")
    assert _files_containing(tmp_path, SECRET) == []  # WAL logs embeddings, not text
    db.close()


def test_store_text_false_text_recovers_from_wal_without_raw_text(tmp_path):
    # store_text=False + WAL text-in: the WAL logs the chunk embeddings, not the
    # body. A crash leaves the WAL on disk; reopening replays it (no re-embedding)
    # to the identical index, and no raw text ever reaches disk.
    secret_doc = "alpha " + SECRET.decode() + " beta gamma delta " * 80  # multi-chunk
    db = LodeDB(
        path=tmp_path,
        model="minilm",
        store_text=False,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    assert db.commit_mode == "wal"
    db.add(secret_doc, id="long")
    db.add("a short note", id="short")
    live = [(h.id, round(h.score, 6)) for h in db.search("alpha beta", k=2)]
    assert db.get_document("long")["chunk_count"] >= 2  # exercises the multi-chunk delta
    assert _files_containing(tmp_path, SECRET) == []

    # Crash: drop the handle, leaving the uncheckpointed WAL on disk; the native
    # engine's writer lock is released on its worker so the test can reopen.
    del db
    gc.collect()

    recovered = LodeDB(
        path=tmp_path,
        model="minilm",
        store_text=False,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        assert recovered.count() == 2
        assert recovered.get_document("long")["chunk_count"] >= 2
        # Replay rebuilt the identical index from the logged embeddings.
        assert [(h.id, round(h.score, 6)) for h in recovered.search("alpha beta", k=2)] == live
        assert _files_containing(tmp_path, SECRET) == []  # still no raw text after replay
    finally:
        recovered.close()


def test_index_text_lexical_recovers_from_wal(tmp_path):
    # index_text=True + store_text=False + WAL: the embedded WAL must carry the
    # per-chunk lexical tokens, or a crash recovers vectors while leaving the
    # lexical postings empty for the recovered docs (silent lexical misses).
    def _open():
        return LodeDB(
            path=tmp_path,
            model="minilm",
            store_text=False,
            index_text=True,
            _embedding_backend=HashEmbeddingBackend(native_dim=384),
        )

    db = _open()
    assert db.commit_mode == "wal"
    db.add("turbine fault code E1234 logged overnight", id="d")
    db.add("a routine maintenance note", id="e")
    assert [h.id for h in db.search("E1234", k=5, mode="lexical")] == ["d"]

    # Crash: drop the handle, leaving the uncheckpointed WAL on disk; the native
    # engine's writer lock is released on its worker so the test can reopen.
    del db
    gc.collect()

    recovered = _open()
    try:
        assert recovered.count() == 2
        # Pre-fix this returned [] (vectors recovered, lexical postings lost).
        assert [h.id for h in recovered.search("E1234", k=5, mode="lexical")] == ["d"]
    finally:
        recovered.close()


def test_store_text_true_retains_caption_under_wal(tmp_path):
    # The complement: with store_text=True the caption is intentionally retained
    # (and replay must keep working), so the WAL drop is conditioned on store_text.
    db = LodeDB.open_vector_store(tmp_path, vector_dim=DIM, store_text=True, commit_mode="wal")
    db.add_vectors([1, 0, 0, 0, 0, 0, 0, 0], id="a", text="kept caption")
    assert db.get_text("a") == "kept caption"
    db.close()
    reopened = LodeDB.open_vector_store(
        tmp_path, vector_dim=DIM, store_text=True, commit_mode="wal"
    )
    assert reopened.get_text("a") == "kept caption"  # survived WAL replay
    reopened.close()
