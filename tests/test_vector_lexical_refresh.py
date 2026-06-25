"""Vector/image upserts must refresh lexical postings, not leave stale ones.

Replacing a text document with a vector (or image) of the same id used to leave
the replaced document's tokens in ``state.document_tokens``, so
``mode="lexical"``/``"hybrid"`` kept matching the old body. A vector document now
sets its tokens from the optional ``text=`` caption (which *should* be lexically
searchable when ``index_text=True``) or to an empty list when there is no caption,
which clears the stale postings. These run with ``index_text=True, store_text=False``
so the lexical index is the persisted-token path, not the raw-text one.
"""

from __future__ import annotations

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend, hash_embedding

DIM = 384


def _open(tmp_path) -> LodeDB:
    return LodeDB(
        path=tmp_path,
        model="minilm",
        store_text=False,
        index_text=True,
        _embedding_backend=HashEmbeddingBackend(native_dim=DIM),
    )


def _vec(i: int) -> list[float]:
    vector = [0.0] * DIM
    vector[i] = 1.0
    return vector


def test_text_then_vector_upsert_clears_stale_lexical(tmp_path):
    db = _open(tmp_path)
    db.add("turbine fault code E1234 logged overnight", id="d")
    assert [h.id for h in db.search("E1234", k=5, mode="lexical")] == ["d"]

    # Replace the text document with a captionless vector at the same id.
    db.add_vectors(_vec(0), id="d")
    # Same handle: the stale posting is gone.
    assert db.search("E1234", k=5, mode="lexical") == []
    db.persist()
    db.close()

    # And after reopen (the .tvlex delta journaled the cleared tokens).
    reopened = _open(tmp_path)
    assert reopened.search("E1234", k=5, mode="lexical") == []
    assert reopened.count() == 1
    reopened.close()


def test_vector_caption_participates_in_lexical(tmp_path):
    db = _open(tmp_path)
    db.add_vectors(_vec(0), id="v", text="diagnostic error E1234 captured")
    # The caption is lexically searchable even though store_text=False (index_text
    # persists payload-derived tokens, not raw text).
    assert [h.id for h in db.search("E1234", k=5, mode="lexical")] == ["v"]
    db.persist()
    db.close()

    reopened = _open(tmp_path)
    assert [h.id for h in reopened.search("E1234", k=5, mode="lexical")] == ["v"]
    reopened.close()


def test_vector_caption_lexical_recovers_from_wal(tmp_path):
    # store_text=False keeps the raw caption out of the WAL, but its tokens are
    # logged (payload-derived), so a crash before checkpoint recovers the caption's
    # lexical postings rather than leaving them empty.
    db = _open(tmp_path)
    assert db.commit_mode == "wal"
    db.add_vectors(_vec(0), id="v", text="sensor fault code E1234 captured")
    assert [h.id for h in db.search("E1234", k=5, mode="lexical")] == ["v"]

    # Crash: leave the uncheckpointed WAL on disk, release the lock to reopen.
    db._engine._release_writer_lock()
    del db

    recovered = _open(tmp_path)
    try:
        assert [h.id for h in recovered.search("E1234", k=5, mode="lexical")] == ["v"]
    finally:
        recovered.close()


class _FakeClip(HashEmbeddingBackend):
    """Text + image fake backend (no download) for the image-replacement path."""

    def __init__(self) -> None:
        super().__init__(native_dim=DIM)
        self.required_model_name = "fake-clip"

    def embed_images(self, images):
        return tuple(hash_embedding(str(image), DIM) for image in images)


def test_image_upsert_clears_stale_text_lexical(tmp_path):
    db = LodeDB(
        path=tmp_path,
        embedder=_FakeClip(),
        store_text=False,
        index_text=True,
    )
    db.add("incident note with serial ABC-789 attached", id="d")
    assert [h.id for h in db.search("ABC-789", k=5, mode="lexical")] == ["d"]
    # Replace the text doc with an image (no caption) -> stale lexical cleared.
    db.add_image("photo", id="d")
    assert db.search("ABC-789", k=5, mode="lexical") == []
    db.close()
