"""store_text=False must keep raw text off disk, including the write-ahead log.

WAL replay re-embeds a text document from its logged body, so the writer's WAL is
a place raw text could otherwise leak even when store_text=False. These tests
assert that no caption or document body ever reaches any on-disk file under
store_text=False, across the vector-in, image, and text-in paths.
"""

from __future__ import annotations

import pytest

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend, hash_embedding

SECRET = b"TOPSECRETRAWTEXT"
DIM = 8


def _files_containing(root, needle: bytes) -> list[str]:
    """Returns the on-disk files under root whose bytes contain needle."""

    return [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and needle in path.read_bytes()
    ]


class _FakeClip(HashEmbeddingBackend):
    """A text + image fake backend (no model download) for the image path."""

    def __init__(self, dim: int = DIM) -> None:
        super().__init__(native_dim=dim)
        self.name = "clip"

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
    # A text-capable (embedder) index with store_text=False resolves to generation,
    # which persists compact codes with no raw text.
    db = LodeDB(path=tmp_path, embedder=_FakeClip(), store_text=False)
    db.add_image("photo-1", id="a", text=SECRET.decode())
    assert db.commit_mode == "generation"
    assert _files_containing(tmp_path, SECRET) == []
    db.close()


def test_add_text_in_no_leak(tmp_path):
    db = LodeDB(
        path=tmp_path,
        model="minilm",
        store_text=False,
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    assert db.commit_mode == "generation"  # text-in + store_text=False -> generation
    db.add(SECRET.decode(), id="d")
    assert _files_containing(tmp_path, SECRET) == []
    db.close()


def test_explicit_wal_with_store_text_false_text_index_rejected(tmp_path):
    with pytest.raises(ValueError, match="store_text=False"):
        LodeDB(
            path=tmp_path,
            model="minilm",
            store_text=False,
            commit_mode="wal",
            _embedding_backend=HashEmbeddingBackend(native_dim=384),
        )


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
