"""Tests for the multimodal image path (``add_image`` / ``search_by_image``).

The functional tests use a deterministic fake CLIP backend so they run offline
with no model download: it maps text and "images" (string tags here) into the
same hashed space, which is exactly the shared-space property that makes
cross-modal retrieval work. A separate test exercises the real ``"clip"`` preset
wiring by construction only (no encode, so no download).
"""

from __future__ import annotations

import pytest

from lodedb import LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend, hash_embedding
from lodedb.local.db import ImageEmbeddingUnsupportedError

DIM = 512


class _FakeClipBackend:
    """Deterministic stand-in for the CLIP backend (text + images, no download).

    Text and images are embedded by hashing a string key, so an image keyed
    ``"cat"`` and the text query ``"cat"`` land on the same vector. That mirrors
    CLIP's shared image/text space closely enough to test the cross-modal wiring.
    """

    name = "clip"
    required_model_name = None
    native_dim = DIM

    def embed_documents(self, texts):
        return tuple(hash_embedding(text, DIM) for text in texts)

    def embed_query(self, text):
        return hash_embedding(text, DIM)

    def embed_images(self, images):
        return tuple(hash_embedding(self._key(image), DIM) for image in images)

    @staticmethod
    def _key(image):
        if isinstance(image, (bytes, bytearray)):
            return bytes(image).decode("utf-8", "ignore")
        return str(image)


def _multimodal_db(path, **kwargs) -> LodeDB:
    return LodeDB(path=path, embedder=_FakeClipBackend(), **kwargs)


def test_add_image_and_cross_modal_text_search(tmp_path):
    db = _multimodal_db(tmp_path)
    db.add_image("cat", id="img-cat", metadata={"path": "cat.jpg"})
    db.add_image("dog", id="img-dog", metadata={"path": "dog.jpg"})
    assert db.count() == 2

    # A text query retrieves the matching image from the shared space.
    text_hits = db.search("cat", k=2)
    assert text_hits[0].id == "img-cat"
    assert text_hits[0].metadata == {"path": "cat.jpg"}

    # An image query retrieves the matching image.
    image_hits = db.search_by_image("dog", k=1)
    assert image_hits[0].id == "img-dog"


def test_add_image_stores_no_raw_bytes(tmp_path):
    db = _multimodal_db(tmp_path)
    db.add_image("cat", id="img-cat", metadata={"path": "cat.jpg"})
    # No raw image payload is retained; the path lives in metadata instead.
    assert db.get("img-cat") is None
    assert db.get_document("img-cat")["metadata"] == {"path": "cat.jpg"}


def test_add_image_optional_caption_text(tmp_path):
    db = _multimodal_db(tmp_path)
    db.add_image("cat", id="img-cat", text="a tabby cat on a sofa")
    assert db.get("img-cat") == "a tabby cat on a sofa"


def test_image_verbs_rejected_without_image_backend(tmp_path):
    # A text-only embedder (no embed_images) cannot serve image verbs.
    db = LodeDB(path=tmp_path, embedder=HashEmbeddingBackend(native_dim=DIM))
    with pytest.raises(ImageEmbeddingUnsupportedError):
        db.add_image("cat")
    with pytest.raises(ImageEmbeddingUnsupportedError):
        db.search_by_image("cat")


def test_image_verbs_rejected_on_vector_only_index(tmp_path):
    db = LodeDB.open_vector_store(tmp_path, vector_dim=DIM)
    with pytest.raises(ImageEmbeddingUnsupportedError):
        db.add_image("cat")


def test_clip_preset_builds_multimodal_index(tmp_path):
    # Constructs the real "clip" preset path: resolves the preset, route profile,
    # and ClipEmbeddingBackend. The model loads lazily on first encode, so opening
    # the index downloads nothing.
    db = LodeDB(path=tmp_path, model="clip")
    assert db.preset is not None
    assert db.preset.multimodal
    assert db._vector_dim == 512
    assert not db.vector_only
    assert callable(getattr(db._embedding_backend, "embed_images", None))
