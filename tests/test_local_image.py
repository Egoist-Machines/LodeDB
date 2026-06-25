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
    required_model_name = "fake-clip"
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


def _text_only_embedder() -> HashEmbeddingBackend:
    # A text-only embedder (no embed_images) with an identity, so construction
    # succeeds and the image verbs fail for the right reason.
    backend = HashEmbeddingBackend(native_dim=DIM)
    backend.required_model_name = "text-only"
    return backend


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


def test_add_images_batch(tmp_path):
    db = _multimodal_db(tmp_path)
    ids = db.add_images(
        [
            {"image": "cat", "id": "c", "metadata": {"path": "cat.jpg"}},
            {"image": "dog", "id": "d", "text": "a dog"},
            {"image": "bird"},  # auto id
        ]
    )
    assert ids[0] == "c" and ids[1] == "d"
    assert ids[2].startswith("doc-")
    assert db.count() == 3
    # The batched encode lands in the same space as single add_image / text.
    assert db.search("cat", k=1)[0].id == "c"
    assert db.search_by_image("dog", k=1)[0].id == "d"
    assert db.get("d") == "a dog"  # caption retained (store_text defaults True)


def test_add_images_requires_image_key(tmp_path):
    db = _multimodal_db(tmp_path)
    with pytest.raises(ValueError, match="image"):
        db.add_images([{"id": "x"}])


def test_add_images_rejected_without_image_backend(tmp_path):
    db = LodeDB(path=tmp_path, embedder=_text_only_embedder())
    with pytest.raises(ImageEmbeddingUnsupportedError):
        db.add_images([{"image": "cat"}])


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
    db = LodeDB(path=tmp_path, embedder=_text_only_embedder())
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


def test_clip_backend_rejects_oversized_image(monkeypatch):
    # The decode guard rejects an image whose pixel count exceeds the limit, before
    # the full decode. Uses a tiny limit so a small image trips it (no model needed).
    pytest.importorskip("PIL")
    from PIL import Image

    from lodedb.engine.embedding_backends import ClipEmbeddingBackend

    monkeypatch.setenv("LODEDB_MAX_IMAGE_PIXELS", "4")  # 2x2 ceiling
    with pytest.raises(ValueError, match="pixel"):
        ClipEmbeddingBackend._load_image(Image.new("RGB", (8, 8)))  # 64 px


def test_add_images_validates_all_items_before_embedding(tmp_path):
    db = _multimodal_db(tmp_path)
    # A bad metadata value on a later item must fail before any image is embedded.
    with pytest.raises(ValueError):
        db.add_images([{"image": "cat"}, {"image": "dog", "metadata": {"bad": ["x"]}}])
    assert db.count() == 0
    assert db.stats()["image_embedding"]["ingest"]["images_embedded"] == 0


def test_image_embedding_metrics_in_stats(tmp_path):
    db = _multimodal_db(tmp_path)
    db.add_image("cat", id="c")
    db.add_images([{"image": "dog"}, {"image": "bird"}])
    db.search_by_image("cat", k=1)
    metrics = db.stats()["image_embedding"]
    # ingest counts the three stored images; query counts the one search.
    assert metrics["ingest"]["images_embedded"] == 3
    assert metrics["ingest"]["encode_calls"] >= 2
    assert metrics["ingest"]["encode_failures"] == 0
    assert metrics["ingest"]["encode_seconds"] >= 0.0
    assert metrics["query"]["images_embedded"] == 1
    assert metrics["query"]["encode_calls"] == 1
