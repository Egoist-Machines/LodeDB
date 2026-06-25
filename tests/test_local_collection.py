"""Tests for LodeCollection: named vector spaces under one root.

Spaces are independent LodeDB indexes that can use different models/dimensions.
These tests use vector-only spaces (and an injected hash backend for the one
preset case) so they run offline with no model download.
"""

from __future__ import annotations

import pytest

from lodedb import LodeCollection, LodeDB
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import ReadOnlyError


def test_plain_lodedb_open_on_collection_root_ignores_manifest(tmp_path):
    # A collection writes collection.json at its root and puts each space in a
    # subdirectory. Opening the root as a plain index must skip the manifest rather
    # than try to parse it as a legacy snapshot (which raised a schema-version error).
    col = LodeCollection(tmp_path)
    col.space("vectors", vector_dim=8).add_vectors([1, 0, 0, 0, 0, 0, 0, 0], id="v")
    col.close()

    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)  # the root, not a space subdir
    assert db.count() == 0  # spaces live in subdirs; the manifest is not an index
    db.close()


def test_spaces_are_independent(tmp_path):
    col = LodeCollection(tmp_path)
    a = col.space("text", vector_dim=8)
    b = col.space("vectors", vector_dim=16)
    a.add_vectors([1, 0, 0, 0, 0, 0, 0, 0], id="t1")
    b.add_vectors([1] + [0] * 15, id="v1")
    b.add_vectors([0, 1] + [0] * 14, id="v2")
    assert a.count() == 1
    assert b.count() == 2
    assert col.spaces() == ["text", "vectors"]
    assert (tmp_path / "text").is_dir()
    assert (tmp_path / "vectors").is_dir()
    col.close()


def test_same_space_returns_same_handle(tmp_path):
    col = LodeCollection(tmp_path)
    first = col.space("x", vector_dim=8)
    second = col.space("x", vector_dim=8)
    assert first is second
    col.close()


def test_manifest_persists_across_reopen(tmp_path):
    col = LodeCollection(tmp_path)
    col.space("vectors", vector_dim=8)
    col.space("notes", model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384))
    col.close()

    reopened = LodeCollection(tmp_path)
    assert reopened.spaces() == ["notes", "vectors"]
    # A vector-only space records model=None; a preset space records its model.
    assert reopened.space_config("vectors") == {
        "kind": "vector",
        "vector_dim": 8,
        "bit_width": 4,
        "store_text": True,
        "index_text": False,
    }
    assert reopened.space_config("notes") == {
        "kind": "preset",
        "model": "minilm",
        "bit_width": 4,
        "store_text": True,
        "index_text": False,
    }
    reopened.close()


def test_custom_embedder_space_records_identity_and_reopens(tmp_path):
    col = LodeCollection(tmp_path)
    space = col.space("notes", embedder=_IdentBackend("my-model"))
    space.add("hello world", id="a")
    space.persist()
    # Recorded as a custom space with its identity, not a false preset.
    assert col.space_config("notes") == {
        "kind": "custom",
        "model_identity": "my-model",
        "bit_width": 4,
        "store_text": True,
        "index_text": False,
    }
    col.close()

    reopened = LodeCollection(tmp_path)
    # The backend can't be persisted, so reopening without one is a clear error.
    with pytest.raises(ValueError, match="custom-embedder space"):
        reopened.space("notes")
    # Reopening with a matching embedder works.
    space2 = reopened.space("notes", embedder=_IdentBackend("my-model"))
    assert space2.count() == 1
    reopened.close()


def test_preset_space_records_effective_bit_width(tmp_path):
    col = LodeCollection(tmp_path)
    # An explicit, conflicting bit_width for a preset space is rejected up front.
    with pytest.raises(ValueError, match="preset"):
        col.space("bad", model="minilm", bit_width=2)
    # A preset space records the preset's real width (4), never a caller value that
    # would not take effect.
    col.space("notes", model="minilm", _embedding_backend=HashEmbeddingBackend(native_dim=384))
    assert col.space_config("notes") == {
        "kind": "preset",
        "model": "minilm",
        "bit_width": 4,
        "store_text": True,
        "index_text": False,
    }
    col.close()


def test_collection_owns_privacy_flags_across_reopen(tmp_path):
    col = LodeCollection(tmp_path)
    col.space("vec", vector_dim=8, store_text=False, index_text=True)
    col.close()

    # Reopening with no flags restores the recorded store_text/index_text, so a
    # privacy-off space never silently flips back to retaining raw text.
    reopened = LodeCollection(tmp_path)
    space = reopened.space("vec")
    assert space.store_text is False
    assert space.index_text is True
    cfg = reopened.space_config("vec")
    assert cfg["store_text"] is False and cfg["index_text"] is True
    # store_text=False refuses raw-text retrieval on the reopened handle too.
    with pytest.raises(ValueError, match="store_text"):
        space.get_text("anything")
    reopened.close()


def test_collection_rejects_conflicting_store_text(tmp_path):
    col = LodeCollection(tmp_path)
    col.space("vec", vector_dim=8, store_text=False)
    col.close()
    reopened = LodeCollection(tmp_path)
    with pytest.raises(ValueError, match="store_text"):
        reopened.space("vec", store_text=True)
    reopened.close()


def test_same_handle_mismatched_config_is_rejected(tmp_path):
    # An already-open space must still validate the requested config, so a
    # mismatched second call fails here rather than returning the cached handle.
    col = LodeCollection(tmp_path)
    col.space("x", vector_dim=8)
    with pytest.raises(ValueError, match="created with"):
        col.space("x", vector_dim=16)
    col.close()


def test_reopen_enforces_space_config(tmp_path):
    col = LodeCollection(tmp_path)
    col.space("emb", vector_dim=8)
    col.close()

    reopened = LodeCollection(tmp_path)
    with pytest.raises(ValueError, match="created with"):
        reopened.space("emb", vector_dim=16)
    reopened.close()


def test_reopen_recorded_space_without_restating_config(tmp_path):
    # A recorded space reopens from its manifest with no config args, including
    # vector-only and clip spaces whose config differs from the plain defaults.
    col = LodeCollection(tmp_path)
    col.space("vec", vector_dim=16)
    col.space("img", model="clip")  # ClipEmbeddingBackend is lazy: no download on open
    col.close()

    reopened = LodeCollection(tmp_path)
    vec = reopened.space("vec")  # no vector_dim -> taken from the manifest
    assert vec.vector_only and vec._vector_dim == 16
    img = reopened.space("img")  # no model -> taken from the manifest
    assert img.preset is not None and img.preset.multimodal
    reopened.close()

    # Read-only reopen of a recorded vector-only space, no config args.
    reader = LodeCollection(tmp_path, read_only=True)
    ro = reader.space("vec")
    assert ro.read_only and ro._vector_dim == 16
    reader.close()


@pytest.mark.parametrize("bad", ["../evil", "a/b", "", ".", "..", "x y"])
def test_invalid_space_name_rejected(tmp_path, bad):
    col = LodeCollection(tmp_path)
    with pytest.raises(ValueError, match="space name"):
        col.space(bad, vector_dim=8)
    col.close()


def test_read_only_collection(tmp_path):
    writer = LodeCollection(tmp_path)
    space = writer.space("vectors", vector_dim=8)
    space.add_vectors([1, 0, 0, 0, 0, 0, 0, 0], id="v1")
    space.persist()
    writer.close()

    reader = LodeCollection(tmp_path, read_only=True)
    ro_space = reader.space("vectors", vector_dim=8)
    assert ro_space.search_by_vector([1, 0, 0, 0, 0, 0, 0, 0], k=1)[0].id == "v1"
    with pytest.raises(ReadOnlyError):
        ro_space.add_vectors([0, 1, 0, 0, 0, 0, 0, 0], id="v2")
    # A space absent from the manifest is not created in read-only mode.
    with pytest.raises(FileNotFoundError):
        reader.space("missing", vector_dim=8)
    reader.close()


def test_read_only_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        LodeCollection(tmp_path / "does-not-exist", read_only=True)


def test_concurrent_space_creation_does_not_lose_spaces(tmp_path):
    # Two handles that both loaded the (empty) manifest then each create a
    # different space: the read-merge-write under the manifest lock keeps both,
    # rather than the second write clobbering the first (last-writer-wins).
    col_a = LodeCollection(tmp_path)
    col_b = LodeCollection(tmp_path)
    col_a.space("alpha", vector_dim=8)
    col_b.space("beta", vector_dim=8)
    col_a.close()
    col_b.close()

    reader = LodeCollection(tmp_path, read_only=True)
    assert reader.spaces() == ["alpha", "beta"]
    reader.close()


class _IdentBackend(HashEmbeddingBackend):
    """A hash backend that declares a model identity."""

    def __init__(self, model_name: str, dim: int = 16) -> None:
        super().__init__(native_dim=dim)
        self.required_model_name = model_name
        self.name = "ident"


def test_collection_space_rejects_wrong_embedder_identity(tmp_path):
    col = LodeCollection(tmp_path)
    space = col.space("notes", embedder=_IdentBackend("model-A"))
    space.add("hello world", id="a")
    space.persist()
    col.close()

    reopened = LodeCollection(tmp_path)
    # Same dimension, different embedder identity for the space: rejected at open.
    with pytest.raises(RuntimeError, match="does not match"):
        reopened.space("notes", embedder=_IdentBackend("model-B"))
    reopened.close()


def test_manifest_publish_failure_rolls_back_space(tmp_path, monkeypatch):
    col = LodeCollection(tmp_path)

    def _boom():
        raise OSError("simulated manifest write failure")

    monkeypatch.setattr(col, "_write_manifest", _boom)
    with pytest.raises(RuntimeError, match="rolled back"):
        col.space("alpha", vector_dim=8)

    # The space was never registered or cached, and its writer lock was released.
    assert col.spaces() == []
    assert "alpha" not in col._open

    # A fresh collection can open the same space (no leaked lock -> no
    # ConcurrentWriterError) and the manifest now publishes cleanly.
    reopened = LodeCollection(tmp_path)
    space = reopened.space("alpha", vector_dim=8)
    space.add_vectors([1, 0, 0, 0, 0, 0, 0, 0], id="v")
    assert reopened.spaces() == ["alpha"]
    reopened.close()
    col.close()


def test_manifest_write_honors_fsync_durability(tmp_path, monkeypatch):
    import lodedb.local.collection as collection_mod

    captured: dict[str, object] = {}
    real_durable_replace = collection_mod.durable_replace

    def _capture(tmp, dst, *, fsync):
        captured["fsync"] = fsync
        return real_durable_replace(tmp, dst, fsync=fsync)

    monkeypatch.setattr(collection_mod, "durable_replace", _capture)
    col = LodeCollection(tmp_path, durability="fsync")
    col.space("vec", vector_dim=8)  # publishes the manifest
    assert captured.get("fsync") is True
    col.close()
