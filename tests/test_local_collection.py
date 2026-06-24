"""Tests for LodeCollection: named vector spaces under one root.

Spaces are independent LodeDB indexes that can use different models/dimensions.
These tests use vector-only spaces (and an injected hash backend for the one
preset case) so they run offline with no model download.
"""

from __future__ import annotations

import pytest

from lodedb import LodeCollection
from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local.db import ReadOnlyError


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
        "model": None,
        "vector_dim": 8,
        "bit_width": 4,
    }
    assert reopened.space_config("notes")["model"] == "minilm"
    reopened.close()


def test_reopen_enforces_space_config(tmp_path):
    col = LodeCollection(tmp_path)
    col.space("emb", vector_dim=8)
    col.close()

    reopened = LodeCollection(tmp_path)
    with pytest.raises(ValueError, match="created with"):
        reopened.space("emb", vector_dim=16)
    reopened.close()


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
