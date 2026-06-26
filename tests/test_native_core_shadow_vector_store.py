from __future__ import annotations

from lodedb import LodeDB


def _onehot(i: int, dim: int = 8) -> list[float]:
    vector = [0.0] * dim
    vector[i] = 1.0
    return vector


def test_open_vector_store_shadow_mode_matches_python_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LODEDB_NATIVE_CORE", "shadow")
    db = LodeDB.open_vector_store(tmp_path, vector_dim=8)
    db.add_vectors(_onehot(0), id="a", metadata={"topic": "ops"})
    db.add_vectors(_onehot(1), id="b", metadata={"topic": "ml"})

    assert [hit.id for hit in db.search_by_vector(_onehot(1), k=2)] == ["b", "a"]
    assert [hit.id for hit in db.search_by_vector(_onehot(0), k=2, filter={"topic": "ops"})] == [
        "a"
    ]
    assert db.remove("a") is True
    assert db.stats()["document_count"] == 1
