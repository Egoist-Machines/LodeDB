from __future__ import annotations

import sys
import types

import numpy as np
import pytest

try:
    import lodedb._turbovec  # noqa: F401
except ImportError:
    import os

    if os.environ.get("LODEDB_ALLOW_MOCK_TURBOVEC") != "1":
        raise

    # Inject pure-Python high-fidelity mock of the Rust TurboVec extension
    mock_turbovec = types.ModuleType("lodedb._turbovec")
    sys.modules["lodedb._turbovec"] = mock_turbovec

    class MockIdMapIndex:
        def __init__(self, dim: int, bit_width: int):
            self.dim = dim
            self.bit_width = bit_width
            self.vectors = {}  # stable_id (int) -> vector (NDArray)
            self.stable_ids = []  # insertion order with swap-remove

        def __len__(self) -> int:
            return len(self.vectors)

        def __contains__(self, stable_id: int) -> bool:
            return int(stable_id) in self.vectors

        def contains(self, stable_id: int) -> bool:
            return int(stable_id) in self.vectors

        def add_with_ids(self, embeddings: np.ndarray, stable_ids: np.ndarray) -> None:
            for emb, sid in zip(embeddings, stable_ids, strict=True):
                sid_uint = int(sid)
                if sid_uint not in self.vectors:
                    self.stable_ids.append(sid_uint)
                self.vectors[sid_uint] = np.asarray(emb, dtype=np.float32)

        def remove(self, stable_id: int) -> bool:
            sid_uint = int(stable_id)
            if sid_uint in self.vectors:
                idx = self.stable_ids.index(sid_uint)
                last_idx = len(self.stable_ids) - 1
                if idx != last_idx:
                    self.stable_ids[idx] = self.stable_ids[last_idx]
                self.stable_ids.pop()
                del self.vectors[sid_uint]
                return True
            return False

        def remove_many(self, stable_ids: np.ndarray) -> int:
            removed = 0
            for sid in stable_ids:
                if self.remove(sid):
                    removed += 1
            return removed

        def search(
            self, queries: np.ndarray, k: int, allowlist: np.ndarray = None
        ) -> tuple[np.ndarray, np.ndarray]:
            queries = np.asarray(queries, dtype=np.float32)
            is_single = queries.ndim == 1
            if is_single:
                queries = queries.reshape(1, -1)

            results_scores = []
            results_ids = []

            candidate_ids = list(self.stable_ids)
            if allowlist is not None:
                allow_set = set(allowlist)
                candidate_ids = [cid for cid in candidate_ids if cid in allow_set]

            if not candidate_ids:
                empty_scores = np.empty((queries.shape[0], 0), dtype=np.float32)
                empty_ids = np.empty((queries.shape[0], 0), dtype=np.uint64)
                if is_single:
                    return empty_scores[0], empty_ids[0]
                return empty_scores, empty_ids

            for q in queries:
                scores = []
                for cid in candidate_ids:
                    vec = self.vectors[cid]
                    # Cosine similarity
                    norm_q = np.linalg.norm(q)
                    norm_v = np.linalg.norm(vec)
                    score = np.dot(q, vec)
                    if norm_q > 0 and norm_v > 0:
                        score = score / (norm_q * norm_v)
                    scores.append((score, cid))
                # Sort by score descending, then by stable_id ascending (deterministic tie-break)
                scores.sort(key=lambda x: (-x[0], x[1]))
                top_k = scores[:k]
                results_scores.append([s[0] for s in top_k])
                results_ids.append([s[1] for s in top_k])

            res_scores = np.asarray(results_scores, dtype=np.float32)
            res_ids = np.asarray(results_ids, dtype=np.uint64)
            if is_single:
                return res_scores[0], res_ids[0]
            return res_scores, res_ids

        def write(self, path: str) -> None:
            import pickle

            with open(path, "wb") as f:
                pickle.dump((self.dim, self.bit_width, self.vectors, self.stable_ids), f)

        @classmethod
        def load(cls, path: str) -> MockIdMapIndex:
            import pickle

            try:
                with open(path, "rb") as f:
                    dim, bit_width, vectors, stable_ids = pickle.load(f)
            except Exception as exc:
                raise RuntimeError(f"Failed to load mock index: {exc}") from exc
            idx = cls(dim, bit_width)
            idx.vectors = vectors
            idx.stable_ids = stable_ids
            return idx

        def bytes_per_vector(self) -> int:
            return self.dim * self.bit_width // 8

        def calibration_fingerprint(self) -> int:
            return 42

        def export_encoded(self, stable_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            bpv = self.bytes_per_vector()
            codes = np.zeros((len(stable_ids), bpv), dtype=np.uint8)
            scales = np.ones(len(stable_ids), dtype=np.float32)
            return codes, scales

        def add_encoded(
            self, stable_ids: np.ndarray, codes: np.ndarray, scales: np.ndarray
        ) -> None:
            for sid in stable_ids:
                sid_uint = int(sid)
                if sid_uint not in self.vectors:
                    self.stable_ids.append(sid_uint)
                self.vectors[sid_uint] = np.zeros(self.dim, dtype=np.float32)

        def rotation_matrix(self) -> np.ndarray:
            return np.eye(self.dim, dtype=np.float32)

        def reconstruct_all(self) -> tuple[np.ndarray, np.ndarray]:
            ids = np.array(self.stable_ids, dtype=np.uint64)
            if not self.stable_ids:
                rows = np.empty((0, self.dim), dtype=np.float32)
            else:
                rows = np.array([self.vectors[sid] for sid in self.stable_ids], dtype=np.float32)
            return ids, rows

        def reconstruct_rows(self, stable_ids: np.ndarray) -> np.ndarray:
            rows = []
            for sid in stable_ids:
                rows.append(self.vectors[int(sid)])
            return np.array(rows, dtype=np.float32)

    mock_turbovec.IdMapIndex = MockIdMapIndex


# --------------------------------------------------------------------------
# Cloud-transfer fixtures: real committed generations in throwaway
# directories, authored through the actual engine (the same commit path a
# user's database goes through) with the deterministic hash embedding
# backend, so the `lodedb.cloud` transfer suites stay fast and network-free.

# Small but non-trivial corpus: enough documents for real base + text artifacts.
DOCUMENTS = [{"text": f"the quick brown document number {i}", "id": f"doc-{i}"} for i in range(6)]


def write_committed_store(path) -> str:
    """Writes one committed generation (with raw text) into `path`; returns its index key."""
    from lodedb import cloud
    from lodedb.engine.embedding_backends import HashEmbeddingBackend
    from lodedb.local.db import LodeDB

    db = LodeDB(
        path=path,
        model="minilm",
        commit_mode="generation",
        _embedding_backend=HashEmbeddingBackend(native_dim=384),
    )
    try:
        db.add_many(DOCUMENTS)
    finally:
        db.close()
    (key,) = cloud.keys(str(path))
    return key


@pytest.fixture()
def committed_store(tmp_path):
    """A LodeDB directory holding one committed generation, as `(dir_path, index_key)`."""
    source = tmp_path / "source"
    source.mkdir()
    return source, write_committed_store(source)


def read_pointer_body(path, key: str) -> dict:
    """Reads the committed root body from `<path>/<key>.commit.json` (raw JSON, no engine)."""
    import json

    with open(path / f"{key}.commit.json", encoding="utf-8") as pointer:
        return json.load(pointer)["body"]
