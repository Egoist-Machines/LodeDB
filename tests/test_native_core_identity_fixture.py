from __future__ import annotations

import json
from pathlib import Path

from lodedb.engine.core import _chunk_id_for_hash, chunk_text, normalized_chunk_hash, sha256_text
from lodedb.engine.turbovec_index import stable_uint64_ids_for_chunk_ids


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "native_core_identity" / "identity.json"


def test_native_core_identity_fixture_matches_python_oracle() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in fixture["hash_cases"]:
        assert sha256_text(case["text"]) == case["sha256"]
        assert normalized_chunk_hash(case["text"]) == case["normalized_chunk_hash"]

    for case in fixture["chunk_cases"]:
        assert list(chunk_text(case["text"], case["limit"])) == case["chunks"]

    for case in fixture["chunk_id_cases"]:
        assert (
            _chunk_id_for_hash(
                case["document_id"],
                chunk_hash=case["chunk_hash"],
                occurrence=case["occurrence"],
            )
            == case["chunk_id"]
        )

    for case in fixture["document_cases"]:
        chunks = list(chunk_text(case["text"], case["limit"]))
        assert chunks == case["chunks"]
        chunk_hashes = [normalized_chunk_hash(chunk) for chunk in chunks]
        assert chunk_hashes == case["chunk_hashes"]
        seen: dict[str, int] = {}
        chunk_ids: list[str] = []
        for chunk_hash in chunk_hashes:
            occurrence = seen.get(chunk_hash, 0)
            seen[chunk_hash] = occurrence + 1
            chunk_ids.append(
                _chunk_id_for_hash(
                    case["document_id"],
                    chunk_hash=chunk_hash,
                    occurrence=occurrence,
                )
            )
        assert chunk_ids == case["chunk_ids"]

    for case in fixture["stable_id_cases"]:
        stable_ids = stable_uint64_ids_for_chunk_ids(tuple(case["chunk_ids"]))
        assert [int(value) for value in stable_ids] == case["stable_ids"]
