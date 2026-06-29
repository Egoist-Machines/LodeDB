from __future__ import annotations

import json
from pathlib import Path

from lodedb.engine._lexical import (
    Bm25Index,
    build_chunk_token_lists,
    reciprocal_rank_fusion,
    tokenize,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "native_core_lexical" / "lexical.json"


def _rounded_rank(rows: list[tuple[str, float]]) -> list[list[object]]:
    return [[unit_id, round(score, 12)] for unit_id, score in rows]


def test_native_core_lexical_fixture_matches_python_oracle() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    for case in fixture["token_cases"]:
        assert tokenize(case["text"]) == case["tokens"]

    corpora = {}
    for corpus in fixture["corpora"]:
        units = corpus["units"]
        corpora[corpus["name"]] = Bm25Index(
            [unit["id"] for unit in units],
            [unit["text"] for unit in units],
            group_ids=[unit["group"] for unit in units],
        )

    for case in fixture["rank_cases"]:
        index = corpora[case.get("corpus", "fixed")]
        allowed = None
        if "allowed_unit_ids" in case:
            allowed = {index.position_of(unit_id) for unit_id in case["allowed_unit_ids"]}
        assert _rounded_rank(
            index.rank(case["query"], limit=case.get("limit"), allowed_indices=allowed)
        ) == case["rank"]

    incremental = fixture["incremental_case"]
    base = incremental["base"]
    index = Bm25Index(
        [unit["id"] for unit in base],
        [unit["text"] for unit in base],
        group_ids=[unit["group"] for unit in base],
    )
    index.replace_group(incremental["replacement_group"], incremental["replacement_units"])
    index.remove_group(incremental["remove_group"])
    assert sorted(index.unit_ids) == incremental["unit_ids"]
    for query, expected in incremental["ranks"].items():
        assert _rounded_rank(index.rank(query)) == expected

    for case in fixture["rrf_cases"]:
        assert _rounded_rank(
            reciprocal_rank_fusion(
                case["rankings"],
                c=case["c"],
                weights=case.get("weights"),
            )
        ) == case["fused"]

    chunk_case = fixture["chunk_token_case"]
    document_tokens = {
        row["document_id"]: row["token_lists"] for row in chunk_case["documents"]
    }
    document_chunk_ids = {
        row["document_id"]: row["chunk_ids"] for row in chunk_case["documents"]
    }
    chunk_ids, token_lists, group_ids = build_chunk_token_lists(
        document_tokens, document_chunk_ids
    )
    assert {
        "chunk_ids": chunk_ids,
        "token_lists": token_lists,
        "group_ids": group_ids,
    } == chunk_case["expected"]
