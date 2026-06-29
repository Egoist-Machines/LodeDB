from __future__ import annotations

import json
import random
from pathlib import Path

from lodedb.engine._lexical import (
    Bm25Index,
    build_chunk_token_lists,
    reciprocal_rank_fusion,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "tests" / "fixtures" / "native_core_lexical" / "lexical.json"


def _rounded_rank(rows: list[tuple[str, float]]) -> list[list[object]]:
    return [[unit_id, round(score, 12)] for unit_id, score in rows]


def _token_cases() -> list[dict[str, object]]:
    texts = [
        "The Quick Brown Fox",
        "fault code E1234 reported",
        "the failure was E1234.",
        "(E1234)",
        "serial ABC-123 shipped",
        "part ABC-123-X",
        "logged on 2024-01-15 today",
        "upgrade to v1.2.3",
        "ratio a/b here",
        "a--b",
        "-E1234-",
        "x... y",
        "/var/log/app.log moved",
        "",
        "!!! ??? ...",
    ]
    return [{"text": text, "tokens": tokenize(text)} for text in texts]


def _rank_cases() -> list[dict[str, object]]:
    units = [
        {
            "id": "u1",
            "text": "the turbine reported fault code e1234 overnight",
            "group": "doc-a",
        },
        {"id": "u2", "text": "quick brown foxes and lazy dogs at noon", "group": "doc-b"},
        {
            "id": "u3",
            "text": "quarterly revenue and operating costs with e1234",
            "group": "doc-c",
        },
        {
            "id": "u4",
            "text": "common token common token rare token",
            "group": "doc-d",
        },
        {
            "id": "u5",
            "text": "another e1234 mention for good measure here",
            "group": "doc-e",
        },
    ]
    index = Bm25Index(
        [unit["id"] for unit in units],
        [unit["text"] for unit in units],
        group_ids=[unit["group"] for unit in units],
    )
    cases: list[dict[str, object]] = []
    for query in ["e1234", "e1234 revenue", "common rare", "foxes", "turbine", "zzz"]:
        cases.append({"name": query, "query": query, "rank": _rounded_rank(index.rank(query))})
    allowed = {index.position_of("u3"), index.position_of("u5")}
    cases.append(
        {
            "name": "allowed_e1234",
            "query": "e1234",
            "allowed_unit_ids": ["u3", "u5"],
            "rank": _rounded_rank(index.rank("e1234", allowed_indices=allowed)),
        }
    )
    cases.append(
        {
            "name": "limit_2",
            "query": "e1234",
            "limit": 2,
            "rank": _rounded_rank(index.rank("e1234", limit=2)),
        }
    )

    rng = random.Random(27)
    vocab = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "e1234",
        "abc-123",
        "rare",
        "common",
        "token",
    ]
    random_units: list[dict[str, str]] = []
    for i in range(16):
        length = rng.randint(3, 10)
        words = [rng.choice(vocab) for _ in range(length)]
        random_units.append({"id": f"r{i:02d}", "text": " ".join(words), "group": f"g{i % 4}"})
    random_index = Bm25Index(
        [unit["id"] for unit in random_units],
        [unit["text"] for unit in random_units],
        group_ids=[unit["group"] for unit in random_units],
    )
    for query in ["alpha e1234", "rare token", "abc-123 gamma", "epsilon missing"]:
        cases.append(
            {
                "name": f"random_{query}",
                "query": query,
                "corpus": "random",
                "rank": _rounded_rank(random_index.rank(query)),
            }
        )
    return [{"name": "fixed", "units": units}, {"name": "random", "units": random_units}], cases


def _incremental_case() -> dict[str, object]:
    base = {
        "docA": {"a#0": "alpha beta gamma", "a#1": "alpha alpha delta"},
        "docB": {"b#0": "gamma e1234 note", "b#1": "delta delta beta"},
        "docC": {"c#0": "epsilon only here", "c#1": "abc-123 serial line"},
    }
    unit_ids: list[str] = []
    texts: list[str] = []
    group_ids: list[str] = []
    for group_id, chunks in base.items():
        for chunk_id, text in chunks.items():
            unit_ids.append(chunk_id)
            texts.append(text)
            group_ids.append(group_id)
    index = Bm25Index(unit_ids, texts, group_ids=group_ids)
    replacement = [
        ["a#7", tokenize("alpha rewritten")],
        ["a#8", tokenize("beta gamma e1234")],
        ["a#9", tokenize("delta epsilon")],
    ]
    index.replace_group("docA", replacement)
    index.remove_group("docB")
    queries = ["alpha", "beta", "gamma", "delta", "epsilon", "e1234", "abc-123"]
    return {
        "base": [
            {"group": group_id, "id": chunk_id, "text": text}
            for group_id, chunks in base.items()
            for chunk_id, text in chunks.items()
        ],
        "replacement_group": "docA",
        "replacement_units": replacement,
        "remove_group": "docB",
        "unit_ids": sorted(index.unit_ids),
        "ranks": {query: _rounded_rank(index.rank(query)) for query in queries},
    }


def _rrf_cases() -> list[dict[str, object]]:
    cases = [
        {"name": "worked", "rankings": [["A", "B", "C"], ["B", "C", "D"]], "c": 60},
        {"name": "dedupe", "rankings": [["A", "A", "B"]], "c": 60},
        {"name": "tie", "rankings": [["b"], ["a"]], "c": 60},
        {"name": "weighted", "rankings": [["A"], ["B"]], "c": 60, "weights": [1.0, 5.0]},
    ]
    for case in cases:
        case["fused"] = _rounded_rank(
            reciprocal_rank_fusion(
                case["rankings"],
                c=case["c"],
                weights=case.get("weights"),
            )
        )
    return cases


def _chunk_token_case() -> dict[str, object]:
    document_tokens = {
        "doc-a": [["alpha", "beta"], ["gamma"], ["extra"]],
        "doc-b": [["delta"]],
        "doc-empty": [],
    }
    document_chunk_ids = {
        "doc-a": ["a#0", "a#1"],
        "doc-b": ["b#0"],
        "doc-missing": ["m#0"],
    }
    chunk_ids, token_lists, group_ids = build_chunk_token_lists(document_tokens, document_chunk_ids)
    return {
        "documents": [
            {
                "document_id": document_id,
                "chunk_ids": document_chunk_ids.get(document_id, []),
                "token_lists": tokens,
            }
            for document_id, tokens in document_tokens.items()
        ],
        "expected": {
            "chunk_ids": chunk_ids,
            "token_lists": token_lists,
            "group_ids": group_ids,
        },
    }


def main() -> None:
    corpora, rank_cases = _rank_cases()
    payload = {
        "token_cases": _token_cases(),
        "corpora": corpora,
        "rank_cases": rank_cases,
        "incremental_case": _incremental_case(),
        "rrf_cases": _rrf_cases(),
        "chunk_token_case": _chunk_token_case(),
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
