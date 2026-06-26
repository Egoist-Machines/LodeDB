"""Deterministic corpora shared by native-core migration fixtures and benchmarks."""

from __future__ import annotations

from typing import Any

TEXT_DOCUMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "doc-alpha",
        "text": "Alpha launch notes mention error code E-1001 and a blue widget.",
        "metadata": {"tenant": "acme", "kind": "note", "year": "2024", "rank": "1"},
    },
    {
        "id": "doc-beta",
        "text": "Beta incident report for serial AX-42 on 2024-06-13.",
        "metadata": {"tenant": "acme", "kind": "incident", "year": "2024", "rank": "2"},
    },
    {
        "id": "doc-gamma",
        "text": "Gamma handbook explains offline vector search and local recovery.",
        "metadata": {"tenant": "zen", "kind": "manual", "year": "2023", "rank": "3"},
    },
)

VECTOR_DOCUMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "vec-alpha",
        "vector": (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "metadata": {"tenant": "acme", "kind": "vector", "year": "2024"},
        "text": "Vector alpha retained payload.",
    },
    {
        "id": "vec-beta",
        "vector": (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "metadata": {"tenant": "zen", "kind": "vector", "year": "2023"},
        "text": "Vector beta retained payload.",
    },
    {
        "id": "vec-gamma",
        "vector": (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        "metadata": {"tenant": "acme", "kind": "vector", "year": "2022"},
        "text": "Vector gamma retained payload.",
    },
)

FILTERS: tuple[dict[str, Any] | None, ...] = (
    None,
    {"tenant": "acme"},
    {"year": {"$gte": 2024}},
    {"$and": [{"tenant": "acme"}, {"kind": {"$ne": "manual"}}]},
)

TEXT_QUERIES: tuple[str, ...] = (
    "error E-1001",
    "offline vector recovery",
    "serial AX-42",
)

VECTOR_QUERIES: tuple[tuple[float, ...], ...] = (
    (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
