"""Concrete expectations ported from OpenKnowledge's chunking.test.ts."""

import json
import sys
from pathlib import Path

import pytest

_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from bench_core import _sanitize_lone_surrogates  # noqa: E402
from okchunk import (  # noqa: E402
    CHUNK_OVERLAP_CHARS,
    CHUNK_TARGET_CHARS,
    MAX_CHUNKS_PER_DOC,
    chunk_document,
)


def test_blank_input_yields_no_chunks() -> None:
    assert chunk_document("") == []
    assert chunk_document("   \n\t  ") == []


def test_short_document_is_one_trimmed_chunk() -> None:
    assert chunk_document("  hello world  ") == ["hello world"]


def test_document_at_target_boundary_stays_one_chunk() -> None:
    text = "a" * CHUNK_TARGET_CHARS
    assert chunk_document(text) == [text]


def test_long_document_splits_with_overlap_and_preserves_content() -> None:
    words = [f"word{index}" for index in range(1_200)]
    chunks = chunk_document(" ".join(words))

    assert len(chunks) > 1
    joined = " ".join(chunks)
    for word in (words[0], words[600], words[-1]):
        assert word in joined

    first_tail = chunks[0][-CHUNK_OVERLAP_CHARS // 2 :]
    assert first_tail.strip().split(" ")[0] in chunks[1]


def test_chunks_never_exceed_target_length() -> None:
    text = "lorem ipsum dolor sit amet " * 2_000
    assert all(len(chunk) <= CHUNK_TARGET_CHARS for chunk in chunk_document(text))


def test_max_chunk_cap_applies_to_pathologically_large_input() -> None:
    text = "x" * CHUNK_TARGET_CHARS * (MAX_CHUNKS_PER_DOC + 50)
    assert len(chunk_document(text)) <= MAX_CHUNKS_PER_DOC


def test_unbroken_run_makes_forward_progress() -> None:
    text = "a" * CHUNK_TARGET_CHARS * 3
    chunks = chunk_document(text)
    assert len(chunks) > 1
    assert len(chunks) <= MAX_CHUNKS_PER_DOC


def test_astral_hard_cut_preserves_parity_and_sanitizes_request_text() -> None:
    chunks = chunk_document("a" * 7_999 + "\U0001f600")

    assert 0xD800 <= ord(chunks[0][-1]) <= 0xDBFF
    with pytest.raises(UnicodeEncodeError):
        json.dumps({"input": [chunks[0]]}, ensure_ascii=False).encode("utf-8")

    sanitized = _sanitize_lone_surrogates(chunks[0])
    serialized = json.dumps({"input": [sanitized]}, ensure_ascii=False).encode("utf-8")
    assert b"\xef\xbf\xbd" in serialized

    well_formed = "Kubernetes \U0001f600"
    assert _sanitize_lone_surrogates(well_formed) == well_formed
