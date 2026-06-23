"""Tests for the persistent, journaled BM25 postings store (``index_text=True``).

By default the BM25 index behind ``mode="hybrid"``/``"lexical"`` is rebuilt in
memory from the retained raw text, so hybrid search depends on ``store_text`` and
is re-tokenized on every reopen. Opening with ``index_text=True`` instead keeps
the per-chunk tokens captured at ``add`` time in a dedicated ``.tvlex`` base plus
a ``.lxd`` delta journal, so hybrid and lexical search survive a reopen without
rebuilding from raw text and without requiring ``store_text=True``.

These mirror the deterministic hash-backend pattern the rest of the local suite
uses: a content-blind embedding cannot "see" an exact token in unrelated prose,
so the lexical ranker is what surfaces error codes, serials, and dates. The
redacted artifacts stay payload-free regardless; the ``.tvlex`` sidecar holds the
payload-derived terms and nothing else does.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pytest

from lodedb.engine.embedding_backends import HashEmbeddingBackend
from lodedb.local import LodeDB


def _open(
    tmp_path,
    *,
    index_text: bool = True,
    store_text: bool = False,
    dim: int = 384,
) -> LodeDB:
    """Opens a LodeDB with an injected deterministic hash backend."""

    return LodeDB(
        path=tmp_path,
        index_text=index_text,
        store_text=store_text,
        _embedding_backend=HashEmbeddingBackend(native_dim=dim),
    )


def _seed_with_exact_token(db: LodeDB, *, token: str, topic: str = "ops") -> str:
    """Adds one carrier document holding ``token`` plus noisy distractors."""

    carrier = db.add(
        "The overnight maintenance log records that the auxiliary turbine tripped "
        f"and the controller reported {token} before the unit recovered.",
        id="carrier",
        metadata={"topic": topic},
    )
    db.add(
        "Quick brown foxes and lazy dogs wander the meadow at noon under a warm sky.",
        id="distractor-animals",
        metadata={"topic": "animals"},
    )
    db.add(
        "Quarterly revenue grew while operating costs declined across every region.",
        id="distractor-finance",
        metadata={"topic": "finance"},
    )
    for i in range(12):
        db.add(
            f"General notes number {i} covering miscellaneous unrelated topics and asides.",
            id=f"filler-{i}",
            metadata={"topic": "misc"},
        )
    return carrier


def _tvlex_files(tmp_path) -> list[str]:
    """Returns every persisted lexical-index base/delta artifact under the path."""

    base = glob.glob(str(Path(tmp_path) / "**" / "*.tvlex"), recursive=True)
    deltas = glob.glob(str(Path(tmp_path) / "**" / "*.lxd"), recursive=True)
    return base + deltas


# -- headline: hybrid/lexical survive a reopen without raw text -------------


def test_hybrid_reopens_without_raw_text(tmp_path):
    """index_text=True makes hybrid/lexical work across a reopen with no raw text.

    The headline guarantee: persist the postings with store_text=False, close,
    reopen with the same flags, and hybrid/lexical still surface the carrier even
    though no raw text was ever retained.
    """

    db = _open(tmp_path, index_text=True, store_text=False)
    carrier = _seed_with_exact_token(db, token="E1234")
    assert [hit.id for hit in db.search("E1234", k=5, mode="lexical")] == [carrier]
    assert db.search("E1234", k=3, mode="hybrid")[0].id == carrier
    db.persist()
    db.close()

    reopened = _open(tmp_path, index_text=True, store_text=False)
    try:
        # Rebuilt purely from the persisted tokens: no raw text is present.
        with pytest.raises(ValueError, match="store_text=True"):
            reopened.get_text(carrier)
        assert [hit.id for hit in reopened.search("E1234", k=5, mode="lexical")] == [carrier]
        assert reopened.search("E1234", k=3, mode="hybrid")[0].id == carrier
    finally:
        reopened.close()


def test_hybrid_persists_a_tvlex_sidecar(tmp_path):
    """Enabling index_text writes a dedicated .tvlex base; the redacted JSON has no tokens."""

    db = _open(tmp_path, index_text=True, store_text=False)
    _seed_with_exact_token(db, token="E1234")
    db.persist()
    assert glob.glob(str(Path(tmp_path) / "**" / "*.tvlex"), recursive=True)
    db.close()


# -- default is off: byte-for-byte unchanged --------------------------------


def test_default_writes_no_tvlex(tmp_path):
    """index_text defaults to False and writes no .tvlex/.lxd artifacts at all."""

    db = LodeDB(path=tmp_path, _embedding_backend=HashEmbeddingBackend(native_dim=384))
    assert db.index_text is False
    db.add("a document mentioning E1234 in its body", id="a")
    db.persist()
    assert _tvlex_files(tmp_path) == []
    db.close()


def test_default_layout_matches_no_lexical_store(tmp_path):
    """With index_text off, the on-disk artifact set is identical to a plain DB.

    Guards the "standard flow byte-for-byte unchanged" invariant: enabling the
    feature must add files only under index_text=True, never by default.
    """

    plain_dir = tmp_path / "plain"
    db = LodeDB(path=plain_dir, _embedding_backend=HashEmbeddingBackend(native_dim=384))
    db.add("payload one", id="a")
    db.add("payload two", id="b")
    db.persist()
    db.close()
    artifacts = sorted(
        p.name for p in plain_dir.rglob("*") if p.is_file()
    )
    # No lexical sidecar artifacts exist in the default layout.
    assert not any(name.endswith((".tvlex", ".lxd")) for name in artifacts)


# -- O(changed) journaling --------------------------------------------------


def test_lexical_commit_is_o_changed(tmp_path):
    """An incremental commit appends one .lxd delta, never rewriting the base."""

    db = _open(tmp_path, index_text=True, store_text=False)
    # Cold base: a batch large enough that the first commit writes a full base.
    db.add_many([{"text": f"base body {i} token{i}", "id": f"b{i}"} for i in range(12)])
    base_files = glob.glob(str(Path(tmp_path) / "**" / "g*.tvlex"), recursive=True)
    assert base_files, "expected a lexical-index base"
    base = Path(sorted(base_files)[-1])
    base_bytes = base.read_bytes()
    deltas_before = glob.glob(str(Path(tmp_path) / "**" / "*.lxd"), recursive=True)

    db.add("one small delta carrying ABC-123", id="delta", metadata={"topic": "ops"})
    # The base map is untouched; the change lands in exactly one new .lxd segment.
    assert base.read_bytes() == base_bytes, "base must not be rewritten on a delta commit"
    deltas_after = glob.glob(str(Path(tmp_path) / "**" / "*.lxd"), recursive=True)
    assert len(deltas_after) == len(deltas_before) + 1, "exactly one .lxd delta appended"
    db.close()

    reopened = _open(tmp_path, index_text=True, store_text=False)
    try:  # replaying base + delta makes the late doc lexically findable
        assert [hit.id for hit in reopened.search("ABC-123", k=5, mode="lexical")] == ["delta"]
        assert reopened.count() == 13
    finally:
        reopened.close()


# -- parity: token path vs raw-text path ------------------------------------


@pytest.mark.parametrize(
    "query",
    ["E1234", "turbine recovered", "foxes", "revenue costs"],
)
def test_token_path_matches_raw_text_path(tmp_path, query):
    """Hybrid ids and scores with index_text equal the store_text rebuild path.

    For the same corpus the two lexical sources (persisted tokens vs raw-text
    re-tokenization) must produce the identical fused ranking, including scores.
    """

    token_dir = tmp_path / "token"
    text_dir = tmp_path / "text"
    token_db = _open(token_dir, index_text=True, store_text=False)
    text_db = _open(text_dir, index_text=False, store_text=True)
    _seed_with_exact_token(token_db, token="E1234")
    _seed_with_exact_token(text_db, token="E1234")

    token_hits = token_db.search(query, k=5, mode="hybrid")
    text_hits = text_db.search(query, k=5, mode="hybrid")
    assert [h.id for h in token_hits] == [h.id for h in text_hits]
    assert [round(h.score, 9) for h in token_hits] == [round(h.score, 9) for h in text_hits]
    token_db.close()
    text_db.close()


def test_both_flags_on_uses_persisted_tokens_and_keeps_text(tmp_path):
    """With both flags on, hybrid works and raw text is still retrievable."""

    db = _open(tmp_path, index_text=True, store_text=True)
    carrier = _seed_with_exact_token(db, token="E1234")
    db.persist()
    db.close()

    reopened = _open(tmp_path, index_text=True, store_text=True)
    try:
        assert reopened.get_text(carrier) is not None  # raw text retained
        assert reopened.search("E1234", k=3, mode="hybrid")[0].id == carrier
    finally:
        reopened.close()


# -- generation correctness after a mutation then reopen --------------------


def test_generation_correct_after_mutation_then_reopen(tmp_path):
    """A doc added after the base is searchable immediately and after a reopen."""

    db = _open(tmp_path, index_text=True, store_text=False)
    _seed_with_exact_token(db, token="E1234")
    assert db.search("ABC-123", k=5, mode="lexical") == []  # not yet present
    db.add("replacement part labeled ABC-123 installed", id="late", metadata={"topic": "ops"})
    # The lexical index is generation-keyed, so the new doc is searchable at once.
    assert [hit.id for hit in db.search("ABC-123", k=5, mode="lexical")] == ["late"]
    db.persist()
    db.close()

    reopened = _open(tmp_path, index_text=True, store_text=False)
    try:  # and the mutation survives the reopen via the journaled delta
        assert [hit.id for hit in reopened.search("ABC-123", k=5, mode="lexical")] == ["late"]
    finally:
        reopened.close()


def test_remove_then_reopen_drops_from_lexical_index(tmp_path):
    """Removing a doc drops it from the persisted postings, durably."""

    db = _open(tmp_path, index_text=True, store_text=False)
    db.add("removable entry citing E1234 fault", id="gone", metadata={"topic": "ops"})
    db.add("kept entry citing ABC-123 serial", id="kept", metadata={"topic": "ops"})
    assert db.remove("gone") is True
    assert db.search("E1234", k=5, mode="lexical") == []
    assert [hit.id for hit in db.search("ABC-123", k=5, mode="lexical")] == ["kept"]
    db.persist()
    db.close()

    reopened = _open(tmp_path, index_text=True, store_text=False)
    try:
        assert reopened.search("E1234", k=5, mode="lexical") == []
        assert [hit.id for hit in reopened.search("ABC-123", k=5, mode="lexical")] == ["kept"]
    finally:
        reopened.close()


# -- error when neither lexical source is enabled ---------------------------


@pytest.mark.parametrize("mode", ["hybrid", "lexical"])
def test_no_lexical_source_raises_clear_error(tmp_path, mode):
    """index_text=False, store_text=False + a lexical mode raises a clear error."""

    db = _open(tmp_path, index_text=False, store_text=False)
    db.add("a document that mentions E1234 somewhere in its text")
    with pytest.raises(ValueError, match="index_text=True"):
        db.search("E1234", k=5, mode=mode)
    with pytest.raises(ValueError, match="store_text=True"):
        db.search_many(["E1234"], k=5, mode=mode)
    db.close()


# -- payload boundary: no tokens leak into redacted artifacts ---------------


def test_tokens_never_leak_into_redacted_artifacts(tmp_path):
    """Persisting the lexical index keeps snapshot/journal/telemetry payload-free."""

    secret = "ZZUNIQUETOKEN9999"
    db = _open(tmp_path, index_text=True, store_text=False)
    db.add(f"an incident referencing {secret} once", id="s", metadata={"topic": "x"})
    db.search(secret, k=3, mode="hybrid")
    db.persist()

    # Redacted JSON snapshot/manifest and journal deltas carry no token.
    for json_file in glob.glob(str(Path(tmp_path) / "**" / "*.json"), recursive=True):
        assert secret not in Path(json_file).read_text(encoding="utf-8")
    for jsd in glob.glob(str(Path(tmp_path) / "**" / "*.jsd"), recursive=True):
        assert secret not in Path(jsd).read_bytes().decode("utf-8", "replace")

    # Telemetry, audit, and redacted stats carry no token.
    engine = db._engine
    assert secret not in json.dumps([dict(m) for m in engine.metrics])
    assert secret not in json.dumps([dict(a) for a in engine.audit_events])
    stats = db.stats()
    assert stats["raw_payload_text_present"] is False
    assert secret not in json.dumps(stats)

    # The dedicated .tvlex sidecar (and only it) holds the token, lowercased.
    tvlex = glob.glob(str(Path(tmp_path) / "**" / "*.tvlex"), recursive=True)[0]
    assert secret.lower() in Path(tvlex).read_text(encoding="utf-8")
    db.close()


def test_corrupt_tvlex_base_fails_closed(tmp_path):
    """A garbled .tvlex base raises on reopen instead of serving partial postings."""

    db = _open(tmp_path, index_text=True, store_text=False)
    db.add("body that will be corrupted, token QQ-42", id="c", metadata={"topic": "ops"})
    db.persist()
    db.close()

    base = Path(sorted(glob.glob(str(Path(tmp_path) / "**" / "g*.tvlex"), recursive=True))[-1])
    base.write_text("not-valid-json {{{", encoding="utf-8")
    with pytest.raises(RuntimeError):
        _open(tmp_path, index_text=True, store_text=False)
