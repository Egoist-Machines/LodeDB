"""Lexical BM25 ranker and Reciprocal Rank Fusion for hybrid retrieval.

This module is intentionally dependency-free (stdlib only) and must not import
``core``. It is the lexical counterpart to :mod:`lodedb.engine._predicate`: a
pure-Python CPU post-step that runs alongside the vector scan and never touches
the TurboVec kernel, the GPU/MPS paths, or the on-disk format.

A BM25 inverted index is payload-derived (it is built from raw chunk text), so
it is held in memory only, rebuilt lazily on the first lexical query after a
mutation, and never written into the redacted ``.json``/``.jsd``/``.tvim``/
``.tvd`` artifacts or into telemetry. See the payload-boundary section of
``docs/architecture.md``.

Three pieces:

- :func:`tokenize`: lowercases and splits on whitespace/punctuation while
  keeping alphanumeric runs intact and preserving internal ``-``/``.``/``/``
  inside code-like tokens, so ``E1234``, ``ABC-123``, and ``2024-01-15`` survive
  as single findable tokens. This is the whole point of the feature: the
  embedding cannot see an exact code, but the lexical index can.
- :class:`Bm25Index`: a classic Okapi BM25 inverted index over a unit space
  (LodeDB indexes chunks), scoring a tokenized query into a ranked list; it
  supports both a one-shot bulk build and incremental add/remove of single
  units so a small mutation does not force a full rebuild.
- :func:`reciprocal_rank_fusion`: fuses ranked id lists with RRF, which needs
  no score normalization and therefore composes cleanly with the vector scores.

BM25 reference (Okapi): ``score(D, Q) = Σ idf(q) · tf·(k1+1) / (tf + k1·(1 - b +
b·|D|/avgdl))``. We use the non-negative Lucene IDF ``ln(1 + (N - df + 0.5) /
(df + 0.5))`` so a term occurring in more than half the corpus never contributes
a negative score. Defaults: ``k1 = 1.2``, ``b = 0.75``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

# Classic Okapi BM25 defaults. k1 governs term-frequency saturation; b governs
# document-length normalization.
BM25_K1 = 1.2
BM25_B = 0.75

# Reciprocal Rank Fusion smoothing constant. 60 is the value Cormack et al.
# found robust on TREC data and the de-facto standard across hybrid search.
RRF_C = 60

# A token is a maximal run of alphanumerics that may contain interior single
# ``-``, ``.``, or ``/`` separators between alphanumerics. The interior-separator
# clause is what keeps code-like tokens whole:
#   ``ABC-123`` -> ["abc-123"], ``2024-01-15`` -> ["2024-01-15"],
#   ``v1.2.3`` -> ["v1.2.3"], ``a/b`` -> ["a/b"].
# Punctuation that is leading, trailing, or doubled is a boundary, so trailing
# sentence punctuation (``E1234.``) and em-dash-style runs split cleanly and do
# not glue onto the token. Matching is applied to the lowercased text.
_TOKEN_RE = re.compile(r"[0-9a-z]+(?:[-./][0-9a-z]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenizes text for lexical matching, preserving codes, serials, and dates.

    Lowercases, then extracts maximal alphanumeric runs while keeping interior
    ``-``/``.``/``/`` separators inside a run, so code-like tokens stay whole and
    findable:

    - ``"Error E1234 on 2024-01-15"`` -> ``["error", "e1234", "on", "2024-01-15"]``
    - ``"serial ABC-123-X"`` -> ``["serial", "abc-123-x"]``
    - ``"see v1.2.3 (build)"`` -> ``["see", "v1.2.3", "build"]``

    Leading/trailing/doubled punctuation is a boundary, so ``"E1234."`` yields
    ``["e1234"]`` and a hyphen run such as ``"a--b"`` splits into ``["a", "b"]``.
    Returns tokens in document order (duplicates preserved) so callers can build
    term frequencies.
    """

    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class Bm25Index:
    """In-memory Okapi BM25 inverted index over a unit (chunk) id space.

    Scores a tokenized query into a ranked unit-id list and is never persisted.
    Two ways to populate it produce an observationally identical index (same
    :meth:`rank` output) for the same set of units:

    - a one-shot bulk build (:meth:`__init__` from texts, or
      :meth:`from_token_lists` from pre-tokenized units), used for the initial
      construction so it stays O(total tokens); and
    - incremental :meth:`add_unit` / :meth:`remove_unit`, used to fold a small
      chunk-level delta into an already-built index without rescanning the rest
      of the corpus.

    Unit positions are stable: a position assigned to a unit id never moves while
    that id is present, so a metadata allowlist expressed as a set of positions
    stays valid across incremental updates (a removed position is simply never
    re-used while the index lives). ``rank`` returns the caller's unit ids
    verbatim. Positions are an internal detail; ids are the stable external key.

    Each unit also belongs to a document *group* (its ``group_id``), so a whole
    document's chunks can be replaced or removed in one call without the caller
    tracking the old chunk ids. :meth:`replace_group` and :meth:`remove_group`
    fold a single document's edit into an already-built index in O(that group's
    units); they leave an index that ranks identically (ids and scores) to a
    fresh bulk build over the final unit set. A unit that is added with no
    explicit group is its own group, so the per-unit :meth:`add_unit` /
    :meth:`remove_unit` API is unchanged.
    """

    __slots__ = (
        "_postings",
        "_doc_freq",
        "_doc_len",
        "_unit_id_by_pos",
        "_pos_by_unit_id",
        "_terms_by_pos",
        "_group_by_pos",
        "_positions_by_group",
        "_n",
        "_total_len",
        "_next_pos",
        "_k1",
        "_b",
    )

    def __init__(
        self,
        unit_ids: Sequence[str],
        texts: Sequence[str],
        *,
        group_ids: Sequence[str] | None = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Builds the inverted index, document lengths, and corpus statistics.

        ``unit_ids`` and ``texts`` are positionally aligned; each text is
        tokenized with :func:`tokenize`. ``group_ids`` (if given) aligns with
        ``unit_ids`` and assigns each unit's document group; it defaults to each
        unit being its own group. To build from already-tokenized units (e.g. a
        persisted postings store), use :meth:`from_token_lists`.
        """

        if len(unit_ids) != len(texts):
            raise ValueError("unit_ids and texts must be the same length")
        self._build(
            unit_ids,
            [tokenize(text) for text in texts],
            group_ids=group_ids,
            k1=k1,
            b=b,
        )

    @classmethod
    def from_token_lists(
        cls,
        unit_ids: Sequence[str],
        token_lists: Sequence[Sequence[str]],
        *,
        group_ids: Sequence[str] | None = None,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> Bm25Index:
        """Builds the index from already-tokenized units, skipping re-tokenization.

        ``unit_ids`` and ``token_lists`` are positionally aligned; each inner
        sequence is one unit's tokens in document order (duplicates preserved, as
        :func:`tokenize` produces). ``group_ids`` (if given) aligns with
        ``unit_ids`` and assigns each unit's document group, defaulting to each
        unit being its own group. This is the constructor for a persisted lexical
        index: the tokens were captured at ingest time, so the on-disk store
        rebuilds an identical index with no raw text and no regex pass.
        """

        if len(unit_ids) != len(token_lists):
            raise ValueError("unit_ids and token_lists must be the same length")
        index = cls.__new__(cls)
        index._build(
            unit_ids,
            [list(tokens) for tokens in token_lists],
            group_ids=group_ids,
            k1=k1,
            b=b,
        )
        return index

    def _build(
        self,
        unit_ids: Sequence[str],
        token_lists: list[list[str]],
        *,
        group_ids: Sequence[str] | None = None,
        k1: float,
        b: float,
    ) -> None:
        """Builds the postings, document lengths, and corpus stats from token lists.

        Assigns positions ``0..N-1`` to the units in order and populates every
        per-position map. ``group_ids`` (if given) aligns with ``unit_ids`` and
        assigns each unit's document group; when ``None`` each unit is its own
        group. Shared by :meth:`__init__` (which tokenizes texts first) and
        :meth:`from_token_lists` (which receives tokens directly), so both bulk
        paths and a sequence of :meth:`add_unit` calls over the same units
        produce an index that ranks identically.
        """

        self._k1 = float(k1)
        self._b = float(b)
        # term -> {pos: term_frequency}; df is the count of distinct posts.
        self._postings: dict[str, dict[int, int]] = {}
        self._doc_freq: dict[str, int] = {}
        self._doc_len: dict[int, int] = {}
        self._unit_id_by_pos: dict[int, str] = {}
        self._pos_by_unit_id: dict[str, int] = {}
        # Distinct terms per position, so a unit can be removed by walking only
        # its own terms instead of scanning the whole vocabulary.
        self._terms_by_pos: dict[int, tuple[str, ...]] = {}
        # Document-group membership: a unit's group (pos -> group id) and the
        # reverse index (group id -> set of positions) so a whole document's
        # chunks can be replaced or removed in O(that group's units).
        self._group_by_pos: dict[int, str] = {}
        self._positions_by_group: dict[str, set[int]] = {}
        self._n = 0
        self._total_len = 0
        self._next_pos = 0
        if group_ids is None:
            for unit_id, tokens in zip(unit_ids, token_lists, strict=True):
                unit_id = str(unit_id)
                self._add_at(unit_id, tokens, unit_id)
        else:
            if len(group_ids) != len(unit_ids):
                raise ValueError("group_ids must align with unit_ids")
            for unit_id, tokens, group_id in zip(
                unit_ids, token_lists, group_ids, strict=True
            ):
                self._add_at(str(unit_id), tokens, str(group_id))

    def _add_at(self, unit_id: str, tokens: Sequence[str], group_id: str) -> None:
        """Inserts ``unit_id`` (assumed absent) at a fresh position in ``group_id``.

        The single insertion primitive shared by the bulk build and
        :meth:`add_unit`; it never checks for an existing id, so callers that
        support upsert must :meth:`remove_unit` first. The unit joins document
        group ``group_id`` (both group maps recorded) so the whole group can
        later be replaced or removed without the caller tracking chunk ids.
        """

        pos = self._next_pos
        self._next_pos += 1
        length = len(tokens)
        self._doc_len[pos] = length
        self._total_len += length
        self._n += 1
        self._unit_id_by_pos[pos] = unit_id
        self._pos_by_unit_id[unit_id] = pos
        self._group_by_pos[pos] = group_id
        self._positions_by_group.setdefault(group_id, set()).add(pos)
        if not tokens:
            self._terms_by_pos[pos] = ()
            return
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, frequency in counts.items():
            posting = self._postings.get(token)
            if posting is None:
                self._postings[token] = {pos: frequency}
                self._doc_freq[token] = 1
            else:
                posting[pos] = frequency
                self._doc_freq[token] += 1
        self._terms_by_pos[pos] = tuple(counts)

    def add_unit(
        self, unit_id: str, tokens: Sequence[str], group_id: str | None = None
    ) -> None:
        """Adds (or upserts) one unit's tokens, keeping corpus stats current.

        If ``unit_id`` is already present it is :meth:`remove_unit`-ed first, so
        a re-add replaces the old tokens rather than double-counting. The new
        unit takes a fresh, never-reused position; existing positions do not
        move, so a previously computed position allowlist stays valid.
        ``group_id`` assigns the unit's document group and defaults to
        ``unit_id`` (so a unit added with no explicit group is its own group).
        """

        unit_id = str(unit_id)
        if unit_id in self._pos_by_unit_id:
            self.remove_unit(unit_id)
        self._add_at(unit_id, tokens, unit_id if group_id is None else str(group_id))

    def _remove_at(self, pos: int) -> None:
        """Removes the unit at ``pos`` and undoes all of its bookkeeping.

        The single removal primitive: it drops the unit's postings and document
        frequencies (walking only its own distinct terms, so it is O(distinct
        terms in the unit), not O(vocabulary)), its length and corpus totals, and
        its unit-id and group maps. Both :meth:`remove_unit` and
        :meth:`remove_group` resolve ids to positions and delegate here.
        """

        for term in self._terms_by_pos.pop(pos, ()):
            posting = self._postings.get(term)
            if posting is None:
                continue
            posting.pop(pos, None)
            if posting:
                self._doc_freq[term] = len(posting)
            else:
                # Last document for this term: drop the term from the vocabulary
                # so its df/idf vanish exactly as in a fresh build without it.
                del self._postings[term]
                del self._doc_freq[term]
        self._total_len -= self._doc_len.pop(pos, 0)
        unit_id = self._unit_id_by_pos.pop(pos, None)
        if unit_id is not None:
            self._pos_by_unit_id.pop(unit_id, None)
        group_id = self._group_by_pos.pop(pos, None)
        if group_id is not None:
            positions = self._positions_by_group.get(group_id)
            if positions is not None:
                positions.discard(pos)
                if not positions:
                    del self._positions_by_group[group_id]
        self._n -= 1

    def remove_unit(self, unit_id: str) -> None:
        """Removes one unit and its postings; a no-op if the id is absent.

        Looks the unit's stable position up and delegates to :meth:`_remove_at`,
        which walks only the unit's own distinct terms, so removal is O(distinct
        terms in the unit), not O(vocabulary).
        """

        pos = self._pos_by_unit_id.get(str(unit_id))
        if pos is None:
            return
        self._remove_at(pos)

    def remove_group(self, group_id: str) -> None:
        """Removes every unit in document group ``group_id``; a no-op if absent.

        Walks only the group's own positions (O(that group's units)), so a whole
        document's chunks drop without the caller tracking the old chunk ids.
        """

        positions = self._positions_by_group.get(str(group_id))
        if not positions:
            return
        for pos in tuple(positions):
            self._remove_at(pos)

    def replace_group(
        self, group_id: str, units: Sequence[tuple[str, Sequence[str]]]
    ) -> None:
        """Replaces document group ``group_id`` with ``units`` (``(unit_id, tokens)``).

        Drops the group's current units (:meth:`remove_group`) then adds each
        given unit into that group, so an edited document's chunks replace its
        old chunks in O(old + new units). Passing an empty ``units`` removes the
        group (drops its old units and adds nothing).
        """

        group_id = str(group_id)
        self.remove_group(group_id)
        for unit_id, tokens in units:
            self.add_unit(unit_id, tokens, group_id)

    @property
    def unit_ids(self) -> frozenset[str]:
        """Returns the set of currently indexed unit ids."""

        return frozenset(self._pos_by_unit_id)

    def position_of(self, unit_id: str) -> int | None:
        """Returns the stable position for ``unit_id``, or ``None`` if absent."""

        return self._pos_by_unit_id.get(str(unit_id))

    def __len__(self) -> int:
        """Returns the number of indexed units."""

        return self._n

    def _idf(self, term: str) -> float:
        """Returns the non-negative Lucene BM25 IDF for a term."""

        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def rank(
        self,
        query: str,
        *,
        limit: int | None = None,
        allowed_indices: set[int] | None = None,
    ) -> list[tuple[str, float]]:
        """Scores a query and returns ``(unit_id, score)`` best-first.

        Only units with a positive BM25 score are returned, so a unit that
        shares no query term is never ranked (a zero-overlap unit must not enter
        fusion just because the corpus is small). ``allowed_indices`` restricts
        scoring to a metadata-filtered subset (stable unit positions, e.g. from
        :meth:`position_of`); ``limit`` caps the returned list. Ties break on
        unit id so the order is deterministic and matches the vector path's tie
        discipline.
        """

        if self._n == 0:
            return []
        query_terms = tokenize(query)
        if not query_terms:
            return []
        # Distinct query terms; repeats in the query do not change BM25 here.
        scores: dict[int, float] = {}
        k1 = self._k1
        b = self._b
        avgdl = (self._total_len / self._n) if self._n else 1.0
        avgdl = avgdl or 1.0
        for term in set(query_terms):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for pos, term_frequency in posting.items():
                if allowed_indices is not None and pos not in allowed_indices:
                    continue
                length = self._doc_len[pos]
                denominator = term_frequency + k1 * (1.0 - b + b * (length / avgdl))
                if denominator <= 0.0:
                    continue
                contribution = idf * (term_frequency * (k1 + 1.0)) / denominator
                scores[pos] = scores.get(pos, 0.0) + contribution
        unit_id_by_pos = self._unit_id_by_pos
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], unit_id_by_pos[item[0]]),
        )
        if limit is not None:
            ranked = ranked[:limit]
        return [(unit_id_by_pos[pos], score) for pos, score in ranked]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    c: int = RRF_C,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuses ranked id lists with Reciprocal Rank Fusion, best-first.

    Each ``ranking`` is an id list ordered best-first (rank 1 is position 0). An
    id's fused score is ``Σ wᵢ / (c + rankᵢ(id))`` summed over the rankers that
    returned it (1-based rank). RRF needs no score normalization, which is why it
    composes a cosine-scored vector list with a BM25-scored lexical list without
    rescaling either. Within a single ranker a repeated id keeps its first
    (best) rank. Ties in the fused score break on the id string so the fused
    order is deterministic. ``weights`` (if given) must align with ``rankings``;
    it defaults to all ones.
    """

    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")
    fused: dict[str, float] = {}
    for ranker_index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else float(weights[ranker_index])
        seen: set[str] = set()
        for position, raw_id in enumerate(ranking):
            unit_id = str(raw_id)
            if unit_id in seen:
                continue
            seen.add(unit_id)
            rank = position + 1
            fused[unit_id] = fused.get(unit_id, 0.0) + weight / (c + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def fuse_unit_rankings(
    vector_unit_ids: Sequence[str],
    lexical_unit_ids: Sequence[str],
    *,
    c: int = RRF_C,
) -> list[str]:
    """Returns the RRF-fused unit-id order for a vector and a lexical ranking.

    Thin convenience over :func:`reciprocal_rank_fusion` for the two-ranker
    hybrid case; returns only the fused id order (scores dropped) since callers
    re-derive a presentation score downstream.
    """

    fused = reciprocal_rank_fusion((vector_unit_ids, lexical_unit_ids), c=c)
    return [unit_id for unit_id, _score in fused]


def build_chunk_texts(
    document_texts: Mapping[str, str],
    document_chunk_ids: Mapping[str, Sequence[str]],
    chunker,  # noqa: ANN001 - callable(text, limit) -> sequence of chunk strings
    chunk_character_limit: int,
) -> tuple[list[str], list[str], list[str]]:
    """Reconstructs ``(chunk_ids, chunk_texts, group_ids)`` from stored document texts.

    Re-chunks each stored document with the same ``chunker`` the ingest path uses
    and zips the result against the document's recorded chunk ids, so the lexical
    index shares the exact chunk id space the vector scan ranks over. A document
    whose stored text is absent (e.g. text retention off for it) contributes no
    chunks. ``group_ids`` carries the owning document id per chunk so the index
    can be built with document-group membership. Returns three positionally
    aligned lists ready for :class:`Bm25Index`.
    """

    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
    group_ids: list[str] = []
    for document_id, ids in document_chunk_ids.items():
        text = document_texts.get(document_id)
        if text is None:
            continue
        pieces = chunker(text, chunk_character_limit)
        # Defensive zip: if a re-chunk ever disagrees in count with the recorded
        # ids (it should not, the chunker is deterministic), align on the shorter
        # so we never mislabel a chunk's text.
        for chunk_id, piece in zip(ids, pieces, strict=False):
            chunk_ids.append(str(chunk_id))
            chunk_texts.append(piece)
            group_ids.append(str(document_id))
    return chunk_ids, chunk_texts, group_ids


def build_chunk_token_lists(
    document_tokens: Mapping[str, Sequence[Sequence[str]]],
    document_chunk_ids: Mapping[str, Sequence[str]],
) -> tuple[list[str], list[list[str]], list[str]]:
    """Flattens per-document token lists into ``(chunk_ids, token_lists, group_ids)``.

    The counterpart to :func:`build_chunk_texts` for the persisted lexical index:
    each document's stored per-chunk token lists are zipped against its recorded
    chunk ids, so the lexical index shares the exact chunk id space the vector
    scan ranks over without re-chunking or re-tokenizing any raw text. A document
    with no stored tokens contributes no chunks. ``group_ids`` carries the owning
    document id per chunk so the index can be built with document-group
    membership. Returns three positionally aligned lists ready for
    :meth:`Bm25Index.from_token_lists`.
    """

    chunk_ids: list[str] = []
    token_lists: list[list[str]] = []
    group_ids: list[str] = []
    for document_id, ids in document_chunk_ids.items():
        chunks = document_tokens.get(document_id)
        if not chunks:
            continue
        # Defensive zip: the chunker is deterministic, so the recorded ids and
        # the captured token lists agree in count; align on the shorter anyway so
        # a mismatch never mislabels a chunk's tokens.
        for chunk_id, tokens in zip(ids, chunks, strict=False):
            chunk_ids.append(str(chunk_id))
            token_lists.append([str(token) for token in tokens])
            group_ids.append(str(document_id))
    return chunk_ids, token_lists, group_ids
