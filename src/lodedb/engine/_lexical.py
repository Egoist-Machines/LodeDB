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

- :func:`tokenize` — lowercases and splits on whitespace/punctuation while
  keeping alphanumeric runs intact and preserving internal ``-``/``.``/``/``
  inside code-like tokens, so ``E1234``, ``ABC-123``, and ``2024-01-15`` survive
  as single findable tokens. This is the whole point of the feature: the
  embedding cannot see an exact code, but the lexical index can.
- :class:`Bm25Index` — a classic Okapi BM25 inverted index over a fixed unit
  space (LodeDB indexes chunks), scoring a tokenized query into a ranked list.
- :func:`reciprocal_rank_fusion` — fuses ranked id lists with RRF, which needs
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
    """In-memory Okapi BM25 inverted index over a fixed unit (chunk) id space.

    Built once per index generation from raw unit texts and reused for every
    lexical query of that generation; it is never persisted. ``unit_ids`` and
    ``texts`` are positionally aligned; ids may repeat callers' chunk ids
    verbatim (they are returned unchanged from :meth:`rank`).
    """

    __slots__ = (
        "_unit_ids",
        "_postings",
        "_doc_freq",
        "_doc_len",
        "_avgdl",
        "_n",
        "_k1",
        "_b",
    )

    def __init__(
        self,
        unit_ids: Sequence[str],
        texts: Sequence[str],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        """Builds the inverted index, document lengths, and corpus statistics."""

        if len(unit_ids) != len(texts):
            raise ValueError("unit_ids and texts must be the same length")
        self._unit_ids: tuple[str, ...] = tuple(str(unit_id) for unit_id in unit_ids)
        self._k1 = float(k1)
        self._b = float(b)
        # term -> {unit_index: term_frequency}
        postings: dict[str, dict[int, int]] = {}
        doc_len: list[int] = []
        total_len = 0
        for index, text in enumerate(texts):
            tokens = tokenize(text)
            doc_len.append(len(tokens))
            total_len += len(tokens)
            if not tokens:
                continue
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            for token, frequency in counts.items():
                postings.setdefault(token, {})[index] = frequency
        self._postings = postings
        self._doc_freq = {term: len(posting) for term, posting in postings.items()}
        self._doc_len = doc_len
        self._n = len(self._unit_ids)
        self._avgdl = (total_len / self._n) if self._n else 0.0

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
        scoring to a metadata-filtered subset (positions into the build-time
        ``unit_ids``); ``limit`` caps the returned list. Ties break on unit id so
        the order is deterministic and matches the vector path's tie discipline.
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
        avgdl = self._avgdl or 1.0
        for term in set(query_terms):
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for unit_index, term_frequency in posting.items():
                if allowed_indices is not None and unit_index not in allowed_indices:
                    continue
                length = self._doc_len[unit_index]
                denominator = term_frequency + k1 * (1.0 - b + b * (length / avgdl))
                if denominator <= 0.0:
                    continue
                contribution = idf * (term_frequency * (k1 + 1.0)) / denominator
                scores[unit_index] = scores.get(unit_index, 0.0) + contribution
        ranked = sorted(
            scores.items(),
            key=lambda item: (-item[1], self._unit_ids[item[0]]),
        )
        if limit is not None:
            ranked = ranked[:limit]
        return [(self._unit_ids[index], score) for index, score in ranked]


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
) -> tuple[list[str], list[str]]:
    """Reconstructs ``(chunk_ids, chunk_texts)`` from stored document texts.

    Re-chunks each stored document with the same ``chunker`` the ingest path uses
    and zips the result against the document's recorded chunk ids, so the lexical
    index shares the exact chunk id space the vector scan ranks over. A document
    whose stored text is absent (e.g. text retention off for it) contributes no
    chunks. Returns two positionally aligned lists ready for :class:`Bm25Index`.
    """

    chunk_ids: list[str] = []
    chunk_texts: list[str] = []
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
    return chunk_ids, chunk_texts
