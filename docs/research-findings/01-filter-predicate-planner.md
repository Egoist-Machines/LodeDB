# Findings 01 — A filter-predicate planner with posting / secondary-index pushdown

> Investigation of [`../research-prompts/01-filter-predicate-planner.md`](../research-prompts/01-filter-predicate-planner.md).
> Branch: `feat/graph-knowledge-memory`. Design input, not a committed decision.

## 1. TL;DR recommendation

Today exactly **two** predicate shapes ride the O(matches) `_MetadataPostingIndex`
allowlist: a flat map of bare-scalar equalities (`{"topic": "ml"}`) and — because
`coerce_sdk_filter`/`$eq` both stringify to the same posting key — nothing else.
**Every** filter that uses an operator map or a logical operator (`$in`, `$gte`,
`$lt`, `$ne`, `$nin`, `$exists`, `$and`, `$or`, `$not`) is routed by
`_is_predicate_filter` to `_scan_filter_allowlist`, an **O(corpus) per-document
Python matcher loop** over `state.chunks.values()` (`core.py:2771`, `:2785-2791`).
This is the dominant cost: a measured local probe shows that for `range_year` at
20k docs, **8.50 ms of the 9.62 ms** end-to-end search (88%) is the Python
allowlist resolution, and it grows ~linearly with corpus. The native TurboVec
masked scan is the *minority* cost on CPU (it block-skips, `search.rs:1314`), but a
larger fraction on GPU (A10 `range_year` = 72.8 ms at 17.5k). This degradation is
**guaranteed to bite the new graph layer**: `KnowledgeGraph._search_index` wraps
every scoped query as `{"$and": [{"kind": ...}, extra_filter]}`
(`knowledge_graph.py:480`), so even pure-equality edge-type traversal
(`relation $in [...]`) takes the slow scan path.

**Recommendation: build a generation-keyed *filter planner* that compiles the
validated predicate tree into set operations over per-field secondary indexes,
all held in memory exactly like the existing `_MetadataPostingIndex` — zero
on-disk format change, zero commit-path cost.** Add three index families:
(a) reuse the existing `(key,value)` postings for `$eq`/`$in`/`$ne`/`$nin`/`$exists`
(negation = complement vs the generation's full doc set; `$exists` = union of a
field's value-postings); (b) a **per-field sorted `(value, doc_ids)` array** for
ordered operators, answered by `bisect` over the sorted values (O(log V + matches));
(c) a **costed residual-scan fallback** that runs the compiled per-document matcher
*only over the smallest already-resolved candidate set* when an operator has no
helpful index (mixed numeric/string fields, lexicographic ranges). Always end by
handing the resolved chunk-id allowlist to the existing TurboVec scan, preserving
the PR #10 batch-allowlist fast path verbatim. This makes ordered/negation filters
sublinear in corpus at fixed selectivity (resolution drops from O(corpus) to
O(matches + log V)), with the same lazy-rebuild, payload-free, O(changed)-commit
properties the posting index already has.

## 2. Current behavior (code-grounded + measured)

### 2.1 The query path and the exact dispatch

Single-query and batch paths converge on one helper:

- `_query_direct_turbovec_index` (single) calls `_build_filter_allowlist(state,
  query_filter)` when a filter is present (`core.py:2169-2181`), then
  `index.search(query, top_k, allowlist_chunk_ids=...)`.
- `_run_direct_batch_group` (batch) groups queries by `_filter_signature`
  (`core.py:2460`) and calls `_build_filter_allowlist` **once per filter group**
  (`core.py:2517`), then `serving.search_batch(..., allowlist_chunk_ids=...)`.
  So `search_many` amortizes one allowlist resolution across all same-filter
  queries in the batch — relevant to the cost model below.

The dispatch that decides fast-vs-slow is `_build_filter_allowlist`
(`core.py:2758-2773`):

```python
def _build_filter_allowlist(self, state, query_filter):
    if _is_predicate_filter(query_filter.get("metadata")):
        return self._scan_filter_allowlist(state, query_filter)      # O(corpus)
    return self._metadata_posting_index(state).allowlist(query_filter) # O(matches)
```

`_is_predicate_filter` (`core.py:4454-4466`) returns `True` iff **any** top-level
metadata entry is a logical operator (key starts with `$`) **or** maps to an
operator dict:

```python
return any(key.startswith("$") or isinstance(value, Mapping)
           for key, value in metadata.items())
```

So the *only* posting-resolvable shape is a flat map of bare scalars
(`{"topic": "ml", "lang": "en"}`), which `_MetadataPostingIndex.allowlist`
(`core.py:379-401`) resolves by intersecting `(key, value)` posting sets —
O(matching docs + their chunks). A single `$eq` operator map (`{"topic": {"$eq":
"ml"}}`) is **semantically identical** but takes the slow path, because
`isinstance(value, Mapping)` is `True`. (Cheap win flagged in §5.)

### 2.2 What the slow path actually does

`_scan_filter_allowlist` (`core.py:2775-2791`) compiles the predicate once
(`_compile_query_filter`, `core.py:4469-4493`, which wraps
`_predicate.compile_metadata_filter`) and then evaluates it against **every chunk**:

```python
matches = _compile_query_filter(query_filter)
return tuple(
    str(chunk.chunk_id)
    for chunk in state.chunks.values()
    if matches(chunk.document_id, document_metadata.get(chunk.document_id, {}))
)
```

The compiled matcher (`_predicate.py:259-318`) is genuinely well-optimized
per-row: operator dispatch is collapsed to a bound closure, `$in`/`$nin` targets
stay tuples, ordered operands are pre-parsed to numbers once
(`_compile_ordered`, `:287`). But it is still a Python call **per chunk in the
corpus**, regardless of selectivity. There is no per-field index for ordered or
negation operators anywhere in the engine — `grep` confirms the only inverted
structure is `_MetadataPostingIndex`, and it only does `(key,value)` equality
lookups.

### 2.3 The native allowlist scan is *not* free either (but it block-skips)

The chunk-id allowlist is translated to stable ids (`turbovec_index.py:97-104`,
`:121-130`) and passed as `kwargs["allowlist"]` to the Rust kernel.
`IdMapIndex::search_with_allowlist` (`id_map.rs:473-494`) builds a **dense
`vec![false; self.inner.len()]` boolean mask over the entire corpus**, then calls
`search_with_mask`. The kernel does **block-level early-exit**: a 32-vector block
is skipped when no slot in it is allowed (`block_has_allowed`, `search.rs:1314`;
wired into the NEON/AVX scoring loops at `search.rs:197, 452, 598, 627, 1382,
1602`). So:

- A **clustered** allowlist (few contiguous blocks) → native scan is sublinear.
- A **sparse** allowlist (scattered postings, or a low-selectivity predicate like
  `$ne` matching ~90% of rows) → block-skipping rarely fires → native scan stays
  ~O(corpus).
- `effective_top_k` is clamped to `min(k, allowlist.size)` (`turbovec_index.py:131-133`),
  and the kernel returns only allowed ids, so the result is the **true top-k of the
  matching subset** — the invariant holds on every path today, and any planner must
  preserve it (it does: we only change *how the id set is computed*, not the scan).

Net: the planner can attack the **Python resolution** cost universally (it is
layout-independent), and it incidentally *improves native block-skipping* whenever
it makes the allowlist smaller/tighter.

### 2.4 Measured latency gradient

**(a) Existing A10 GPU benchmark** (`benchmarks/graph_memory/results/results_a10.json`,
govreport, 17,517 docs, k=10, 256 queries, `device=cuda`, mean ms):

| case | predicate | path | mean ms | ×eq |
|------|-----------|------|--------:|----:|
| `no_filter` | — | unfiltered | 7.50 | 0.53 |
| `eq_topic` | `{topic: ml}` | **posting** | 14.18 | 1.00 |
| `in_topic_3` | `topic $in [3]` | scan | 38.07 | 2.69 |
| `and_topic_year` | `$and[eq, $gte]` | scan | 34.24 | 2.42 |
| `exists_topic` | `topic $exists` | scan | 47.28 | 3.34 |
| `ne_topic` | `topic $ne ml` | scan | 50.96 | 3.59 |
| `gte_year` | `year $gte 2013` | scan | 56.65 | 4.00 |
| `range_year` | `year $gte 2010 $lt 2015` | scan | 72.81 | **5.14** |

Ordered/negation predicates are 2.4–5.1× the equality posting path and 4.6–9.7×
the unfiltered scan, at only ~17.5k docs.

**(b) Local CPU probe** (`/tmp/filter_probe.py`, run via `uv run --no-sync`,
`HashEmbeddingBackend(native_dim=384)`, scattered metadata identical to the
benchmark `_synth_metadata`: `topic = TOPICS[i%10]`, `year = 2000 + i%26`; k=10,
60 iters/case; **decomposes** end-to-end vs the Python `_build_filter_allowlist`):

| case | E2E 5k | E2E 20k | py-resolve 20k | py % of E2E (20k) | allowlist size 20k |
|------|-------:|--------:|---------------:|------------------:|-------------------:|
| `no_filter` | 0.16 | 0.34 | — | — | — |
| `eq_topic` (posting) | 0.55 | 1.00 | 0.19 | 19% | 2,000 |
| `in_topic_3` | 1.20 | 4.88 | 3.41 | 70% | 6,000 |
| `and_topic_year` | 1.45 | 5.62 | 4.63 | 82% | 923 |
| `ne_topic` | 1.52 | 5.52 | 3.44 | 62% | 18,000 |
| `exists_topic` | 1.48 | 5.28 | 3.16 | 60% | 20,000 |
| `gte_year` | 2.09 | 7.16 | 5.68 | 79% | 9,997 |
| `range_year` | 2.73 | 9.62 | **8.50** | **88%** | 3,845 |

Three things this nails down:

1. **The Python per-document scan dominates predicate-filter latency** (60–88% of
   E2E at 20k) and **scales ~linearly with corpus** (`range_year` py-resolve:
   2.34 ms → 8.50 ms for 5k → 20k, ≈3.6× for 4× corpus). This is the bottleneck a
   planner removes.
2. **`eq_topic` confirms the posting path is cheap**: 0.19 ms resolve at 20k for a
   2,000-chunk allowlist — O(matches), corpus-independent in resolution.
3. **A clustered-vs-scattered run barely moved E2E** (`range_year` 9.62 scattered
   vs 9.95 clustered at 20k) because the Python scan cost is layout-independent —
   confirming that block-skipping helps only the already-small native fraction, and
   the real lever is the resolution algorithm.

**Conclusion for §1**: ordered and negation predicates degrade filtered search
toward a full scan as the corpus grows, and the degradation is overwhelmingly in
the Python resolution loop, not the SIMD kernel.

## 3. Prior art (cited)

| System | Equality | Range (`$gt/$lt`) | Negation (`$ne/$nin`) | Existence | Planner |
|--------|----------|-------------------|------------------------|-----------|---------|
| **Qdrant** | hash/struct payload index | mutable **interval tree** `NumericIndex` (configurable `range`/`lookup` flags) | filter-context check | presence | **cardinality estimator** chooses prefilter (payload index → id set into HNSW) vs in-traversal postfilter vs full scan, gated by `full_scan_threshold` |
| **Weaviate** | **roaring-bitmap** inverted index (`indexFilterable`) → allow-list into HNSW | dedicated **`indexRangeFilters`** = roaring-bitmap *slices* (64-bit values) | bitmap complement | bitmap presence | picks range vs filterable index per operator |
| **LanceDB / Lance** | **BITMAP** scalar index (low cardinality) | **BTREE** scalar index (sorted, binary search; high cardinality) | scan / mask | IS NULL via index | prefilter passes id mask to vector index `filter_row_ids()`; fragment-skipping |
| **Lucene / Elasticsearch** | inverted index (terms) | **BKD tree** (block-KD) over point values; sparse→dense `DocIdSetBuilder` | scan / `MUST_NOT` over a docset | norms/doc-values | per-segment cost |
| **Apache Pinot** | inverted bitmap | **bit-sliced `RangeBitmap`** — range-encoded bit slices, range query = unions/intersections of slices, vectorised | bitmap complement | — | — |
| **MongoDB** | B-tree seek | B-tree **range scan** | **`$ne`/`$nin` are not selective → effectively a full index/collection scan**; docs explicitly advise rewriting to `$in`/`$or` | sparse index | cost-based |
| **SQLite** | B-tree seek | B-tree **range scan** (one binary search + walk) | scan | partial index | cost-based, "stops at first range" |

Synthesized lessons that shape the recommendation:

1. **Every serious system keeps equality on an inverted/bitmap structure and adds
   a *separate* ordered structure for ranges** — Qdrant's interval tree, Weaviate's
   range-filter roaring slices, LanceDB's BTREE, Lucene's BKD, Pinot's bit-sliced
   RangeBitmap. None tries to answer ranges from equality postings. LodeDB has only
   the equality half today.
   ([Qdrant DeepWiki](https://deepwiki.com/qdrant/qdrant/4-search-and-query-processing),
   [Weaviate inverted-index docs](https://docs.weaviate.io/weaviate/config-refs/indexing/inverted-index),
   [LanceDB scalar indexes](https://lancedb.com/docs/indexing/scalar-index/),
   [Lucene BKD DeepWiki](https://deepwiki.com/apache/lucene/4.3-bkd-trees-and-point-values),
   [Pinot RangeBitmap](https://richardstartin.github.io/posts/range-bitmap-index))

2. **Filtered vector search universally resolves the predicate to an id allow-list
   first, then pushes it into the ANN/exact scan** — Weaviate "builds an allow-list
   from the inverted index first, then passes that constrained set into HNSW";
   LanceDB passes a prefilter id mask to `filter_row_ids()`; Qdrant feeds
   `iter_filtered_points()` into search. **This is exactly LodeDB's PR #10
   batch-allowlist contract** — so the planner slots in *behind* the same boundary
   without touching the scan.
   ([Weaviate filtering](https://docs.weaviate.io/weaviate/concepts/filtering),
   [Lance filtering/row-masking](https://deepwiki.com/lancedb/lance/5.4-filtering-and-row-masking))

3. **Negation is not index-friendly and is usually answered as a complement or a
   scan.** Roaring bitmaps make complement (flip) a cheap container-level op
   ([roaring complement](https://arxiv.org/pdf/1709.07821)); MongoDB just treats
   `$ne/$nin` as non-selective and effectively scans
   ([Mongo $ne/$nin pitfalls](https://www.mongodb.com/docs/manual/core/query-optimization/)).
   So computing `$ne` as `full_set − posting(value)` is the principled cheap move,
   and a residual scan over a small candidate set is the accepted fallback.

4. **A cardinality-estimating planner that can *decline* to use an index is the
   norm, not the exception.** Qdrant's `full_scan_threshold` / `PlainPayloadIndex`
   and SQLite's cost model both fall back to a scan when an index would not pay off.
   This validates the "costed residual scan" leg of the design — a planner that is
   honest about when *not* to build an id set is correct, not a cop-out.
   ([Qdrant indexing](https://qdrant.tech/documentation/manage-data/indexing/),
   [SQLite query planner](https://www.sqlite.org/queryplanner.html))

## 4. Recommended design

### 4.1 Where it lives and how it stays within invariants

A new pure module **`src/lodedb/engine/_filter_plan.py`** holds the plan
representation and the planner. It is **stdlib-only** (`bisect`, `dataclasses`,
`collections.abc`, `typing`) and **must not import `core`** — same rule as
`_predicate.py`. It depends on `_predicate` only for the residual-scan compile.
This keeps `_predicate.py` importable without `core` (the import-boundary guard
stays green) and adds no PyPI dependency (no roaring-bitmap library, no numpy in
this module).

`core.py` owns the **index storage** (analogous to `_MetadataPostingIndex`) and
the binding of a plan to a generation, because the indexes live on engine state.
The planner itself (`_filter_plan.py`) is given the index objects and returns a
resolved `set[str]` of document ids — it never reaches into `core`.

Invariant compliance, point by point:

- **Payload-free artifacts**: the secondary indexes store only the same redacted
  `(key, value)` strings already in `state.document_metadata` (which is already in
  the redacted `.json` snapshot) plus doc-id sets. No raw text, no vectors. And —
  critically — **nothing new is persisted** (see §4.5), so no artifact format
  changes at all.
- **O(changed) commit path**: the indexes are **built lazily, keyed by generation**,
  exactly like `_metadata_posting_index` (`core.py:2734-2756`). The commit path is
  untouched; the first filtered query of a new generation rebuilds them (or, as an
  optimization, see §4.4 incremental option). No `prepare()`, no repack, no
  O(corpus) work added to commit.
- **Lean deps**: stdlib only.
- **True top-k**: the planner only changes *how the chunk-id allowlist is computed*.
  The allowlist still flows into `index.search(..., allowlist_chunk_ids=...)`
  unchanged, so top-k semantics are byte-identical to today.

### 4.2 Plan representation

A small frozen-dataclass IR — a tree of set-producing nodes — emitted by the
planner from the *already-validated* predicate (it trusts the grammar, like
`compile_metadata_filter`):

```python
# _filter_plan.py  (illustrative)
@dataclass(frozen=True)
class PostingUnion:        # $eq / $in  -> union of (key,value) postings
    field: str
    values: tuple[str, ...]

@dataclass(frozen=True)
class Complement:          # $ne / $nin / $not / $exists:False -> full_set - inner
    inner: "PlanNode"

@dataclass(frozen=True)
class FieldPresence:       # $exists:True -> union of all postings for `field`
    field: str

@dataclass(frozen=True)
class RangeScan:           # $gt/$gte/$lt/$lte (numeric) -> bisect a sorted index
    field: str
    lo: float | None; lo_inclusive: bool
    hi: float | None; hi_inclusive: bool

@dataclass(frozen=True)
class Intersect:           # $and (and AND-ed sibling fields)
    children: tuple["PlanNode", ...]

@dataclass(frozen=True)
class Union:               # $or
    children: tuple["PlanNode", ...]

@dataclass(frozen=True)
class ResidualScan:        # no helpful index: run compiled matcher over a candidate set
    predicate: Callable[[Mapping[str, str]], bool]   # from _predicate
    over: "PlanNode | None"  # restrict to this candidate set; None = whole corpus

PlanNode = PostingUnion | Complement | FieldPresence | RangeScan | Intersect | Union | ResidualScan
```

The IR is **id-set-valued**: every node evaluates to a `set[str]` of *document*
ids (chunk expansion happens once at the end, reusing
`_MetadataPostingIndex._chunks_by_document`). Operating on document ids — not chunk
ids — is correct because the predicate is a function of document metadata (a
document's chunks all share its metadata), and it keeps the intermediate sets
~`chunk_count / avg_chunks_per_doc` smaller.

### 4.3 Compiling each operator to a plan node

| Predicate | Plan node | Index used | Resolution cost |
|-----------|-----------|------------|-----------------|
| `{field: scalar}`, `field: {$eq: v}` | `PostingUnion(field, (v,))` | `(key,value)` postings | O(matches) |
| `field: {$in: [..]}` | `PostingUnion(field, values)` | postings | O(Σ matches) |
| `field: {$ne: v}` | `Complement(PostingUnion(field,(v,)))` | postings + full set | O(matches + \|full\|) set-diff |
| `field: {$nin: [..]}` | `Complement(PostingUnion(field,values))` | postings + full set | O(Σ matches + \|full\|) |
| `field: {$exists: True}` | `FieldPresence(field)` | per-field value index | O(matches) |
| `field: {$exists: False}` | `Complement(FieldPresence(field))` | per-field + full set | O(matches + \|full\|) |
| `field: {$gte: a, $lt: b}` (numeric) | `RangeScan(field, a, True, b, False)` | **sorted-value index** | O(log V + matches) |
| `field: {$gt: ...}` (operand non-numeric ⇒ lexicographic) | `ResidualScan` over best sibling | — | O(candidate) |
| `$and: [...]`, AND-ed sibling fields | `Intersect(children)` | — | smallest-first intersect |
| `$or: [...]` | `Union(children)` | — | union |
| `$not: f` | `Complement(plan(f))` | + full set | set-diff |

**Mixed-field AND-ing**: a node like `{topic: "ml", year: {$gte: 2013}}` becomes
`Intersect(PostingUnion(topic,("ml",)), RangeScan(year, 2013, True, None, _))`. The
evaluator orders `Intersect` children by **ascending estimated cardinality**
(cheapest, smallest set first) so the range leg can, when beneficial, be evaluated
as a `ResidualScan(over=<the topic posting set>)` — i.e. the residual matcher runs
over just the `topic=ml` docs, not the corpus. This is the planner's core trick and
directly mirrors Qdrant's "narrow candidates before vector computation."

### 4.4 The secondary index (ordered fields) — structure, build, maintenance

Add one structure per *ordered-queried* field. Because metadata is stored as
strings (`_predicate.py` docstring), the ordered index is **numeric-typed**: it
indexes only the values of a field that parse as finite numbers (via
`_predicate._as_number`); non-numeric values stay out of it and force the
lexicographic residual path for that field (rare; matches the existing
"lexicographic fallback" semantics).

```python
class _OrderedFieldIndex:      # lives in core.py, built lazily like the posting index
    __slots__ = ("values", "doc_ids")   # parallel arrays, sorted by value
    # values:  list[float]              (ascending)
    # doc_ids: list[frozenset[str]]     (docs whose stored `field` == values[i])
```

- **Range query** `lo..hi`: two `bisect` calls give the index window
  `[i, j)`; union the `doc_ids` frozensets in that slice. Cost **O(log V + matches)**
  in resolution — sublinear in corpus at fixed selectivity, which is the success
  criterion. (This is the LanceDB BTREE / SQLite range-scan idea, minimal-form.)
- The structure is a sorted **array**, not a tree — it is rebuilt per generation
  (never mutated in place), so insertion cost is irrelevant; only build + query
  cost matter, and a sorted array beats a tree on both for a build-once structure.

**Build cost & where it runs.** Two options; recommend **(A)** first, with (B) as a
follow-up if rebuild latency on huge corpora shows up:

- **(A) Lazy per-generation rebuild (recommended, ships first).** Build all needed
  secondary indexes in `_metadata_posting_index`'s sibling builder, on the **first
  filtered query of a generation**, from `state.document_metadata`. This is the
  *exact* model the posting index uses today (`core.py:2734-2756`) — **the commit
  path stays O(changed)**, the cost is one O(corpus) pass amortized over all queries
  of that generation, off the write path. We already pay an O(corpus) pass there to
  build postings; adding sorted arrays is the same asymptotic pass.
- **(B) Incremental per-upsert maintenance (optional).** If even the lazy rebuild is
  too heavy at 1M docs, maintain the posting map incrementally inside
  `upsert_documents`/`upsert_vectors` (which already write
  `state.document_metadata`, `core.py:1087, 1274, 1302, 1387, 1403`, and pop on
  delete, `:1511`): add/remove the doc id from `postings[(k,v)]` and from the
  ordered field's set. **This is genuinely O(changed)** — only the upserted/deleted
  docs' (k,v) pairs are touched — so it respects the invariant. The sorted-array
  *materialization* for range queries can still be derived lazily from the
  incrementally-maintained `value -> doc_ids` map on first range query of a
  generation (sorting V distinct values, not N docs). I recommend deferring (B);
  (A) already delivers the asymptotic win and matches the current pattern exactly.

**Negation / existence reuse the equality postings** — no extra structure:
- `$exists:True(field)` = union of `postings[(field, v)]` over all `v` for that
  field. Maintain a tiny `fields -> set[value]` map (or derive from posting keys) so
  this is O(values + matches).
- `$ne`/`$nin`/`$not`/`$exists:False` = `full_doc_set − inner`. `full_doc_set` is
  `set(self._chunks_by_document)` / `state.document_metadata.keys()`, already
  available. Set-difference is O(\|full\|) — acceptable, and matches every prior-art
  system's treatment of negation; the *real* speedup for negation comes when it is
  AND-ed with a selective positive predicate, where the evaluator computes the
  positive set first and runs the negation as a residual filter over it
  (`ResidualScan(over=positive)`), never materializing the full complement.

### 4.5 On-disk compatibility / migration

**There is nothing to migrate, and no reader can break.** The current
`_MetadataPostingIndex` is **purely in-memory**, rebuilt lazily and **never
serialized** — confirmed: the only references to `_metadata_posting_indexes` are
construction (`core.py:560`), eviction on `delete_index` (`:852`), and lazy
get/build (`:2744, :2755`); it appears in **no** persist/`durable_replace`/commit
path. The proposed secondary indexes follow the identical model: built in memory
from `state.document_metadata` (which is already persisted in the redacted
snapshot/journal), keyed by generation, dropped on mutation. Therefore:

- **No new on-disk artifact, no schema bump, no `.commit.json` change.** A v0.1.x
  store, a current store, and a post-planner store are **byte-identical on disk**.
- Old readers (and old writers) are unaffected; new code rebuilds the in-memory
  structures from data they already have.
- The legacy `<key>.json`-without-commit-manifest fallback path is untouched.

This is the cleanest possible migration story and is the single strongest reason to
keep the indexes in-memory rather than persisting them.

### 4.6 Cost model / when no index helps

The planner attaches an estimated cardinality to each node (postings: `len(set)`;
range: `j - i` window count over `doc_ids` sizes; complement: `|full| - |inner|`;
intersect: min child; union: sum) and uses it to:

1. **Order `Intersect` children smallest-first**, and evaluate later children as
   `ResidualScan(over=<running intersection>)` when their own index is absent or
   their standalone cardinality exceeds the running set — so a residual scan is
   **always over the smallest candidate set, never the corpus**.
2. **Decline indexing when it would not pay** (the Qdrant `full_scan_threshold`
   lesson): if the *entire* predicate is a single low-selectivity residual (e.g. a
   lone lexicographic `$gt` on a high-cardinality string field with no AND partner),
   fall back to today's `_scan_filter_allowlist` over the whole corpus — same cost
   as now, never worse. The planner's contract is "**never slower than the status
   quo, asymptotically faster whenever a positive index or an AND-partner exists.**"

A lone, low-selectivity ordered predicate over a *numeric* field still wins
(sorted-array window beats full scan); the residual-corpus fallback is reserved for
*non-numeric* ordered operands and the (rare) all-negation-no-positive query, where
prior art agrees a scan is the right call.

## 5. Tradeoffs, risks, invariant compliance

- **Cheap immediate win, independent of the whole planner**: route `$eq` and
  single-value `$in` operator-maps through the posting index. `_is_predicate_filter`
  could special-case "operator map whose only keys are `$eq`/`$in`" and lower it to
  the posting `allowlist` instead of the scan. This alone moves
  `KnowledgeGraph` edge-type traversal (`relation $in [...]`) and any `{$eq}` filter
  onto the fast path with ~10 lines and no new structure. **Caveat**: it must run
  *after* the `$and` flattening (graph wraps everything in `$and`), so the real win
  needs the planner's `$and`→intersect lowering, not just the `_is_predicate_filter`
  tweak. Flagged as a follow-up task, not done here (hard constraint: report only).
- **Memory**: secondary indexes add memory proportional to distinct `(key,value)`
  pairs + per-ordered-field sorted arrays — same order as the existing posting
  index, which already holds every `(key,value)→docs`. Bounded and redacted.
- **Lazy-rebuild latency spike**: option (A) pays an O(corpus) build on the first
  filtered query after each mutation — *identical* to the posting index's current
  behavior, so no new risk class; option (B) removes even that. The frozen-set
  `doc_ids` sharing keeps the sorted array's memory modest.
- **Correctness risks to watch**: (1) numeric/lexicographic split must match
  `_predicate._as_number` exactly so planner results == matcher results — the
  validation plan checks this with a differential test (§6); (2) `$ne`/`$exists:False`
  semantics include *missing-field* docs (`_predicate.py:274-283`) — the
  `Complement` over `full_doc_set` naturally includes them (a doc missing the field
  is in `full` but not in `posting(value)`), which is exactly right; (3) `$or` of a
  cheap posting and an expensive residual must union *resolved sets*, not short-circuit
  per-doc — the IR handles this since each branch resolves to a set.
- **Invariants**: payload-free ✓ (only redacted k/v + ids, nothing persisted),
  O(changed) commit ✓ (lazy/generation-keyed, or genuinely incremental in (B)),
  stdlib-only `_filter_plan.py` importable without `core` ✓, true-top-k ✓ (scan
  contract unchanged), back-compat ✓ (no on-disk change).

## 6. Validation plan (precise impl + expected delta)

I ran a **decomposition PoC** (§2.4b, `/tmp/filter_probe.py`) that *measures* the
cost being removed; I did **not** build the planner (hard constraint: report-only,
no source edits). The build-and-validate plan:

1. **Implement** `_filter_plan.py` (IR + planner + evaluator) and the
   `_OrderedFieldIndex` / presence map in `core.py`, lazy-built alongside
   `_metadata_posting_index`. Route `_build_filter_allowlist` through the planner;
   keep `_scan_filter_allowlist` as the residual fallback. ~300–400 LoC, no
   dependency change.
2. **Differential correctness test** (decisive, cheap): for a few thousand random
   predicates over a synthetic corpus, assert the planner's resolved chunk-id set
   **equals** `_scan_filter_allowlist`'s set. This guarantees the optimization is
   behavior-preserving (catches the numeric/lexicographic and missing-field edge
   cases). Add to the local suite (`tests/`).
3. **Benchmark**: `benchmarks/graph_memory/`, the `filters` sub-benchmark
   (`run_filter_bench`, `graph_memory_bench.py:220`), which already measures
   `gte_year`, `range_year`, `ne_topic`, `exists_topic`, `in_topic_3`,
   `and_topic_year` by latency. Run on Modal A10 + L40S at **50k, 200k, 1M** docs
   (`modal_bench.py`), comparing planner vs `main`.
   - **Success criterion**: `range_year`/`gte_year` mean-ms grows **sublinearly**
     with corpus at fixed selectivity (planner) vs the ~linear growth on `main`.
     From the local decomposition (range_year py-resolve 8.50 ms → ~`log V + matches`),
     expect the resolution component to drop by **>10×** at 1M docs, pulling
     `range_year`/`gte_year` from the slowest cases down toward the `in_topic_3`/`eq`
     band. `and_topic_year` should approach `eq_topic` (selective posting +
     residual-over-candidates). Negation-only `ne_topic`/`exists_topic` improve from
     the complement (no per-doc Python), bounded by the O(\|full\|) set-diff.
   - **No-regression gates**: (a) `eq_topic` and `no_filter` unchanged (planner must
     not add overhead to the already-fast posting/unfiltered paths); (b) the
     existing exact-filter and PR #10 batch-allowlist benchmarks unchanged
     (the scan contract is identical). Confirm `search_many` still resolves one
     allowlist per filter group.
4. **Per-generation build-cost check**: measure the lazy index-build time at 1M docs
   to confirm option (A) is acceptable; if not, land option (B) incremental
   maintenance (still O(changed), verified by asserting commit-path timing is flat
   in corpus size).

## 7. Open questions

1. **(A) lazy vs (B) incremental** for ordered-index build at the 1M-doc tier — does
   the first-query rebuild spike justify the extra upsert-path bookkeeping of (B)?
   (Validation step 4 decides; (A) ships first regardless.)
2. **Numeric-index typing policy**: index only numeric-parseable values per field
   (recommended), or also maintain a lexicographically-sorted string array for
   non-numeric ordered fields? The latter doubles range coverage but adds memory and
   a second structure; defer until a real lexicographic-range use case appears
   (graph temporal/score fields are numeric).
3. **`$or` across heterogeneous fields**: when one branch is a selective posting and
   another is a full-corpus residual, is unioning the resolved sets always cheaper
   than a single corpus residual over the whole `$or`? The cost model should pick;
   needs a threshold tuned against the benchmark.
4. **Compound-key fast path** for the graph layer's ubiquitous `{kind, type}` AND —
   worth a synthesized `(kind,type)` posting key, or does smallest-first intersect of
   two single-field postings already suffice? (Likely suffices; measure.)
5. **Should the `_is_predicate_filter` `$eq`/`$in` cheap-lowering ship independently**
   ahead of the full planner as a low-risk interim PR, given it needs the `$and`
   flattening to actually help the graph layer?
