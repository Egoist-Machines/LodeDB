# Findings: engine-side filtered enumeration (O(matches), not O(corpus))

Investigation of [`research-prompts/02`](../research-prompts/02-engine-side-filtered-enumeration.md).
Code-grounded, measured, and cited. **Design input, not a committed decision.**

---

## 1. TL;DR

- `LodeDB.list_documents(filter=)` is **O(corpus)** today: the SDK calls the engine's
  unfiltered `list_documents()`, which materializes a redacted payload for *every*
  document (`core.py:1849-1852`), then applies the compiled predicate in Python
  (`db.py:643-662`). Cost tracks corpus size, not result-set size.
- Measured (local, `HashEmbeddingBackend(native_dim=384)`, **50 matches held fixed**
  as the corpus grows): `list_documents(filter={"kind":"target"})` rises
  **2.1 ms → 612 ms** from 2k → 200k docs (~linear). A prototype O(matches) resolver
  on the *same* in-memory state stays **flat at ~0.015 ms** (and `count` at
  ~0.004 ms) — a ~40,000× gap at 200k. Full table in §2.
- The engine already owns the structure to fix this: the generation-keyed
  `_MetadataPostingIndex` (`core.py:356-401`), reached through the shared
  `_build_filter_allowlist` (`core.py:2758-2773`). Its `allowlist()` internally
  resolves a **document set** (`core.py:382-397`) before expanding to chunk ids —
  enumeration is exactly "stop at the document set."
- **Recommended design** (§4): add a posting-level `document_allowlist()` that returns
  the matching *doc-id set* (no chunk expansion); an engine
  `list_documents(filter=, after=, limit=)` that resolves that set and materializes
  only those redacted records via the existing `_document_resource_payload`
  (`core.py:4496-4504`), with a stable-id keyset cursor for streaming; and a
  `count(filter=)` returning `len(set)` with **zero** record materialization. Wire
  `LodeDB.list_documents(filter=)` + a new `LodeDB.count(filter=)` to it. Predicate
  filters the postings can't resolve route through the **same shared resolver** that
  prompt 01's planner upgrades — one resolution path for search and enumeration.
- All three invariants hold: records stay payload-free; the posting index is built
  lazily off the committed snapshot (no commit-path cost, `core.py:2734-2756`); and a
  `read_only=True` handle already serves a consistent single-generation snapshot
  (`core.py:574-580`) — enumeration needs no new isolation mechanism. **No on-disk
  format change**; existing readers unaffected.

---

## 2. Current behavior (code-grounded + measured)

### 2.1 The SDK path is O(corpus)

`LodeDB.list_documents(filter=)` (`src/lodedb/local/db.py:613-662`):

```python
raw = self._index.list_documents()                 # db.py:636  -> materialize ALL
records = [_public_document_record(r) for r in raw] # db.py:643  -> project ALL
...
for record in records:                              # db.py:656  -> filter in Python
    if allow_ids is not None and record["id"] not in allow_ids: continue
    if predicate is not None and not predicate(record["metadata"]): continue
```

The docstring is explicit that "the match currently runs in-process over the
enumerated records" (`db.py:630-632`). `LodeIndex.list_documents()`
(`src/lodedb/engine/index.py:338-351`) takes **no filter argument** — it forwards to
the engine and returns every record.

The engine's `LodeEngine.list_documents` (`src/lodedb/engine/core.py:1838-1853`):

```python
documents = [
    _document_resource_payload(state, document_id)      # core.py:1850
    for document_id in sorted(state.document_hashes)    # core.py:1851  -> O(corpus), sorted
]
```

So each call is **O(N log N)** in corpus size (the `sorted`) plus an O(N) payload
build plus an O(N) Python predicate pass in the SDK — at a result size that may be
tiny. `db.count()` (`db.py:676-679`) returns the *unfiltered* total from
`stats()["document_count"]` (`core.py:1982`); there is **no `count(filter=)`** anywhere.

By contrast, `get_document(id)` is already O(1): `document_id in state.document_hashes`
then one `_document_resource_payload` (`core.py:1856-1872`). The asymmetry is the
whole problem — point reads are indexed, set reads are not.

### 2.2 The structure to fix it already exists

`_MetadataPostingIndex` (`core.py:356-401`) maps `(key, value) -> {document_id}` plus
`document_id -> [chunk_id]`. Its `allowlist()` (`core.py:379-401`) **already computes
the matching document set** before expanding to chunks:

```python
document_set = set(posting) if document_set is None else (document_set & posting)  # 389
...
for document_id in document_set:                       # 399  <-- the set enumeration wants
    chunk_ids.extend(self._chunks_by_document.get(document_id, ()))                # 400
```

For search, the chunk expansion (lines 398-401) is necessary — the TurboVec scan is
keyed by chunk. For **enumeration there are no chunks to scan**: we want
`document_set` and stop. The posting index is built lazily and cached per generation
(`_metadata_posting_index`, `core.py:2734-2756`), so it adds **nothing** to the
write/commit path — it is rebuilt on the first filtered read of a new generation,
exactly like the resident GPU session.

`_build_filter_allowlist` (`core.py:2758-2773`) is the single dispatch point:
equality filters → posting index; predicate filters (`_is_predicate_filter`,
`core.py:4454-4466`) → `_scan_filter_allowlist` (a compiled O(corpus) scan,
`core.py:2775-2791`). It is called from both the single-query (`core.py:2174`) and
batch (`core.py:2517`) paths — this is the function prompt 01's planner refactors, and
the function enumeration must share (§4.5).

### 2.3 Measured scaling (local PoC)

Harness: `/tmp/enum_bench.py` (scratch). One vector per doc via `add_vectors_many`
(1 chunk/doc) using `HashEmbeddingBackend(native_dim=384)` — no model download.
**`kind="target"` is carried by exactly 50 docs at every corpus size**, so the result
set is fixed and only the corpus grows. Median of 7 runs, posting index pre-warmed for
both paths. `proto_*` is the §4 design prototyped against the live engine's private
internals (no repo file modified):

| corpus  | current `list(filter=)` | proto `list(filter=)` | proto `count(filter=)` | matches |
|--------:|------------------------:|----------------------:|-----------------------:|--------:|
|   2,000 |                2.101 ms |             0.0154 ms |              0.0039 ms |      50 |
|  10,000 |               10.821 ms |             0.0153 ms |              0.0039 ms |      50 |
|  50,000 |              132.554 ms |             0.0158 ms |              0.0040 ms |      50 |
| 200,000 |              611.809 ms |             0.0152 ms |              0.0040 ms |      50 |

The current path grows **~291×** (2k→200k, near-linear in corpus); the prototype is
**flat** (corpus-independent), confirming O(matches). At 200k the speedup is ~40,000×
for `list` and ~150,000× for `count`. This is precisely the prompt's success criterion
— "latency scales with the result-set size, not the corpus size" — and the shape a
1-hop graph traversal needs to stay flat as the graph grows.

> The prototype's equality resolver (`/tmp/enum_bench.py:proto_resolve_doc_ids`)
> intersects `posting._postings[(k,v)]` sets and returns `sorted(doc_set)`, then
> `_document_resource_payload` materializes only those ids — i.e. the §4 design, run
> end-to-end against real engine state.

---

## 3. Prior art (cited)

Filter-only enumeration and count-without-materialization are standard primitives;
the design below mirrors well-trodden patterns.

- **Qdrant — `scroll` and `count`.** `scroll` "returns all points in a page-by-page
  manner … looks only for points which satisfy filtering conditions" with **no query
  vector**, sorted by id, paged via a `next_page_offset` cursor (the docs warn that
  "passing huge offset values can cause performance issues; use the next_page_offset
  cursor pattern instead") — a keyset cursor, not numeric offset. `count` "Counts the
  number of points that match a specified filtering condition" and "does **not**
  require a query vector," with an `exact` flag: *"If true, count exact number of
  points. If false, count approximate number of points faster."* LodeDB is exact by
  construction, so `count(filter=)` is always the `exact=true` analog. This is the
  direct template for §4. [1][2]

- **Lucene — `Weight#count(LeafReaderContext)` (LUCENE-9620 / apache/lucene#10660).**
  Lucene moved `count` to `Weight` so a query can "return cardinality information
  directly without collecting hits"; `TermQuery` answers from the term's postings
  `docFreq`, `MatchAllDocsQuery` from `maxDoc`, and only otherwise falls back to a
  `BulkScorer + TotalHitCountCollector`. This is exactly the proposed split:
  `count(filter=)` returns `len(doc_set)` straight from postings for the resolvable
  case, and falls back to a costed scan otherwise. `ConstantScoreQuery` likewise
  "strips off all scores" and "does not wrap the inner weight … when scores are not
  required" — the enumeration analog of dropping ranking entirely. [3][4]

- **LanceDB — filter-only scan.** `.where()` "filter data without search" returns rows
  by metadata predicate with no vector, and the docs recommend a **scalar index on
  frequently filtered columns** plus a `limit` on large tables — validating both the
  posting/secondary-index pushdown (prompt 01) and the bounded-page default here. [5]

- **SQLite — keyset / seek pagination.** The seek method "pages from the last row
  already returned" via a `WHERE (sort_key, id) > (cursor)` cursor and "doesn't count
  rows … give me the N rows after the last one I saw," yielding "constant-time
  performance regardless of table size." A stable tie-breaker id "prevents duplicates
  and missing rows." This is the cursor contract for the streaming variant in §4.3. [6]

**Takeaway:** every mature system separates (a) *ranked top-k search* from (b)
*filter-only enumeration* and (c) *count*, resolves (b)/(c) from an index when the
predicate allows, streams (b) via a **stable-id keyset cursor** (not numeric offset),
and answers (c) from index cardinality without materializing rows. LodeDB already has
the index (`_MetadataPostingIndex`) and the redacted projector
(`_document_resource_payload`); it is missing the document-level entry points.

---

## 4. Recommended design

One shared resolver, three thin entry points (enumerate / stream / count), and SDK
wiring. No new on-disk artifact.

### 4.1 Posting-index: a document-level resolver

Add to `_MetadataPostingIndex` (`core.py:356`) a sibling to `allowlist()` that stops
at the document set — no chunk expansion:

```python
def document_allowlist(self, query_filter: Mapping[str, Any]) -> set[str] | None:
    """Matching DOCUMENT ids for an equality+document_ids filter, or None for match-all.

    Identical resolution to allowlist() up to line 397, but returns the doc set
    instead of the chunk expansion. None signals 'no metadata/document_ids
    constraint' so the caller can iterate the full id map without building a set.
    """
    document_set: set[str] | None = None
    metadata = query_filter.get("metadata")
    if metadata:
        for key, value in metadata.items():
            posting = self._postings.get((str(key), str(value)))
            if not posting:
                return set()                      # empty -> no matches
            document_set = set(posting) if document_set is None else (document_set & posting)
            if not document_set:
                return set()
    document_ids = query_filter.get("document_ids")
    if document_ids is not None:
        requested = {str(d) for d in document_ids}
        document_set = requested if document_set is None else (document_set & requested)
    return document_set                            # None == match-all
```

`allowlist()` can be refactored to call this and then expand to chunks, removing the
duplication (lines 382-397 become a single source of truth). Cost: O(matching docs)
for the intersection, independent of corpus.

### 4.2 Engine: `_resolve_document_ids` (shared) + `list_documents(filter=)`

A single engine helper resolves the matching id set, mirroring
`_build_filter_allowlist`'s equality/predicate split but at the **document** grain:

```python
def _resolve_document_ids(self, state, query_filter) -> list[str]:
    if not query_filter:                                         # match-all
        return sorted(state.document_hashes)
    if _is_predicate_filter(query_filter.get("metadata")):
        # Predicate path: doc-level compiled scan (see 4.5). O(corpus) UNTIL
        # prompt 01's planner resolves it from postings/secondary indexes.
        matches = _compile_query_filter(query_filter)
        dm = state.document_metadata
        return sorted(d for d in state.document_hashes
                      if matches(d, dm.get(d, {})))
    doc_set = self._metadata_posting_index(state).document_allowlist(query_filter)
    if doc_set is None:
        return sorted(state.document_hashes)
    return sorted(doc_set)
```

Then add an **overload-compatible** filter to the engine endpoint
(`core.py:1838`) — `query_filter` defaults to `None`, so the existing unfiltered call
and on-the-wire shape are unchanged:

```python
@_synchronized
def list_documents(self, *, context, index_id=None, query_filter=None,
                   after=None, limit=None) -> EngineResponse:
    state = self._index_for_context(context, index_id=index_id, operation="list_documents")
    if isinstance(state, EngineResponse):
        return state
    try:
        validated = _validate_query_filter(query_filter)        # core.py:4364, reused
    except ValueError as exc:
        return self._error(400, str(exc), "list_documents", context.client_id)
    ids = self._resolve_document_ids(state, validated)          # sorted, stable
    page, next_cursor = _page_after(ids, after=after, limit=limit)   # 4.3
    documents = [_document_resource_payload(state, d) for d in page] # only matches
    body = {"status": "ok", "documents": documents}
    if next_cursor is not None:
        body["next_after"] = next_cursor
    return EngineResponse(200, body)
```

`_validate_query_filter` (`core.py:4364-4388`) already accepts exactly
`{"metadata", "document_ids"}` and validates the predicate grammar — **reuse it
verbatim** so enumeration and search share one validator (and one trust boundary).
Materialization touches only `page`, so cost is **O(matches resolved + page size)**.

### 4.3 Streaming / iterator semantics (large result sets)

Resolving a 1M-match set into one list is itself O(matches); for the
"return *everything*" graph cases we want bounded memory. Use a **stable-id keyset
cursor**, per Qdrant `scroll` [1] and SQLite seek [6] — not a numeric offset:

- `after` = the last document id returned; `limit` = page size (default e.g. 1000,
  `None` = no cap for back-compat with today's unbounded return).
- `_page_after(ids, after, limit)` slices the **sorted** id list at the first id `>
  after` and returns `(page, next_after)` where `next_after` is the last id of a full
  page or `None` when drained. Because ids are sorted and unique (`document_hashes`
  keys), the cursor is total and stable within a generation — no duplicates, no gaps.
- SDK iterator (ergonomic, generator-based):

  ```python
  def iter_documents(self, *, filter=None, page_size=1000):
      after = None
      while True:
          resp = self._index.list_documents(query_filter=_normalize_filter(filter),
                                             after=after, limit=page_size)
          docs = resp["documents"]
          if not docs:
              return
          yield from (_public_document_record(d) for d in docs)
          after = resp.get("next_after")
          if after is None:
              return
  ```

  `list_documents(filter=)` stays the eager list (back-compat); `iter_documents` is the
  streaming addition. The graph layer's `reindex()` (which scans **all** nodes then
  **all** edges, `knowledge_graph.py:373,384`) switches to `iter_documents` so a 1M-edge
  rebuild streams at bounded memory.

**Snapshot consistency across pages:** the cursor is only coherent if every page reads
the *same* generation. A `read_only=True` handle is pinned to one committed generation
for its lifetime (§4.6), so it is the correct handle for a long scroll. On a writable
handle, a concurrent commit between pages advances the generation and rebuilds the
posting index (`core.py:2743-2746`); document the scroll as "consistent within a
generation; for a stable long scroll over a changing store, use
`open_readonly`." (Qdrant carries the same caveat for `scroll` under concurrent
writes.)

### 4.4 `count(filter=)` — no materialization

```python
@_synchronized
def count_documents(self, *, context, index_id=None, query_filter=None) -> EngineResponse:
    state = self._index_for_context(context, index_id=index_id, operation="count_documents")
    if isinstance(state, EngineResponse):
        return state
    validated = _validate_query_filter(query_filter)
    if not validated:
        n = len(state.document_hashes)                         # match-all: O(1)
    elif _is_predicate_filter(validated.get("metadata")):
        matches = _compile_query_filter(validated)             # residual scan (4.5)
        dm = state.document_metadata
        n = sum(1 for d in state.document_hashes if matches(d, dm.get(d, {})))
    else:
        doc_set = self._metadata_posting_index(state).document_allowlist(validated)
        n = len(state.document_hashes) if doc_set is None else len(doc_set)
    return EngineResponse(200, {"status": "ok", "count": n})
```

SDK: extend `LodeDB.count` to accept an optional `filter` (keeping
`count()` == total). For equality this is **O(matching docs)** with **zero**
`_document_resource_payload` calls — the Lucene `Weight#count` pattern [3], measured at
~0.004 ms above regardless of corpus. `KnowledgeGraph.stats()` (`knowledge_graph.py:396`)
can then report `db.count(filter={"kind":"node"})` / `{"kind":"edge"}` cheaply instead
of an unfiltered total.

### 4.5 Interaction with prompt 01 (the predicate planner) — one resolution path

Today predicate filters (`$gte`, `$ne`, `$nin`, `$exists`, `$not`, and `$and`/`$or`
that aren't pure equality) fall to a compiled **chunk** scan
(`_scan_filter_allowlist`, `core.py:2775-2791`). The §4.2/§4.4 fallback uses a
**document** scan (`state.document_hashes` × compiled predicate) — same predicate
(`_compile_query_filter`, `core.py:4469-4493`), but iterated once per document instead
of once per chunk, which is strictly cheaper for enumeration (no chunk fan-out).

The key alignment: prompt 01 turns `_build_filter_allowlist` into a **planner** that
resolves predicate operators against postings + new secondary indexes (sorted index
for ordered ops, presence set for `$exists`) and only residual-scans what no index
covers. Enumeration must consume the **document-grain output of that same planner**.
Concretely, the planner should expose a document-level resolver —

```python
def resolve_document_ids(self, state, query_filter) -> set[str] | None: ...
```

— that `allowlist()` (chunk-grained, for search) and `_resolve_document_ids`
(enumeration) both call. When prompt 01 lands, `_resolve_document_ids`'s
`_is_predicate_filter` branch is replaced by `planner.resolve_document_ids(...)`, and
ordered/negation enumeration becomes O(matches) for free — no enumeration-specific
index work. **Build order:** ship §4.1-§4.4 now (equality + `document_ids` go
O(matches) immediately, which is the dominant graph workload —
`{"src": X}`, `{"kind": "edge"}`, `{"type": "Person"}`); predicate enumeration inherits
the speedup when 01 ships. Until then it is no worse than today (same compiled
predicate, fewer iterations).

### 4.6 Read-only behavior

No new mechanism required. A `read_only=True` handle takes no writer lock and loads
each index's single committed generation at open via `_load_persisted_indexes()`
(`core.py:574-580`); the in-memory `state` it serves is a consistent snapshot. The
posting index is generation-keyed (`core.py:2734-2756`) and built lazily on first
filtered read off that snapshot — so `list_documents(filter=)`, `iter_documents`, and
`count(filter=)` all resolve against one frozen generation. The existing test
`tests/test_local_enumeration.py:test_enumeration_on_readonly_handle` already asserts
filtered enumeration on a reader; it keeps passing, and a multi-page scroll on a reader
is inherently stable. Enumeration is **read-only-safe by construction** and is the
recommended handle for long scrolls under concurrent writers.

---

## 5. Tradeoffs / risks / invariant compliance

**Invariants (AGENTS.md):**

- *Payload-free records.* Every path materializes via the unchanged
  `_document_resource_payload` (`core.py:4496-4504`) → `{document_id, metadata,
  chunk_count, content_hash}`. `count` materializes nothing. The SDK projector
  `_public_document_record` (`db.py:791-805`) is unchanged. ✔
- *Snapshot isolation incl. read-only.* §4.6 — reads one committed generation; no lock
  on read-only handles; cursor coherent within a generation. ✔
- *O(changed) commits.* The posting index is built lazily on the first filtered **read**
  of a generation and cached (`core.py:2743-2756`); the new `document_allowlist` reads
  that same cache. **Nothing is added to the upsert/delete/commit path.** ✔
- *No on-disk format change.* The design adds **no persisted artifact** — the posting
  index is an in-memory derived structure rebuilt from `state.document_metadata`.
  Existing `.json`/`.jsd`/`.tvim`/`.tvd`/commit-manifest readers are untouched; an old
  store opens and enumerates with zero migration. ✔ (Prompt 01's *secondary* indexes
  are a separate, opt-in, also-derived concern; this report needs none.)

**Tradeoffs / risks:**

1. **First-read posting build is O(corpus) once per generation.** A cold filtered read
   (or the first after any mutation) pays a one-time O(corpus) build of
   `_metadata_posting_index`. This is *existing* behavior shared with filtered search,
   not new cost, and it amortizes across all subsequent filtered reads/searches of that
   generation. For a write-heavy store where every read sees a new generation, the
   build dominates — but that is already true for filtered `search`, and prompt 01's
   incremental-maintenance discussion applies equally. Worth a benchmark column
   (cold vs warm).
2. **Memory.** Eager `list_documents(filter=)` still builds the full result list (now
   only of matches, so far smaller than today's full corpus). Truly large sets should
   use `iter_documents` / `count`. Document this; default `iter_documents` page size
   bounds peak memory.
3. **`sorted()` on the resolved set.** Keeping results id-sorted preserves a stable
   cursor and matches the engine's current `sorted(state.document_hashes)` contract
   (`core.py:1851`). Cost is O(M log M) in *matches*, not corpus — negligible vs the
   O(corpus) it replaces. Existing tests treat results as sets
   (`test_local_enumeration.py:68-126`), so ordering is not a breaking concern.
4. **Predicate enumeration stays O(corpus) until prompt 01.** Honest scope: equality +
   `document_ids` (the dominant graph traversal filters) go O(matches) now; ordered /
   negation enumeration improves only when the planner lands (§4.5). It is never *worse*
   than today.
5. **Behavioral parity.** The new engine-resolved result set must be **identical** to
   today's Python-filtered set. Mitigation: both compile the same `_predicate` grammar;
   add a property test asserting `set(engine_filtered) == set(python_filtered)` across
   the `test_local_enumeration.py` filter matrix before switching the SDK default.

---

## 6. Prototype / validation plan

**Already prototyped (§2.3):** the equality resolver + payload materialization is
implemented in `/tmp/enum_bench.py` against live engine internals and measured flat
(O(matches)) vs the current O(corpus) path. That validates the core claim locally.

**Implementation (small, localized):**

1. `_MetadataPostingIndex.document_allowlist()` + refactor `allowlist()` to reuse it
   (`core.py:356-401`).
2. `LodeEngine._resolve_document_ids`, `list_documents(query_filter=, after=, limit=)`,
   `count_documents(query_filter=)`, plus `_page_after` (`core.py` near 1838).
3. `LodeIndex.list_documents(filter=, after=, limit=)` + `count_documents(filter=)`
   passthrough (`index.py:338`).
4. SDK: route `LodeDB.list_documents(filter=)` to the engine filter, add
   `LodeDB.count(filter=)` and `LodeDB.iter_documents(...)` (`db.py:613-679`); keep the
   Python predicate **only** as the fallback the engine returns for an unresolved
   predicate (until 01).
5. `KnowledgeGraph.reindex()` → `iter_documents`; `KnowledgeGraph.stats()` →
   `count(filter=)` (`knowledge_graph.py:354-396`).

**Tests:** extend `tests/test_local_enumeration.py` — engine-vs-Python set equality
across the existing filter matrix; cursor completeness/no-duplicates over multiple
pages; `count(filter=)` equals `len(list_documents(filter=))`; read-only multi-page
scroll stability; empty-store and match-all paths.

**Benchmark (`benchmarks/graph_memory/`), expected deltas:**

- Add an `enumeration` sub-benchmark beside `filters`
  (`graph_memory_bench.py:220-266`): for corpus 10k → 1M, hold matches fixed and
  measure `list_documents(filter=eq)`, `count(filter=eq)`, and a **1-hop "all edges of
  a node"** (`{"src": "n…"}`) — expect **flat** latency vs corpus (current: linear), per
  the §2.3 local shape. Report cold (post-mutation) vs warm (cached posting index).
- Add a `reindex` case: full-graph rebuild via `iter_documents` at 10k/100k/1M edges —
  expect bounded memory and throughput limited by re-embedding/topology, not by
  repeated O(corpus) enumeration.
- Modal (A10/L40S) `main_a10` confirms at 50k–1M; assert **no regression** on existing
  `filters`, `vector_in`, and `graph` cases (enumeration adds an in-memory structure
  only; the scan/commit paths are untouched).

---

## 7. Open questions

1. **Default page size & cap.** Should `list_documents(filter=)` stay unbounded
   (today's contract) or gain a default `limit` with `iter_documents` as the unbounded
   escape hatch? LanceDB recommends a `limit` on large tables [5]; Qdrant defaults
   `scroll` to 10 [1]. Proposal: keep `list_documents` eager+unbounded for
   back-compat, make `iter_documents` the bounded streaming primitive.
2. **Cursor opacity.** Expose the raw last-id as `after`, or an opaque token
   (id + generation) so a reused stale cursor across a generation boundary can be
   detected and rejected rather than silently skewing? Opaque is safer for the writable
   handle; raw-id is simpler for the read-only scroll.
3. **`count` exactness signaling.** LodeDB is always exact, so no `exact` flag is needed
   — but should the API mirror Qdrant's `{count: N}` shape [2] for familiarity, or
   return a bare int? (SDK bare int; engine `{"count": N}`.)
4. **Convergence with prompt 01's planner interface.** Lock the shared
   `resolve_document_ids(state, filter) -> set[str] | None` signature now so both
   efforts target it, avoiding a second refactor when 01 lands (§4.5).
5. **`document_ids`-only filters.** A pure `{"document_ids":[...]}` enumeration is just
   a batched `get_document` — should it short-circuit straight to per-id payloads
   (skipping the posting index entirely) for the smallest possible cost? (Trivial:
   `document_allowlist` returns `requested` directly when no metadata constraint.)

---

### Sources

- [1] Qdrant — Scroll points (filter-only, no vector, `next_page_offset` cursor):
  https://api.qdrant.tech/api-reference/points/scroll-points
- [2] Qdrant — Count points (`exact` true/false, no query vector):
  https://api.qdrant.tech/api-reference/points/count-points
- [3] Lucene — `Weight#count(LeafReaderContext)` (LUCENE-9620 / cardinality without
  collecting hits): https://github.com/apache/lucene/issues/10660
- [4] Lucene — `ConstantScoreQuery` (strips scoring; skips inner weight when scores
  not required): https://lucene.apache.org/core/9_9_1/core/org/apache/lucene/search/ConstantScoreQuery.html
- [5] LanceDB — Metadata filtering (`.where()` filter-only scan; scalar index + limit):
  https://docs.lancedb.com/search/filtering
- [6] SQLite/SQL — Seek (keyset) pagination (stable-id cursor, constant time, "doesn't
  count rows"): https://alexanderobregon.substack.com/p/pagination-with-the-seek-method-in
