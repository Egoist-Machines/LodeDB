# Findings: remove the per-hit metadata N+1 in search hydration

Answers [`../research-prompts/05-batched-metadata-hydration.md`](../research-prompts/05-batched-metadata-hydration.md).
Code-grounded, measured on this branch (`feat/graph-knowledge-memory`), prior art cited.
Design input — not a committed decision.

---

## 1. TL;DR

Adopt **Option 2 — inline the already-in-memory redacted metadata in the engine
result rows** and have the SDK build hits straight from the row. It is clean,
payload-safe, and needs no new engine method, no new round-trips, and no API
change. The plumbing already exists: `_materialize_query_results` attaches
`candidate["metadata"]` today whenever the query's `include` carries
`"metadata"` ([`core.py:3892-3893`](../../src/lodedb/engine/core.py#L3892)), the
forbidden-key contract already permits `metadata`
([`index.py:594`](../../src/lodedb/engine/index.py#L594)), and the only reason
the N+1 exists is that the SDK never sets that include flag and re-fetches each
hit instead.

The change is two lines of intent: pass `include=("metadata",)` on the engine
queries that back `search`/`search_many`/`search_by_vector`/`search_many_by_vector`,
and have `_hits_from_result_rows` read `row["metadata"]` instead of calling
`_metadata_for_document(id)` per row.

**Measured** (5 000-doc bounded corpus, `HashEmbeddingBackend(native_dim=384)`,
warm SIMD layout, NEON CPU scan, this machine):

| path | get_document calls | end-to-end search latency |
|------|--------------------|----------------------------|
| `search(k=100)` today | **100** (one per hit) | mean 0.821 ms |
| `search(k=100)` Option 2 | **0** | mean 0.351 ms (**−57%**) |
| `search(k=1000)` today | **1000** | mean 8.672 ms |
| `search(k=1000)` Option 2 | **0** | mean 4.621 ms (**−47%**) |
| `search_many(batch=64, k=100)` today | **6400** (= batch·k) | p50 41.2 ms |

The isolated metadata-hydration step alone (holding the scan constant) is
**18–30× faster** as a single batched in-memory read than as the per-hit
`get_document` loop. Parity is exact: inlined metadata is byte-identical to the
N+1 path and leaks no forbidden payload key.

Option 1 (a batched `get_document`-style engine method) also removes the N+1 and
is a fine fallback, but it adds a new public engine method, still copies metadata
that the engine already held one frame earlier in the same call, and does a
second pass over ids the engine just produced. Option 2 dominates it on every
axis here, so Option 1 is recommended *only* if a future refactor makes the
materializer unable to see `state.document_metadata` (it can today).

---

## 2. Current behavior (code-grounded + measured)

### 2.1 The N+1, exactly

Every public search verb funnels through one hydration helper:

- `LodeDB.search` → `self._index.query(...)`
  ([`db.py:411-416`](../../src/lodedb/local/db.py#L411)) →
  `_hits_from_result_rows`.
- `LodeDB.search_many` → `self._index.query_batch(...)` → one
  `_hits_from_result_rows` per query
  ([`db.py:443-460`](../../src/lodedb/local/db.py#L443)).
- `LodeDB.search_by_vector` → `self._index.query_vector(...)` →
  `_hits_from_result_rows` ([`db.py:483-488`](../../src/lodedb/local/db.py#L483)).
- `LodeDB.search_many_by_vector` → `self._index.query_vectors_batch(...)` → one
  `_hits_from_result_rows` per query
  ([`db.py:509-525`](../../src/lodedb/local/db.py#L509)).

`_hits_from_result_rows` loops over result rows and, for each one, calls
`_metadata_for_document(document_id)`
([`db.py:742-759`](../../src/lodedb/local/db.py#L742)):

```python
# db.py:752-757
hits.append(
    LodeSearchHit(
        score=float(row["score"]),
        id=document_id,
        metadata=self._metadata_for_document(document_id),   # one call PER hit
    )
)
```

`_metadata_for_document` is a full engine round-trip:
`self._index.get_document(document_id)`
([`db.py:732-740`](../../src/lodedb/local/db.py#L736)) →
`LodeIndex.get_document`
([`index.py:353-362`](../../src/lodedb/engine/index.py#L353)) →
`LodeEngine.get_document`. That engine method is **`@_synchronized`**
([`core.py:1855-1856`](../../src/lodedb/engine/core.py#L1855)): each per-hit call
re-acquires the reentrant `_op_lock`, mints a fresh `EngineRequestContext` with
`datetime.now(tz=UTC)` ([`index.py:418-424`](../../src/lodedb/engine/index.py#L418)),
re-validates the id, checks membership in `state.document_hashes`, and rebuilds a
record dict via `_document_resource_payload`
([`core.py:4496-4504`](../../src/lodedb/engine/core.py#L4496)). So the "pure dict
lookup" the prompt hypothesizes is actually a lock acquisition + clock read +
validation + dict rebuild **per hit** — which is why the measured per-hit gap is
18–30×, not ~1×.

So `search(k=N)` issues **N** `get_document` round-trips after the scan;
`search_many` with `B` queries of `k` issues **B·N**.

### 2.2 The engine already holds the metadata — and can already inline it

`_materialize_query_results` builds the result rows from the TurboVec scan output
and, in the same loop, already reads `state.document_metadata` to apply the filter
([`core.py:3885-3897`](../../src/lodedb/engine/core.py#L3885)):

```python
# core.py:3886-3894
matches_filter = _compile_query_filter(query_filter)
document_metadata = state.document_metadata
for candidate in candidates:
    document_id = str(candidate["document_id"])
    if not matches_filter(document_id, document_metadata.get(document_id, {})):
        continue
    if QUERY_INCLUDE_METADATA in include:
        candidate["metadata"] = dict(document_metadata.get(document_id, {}))   # <-- already here
    filtered.append(candidate)
```

The metadata is already resident in `state.document_metadata` (a
`dict[str, dict[str, str]]`, [`core.py:287`](../../src/lodedb/engine/core.py#L287),
populated on every ingest at
[`core.py:1087`](../../src/lodedb/engine/core.py#L1087),
[`:1302`](../../src/lodedb/engine/core.py#L1302),
[`:1403`](../../src/lodedb/engine/core.py#L1403)). When `include` carries
`"metadata"`, the row already gets a redacted metadata dict — the **same** dict
`get_document` would return. `QUERY_INCLUDE_METADATA == "metadata"` and is the
only include value `_validate_query_includes` accepts
([`core.py:4391-4404`](../../src/lodedb/engine/core.py#L4391)).

The gap is purely on the SDK side: **no `LodeDB` search verb ever sets
`include`** (confirmed — there is no `include=` argument in any `db.py` query
call), so `include=()`, the `if` is skipped, and the SDK falls back to the
per-hit `get_document`.

### 2.3 The payload-free contract already permits metadata

`_search_result_from_payload` forbids exactly `{"text", "chunk_text",
"document_text", "embedding", "raw_payload"}`
([`index.py:594-600`](../../src/lodedb/engine/index.py#L594)) — `metadata` is
**not** forbidden. The metadata stored in `state.document_metadata` is the
engine's validated string→string map (`_validate_metadata`), never text or
vectors, so inlining it stays inside the redaction guarantee.

> Note: `EngineSearchResult` (the *typed* `LodeIndex.search`/`search_batch` path,
> [`index.py:21-27`](../../src/lodedb/engine/index.py#L21)) drops metadata and is
> **not** on the SDK hot path — `LodeDB.search` uses the dict-returning
> `query`/`query_batch` instead ([`db.py:411`](../../src/lodedb/local/db.py#L411)).
> Option 2 therefore needs no change to `EngineSearchResult`. (It is used by
> internal callers/tests; surfacing metadata there is optional polish.)

### 2.4 Measured overhead (the deliverable numbers)

Harness: `uv run --no-sync python`, 5 000 docs (3 metadata keys each), warm
layout, 60 iters/k (25 for batches). `get_document` calls counted by wrapping
`db._index` with a pass-through counter (zero behavior change).

**Current `search(k)` — latency and call count:**

| k | results | get_document/query | p50 | mean |
|---|---------|--------------------|-----|------|
| 10 | 10 | **10.0** | 0.206 ms | 0.227 ms |
| 50 | 50 | **50.0** | 0.401 ms | 0.463 ms |
| 100 | 100 | **100.0** | 0.742 ms | 0.821 ms |
| 500 | 500 | **500.0** | 4.226 ms | 4.354 ms |
| 1000 | 1000 | **1000.0** | 8.483 ms | 8.672 ms |

Exactly k calls per query — the N+1, measured.

**`search_many` — the batch multiplication:**

| batch | k | total hits | get_document/call | p50 |
|-------|---|------------|-------------------|-----|
| 1 | 100 | 100 | **100.0** | 0.634 ms |
| 8 | 100 | 800 | **800.0** | 5.839 ms |
| 32 | 100 | 3200 | **3200.0** | 21.70 ms |
| 64 | 100 | 6400 | **6400.0** | 41.21 ms |

Calls scale exactly as **batch · k** (O(N·batch), per the success criteria).

**Isolated metadata step (scan held constant): per-hit `get_document` loop vs a
single batched in-memory read of `state.document_metadata`:**

| k | per-hit loop p50 | batched in-mem p50 | step speedup |
|---|------------------|---------------------|--------------|
| 10 | 0.0269 ms | 0.0010 ms | **26.9×** |
| 100 | 0.2875 ms | 0.0112 ms | **25.7×** |
| 500 | 1.8870 ms | 0.0631 ms | **29.9×** |
| 1000 | 3.8954 ms | 0.2130 ms | **18.3×** |

The metadata step is the dominant non-scan cost at high k: at k=1000 the per-hit
loop (3.90 ms) is ~45% of the whole `search` (8.67 ms).

So the overhead is *not* "small per call" — the `@_synchronized` lock + context
construction inflate each lookup well past a dict access, and at high k / large
batch it is the single biggest avoidable cost in hydration.

---

## 3. Prior art (cited)

Returning metadata/payload inline with ANN results is the norm; the design axis
is *opt-in and projected* vs *always-on*, and *in-row* vs *second lookup*.

- **Qdrant** returns id+score only by default; payload is opt-in via
  `with_payload` (and vectors via `with_vectors`). It is explicitly a cost knob:
  one report measured ~8 s with `with_payload=true` vs ~1.2 s without on 10 M
  points, because Qdrant payload can live off-heap/on-disk. The lesson is the
  *shape* (opt-in, projectable), not the cost — Qdrant pays I/O that LodeDB does
  not, since LodeDB metadata is already in-process. ([Qdrant: Search](https://qdrant.tech/documentation/search/search/),
  [Qdrant #50 "Allow to include payload … into search result"](https://github.com/qdrant/qdrant/issues/50))
- **pgvector** keeps embeddings and metadata in the same row/transaction, so a
  single `SELECT` returns `(id, content, metadata, similarity)` with **no second
  lookup** — the canonical argument for inlining. ([Encore: pgvector guide](https://encore.dev/blog/you-probably-dont-need-a-vector-database))
- **LanceDB** returns the selected metadata columns inline with each hit (plus
  `_distance`) in one query, and recommends projecting only the columns you need
  for best performance. ([LanceDB: Query](https://lancedb.github.io/lancedb/js/classes/Query/),
  [LanceDB: filtering](https://docs.lancedb.com/search/filtering))

Convergent takeaway: **inline metadata in the result row, in one pass, and keep
it the only redacted projection** — exactly Option 2. LodeDB's twist is that its
metadata is already in RAM during materialization, so inlining is nearly free (no
extra I/O, unlike Qdrant), and there is no reason to prefer a second lookup.

---

## 4. Recommended design (Option 2: inline in the engine row)

### 4.1 Engine — already done; just exercise it

No engine change is strictly required. `_materialize_query_results` already
inlines metadata when `include` contains `"metadata"`
([`core.py:3892-3893`](../../src/lodedb/engine/core.py#L3892)). The metadata dict
it copies is the same redacted map `_document_resource_payload` returns. The cost
is one extra `dict(...)` copy per *kept* row — work the per-hit path does anyway,
minus the lock/context/validation overhead and minus the second pass.

### 4.2 SDK — set the include, read the row

Two minimal edits in `src/lodedb/local/db.py`:

1. **Request metadata inline** on each engine query that backs a search verb. The
   `include` field is already plumbed through `EngineQuery`/`query`/`query_batch`
   ([`index.py:169-211`](../../src/lodedb/engine/index.py#L169)). For the
   text-batch path it is already accepted per item
   ([`index.py:556`](../../src/lodedb/engine/index.py#L556)). Two small surface
   gaps to close so all four verbs are symmetric:
   - `LodeIndex.query_vector` builds its `EngineQuery` with no `include`
     ([`index.py:240-247`](../../src/lodedb/engine/index.py#L240)) — add an
     `include` parameter (default `()`).
   - `_query_vector_from_item` likewise omits `include`
     ([`index.py:537-542`](../../src/lodedb/engine/index.py#L537)) — read an
     optional `"include"` key.
   Then have the four `LodeDB` search verbs pass `include=("metadata",)` (the
   single accepted value, [`core.py:4401`](../../src/lodedb/engine/core.py#L4401)).

2. **Hydrate from the row** — change `_hits_from_result_rows`
   ([`db.py:742-759`](../../src/lodedb/local/db.py#L742)) to read the inlined
   metadata instead of calling `_metadata_for_document`:

   ```python
   def _hits_from_result_rows(self, rows):
       ...
       for row in rows:
           ...
           raw_meta = row.get("metadata", {})
           metadata = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
           hits.append(LodeSearchHit(score=float(row["score"]),
                                     id=document_id, metadata=metadata))
       return hits
   ```

   `_metadata_for_document` ([`db.py:732-740`](../../src/lodedb/local/db.py#L732))
   becomes dead and can be removed (or kept as a by-id helper — it duplicates the
   public `get_document`, so removal is cleaner).

### 4.3 Why this is the clean option

- **Zero new round-trips, zero new public API.** No batched-`get_document`
  method to design, name, document, and test. The contract surface is unchanged.
- **One pass.** The engine emits metadata in the same loop that already walks
  `state.document_metadata` for filtering; the SDK never re-walks the ids.
- **Dedup is automatic and correct.** The materializer keys on each result row's
  `document_id`; if a multi-chunk document ever produced two rows, each row
  carries its own (identical) metadata — no special-casing. (Empirically, the
  current single-chunk corpora produce one row per unique id: 200 rows → 200
  unique ids, so there is nothing to dedup today, but the design is robust if
  that changes.)
- **Same redaction guarantee.** Inlined value is the validated string→string
  metadata map, never text/vectors; the forbidden-key check still passes
  (verified — see §6).

### 4.4 Public shape unchanged

`LodeSearchHit(score, id, metadata)` and its tuple unpacking
([`db.py:55-90`](../../src/lodedb/local/db.py#L55)) are untouched. Callers,
including `KnowledgeGraph.semantic_nodes`/`search_subgraph`
([`knowledge_graph.py:257-350`](../../src/lodedb/graph/knowledge_graph.py#L257)),
see byte-identical hits — they just arrive without the N+1.

---

## 5. Tradeoffs / risks / invariant compliance

| Concern | Assessment |
|---------|------------|
| **Payload-free invariant** | Held. Only `state.document_metadata` (validated string→string) is inlined; `_search_result_from_payload`'s forbidden set ([`index.py:594`](../../src/lodedb/engine/index.py#L594)) excludes `metadata`. Verified: row keys are exactly `{chunk_id, document_id, metadata, score}`, no forbidden key. |
| **Public hit shape** | Unchanged — `(score, id, metadata)` / `LodeSearchHit` identical. |
| **Result equivalence** | Verified identical to the N+1 path (k=100): same ids, same metadata, same order. |
| **Commit-path O(changed)** | Untouched — this is a read/materialize change only; no ingest, sync, or persistence code is touched. |
| **Extra per-row copy** | The engine now `dict(...)`-copies metadata for every kept row even when a caller ignores `.metadata`. This is cheaper than the per-hit `get_document` it replaces (measured net −47–57%), and matches the prior contract that always populated `.metadata`. If a future caller wanted ids-only, add an opt-out (mirror Qdrant `with_payload=False`); not needed now. |
| **Slight behavior nuance** | `_metadata_for_document` swallows races returning `{}` ([`db.py:737-738`](../../src/lodedb/local/db.py#L737)); the inline path reads the same snapshot the scan ran against, so it is *more* consistent (metadata and scoring come from one frame), not less. |
| **`get` semantics** | Unaffected — raw text still flows only through `get`/`get_text`/`get_texts`; this changes only redacted-metadata delivery. |
| **Read-only handles** | Unaffected — pure read path; no lock/persist implications. |

**Net:** strictly fewer engine calls, less lock contention (the shared-engine
`serve` path no longer re-enters `_op_lock` N times per query), identical
results, same redaction. No CUDA/MPS path is touched.

---

## 6. Prototype / validation plan

### 6.1 Measured PoC (already run — evidence in §2.4)

The numbers in §2.4 come from a working PoC under `/tmp` (scratch, not committed)
that:
- counts `get_document` calls by wrapping `db._index` with a pass-through
  counter → proves k and B·k call counts;
- times the real `search(k)` per k → the "current" column;
- times a **simulated Option 2** that calls `real_index.query(..., include=("metadata",))`
  and builds hits straight from `row["metadata"]` with **zero** `get_document`
  → −47% (k=1000) to −57% (k=100) end-to-end;
- asserts parity + payload safety:
  `{"metadata_identical": True, "forbidden_keys_leaked": [], "row_keys":
  ["chunk_id", "document_id", "metadata", "score"]}`.

This already validates correctness and the win on the exact `include` hook the
real change would flip — the production change is just moving that hook from the
PoC into `db.py`.

### 6.2 Implementation steps (precise)

1. `index.py`: add `include: Iterable[str] = ()` to `LodeIndex.query_vector`
   ([:228](../../src/lodedb/engine/index.py#L228)) and thread it into its
   `EngineQuery`; read `"include"` in `_query_vector_from_item`
   ([:514](../../src/lodedb/engine/index.py#L514)).
2. `db.py`: pass `include=("metadata",)` in `search`/`search_many`/
   `search_by_vector`/`search_many_by_vector`
   ([:411](../../src/lodedb/local/db.py#L411), [:443](../../src/lodedb/local/db.py#L443),
   [:483](../../src/lodedb/local/db.py#L483), [:509](../../src/lodedb/local/db.py#L509)).
3. `db.py`: rewrite `_hits_from_result_rows` to read `row["metadata"]`; delete the
   now-dead `_metadata_for_document`.
4. (Optional polish) carry `metadata` on `EngineSearchResult` + relax
   `_search_result_from_payload` to keep it, so the typed `LodeIndex.search`
   path matches.

### 6.3 Regression / parity gates

- Existing SDK suite (search/search_many/by_vector + KnowledgeGraph) must pass
  unchanged — hits are byte-identical, so no test should need editing beyond any
  that *assert on call counts*.
- `tests/test_import_boundary.py` unaffected (no new imports).
- Add a parity assertion: for a seeded corpus, `search(k=K)` hits equal the
  metadata-inlined hits for K ∈ {1, 10, 100}, and the result rows expose no
  forbidden key (reuse the §6.1 check).

### 6.4 Benchmark sweep (the success-criteria deliverable)

Extend the `vector_in` sub-benchmark in
[`benchmarks/graph_memory/graph_memory_bench.py`](../../benchmarks/graph_memory/graph_memory_bench.py)
(`run_vector_in_bench`, [:114](../../benchmarks/graph_memory/graph_memory_bench.py#L114))
to **sweep k** instead of a single `top_k`:

- Loop `k ∈ {10, 50, 100, 500, 1000}` over the existing byte-identical text-in /
  vector-in indexes; for each k record `search`/`search_by_vector` latency
  (`_latency_summary`) into a `search_latency_by_k` map, keeping the existing
  `topk_overlap` parity check at one k.
- Optionally add a small `search_many` sweep over `batch ∈ {1, 8, 32, 64}` at a
  fixed k to surface the B·k effect.
- Run before/after the §6.2 change on **Modal** (`modal_bench.py::main_a10`,
  full spec `top_k`/sweep) and locally (`--dataset synthetic`).

**Expected delta** (from §2.4, scan-bound corpus): negligible at k=10
(`get_document` overhead ≈ scan), then growing to roughly **−45% to −57%** at
k≥100, and proportional removal of B·k `get_document` calls in the `search_many`
sweep (e.g. 6400 → 0 at batch=64, k=100). On Modal with the real MiniLM backend
and GovReport, single-query absolute latency is embedding-dominated, so report
the *post-embedding* search latency (the bundle already separates
`query_embedding_latency_ms` / `query_search_latency_ms` in the engine response,
[`core.py:1644-1645`](../../src/lodedb/engine/core.py#L1644)) to isolate the
hydration win; the batched paths show the largest absolute gains.

---

## 7. Open questions

1. **Always-inline vs opt-out?** Should the engine always inline (current
   recommendation, matches the pre-existing `.metadata` contract) or expose a
   `with_payload=False`-style opt-out for ids-only callers (Qdrant/LanceDB do)?
   No current caller wants ids-only, so defer — but note `KnowledgeGraph`
   sometimes only needs `node_id`/`edge_id` from metadata
   ([`knowledge_graph.py:283`](../../src/lodedb/graph/knowledge_graph.py#L283)),
   which the inlined metadata already supplies for free.
2. **Typed-path metadata (§6.2 step 4).** Worth surfacing metadata on
   `EngineSearchResult` for the typed `LodeIndex.search` path, or leave that path
   metadata-free since the SDK doesn't use it? Low stakes either way.
3. **Multi-chunk documents.** No current path produces multiple result rows for
   one document id (vector-in is single-chunk; text docs are capped to one chunk
   in the benchmark). If chunk-level results ever surface, confirm hydration
   stays per-row (it does by construction) and decide whether the SDK should
   dedup to document granularity — orthogonal to this N+1 fix.
4. **Interaction with batched reads (`research-prompts/06`).** A batched
   `get_documents([...])` engine method (Option 1) would still be useful for the
   *non-search* by-id reads in the graph layer (resolving edge endpoints in
   `k_hop`/`neighbors`). Option 2 fixes the search N+1; Option 1's batched read
   remains the right tool for those enumeration-side reads — they are
   complementary, not competing.
