# Findings: a bring-your-own-vectors index mode (arbitrary dim, no embedder)

> Research prompt: [`docs/research-prompts/03-arbitrary-dim-vector-only-index.md`](../research-prompts/03-arbitrary-dim-vector-only-index.md).
> Method: claims grounded in code (`file:line`); TurboVec behavior read from
> vendored sources + committed benchmark JSON; prior art from web sources (cited);
> feasibility proven by measurement under `/tmp` against the real compiled
> `lodedb._turbovec` extension (`uv run --no-sync`, no rebuild, no model download).

## 1. TL;DR

- **The capability already exists in the engine; only the local SDK pins it shut.**
  A `LodeEngine` constructed with `route_policy=None` **and** `embedding_backend=None`
  accepts a caller-chosen `native_dim`, ingests vector-in documents at that dim,
  queries by vector, persists, and **reopens with the dim re-enforced** — with **zero
  core changes**. Proven at dim 1536: `create_index` → 201, `upsert_vectors` of 50×
  1536-d → 200, top-1 self-match score `1.001`, identical results after a fresh
  reopen, and a 768-d query into the reopened index correctly rejected with
  `400 query vector dimension 768 does not match index native_dim 1536`
  (transcript in §2.4).
- **What blocks it today is entirely in `src/lodedb/local/`**: `LodeDB.__init__` always
  resolves a preset (`resolve_preset(model)`, `db.py:171`), always builds a
  `SentenceTransformerEmbeddingBackend` from it (`db.py:184-189` →
  `backends.py:133`), and funnels every vector through `self.preset.native_dim`
  (`db.py:331, 368, 482, 511` → `_prepare_vector`, `db.py:850-851`). The preset's
  `route_policy` then pins `model`/`native_dim` in the engine
  (`core.py:715-724`, `core.py:699-705`).
- **Recommended construction API:** `LodeDB(path, embedding="none", vector_dim=1536)`
  (sugar: `LodeDB.open_vector_store(path, vector_dim=1536)`). Internally:
  `preset=None`, `embedding_backend=None`, pass a **vector-only route profile** (or
  `route_policy=None`) so `_index_shape_from_profile` takes the caller-shape branch
  (`core.py:2109-2112`); record a stable redacted identity (`model="external"`,
  `task="vector-only"`, `route_profile="custom"`) which the snapshot **already
  persists and re-enforces** (`core.py:4822-4825`, restored at `core.py:4903-4906`;
  on-disk JSON shown in §2.4). Make `add`/`add_many`/`search`/`search_many` raise a
  clear `VectorOnlyIndexError` (the engine otherwise *silently* embeds text with the
  `HashEmbeddingBackend` fallback — §2.3).
- **Lean deps:** make the `sentence-transformers` import lazy. It is a hard runtime
  dependency today (`pyproject.toml:43`, comment at `:36` notes it "pulls
  torch/transformers"); a vector-only install never needs it.
- **TurboVec holds recall at 1536/3072 and is sub-millisecond.** From the vendored
  benchmark JSON: 4-bit recall@1 = **0.974** at both 1536 and 3072 (≈1.0 by k≥4);
  2-bit recall@1 = 0.872–0.929; memory is **8.0×** (4-bit) / ~16× (2-bit) vs fp32 and
  scales linearly with dim (1536-d 4-bit = **73.6 MB / 100k**, 3072-d = 146.9 MB);
  ARM scan latency 1536-d 4-bit = **0.185 ms/query** (multi-thread) / 1.992 ms
  (single-thread), 3072-d = 0.375 / 3.968 ms — faster than the FAISS `IndexPQ`
  reference in every cell (§4.5).
- **One real correctness risk to design around:** TurboVec's TQ+ per-coordinate
  calibration needs **≥1000 vectors** in the *first* batch (`TQPLUS_MIN_SAMPLES = 1000`,
  `encode.rs:45`), and the calibration is **frozen on the first cold build** and reused
  for all later `add_with_ids` (engine sync: cold build only when `previous is None`,
  `core.py:2815-2832`; incremental adds at `core.py:2898`). A trickle-ingested graph
  memory (facts added one at a time) freezes **identity calibration** on the first add
  and **permanently misses the 4-bit recall lift** even after growing to 1M rows
  (`lib.rs:745-754`, `encode.rs:39-45`). Mitigation in §4.6/§5.2.

---

## 2. Current behavior + the exact validators/dim guards that block arbitrary dims

### 2.1 The dimension is hard-wired to a preset at the SDK boundary

`LodeDB.__init__` always picks a preset and always builds a real embedding backend:

```text
src/lodedb/local/db.py
171   self.preset: LocalModelPreset = resolve_preset(model)   # only "minilm"/"bge"
184-189   backend, self.embedding_resolution = build_local_embedding_backend(self.preset, ...)
190   self._embedding_backend = backend
219   route_policy=self.preset.route_policy,                   # pins model + native_dim
```

`resolve_preset` (`presets.py:78-85`) raises for any name other than `minilm`/`bge`,
and each preset's `native_dim`/`model` come straight from a route policy
(`presets.py:46-49`, `route_profiles.py:31,44`: 768 and 384). Every vector-in entry
point validates against `self.preset.native_dim`:

```text
src/lodedb/local/db.py
331   prepared = _prepare_vector(vector, self.preset.native_dim, normalize=normalize)   # add_vectors
368   _prepare_vector(vector, self.preset.native_dim, ...)                              # add_vectors_many
482   prepared = _prepare_vector(vector, self.preset.native_dim, ...)                   # search_by_vector
511   _prepare_vector(vector, self.preset.native_dim, ...)                              # search_many_by_vector
850-851   if len(values) != dim: raise ValueError(f"vector must have dimension {dim}, got {len(values)}")
```

So at the SDK there is **no way to express a dim other than 384 or 768** — the only
two values `self.preset.native_dim` can take.

### 2.2 The engine-side validator chain (what *would* fire even if the SDK let a dim through)

`create_index` runs three relevant guards, in order:

```text
src/lodedb/engine/core.py
677-682   model, provider, task, native_dim = self._index_shape_from_profile(...)
699-705   if self.embedding_backend is not None and native_dim != self.embedding_backend.native_dim:
              error 400 "native_dim must match configured embedding backend"
706-713   if self.embedding_backend.required_model_name and model != ...: 
              error 400 "model must match configured embedding backend"
715-724   route_error = self.route_policy.validate_index_request(model, provider, task, native_dim, route)
```

- **`_index_shape_from_profile` (`core.py:2099-2120`)** is the keystone. When
  `route_policy is not None`, passing *any* of `model/provider/task/native_dim` raises
  `"index shape is derived from route profile"` (`:2113-2114`) and the shape is taken
  verbatim from the policy (`:2115-2120`). **When `route_policy is None`, it accepts
  the caller's shape** (`:2109-2112`). This single branch is why `route_policy=None`
  unlocks arbitrary dim.
- **`EngineRoutePolicy.validate_index_request` (`core.py:194-218`)** rejects
  `native_dim != self.native_dim` (`:211-212`) and `model != self.model` (`:205-206`).
  This is the validator the prompt names; it only runs when a policy is attached.
- **The embedding-backend match (`core.py:699-705`)** is the one that actually fires
  in the SDK today (see S2 below), because the SDK always attaches a backend.

The runtime ingest/query paths re-check the dim against the *persisted* `state.native_dim`,
so they enforce consistency independent of how the index was created:

```text
core.py:1352-1360   upsert/_ingest_vectors: array.shape[0] != native_dim -> 400 "vector dimension ... does not match index native_dim"
core.py:1575-1582   query: len(query.embedding) != state.native_dim -> 400
core.py:1676-1683   query_batch: same guard
core.py:4046-4049   _validate_direct_turbovec_snapshot: index.dim != state.native_dim -> ValueError on load
```

### 2.3 The silent-text-embedding trap (why the SDK must add an explicit guard)

`_embedding_backend_for_state` **falls back to a fixture backend** when none is configured:

```text
core.py:2130-2137
  if self.embedding_backend is not None:
      if self.embedding_backend.native_dim != state.native_dim: raise ...
      return self.embedding_backend
  return HashEmbeddingBackend(native_dim=state.native_dim)   # <-- silent fallback
```

Measured: with `route_policy=None, embedding_backend=None`, a **text-in**
`upsert_documents("hello world")` returns **`200, embedded 1`** — it embedded the text
with `HashEmbeddingBackend`, producing a meaningless hash vector that would be mixed
with the caller's real BYO vectors and scored against them. Therefore the
"this index is vector-only" rejection **must live in the SDK** (`LodeDB.add`/`search`),
not be left to the engine. (The engine fallback is correct and useful for fixtures;
we just must not reach it from a vector-only handle.)

### 2.4 Observed tracebacks (real, under `/tmp`)

**S1 — what already works at 384** (`LodeDB(model="minilm",
_embedding_backend=HashEmbeddingBackend(native_dim=384))`): `add_vectors` +
`search_by_vector` succeed; top-1 self-match score `1.001`; text-in `add()` also works.
This is the existing internal hook the prompt points to. It works *only* because the
hash backend's dim (384) equals the minilm preset's dim.

**S2 — the blocker** (same but `HashEmbeddingBackend(native_dim=1536)`, preset still
minilm):

```text
File ".../src/lodedb/local/db.py", line 228, in __init__
    self._ensure_index()                       # -> self._index.create(...)
File ".../src/lodedb/engine/index.py", line 431, in _unwrap
    raise EngineError(...)
lodedb.engine.index.EngineError: native_dim must match configured embedding backend
```

Raised at **`core.py:699-705`**: the preset's route policy says 384, the injected
backend says 1536, so `create_index` 400s.

**S3 — SDK boundary** (a 1536-d vector into a 384 index):

```text
File ".../src/lodedb/local/db.py", line 331, in add_vectors
    prepared = _prepare_vector(vector, self.preset.native_dim, normalize=normalize)
File ".../src/lodedb/local/db.py", line 851, in _prepare_vector
    raise ValueError(f"vector must have dimension {dim}, got {len(values)}")
ValueError: vector must have dimension 384, got 1536
```

**Feasibility proof — the engine already does the right thing at 1536** (constructed
directly with `route_policy=None, embedding_backend=None`):

```text
create_index status: 201 | native_dim: 1536 (internal)
upsert_vectors status: 200 | embedded: 50
query status: 200 | top: [('n7', 1.001), ('n33', 0.07), ('n23', 0.046)]
REOPEN query status: 200 | top: [('n7', 1.001), ('n33', 0.07), ('n23', 0.046)]   # fresh LodeEngine, restored from snapshot
768-d query into the 1536 index -> 400 "query vector dimension 768 does not match index native_dim 1536"
```

The persisted snapshot on disk (generation-addressed `<key>.gen/g*.json`) records
exactly the redacted identity needed for reopen enforcement:

```text
g1.json/g2.json -> native_dim=1536  model="external"  task="vector-only"
                    storage_profile="turbovec_direct"  turbovec_bit_width=4
```

(`_state_header_payload`, `core.py:4811-4836`; restored by `_state_from_payload`,
`core.py:4903-4906`. With `route_policy=None`, `route_profile` defaults to the literal
`"custom"`, `core.py:4803/4860`.) **No snapshot schema change is required** — schema is
still `version: 1` and the dim/model/task fields already exist and round-trip.

---

## 3. Prior art (cited)

### 3.1 How mainstream stores expose a pure bring-your-own-vectors mode

- **Raw-vector-in is the established default in 3 of 4 stores.** Qdrant's core `upsert`
  takes precomputed vectors; collections are created with
  `VectorParams(size=, distance=)` and the dimensionality is fixed per collection and
  enforced on insert, with the error `Wrong input: Vector dimension error: expected
  dim: <N>, got <M>`. (https://qdrant.tech/documentation/concepts/collections/;
  observed error https://github.com/microsoft/semantic-kernel/issues/13587)
- **pgvector**: a `vector(N)` column; `N` is enforced on every insert by
  `CheckExpectedDim` → `expected %d dimensions, not %d`. No built-in embedder. Type
  maxima: `vector` ≤ 2000 (indexable), `halfvec` ≤ 4000.
  (https://github.com/pgvector/pgvector/blob/master/README.md;
  https://github.com/pgvector/pgvector/blob/master/src/vector.c)
- **LanceDB**: the bring-your-own path is a plain Arrow schema with
  `pa.list_(pa.float32(), N)` (a `FixedSizeList`), so `N` is fixed by schema; raw
  vectors insert directly. (https://docs.lancedb.com/tables/create)
- **Chroma is the outlier** — text-embedding is the *default* (passing `documents=`
  auto-embeds with `all-MiniLM-L6-v2`, dim 384); to bring your own vectors you
  explicitly pass `embeddings=`. Dimensionality is *deferred*: not declared at
  creation, then locked on first insert; a mismatch raises `InvalidDimensionException`
  (`Embedding dimension X does not match collection dimensionality Y`).
  (https://docs.trychroma.com/docs/embeddings/embedding-functions;
  https://cookbook.chromadb.dev/faq/)
- **Model-identity tracking is essentially absent across the field** — Qdrant,
  pgvector, and Chroma record only the numeric dim, so two *different* same-dim models
  mix silently. LanceDB is the lone (opt-in) exception: its embedding-function registry
  serializes model name + params into the table's Arrow schema metadata and reapplies
  it on reopen. (https://deepwiki.com/lancedb/lancedb/8.3-embedding-functions) —
  **Recording a redacted model-identity string alongside the LodeDB index would be a
  genuine, citable differentiator, not a reinvention; and LodeDB already persists
  `model`/`task` in the snapshot (§2.4), so it is nearly free.**

**Implication for LodeDB:** the recommended design (fix dim at *creation*, enforce on
insert with a clear error, persist a redacted model identity) matches the
Qdrant/pgvector pattern (the stronger one) rather than Chroma's defer-and-lock.

### 3.2 What graph-memory frameworks expect from a vector backend

**All three own their embedder, embed text upstream, and hand precomputed float
vectors to the store. None expect the store to embed.** A vector-in / search-by-vector
API at an arbitrary dim is exactly the interface they want.

- **Graphiti (getzep/graphiti)**: `EmbedderClient` abstraction (`async create(...) ->
  list[float]`, `create_batch`). Default model `text-embedding-3-small`, but the
  default **dim is 1024**, generated as the full 1536 then **hard-sliced**
  `embedding[: self.config.embedding_dim]` (flagged upstream as a quality bug). Vector
  search runs as cosine **inside the graph DB via Cypher** (`node_similarity_search`,
  `edge_similarity_search` in `search_utils.py`); the graph backends are **Neo4j,
  FalkorDB, Amazon Neptune, Kuzu**. There is no standalone vector-store driver, so a
  LodeDB integration would target the `*_similarity_search` seam or a thin store wrapper
  that Graphiti's search calls.
  (https://raw.githubusercontent.com/getzep/graphiti/main/graphiti_core/embedder/openai.py;
  https://github.com/getzep/graphiti/issues/1087;
  https://raw.githubusercontent.com/getzep/graphiti/main/graphiti_core/search/search.py;
  https://github.com/getzep/graphiti)
- **cognee (topoteretes/cognee)** — *the cleanest single drop-in target*: a documented
  abstract `VectorDBInterface` with named methods:
  `has_collection`, `create_collection(collection_name, payload_schema=None)`,
  `create_data_points(collection_name, data_points)`,
  `search(collection_name, query_text=None, query_vector=None, limit, with_vector, ...)`,
  `batch_search`, `retrieve`, `delete_data_points`, `prune`, and `embed_data(data) ->
  List[List[float]]`. The adapter holds an **injected** `embedding_engine` (default
  `text-embedding-3-large`, **dim 3072**) and embeds *before* writing; the underlying DB
  only ever sees vectors. Core ships **LanceDB**; community adapters add Qdrant, Redis,
  OpenSearch, Azure AI Search.
  (https://raw.githubusercontent.com/topoteretes/cognee/main/cognee/infrastructure/databases/vector/vector_db_interface.py;
  https://raw.githubusercontent.com/topoteretes/cognee/main/cognee/infrastructure/databases/vector/lancedb/LanceDBAdapter.py;
  https://docs.cognee.ai/contributing/adding-providers/adding-new-vector-database)
- **Letta (formerly MemGPT)**: archival memory is a `Passage` store via
  `PassageManager`/`ArchiveManager`; `insert_passage` calls
  `embedding_client.request_embeddings(...)` then stores. Default `embedding_dim` 1536
  (`letta-free`/OpenAI). For pgvector, embeddings are **zero-padded to
  `MAX_EMBEDDING_DIM = 2048`**. Backends: pgvector (default self-hosted), Turbopuffer
  (Cloud), Pinecone. Tighter coupling than cognee (no single pluggable vector ABC).
  (https://raw.githubusercontent.com/letta-ai/letta/main/letta/schemas/embedding_config.py;
  https://raw.githubusercontent.com/letta-ai/letta/main/letta/services/passage_manager.py;
  https://deepwiki.com/letta-ai/letta/3-memory-system)

**Default dims diverge — a BYO store must be dim-agnostic:** Graphiti **1024** (sliced
from 1536), cognee **3072**, Letta **1536** (padded to 2048 on pgvector). This is the
core argument for arbitrary `vector_dim` rather than a third preset.

### 3.3 Quantization of high-dim embeddings — recall (corroborates §4.5)

- **Binary quantization** of OpenAI embeddings is unusually strong and *improves with
  dimension*: `text-embedding-3-large` @ 3072-d → **0.9966 recall** (3× oversample +
  rescore), vs 0.9826 truncated to 1536-d; `text-embedding-3-small` @ 1536-d → 0.9847.
  Qdrant explicitly warns BQ gives "poorer results for small embeddings i.e. less than
  1024 dimensions." (https://qdrant.tech/articles/binary-quantization-openai/;
  https://qdrant.tech/articles/binary-quantization/)
- **int8 scalar quantization** retains ~99% with rescoring across models
  (https://huggingface.co/blog/embedding-quantization); Qdrant reports near-zero recall
  loss at 4× compression (https://qdrant.tech/articles/scalar-quantization/).
- **Matryoshka (MRL) truncation** is orthogonal and composable: OpenAI states a
  `text-embedding-3-large` vector "can be shortened to a size of 256 while still
  outperforming an unshortened `text-embedding-ada-002` embedding with a size of 1536."
  (https://developers.openai.com/api/docs/guides/embeddings; MRL: arXiv:2205.13147)

The takeaway: the high-dim regime BYO callers bring (1536/3072) is *exactly* where
aggressive quantization keeps recall — which is what TurboVec's own benchmarks show.

---

## 4. Recommended design

A vector-only mode that **extends, does not fork** the existing paths. The engine needs
**no changes**; the work is in `src/lodedb/local/` plus a dependency-laziness change.

### 4.1 Construction API (`src/lodedb/local/db.py`)

```python
db = LodeDB(path, embedding="none", vector_dim=1536)        # explicit
db = LodeDB.open_vector_store(path, vector_dim=1536, bit_width=4)   # sugar
```

New keyword args on `__init__`: `embedding: str = <preset name>` (accept the sentinel
`"none"`) and `vector_dim: int | None = None`. Validation: `vector_dim` is **required**
when `embedding="none"` and **forbidden** otherwise; `1 <= vector_dim <= 65536`
(TurboVec `MAX_DIM`, `lib.rs:72`). Add a `self.vector_only: bool` flag.

When `embedding="none"`:
- **Do not** call `resolve_preset` / `build_local_embedding_backend`. Set
  `self.preset = None`, `self._embedding_backend = None`.
- Build the engine with **`route_policy=` a dedicated vector-only profile** (preferred
  over `route_policy=None`; see §4.2) and `embedding_backend=None`. The
  `EngineSecurityConfig.route_profile` becomes `"vector-only"` (it is redacted display
  metadata only — `core.py:4519` via `_client_route_profile`).
- Drive `create`/ingest at `vector_dim`. The dim plumbs through to the engine via the
  vector-only route profile's `native_dim` (no `native_dim=` kwarg crosses
  `_index_shape_from_profile`, satisfying its policy branch).

`_prepare_vector` calls switch from `self.preset.native_dim` to a
`self._vector_dim` property (`= self.preset.native_dim` in preset mode, `= vector_dim`
in vector-only mode), so `add_vectors`/`search_by_vector` work unchanged at the chosen
dim.

### 4.2 Route-profile vs validator changes — prefer a dedicated profile

Two options satisfy `_index_shape_from_profile`:

1. **`route_policy=None`** — minimal, *proven working* in §2.4. But it makes the engine
   permissive (caller supplies arbitrary `model/provider/task/native_dim`) and loses the
   `require_turbovec_available` guard at `core.py:725-732` (which is gated on
   `route_policy is not None and index_backend == DIRECT_TURBOVEC_STORAGE_PROFILE`).
2. **A parameterized vector-only `EngineRoutePolicy`** (recommended). Add a constructor
   like `vector_only_route_policy(native_dim, bit_width)` that returns:

   ```python
   EngineRoutePolicy(
       profile="vector-only", label="External vectors (bring-your-own)",
       client_note="caller-supplied embeddings; no internal model",
       model="external", provider="external", task="vector-only",
       native_dim=native_dim, method_template=f"direct_turbovec_full{native_dim}_bw{bit_width}",
       index_backend="turbovec_direct", turbovec_bit_width=bit_width,
   )
   ```

   This keeps the route-policy contract intact: `_index_shape_from_profile` derives the
   shape from the policy (so the SDK passes **no** shape kwargs), `validate_index_request`
   passes (model/dim equal the policy's own values), and the TurboVec availability guard
   still runs. The route registry's `select_route` is permissive for an unknown
   `model="external"` (it returns a default decision, not an error — confirmed: §2.4's
   `create_index` succeeded), and `validate_index_request` only consults the registry
   when `index_backend != turbovec_direct` (`core.py:213-217`), which is not our case.

   This profile is **not** registered in `CLIENT_ROUTE_POLICIES` (that dict is keyed by
   the four preset names and surfaced in the manifest); it is constructed on demand by
   the local layer, mirroring how `LocalModelPreset.route_policy` already returns a
   policy object (`presets.py:33-37`).

`model="external"` / `task="vector-only"` is the **stable redacted identity** the prompt
asks for: it is what lands in the persisted snapshot (`core.py:4822-4824`) and what a
reopen restores and re-binds, so two indexes built "vector-only" are mutually consistent
and an attempt to reopen a 384 preset index *as* vector-only (or vice versa) is caught by
the existing snapshot dim/identity round-trip.

### 4.3 Text-in methods raise on a vector-only index

Add a guard at the top of `add`, `add_many`, `search`, `search_many`:

```python
if self.vector_only:
    raise VectorOnlyIndexError(
        "this index is vector-only (embedding='none'); use add_vectors/search_by_vector"
    )
```

This is **required** because the engine would otherwise silently embed the text with the
`HashEmbeddingBackend` fallback (§2.3, measured `200, embedded 1`). `get`/`get_text`
already return `None` for vector-in docs (no text captured — `core.py:983-984`), which is
the correct behavior and needs no change. `VectorOnlyIndexError(RuntimeError)` is a new
public exception alongside `ReadOnlyError` (`db.py:93`).

### 4.4 Snapshot dim/mode recording (no schema change)

Already handled. `native_dim`, `model`, `task`, `storage_profile`, `turbovec_bit_width`
are in the schema-v1 header (`core.py:4811-4836`) and round-trip on reopen
(`core.py:4903-4913`), verified on disk in §2.4. Reopen enforcement comes for free from
the runtime guards (`core.py:1575-1582`, `:1352-1360`) and the load-time snapshot check
(`core.py:4046-4049`). **Mixing models in one index stays prevented** exactly as today:
the `model` field is fixed at create and the dim guard rejects off-dim vectors; recording
`model="external"` documents intent (callers must still only mix one external model's
vectors — same contract as the text path's "same model only" note, `db.py:323-327`).

### 4.5 Quantizer behavior at large dim (1536/3072) — quantified from vendored benchmarks

All figures from `third_party/turbovec/benchmarks/results/*.json` (TurboVec's own suite;
TQ = TurboVec, FAISS reference = `IndexPQ`):

| dim | bits | recall@1 | recall@2 | recall@4 | recall@8 | FAISS recall@1 |
|----:|-----:|---------:|---------:|---------:|---------:|---------------:|
| 1536 | 4 | **0.974** | 0.998 | 1.000 | 1.000 | 0.966 |
| 3072 | 4 | **0.974** | 0.999 | 1.000 | 1.000 | 0.972 |
| 1536 | 2 | 0.891 | 0.979 | 0.998 | 1.000 | 0.872 |
| 3072 | 2 | 0.929 | 0.988 | 0.999 | 1.000 | 0.912 |

(`recall_d{1536,3072}_{2,4}bit.json`.) **4-bit holds recall at high dim** (≈ FAISS PQ
m=dim/2, nbits=8, and slightly above it). 2-bit takes a recall@1 discount but recovers
to ≈1.0 by k≥4 — consistent with the existing 2-bit tier's "qrel recall
indistinguishable from 4-bit" note (`route_profiles.py:55-57`) and with the binary-
quantization-on-OpenAI prior art in §3.3.

**Memory** (`compression.json`, n=100k): scales linearly with dim; 4-bit ≈ 8.0×, 2-bit
≈ 16× vs fp32.

| dim | bits | fp32 MB | index MB | ratio | bytes/vector |
|----:|-----:|--------:|---------:|------:|-------------:|
| 1536 | 4 | 585.9 | **73.6** | 8.0× | ~772 |
| 1536 | 2 | 585.9 | 37.0 | 15.8× | ~388 |
| 3072 | 4 | 1171.9 | **146.9** | 8.0× | ~1540 |
| 3072 | 2 | 1171.9 | 73.6 | 15.9× | ~772 |

At 1M × 1536-d 4-bit ≈ **736 MB** of persisted codes (vs ~5.9 GB fp32).

**Scan latency** (ARM = Apple Silicon NEON, `speed_d*_4bit_arm_*.json`); TQ beats the
FAISS reference in every cell:

| dim | bits | single-thread ms/q | multi-thread ms/q | FAISS ST ms/q |
|----:|-----:|-------------------:|------------------:|--------------:|
| 1536 | 4 | 1.992 | **0.185** | 2.45 |
| 3072 | 4 | 3.968 | **0.375** | 4.925 |

Latency scales ~linearly with dim (exact-scan over packed codes), so 3072 is ~2× the
cost of 1536 — still sub-millisecond multi-threaded. This is the *uncompromised exact
scan* LodeDB advertises; no ANN graph, so recall is a function of quantization only,
exactly as the table shows.

### 4.6 Lazy embedder import (lean deps)

`sentence-transformers` is a **hard** dependency (`pyproject.toml:43`; comment `:36`:
"sentence-transformers = default embedding backend (pulls torch/transformers)"). The
class already lazy-imports the model itself (`embedding_backends.py:118-119`), but the
*module* import chain is eager: `db.py:38-41` → `backends.py:20-23` imports
`SentenceTransformerEmbeddingBackend` at module load. Two changes make a vector-only
install skip torch entirely:

1. Defer the `from lodedb.local.backends import build_local_embedding_backend` to inside
   the preset branch of `__init__` (it is only needed when `embedding != "none"`).
2. Move `sentence-transformers` (and torch, transitively) from `[project].dependencies`
   to `[project.optional-dependencies]` (e.g. `text = ["sentence-transformers>=3.0.0"]`),
   keeping `numpy` + the bundled `lodedb._turbovec` extension as the only hard deps. A
   preset (text-in) install becomes `pip install lodedb[text]`; the vector-only install
   is the lean default. `HashEmbeddingBackend` lives in `engine.embedding_backends`
   (numpy-only, `embedding_backends.py:26-47`) and stays importable for fixtures.

This is a **dependency reduction**, satisfying the invariant that a vector-only index
*shrinks* the surface rather than adding to it.

---

## 5. Tradeoffs / risks / invariant compliance

### 5.1 Invariant compliance (the payload-free + commit guarantees)

- **Payload-free artifacts:** vector-in already discards raw vectors after encoding
  (`_discard_direct_turbovec_transient_embeddings`, `core.py:4590-4613`, zeros the
  transient rows post-sync; called at `core.py:1444`). Only the quantized codes persist
  (`.tvim`/`.tvd`); the `.json`/`.jsd` snapshot carries references + metadata only
  (`_chunk_row_payload`, `core.py:4790-4796` — no vector). Arbitrary dim changes the
  width of the codes, not the boundary. **Compliant, unchanged.**
- **O(changed) commits + generation-addressed manifest:** the vector-only mode rides the
  identical `upsert_vectors` → `_ingest_vectors` → `_finalize_document_ingest` →
  `_sync_direct_turbovec_index` path (`core.py:955-997, 1319-1408, 2793+`), which is the
  same atomic-commit + incremental-delta path the text path uses. **Reused verbatim,
  not forked.**
- **Lean deps:** §4.6 *removes* sentence-transformers/torch from the default install.
  **Improves the invariant.**

### 5.2 Risks

- **(High) Calibration starvation on trickle ingest — the one real risk.**
  `TQPLUS_MIN_SAMPLES = 1000` (`encode.rs:45`): the data-dependent per-coordinate TQ+
  calibration that delivers the 4-bit recall in §4.5 is only fitted when the *first*
  encode batch has ≥1000 rows; below that, an **identity calibration** is frozen and
  reused for every later add (`encode.rs:146-149`, comment `:39-45`). The engine cold-
  builds only when `previous is None` (`core.py:2815-2832`) and otherwise calls
  `add_with_ids` against the existing calibration (`core.py:2898`). So a graph memory
  that adds facts one or a few at a time — the canonical Graphiti/cognee/Letta access
  pattern — would freeze identity calibration on the very first `add_vectors` and
  **never** get the recall lift, even at 1M rows (`lib.rs:745-754`). The §2.4 proof used
  a single 50-vector batch, which is below the threshold, so it ran on identity
  calibration (top-1 still 1.001 because the query equals a stored vector — identity
  calibration costs *ranking* recall on near-neighbors, not exact self-match).
  **Mitigations** (pick one, document the chosen one): (a) expose a `bulk_add_vectors`
  /"build" entry that batches ≥1000 vectors into the cold build so calibration fits once
  (cleanest for an adapter that loads an existing corpus); (b) add an engine "recalibrate
  on next full repack" trigger that refits once the corpus crosses 1000 rows
  (`TurboQuantIndex` exposes `calibration_fitted()` to detect the identity state —
  `lib.rs:754`); (c) document that vector-only indexes intended for high recall should be
  seeded with ≥1000 vectors before incremental use. This is also a pre-existing
  property of the text path; it is just far more likely to bite trickle-fed graph memory.
- **(Medium) Silent text embedding** if the §4.3 guard is missing — addressed by the
  guard; flagged here because it is a correctness landmine (§2.3).
- **(Low) Default bit width.** Presets pin 4-bit (`route_profiles.py:35,47`); the
  vector-only profile should default to 4-bit (recall §4.5) and expose `bit_width=2` as
  the documented storage-constrained opt-in, mirroring the existing 2-bit tier.
- **(Low) `model="external"` is non-specific.** Two genuinely different external models
  at the same dim still mix without complaint (same gap as Qdrant/pgvector, §3.1).
  Optional hardening: accept `model_identity="text-embedding-3-large"` and persist it in
  the snapshot `model` field so reopen rejects a mismatched identity. Cheap (field
  already persisted); defer unless an integration asks for it.
- **(Low) GPU/MPS resident scan at high dim.** The CUDA/MPS sessions estimate bytes from
  dim × rows (`core.py:331-343` telemetry); the memory-admission gate already exists, so
  a 3072-d 1M corpus simply may decline GPU residency and fall back to the (sub-ms) NEON
  kernel. No correctness impact; worth a benchmark note (§6).

### 5.3 What explicitly does NOT change

The engine (`core.py`), the commit manifest, the TurboVec delta store, and the snapshot
schema. The entire change is `src/lodedb/local/db.py` + a small
`src/lodedb/local/presets.py`/route-profile helper + a `pyproject.toml` dependency move.
This is why the §2.4 proof works against the *current* compiled extension with no rebuild.

---

## 6. Prototype / validation plan

### 6.1 Correctness (the success criterion) — extends today's proof

A test mirroring §2.4 but through the *public* SDK once §4 lands:

```python
db = LodeDB(tmp, embedding="none", vector_dim=1536)
ids = db.add_vectors_many([{ "vector": v_i, "id": f"n{i}", "metadata": {"type": "Person"}}
                           for i, v_i in enumerate(corpus)])   # corpus >= 1000 (calibration)
hits = db.search_by_vector(corpus[7], k=10)
assert hits[0].id == "n7"
# recall vs brute-force oracle (cosine argsort over the raw fp32 corpus):
assert topk_overlap(hits, oracle(corpus, corpus[7], k=10)) >= 0.97   # cf. §4.5 4-bit @1536
# text-in is refused:
with pytest.raises(VectorOnlyIndexError): db.add("text")
# reopen enforces dim:
db.close(); db2 = LodeDB(tmp, embedding="none", vector_dim=1536)
assert db2.search_by_vector(corpus[7], k=1)[0].id == "n7"
with pytest.raises(ValueError): db2.add_vectors([0.1]*768)   # off-dim rejected
```

Place under `tests/` next to the existing vector-in tests; run with `uv run --no-sync`.
The brute-force-oracle recall assertion is the prompt's "preserve recall vs a
brute-force oracle" criterion and should be run **with ≥1000 seed vectors** so TQ+
calibration is fitted (§5.2).

### 6.2 Thin cognee adapter sketch (cleanest target — §3.2)

cognee's `VectorDBInterface` maps almost 1:1 onto the LodeDB SDK:

```python
class LodeDBAdapter(VectorDBInterface):
    def __init__(self, path, embedding_engine):
        self.db = LodeDB(path, embedding="none", vector_dim=embedding_engine.dimensions)
        self.embedding_engine = embedding_engine          # cognee owns embedding
    async def create_collection(self, name, payload_schema=None):  # LodeDB is single-index;
        ...                                                        #   namespace via a metadata tag
    async def create_data_points(self, name, data_points):
        vectors = await self.embed_data([DataPoint.get_embeddable_data(dp) for dp in data_points])
        self.db.add_vectors_many([{ "vector": v, "id": dp.id,
                                    "metadata": {"collection": name, **flatten(dp.payload)}}
                                   for dp, v in zip(data_points, vectors)])
    async def search(self, name, query_text=None, query_vector=None, limit=10, **_):
        if query_text and query_vector is None:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]
        return self.db.search_by_vector(query_vector, k=limit,
                                        filter={"collection": name})
    async def embed_data(self, data): return await self.embedding_engine.embed_text(data)
    async def delete_data_points(self, name, ids): [self.db.remove(i) for i in ids]
```

This requires **no engine change** — it is pure SDK glue. (Graphiti is a poorer first
target: it runs similarity in Cypher and has no standalone store driver — §3.2 — so it
would need a `*_similarity_search` shim, not a clean adapter.) The adapter is a
proof-of-concept; productionizing collection semantics (cognee creates many collections;
LodeDB is one index) means either a metadata `collection` tag + filter (shown) or one
LodeDB dir per collection.

### 6.3 Ingest/query benchmark at 100k–1M external vectors (Modal A10/L40S)

Extend `benchmarks/graph_memory/` (metrics-only, per `docs/graph.md`). Synthetic random
unit vectors at dim ∈ {1536, 3072}, n ∈ {100k, 1M}, bit_width ∈ {4, 2}:

- **Ingest**: one bulk `add_vectors_many` (≥1000 to fit calibration, §5.2), measure
  cold-build encode time + persisted bytes. *Expected* from §4.5: 1536-d 4-bit ≈ 736 MB
  / 1M; 3072-d 4-bit ≈ 1.47 GB / 1M.
- **Query**: `search_many_by_vector` batches; report p50/p99 ms/query and recall@10 vs an
  fp32 brute-force oracle on a 10k held-out query set. *Expected*: recall@10 ≈ 1.0
  (4-bit), p50 sub-ms multi-thread on CPU (§4.5); on A10/L40S the resident GPU scan
  applies for 1536-d but may decline for 3072-d×1M on memory admission (§5.2) — capture
  `gpu_stage_one_status`/`gpu_fallback_reason` from the existing telemetry
  (`core.py:1631-1634`).
- **Parity**: confirm `search_by_vector` == `search_many_by_vector` top-k (the existing
  `topk_overlap == 1.0` check, `docs/graph.md:74-76`).

Reuse the Modal harness pattern (`modal run benchmarks/graph_memory/modal_bench.py::...`,
`docs/graph.md:148-150`); the build gotcha (maturin/third_party) is in the Modal-GPU-
benchmarks memory note.

---

## 7. Open questions

1. **Calibration policy (§5.2) is the real decision.** Do we (a) require a ≥1000-vector
   bulk seed via a documented `build`/`bulk_add_vectors` entry, (b) add an engine
   recalibrate-on-repack trigger, or (c) just document the floor? (a) is the smallest
   change and matches how adapters load an existing corpus; (b) is the most robust for
   true trickle ingest but touches the engine (which §4 otherwise avoids). Recommend (a)
   for v1, (b) as a tracked follow-up.
2. **Route-policy `None` vs a vector-only profile (§4.2).** Confirm the
   `require_turbovec_available` guard (`core.py:725-732`) is worth preserving for the
   local SDK (it is, since the whole product is TurboVec-backed) — which argues for the
   dedicated profile over `route_policy=None`.
3. **Collection semantics for cognee (§6.2).** Metadata-tag-in-one-index vs one-dir-per-
   collection. The tag approach reuses the engine-side filter pushdown (`research-prompts/02`);
   per-dir gives isolation but multiplies writer locks. Likely tag-based, but confirm
   against cognee's `prune`/`has_collection` expectations.
4. **Should `model_identity` be a first-class arg (§5.2)?** Persisting a caller-supplied
   `"text-embedding-3-large"` in the snapshot `model` field would let reopen reject a
   mismatched embedder for ~free and would be a genuine differentiator vs Qdrant/pgvector
   (§3.1). Defer unless an integration asks, but it is cheap.
5. **`vector_dim` upper bound.** TurboVec caps at `MAX_DIM = 65536` (`lib.rs:72`); the
   benchmarked regime is ≤3072. Do we cap the SDK lower (e.g. 4096) until larger dims are
   benchmarked, or expose the full range with a "unbenchmarked above 3072" note?
