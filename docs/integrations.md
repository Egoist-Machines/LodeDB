# Integrations roadmap

This document tracks LodeDB's integrations into the open-source RAG and agent ecosystem:
what ships today, what's in progress, and what's planned. It's a living tracker; status and
star counts are approximate and dated (last reviewed 2026-07).

LodeDB is a strong fit wherever a project already treats vector storage as a pluggable
backend, or where its current local default makes LodeDB's strengths easy to show:

- **Local-first, no server.** In-process, on-disk, no daemon, no account, no API key.
- **Exact recall by default.** A brute-force exact scan is the default and the authority, so
  results don't drift with index size. An opt-in IVF-style ANN (`ann="cluster"`) is available for
  large corpora; it re-scores its candidates exactly (exact scores) but is approximate (a neighbor
  in an unprobed cluster can be missed), so exact scan stays the default.
- **Hybrid retrieval.** `search` / `search_many` take a `mode` of `"vector"`, `"hybrid"`, or
  `"lexical"`. Hybrid fuses an Okapi BM25 lexical ranker with the vector scan via Reciprocal
  Rank Fusion, so exact tokens an embedding misses (error codes, serials, dates) are recovered
  when they appear in the document body.
- **GPU-resident batch search.** Batched `search_many` reaches ~24k q/s (A10) / ~50k q/s
  (L40S) at batch 1024, vs an ~8.5k–10.4k q/s CPU ceiling, recall unchanged. The crossover is
  around **batch 50**, so the win shows up under batched or concurrent load, not single
  interactive queries (`[gpu]` is CUDA, Linux only; an opt-in Apple MPS scan also exists, off
  by default).
- **O(changed) delta persistence.** Commits write only changed rows, staying sub-millisecond
  at 1M vectors (173×–1,308× faster than a full rewrite). Best shown in apps with frequent
  uploads, workspace sync, or chat-memory updates.

## Integration patterns

1. **Framework adapter (in-process).** Implement the host framework's native vector-store
   interface and keep LodeDB fully in-process. This is the cleanest path and yields
   upstream-quality, reusable integrations. Used for LangChain and LlamaIndex.
2. **App provider.** Plug LodeDB into an application's vector-DB provider/registry layer
   directly (Python apps), or via a thin **local bridge** for a first demo when the app is
   JS/TS and LodeDB ships as a Python package. `lodedb serve` binds to loopback by default and
   can bind to a trusted private-network address for LAN demos; it is unauthenticated and is
   not a public-network service.

## LodeDB as a local embeddings provider

`lodedb serve` exposes `POST /v1/embeddings` (and `POST /embeddings`), an OpenAI-compatible
embeddings endpoint backed by LodeDB's local ONNX embedding runtime. Any client that can point
an OpenAI-style `baseUrl` at a custom host can use LodeDB as a local embeddings provider without
an API key. Documents and queries never leave the machine. The embedding model itself is downloaded
once from Hugging Face on first use, so prefetch it before going offline or operating air-gapped. By
default, the server binds to loopback and is unauthenticated. The `Authorization` header is accepted
and ignored.

The request's `model` field is echoed back in the response; it does not select the embedding model.
The preset selected at `lodedb serve --model ...` is authoritative. If `dimensions` is present,
it must match the preset's native dimension or the request is rejected: `minilm` is 384, `bge` is
768, and `clip` is 512. The OpenAI wire format has no query/document role, so `minilm` is the
recommended preset because it is symmetric.

For [OpenKnowledge](https://github.com/inkeep/open-knowledge) (`inkeep/open-knowledge`),
semantic search consumes this API and whitelists `http://localhost`:

```bash
lodedb serve --model minilm --port 8088
```

```yaml
# <wiki>/.ok/local/config.yml
search:
  semantic:
    enabled: true
    baseUrl: http://127.0.0.1:8088
    model: minilm
    dimensions: 384
```

OpenKnowledge requires a non-empty API key before it will enable semantic search. Set
`OK_EMBEDDINGS_API_KEY` to any placeholder value, such as `local-placeholder`. It is only ever
sent to the loopback server, which ignores it.

## Migrating onto LodeDB

The `lodedb migrate` toolkit moves an existing store onto LodeDB along an
inspect -> plan -> dry-run -> run -> validate path, without hand-written export scripts and
without touching the source store (it stays as the rollback path). Detection routes a framework
owner (LangChain, LlamaIndex, mem0) to the framework path even when it is backed by pgvector or
Qdrant, and a project that wires a provider such as pgvector directly to the provider-first path.
Reports and the `migration.json` manifest are payload-free, and connection strings are redacted.

Two public agent pages cover the two entry points:

- [migrate-agent](migrate-agent.md): framework-first migration (LangChain, LlamaIndex, mem0).
  Source: `https://egoistmachines.com/lodedb/migrate-agent`.
- [install-agent](install-agent.md): provider-first install-and-migrate for direct providers
  (pgvector first), which hands framework projects off to the page above.
  Source: `https://egoistmachines.com/lodedb/install-agent`.

```bash
lodedb migrate inspect  --project . --json
lodedb migrate plan     --project . --target ./data/lodedb --out lodedb-migration-plan.md
lodedb migrate run      --plan lodedb-migration-plan.json --target ./data/lodedb --dry-run
lodedb migrate run      --plan lodedb-migration-plan.json --target ./data/lodedb --write
lodedb migrate validate --manifest ./data/lodedb/migration.json
```

## Status

| Target | Type | Stars≈ | Pattern | Status |
|---|---|---:|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | Framework | 139k | `VectorStore` adapter (`lodedb[langchain]`) | ✅ Shipped |
| [LlamaIndex](https://github.com/run-llama/llama_index) | Framework | 50k | `VectorStore` + `PropertyGraphStore` adapters (`lodedb[llama-index]`) | ✅ Shipped ([#7](https://github.com/Egoist-Machines/LodeDB/pull/7)) |
| [mem0](https://github.com/mem0ai/mem0) | Agent memory | 59k | `VectorStoreBase` provider (`lodedb[mem0]`) | ✅ Shipped ([#16](https://github.com/Egoist-Machines/LodeDB/pull/16)) |
| [cognee](https://github.com/topoteretes/cognee) | Graph memory | 7k | `VectorDBInterface` provider (`lodedb[cognee]`) | ✅ Shipped ([#74](https://github.com/Egoist-Machines/LodeDB/pull/74)) |
| [PrivateGPT](https://github.com/zylon-ai/private-gpt) | Local RAG app | 57k | LlamaIndex provider shim + `settings.yaml` key | ✅ Shipped |
| [OpenKnowledge](https://github.com/inkeep/open-knowledge) | LLM wiki / editor (TS) | n/a | OpenAI-compatible embeddings endpoint (`lodedb serve`) | ✅ Shipped |
| [Haystack](https://github.com/deepset-ai/haystack) | Framework | 25k | `DocumentStore` protocol | 📋 Backlog |
| [txtai](https://github.com/neuml/txtai) | Framework | n/a | `custom` backend (resolvable class string) | 📋 Backlog |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | Local RAG app (JS) | 62k | Provider layer + local bridge | 📋 Backlog |
| [Open WebUI](https://github.com/open-webui/open-webui) | Local RAG app (Py) | 140k | `VECTOR_DB` provider | 📋 Backlog |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Agent app (TS) | 53k | LangChain vector-store node / bridge | 📋 Backlog |
| [kotaemon](https://github.com/Cinnamon/kotaemon) | Local RAG app | 26k | `BaseVectorStore` via `KH_VECTORSTORE` (no extra needed) | ✅ Shipped |
| [LocalGPT](https://github.com/PromtEngineer/localGPT) | Local RAG app | 22k | Replace bundled LanceDB | 📋 Backlog |
| [Msty](https://msty.app/) | Local AI workspace | n/a | Knowledge Stack storage/index bridge | 🔬 Evaluating |
| [GPT4All](https://github.com/nomic-ai/gpt4all) | Local LLM app | n/a | LocalDocs storage backend | 🔬 Evaluating |
| [DocsGPT](https://github.com/arc53/DocsGPT) | Private AI / enterprise search | 18k | Vector-store provider | 🔬 Evaluating |
| [Frigate](https://github.com/blakeblackshear/frigate) | Local NVR / edge vision | n/a | Semantic-search vector sidecar | 🔬 Evaluating |
| [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) | Local document management | n/a | Semantic document-similarity sidecar | 🔬 Evaluating |
| [Dify](https://github.com/langgenius/dify) | RAG platform (TS) | 145k | `VECTOR_STORE` provider | 🔬 Evaluating (heavy stack) |
| [Khoj](https://github.com/khoj-ai/khoj) | App | 35k | config-key backend | ⛔ Blocked (AGPL-3.0) |

Legend: ✅ shipped · 🚧 in review · 🔜 planned · 📋 backlog · 🔬 evaluating · ⛔ blocked.

## 2026-07 target review

A July research pass looked for products where an embedded, low-footprint vector store removes
specific friction: local RAG rebuilds, private memory, media search, or document search without a
separate Chroma/Qdrant/LanceDB service. The result was a prioritization pass, not a new
integration class.

Overlap with current work:

- **Already shipped:** mem0 and PrivateGPT. Do not start new adapters here; keep the next work to
  benchmarks, examples, migration docs, and upstream usage notes.
- **In review:** cognee. Finish the existing adapter and the cognee-community package before adding
  another graph-memory target.
- **Already on the backlog:** Haystack, txtai, AnythingLLM, Open WebUI, Flowise, kotaemon,
  LocalGPT, and Dify. Keep them in the existing sequence below; the research only changes the
  order inside that queue.
- **Blocked / licensing review:** Khoj remains blocked by AGPL-3.0. Immich is also AGPL-3.0, so
  treat it as a visibility demo or local sidecar only unless legal review says otherwise.

Recommended order from this pass:

1. **kotaemon.** ✅ Done (see Recently shipped): the `KH_VECTORSTORE` seam took a
   dependency-free adapter and a settings change, benchmarked ~9× ingest / ~10× incremental
   add / ~40× disk vs the Chroma default. Next step is the upstream conversation.
2. **Haystack and txtai.** These are still the best ecosystem multipliers. They should precede
   most one-off app integrations unless an app partner is ready to test.
3. **Open WebUI and DocsGPT.** Both are Python-heavy private AI apps with user-visible document
   search. Start with a provider proof of concept and a benchmark against their current local
   default.
4. **Msty and GPT4All.** Both have the exact local-docs workload. The first useful step is a
   loopback/local-service prototype because neither is naturally a Python package consumer.
5. **AnythingLLM and Flowise.** Broad distribution, but JS/TS. Reuse the same local bridge used for
   Msty/GPT4All before considering native bindings.
6. **Frigate and Paperless-ngx.** These are not generic RAG apps, but both have local, private
   corpora and search features where compact semantic indexes matter. Treat them as second-wave
   demos after the local-RAG apps.

Watch-only until there is an owner:
[Obsidian](https://obsidian.md/), [Anytype](https://anytype.io/),
[Jan](https://github.com/janhq/jan), [AFFiNE](https://github.com/toeverything/AFFiNE),
[Zotero](https://github.com/zotero/zotero), [Ente](https://github.com/ente-io/ente),
and [DEVONthink](https://www.devontechnologies.com/apps/devonthink). Each needs a plugin,
Swift/mobile binding, or proprietary-app relationship before it becomes actionable.

## Sequencing

**Done.** LangChain, LlamaIndex (vector + property-graph), and mem0
([#16](https://github.com/Egoist-Machines/LodeDB/pull/16)) have landed, so the framework
foundation is in place and every app built on those abstractions is reachable without a new
adapter. **PrivateGPT** ships on top of it as a provider shim plus one config key: its store
layer *is* LlamaIndex's `BasePydanticVectorStore`, so no new adapter was needed.

**Now.** Framework breadth: **Haystack** and **txtai**, both low-effort drop-in adapters. The
first local-RAG app demo, **kotaemon**, has shipped (see Recently shipped); its remaining work
is the upstream PR/usage-note conversation with Cinnamon/kotaemon.

**Later.** High-visibility local apps. Lead with permissively licensed Python apps (Open WebUI,
DocsGPT, LocalGPT); the JS/TS or desktop apps (AnythingLLM, Flowise, Msty, GPT4All, Dify) need
the local-bridge step first.

## Recently shipped

### kotaemon
- **Repo:** Cinnamon/kotaemon (~26k★, Apache-2.0). Local, open-source RAG UI for chatting
  with documents. **Install:** none — the adapter is dependency-free (it duck-types
  kotaemon's `DocumentWithEmbedding` and LlamaIndex's `MetadataFilters` rather than
  importing either package), so plain `lodedb` inside kotaemon's environment is enough.
- **Pattern:** `lodedb.local.integrations.kotaemon.LodeDBVectorStore` implements kotaemon's
  `BaseVectorStore` contract (`add` / `delete` / `query` / `drop`, plus `count` and
  `__persist_flow__`). kotaemon resolves its vector store from a dotted `__type__` path via
  theflow's `deserialize`, so selection is a settings change only — no kotaemon fork, no
  registration call:

  ```python
  # flowsettings.py
  KH_VECTORSTORE = {
      "__type__": "lodedb.local.integrations.kotaemon.LodeDBVectorStore",
      "path": str(KH_USER_DATA_DIR / "vectorstore"),
  }
  ```

  kotaemon's `get_vectorstore` injects `collection_name` per index; each collection is one
  LodeDB store under `<path>/<collection_name>`. See
  [`examples/kotaemon_store.py`](../examples/kotaemon_store.py).
- **Why it fits:** kotaemon's default is a Chroma collection per file index, rebuilt chunk
  batch by chunk batch as users upload files — exactly the incremental-write workload
  LodeDB's O(changed) delta persistence targets, and the app is local-first so a zero-server
  embedded store is a category match.
- **Embeddings:** kotaemon owns embedding (its pipelines pass precomputed vectors), so the
  adapter uses LodeDB's vector-in path. kotaemon never configures an embedding dimension —
  the store meets whatever model the user selected — so the LodeDB index is created lazily on
  first `add` and its shape is recorded in a `kotaemon_store.json` sidecar for reopens.
  Dimensions that are not a multiple of 8 (a LodeDB index requirement) are zero-padded up,
  which changes neither norms nor dot products, so cosine scores are exact. Chunk text and
  documents stay in kotaemon's docstore (retrieval re-reads them by id); the adapter keeps
  only vectors, ids, and scalar filter metadata (e.g. `file_id`), with each row's id mirrored
  into a reserved `kh::id` metadata key so kotaemon's chunk-id `scope` (`doc_ids`) and the
  `file_id IN [...]` `MetadataFilters` push down into LodeDB's metadata planner.
- **Validated:** kotaemon's own vector-store test shapes pass against the adapter (mirrored
  in `tests/test_kotaemon_adapter.py`, which needs no kotaemon install), and a live check in
  a kotaemon 0.11 environment ran the real `deserialize(KH_VECTORSTORE)` path, real
  `DocumentWithEmbedding` / `MetadataFilters` objects, and top-1 parity with the Chroma
  default on normalized vectors (14/14 checks).
- **Benchmark (kotaemon interface, 20k × 384-dim normalized vectors, batches of 100, one
  process per backend, M-series CPU):** vs the default `ChromaVectorStore`, byte-identical
  vectors/ids/metadata/top-k on both sides — ingest 1.8s vs 16.9s (~9×); per-100-chunk add
  (= durable commit at this seam) p50 8.6ms / p95 8.9ms / p99 9.0ms vs p50 83.9ms / p95
  111ms / p99 126ms (the O(changed) tail story); query p50 0.17ms / p95 0.22ms vs 64.2ms /
  65.1ms; 500-chunk scoped query p50 0.31ms vs 64.5ms; disk 10.5MB vs 424MB (~40×); peak RSS
  ~even (592MB vs 577MB, both dominated by the benchmark fixture); top-1 accuracy vs a
  brute-force oracle 1.000 vs 0.440 (near-tie random vectors are adversarial for HNSW;
  real-embedding gaps are smaller, but exact scan cannot miss the true neighbor). Chroma's
  cold reopen is faster (0.07s vs 0.43s). Batch-query latency is N/A at this seam —
  kotaemon's `BaseVectorStore` has no batch-query API (retrieval issues single queries);
  LodeDB's batch path is covered by the repo's own benchmarks. Score scale differs by
  design: LodeDB returns cosine similarity, Chroma's default collection ranks by L2
  (identical ordering on normalized embeddings). Kotaemon has no retrieval test set, so
  quality checks use the brute-force oracle; artifacts are metrics-only (counts, latencies,
  sizes).
- **Scope:** vector store only; kotaemon's docstore (LanceDB/Elasticsearch/SimpleFile) and
  graph indices are separate seams and unchanged. Upstream, kotaemon bundles store choices in
  `libs/kotaemon/kotaemon/storages/vectorstores/`; a PR there would add a thin subclass plus
  a `pip install lodedb` extra in their docs — the settings-path integration above works
  today without it.

### cognee ([#74](https://github.com/Egoist-Machines/LodeDB/pull/74), in review)
- **Repo:** topoteretes/cognee (~7k★, Apache-2.0). AI memory that builds a knowledge graph
  over your data. **Install:** `lodedb[cognee]` (`cognee>=1.1.0,<2`).
- **Pattern:** `CogneeLodeDBAdapter` implements cognee's `VectorDBInterface`; call
  `register_cognee_adapter()` to register the `"lodedb"` provider, then select it with
  `cognee.config.set_vector_db_config({"vector_db_provider": "lodedb", "vector_db_url": "<dir>"})`.
  See [`examples/cognee_provider.py`](../examples/cognee_provider.py). Also packaged for the
  [cognee-community](https://github.com/topoteretes/cognee-community) plugin repo as
  `cognee-community-vector-adapter-lodedb`, whose `register` module registers this same adapter.
- **Why it fits:** cognee is a graph-memory layer whose vector index accrues via incremental
  data-point adds and NodeSet re-tagging, which is exactly the workload LodeDB's O(changed) delta
  persistence is built for. It ships LanceDB as its embedded local default, so a zero-server exact
  store is a category match.
- **Embeddings:** cognee owns embedding (its `EmbeddingEngine`), so the adapter uses LodeDB's
  vector-in path. Each cognee collection is one LodeDB index under `vector_db_url`; the serialized
  DataPoint payload goes to LodeDB's raw-text sidecar (so `retrieve` and `include_payload` searches
  return it), while `belongs_to_set` membership is stored as scalar presence keys so cognee's
  `node_name` (NodeSet) filtering pushes into the metadata planner. cognee ranks by cosine distance
  (lower is better), so searches report `1 - similarity`.
- **Scope:** vector store (`VectorDBInterface`), including `create_data_points` / `search` /
  `batch_search` / `retrieve` / `delete_data_points` / `create_vector_index` / `index_data_points`
  / `prune`, plus `remove_belongs_to_set_tags` and `upsert_raw_vectors` for dataset-deletion
  consistency and system-owned vectors. cognee's graph and relational stores are separate providers
  and unchanged.

### mem0 ([#16](https://github.com/Egoist-Machines/LodeDB/pull/16))
- **Repo:** mem0ai/mem0 (~59k★, Apache-2.0). **Install:** `lodedb[mem0]` (`mem0ai>=2.0.0`).
- **Pattern:** `LodeDBVectorStore` implements mem0's `VectorStoreBase`; call
  `register_mem0_provider()` to register the `"lodedb"` provider, then select it with
  `Memory.from_config({"vector_store": {"provider": "lodedb", "config": {...}}})`. See
  [`examples/mem0_store.py`](../examples/mem0_store.py).
- **Why it fits:** mem0 is an agent-memory layer where memories accrue via incremental adds,
  which is exactly the workload LodeDB's O(changed) delta persistence is built for. It already
  ships FAISS / Chroma / embedded-Qdrant, so a local zero-server store is a category match, not
  a stretch.
- **Embeddings:** mem0 owns embedding, so the adapter uses LodeDB's vector-in path rather than
  the text path. Full payloads (raw memory text and list-valued fields) go to LodeDB's raw-text
  sidecar, while scalar fields such as `user_id` / `agent_id` / `run_id` stay in metadata so
  mem0's filtered reads stay exact.
- **Scope:** vector-store only. mem0's v2 line dropped the optional OSS graph-memory layer, so
  there is no graph adapter; for a LodeDB-backed graph use `lodedb.graph.KnowledgeGraph` or the
  LlamaIndex `LodeDBPropertyGraphStore`.

### PrivateGPT (provider shim on the LlamaIndex adapter)
- **Repo:** zylon-ai/private-gpt (~57k★, Apache-2.0). Marquee "100% private" RAG.
- **Pattern:** PrivateGPT's store layer *is* LlamaIndex's `BasePydanticVectorStore`, selected by
  `vectorstore.database` in `settings.yaml`. Concretely, `VectorStoreComponent` builds
  `{"qdrant": QdrantVectorStoreFactory, **_PROVIDERS}` and resolves `vectorstore.database` against
  it; providers register into `_PROVIDERS` via
  `private_gpt.components.vector_store.factory.register_vector_store(database, provider)`, where a
  provider is a `VectorStoreFactory` (ABC) with `vector_store(collection) -> BasePydanticVectorStore`.
  The `database` field is a free-form `str` (no `Literal` allow-list), so `database: lodedb` is
  accepted once the provider is registered.
- **What LodeDB ships:** `lodedb.local.integrations.privategpt`, a `VectorStoreFactory` subclass
  that reads PrivateGPT settings and builds the existing text-path `LodeDBVectorStore` (one local
  LodeDB index per collection), plus `register_lodedb_provider()` that registers it under
  `"lodedb"`. It reuses the LlamaIndex adapter from [#7](https://github.com/Egoist-Machines/LodeDB/pull/7);
  it is **not** a new adapter. Needs `lodedb[llama-index]` installed into PrivateGPT's environment.
- **What must live in PrivateGPT's process:** `_PROVIDERS` is process-local with no entry-point
  auto-discovery, so registration has to be *triggered* there, with one line
  (`from lodedb.local.integrations.privategpt import register_lodedb_provider; register_lodedb_provider()`)
  in a tiny launcher or near the top of `private_gpt/__main__.py`, before the
  `VectorStoreComponent` is built. Then set in `settings.yaml`:

  ```yaml
  vectorstore:
    database: lodedb        # or PGPT_VECTORSTORE=lodedb
  lodedb:                   # optional; defaults shown
    path: local_data/lodedb
    model: minilm           # "minilm" (fast) or "bge" (quality)
    device: auto            # auto | cpu | mps | cuda
    store_text: true        # keep on for hybrid/lexical retrieval
    index_text: true        # persist the lexical index; omit to follow store_text
  ```

  This needs no PrivateGPT fork (`register_vector_store` is the documented extension seam), but
  the trigger and the YAML keys are PrivateGPT-side and cannot be set from LodeDB. See
  [`examples/privategpt_provider.py`](../examples/privategpt_provider.py).
- **Text-path:** like the LlamaIndex adapter, LodeDB embeds text itself, so PrivateGPT's own
  embedding model is bypassed for storage/query and `vectorstore.embed_dim` is informational here;
  point PrivateGPT at a cheap/mock embedding to avoid a redundant embedding call.
- **Caveat:** this targets PrivateGPT's current factory-registry store layer. On a much older
  PrivateGPT whose `VectorStoreComponent` is a hardcoded `if/elif` over `database` with no
  `register_vector_store`, there is no clean seam without a fork; wire `LodeDBVectorStore` in by
  hand there instead.

## Planned developments

### Haystack and txtai (low-effort framework breadth)
- **Haystack** (~25k★, Apache-2.0): duck-typed `DocumentStore` protocol (≈6 methods), no base
  class to inherit.
- **txtai** (Apache-2.0): accepts a `custom` backend as a fully-resolvable class string, a
  drop-in adapter with no fork. (txtai already lists `turbovec` among its ANN backends, so its
  users already trust the same core; LodeDB slots in as a distinct exact and persistent
  backend.)

### Local RAG apps (visibility demos)
- **LocalGPT** (~22k★, MIT): bundles LanceDB (`lancedb_uri`), a contained code swap.
- **Open WebUI** (~140k★): `VECTOR_DB` selector; its docs warn the default Chroma/SQLite path
  is unsafe under multiple workers or replicas, so LodeDB is both a performance and a *simpler
  local persistence* story. Note its license has branding requirements (see Licensing).
- **DocsGPT** (~18k★, MIT): private AI / enterprise-search app with local-model support and
  document ingestion. Evaluate a vector-store provider rather than a fork.
- **Msty**: Knowledge Stacks ingest files, folders, Obsidian vaults, notes, and transcripts, then
  search them locally. Start with a local bridge or native binding plan.
- **GPT4All**: LocalDocs indexes folders into embedding vectors for private desktop RAG. The first
  step is to benchmark LodeDB as a LocalDocs storage backend against the current index.
- **Frigate**: semantic search over tracked-object thumbnails and descriptions runs locally and
  has explicit RAM/GPU pressure. Keep this as an edge/media demo, not a core RAG adapter.
- **Paperless-ngx**: local OCR document archive with full-text search and "more like this"; a
  semantic sidecar can improve similarity search without replacing the document store.
- **AnythingLLM** (~62k★, JS) / **Flowise** (~53k★, TS): reach via the provider/node layer;
  both need a local bridge for a first demo (see language boundary below).

### Watch list (agent-memory / knowledge layer)
Same sweet spot as mem0; surfaced by research but not yet scoped:
[Letta/MemGPT](https://github.com/letta-ai/letta),
[Microsoft AutoGen](https://github.com/microsoft/autogen),
[CrewAI](https://github.com/crewAIInc/crewAI),
[Graphiti](https://github.com/getzep/graphiti)
(cognee has shipped an adapter, see below).
LodeDB now ships a knowledge-graph layer (`lodedb.graph.KnowledgeGraph`) and a LlamaIndex
`PropertyGraphStore` adapter, which strengthens the fit for the graph-memory targets here.

## Design constraints that shape adapters

- **Text-path by default, with a vector-in path available.** The high-level `LodeDB` SDK is
  text-in and embeds internally (`add` / `search` take strings), and the LangChain and
  LlamaIndex vector-store adapters use that path, so the host framework's own embedding model
  is bypassed (LangChain's contract is text-based so this is invisible; the LlamaIndex
  vector-store adapter sets `is_embedding_query=False`). A precomputed-vector path now also
  exists: `add_vectors` / `add_vectors_many` / `search_by_vector` / `search_many_by_vector`,
  plus a vector-only index (`LodeDB(vector_dim=...)` / `open_vector_store`) with no internal
  embedder. The LlamaIndex `PropertyGraphStore` adapter uses this mode to honor any host
  `embed_model`, and the mem0 adapter uses it because mem0 owns embedding, so an adapter that
  needs the host's own embeddings is no longer blocked.
- **Metadata filters support comparison operators and boolean composition.** `filter=` accepts
  `$eq` / `$ne` / `$gt` / `$gte` / `$lt` / `$lte` / `$in` / `$nin` / `$exists` with `$and` /
  `$or` / `$not` composition; a bare scalar stays exact-match. The engine also exposes metadata
  enumeration (`list_documents` / `count`) and by-id reads (`get`). Adapters can therefore back
  filter, delete-by-filter, and get-by-id features rather than refusing them. Filter shapes
  outside this grammar (for example substring match) still raise clearly rather than silently
  degrade.
- **Language boundary.** LodeDB is a Python package with an embedded Rust core. JS/TS targets
  (AnythingLLM, Flowise, Dify) need a thin local bridge for a first demo before any
  native binding is worth it. Use loopback by default; private-network binds are only for
  trusted LAN demos because the bridge is intentionally unauthenticated. Land clean
  in-process integrations in Python ecosystems first.
- **GPU acceleration is opt-in.** The CUDA-resident exact scan ships in the `[gpu]` extra
  (cupy, Linux/CUDA), and the batch-throughput figures above are CUDA measurements. An opt-in
  Apple MPS resident scan also exists (`LODEDB_MPS_DIRECT_TURBOVEC`), but it's off by default
  and NEON stays the default on Apple Silicon, since MPS was not faster than NEON on the
  measured M1. Everywhere else the CPU scan is the source of truth.

## Shared benchmark plan

Reuse one harness across targets so results are comparable:

- **Datasets:** BEIR **SciFact**, **FiQA-2018**, **NFCorpus** (judged relevance for Recall@k
  and NDCG@k), plus a synthetic dense-vector stress case at **100k and 1M** vectors for
  batch-search and delta-write behavior.
- **Metrics:** ingest throughput (chunks/s); persist latency p50/p95/p99; single-query
  latency; **batch latency at 1 / 16 / 64 / 256 / 1024**; query throughput; Recall@10;
  NDCG@10; disk size; RSS/VRAM; cold-start reopen time; and concurrent-user error rate for app
  targets.
- **Fairness:** fix the **same local embedding model and dimension** (for example a BGE preset)
  and the **same chunking policy and top-k** across baseline and LodeDB; cross-check exact
  recall against a brute-force oracle on the same embeddings.
- **Always include a batched and a concurrent mode.** Single-query comparisons hide LodeDB's
  GPU and concurrency strengths (crossover ≈ batch 50).
- **Demo success rule:** no regression in retrieval quality; a clear win in write/update
  latency; and at least one mode where batch throughput or concurrent responsiveness is plainly
  better. GPU is a second layer of benefit, not the only one.

## Licensing notes

Lead with permissively licensed ecosystems (Apache-2.0 / MIT): LangChain, LlamaIndex,
Haystack, PrivateGPT, mem0, kotaemon, LocalGPT, txtai.

- **Khoj** is **AGPL-3.0**: strong copyleft, treated as blocked for integrations that need
  permissive or closed redistribution.
- **Open WebUI** and **Dify** ship source-available licenses with extra conditions (Open WebUI
  has branding requirements; Dify adds conditions on multi-tenant/commercial use). They're fine
  as visibility demos, but verify terms before contributing code upstream.

Licenses and star counts should be re-verified before relying on them externally.
