# Integrations roadmap

This document tracks LodeDB's integrations into the open-source RAG and agent ecosystem:
what ships today, what's in progress, and what's planned. It's a living tracker; status and
star counts are approximate and dated (last reviewed 2026-06).

LodeDB is a strong fit wherever a project already treats vector storage as a pluggable
backend, or where its current local default makes LodeDB's strengths easy to show:

- **Local-first, no server.** In-process, on-disk, no daemon, no account, no API key.
- **Exact recall.** A brute-force exact scan rather than ANN, so results don't drift with
  index size.
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

## Status

| Target | Type | Stars≈ | Pattern | Status |
|---|---|---:|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | Framework | 139k | `VectorStore` adapter (`lodedb[langchain]`) | ✅ Shipped |
| [LlamaIndex](https://github.com/run-llama/llama_index) | Framework | 50k | `VectorStore` + `PropertyGraphStore` adapters (`lodedb[llama-index]`) | ✅ Shipped ([#7](https://github.com/Egoist-Machines/LodeDB/pull/7)) |
| [mem0](https://github.com/mem0ai/mem0) | Agent memory | 59k | `VectorStoreBase` provider (`lodedb[mem0]`) | ✅ Shipped ([#16](https://github.com/Egoist-Machines/LodeDB/pull/16)) |
| [PrivateGPT](https://github.com/zylon-ai/private-gpt) | Local RAG app | 57k | LlamaIndex provider shim + `settings.yaml` key | ✅ Shipped |
| [Haystack](https://github.com/deepset-ai/haystack) | Framework | 25k | `DocumentStore` protocol | 📋 Backlog |
| [txtai](https://github.com/neuml/txtai) | Framework | n/a | `custom` backend (resolvable class string) | 📋 Backlog |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | Local RAG app (JS) | 62k | Provider layer + local bridge | 📋 Backlog |
| [Open WebUI](https://github.com/open-webui/open-webui) | Local RAG app (Py) | 140k | `VECTOR_DB` provider | 📋 Backlog |
| [Flowise](https://github.com/FlowiseAI/Flowise) | Agent app (TS) | 53k | LangChain vector-store node / bridge | 📋 Backlog |
| [kotaemon](https://github.com/Cinnamon/kotaemon) | Local RAG app | 26k | Swappable backend (`flowsettings.py`) | 📋 Backlog |
| [LocalGPT](https://github.com/PromtEngineer/localGPT) | Local RAG app | 22k | Replace bundled LanceDB | 📋 Backlog |
| [Dify](https://github.com/langgenius/dify) | RAG platform (TS) | 145k | `VECTOR_STORE` provider | 🔬 Evaluating (heavy stack) |
| [Khoj](https://github.com/khoj-ai/khoj) | App | 35k | config-key backend | ⛔ Blocked (AGPL-3.0) |

Legend: ✅ shipped · 🚧 in review · 🔜 planned · 📋 backlog · 🔬 evaluating · ⛔ blocked.

## Sequencing

**Done.** LangChain, LlamaIndex (vector + property-graph), and mem0
([#16](https://github.com/Egoist-Machines/LodeDB/pull/16)) have landed, so the framework
foundation is in place and every app built on those abstractions is reachable without a new
adapter. **PrivateGPT** ships on top of it as a provider shim plus one config key: its store
layer *is* LlamaIndex's `BasePydanticVectorStore`, so no new adapter was needed.

**Now.** Framework breadth: **Haystack** and **txtai**, both low-effort drop-in adapters.

**Later.** High-visibility local apps. Lead with permissively licensed Python apps (kotaemon,
LocalGPT, Open WebUI); the JS/TS apps (AnythingLLM, Flowise, Dify) need the local-bridge step
first.

## Recently shipped

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
    index_text: false
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
- **kotaemon** (~26k★, Apache-2.0): swappable vector stores including an InMemory option via
  `flowsettings.py`; LodeDB is a strict upgrade (persistent, exact, fast).
- **LocalGPT** (~22k★, MIT): bundles LanceDB (`lancedb_uri`), a contained code swap.
- **Open WebUI** (~140k★): `VECTOR_DB` selector; its docs warn the default Chroma/SQLite path
  is unsafe under multiple workers or replicas, so LodeDB is both a performance and a *simpler
  local persistence* story. Note its license has branding requirements (see Licensing).
- **AnythingLLM** (~62k★, JS) / **Flowise** (~53k★, TS): reach via the provider/node layer;
  both need a local bridge for a first demo (see language boundary below).

### Watch list (agent-memory / knowledge layer)
Same sweet spot as mem0; surfaced by research but not yet scoped:
[Letta/MemGPT](https://github.com/letta-ai/letta),
[Microsoft AutoGen](https://github.com/microsoft/autogen),
[CrewAI](https://github.com/crewAIInc/crewAI),
[cognee](https://github.com/topoteretes/cognee),
[Graphiti](https://github.com/getzep/graphiti).
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
