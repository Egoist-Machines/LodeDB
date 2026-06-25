# LodeDB migration agent instructions

Public page source for `https://egoistmachines.com/lodedb/migrate-agent`. Paste this into an
agent run, or link it, when migrating an existing LangChain, LlamaIndex, or mem0 vector store
onto LodeDB. The matching CLI is `lodedb migrate`.

You are migrating an existing LangChain, LlamaIndex, or mem0 vector store to LodeDB. Work in the
user's application repo. Do not delete or modify the existing store. Use a branch or a patch.
Stop and ask if the current provider or source path is ambiguous.

## What the toolkit guarantees

- The source store is never modified or deleted. It stays as the rollback path.
- The new LodeDB store is written to `<target>.tmp` first, reopened read-only, validated, and
  only then moved into the target path. A real run never half-writes the target.
- Reports and the `migration.json` manifest are payload-free: counts, bytes, timings, id
  hashes, dimensions, versions, and warnings only. No raw documents, queries, vectors,
  embeddings, payloads, or credentials.

## 1. Identify the framework and provider

Search the repo for:

- LangChain: `VectorStore`, `InMemoryVectorStore`, `Chroma`, `QdrantVectorStore`, `Faiss`,
  `LanceDB`, sqlite-vec, pgvector.
- LlamaIndex: `StorageContext`, `VectorStoreIndex`, `SimpleVectorStore`, `persist_dir`,
  `vector_store`.
- mem0: `Memory.from_config`, `MemoryConfig`, `vector_store.provider`.

Record the current provider, persistence path, collection or table, document count if
available, and whether the app passes text or precomputed vectors.

`lodedb migrate inspect` does this for you and prints the routing decision:

```bash
lodedb migrate inspect --project . --json
```

If it reports `"route": "ambiguous"`, re-run with `--framework langchain|llama-index|mem0`.

## 2. Choose the LodeDB optional extra

- LangChain only: install `lodedb[langchain]`.
- LlamaIndex only: install `lodedb[llama-index]`.
- mem0 only: install `lodedb[mem0]`.
- Multiple frameworks: combine extras, for example `lodedb[langchain,llama-index,mem0]`.
- CUDA batch search on Linux only: add `gpu` after confirming the host has CUDA, for example
  `lodedb[mem0,gpu]`.
- Do not install unrelated extras.

Use the repo's package manager. Prefer `uv add`, Poetry, or the existing requirements file when
present; otherwise use pip. `lodedb migrate inspect` prints the exact install command for the
detected package manager.

## 3. Inspect and plan

```bash
lodedb doctor --json
lodedb migrate inspect --project . --json
lodedb migrate plan --project . --target ./data/lodedb --out lodedb-migration-plan.md
```

`plan` writes the Markdown plan and a sibling `lodedb-migration-plan.json`. Read the plan and
confirm the detected framework, provider, source path, target path, count, embedding model or
dimension, validation thresholds, the code/config switch snippet, and any warnings.

## 4. Migrate safely

Run a dry run first. It reads the source and confirms feasibility without writing anything:

```bash
lodedb migrate run --plan lodedb-migration-plan.json --target ./data/lodedb --dry-run
```

Then run the migration into a new target path with `--write` and validate it (a run with no
`--write` stays a dry run, so the migration step is explicit):

```bash
lodedb migrate run --plan lodedb-migration-plan.json --target ./data/lodedb --write
lodedb migrate validate --manifest ./data/lodedb/migration.json
```

Do not remove the source store. Keep it as the rollback path. If `run` exits non-zero because
validation failed, leave the app on the old provider and report the failed step.

## 5. Switch the application

Apply the generated code or config switch from the plan. The shapes per framework:

- LangChain: construct `lodedb.local.integrations.langchain.LodeDBVectorStore` over a
  `LodeDB(path=...)` handle and use it as the retriever's vector store.
- LlamaIndex: construct `StorageContext.from_defaults(vector_store=LodeDBVectorStore.from_path(...))`.
  Because the LodeDB adapter embeds text internally, pass `VectorStoreIndex` a cheap
  `MockEmbedding`; the vectors it computes are discarded.
- mem0: call `register_mem0_provider()` before `Memory.from_config(...)` and set the
  `vector_store` block to `{"provider": "lodedb", "config": {"path": ..., "collection_name": ...,
  "embedding_model_dims": ...}}`.

Run the application's retrieval tests, plus one restart/reopen test. If validation fails, leave
the app on the old provider and report the failing step.

## Migration model

Two replay modes, matching the shipped adapters:

- Text replay (LangChain, LlamaIndex). Export canonical text, ids, and metadata; LodeDB embeds
  with the selected preset and retains text in its raw-text sidecar by default, so reopen-safe
  retrieval and hybrid search work. Ranking parity with the source embeddings is not promised;
  validation checks count, metadata and filter behavior, stored-text recovery after reopen, and
  a representative query-overlap sample.
- Vector preserve (mem0). mem0 owns embeddings, so ids, vectors, and full payloads are copied
  verbatim. The full mem0 payload JSON is kept in LodeDB's raw-text sidecar, and scalar filter
  keys such as `user_id`, `agent_id`, and `run_id` stay in metadata so filtered reads stay
  exact. The embedding dimension must be a positive multiple of 8.

## Validation

`lodedb migrate run` validates the written store before moving it into place, and
`lodedb migrate validate` re-checks it from the manifest:

- source count equals target count, or every skipped row has a recorded reason;
- ids and metadata survive for a sample set;
- retained text can be fetched after reopen when text is stored;
- the persisted-index audit (`audit_persisted_index_snapshots`) passes and reports no raw text.

## Scope in this release

- Default and local source stores are wired end to end: LangChain `InMemoryVectorStore`,
  LlamaIndex `SimpleVectorStore` (persisted `StorageContext` docstore), and mem0 Qdrant.
- Production source stores (Chroma, remote Qdrant, FAISS, LanceDB, sqlite-vec, and pgvector
  beneath a framework) are tracked as follow-up source modules. The plan and routing already
  recognize them; the read-only exporters land next.
- The direct (non-framework) provider path, starting with pgvector, is documented on the
  companion [install-agent page](https://egoistmachines.com/lodedb/install-agent).
