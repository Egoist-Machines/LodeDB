# LodeDB local install and provider migration instructions

Public page source for `https://egoistmachines.com/lodedb/install-agent`. Paste this into an
agent run, or link it, when helping a local project move from its current vector provider to
LodeDB. The matching CLI is `lodedb migrate`.

You are helping migrate this local project from its current vector provider to LodeDB. Work in
the project checkout. Do not delete the existing provider data, database tables, collections,
Docker volumes, or config. Stop and ask for confirmation if the current provider, source
location, embedding dimension, or package manager is ambiguous.

This page is the provider-first entry point. It is a router: if the project uses LangChain,
LlamaIndex, or mem0, it hands off to the framework migration toolkit on the
[migrate-agent page](https://egoistmachines.com/lodedb/migrate-agent) instead of attempting a direct provider export. Direct
migration here is for projects that use a store such as pgvector directly from application code.

## 1. Identify the project and package manager

Check for `pyproject.toml`, `uv.lock`, `poetry.lock`, `requirements.txt`, `setup.py`,
`package.json`, Docker Compose files, and application config. Use the project's existing package
manager. Do not introduce a new one unless the user asks.

## 2. Identify whether this is a framework migration or a direct-provider migration

Search source and config for framework ownership first:

- LangChain: `VectorStore`, `Retriever`, `InMemoryVectorStore`, `QdrantVectorStore`, `Chroma`,
  `FAISS`/`Faiss`, `LanceDB`, or LangChain retriever construction.
- LlamaIndex: `StorageContext`, `VectorStoreIndex`, `SimpleVectorStore`, `persist_dir`,
  `docstore`, `index_store`, or LlamaIndex vector-store construction.
- mem0: `Memory.from_config`, `MemoryConfig`, `vector_store.provider`, or mem0 provider config.

The routing rule is: **framework detection wins over direct-provider detection.** If LangChain,
LlamaIndex, or mem0 owns the vector store, switch to the framework migration toolkit on the
[migrate-agent page](https://egoistmachines.com/lodedb/migrate-agent). Do not continue with a direct pgvector/Qdrant/Chroma
export unless the plan explicitly shows the framework is not the owner. If a project uses
pgvector *through* LangChain, this is a LangChain migration, not a direct pgvector migration.

If no framework owner is found, search for direct providers:

- pgvector: `pgvector`, `CREATE EXTENSION vector`, `vector(...)`, `embedding` columns, `psycopg`,
  `asyncpg`, SQLAlchemy models, Alembic migrations, Postgres connection strings.
- Qdrant: `qdrant_client`, `QdrantClient`, collection names, local `path`, or `url`/`host`.
- Chroma: `chromadb`, `Chroma`, `persist_directory`, collection names.
- LanceDB: `lancedb`, `LanceDB`, `uri`, table names.
- sqlite-vec: `sqlite_vec`, `vec0`, SQLite files, vector virtual tables.
- FAISS: `faiss`, `save_local`, `load_local`, index files, docstore files.

`lodedb migrate inspect` applies all of this, including the framework-wins routing rule, and
prints the decision:

```bash
lodedb migrate inspect --project . --json
```

A framework owner returns `"route": "framework"` with `"next"` pointing at the framework
toolkit. A direct provider returns `"route": "provider"`. Ambiguity returns
`"route": "ambiguous"`; re-run with `--framework` or `--provider`.

## 3. Pick the LodeDB install command

Install only what is needed. The base `lodedb` is a vector store with no embedding runtime; add
the `[embeddings]` extra only when LodeDB itself embeds text:

- Vector-preserve migration (the app owns embeddings; you insert precomputed vectors): `lodedb`.
- Text-owned migration (LodeDB embeds text going forward, `model=...`): `lodedb[embeddings]`
  (ONNX, torch-free), or `lodedb[embeddings,torch]` to add the PyTorch runtime.
- LangChain code path: `lodedb[langchain]`, then follow the framework toolkit.
- LlamaIndex code path: `lodedb[llama-index]`, then follow the framework toolkit.
- mem0 code path: `lodedb[mem0]`, then follow the framework toolkit.
- CUDA batch search on Linux only, after confirming CUDA: add `gpu`, for example `lodedb[gpu]`.

Use the existing package manager, for example `uv add lodedb`, `poetry add lodedb`, or update
`requirements.txt`. `lodedb migrate inspect` prints the exact command for the detected manager.
Run `lodedb doctor --json` after install.

## 4. Plan before migrating

```bash
lodedb migrate inspect --project . --json
lodedb migrate plan --provider pgvector --source "$DATABASE_URL" --table documents \
  --target ./data/lodedb --out lodedb-migration-plan.md
```

`plan` writes the Markdown plan and a sibling `.json`. Read it and confirm whether it chose the
framework path or the direct-provider path, plus provider, source, target, dimension, counts,
install command, warnings, and rollback. A credentialed connection string is never written into
the plan or the manifest; only its redacted form and a fingerprint are stored, so you re-supply
the connection at run time with `--source`.

## 5. Migrate safely

If the plan routes to a framework, follow the framework migration commands on the
[migrate-agent page](https://egoistmachines.com/lodedb/migrate-agent). If it stays on the direct-provider path, run a dry run
first, then write a new LodeDB target path:

```bash
lodedb migrate run --plan lodedb-migration-plan.json --source "$DATABASE_URL" \
  --target ./data/lodedb --dry-run
lodedb migrate run --plan lodedb-migration-plan.json --source "$DATABASE_URL" \
  --target ./data/lodedb --write
lodedb migrate validate --manifest ./data/lodedb/migration.json
```

Do not drop old tables, delete old collections, remove Docker volumes, or erase old files. The
source provider is read-only throughout: the toolkit issues only `SELECT` and read-only catalog
probes against Postgres, never `DROP` / `DELETE` / `TRUNCATE` / `UPDATE` / schema or index
changes. A non-local source host is refused unless you pass `--allow-remote-source` after
confirming the host is safe to read.

## 6. Switch the app

Apply the generated code or config switch from the plan. Prefer the narrowest change: replace
the provider construction, keep the same embedding model where vector-preserve mode is used, and
keep the old config as rollback until validation and application tests pass. The two direct-SDK
shapes the plan generates:

Vector-preserve (the app owns embeddings):

```python
from lodedb import LodeDB

db = LodeDB.open_vector_store("./data/lodedb", vector_dim=1536)
hits = db.search_by_vector(query_embedding, k=10, filter={"metadata": {"tenant_id": tenant_id}})
```

Text-owned (LodeDB owns embeddings going forward; needs `lodedb[embeddings]`):

```python
from lodedb import LodeDB

db = LodeDB("./data/lodedb", model="bge")
hits = db.search(query_text, k=10, filter={"metadata": {"tenant_id": tenant_id}})
```

Run the application's retrieval tests plus one restart/reopen test. If validation fails, leave
the app on the old provider and report the failing step.

## Provider migration modes

The plan states which mode it uses and why. It does not silently switch from framework handoff
to direct export, or from vector preserve to text replay, because either can change behavior and
rankings.

- Framework handoff: if the app uses LangChain, LlamaIndex, or mem0, route to the framework
  toolkit. This takes priority over the direct provider beneath the framework.
- Vector preserve: export id, vector, text, and metadata, then insert into
  `LodeDB.open_vector_store(path, vector_dim=N)`. Preferred for direct pgvector where vectors are
  available and the application already owns embedding.
- Text replay: export id, text, and metadata, then insert through `LodeDB(path=..., model=...)`.
  Useful when vectors are not exportable or the app wants LodeDB to own embeddings going forward.

## pgvector path

For direct pgvector usage the toolkit supports read-only inspection and export for common
schemas:

- It detects vector column types such as `vector(1536)`, the table, and the id / text / metadata
  columns, auto-detecting common names (`id`/`uuid`/`custom_id`, `content`/`document`/`text`,
  `embedding`/`vector`, `metadata`/`cmetadata`) and accepting overrides:
  `--id-column`, `--text-column`, `--vector-column`, `--metadata-column`.
- It reads the dimension from the declared column type, refusing to migrate when the dimension is
  missing or inconsistent; a row whose vector width disagrees is skipped with a recorded reason
  rather than corrupting the index.
- It exports rows in stable primary-key order in fixed batches, so a long export is stable even
  under concurrent application writes.
- The connection string is redacted from every log, plan, manifest, and generated patch.

## Scope in this release

- The direct provider supported end to end is **pgvector**. Qdrant, Chroma, LanceDB, sqlite-vec,
  and FAISS are detected by `inspect` and reported, but their direct exporters are tracked as
  follow-up provider modules. Until then, those projects that wire the store through a framework
  still migrate today through the framework path.
- The framework path (LangChain, LlamaIndex, mem0) is documented on the
  [migrate-agent page](https://egoistmachines.com/lodedb/migrate-agent).
