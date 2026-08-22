# Managed cloud roadmap

This document maps a path from LodeDB, an embedded local-first vector database, to an
optional managed cloud companion, following the model DuckDB established with MotherDuck:
the local engine stays useful and primary on its own, and the cloud is an opt-in companion
that adds durable sync, managed serving, sharing, and scale. It is a direction, not a
commitment. The near-term, low-risk slice is called out under "Scope of investment", and
the managed-service milestones are explicitly demand-gated.

## North star

Lode Cloud is a companion, not a replacement.

1. Local LodeDB stays offline-first, in-process, and no-auth by default. `import lodedb`
   never loads auth or network code.
2. Cloud mode is opt-in through a separate package extra, a separate client class, and a
   separate CLI command group.
3. The first managed capabilities are durable sync and backup, then managed read-only
   serving of committed indexes.
4. Hybrid local/cloud query routing comes last, and it is retrieval routing plus result
   fusion, not relational operator splitting.

### The MotherDuck model and the honest translation

MotherDuck is not "DuckDB on a server". It keeps the embedded engine authoritative on the
client and adds, in the cloud, a service layer (identity, authz, sharing), per-user
isolated compute, object-storage-backed durable state with copy-on-write differential
storage (which is what enables zero-copy clone, branch, and time travel), and a hybrid
planner that can split one SQL query across local and cloud. A client opts in with one line
(`ATTACH 'md:'`) authenticated by a token.

The translation to LodeDB is mostly direct, with one deliberate exception:

- A local client attaching a cloud database becomes a LodeDB handle syncing a committed
  generation to and from a managed remote.
- Differential, copy-on-write storage maps onto a capability LodeDB already has. The
  on-disk format is generation-addressed: committed state lives in immutable,
  content-checksummed `g<epoch>.*` artifacts under `<key>.gen/`, pinned by a single atomic
  root manifest, `<key>.commit.json`. Cloud storage becomes "ship the checksummed artifacts
  to object storage and publish the manifest", not a new storage engine.
- Per-user isolated compute maps onto a per-tenant engine instance serving that tenant's
  path.
- The service layer maps onto a new authenticated service, the one component LodeDB
  deliberately does not have today.
- Hybrid SQL execution does not map. LodeDB has no relational optimizer and no query plan to
  split; a single exact nearest-neighbour scan over one index is not partitionable the way a
  SQL join is, so copying bridge operators would be overbuilding. The honest analogue is
  routing a whole query to local or cloud, or running it on both and fusing the results.

## Invariants the cloud must preserve

LodeDB's identity rests on properties enforced in code today. The cloud companion preserves
all four on the client and surfaces them as deliberate, opt-in choices on the server.

1. **Loopback or private bind only.** `is_private_bind_host` (`engine/core.py`) rejects any
   public bind, and the local HTTP server (`local/server.py`) refuses a non-private host.
   The embedded engine is never exposed publicly. The cloud serving tier is a separate,
   authenticated service that fronts its own local engine; the bind guard is never relaxed.
2. **Payload boundary.** Redacted artifacts (`.json`, `.jsd`, `.tvim`, `.tvd`) never carry
   text. Raw text lives only in the opt-in `.tvtext` base plus `.txd` journal, and lexical
   terms in the opt-in `.tvlex` base plus `.lxd` journal (`local/db.py`,
   `engine/_commit_manifest.py`). Anything that leaves the machine respects this: text and
   lexical sync are explicit per-store opt-ins, off by default, and control-plane telemetry
   stays metrics-only.
3. **Single-writer, crash-atomic commits.** One writer per path via an OS advisory lock
   (`engine/_filelock.py`); every commit is sealed by the atomic `<key>.commit.json` swap,
   which is the only thing that commits a generation; recovery rolls back to the last
   committed generation (`engine/core.py`). The cloud extends this to one writer per remote
   path via a lease, it does not replace it.
4. **Lean core.** Runtime dependencies are numpy, typer, sentence-transformers, and pyyaml,
   and an import-boundary test (`tests/test_import_boundary.py`) guards against growth. All
   cloud code lives behind a new optional extra and a new top-level package, importing
   nothing into the embedded path.

## Core idea: reuse the on-disk format as the cloud substrate

The single most important decision is to treat the existing generation-addressed commit
format as the differential-storage layer rather than inventing a second one. It already
provides what a cloud durable store needs:

- **Immutability.** Bases are epoch-addressed and never overwritten in place, so every
  committed generation is a stable snapshot.
- **Content addressing.** Artifacts and the manifest body are sha256-checksummed
  (`body_sha256`), so blobs deduplicate by checksum and integrity is verifiable on restore.
- **Atomic commit.** Publishing one new manifest pointer commits a whole generation; readers
  see old-then-new, never torn.
- **O(changed) updates.** Writes append delta journals onto a fixed base epoch, so a cloud
  push uploads only new segments or a new base, inheriting the engine's existing
  incremental-commit property.

Clone, branch, and time travel then fall out as catalog operations over immutable
generations rather than new storage machinery.

One coupling to manage: the on-disk manifest schema (`COMMIT_MANIFEST_SCHEMA_VERSION`) and
the cloud wire protocol must be versioned independently, so a future on-disk bump does not
break deployed clients, and a cloud that retains generations long-term must keep reading
older format versions as the engine evolves.

## Milestones

### Milestone 0: Boundaries and this document

Land this roadmap as `docs/cloud-roadmap.md`, linked from the README enterprise note and the
architecture overview. No cloud runtime dependencies and no code. Fix the cloud user stories
(back up a local index, query a managed copy without holding a local writer, share read-only
snapshots, run GPU batch serving, BYOC later) and the privacy policy (text and lexical
artifacts are payload-bearing and require explicit policy). This is the content of the first
pull request.

### Milestone 1: Artifact-store abstraction and manifest export/import

Introduce a storage abstraction for committed artifacts with the local filesystem as the
default implementation, modelled on the existing root pointer and generation artifacts (no
new format). Interface shape:

```python
class ArtifactStore:
    def read_bytes(self, name: str) -> bytes: ...
    def write_bytes_if_absent(self, name: str, data: bytes, sha256: str) -> None: ...
    def compare_and_swap_pointer(self, key: str, old_generation: int | None, new_body: dict) -> None: ...
```

Add a read-only generation-inventory helper that, given a path and index key, lists every
artifact the live `<key>.commit.json` references with its checksum, size, and kind,
distinguishing a new delta segment on an existing base epoch from a new base epoch (so push
stays O(changed)). Add a manifest export/import that packages a committed generation without
reading uncommitted WAL tails, so only committed state ships.

- Modules: `engine/artifact_store.py` (interface), `engine/local_artifact_store.py`
  (default), and a read-only inventory helper. The only engine-side change is read-only.
- Decision: object-storage backends (S3, GCS, Azure) live in the optional cloud extra or a
  cloud-service package, never in the open-core runtime dependencies.
- Risk: object stores have no `os.replace`. The root-pointer commit needs a
  strongly-consistent conditional write or a metadata-service compare-and-swap, never
  eventually-consistent list operations.

### Milestone 2: Backup and restore (self-hosted or BYOC object store)

The first real cloud feature, and the de-risking one. Upload the current committed
generation to object storage, and restore a generation into a local directory, verifying
manifest checksums before accepting it. Restore reuses the engine recovery path, so a torn
restore self-heals to the last good generation.

- Text policy: the default uploads only redacted artifacts; `--include-text` adds
  `.tvtext`/`.txd`; `--include-lexical` adds `.tvlex`/`.lxd` (lexical terms are
  payload-derived and sensitive).
- CLI: `lodedb cloud push`, `lodedb cloud pull`, `lodedb cloud status` (compare local and
  remote generation numbers and counts).
- This proves cloud durability without changing writer semantics. Combined with Milestone 1
  it is shippable as self-hosted or BYOC sync before any managed service exists.

### Milestone 3: Control plane

The service layer LodeDB lacks: a separate authenticated service, outside the engine path,
providing identity, orgs, projects, and tokens; a catalog of databases, indexes, snapshots,
owners, and share grants; generation-pointer metadata with the strongly-consistent
compare-and-swap from Milestone 1; a single-writer lease per index; and a metrics-only audit
log.

- Client: a separate `lodedb.cloud` package and a `LodeDBCloud` client, plus a `lodedb cloud`
  CLI group (login, sync, pull, status, share, serve). Cloud auth stays out of `LodeDB(...)`.
  A thin ergonomic facade may lazily import the cloud client only when a remote is requested,
  so the import boundary holds while integrations still get a single handle.
- Metadata DB: a strongly-transactional store (Postgres class) for the managed service. The
  interface stays abstract so a local single-node implementation serves development, tests,
  and self-host.
- Risk: this is the first networked auth surface and needs a security review. Tokens and
  control-plane metadata must never contain documents, queries, vectors, embeddings, or
  credentials.

### Milestone 4: Managed read-only serving

A cloud worker materialises a committed generation onto an ephemeral local SSD cache, opens
it with `read_only=True` (which takes no writer lock and loads the exact committed manifest),
and serves search, search_many, get, and stats over authenticated HTTPS. Many concurrent
readers per tenant are safe and lock-free.

- This is a separate authenticated service, never the loopback dev server with auth bolted
  on.
- Raw-text serving is a separate permission: vector search and metadata can be allowed while
  get and inline text are denied.

### Milestone 5: Managed GPU batch serving

The cloud is where GPUs live, so batched search_many on a CUDA worker is the cloud's
throughput advantage. The engine already has a GPU-resident exact scan
(`engine/gpu_turbovec.py`, lazy and optional), with the CPU scan as the source of truth and
fallback. Optimise for batch and concurrency, not single-query latency; preserve exactness
and CPU fallback; keep a per-index resident GPU cache with eviction and a per-tenant quota so
one large index cannot starve a worker.

### Milestone 6: Cloud writes with a single-writer lease

A managed writer per index. The worker acquires a control-plane lease (not a filesystem
lock), opens a writable cached copy, applies add and remove, commits through the existing
generation path, uploads changed artifacts, then advances the cloud root pointer by
compare-and-swap. One active writer per index; the local-writer-plus-push mode (Milestone 2)
and the cloud-writer mode are mutually exclusive per path via the lease.

- WAL caveat: WAL mode does not expose uncheckpointed writes to lock-free readers, so remote
  writes force a generation commit (checkpoint) per request until a cloud reader-invalidation
  protocol exists.
- Server-side embedding uses the same presets; arbitrary user embedding code in workers is
  out of scope initially.

### Milestone 7: Snapshot sharing, attach, clone, branch, time travel

Immutable snapshot IDs for committed generations, and read-only share grants to a snapshot or
to a latest-published pointer. `attach` pulls a shared snapshot into a local read-only path
queried with `open_readonly`; later, remote read-only attach through the cloud client.

- Clone is a new catalog pointer to an existing manifest; branch forks the pointer and
  accepts independent future generations; time travel selects an older generation. All are
  catalog operations because generations are immutable and independently loadable.
- Sharing must not expose raw text or lexical terms unless the owner published those
  artifacts and the grantee has permission.
- Highest correctness risk: blob garbage collection must become reference-counted across
  branches before any remote deletion. The local keep-last-N-epochs rule
  (`DEFAULT_EPOCHS_RETAINED`) does not hold once a generation can be referenced by multiple
  branches; gate remote GC behind catalog-level reference counting.

### Milestone 8: Bidirectional sync and conflict model

Local push and pull tracking base, local head, and remote head. Divergent histories are not
auto-merged: offer explicit `--force-push` and `--force-pull`, with logical add and remove
merge from WAL history as later work. The sync unit is a committed generation, not arbitrary
files. Conservative and explicit by default.

### Milestone 9: Unified handle and hybrid retrieval routing

A client that holds local and remote indexes and routes a query: local only, remote only, or
fan-out to both and fuse. Implemented as a separate cloud client with an optional thin facade
over the existing `LodeDB`, so the adapters (LangChain, LlamaIndex, mem0) and the MCP server
can target a cloud-backed handle, with the cloud client lazily imported.

- The fan-out merge is exact: because scoring is exact cosine and results are
  `(score, id, metadata)`, merging two top-k lists is just a re-sort by score, with no
  approximation, unlike merging approximate-nearest-neighbour shards. Cross-source fusion
  reuses the existing reciprocal-rank-fusion logic.
- Correctness requirement: fused or synced results are only comparable if both ends embed
  with the identical model, version, and normalisation. A synced store carries an implicit
  embedding-model contract; pin and record it, and refuse cross-model fusion.
- This is routing, not operator splitting: a whole query goes to one side, or to both and
  merges. No bridge operators.

### Milestone 10: Hosted MCP and agent-memory service

A hosted, authenticated MCP endpoint backed by cloud indexes, mirroring the local stdio tools
(`lodedb_add`, `lodedb_search`, `lodedb_get` only with text permission, `lodedb_remove`,
`lodedb_stats`). The local stdio MCP is unchanged and stays local-first. Text exposure is
deny-by-default for shared indexes.

### Milestone 11: BYOC and on-prem

Helm or Terraform for the control plane, workers, object storage, and metadata DB;
customer-managed bucket and KMS; an air-gapped mode with no managed-service callbacks. Keep
BYOC operationally close to the managed service to avoid a forked product. This matches the
enterprise note already in the README.

## Sequencing

- Milestone 0, then 1, then 2 form the low-risk core: plumbing over an existing format,
  shippable as self-hosted or BYOC sync with no control plane.
- Milestone 3 is the inflection point where real new surface area and security exposure
  begin.
- Milestones 4, 5, and 6 build managed serving and then managed writes on top of the control
  plane.
- Milestones 7 and 8 add sharing and sync semantics over immutable generations.
- Milestone 9 adds the unified handle and hybrid routing; Milestone 10 the hosted MCP;
  Milestone 11 the BYOC packaging.

## Cross-cutting decisions

- Reuse the generation-addressed format as the sync substrate; do not invent a second storage
  engine.
- Extend, never relax, the four invariants. Cloud code is a separate package and a separate
  process.
- Replace local writer locks with cloud leases, not network filesystems (the file lock is
  documented as unreliable on NFS and SMB).
- Object storage holds immutable artifacts; a strongly-consistent metadata store holds the
  root pointer and performs the commit compare-and-swap.
- The public SDK contract (search, search_many, add, get, the `(score, id, metadata)` shape,
  and open_readonly) is the cloud's API contract, so existing surfaces inherit cloud backing.
- Build remote serving before remote writes, and snapshot sharing before multi-writer
  collaboration.

## Risks

- Atomic root-pointer commit on object storage needs a real compare-and-swap or metadata
  transaction, not `os.replace` and not eventually-consistent listing.
- WAL mode is single-writer and does not expose uncheckpointed writes to lock-free readers;
  cloud writes must checkpoint.
- Uploading `.tvtext` or `.tvlex` changes the privacy posture; both are opt-in and off by
  default.
- Multi-writer merge is not supported by the engine; the lease keeps one writer per index.
- Content addressing deduplicates by sha256; in a multi-tenant store this must be namespaced
  and authorised per tenant, so a known checksum cannot probe for or fetch another tenant's
  blob.
- Hybrid fusion is only correct across an identical embedding model and version; pin and
  record the model with the store.
- Cross-branch blob GC needs reference counting before any deletion.
- Cold starts may be dominated by model loading and index hydration; cache hot generations on
  SSD and keep resident workers.

## Out of scope

- A SQL or relational query optimizer, or MotherDuck-style bridge operators. Cross-source
  result fusion is the analogue.
- Multi-writer or CRDT merge.
- Changes to the TurboVec kernel, the CUDA or MPS scan, or the on-disk format to suit the
  cloud.
- Relaxing the loopback bind or the metrics-only telemetry of the embedded engine.
- Adding auth or network code to `LodeDB(...)` or to the local stdio MCP.
- New mandatory runtime dependencies.
- A browser or WASM client.

## Scope of investment

The architecture is grounded in what the engine already does, but it describes a multi-quarter
program that turns a library into a multi-tenant service with auth, billing, isolation, and
on-call. Treat it in two parts:

- **Near-term, low-risk, and on-brand:** Milestones 0, 1, and 2. These ship self-hosted or
  BYOC durable sync and backup, reuse the existing format, add no mandatory dependencies, and
  match the existing enterprise note without committing to running a managed service.
- **Demand-gated:** Milestone 3 onward (the control plane and everything above it). Build
  these against real customer demand, since each adds operational and security surface that
  contradicts the no-server positioning if shipped speculatively.

## First pull request and next step

This pull request adds this document and links it from the README enterprise note and the
architecture overview. It adds no code and no dependencies. The first implementation pull
request is Milestone 1: the artifact-store interface and manifest export/import over the
existing commit manifest, with the object-storage backend kept in the optional cloud extra.
