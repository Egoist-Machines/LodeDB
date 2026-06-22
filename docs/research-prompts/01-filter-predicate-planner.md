# Research prompt: a filter-predicate planner with posting/secondary-index pushdown

## Context

You are working in the LodeDB repo (a local-first, embedded, exact vector
database; Apache-2.0 Python SDK over a vendored Rust TurboVec core). At v0.1.2 the
metadata filter supports a Mongo-style predicate grammar —
`$eq/$ne/$gt/$gte/$lt/$lte/$in/$nin/$exists` plus `$and/$or/$not` — implemented in
`src/lodedb/engine/_predicate.py` (validation + a compiled per-document matcher).

Filters are applied during query as an **allowlist pushed into the TurboVec scan**
(see `LodeDB.search`'s docstring and the batch-allowlist work in PR #10). The
allowlist is resolved from a generation-keyed inverted index, `_MetadataPostingIndex`
in `src/lodedb/engine/core.py` (search for the class and its `allowlist(...)`
usage). Exact `(key, value)` postings make `$eq` / `$in` resolution O(matches).

This matters now because the new graph/knowledge-graph layer (`src/lodedb/graph/`)
and agent-memory use cases lean hard on filters that are **not** plain equality:
edge-type traversal (`relation $in [...]`), temporal validity (`created_at $gte T`,
`t_invalid $exists False`), and confidence thresholds (`score $gte 0.7`).

## The problem to investigate

Determine exactly how each predicate operator is resolved today on the query path:
which operators ride the `_MetadataPostingIndex` allowlist (O(matches)) versus which
fall back to evaluating the compiled per-document matcher across the candidate set
(potentially O(corpus)). In particular trace `$gte/$lt/$ne/$nin/$exists/$not` and
the `$and`/`$or` composition. Confirm whether an ordered predicate over a
high-cardinality field degrades filtered search toward a full scan as the corpus
grows.

Then design a **filter planner**: a compile step that turns a validated predicate
tree into an execution plan over the available indexes —

- `$eq`/`$in` → union of posting sets;
- `$and` → intersect child candidate sets; `$or` → union; `$not` → complement vs the
  generation's full id set;
- ordered (`$gt/$gte/$lt/$lte`) and `$ne/$nin/$exists` → either a new **secondary
  index** (e.g. a sorted structure per ordered field, or a presence set for
  `$exists`) or an explicit, costed residual scan over the smallest candidate set;
- always end by pushing the resolved id allowlist into the existing TurboVec scan so
  the batch-allowlist fast path (PR #10) is preserved.

## Deliverable

A written design (and, if tractable, a prototype) covering: the plan representation;
which secondary indexes to add and their build/commit cost; how the planner stays
within LodeDB's invariants; the fallback/cost model when no index helps; and a
migration story for existing on-disk indexes (the format must not break readers).

## Invariants to respect (from AGENTS.md / docs/architecture.md)

- **Payload-free artifacts**: any new index persists only redacted keys/values
  already allowed in metadata — never raw text/vectors.
- **O(changed) commit path**: secondary-index maintenance must stay incremental per
  upsert; do not add an O(corpus) repack to the commit path.
- **Lean dependencies**: prefer stdlib / the existing stack; `_predicate.py` is
  deliberately stdlib-only and must stay importable without `core`.
- Filtered search must keep returning the **true top-k of the matching subset**, not
  a post-filtered slice.

## Success criteria

Filtered `search`/`search_many` latency for ordered/negation predicates becomes
sublinear in corpus size at fixed selectivity (or you show, with numbers, why a
costed residual scan is the right call). Validate with
`benchmarks/graph_memory/` (the `filters` sub-benchmark already measures latency by
predicate kind/selectivity) on Modal A10/L40S at 50k–1M docs, and confirm no
regression on the existing exact-filter and batch-allowlist benchmarks.
