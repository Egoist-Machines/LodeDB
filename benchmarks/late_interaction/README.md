# Late-interaction (MaxSim) vs single-vector page embeddings

Stage 2 of [issue #25](https://github.com/Egoist-Machines/LodeDB/issues/25):
does late-interaction retrieval (`LodeLateInteractionIndex`, MaxSim over a set of
patch vectors per page) recover the true ranking better than one mean-pooled
vector per page, and at what cost?

Both indexes are fed the **same** synthetic multi-vector documents, so this
measures storage and scan, not an encoder. Ground truth is the exact brute-force
MaxSim top-k over every document's full-precision patches (the metric ColPali /
ColQwen optimise), and both indexes are scored against it. Real ViDoRe numbers
need a bring-your-own ColPali encoder; the harness takes the same shape on real
embeddings.

```bash
uv run python benchmarks/late_interaction/run.py
uv run python benchmarks/late_interaction/run.py --docs 2000 --queries 200 --candidate-depth 32
```

## Result

The quality case holds decisively, and with the resident exact scan it comes at
low query latency. On 800 synthetic pages (64 patches each, dim 128):

| metric | late interaction | single vector (pooled) |
|---|---:|---:|
| recall@10 vs exact MaxSim | **1.00** | 0.05 |
| mean query latency | 1.5 ms | 0.13 ms |
| resident build (one-time) | 0.34 s | -- |
| ingest (800 docs) | 25.9 s | 0.23 s |
| on-disk footprint | 127 MB | 0.3 MB |

Pooling a page to one vector destroys the per-patch signal MaxSim depends on, so
its recall against the MaxSim ranking is near zero regardless of tuning. Late
interaction returns the exact MaxSim ranking (recall 1.0): the default search
path holds every patch in one in-memory matrix and scores the whole corpus with a
single GEMM plus a segmented max, so there is no candidate-recall loss and no
per-candidate read-back.

### How the query path got here

An earlier two-stage path (per-query-token quantized scan to gather candidates,
then read each candidate's patches back and rescore) ran ~110 ms/query at
recall 0.73. A per-stage profile showed scoring was ~1% of that; candidate
generation (~61%) and patch loading (~38%) dominated. Two changes removed both:

- **Resident exact scan** (default for unfiltered queries within a memory budget):
  skips candidate generation and read-back entirely -- ~110 ms -> ~1.5 ms, and
  exact (recall 0.73 -> 1.0). One-time build cost is reported as
  `resident_build_seconds`.
- The two-stage **indexed** path (used for filtered queries, or corpora over the
  resident budget) was also sped up ~2x by dropping a redundant patch-row filter
  and batching the candidate read.

## What's left for stage 3

Query latency is handled. The remaining lever is **footprint**: the index is
~400x the single-vector store (one row per patch plus the full-precision sidecar
the exact scan reads back), and the resident matrix is full-precision in memory.
Native quantized multi-vector storage in the TurboVec core (plus patch pooling on
the encoder side) is what would shrink both, and would extend the fast exact path
to corpora past the resident budget. The native MaxSim kernel for scoring is
already in the core (`scoring="native"`).

All output is metrics-only: counts, bytes, latency, recall; never vectors.
