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
uv run python benchmarks/late_interaction/run.py --docs 2000 --queries 200 --storage float16
```

## Result

The quality case holds decisively, at low query latency. On 800 synthetic pages
(64 patches each, dim 128), late interaction returns the exact MaxSim ranking that
mean-pooling cannot:

| metric | late interaction (float32) | single vector (pooled) |
|---|---:|---:|
| recall@10 vs exact MaxSim | **1.00** | 0.05 |
| mean query latency | 1.5 ms | 0.2 ms |
| resident build (one-time) | 0.05 s | -- |
| ingest (800 docs) | 0.57 s | 0.23 s |
| on-disk footprint | 35 MB | 0.3 MB |

Pooling a page to one vector destroys the per-patch signal MaxSim depends on, so
its recall against the MaxSim ranking is near zero regardless of tuning. Late
interaction holds every patch in one in-memory matrix and scores the whole corpus
with a single GEMM plus a segmented max -- no candidate-recall loss, no
per-candidate read-back.

### Storage precision (`--storage`)

Each document's patch matrix is stored at a chosen precision. All three return the
exact-MaxSim ranking on this set except int8, which is within ~2%; float32 is the
default for its query speed, with float16 / int8 trading a little for footprint:

| storage | on-disk | ingest | query | recall@10 |
|---|---:|---:|---:|---:|
| **float32** (default) | 35 MB | 0.67 s | 1.5 ms | 1.000 |
| float16 | 18 MB | 0.48 s | 3.6 ms | 1.000 |
| int8 | 9 MB | 0.37 s | 2.7 ms | 0.984 |

### How it got here

The first prototype stored one engine row per patch and ran a two-stage
quantized-candidate-then-rescore query: ~110 ms/query at recall 0.73, and ~127 MB
on disk. A per-stage profile showed scoring was ~1% of the query; candidate
generation (~61%) and patch read-back (~38%) dominated. Three changes removed all
of it:

- **Resident exact scan** (default, unfiltered, within `resident_max_bytes`):
  scores the whole corpus from one in-memory matrix -- no candidate scan, no
  read-back. ~110 ms -> a few ms, recall 0.73 -> 1.0.
- **One row per document** (the whole patch matrix in a single row): 39-54x faster
  ingest and far fewer rows; filtered queries score the matching subset
  exhaustively, and corpora over the resident budget stream from disk (exact,
  constant memory).
- **Reduced-precision storage** (optional, `storage="float16"`/`"int8"`): 7-14x
  smaller on disk and up to 2x more pages resident in RAM, at recall 1.0 (float16)
  / ~0.98 (int8). The default `float32` keeps the fastest query and bit-exactness.

All output is metrics-only: counts, bytes, latency, recall; never vectors.
