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

The quality case holds decisively. On 800 synthetic pages (64 patches each,
dim 128), late interaction recovers the exact-MaxSim ranking that mean-pooling
cannot:

| metric | late interaction | single vector (pooled) |
|---|---:|---:|
| recall@10 vs exact MaxSim | **0.73** | 0.05 |
| mean query latency | 114 ms | 0.15 ms |
| ingest (800 docs) | 30.1 s | 0.29 s |
| on-disk footprint | 127 MB | 0.3 MB |

Pooling a page to one vector destroys the per-patch signal MaxSim depends on, so
its recall against the MaxSim ranking is near zero regardless of tuning. Late
interaction's recall is bounded by candidate generation, not by scoring -- the
rescore is exact, so deeper candidate scans converge to perfect recall:

| candidate_depth | recall@10 | mean query ms |
|---:|---:|---:|
| 8 | 0.66 | 77 |
| 32 | 0.97 | 202 |
| 96 | 1.00 | 171 |

(400 docs, 100 queries. Pooled recall stays ~0.06 across all depths.)

## What this says about stage 3

The quality case for late interaction is clear. The cost is the open problem, and
it is exactly what native support would address:

- **Latency** is 100-200 ms because candidate generation and the MaxSim rescore
  run per query in Python over many patch rows. A native multi-vector store with a
  MaxSim kernel removes the per-candidate Python loop and the row-at-a-time scan.
- **Footprint** is ~400x the single-vector index: one stored row per patch, plus
  the float32 patch sidecar the exact rescore reads back. Native quantized
  multi-vector storage (and patch pooling on the encoder side) is what keeps the
  index compact, which is otherwise the project's main advantage.

So stage 3 (native multi-vector storage + a MaxSim scoring kernel in the TurboVec
core) is warranted: it targets the latency and footprint this prototype trades
away, while the prototype already validates the retrieval quality and the API.

All output is metrics-only: counts, bytes, latency, recall; never vectors.
