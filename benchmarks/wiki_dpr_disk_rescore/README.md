# wiki_dpr disk-rescore benchmark

This harness reconstructs the vector-only benchmark described in
[issue #79](https://github.com/Egoist-Machines/LodeDB/issues/79). It measures LodeDB against a
shared, durable `kenhktsui/wiki_dpr_e5` vector corpus, not an embedding model. The full corpus is
about 80 GB of parquet input: 21,015,300 rows of normalized 768-dimensional fp32 vectors.

The benchmark reports index-fidelity recall@100 against prepared exact fp32 top-100 ground truth,
sequential single-query latency, closed-loop concurrency-4 QPS and request latency, batched QPS,
and store footprint. It has exact 4-bit scan and cluster-prune ANN configurations today. The
rescore dtype, oversample, compact-layout, open-time nprobe, and block-skip measurements have
CLI support for `feat/cluster-layout-rescore`; selecting an unavailable feature exits with a clear
engine-branch requirement.

## Dataset convention

`data_prep.py` streams Parquet batches into a `base.f32` memmap. It samples queries as corpus row
indices. A sampled vector stays in the corpus, so its own row is present in its exact top-100
ground truth. This self-retrieval convention is intentional and applies to every configuration.
The preparation manifest captures row count, dimension, seed, selected row indices, source shard
names and sizes, and artifact file names. No document text, query text, or embeddings are written
to result JSON.

The top-100 reference uses a blocked fp32 matrix multiply over the persisted memmap. It is created
once during preparation, then the serving harness only reads it. This keeps recall independent of
the engine being measured.

## Local M5 repro, 1M rows

Install the normal development environment, obtain the dataset shards separately, then run:

```sh
.venv/bin/python benchmarks/wiki_dpr_disk_rescore/data_prep.py \
  --shards-dir /path/to/wiki_dpr_e5/data \
  --out /tmp/wiki-dpr-1m --target-rows 1000000 --n-queries 1000 --seed 42

.venv/bin/python benchmarks/wiki_dpr_disk_rescore/lodedb_bench.py \
  --data /tmp/wiki-dpr-1m --store /tmp/wiki-dpr-stores/exact_bw4 \
  --loop-seconds 20 --out benchmarks/wiki_dpr_disk_rescore/results/exact_bw4.json

.venv/bin/python benchmarks/wiki_dpr_disk_rescore/lodedb_bench.py \
  --data /tmp/wiki-dpr-1m --store /tmp/wiki-dpr-stores/ann1000_np16 \
  --ann-clusters 1000 --ann-nprobe 16 --loop-seconds 20 \
  --out benchmarks/wiki_dpr_disk_rescore/results/ann1000_np16.json

.venv/bin/python benchmarks/wiki_dpr_disk_rescore/report.py \
  --results benchmarks/wiki_dpr_disk_rescore/results
```

The issue's Apple M5 baseline for later parity work was 1M rows, 4-bit exact scan: recall 0.9401,
22 ms sequential latency, 45.9 closed-loop concurrency-4 QPS, and 329 batch QPS. Its
`clusters=1000, nprobe=16` ANN point was recall 0.877, 12.5 ms sequential latency, and 76.9 QPS.
Those numbers are historical issue context, not output from this reconstructed harness.

## Modal repro, full 21M rows

The Modal image is CPU-only and compiles the repository workspace with OpenBLAS, matching the
Linux TurboVec build requirement. It stores source shards, prepared arrays, and built stores in the
`wiki-dpr-e5` volume. It copies each store to container-local disk before serving and records that
copy time separately in the returned JSON.

```sh
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::download
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::prepare --target-rows 21015300
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::ground_truth --target-rows 21015300

modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::build \
  --labels exact_bw4,ann1000,ann4096 --target-rows 21015300
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::main \
  --labels exact_bw4,ann1000_np16,ann4096_np64 --target-rows 21015300
```

For a cheap end-to-end check that does not download from Hugging Face, use:

```sh
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::smoke
```

The 21M serving path currently needs about 128 GB RAM because opening the store materializes the
vectors. The Modal serving function uses 4 CPUs for parity with the published Qdrant nodes; its
memory allocation is intentionally larger than that CPU count suggests.

## Measurement notes

Sequential latency warms ten queries, then runs every prepared query once. Closed-loop throughput
uses one shared LodeDB handle and N worker threads. Each thread loops over a shuffled query stream
for the configured duration, and a request is measured from immediately before `search_by_vector`
until it returns. It does not use an open-loop request queue. Batched QPS uses
`search_many_by_vector_arrays` for batches of 64, 256, and 1000 vectors.

Every distinct ANN cluster layout gets its own store directory because clustering is a create-time
choice. Reopen-time nprobe and rescore oversample values are session overrides, so their serve
configurations reuse that durable store. The first ANN query is separately timed as cluster-index
construction, not included in sequential latency.

`report.py` includes two fixed published comparison references, labeled not rerun: Qdrant at 111.9
aggregate QPS, 37.3 per node, and 0.9596 recall on three 4vCPU/16GB nodes; Elastic DiskBBQ at 32.4
aggregate QPS, 10.8 per node, and 0.96 recall on three 7vCPU/26GB nodes.

## Results policy

Commit only metrics-only JSON to `results/`: counts, timings, rates, recall, store bytes, and
environment provenance. Do not commit prepared vectors, Parquet shards, stores, document text,
query vectors, or raw query data.
