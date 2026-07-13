# wiki_dpr disk-rescore benchmark

This harness reconstructs the vector-only benchmark described in
[issue #79](https://github.com/Egoist-Machines/LodeDB/issues/79). It measures LodeDB against a
shared, durable `kenhktsui/wiki_dpr_e5` vector corpus, not an embedding model. The full corpus is
about 80 GB of parquet input: 21,015,300 rows of normalized 768-dimensional fp32 vectors.

The benchmark reports index-fidelity recall@100 against prepared exact fp32 top-100 ground truth,
sequential single-query latency, closed-loop concurrency-4 QPS and request latency, batched QPS,
and store footprint. It has exact 4-bit scan, cluster-prune ANN, original-precision rescore,
compact-layout, open-time nprobe, and block-skip configurations.

## Dataset convention

`data_prep.py` streams Parquet batches into a `base.f32` memmap. It samples queries as corpus row
indices. A sampled vector stays in the corpus, so its own row is present in its exact top-100
ground truth. This self-retrieval convention is intentional and applies to every configuration.
Preparation requires an immutable source commit digest. The version-2 manifest commits the source
revision, row count, dimension, seed, selected row indices, artifact SHA-256 values, a corpus ID,
and an evaluation ID covering queries plus ground truth. Corpus, query, and ground-truth payloads
use generation-addressed names; `manifest.json` is atomically replaced only after every new payload
is complete, so a crash cannot make a prior manifest point at a truncated replacement. No document
text, query text, or embeddings are written to result JSON.

The top-100 reference uses a blocked fp32 matrix multiply over the persisted memmap. It is created
once during preparation, then the serving harness only reads it. This keeps recall independent of
the engine being measured.

## Local M5 repro, 1M rows

Install the normal development environment, obtain the dataset shards separately, then run:

Benchmark-only preparation/remote dependencies are intentionally not base runtime dependencies:

```sh
uv pip install pyarrow huggingface-hub modal
```

```sh
.venv/bin/python benchmarks/wiki_dpr_disk_rescore/data_prep.py \
  --shards-dir /path/to/wiki_dpr_e5/data \
  --source-revision <40-character-dataset-commit> \
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

Historical issue numbers and results created before the version-2 provenance contract are context,
not output accepted by this harness. They are intentionally excluded from generated parity tables.

## Modal repro, full 21M rows

The Modal image is CPU-only and compiles the repository workspace with OpenBLAS, matching the
Linux TurboVec build requirement. It stores source shards, prepared arrays, and built stores in the
`wiki-dpr-e5` volume. It copies each store to container-local disk before serving and records that
copy time separately in the returned JSON.

```sh
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::download \
  --revision <40-character-dataset-commit>
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::prepare \
  --target-rows 21015300 --source-revision <same-dataset-commit>
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::ground_truth --target-rows 21015300

modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::build \
  --labels exact_bw4,ann1000,ann4096 --target-rows 21015300
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::main \
  --labels exact_bw4,ann1000_np16,ann4096_np64 --target-rows 21015300
```

For a cheap end-to-end check that does not download from Hugging Face, use:

```sh
modal run benchmarks/wiki_dpr_disk_rescore/modal_bench.py::smoke \
  --git-sha "$(git rev-parse HEAD)"
```

The 21M serving path currently needs about 128 GB RAM because opening the store materializes the
vectors. The Modal serving function fixes serving at 4 CPUs for repeatability; its memory allocation
is intentionally larger than that CPU count suggests. CPU count alone does not make an external
system result comparable.

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

The pre-change physical-layout control is marked historical and cannot be built by current code.
It is served only when its store config proves builder commit
`5e54fa53f51986268eee8b77712ac488e7b9aa97` and layout ID
`cluster-insertion-order-v0`; otherwise the sweep skips it. `report.py` shows corpus rows, query
count, evaluation ID, requested concurrency, and actual measurement duration. It refuses legacy
JSON and warns rather than presenting a parity table when populations or evaluation IDs differ.
External published systems are omitted unless rerun on the exact same corpus and query set.

## Results policy

Commit only schema-version-2 metrics JSON to `results/`: identities, counts, timings, rates, recall,
store bytes, and environment/build provenance. Resume validates dataset ID, store ID, builder/layout
identity, serve overrides, query count, loop duration, and concurrency; a mismatched file fails
closed instead of being kept. Results are atomically published. Do not commit prepared vectors,
Parquet shards, stores, document text, query vectors, or raw query data.
