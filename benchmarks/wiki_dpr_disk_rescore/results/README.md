# Results

Only schema-version-2 result JSON belongs here. Each file must pass
`lodedb_bench.validate_result_schema` and bind an immutable dataset evaluation ID, store ID,
builder commit, physical-layout ID, query count, requested loop duration, concurrency, and serve
overrides.

The earlier 1M result files were removed because they predated this provenance contract. In
particular, their dataset artifacts and pre-change layout store could not be distinguished from
same-shaped replacements. Rerun the harness to publish comparable results; do not copy legacy
numbers back into this directory.
