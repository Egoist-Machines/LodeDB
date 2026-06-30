# Contributing to LodeDB

Bug reports, feature requests, and pull requests are welcome at
<https://github.com/Egoist-Machines/LodeDB/issues>.

## Development setup

LodeDB uses [uv](https://docs.astral.sh/uv/). The vendored TurboVec crate is built from
source on first sync (maturin / pyo3), so you need a Rust toolchain and a CBLAS provider:
the Accelerate framework on macOS, OpenBLAS on Linux (`sudo apt-get install libopenblas-dev`).

```bash
uv sync --extra dev --extra embeddings --extra torch --extra mcp --extra langchain   # build the venv
uv run pytest -q                                                # run the test suite
uv run ruff check .                                             # lint (line length 100)
uv run ruff format .                                            # format
```

## Conventions

- **Style:** ruff with `E, F, I, UP, B` selected, line length 100. Run `ruff check` and
  `ruff format` before opening a PR.
- **Types:** the package ships inline type hints (`py.typed`); keep new code annotated.
- **Tests:** add tests under `tests/` for new behavior, and keep
  `tests/test_import_boundary.py` green.

## Ground rules

These are load-bearing — PRs that break them won't be merged:

- **Privacy.** Telemetry and the redacted artifacts (the `.json` snapshot, `.jsd` journal,
  `.tvim`/`.tvd` sidecars, and audit log) never carry raw documents, queries, chunks,
  embeddings, or credentials — only counts, bytes, latency, ids, and timestamps. The one
  exception is the original document text, which is retained by default in a *separate*
  `.tvtext` sidecar so `db.get(id)` can return it; pass `store_text=False` to keep no text
  on disk. No telemetry/redacted path ever reads that sidecar.
- **Lean dependencies.** The base runtime PyPI set is `numpy`, `typer`, `pyyaml` — a
  dependency-light vector store; the patched TurboVec core is vendored and bundled into the
  wheel as `lodedb._turbovec`, not a PyPI dependency. Built-in text embedding is opt-in:
  `[embeddings]` (`onnxruntime` + `transformers`) and `[torch]` (`sentence-transformers`), with
  Optimum in `[onnx-export]`. The embedding runtimes and every heavier dep load lazily:
  importing `lodedb` must not pull `onnxruntime`, `transformers`, `sentence-transformers`,
  `torch`, `faiss`, `modal`, `matplotlib`, and friends — even when installed
  (`tests/test_import_boundary.py` enforces this).
- **Licensing.** LodeDB is Apache-2.0. The vendored TurboVec core under
  `third_party/turbovec/` is MIT — preserve its `LICENSE` and the top-level `NOTICE`. No
  GPL/AGPL/LGPL dependencies.
- **Local stays local.** The local path is no-auth, loopback/private-network, CPU-scan.
  Private-network binds are for trusted LANs only; do not expose `serve` or `mcp` to public or
  untrusted networks without an explicit auth layer. The CUDA scan is an opt-in `[gpu]` extra
  and stays lazy.
