# AGENTS.md

Guidance for humans and AI coding agents working in the LodeDB repository.

## What this is

LodeDB is an open-source (Apache-2.0), local-first embedded vector database. The public
surface is `import lodedb` plus the `lodedb` CLI; everything lives under `src/lodedb/`.

```
src/lodedb/__init__.py   public API: LodeDB, LodeSearchHit, the CLI
src/lodedb/local/        local-first SDK: db, CLI, MPS/CPU embedding selector, dev
                         server, MCP server, LangChain adapter
src/lodedb/engine/       engine core: index / storage / embedding / route profiles
                         (LodeEngine in core.py, LodeIndex in index.py)
third_party/turbovec/    vendored MIT compact vector core (preserve LICENSE)
tests/                   local SDK suite + import-boundary guard
```

## Hard rules (non-negotiable)

- **Privacy / metrics-only:** telemetry and the redacted artifacts (`.json` snapshot, `.jsd`
  journal, `.tvim`/`.tvd` sidecars, audit log) never carry raw documents, queries, chunks,
  embeddings, or credentials — only counts / bytes / latency / ids / timestamps. The one
  exception is original document text, retained by default in a *separate* `.tvtext` sidecar
  for `db.get(id)`; pass `store_text=False` to keep no text on disk. No telemetry/redacted
  path reads that sidecar.
- **Lean dependencies:** runtime PyPI deps are `numpy`, `typer`,
  `sentence-transformers`, `pyyaml` (+ extras `[mcp]`, `[langchain]`, `[gpu]`). The patched
  TurboVec core is vendored under `third_party/turbovec/` and bundled into the wheel as
  `lodedb._turbovec` — not a PyPI dependency.
  Importing LodeDB must **not** load `faiss` / `modal` / `mteb` / `datasets` / `matplotlib`
  / `sklearn` — `tests/test_import_boundary.py` fails closed (fresh subprocess) if it ever
  does. (`scikit-learn` may be *installed* transitively via `sentence-transformers`; the
  guard checks what *loads*, not what is installed.)
- **Licensing:** our code is Apache-2.0; preserve the upstream **MIT** notice on
  `third_party/turbovec/` (see `NOTICE`). No GPL/AGPL/LGPL dependencies.
- **Local path stays local:** the local SDK runs in a no-auth, loopback, CPU-scan profile.
  Don't add auth, network, or server requirements to it. The CUDA scan (`[gpu]`) is an
  opt-in extra and must stay lazy — importing LodeDB never requires CuPy.
- **Concurrency invariants:** a writable handle holds the cross-process single-writer lock
  (`engine/_filelock.py`) for its lifetime; a `read_only=True` handle takes **no** lock and
  must never persist (`_persist_state` enforces this). Public `LodeEngine` operations run
  under one reentrant in-process lock (`@_synchronized`) so the threaded `serve` can share a
  handle. Persist every file via `durable_replace` (temp + `os.replace`, with the fsync gated
  on `durability="fsync"`) — never write a persisted artifact in place.

## Develop

```bash
uv sync --extra dev --extra mcp --extra langchain   # build venv (compiles TurboVec)
uv run pytest -q                                                # run the suite
uv run ruff check .                                             # lint (line-length 100)
```

The vendored TurboVec crate is built from source by `uv sync` (maturin / pyo3). It links a
CBLAS provider: the Accelerate framework on macOS, OpenBLAS on Linux (`apt-get install
libopenblas-dev`).
