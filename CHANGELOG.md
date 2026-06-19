# Changelog

All notable changes to LodeDB are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes yet._

## [0.1.0] - 2026-06-19

First public release.

### Added

- **Local-first, privacy-by-default vector database.** Embeds into your process; your
  data never leaves the machine. No accounts, no network calls on the core path.
- **Compact storage** via the bundled, patched **TurboVec** core (2/4-bit codes), shipped
  as the `lodedb._turbovec` extension. Wheels are `abi3`, so one wheel covers all supported
  Python versions on a platform and there is nothing to compile on install.
- **Local embeddings** through `sentence-transformers` on CPU, CUDA, or Apple MPS, with
  built-in model presets (`LOCAL_MODEL_PRESETS`).
- **Delta persistence**: on-disk index (`.tvim` / `.tvd`) plus a journal (`.jsd`) for fast
  incremental updates, and an optional `.tvtext` raw-text sidecar gated by `store_text`
  (default `True`; set `store_text=False` to keep no document text on disk).
- **Python API** — `LodeDB` with `add`, `search`, `search_many`, `get`, and `persist`;
  results returned as `LodeSearchHit`.
- **`lodedb` CLI** — `doctor`, `index`, `query`, `get`, `benchmark`, `serve`, `mcp`.
- **Local HTTP dev server** (`lodedb serve`) — loopback-only, no auth, metrics-only
  telemetry; exposes `/healthz`, `/stats`, `/search`, and `/get`.
- **Optional extras**:
  - `[gpu]` — opt-in CUDA-resident exact scan (cupy; Linux/CUDA only).
  - `[mcp]` — stdio MCP server so coding agents can use LodeDB as local memory.
  - `[langchain]` — LangChain `VectorStore` adapter.

### Notes

- Runtime PyPI dependencies are kept lean: `numpy`, `typer`, `sentence-transformers`,
  `pyyaml`. Heavier research dependencies are not imported on the core path; this is
  enforced by `tests/test_import_boundary.py`.
- LodeDB is licensed under Apache-2.0. The vendored TurboVec core
  (`third_party/turbovec/`) is MIT — see [`NOTICE`](NOTICE).

[Unreleased]: https://github.com/Egoist-Machines/LodeDB/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Egoist-Machines/LodeDB/releases/tag/v0.1.0
