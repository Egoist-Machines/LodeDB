## What & why

Brief summary of the change and the motivation.

## How it was tested

- [ ] `uv run pytest -q`
- [ ] `uv run ruff check .`
- [ ] Manual check (describe):

## Ground rules (see [CONTRIBUTING.md](CONTRIBUTING.md))

- [ ] No raw documents / queries / embeddings / credentials added to logs, telemetry, or the redacted artifacts
- [ ] Runtime deps stayed lean; `tests/test_import_boundary.py` is green (no new heavy imports at `import lodedb`)
- [ ] Licensing preserved (our code Apache-2.0; vendored TurboVec MIT `LICENSE` + top-level `NOTICE` intact)
- [ ] New behavior has tests
