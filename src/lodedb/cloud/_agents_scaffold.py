"""Generators for `lodedb cloud init --agents`: the artifacts that make a consumer
repo one-shot for a coding agent, generated from the repo's REAL link (host,
org, environment, store) rather than placeholders.

Three artifacts, all safe to commit (no secrets — credentials stay in env
vars or ~/.orecloud):

- `.claude/skills/orecloud/SKILL.md` — Agent Skills spec (name + description
  frontmatter, short body); read by Claude Code, Cursor, Codex, and Copilot.
  Ours to regenerate: rewritten on every run so it tracks the link.
- An `## OreCloud` section appended to `AGENTS.md` (created if absent);
  never touched again once the heading exists — the section is theirs to
  edit after the first drop.
- An `orecloud-control` entry in `.mcp.json` pointing at the control-plane
  MCP endpoint with `${ORECLOUD_TOKEN}` env expansion; merged into an
  existing file, skipped if that file is unparseable or already has the
  entry.
"""

from __future__ import annotations

import json
from pathlib import Path

AGENTS_SECTION_HEADING = "## OreCloud"
MCP_SERVER_NAME = "orecloud-control"


def skill_markdown(host: str, org: str, environment: str, store: str) -> str:
    """The .claude/skills/orecloud/SKILL.md body for one linked repo."""
    target = f"{org}/{environment}/{store}"
    return f"""---
name: orecloud
description: Work with this repo's OreCloud-managed LodeDB search store ({target} on {host}). Covers connecting via the Python SDK, asynchronous write semantics (seq / min_seq), semantic search, the lodedb cloud CLI's JSON output and exit codes, and MCP wiring. Use when adding, searching, or syncing documents against OreCloud, or when provisioning stores and API keys.
---

# OreCloud

This repository is linked to a managed OreCloud remote (recorded in
`orecloud.toml` — committable, no secrets):

- host: `{host}`
- target: `{target}`

## Connect (Python SDK)

```python
from lodedb.cloud import Client  # pip install "lodedb[cloud]"

client = Client(token=..., host="{host}")            # an ore_sk_ key carries
store = client.store("{store}")                     # its org/environment
store.add_many([{{"text": "hello", "metadata": {{"kind": "note"}}}}])
hits = store.search("greeting", k=3)
```

Tokens come from the `ORECLOUD_TOKEN` env var (usually the repo's `.env`,
written by `lodedb cloud init --agents`) or `lodedb cloud tokens mint --env-file .env`
— the secret goes straight to the file, never to stdout (scopes: `write`,
`read:search`, `read:text`; use the narrowest set). Mint keys scoped to this
environment (`{environment}`) so the client binds itself; only unscoped
personal tokens pass `org=`/`environment=`. Never hard-code or print a token.

## Write semantics (important)

Writes are asynchronous: `add`/`add_many` return once the write is durably
ACCEPTED (returning the document ids), BEFORE it is searchable — visibility
follows within seconds. The SDK handles read-your-writes automatically: a
search on the same handle waits briefly for that handle's own writes, and
`store.wait_for(store.last_write_id)` blocks until a write is fully applied.
Only when calling the raw HTTP API directly: pass the acceptance's `seq` to
search as `min_seq`, and treat HTTP 425 as "not folded yet — retry after a
moment", never as failure.

## CLI

The `lodedb cloud` CLI prints JSON whenever stdout is not a terminal (force
with `--json`/`--no-json`), so its output is directly parseable:

```bash
lodedb cloud store list                      # registered stores (JSON when piped)
lodedb cloud store create <name>             # register + print a connect snippet
lodedb cloud sync <local-dir> cloud <key>    # converge a local LodeDB with the cloud
lodedb cloud status <local-dir> cloud <key>  # read-only preview of a sync
```

Errors land on stderr as `error: …` plus a `hint: …` line naming the exact
next command, with exit codes by class: 2 usage, 3 auth, 4 not found,
5 refused (quota/conflict — do not retry blindly), 6 transient (retry with
backoff). Confirmation prompts never hang a pipe: destructive verbs need
`--yes` when non-interactive.

## MCP

The store itself is an MCP server (search/add tools):
`{host}/mcp/{target}` with `Authorization: Bearer <key>`.
Provisioning (environments, stores, keys, rollback) is the separate
control-plane MCP server at `{host}/mcp/control` — see `.mcp.json`.

## Boundaries

- Never commit tokens or `~/.orecloud/credentials.json`; `orecloud.toml`
  and this skill are safe to commit.
- A 402 answer means a plan limit was reached; reads keep serving —
  surface the message instead of retrying.
"""


def agents_md_section(host: str, org: str, environment: str, store: str) -> str:
    """The `## OreCloud` section appended to the consumer repo's AGENTS.md."""
    target = f"{org}/{environment}/{store}"
    return f"""{AGENTS_SECTION_HEADING}

This repo uses an OreCloud-managed LodeDB store: `{target}` on `{host}`
(link recorded in `orecloud.toml`; no secrets in either file).

```bash
lodedb cloud status <local-dir> cloud <key>   # what a sync would do (read-only)
lodedb cloud sync <local-dir> cloud <key>     # converge local and cloud
lodedb cloud store list --environment {environment}   # registered stores
```

- Credentials: set `ORECLOUD_TOKEN` (plus `ORECLOUD_HOST` when the control
  plane is not the hosted default), or run `lodedb cloud login --host {host}`
  once. Never commit tokens.
- The CLI prints JSON when piped; errors carry a `hint:` line and per-class
  exit codes (3 auth, 4 not found, 5 refused, 6 transient).
- SDK writes are asynchronous: `add_many` returns the document ids once the
  write is accepted, before it is searchable. The SDK handle does
  read-your-writes automatically; `store.wait_for(store.last_write_id)`
  blocks until the write applies. (Raw HTTP callers pass the acceptance's
  `seq` as `min_seq` and treat HTTP 425 as "retry shortly".)
- Full guide: `.claude/skills/orecloud/SKILL.md` (or {host}/llms.txt).
"""


def mcp_server_entry(host: str) -> dict:
    """The .mcp.json server entry for the control-plane MCP endpoint."""
    return {
        "type": "http",
        "url": f"{host}/mcp/control",
        "headers": {"Authorization": "Bearer ${ORECLOUD_TOKEN}"},
    }


def scaffold_agent_artifacts(
    local_dir: str | Path, *, host: str, org: str, environment: str, store: str
) -> tuple[list[str], list[str]]:
    """Writes the three artifacts into `local_dir`. Returns (written, notes):
    repo-relative paths that were created/updated, and human-readable notes
    for anything deliberately left alone. Paths use forward slashes on every
    platform — they land in the CLI's JSON output, so the shape is a contract,
    not a display choice."""
    root = Path(local_dir)
    written: list[str] = []
    notes: list[str] = []

    skill_path = root / ".claude" / "skills" / "orecloud" / "SKILL.md"
    skill_path.parent.mkdir(parents=True, exist_ok=True)
    skill_path.write_text(skill_markdown(host, org, environment, store), encoding="utf-8")
    written.append(skill_path.relative_to(root).as_posix())

    agents_path = root / "AGENTS.md"
    if agents_path.exists():
        existing = agents_path.read_text(encoding="utf-8")
        if AGENTS_SECTION_HEADING in existing:
            # The section is the repo's to edit after the first drop —
            # regenerating over their changes would be data loss.
            notes.append(f"AGENTS.md already has an '{AGENTS_SECTION_HEADING}' section; left as is")
        else:
            joiner = (
                "" if existing.endswith("\n\n") else ("\n" if existing.endswith("\n") else "\n\n")
            )
            agents_path.write_text(
                existing + joiner + agents_md_section(host, org, environment, store),
                encoding="utf-8",
            )
            written.append("AGENTS.md")
    else:
        agents_path.write_text(
            agents_md_section(host, org, environment, store), encoding="utf-8"
        )
        written.append("AGENTS.md")

    mcp_path = root / ".mcp.json"
    entry = mcp_server_entry(host)
    if mcp_path.exists():
        try:
            config = json.loads(mcp_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            notes.append(".mcp.json exists but is not valid JSON; left as is")
            return written, notes
        if not isinstance(config, dict):
            notes.append(".mcp.json exists but is not a JSON object; left as is")
            return written, notes
        servers = config.setdefault("mcpServers", {})
        if MCP_SERVER_NAME in servers:
            notes.append(f".mcp.json already has an '{MCP_SERVER_NAME}' server; left as is")
            return written, notes
        servers[MCP_SERVER_NAME] = entry
    else:
        config = {"mcpServers": {MCP_SERVER_NAME: entry}}
    mcp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    written.append(".mcp.json")
    return written, notes
