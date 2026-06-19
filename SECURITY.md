# Security Policy

LodeDB is a local-first, in-process library: it runs inside your own application, stores
its index on local disk, and makes no outbound network calls except a one-time Hugging
Face download of embedding-model weights. There is no hosted LodeDB service to attack
remotely — the security surface is the library running on your machine.

## Reporting a vulnerability

Please report security issues **privately** — do not open a public GitHub issue.

- **Preferred:** open a private advisory via GitHub Security Advisories at
  <https://github.com/Egoist-Machines/LodeDB/security/advisories/new>.
- **Or email** <oss@egoistmachines.com> with steps to reproduce, the affected version
  (`python -c "import lodedb; print(lodedb.__version__)"`), and impact.

We aim to acknowledge a report within 5 business days and to share a remediation timeline
after triage. Please give us a reasonable window to ship a fix before public disclosure;
we're happy to credit reporters who want it.

## Supported versions

LodeDB is pre-1.0. Security fixes target the latest released `0.x` version only. Pin a
version and watch releases until a stable line is published.

## Scope and operational notes

- The bundled dev server (`lodedb serve`) and MCP server (`lodedb mcp`) are
  **unauthenticated and intended for loopback / local use only**. Do not expose them to
  untrusted networks.
- Original document text is retained **by default** in a local `.tvtext` sidecar. Treat
  that directory as sensitive, or open the database with `store_text=False` to keep no
  text on disk. See [`docs/architecture.md`](docs/architecture.md) for the
  persistence / payload boundary.
- The vendored TurboVec core under `third_party/turbovec/` is upstream MIT code. Issues
  specific to upstream TurboVec are best reported to that project, but feel free to flag
  anything you find through the channels above and we'll help coordinate.
