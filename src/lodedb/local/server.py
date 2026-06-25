"""Minimal local HTTP server for LodeDB (dev convenience, no auth).

A thin local HTTP loop over :class:`LodeDB` for quick experimentation from
non-Python clients on your own machine. It binds to loopback by default and
refuses non-private hosts; private-network binds are an intentional trusted-LAN
mode. It carries no auth because the local embedded mode is no-auth, so never
expose it to public or untrusted networks. Raw documents and queries are never
logged; by default it logs nothing. ``POST /get`` returns a stored document's raw
text by id; it is available unless the server was started with
``serve --no-store-text``.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lodedb.engine.core import is_private_bind_host
from lodedb.local.db import LodeDB

# Cap request bodies on the loopback dev server (defensive). Oversized bodies
# get a 400 instead of an unbounded read.
_MAX_BODY_BYTES = 64 * 1024 * 1024


def build_local_handler(db: LodeDB) -> type[BaseHTTPRequestHandler]:
    """Builds a request handler bound to one open :class:`LodeDB` instance."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args: object) -> None:  # noqa: D401 - silence access log
            """Suppresses the default access log to avoid leaking request lines."""

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length <= 0:
                return {}
            if length > _MAX_BODY_BYTES:
                raise ValueError("request body too large")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            if self.path == "/healthz":
                self._send(200, {"status": "ok"})
            elif self.path == "/stats":
                self._send(200, db.stats())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            try:
                payload = self._read_json()
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "invalid JSON body"})
                return
            try:
                if self.path == "/add":
                    doc_id = db.add(
                        payload["text"],
                        id=payload.get("id"),
                        metadata=payload.get("metadata"),
                    )
                    self._send(200, {"id": doc_id, "count": db.count()})
                elif self.path == "/search":
                    hits = db.search(
                        payload["query"],
                        k=int(payload.get("k", 10)),
                        filter=payload.get("filter"),
                    )
                    self._send(
                        200,
                        {
                            "results": [
                                {"score": h.score, "id": h.id, "metadata": h.metadata}
                                for h in hits
                            ]
                        },
                    )
                elif self.path == "/remove":
                    self._send(200, {"removed": db.remove(payload["id"]), "count": db.count()})
                elif self.path == "/get":
                    text = db.get(payload["id"])
                    if text is None:
                        self._send(404, {"error": "document not found"})
                    else:
                        self._send(200, {"id": payload["id"], "text": text})
                else:
                    self._send(404, {"error": "not found"})
            except KeyError as exc:
                self._send(400, {"error": f"missing field: {exc}"})
            except ValueError as exc:
                self._send(400, {"error": str(exc)})

    return _Handler


def serve_local(
    *,
    path: str | Path,
    model: str = "minilm",
    device: str = "auto",
    embedding_runtime: str = "auto",
    host: str = "127.0.0.1",
    port: int = 8088,
    store_text: bool = True,
    durability: str | None = None,
    commit_mode: str | None = None,
) -> None:
    """Opens an :class:`LodeDB` and serves it on a local/private HTTP loop (blocking).

    Raw-text storage is on by default so ``POST /get`` can return a document's
    original text by id; pass ``store_text=False`` to opt out. ``durability``
    (``"fast"``/``"fsync"``) controls power-loss durability of each commit; this
    is the writer that holds the path, so its requests are serialized by the
    engine's in-process lock. ``commit_mode`` (``"generation"``/``"wal"``)
    selects the per-mutation commit path; ``"wal"`` checkpoints the log into a
    generation on shutdown via the ``db.close()`` below.
    """

    if not is_private_bind_host(host):
        raise ValueError("LodeDB local server host must be loopback or private network")
    db = LodeDB(
        path=path,
        model=model,
        device=device,
        embedding_runtime=embedding_runtime,
        store_text=store_text,
        durability=durability,
        commit_mode=commit_mode,
    )
    handler = build_local_handler(db)
    server = ThreadingHTTPServer((host, port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        db.close()
