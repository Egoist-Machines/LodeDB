# Native Core WAL Payloads

The native WAL reader/writer mirrors the Python frame format: magic/version
header, big-endian body length, `op\n` plus compact JSON payload, and a CRC32
over length plus body. A torn final frame is dropped; a bad interior frame fails
closed.

Payload classes:

- `upsert_documents`: may contain raw document text. It is the writer's private
  recovery log, not telemetry or a redacted artifact.
- `apply_embedded_documents`: carries chunk embeddings and derived tokens for
  `store_text=False` recovery; it does not carry raw document text.
- `upsert_vectors`: carries caller vectors and redacted metadata. Optional text
  is present only when raw-text storage is enabled.
- `delete_documents` and `update_document_payload`: carry ids plus redacted
  metadata/text-update intent needed for replay.

Checkpoints write a fresh generation first and truncate the WAL only after the
root manifest has been published, so replay after a crash is idempotent.
