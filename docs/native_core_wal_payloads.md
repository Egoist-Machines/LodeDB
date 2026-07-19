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

## Segment files

A WAL *segment* is an immutable standalone blob in exactly the file format
above (header plus CRC-framed records), produced by `encode_wal_segment` with
no store open — the building block for out-of-band ingest (see
`lodedb.local.segments`). Two deliberate differences from the on-disk
`<key>.wal`:

- **LSNs are stamped at fold time, not at encode time.** Segment records carry
  no `lsn` key on the wire; the folding side stamps consecutive LSNs from a
  floor above the store's committed `applied_lsn` and applies them in memory
  (`apply_wal_records`), publishing one O(changed) generation delta per fold
  batch. A segment that already carries LSNs is refused (it signals a re-fold
  of an already-folded segment).
- **Decoding is strict.** A segment is complete by construction, so a short
  header, torn tail, or trailing CRC failure means a corrupt transfer and fails
  closed — unlike the crash-tolerant `<key>.wal` reader, which drops a torn
  final frame.

Raw document text enters a segment only when the plan was built with
`store_text=True`, mirroring the `<key>.wal` policy.
