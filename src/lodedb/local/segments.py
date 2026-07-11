"""WAL segment files: store-free planning, encoding, and folding.

Advanced building blocks for out-of-band ingest (e.g. a managed cloud's
multi-writer pipeline): a *writer* chunks and embeds documents with no open
store, encodes them into an immutable segment file in LodeDB's WAL frame
format, and ships the bytes; a *fold orchestrator* later stamps log sequence
numbers and folds each segment into a warm writable :class:`~lodedb.LodeDB`
handle opened in ``commit_mode="generation"``, publishing one O(changed)
generation delta per fold batch via :meth:`LodeDB.persist`.

This module is deliberately not re-exported from ``lodedb``'s root: local
applications should use :class:`~lodedb.LodeDB` / :class:`~lodedb.Appender`.
Segments carry raw document text only when planned with ``store_text=True``
(the same privacy policy as ``<key>.wal``).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from lodedb import _native_core
from lodedb.local.db import _LOCAL_INDEX_ID, LodeDB, _coerce_metadata

__all__ = [
    "plan_documents",
    "build_embedded_documents_record",
    "delete_documents_record",
    "encode_segment",
    "decode_segment",
    "fold_segment",
]


def plan_documents(
    documents: Iterable[Mapping[str, Any]],
    *,
    store_text: bool = False,
    index_text: bool = False,
    chunk_character_limit: int = 900,
) -> dict[str, Any]:
    """Chunks ``{"text", "id", "metadata"?}`` documents into the appender-shaped
    ingest plan without an open store; every chunk is marked for embedding.

    ``chunk_character_limit`` and the text flags MUST match the target store's
    writer (:class:`~lodedb.LodeDB` defaults: 900, ``store_text=True``) or chunk
    ids and text retention diverge at fold time. Both text flags default **off**
    for privacy, matching :meth:`Appender.open`. Ids are required and non-empty:
    auto-generated ids would collide across concurrent writers. Metadata is
    coerced to the engine's string->string model.
    """

    prepared: list[dict[str, Any]] = []
    for item in documents:
        text = item.get("text")
        if text is None or not str(text).strip():
            raise ValueError("each document needs a non-empty 'text'")
        document_id = item.get("id")
        if document_id is None or not str(document_id).strip():
            raise ValueError("each document needs a non-empty 'id'")
        prepared.append(
            {
                "document_id": str(document_id),
                "text": str(text),
                "metadata": _coerce_metadata(item.get("metadata")),
            }
        )
    if not prepared:
        raise ValueError("plan_documents requires at least one document")
    plan_json = _native_core.plan_segment_documents(
        json.dumps(prepared),
        bool(store_text),
        bool(index_text),
        int(chunk_character_limit),
    )
    plan = json.loads(plan_json)
    if not isinstance(plan, dict):
        raise RuntimeError("native core returned a non-object ingest plan")
    return plan


def build_embedded_documents_record(
    plan: Mapping[str, Any],
    embeddings: Iterable[Iterable[float]],
    *,
    vector_dim: int,
) -> dict[str, Any]:
    """Returns the ``{"op": "apply_embedded_documents", "payload": ...}`` record
    for a plan plus one embedding per ``plan["chunks_to_embed"]``, in order.

    The embedding count, ``vector_dim``, and finiteness are validated natively,
    so a bad record can never be encoded (and never uploaded). The payload is
    byte-identical to the WAL record :meth:`Appender.append_text_many` logs for
    the same input.
    """

    import numpy as np

    rows = [np.asarray(row, dtype=np.float32) for row in embeddings]
    matrix = (
        np.ascontiguousarray(rows, dtype=np.float32)
        if rows
        else np.zeros((0, 0), dtype=np.float32)
    )
    payload_json = _native_core.build_embedded_documents_payload(
        json.dumps(dict(plan)), matrix, int(vector_dim)
    )
    return {"op": "apply_embedded_documents", "payload": json.loads(payload_json)}


def delete_documents_record(ids: str | Iterable[str]) -> dict[str, Any]:
    """Returns the ``{"op": "delete_documents", "payload": ...}`` record deleting
    the given document ids; a bare string is one id. Ids must be non-empty.
    Pure Python: the payload is trivial and needs no native validation."""

    if isinstance(ids, str):
        ids = [ids]
    document_ids = [str(document_id) for document_id in ids]
    if not document_ids:
        raise ValueError("delete_documents_record requires at least one document id")
    if any(not document_id.strip() for document_id in document_ids):
        raise ValueError("document ids must be non-empty")
    return {"op": "delete_documents", "payload": {"document_ids": document_ids}}


def encode_segment(records: Sequence[Mapping[str, Any]]) -> bytes:
    """Encodes ``{"op", "payload"}`` records into an immutable LodeDB WAL-format
    segment (``EELWAL01`` header plus CRC-framed records, LSNs unassigned).

    At least one record is required, ops must be natively replayable, and
    records must be unstamped -- all fail closed here, before any upload.
    """

    rows = [dict(record) for record in records]
    return bytes(_native_core.encode_wal_segment(json.dumps(rows)))


def decode_segment(data: bytes) -> list[dict[str, Any]]:
    """Strictly decodes segment bytes to ``[{"op", "payload", "lsn"}, ...]``.

    Unlike the crash-tolerant WAL file reader, any torn or corrupt frame raises:
    a segment is a complete immutable blob, so truncation means a corrupt
    download. For tests and writer-side validation.
    """

    records = json.loads(_native_core.decode_wal_segment(bytes(data)))
    if not isinstance(records, list):
        raise RuntimeError("native core returned a non-list segment decode")
    return [dict(record) for record in records]


def fold_segment(db: LodeDB, data: bytes, *, first_lsn: int) -> int:
    """Folds one downloaded segment into a warm writable ``db`` opened with
    ``commit_mode="generation"``, returning the number of records applied.

    Decodes strictly, stamps LSNs ``first_lsn .. first_lsn + n - 1`` (a segment
    already carrying LSNs is refused), and applies in memory only: call
    :meth:`LodeDB.persist` after the fold batch to publish one O(changed)
    generation delta. Records at or below the store's applied watermark skip
    for refold idempotence -- compare the returned count to the segment's
    record count and treat an unexpected shortfall as an LSN-allocation bug
    (stamp from ``max(committed applied_lsn, last stamped) + 1``; see
    :meth:`LodeDB.applied_lsn`). On any exception the in-memory state may be
    partially applied (disk is untouched): abandon the handle with
    :meth:`LodeDB.discard` -- never :meth:`LodeDB.close`, which would persist
    the partial batch -- and reopen before retrying.
    """

    db._require_writable()
    with db._op_lock:
        return int(
            db._native_vector_engine.fold_wal_segment(
                _LOCAL_INDEX_ID, bytes(data), int(first_lsn)
            )
        )
