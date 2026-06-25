"""Read-only exporter for a direct pgvector table (vector-preserve).

This is the first-class direct provider for issue #35: a Postgres table with a
``vector(N)`` column used directly from application code (psycopg/asyncpg/SQLAlchemy
or raw SQL), not through a framework. The importer connects with the application's
configured DSN (redacted from every log and report), reads the table's id / text /
vector / metadata columns in stable primary-key order, and streams each row as a
vector-preserve :class:`ExportedRow`. The application already owns the embeddings,
so vectors are copied verbatim.

Strict read-only contract: the importer issues only ``SELECT`` (and read-only
catalog probes). It never runs ``DROP`` / ``DELETE`` / ``TRUNCATE`` / ``UPDATE`` /
``INSERT`` / schema or index changes. The DSN is opened lazily through an injectable
``connect`` callable (so tests can drive a fake DB-API connection), and a non-local
host requires an explicit remote override.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from typing import Any

from lodedb.local.migrate.report import is_local_source, redact_connection_string
from lodedb.local.migrate.sources.base import (
    MODE_VECTOR_PRESERVE,
    ExportedRow,
    SourceExport,
    SourceExportError,
)

# A conservative identifier guard: table/column names are quoted into SQL, so they
# must be plain identifiers (optionally schema-qualified) to keep injection off the table.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")
_EXPORT_BATCH = 500
# Default column names matched against common direct-pgvector and LangChain pgvector schemas.
_DEFAULT_ID_COLUMNS = ("id", "uuid", "custom_id", "pk")
_DEFAULT_TEXT_COLUMNS = ("text", "content", "document", "chunk", "body", "page_content")
_DEFAULT_VECTOR_COLUMNS = ("embedding", "vector", "embeddings", "vec")
_DEFAULT_METADATA_COLUMNS = ("metadata", "cmetadata", "meta", "payload")


def _default_connect(dsn: str) -> Any:
    """Opens a psycopg(3) connection for ``dsn`` (the default DB-API factory)."""

    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - exercised only without psycopg
        raise SourceExportError(
            "exporting a pgvector table needs psycopg (psycopg[binary]); install it in the "
            "source project's environment, or pass a connection factory"
        ) from exc
    return psycopg.connect(dsn)


def _quote_ident(name: str) -> str:
    """Quotes a (possibly schema-qualified) identifier after validating each part."""

    parts = name.split(".")
    for part in parts:
        if not _IDENT_RE.match(part):
            raise SourceExportError(f"unsafe SQL identifier {name!r}")
    return ".".join(f'"{part}"' for part in parts)


def _split_schema_table(table: str) -> tuple[str, str]:
    """Splits ``schema.table`` (default schema ``public``) into ``(schema, table)``."""

    if "." in table:
        schema, _, name = table.partition(".")
        return schema, name
    return "public", table


class PgVectorExport(SourceExport):
    """Streams a pgvector table as vector-preserve rows, read-only."""

    def __init__(
        self,
        *,
        dsn: str,
        table: str,
        id_column: str | None = None,
        text_column: str | None = None,
        vector_column: str | None = None,
        metadata_column: str | None = None,
        vector_dim: int | None = None,
        allow_remote: bool = False,
        connect: Callable[[str], Any] | None = None,
    ) -> None:
        """Connects (read-only), resolves columns, and reads the vector dimension."""

        if not is_local_source(dsn) and not allow_remote:
            raise SourceExportError(
                "refusing to connect to a non-local Postgres host without an explicit override; "
                "re-run with --allow-remote-source after confirming the host is safe to read"
            )
        for value in (table, id_column, text_column, vector_column, metadata_column):
            if value is not None:
                _quote_ident(value)
        connect = connect or _default_connect
        conn = connect(dsn)
        try:
            columns = _table_columns(conn, table)
            resolved = _resolve_columns(
                columns,
                id_column=id_column,
                text_column=text_column,
                vector_column=vector_column,
                metadata_column=metadata_column,
            )
            dim = vector_dim or _vector_dimension(conn, table, resolved["vector"])
            if dim is None:
                raise SourceExportError(
                    f"could not determine the vector dimension of {table}.{resolved['vector']}; "
                    "pass --vector-dim"
                )
            count = _row_count(conn, table)
        except SourceExportError:
            _close(conn)
            raise
        warnings: list[str] = []
        if resolved["text"] is None:
            warnings.append(
                "no text column detected; rows export as vector-preserve without payload text "
                "(pass --text-column to retain text, or migrate vector-only)"
            )
        super().__init__(
            framework=None,
            provider="pgvector",
            mode=MODE_VECTOR_PRESERVE,
            location=redact_connection_string(dsn),
            vector_dim=int(dim),
            count=count,
            warnings=warnings,
            notes={
                "table": table,
                "id_column": resolved["id"],
                "text_column": resolved["text"],
                "vector_column": resolved["vector"],
                "metadata_column": resolved["metadata"],
            },
        )
        self._conn = conn
        self._table = table
        self._cols = resolved
        self._dim = int(dim)

    def iter_rows(self) -> Iterator[ExportedRow]:
        """Streams rows in stable primary-key order, in fixed batches.

        Uses keyset pagination ordered by the id column so a long export is stable
        even if the source is being written concurrently by the application.
        """

        select_cols = [self._cols["id"], self._cols["vector"]]
        if self._cols["text"]:
            select_cols.append(self._cols["text"])
        if self._cols["metadata"]:
            select_cols.append(self._cols["metadata"])
        quoted = ", ".join(_quote_ident(col) for col in select_cols)
        table = _quote_ident(self._table)
        id_q = _quote_ident(self._cols["id"])

        last_id: Any = None
        while True:
            if last_id is None:
                sql = f"SELECT {quoted} FROM {table} ORDER BY {id_q} LIMIT %s"
                params: tuple[Any, ...] = (_EXPORT_BATCH,)
            else:
                sql = (
                    f"SELECT {quoted} FROM {table} WHERE {id_q} > %s "
                    f"ORDER BY {id_q} LIMIT %s"
                )
                params = (last_id, _EXPORT_BATCH)
            cursor = self._conn.cursor()
            try:
                cursor.execute(sql, params)
                rows = cursor.fetchall()
            finally:
                _close(cursor)
            if not rows:
                break
            for row in rows:
                raw_id = row[0]
                last_id = raw_id
                offset = 2
                text = None
                if self._cols["text"]:
                    text = row[offset]
                    offset += 1
                metadata: dict[str, Any] = {}
                if self._cols["metadata"]:
                    metadata = _coerce_metadata(row[offset])
                yield ExportedRow(
                    id=str(raw_id),
                    text=text if isinstance(text, str) else None,
                    metadata=metadata,
                    vector=_coerce_pg_vector(row[1], self._dim),
                )
            if len(rows) < _EXPORT_BATCH:
                break

    def close(self) -> None:
        """Closes the read-only Postgres connection."""

        _close(self._conn)


def _table_columns(conn: Any, table: str) -> dict[str, str]:
    """Returns ``{column_name: data_type}`` for a table via information_schema (read-only)."""

    schema, name = _split_schema_table(table)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s",
            (schema, name),
        )
        rows = cursor.fetchall()
    finally:
        _close(cursor)
    if not rows:
        raise SourceExportError(f"table {table!r} was not found (or has no columns)")
    return {str(r[0]): str(r[1]) for r in rows}


def _resolve_columns(
    columns: dict[str, str],
    *,
    id_column: str | None,
    text_column: str | None,
    vector_column: str | None,
    metadata_column: str | None,
) -> dict[str, str | None]:
    """Resolves id/text/vector/metadata column names, auto-detecting where unset."""

    names = set(columns)

    def pick(explicit: str | None, candidates: tuple[str, ...], *, required: bool, what: str):
        if explicit is not None:
            if explicit not in names:
                raise SourceExportError(f"{what} column {explicit!r} is not in table columns")
            return explicit
        for candidate in candidates:
            if candidate in names:
                return candidate
        if required:
            raise SourceExportError(
                f"could not auto-detect the {what} column; pass --{what.replace('_', '-')}-column"
            )
        return None

    vector = pick(vector_column, _DEFAULT_VECTOR_COLUMNS, required=True, what="vector")
    id_col = pick(id_column, _DEFAULT_ID_COLUMNS, required=True, what="id")
    text = pick(text_column, _DEFAULT_TEXT_COLUMNS, required=False, what="text")
    metadata = pick(metadata_column, _DEFAULT_METADATA_COLUMNS, required=False, what="metadata")
    return {"id": id_col, "text": text, "vector": vector, "metadata": metadata}


def _vector_dimension(conn: Any, table: str, vector_column: str) -> int | None:
    """Reads the declared ``vector(N)`` dimension from pg_catalog, else samples one row.

    Prefers the declared type modifier (``atttypmod``) so an empty table still has a
    known dimension; falls back to measuring one stored vector. Both are read-only.
    """

    schema, name = _split_schema_table(table)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT a.atttypmod FROM pg_attribute a "
            "JOIN pg_class c ON c.oid = a.attrelid "
            "JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname = %s AND c.relname = %s AND a.attname = %s",
            (schema, name, vector_column),
        )
        row = cursor.fetchone()
    except Exception:  # noqa: BLE001 - catalog probe is best-effort; fall back to sampling
        row = None
    finally:
        _close(cursor)
    if row is not None and isinstance(row[0], int) and row[0] > 0:
        return int(row[0])

    cursor = conn.cursor()
    try:
        cursor.execute(
            f"SELECT {_quote_ident(vector_column)} FROM {_quote_ident(table)} "
            f"WHERE {_quote_ident(vector_column)} IS NOT NULL LIMIT 1"
        )
        sample = cursor.fetchone()
    finally:
        _close(cursor)
    if not sample or sample[0] is None:
        return None
    measured = _coerce_pg_vector(sample[0], None)
    return len(measured) if measured else None


def _row_count(conn: Any, table: str) -> int | None:
    """Returns the table row count (read-only ``SELECT COUNT(*)``), or ``None`` on failure."""

    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {_quote_ident(table)}")
        row = cursor.fetchone()
    except Exception:  # noqa: BLE001 - count is informational only
        return None
    finally:
        _close(cursor)
    return int(row[0]) if row and isinstance(row[0], int) else None


def _coerce_pg_vector(value: Any, dim: int | None) -> list[float] | None:
    """Parses a pgvector value (list/tuple, or the ``[1,2,3]`` text form) to floats."""

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        components = value
    elif isinstance(value, str):
        stripped = value.strip().strip("[]").strip()
        if not stripped:
            return None
        components = stripped.split(",")
    else:
        return None
    try:
        out = [float(component) for component in components]
    except (TypeError, ValueError):
        return None
    if dim is not None and len(out) != dim:
        # An inconsistent-width row is dropped to ``None`` so the runner records it as a
        # skipped/dimension-mismatch row rather than corrupting the index.
        return None
    return out


def _coerce_metadata(value: Any) -> dict[str, Any]:
    """Coerces a JSON/JSONB metadata column value to a dict (best-effort)."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _close(handle: Any) -> None:
    """Closes a cursor/connection, ignoring errors (read-only cleanup)."""

    close = getattr(handle, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001 - best-effort close
            pass
