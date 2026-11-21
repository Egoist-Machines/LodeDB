"""OpenKnowledge's dependency-free document chunker, ported from TypeScript."""

from __future__ import annotations

from collections.abc import Iterable

CHUNK_TARGET_CHARS = 8_000
CHUNK_OVERLAP_CHARS = 400
MAX_CHUNKS_PER_DOC = 80
CHUNK_CONFIG_ID = f"c{CHUNK_TARGET_CHARS}-o{CHUNK_OVERLAP_CHARS}-m{MAX_CHUNKS_PER_DOC}"

# ECMAScript WhiteSpace plus LineTerminator code points used by String.prototype.trim().
_JS_TRIM_CODE_UNITS = frozenset(
    "\u0009\u000a\u000b\u000c\u000d\u0020\u00a0\u1680"
    "\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007"
    "\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"
)


def _to_js_code_units(text: str) -> str:
    """Returns a Python string whose characters are JavaScript UTF-16 code units."""

    raw = text.encode("utf-16-le", "surrogatepass")
    return "".join(
        chr(int.from_bytes(raw[index : index + 2], "little")) for index in range(0, len(raw), 2)
    )


def _from_js_code_units(code_units: str) -> str:
    """Reassembles a Python string from JavaScript UTF-16 code units."""

    raw = b"".join(unit.encode("utf-16-le", "surrogatepass") for unit in code_units)
    return raw.decode("utf-16-le", "surrogatepass")


def _js_trim(code_units: str) -> str:
    """Implements String.prototype.trim() over the UTF-16 representation."""

    start = 0
    end = len(code_units)
    while start < end and code_units[start] in _JS_TRIM_CODE_UNITS:
        start += 1
    while end > start and code_units[end - 1] in _JS_TRIM_CODE_UNITS:
        end -= 1
    return code_units[start:end]


def javascript_length(text: str) -> int:
    """Returns ``text.length`` as JavaScript would report it."""

    return len(text.encode("utf-16-le", "surrogatepass")) // 2


def _last_index_of_boundary(code_units: str, end: int) -> int:
    """Matches Math.max(text.lastIndexOf(' ', end), text.lastIndexOf('\\n', end))."""

    return max(code_units.rfind(" ", 0, end + 1), code_units.rfind("\n", 0, end + 1))


def chunk_document(
    text: str,
    *,
    target_chars: int | None = None,
    overlap_chars: int | None = None,
    max_chunks: int | None = None,
) -> list[str]:
    """Splits text exactly as OpenKnowledge's ``chunkDocument`` does.

    The TypeScript implementation measures character positions in UTF-16 code units.
    Keeping that behavior matters for documents containing astral Unicode characters.
    """

    target = max(1, CHUNK_TARGET_CHARS if target_chars is None else target_chars)
    requested_overlap = CHUNK_OVERLAP_CHARS if overlap_chars is None else overlap_chars
    overlap = max(0, min(requested_overlap, target - 1))
    chunk_cap = MAX_CHUNKS_PER_DOC if max_chunks is None else max_chunks
    source = _to_js_code_units(text)

    if not _js_trim(source):
        return []
    if len(source) <= target:
        return [_from_js_code_units(_js_trim(source))]

    chunks: list[str] = []
    start = 0
    while start < len(source) and len(chunks) < chunk_cap:
        end = min(len(source), start + target)
        if end < len(source):
            boundary = _last_index_of_boundary(source, end)
            if boundary > start + target // 2:
                end = boundary
        piece = _js_trim(source[start:end])
        if piece:
            chunks.append(_from_js_code_units(piece))
        if end >= len(source):
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return chunks


def chunkDocument(  # noqa: N802 - matches the OpenKnowledge public helper name.
    text: str,
    options: dict[str, int] | None = None,
) -> list[str]:
    """Compatibility spelling and options shape of OpenKnowledge's TypeScript helper."""

    values: Iterable[tuple[str, int]] = () if options is None else options.items()
    kwargs = {
        "target_chars": None,
        "overlap_chars": None,
        "max_chunks": None,
    }
    for key, value in values:
        if key == "targetChars":
            kwargs["target_chars"] = value
        elif key == "overlapChars":
            kwargs["overlap_chars"] = value
        elif key == "maxChunks":
            kwargs["max_chunks"] = value
    return chunk_document(text, **kwargs)
