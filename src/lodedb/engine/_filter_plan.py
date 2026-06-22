"""Set-based resolution of a validated metadata filter to a document-id set.

Given a generation's in-memory per-field value indexes, this resolves a
*validated* metadata filter to the matching document-id set using set operations
and bisect, instead of the O(corpus) per-document compiled-matcher scan that
``LodeEngine._scan_filter_allowlist`` runs. Comparison semantics mirror
``_predicate`` exactly: this module reuses its number parsing and ordered
comparators, so an index-resolved set is identical to the compiled-matcher
result. (That identity is asserted in tests against the reference scan, which is
retained as the oracle.)

Stdlib-only; it must not import ``core``. ``_MetadataPostingIndex`` uses it for
both filtered search (the document set is expanded to chunk ids) and filtered
enumeration / count (the document set is used directly).
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from typing import Any

from lodedb.engine._predicate import _ORDERED_CMP, _as_number


class FieldIndex:
    """Per-key value index for one metadata field within a generation.

    Holds the docs carrying the key, a value -> docs map, and the field's numeric
    values sorted for bisect (with the non-numeric values kept aside so a numeric
    range query does not have to scan every distinct value).
    """

    __slots__ = ("docs", "value_docs", "_num_keys", "_num_values", "_nonnumeric_values")

    def __init__(self) -> None:
        self.docs: set[str] = set()
        self.value_docs: dict[str, set[str]] = {}
        self._num_keys: list[float] = []
        self._num_values: list[str] = []
        self._nonnumeric_values: list[str] = []

    def finalize(self) -> None:
        """Partitions distinct values into a sorted numeric array + the rest."""

        numeric: list[tuple[float, str]] = []
        nonnumeric: list[str] = []
        for value in self.value_docs:
            number = _as_number(value)
            if number is None:
                nonnumeric.append(value)
            else:
                numeric.append((number, value))
        numeric.sort()
        self._num_keys = [number for number, _ in numeric]
        self._num_values = [value for _, value in numeric]
        self._nonnumeric_values = nonnumeric

    def numeric_values_satisfying(self, op: str, operand_number: float) -> list[str]:
        """Returns value strings whose numeric value satisfies ``op operand`` (bisect)."""

        keys = self._num_keys
        if op == "$gt":
            return self._num_values[bisect.bisect_right(keys, operand_number):]
        if op == "$gte":
            return self._num_values[bisect.bisect_left(keys, operand_number):]
        if op == "$lt":
            return self._num_values[: bisect.bisect_left(keys, operand_number)]
        return self._num_values[: bisect.bisect_right(keys, operand_number)]  # $lte


_EMPTY_FIELD = FieldIndex()


def build_field_indexes(
    document_metadata: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, FieldIndex], set[str]]:
    """Builds per-field value indexes and the full document-id set for a generation."""

    fields: dict[str, FieldIndex] = {}
    all_docs: set[str] = set()
    for document_id, metadata in document_metadata.items():
        doc = str(document_id)
        all_docs.add(doc)
        for key, value in metadata.items():
            field = fields.get(str(key))
            if field is None:
                field = FieldIndex()
                fields[str(key)] = field
            field.docs.add(doc)
            field.value_docs.setdefault(str(value), set()).add(doc)
    for field in fields.values():
        field.finalize()
    return fields, all_docs


def resolve(
    metadata_filter: Mapping[str, Any],
    fields: Mapping[str, FieldIndex],
    all_docs: set[str],
) -> set[str]:
    """Resolves a validated metadata filter to the matching document-id set.

    Entries of a node are AND-ed (Mongo semantics, matching ``_predicate``);
    ``$and``/``$or``/``$not`` recurse. Returns a fresh set the caller may mutate.
    """

    result: set[str] | None = None
    for key, spec in metadata_filter.items():
        if key == "$and":
            docs = _resolve_and(spec, fields, all_docs)
        elif key == "$or":
            docs = set()
            for sub in spec:
                docs |= resolve(sub, fields, all_docs)
        elif key == "$not":
            docs = all_docs - resolve(spec, fields, all_docs)
        else:
            docs = _resolve_field(key, spec, fields, all_docs)
        result = docs if result is None else (result & docs)
        if not result:
            return set()
    return set(all_docs) if result is None else result


def _resolve_and(
    subs: list[Mapping[str, Any]],
    fields: Mapping[str, FieldIndex],
    all_docs: set[str],
) -> set[str]:
    result: set[str] | None = None
    for sub in subs:
        docs = resolve(sub, fields, all_docs)
        result = docs if result is None else (result & docs)
        if not result:
            return set()
    return set(all_docs) if result is None else result


def _resolve_field(
    field: str,
    spec: Any,
    fields: Mapping[str, FieldIndex],
    all_docs: set[str],
) -> set[str]:
    index = fields.get(field, _EMPTY_FIELD)
    if not isinstance(spec, Mapping):
        # Bare scalar is exact-match sugar for $eq.
        return set(index.value_docs.get(str(spec), ()))
    result: set[str] | None = None
    for op, operand in spec.items():
        docs = _resolve_operator(op, operand, index, all_docs)
        result = docs if result is None else (result & docs)
        if not result:
            return set()
    return set(all_docs) if result is None else result


def _resolve_operator(
    op: str,
    operand: Any,
    index: FieldIndex,
    all_docs: set[str],
) -> set[str]:
    if op == "$eq":
        return set(index.value_docs.get(str(operand), ()))
    if op == "$ne":
        # A missing key satisfies $ne (its absent value is never equal to operand).
        return all_docs - index.value_docs.get(str(operand), set())
    if op == "$in":
        docs: set[str] = set()
        for value in operand:
            docs |= index.value_docs.get(str(value), set())
        return docs
    if op == "$nin":
        excluded: set[str] = set()
        for value in operand:
            excluded |= index.value_docs.get(str(value), set())
        return all_docs - excluded
    if op == "$exists":
        return set(index.docs) if operand else (all_docs - index.docs)
    # Ordered ($gt/$gte/$lt/$lte): a missing key never satisfies an ordered op.
    return _resolve_ordered(op, str(operand), index)


def _resolve_ordered(op: str, operand: str, index: FieldIndex) -> set[str]:
    compare = _ORDERED_CMP[op]
    operand_number = _as_number(operand)
    docs: set[str] = set()
    if operand_number is None:
        # Non-numeric operand: every comparison is lexicographic (matching
        # _predicate), so compare each stored value string to the operand.
        for value, value_docs in index.value_docs.items():
            if compare(value, operand):
                docs |= value_docs
        return docs
    # Numeric operand: numeric-parseable stored values compare numerically
    # (resolved by bisect); non-numeric stored values compare lexicographically
    # against the operand string (kept to the small non-numeric set).
    for value in index.numeric_values_satisfying(op, operand_number):
        docs |= index.value_docs[value]
    for value in index._nonnumeric_values:
        if compare(value, operand):
            docs |= index.value_docs[value]
    return docs
