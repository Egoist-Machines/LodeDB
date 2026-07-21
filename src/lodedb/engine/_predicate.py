"""Metadata filter predicate engine for query-time document filtering.

This module is intentionally dependency-free (stdlib only) and must not import
``core``; it is consumed by both the engine trust boundary
(``LodeEngine._validate_query_filter`` validates; ``core._compile_query_filter``
compiles the per-document matcher used by the corpus scan) and the SDK ergonomics
layer (``lodedb.local.db._normalize_filter``).

Grammar (backward compatible; a bare scalar is exact-match sugar for ``$eq``):

    metadata-filter := { (field | logical)* }            # entries are AND-ed
    field           := <name> : (scalar | operator-map)
    operator-map    := { (<comparison-op> : operand)+ }  # operators are AND-ed
    logical         := $and:[filter,...] | $or:[filter,...] | $not:filter

    comparison-op   := $eq | $ne | $gt | $gte | $lt | $lte | $in | $nin | $exists
    operand($in/$nin) := [scalar, ...]
    operand($exists)  := bool
    operand(other)    := scalar  (str | int | float | bool | None)

Stored metadata is always a flat ``str -> str`` map (the engine stringifies on
write and never migrates it). Filtering therefore compares strings, with one
exception: the ordered operators (``$gt/$gte/$lt/$lte``) parse *both* the stored
value and the operand as numbers and compare numerically when both parse (and
neither is NaN); otherwise they fall back to a lexicographic string compare.

Two stringify rules exist so each interface stays self-consistent without
changing the on-disk format:

- :func:`coerce_sdk_filter` uses the SDK rule (``True`` -> ``"true"``), matching
  how ``lodedb.local.db._coerce_metadata`` writes document metadata, so an SDK
  caller's bool filter matches SDK-stored data.
- :func:`validate_metadata_filter` uses the engine rule (``str(True)`` ->
  ``"True"``), matching ``core._validate_metadata`` storage, so a direct-engine /
  HTTP caller stays self-consistent. SDK-coerced operands are already strings, so
  this second pass is a no-op for them.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Callable, Mapping
from typing import Any

ORDERED_OPERATORS = frozenset({"$gt", "$gte", "$lt", "$lte"})
COMPARISON_OPERATORS = frozenset(
    {"$eq", "$ne", "$in", "$nin", "$exists"} | ORDERED_OPERATORS
)
LOGICAL_OPERATORS = frozenset({"$and", "$or", "$not"})

# Fail closed on pathological nesting rather than risk a recursion blow-up on a
# hostile filter (e.g. an agent relaying an attacker-shaped predicate).
_MAX_DEPTH = 32

# Sentinel for "key absent" so a single ``dict.get`` distinguishes a missing key
# from any stored string without a second ``in`` lookup on the per-document path.
_MISSING = object()

# Ordered operators compiled to bound comparators (resolved once, not per row).
_ORDERED_CMP = {
    "$gt": operator.gt,
    "$gte": operator.ge,
    "$lt": operator.lt,
    "$lte": operator.le,
}


def validate_metadata_filter(metadata: Any) -> dict[str, Any]:
    """Validates a metadata filter at the engine trust boundary and stringifies operands.

    This is authoritative: it runs for every query regardless of caller (SDK,
    HTTP dev server, or direct engine use). Raises ``ValueError`` with a precise
    message on any grammar violation.
    """

    if not isinstance(metadata, Mapping) or not metadata:
        raise ValueError("filter.metadata must be a nonempty object")
    return _walk(metadata, _engine_stringify, 0)


def coerce_sdk_filter(metadata: Any) -> dict[str, Any]:
    """Coerces a metadata filter from the SDK using the SDK bool rule.

    Mirrors the document-metadata coercion (``True`` -> ``"true"``) so a bool
    filter from the SDK matches SDK-stored metadata. The engine re-validates the
    result; this pass exists to canonicalize operands before the engine sees them.
    """

    return _walk(metadata, _sdk_stringify, 0)


def matches_metadata_filter(
    document_metadata: Mapping[str, str],
    metadata_filter: Mapping[str, Any],
) -> bool:
    """Returns whether a document's stored metadata satisfies a validated filter.

    Convenience for a one-off match: it compiles then evaluates. To test many
    documents against the same filter, call :func:`compile_metadata_filter` once
    and reuse the predicate (this is what the engine's corpus scan does).
    """

    return _compile_node(metadata_filter)(document_metadata)


def compile_metadata_filter(
    metadata_filter: Mapping[str, Any],
) -> Callable[[Mapping[str, str]], bool]:
    """Compiles a validated filter into a reusable ``metadata -> bool`` predicate.

    The filter must already be validated (see :func:`validate_metadata_filter`);
    compilation trusts the grammar and does not re-check it. It hoists everything
    independent of the document out of the per-document path: operator dispatch
    collapses into a bound comparator, ``$in``/``$nin`` targets stay tuples, and
    ordered operands are parsed to numbers once instead of re-running ``float()``
    on the constant operand for every row scanned. The returned predicate is pure
    and may be reused across an entire corpus scan.
    """

    return _compile_node(metadata_filter)


# --- validation / coercion -------------------------------------------------


def _walk(node: Any, stringify: Any, depth: int) -> dict[str, Any]:
    if depth > _MAX_DEPTH:
        raise ValueError("filter is nested too deeply")
    if not isinstance(node, Mapping):
        raise ValueError("filter must be an object")
    if not node:
        raise ValueError("filter must be a non-empty object")
    result: dict[str, Any] = {}
    for key, spec in node.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("filter keys must be non-blank strings")
        if key in LOGICAL_OPERATORS:
            result[key] = _walk_logical(key, spec, stringify, depth)
        elif key.startswith("$"):
            raise ValueError(f"unsupported filter operator {key!r} at field level")
        else:
            result[key] = _walk_field(key, spec, stringify, depth)
    return result


def _walk_logical(op: str, spec: Any, stringify: Any, depth: int) -> Any:
    if op == "$not":
        return _walk(spec, stringify, depth + 1)
    if not isinstance(spec, list) or not spec:
        raise ValueError(f"{op} requires a non-empty list of filters")
    return [_walk(sub, stringify, depth + 1) for sub in spec]


def _walk_field(field: str, spec: Any, stringify: Any, depth: int) -> Any:
    if isinstance(spec, Mapping):
        if not spec:
            raise ValueError(f"operator map for field {field!r} must be non-empty")
        operators: dict[str, Any] = {}
        for op, operand in spec.items():
            if op not in COMPARISON_OPERATORS:
                raise ValueError(f"unsupported operator {op!r} for field {field!r}")
            operators[op] = _validate_operand(field, op, operand, stringify)
        return operators
    # Bare scalar is exact-match sugar for ``$eq``.
    return _scalar(field, spec, stringify)


def _validate_operand(field: str, op: str, operand: Any, stringify: Any) -> Any:
    if op == "$exists":
        if not isinstance(operand, bool):
            raise ValueError(f"$exists for field {field!r} requires a boolean")
        return operand
    if op in ("$in", "$nin"):
        if not isinstance(operand, (list, tuple)) or not operand:
            raise ValueError(f"{op} for field {field!r} requires a non-empty list")
        return tuple(_scalar(field, item, stringify) for item in operand)
    return _scalar(field, operand, stringify)


def _scalar(field: str, value: Any, stringify: Any) -> str:
    if isinstance(value, Mapping) or isinstance(value, (list, tuple)):
        raise ValueError(f"filter operand for field {field!r} must be a scalar value")
    return stringify(value)


def _sdk_stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    raise ValueError("filter operand must be a string, number, boolean, or null")


def _engine_stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bool, int, float, str)):
        return str(value)
    raise ValueError("filter operand must be a string, number, boolean, or null")


# --- compilation -----------------------------------------------------------
#
# A validated filter is compiled once into a predicate (a tree of closures) and
# then evaluated against many documents. Entries of a node are AND-ed (Mongo
# semantics); ``$and``/``$or``/``$not`` recurse into sub-nodes. Each field
# operator becomes a closure over its (pre-parsed) operand, so the per-document
# call does no operator-string dispatch and no operand parsing.


def _compile_node(node: Mapping[str, Any]) -> Callable[[Mapping[str, str]], bool]:
    predicates: list[Callable[[Mapping[str, str]], bool]] = []
    for key, spec in node.items():
        if key == "$and":
            predicates.append(_compile_all([_compile_node(sub) for sub in spec]))
        elif key == "$or":
            predicates.append(_compile_any([_compile_node(sub) for sub in spec]))
        elif key == "$not":
            inner = _compile_node(spec)
            predicates.append(lambda meta, _inner=inner: not _inner(meta))
        else:
            predicates.append(_compile_field(key, spec))
    return _compile_all(predicates)


def _compile_all(
    predicates: list[Callable[[Mapping[str, str]], bool]],
) -> Callable[[Mapping[str, str]], bool]:
    if len(predicates) == 1:
        return predicates[0]
    children = tuple(predicates)

    def _all(meta: Mapping[str, str]) -> bool:
        for predicate in children:
            if not predicate(meta):
                return False
        return True

    return _all


def _compile_any(
    predicates: list[Callable[[Mapping[str, str]], bool]],
) -> Callable[[Mapping[str, str]], bool]:
    children = tuple(predicates)

    def _any(meta: Mapping[str, str]) -> bool:
        for predicate in children:
            if predicate(meta):
                return True
        return False

    return _any


def _compile_field(field: str, spec: Any) -> Callable[[Mapping[str, str]], bool]:
    if not isinstance(spec, Mapping):
        # Bare scalar is exact-match sugar for ``$eq``.
        return lambda meta, _f=field, _v=spec: meta.get(_f, _MISSING) == _v
    return _compile_all(
        [_compile_operator(field, op, operand) for op, operand in spec.items()]
    )


def _compile_operator(
    field: str, op: str, operand: Any
) -> Callable[[Mapping[str, str]], bool]:
    if op == "$eq":
        return lambda meta, _f=field, _v=operand: meta.get(_f, _MISSING) == _v
    if op == "$ne":
        # A missing key satisfies $ne: the sentinel never equals an operand.
        return lambda meta, _f=field, _v=operand: meta.get(_f, _MISSING) != _v
    if op == "$in":
        return lambda meta, _f=field, _v=operand: meta.get(_f, _MISSING) in _v
    if op == "$nin":
        return lambda meta, _f=field, _v=operand: meta.get(_f, _MISSING) not in _v
    if op == "$exists":
        if operand:
            return lambda meta, _f=field: _f in meta
        return lambda meta, _f=field: _f not in meta
    return _compile_ordered(field, op, operand)


def _compile_ordered(
    field: str, op: str, operand: str
) -> Callable[[Mapping[str, str]], bool]:
    compare = _ORDERED_CMP[op]
    operand_number = _as_number(operand)
    if operand_number is None:
        # The operand is not a finite number, so a numeric compare is impossible
        # (both sides must parse); it is always lexicographic, and the stored
        # value never needs parsing.
        def _ordered_str(
            meta: Mapping[str, str], _f: str = field, _v: str = operand
        ) -> bool:
            stored = meta.get(_f, _MISSING)
            return stored is not _MISSING and compare(stored, _v)

        return _ordered_str

    def _ordered_num(
        meta: Mapping[str, str],
        _f: str = field,
        _v: str = operand,
        _vn: float = operand_number,
    ) -> bool:
        stored = meta.get(_f, _MISSING)
        if stored is _MISSING:
            return False
        stored_number = _as_number(stored)
        if stored_number is None:
            return compare(stored, _v)  # stored not numeric -> lexicographic
        return compare(stored_number, _vn)

    return _ordered_num


def _as_number(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    # NaN is unordered; fall back to a string compare so ordering stays total.
    if math.isnan(value):
        return None
    return value
