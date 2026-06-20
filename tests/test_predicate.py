"""Unit tests for the metadata filter predicate engine (lodedb.engine._predicate).

These exercise the grammar, the two stringify rules, and the full comparison
semantics matrix directly, without the engine/SDK/TurboVec stack.
"""

from __future__ import annotations

import pytest

from lodedb.engine._predicate import (
    coerce_sdk_filter,
    compile_metadata_filter,
    matches_metadata_filter,
    validate_metadata_filter,
)


def _matches(doc: dict[str, str], raw_filter: dict) -> bool:
    """Validates a raw filter (engine rule) then evaluates it against a document."""

    return matches_metadata_filter(doc, validate_metadata_filter(raw_filter))


# --- equality / bare scalar -------------------------------------------------


@pytest.mark.parametrize("spec", ["1", {"$eq": "1"}, {"$eq": 1}])
def test_eq_and_bare_scalar_match(spec):
    assert _matches({"a": "1"}, {"a": spec}) is True
    assert _matches({"a": "2"}, {"a": spec}) is False
    assert _matches({"b": "1"}, {"a": spec}) is False  # missing key


def test_ne_excludes_matches_and_includes_missing():
    assert _matches({"a": "1"}, {"a": {"$ne": "2"}}) is True
    assert _matches({"a": "1"}, {"a": {"$ne": "1"}}) is False
    assert _matches({"b": "1"}, {"a": {"$ne": "1"}}) is True  # missing -> $ne true


# --- ordered comparisons ----------------------------------------------------


def test_numeric_range():
    doc = {"year": "2020"}
    assert _matches(doc, {"year": {"$gte": 2020}}) is True
    assert _matches(doc, {"year": {"$gt": 2020}}) is False
    assert _matches(doc, {"year": {"$lt": 2021}}) is True
    assert _matches(doc, {"year": {"$gte": 2019, "$lt": 2025}}) is True  # AND
    assert _matches(doc, {"year": {"$gte": 2021}}) is False
    assert _matches({"other": "x"}, {"year": {"$gte": 2020}}) is False  # missing


def test_ordered_non_numeric_is_lexicographic():
    doc = {"name": "banana"}
    assert _matches(doc, {"name": {"$gt": "apple"}}) is True
    assert _matches(doc, {"name": {"$lt": "cherry"}}) is True
    assert _matches(doc, {"name": {"$gt": "cherry"}}) is False


def test_ordered_mixed_type_falls_back_to_string_compare():
    # stored non-numeric, operand numeric -> only one parses -> lexicographic.
    assert _matches({"x": "abc"}, {"x": {"$gt": 1}}) is True  # "abc" > "1"
    assert _matches({"x": "abc"}, {"x": {"$lt": 1}}) is False


def test_nan_falls_back_to_string_compare_deterministically():
    # NaN is unordered; both sides must parse as finite numbers to compare numerically.
    assert _matches({"v": "nan"}, {"v": {"$gt": "0"}}) is True  # "nan" > "0"
    assert _matches({"v": "5"}, {"v": {"$gt": "nan"}}) is False  # "5" < "nan"


def test_inf_compares_numerically():
    assert _matches({"v": "1e9"}, {"v": {"$lt": "inf"}}) is True
    assert _matches({"v": "inf"}, {"v": {"$gte": "1e9"}}) is True


# --- membership / existence -------------------------------------------------


def test_in_and_nin():
    assert _matches({"c": "x"}, {"c": {"$in": ["x", "y"]}}) is True
    assert _matches({"c": "z"}, {"c": {"$in": ["x", "y"]}}) is False
    assert _matches({"c": "z"}, {"c": {"$nin": ["x", "y"]}}) is True
    assert _matches({"d": "x"}, {"c": {"$in": ["x"]}}) is False  # missing -> false
    assert _matches({"d": "x"}, {"c": {"$nin": ["x"]}}) is True  # missing -> true


def test_in_coerces_numeric_members():
    assert _matches({"year": "2020"}, {"year": {"$in": [2020, 2021]}}) is True


def test_exists():
    assert _matches({"a": "1"}, {"a": {"$exists": True}}) is True
    assert _matches({"a": "1"}, {"a": {"$exists": False}}) is False
    assert _matches({"b": "1"}, {"a": {"$exists": True}}) is False
    assert _matches({"b": "1"}, {"a": {"$exists": False}}) is True


def test_exists_true_for_explicit_null_stored_as_empty_string():
    # A None document value is stored as "" by the engine -> the key is present.
    assert _matches({"n": ""}, {"n": {"$exists": True}}) is True
    assert _matches({"n": ""}, {"n": {"$exists": False}}) is False


# --- logical composition ----------------------------------------------------


def test_and():
    flt = {"$and": [{"a": "1"}, {"b": "2"}]}
    assert _matches({"a": "1", "b": "2"}, flt) is True
    assert _matches({"a": "1"}, flt) is False


def test_or():
    flt = {"$or": [{"a": "1"}, {"b": "2"}]}
    assert _matches({"b": "2"}, flt) is True
    assert _matches({"c": "3"}, flt) is False


def test_not():
    assert _matches({"a": "2"}, {"$not": {"a": "1"}}) is True
    assert _matches({"a": "1"}, {"$not": {"a": "1"}}) is False


def test_nested_logical_and_field_mix():
    flt = {"$or": [{"year": {"$gte": 2020}}, {"topic": "x"}]}
    assert _matches({"year": "2021"}, flt) is True
    assert _matches({"topic": "x"}, flt) is True
    assert _matches({"year": "2019", "topic": "y"}, flt) is False

    mixed = {"a": "1", "$or": [{"b": "2"}, {"c": "3"}]}
    assert _matches({"a": "1", "c": "3"}, mixed) is True
    assert _matches({"a": "1", "d": "4"}, mixed) is False  # the $or fails
    assert _matches({"b": "2", "c": "3"}, mixed) is False  # a != 1


# --- the two stringify rules ------------------------------------------------


def test_engine_rule_uses_capitalized_bool():
    # Direct-engine / HTTP path: validate stringifies bool via str() -> "True".
    validated = validate_metadata_filter({"fresh": {"$eq": True}})
    assert validated == {"fresh": {"$eq": "True"}}
    assert matches_metadata_filter({"fresh": "True"}, validated) is True
    assert matches_metadata_filter({"fresh": "true"}, validated) is False


def test_sdk_rule_uses_lowercase_bool():
    # SDK path: coerce lowercases bools to match SDK-stored metadata.
    coerced = coerce_sdk_filter({"fresh": True})
    assert coerced == {"fresh": "true"}
    # Re-validating the coerced filter is a no-op (operand already a string).
    assert validate_metadata_filter(coerced) == {"fresh": "true"}
    assert matches_metadata_filter({"fresh": "true"}, coerced) is True


def test_coerce_sdk_filter_shapes():
    assert coerce_sdk_filter({"year": 2020}) == {"year": "2020"}
    assert coerce_sdk_filter({"fresh": False}) == {"fresh": "false"}
    assert coerce_sdk_filter({"a": None}) == {"a": ""}
    assert coerce_sdk_filter({"c": {"$in": [1, 2]}}) == {"c": {"$in": ("1", "2")}}
    assert coerce_sdk_filter({"a": {"$exists": True}}) == {"a": {"$exists": True}}


# --- invalid grammar --------------------------------------------------------


def test_empty_filter_rejected():
    with pytest.raises(ValueError, match="nonempty object"):
        validate_metadata_filter({})


@pytest.mark.parametrize(
    "bad",
    [
        {"$foo": 1},  # unknown operator at field level
        {"a": {"$foo": 1}},  # unknown operator in map
        {"a": {"$in": "x"}},  # $in not a list
        {"a": {"$in": []}},  # $in empty
        {"a": {"$nin": []}},  # $nin empty
        {"$and": {"a": "1"}},  # $and not a list
        {"$and": []},  # $and empty
        {"$or": []},  # $or empty
        {"$not": [1]},  # $not not a mapping
        {"a": {"$exists": "yes"}},  # $exists not a bool
        {"a": {}},  # empty operator map
        {"a": {"$eq": {"x": 1}}},  # operand not scalar
        {"a": {"$eq": ["x"]}},  # operand is a list
    ],
)
def test_invalid_grammar_raises(bad):
    with pytest.raises(ValueError):
        validate_metadata_filter(bad)


def test_excessive_nesting_fails_closed():
    flt: dict = {"a": "1"}
    for _ in range(40):
        flt = {"$not": flt}
    with pytest.raises(ValueError, match="nested too deeply"):
        validate_metadata_filter(flt)


# --- compile-once reuse -----------------------------------------------------


def test_compile_metadata_filter_is_reusable_and_agrees_with_oneshot():
    """A compiled predicate is reused across documents and matches the one-shot path.

    The engine compiles a filter once and evaluates it against the whole corpus;
    this pins that the compiled predicate is a single reusable callable whose
    verdicts are identical to ``matches_metadata_filter`` (which compiles per call).
    """

    validated = validate_metadata_filter(
        {"$or": [{"year": {"$gte": 2020}}, {"topic": "x"}], "lang": {"$ne": "fr"}}
    )
    predicate = compile_metadata_filter(validated)
    docs = [
        {"year": "2021", "lang": "en"},
        {"topic": "x", "lang": "en"},
        {"year": "2019", "topic": "y", "lang": "en"},
        {"year": "2021", "lang": "fr"},  # $ne excludes
        {},  # missing keys: $or fails
    ]
    # The same compiled object handles every document and agrees with one-shot.
    assert [predicate(doc) for doc in docs] == [
        matches_metadata_filter(doc, validated) for doc in docs
    ]
    assert [predicate(doc) for doc in docs] == [True, True, False, False, False]
