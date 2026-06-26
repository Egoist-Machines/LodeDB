from __future__ import annotations

import json
from pathlib import Path

from lodedb.engine._predicate import (
    coerce_sdk_filter,
    compile_metadata_filter,
    validate_metadata_filter,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "native_core_predicate" / "predicate.json"


def _matches(documents: list[dict], validated: dict) -> list[str]:
    predicate = compile_metadata_filter(validated)
    return [document["id"] for document in documents if predicate(document["metadata"])]


def _jsonable(value: object) -> object:
    return json.loads(json.dumps(value))


def test_native_core_predicate_fixture_matches_python_oracle() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    documents = fixture["documents"]

    for case in fixture["cases"]:
        engine_validated = validate_metadata_filter(case["filter"])
        assert _jsonable(engine_validated) == case["engine_validated"]
        assert _matches(documents, engine_validated) == case["engine_matches"]

        sdk_validated = coerce_sdk_filter(case["filter"])
        assert _jsonable(sdk_validated) == case["sdk_validated"]
        assert _matches(documents, sdk_validated) == case["sdk_matches"]

    for bad_filter in fixture["invalid_filters"]:
        try:
            validate_metadata_filter(bad_filter)
        except ValueError:
            pass
        else:  # pragma: no cover - assertion branch
            raise AssertionError(f"filter should be invalid: {bad_filter!r}")
