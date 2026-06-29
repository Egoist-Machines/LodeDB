use std::collections::BTreeMap;

use lodedb_core::filter::{coerce_sdk_filter, matches_metadata_filter, validate_metadata_filter};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct PredicateFixtures {
    documents: Vec<Document>,
    cases: Vec<PredicateCase>,
    invalid_filters: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct Document {
    id: String,
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct PredicateCase {
    filter: Value,
    engine_validated: Value,
    sdk_validated: Value,
    engine_matches: Vec<String>,
    sdk_matches: Vec<String>,
}

fn fixtures() -> PredicateFixtures {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/native_core_predicate/predicate.json"
    ))
    .expect("predicate fixture must parse")
}

fn matching_ids(documents: &[Document], validated: &Value) -> Vec<String> {
    documents
        .iter()
        .filter(|document| matches_metadata_filter(&document.metadata, validated).unwrap())
        .map(|document| document.id.clone())
        .collect()
}

#[test]
fn predicate_fixtures_match_python_oracle() {
    let fixtures = fixtures();
    for case in fixtures.cases {
        let engine_validated = validate_metadata_filter(&case.filter).unwrap();
        assert_eq!(engine_validated, case.engine_validated);
        assert_eq!(
            matching_ids(&fixtures.documents, &engine_validated),
            case.engine_matches
        );

        let sdk_validated = coerce_sdk_filter(&case.filter).unwrap();
        assert_eq!(sdk_validated, case.sdk_validated);
        assert_eq!(
            matching_ids(&fixtures.documents, &sdk_validated),
            case.sdk_matches
        );
    }
}

#[test]
fn invalid_predicate_fixtures_fail_closed() {
    for filter in fixtures().invalid_filters {
        assert!(validate_metadata_filter(&filter).is_err());
        assert!(coerce_sdk_filter(&filter).is_err());
    }
}
