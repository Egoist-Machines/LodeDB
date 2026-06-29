use std::collections::BTreeMap;

use lodedb_core::filter::{
    build_field_indexes, coerce_sdk_filter, expand_doc_ids_to_chunk_ids, matches_metadata_filter,
    resolve_filter, validate_metadata_filter, DocSet,
};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize)]
struct PredicateFixtures {
    documents: Vec<Document>,
    cases: Vec<PredicateCase>,
}

#[derive(Debug, Deserialize)]
struct Document {
    id: String,
    metadata: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct PredicateCase {
    filter: Value,
    engine_matches: Vec<String>,
    sdk_matches: Vec<String>,
}

fn fixtures() -> PredicateFixtures {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/native_core_predicate/predicate.json"
    ))
    .expect("predicate fixture must parse")
}

fn metadata_by_id(documents: &[Document]) -> BTreeMap<String, BTreeMap<String, String>> {
    documents
        .iter()
        .map(|document| (document.id.clone(), document.metadata.clone()))
        .collect()
}

fn matching_ids(documents: &[Document], validated: &Value) -> Vec<String> {
    documents
        .iter()
        .filter(|document| matches_metadata_filter(&document.metadata, validated).unwrap())
        .map(|document| document.id.clone())
        .collect()
}

fn sorted_set(ids: &[String]) -> DocSet {
    ids.iter().cloned().collect()
}

#[test]
fn filter_plan_fixtures_match_python_oracle() {
    let fixtures = fixtures();
    let metadata = metadata_by_id(&fixtures.documents);
    let (fields, all_docs) = build_field_indexes(&metadata);

    for case in fixtures.cases {
        let engine_validated = validate_metadata_filter(&case.filter).unwrap();
        let engine_planned = resolve_filter(&engine_validated, &fields, &all_docs).unwrap();
        assert_eq!(engine_planned, sorted_set(&case.engine_matches));
        assert_eq!(
            matching_ids(&fixtures.documents, &engine_validated),
            case.engine_matches
        );

        let sdk_validated = coerce_sdk_filter(&case.filter).unwrap();
        let sdk_planned = resolve_filter(&sdk_validated, &fields, &all_docs).unwrap();
        assert_eq!(sdk_planned, sorted_set(&case.sdk_matches));
        assert_eq!(
            matching_ids(&fixtures.documents, &sdk_validated),
            case.sdk_matches
        );
    }
}

#[test]
fn expands_document_allowlist_to_chunk_allowlist() {
    let document_ids: DocSet = ["d0".to_string(), "d2".to_string(), "missing".to_string()]
        .into_iter()
        .collect();
    let document_chunks = BTreeMap::from([
        (
            "d0".to_string(),
            vec!["d0:0".to_string(), "d0:1".to_string()],
        ),
        ("d1".to_string(), vec!["d1:0".to_string()]),
        ("d2".to_string(), vec!["d2:0".to_string()]),
    ]);

    assert_eq!(
        expand_doc_ids_to_chunk_ids(&document_ids, &document_chunks),
        ["d0:0".to_string(), "d0:1".to_string(), "d2:0".to_string()]
            .into_iter()
            .collect()
    );
}
