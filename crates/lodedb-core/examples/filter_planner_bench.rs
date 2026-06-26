use std::collections::BTreeMap;
use std::time::Instant;

use lodedb_core::filter::{build_field_indexes, coerce_sdk_filter, resolve_filter};
use serde::Deserialize;
use serde_json::{json, Value};

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
}

fn main() {
    let iterations = parse_iterations();
    let fixtures: PredicateFixtures = serde_json::from_str(include_str!(
        "../../../tests/fixtures/native_core_predicate/predicate.json"
    ))
    .expect("predicate fixture must parse");
    let metadata: BTreeMap<String, BTreeMap<String, String>> = fixtures
        .documents
        .iter()
        .map(|document| (document.id.clone(), document.metadata.clone()))
        .collect();
    let filters: Vec<Value> = fixtures
        .cases
        .iter()
        .map(|case| coerce_sdk_filter(&case.filter).expect("fixture filter must validate"))
        .collect();
    let (fields, all_docs) = build_field_indexes(&metadata);

    let start = Instant::now();
    let mut checksum = 0usize;
    for _ in 0..iterations {
        for filter in &filters {
            checksum += resolve_filter(filter, &fields, &all_docs)
                .expect("filter plan must resolve")
                .len();
        }
    }
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{}",
        json!({
            "name": "rust_planner",
            "cases": filters.len(),
            "documents": metadata.len(),
            "iterations": iterations,
            "elapsed_ms": round_millis(elapsed_ms),
            "checksum": checksum,
        })
    );
}

fn parse_iterations() -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--iterations" {
            return args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(10_000);
        }
    }
    10_000
}

fn round_millis(value: f64) -> f64 {
    (value * 1000.0).round() / 1000.0
}
