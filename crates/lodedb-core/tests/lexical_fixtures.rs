use std::collections::{BTreeMap, BTreeSet};

use lodedb_core::lexical::bm25::DocumentTokenLists;
use lodedb_core::lexical::{build_chunk_token_lists, reciprocal_rank_fusion, tokenize, Bm25Index};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct LexicalFixtures {
    token_cases: Vec<TokenCase>,
    corpora: Vec<Corpus>,
    rank_cases: Vec<RankCase>,
    incremental_case: IncrementalCase,
    rrf_cases: Vec<RrfCase>,
    chunk_token_case: ChunkTokenCase,
}

#[derive(Debug, Deserialize)]
struct TokenCase {
    text: String,
    tokens: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct Corpus {
    name: String,
    units: Vec<Unit>,
}

#[derive(Debug, Deserialize)]
struct Unit {
    id: String,
    text: String,
    group: String,
}

#[derive(Debug, Deserialize)]
struct RankCase {
    query: String,
    rank: Vec<(String, f64)>,
    corpus: Option<String>,
    limit: Option<usize>,
    allowed_unit_ids: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct IncrementalCase {
    base: Vec<Unit>,
    replacement_group: String,
    replacement_units: Vec<(String, Vec<String>)>,
    remove_group: String,
    unit_ids: Vec<String>,
    ranks: BTreeMap<String, Vec<(String, f64)>>,
}

#[derive(Debug, Deserialize)]
struct RrfCase {
    rankings: Vec<Vec<String>>,
    c: f64,
    weights: Option<Vec<f64>>,
    fused: Vec<(String, f64)>,
}

#[derive(Debug, Deserialize)]
struct ChunkTokenCase {
    documents: Vec<DocumentTokenLists>,
    expected: ChunkTokenExpected,
}

#[derive(Debug, Deserialize, PartialEq)]
struct ChunkTokenExpected {
    chunk_ids: Vec<String>,
    token_lists: Vec<Vec<String>>,
    group_ids: Vec<String>,
}

fn fixtures() -> LexicalFixtures {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/native_core_lexical/lexical.json"
    ))
    .expect("lexical fixture must parse")
}

fn build_index(units: &[Unit]) -> Bm25Index {
    let unit_ids = units.iter().map(|unit| unit.id.clone()).collect::<Vec<_>>();
    let texts = units
        .iter()
        .map(|unit| unit.text.clone())
        .collect::<Vec<_>>();
    let group_ids = units
        .iter()
        .map(|unit| unit.group.clone())
        .collect::<Vec<_>>();
    Bm25Index::from_texts(&unit_ids, &texts, Some(&group_ids)).unwrap()
}

fn rounded_rank(rows: Vec<(String, f64)>) -> Vec<(String, f64)> {
    rows.into_iter()
        .map(|(unit_id, score)| {
            (
                unit_id,
                (score * 1_000_000_000_000.0).round() / 1_000_000_000_000.0,
            )
        })
        .collect()
}

#[test]
fn tokenizer_matches_python_oracle() {
    for case in fixtures().token_cases {
        assert_eq!(tokenize(&case.text), case.tokens);
    }
}

#[test]
fn bm25_rank_cases_match_python_oracle() {
    let fixtures = fixtures();
    let corpora = fixtures
        .corpora
        .iter()
        .map(|corpus| (corpus.name.clone(), build_index(&corpus.units)))
        .collect::<BTreeMap<_, _>>();

    for case in fixtures.rank_cases {
        let corpus = case.corpus.as_deref().unwrap_or("fixed");
        let index = corpora.get(corpus).expect("fixture corpus must exist");
        let allowed = case.allowed_unit_ids.map(|unit_ids| {
            unit_ids
                .iter()
                .filter_map(|unit_id| index.position_of(unit_id))
                .collect::<BTreeSet<_>>()
        });
        assert_eq!(
            rounded_rank(index.rank(&case.query, case.limit, allowed.as_ref())),
            case.rank,
            "rank mismatch for query {:?}",
            case.query
        );
    }
}

#[test]
fn incremental_replace_group_matches_python_oracle() {
    let case = fixtures().incremental_case;
    let mut index = build_index(&case.base);
    index.replace_group(&case.replacement_group, &case.replacement_units);
    index.remove_group(&case.remove_group);
    assert_eq!(
        index.unit_ids(),
        case.unit_ids.into_iter().collect::<BTreeSet<_>>()
    );
    for (query, expected) in case.ranks {
        assert_eq!(rounded_rank(index.rank(&query, None, None)), expected);
    }
}

#[test]
fn rrf_matches_python_oracle() {
    for case in fixtures().rrf_cases {
        assert_eq!(
            rounded_rank(
                reciprocal_rank_fusion(&case.rankings, case.c, case.weights.as_deref()).unwrap()
            ),
            case.fused
        );
    }
}

#[test]
fn chunk_token_lists_match_python_oracle() {
    let case = fixtures().chunk_token_case;
    let (chunk_ids, token_lists, group_ids) = build_chunk_token_lists(&case.documents);
    assert_eq!(
        ChunkTokenExpected {
            chunk_ids,
            token_lists,
            group_ids,
        },
        case.expected
    );
}
