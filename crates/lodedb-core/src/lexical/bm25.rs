//! Okapi BM25 inverted index matching the Python lexical oracle.

use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreErrorCode};
use crate::lexical::tokenize::tokenize;

/// Classic Okapi BM25 term-frequency saturation constant.
pub const BM25_K1: f64 = 1.2;
/// Classic Okapi BM25 document-length normalization constant.
pub const BM25_B: f64 = 0.75;

/// In-memory Okapi BM25 index over a unit/chunk id space.
#[derive(Debug, Clone)]
pub struct Bm25Index {
    postings: HashMap<String, HashMap<usize, usize>>,
    doc_freq: HashMap<String, usize>,
    doc_len: HashMap<usize, usize>,
    unit_id_by_pos: HashMap<usize, String>,
    pos_by_unit_id: HashMap<String, usize>,
    terms_by_pos: HashMap<usize, Vec<String>>,
    group_by_pos: HashMap<usize, String>,
    positions_by_group: HashMap<String, HashSet<usize>>,
    n: usize,
    total_len: usize,
    next_pos: usize,
    k1: f64,
    b: f64,
}

impl Bm25Index {
    /// Builds a BM25 index from raw unit texts.
    pub fn from_texts(
        unit_ids: &[String],
        texts: &[String],
        group_ids: Option<&[String]>,
    ) -> Result<Self, CoreError> {
        if unit_ids.len() != texts.len() {
            return invalid("unit_ids and texts must be the same length");
        }
        let token_lists = texts.iter().map(|text| tokenize(text)).collect::<Vec<_>>();
        Self::from_token_lists(unit_ids, &token_lists, group_ids)
    }

    /// Builds a BM25 index from already-tokenized units.
    pub fn from_token_lists(
        unit_ids: &[String],
        token_lists: &[Vec<String>],
        group_ids: Option<&[String]>,
    ) -> Result<Self, CoreError> {
        if unit_ids.len() != token_lists.len() {
            return invalid("unit_ids and token_lists must be the same length");
        }
        if group_ids.is_some_and(|groups| groups.len() != unit_ids.len()) {
            return invalid("group_ids must align with unit_ids");
        }
        let mut index = Self::empty();
        for (position, unit_id) in unit_ids.iter().enumerate() {
            let group_id = group_ids
                .and_then(|groups| groups.get(position))
                .unwrap_or(unit_id);
            index.add_at(unit_id.clone(), &token_lists[position], group_id.clone());
        }
        Ok(index)
    }

    /// Creates an empty BM25 index.
    pub fn empty() -> Self {
        Self {
            postings: HashMap::new(),
            doc_freq: HashMap::new(),
            doc_len: HashMap::new(),
            unit_id_by_pos: HashMap::new(),
            pos_by_unit_id: HashMap::new(),
            terms_by_pos: HashMap::new(),
            group_by_pos: HashMap::new(),
            positions_by_group: HashMap::new(),
            n: 0,
            total_len: 0,
            next_pos: 0,
            k1: BM25_K1,
            b: BM25_B,
        }
    }

    /// Adds or upserts one tokenized unit.
    pub fn add_unit(&mut self, unit_id: &str, tokens: &[String], group_id: Option<&str>) {
        if self.pos_by_unit_id.contains_key(unit_id) {
            self.remove_unit(unit_id);
        }
        let group_id = group_id.unwrap_or(unit_id).to_string();
        self.add_at(unit_id.to_string(), tokens, group_id);
    }

    /// Removes one unit; no-op if absent.
    pub fn remove_unit(&mut self, unit_id: &str) {
        if let Some(pos) = self.pos_by_unit_id.get(unit_id).copied() {
            self.remove_at(pos);
        }
    }

    /// Removes all units in a document group; no-op if absent.
    pub fn remove_group(&mut self, group_id: &str) {
        let Some(positions) = self.positions_by_group.get(group_id) else {
            return;
        };
        for pos in positions.iter().copied().collect::<Vec<_>>() {
            self.remove_at(pos);
        }
    }

    /// Replaces one document group with the provided `(unit_id, tokens)` units.
    pub fn replace_group(&mut self, group_id: &str, units: &[(String, Vec<String>)]) {
        self.remove_group(group_id);
        for (unit_id, tokens) in units {
            self.add_unit(unit_id, tokens, Some(group_id));
        }
    }

    /// Returns the stable position for a unit id.
    pub fn position_of(&self, unit_id: &str) -> Option<usize> {
        self.pos_by_unit_id.get(unit_id).copied()
    }

    /// Returns indexed unit ids.
    pub fn unit_ids(&self) -> BTreeSet<String> {
        self.pos_by_unit_id.keys().cloned().collect()
    }

    /// Returns the number of indexed units.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Returns whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Scores a query and returns `(unit_id, score)` best-first.
    pub fn rank(
        &self,
        query: &str,
        limit: Option<usize>,
        allowed_indices: Option<&BTreeSet<usize>>,
    ) -> Vec<(String, f64)> {
        if self.n == 0 {
            return Vec::new();
        }
        let query_terms = tokenize(query).into_iter().collect::<BTreeSet<_>>();
        if query_terms.is_empty() {
            return Vec::new();
        }
        let mut scores: HashMap<usize, f64> = HashMap::new();
        let avgdl = if self.n == 0 {
            1.0
        } else {
            let value = self.total_len as f64 / self.n as f64;
            if value == 0.0 {
                1.0
            } else {
                value
            }
        };
        for term in query_terms {
            let Some(posting) = self.postings.get(&term) else {
                continue;
            };
            let idf = self.idf(&term);
            if idf <= 0.0 {
                continue;
            }
            for (pos, term_frequency) in posting {
                if allowed_indices.is_some_and(|allowed| !allowed.contains(pos)) {
                    continue;
                }
                let length = *self.doc_len.get(pos).unwrap_or(&0) as f64;
                let term_frequency = *term_frequency as f64;
                let denominator =
                    term_frequency + self.k1 * (1.0 - self.b + self.b * (length / avgdl));
                if denominator <= 0.0 {
                    continue;
                }
                let contribution = idf * (term_frequency * (self.k1 + 1.0)) / denominator;
                *scores.entry(*pos).or_insert(0.0) += contribution;
            }
        }
        let mut ranked = scores
            .into_iter()
            .filter_map(|(pos, score)| self.unit_id_by_pos.get(&pos).map(|id| (id.clone(), score)))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        if let Some(limit) = limit {
            ranked.truncate(limit);
        }
        ranked
    }

    fn add_at(&mut self, unit_id: String, tokens: &[String], group_id: String) {
        let pos = self.next_pos;
        self.next_pos += 1;
        self.doc_len.insert(pos, tokens.len());
        self.total_len += tokens.len();
        self.n += 1;
        self.unit_id_by_pos.insert(pos, unit_id.clone());
        self.pos_by_unit_id.insert(unit_id, pos);
        self.group_by_pos.insert(pos, group_id.clone());
        self.positions_by_group
            .entry(group_id)
            .or_default()
            .insert(pos);

        let mut counts: HashMap<String, usize> = HashMap::new();
        for token in tokens {
            *counts.entry(token.clone()).or_insert(0) += 1;
        }
        for (token, frequency) in &counts {
            self.postings
                .entry(token.clone())
                .or_default()
                .insert(pos, *frequency);
            *self.doc_freq.entry(token.clone()).or_insert(0) += 1;
        }
        self.terms_by_pos.insert(pos, counts.into_keys().collect());
    }

    fn remove_at(&mut self, pos: usize) {
        for term in self.terms_by_pos.remove(&pos).unwrap_or_default() {
            if let Some(posting) = self.postings.get_mut(&term) {
                posting.remove(&pos);
                if posting.is_empty() {
                    self.postings.remove(&term);
                    self.doc_freq.remove(&term);
                } else {
                    self.doc_freq.insert(term, posting.len());
                }
            }
        }
        self.total_len -= self.doc_len.remove(&pos).unwrap_or(0);
        if let Some(unit_id) = self.unit_id_by_pos.remove(&pos) {
            self.pos_by_unit_id.remove(&unit_id);
        }
        if let Some(group_id) = self.group_by_pos.remove(&pos) {
            if let Some(positions) = self.positions_by_group.get_mut(&group_id) {
                positions.remove(&pos);
                if positions.is_empty() {
                    self.positions_by_group.remove(&group_id);
                }
            }
        }
        self.n -= 1;
    }

    fn idf(&self, term: &str) -> f64 {
        let df = *self.doc_freq.get(term).unwrap_or(&0);
        if df == 0 {
            return 0.0;
        }
        (1.0 + (self.n as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln()
    }
}

/// Flattened persisted-token input for one document.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DocumentTokenLists {
    pub document_id: String,
    pub chunk_ids: Vec<String>,
    pub token_lists: Vec<Vec<String>>,
}

/// Flattens per-document token lists into `(chunk_ids, token_lists, group_ids)`.
pub fn build_chunk_token_lists(
    documents: &[DocumentTokenLists],
) -> (Vec<String>, Vec<Vec<String>>, Vec<String>) {
    let mut chunk_ids = Vec::new();
    let mut token_lists = Vec::new();
    let mut group_ids = Vec::new();
    for document in documents {
        if document.token_lists.is_empty() {
            continue;
        }
        for (chunk_id, tokens) in document.chunk_ids.iter().zip(&document.token_lists) {
            chunk_ids.push(chunk_id.clone());
            token_lists.push(tokens.clone());
            group_ids.push(document.document_id.clone());
        }
    }
    (chunk_ids, token_lists, group_ids)
}

fn invalid<T>(message: impl Into<String>) -> Result<T, CoreError> {
    Err(CoreError::new(CoreErrorCode::InvalidArgument, message))
}

#[cfg(test)]
mod tests {
    use super::{Bm25Index, DocumentTokenLists};
    use crate::lexical::bm25::build_chunk_token_lists;
    use crate::lexical::tokenize;

    #[test]
    fn ranks_and_filters_by_allowed_positions() {
        let index = Bm25Index::from_texts(
            &["u0".to_string(), "u1".to_string(), "u2".to_string()],
            &[
                "match here".to_string(),
                "match again".to_string(),
                "no overlap".to_string(),
            ],
            None,
        )
        .unwrap();
        let allowed = [index.position_of("u1").unwrap()].into_iter().collect();
        assert_eq!(index.rank("match", None, Some(&allowed))[0].0, "u1");
    }

    #[test]
    fn replace_group_matches_rebuild() {
        let mut index = Bm25Index::from_texts(
            &["a#0".to_string(), "a#1".to_string(), "b#0".to_string()],
            &[
                "alpha beta".to_string(),
                "gamma".to_string(),
                "delta epsilon".to_string(),
            ],
            Some(&["docA".to_string(), "docA".to_string(), "docB".to_string()]),
        )
        .unwrap();
        index.replace_group(
            "docA",
            &[("a#2".to_string(), tokenize("alpha gamma e1234"))],
        );
        let fresh = Bm25Index::from_texts(
            &["a#2".to_string(), "b#0".to_string()],
            &["alpha gamma e1234".to_string(), "delta epsilon".to_string()],
            Some(&["docA".to_string(), "docB".to_string()]),
        )
        .unwrap();
        assert_eq!(index.unit_ids(), fresh.unit_ids());
        assert_eq!(
            index.rank("e1234", None, None),
            fresh.rank("e1234", None, None)
        );
    }

    #[test]
    fn flattens_chunk_token_lists() {
        let documents = vec![
            DocumentTokenLists {
                document_id: "doc-a".to_string(),
                chunk_ids: vec!["a#0".to_string(), "a#1".to_string()],
                token_lists: vec![vec!["alpha".to_string()], vec!["beta".to_string()]],
            },
            DocumentTokenLists {
                document_id: "doc-b".to_string(),
                chunk_ids: vec!["b#0".to_string()],
                token_lists: vec![vec!["gamma".to_string()]],
            },
        ];
        assert_eq!(
            build_chunk_token_lists(&documents),
            (
                vec!["a#0".to_string(), "a#1".to_string(), "b#0".to_string()],
                vec![
                    vec!["alpha".to_string()],
                    vec!["beta".to_string()],
                    vec!["gamma".to_string()]
                ],
                vec![
                    "doc-a".to_string(),
                    "doc-a".to_string(),
                    "doc-b".to_string()
                ]
            )
        );
    }
}
