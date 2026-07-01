//! Okapi BM25 inverted index matching the Python lexical oracle.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::error::{CoreError, CoreErrorCode};
use crate::lexical::tokenize::tokenize;

/// Classic Okapi BM25 term-frequency saturation constant.
pub const BM25_K1: f64 = 1.2;
/// Classic Okapi BM25 document-length normalization constant.
pub const BM25_B: f64 = 0.75;

/// In-memory Okapi BM25 index over a unit/chunk id space.
///
/// Mutation keeps the `postings`/`doc_freq`/`doc_len` maps as the authoritative
/// builder + delta layer. Scoring runs off a compact [`ServingIndex`] snapshot
/// that is rebuilt lazily on the first query after any mutation (the `serving`
/// cache), so a query never pays for the map indirection and the ranker can use
/// MaxScore pruning against precomputed per-term score upper bounds. The
/// snapshot is derived state, so it never changes what `rank` returns; it only
/// changes how fast the top-k is found.
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
    /// Compact scoring snapshot; `None` means "dirty, rebuild on next query".
    /// Interior mutability keeps `rank(&self, ...)` read-only for the engine.
    serving: RefCell<Option<ServingIndex>>,
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
            serving: RefCell::new(None),
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
    ///
    /// Semantically identical to a full BM25 scan of every matching unit
    /// followed by a sort on `(score desc, unit_id asc)`, but a bounded top-k
    /// request (`limit = Some(k)`, the engine's only query shape) is answered by
    /// MaxScore over the compact serving snapshot: it keeps a size-`k` heap and
    /// skips postings that provably cannot enter it, so it stops fully sorting
    /// all matches. The returned ids, order, and scores are bit-for-bit what the
    /// exhaustive path would return. `limit = None` (score everything) has no
    /// top-k threshold to prune against, so it uses the exhaustive path directly.
    pub fn rank(
        &self,
        query: &str,
        limit: Option<usize>,
        allowed_indices: Option<&BTreeSet<usize>>,
    ) -> Vec<(String, f64)> {
        match limit {
            // No bound: nothing to prune against, so score exhaustively.
            None => self.rank_reference(query, None, allowed_indices),
            // A zero-length request is empty, matching `truncate(0)`.
            Some(0) => Vec::new(),
            Some(k) => self.rank_maxscore(query, k, allowed_indices),
        }
    }

    /// Exhaustive BM25 scan kept as the correctness reference for [`rank`].
    ///
    /// This is the original ranker: it scores every unit that shares a query
    /// term, then sorts on `(score desc, unit_id asc)`. It is the specification
    /// the MaxScore path is differential-tested against, and it backs the
    /// unbounded (`limit = None`) query shape.
    fn rank_reference(
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
        let avgdl = self.avgdl();
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

    /// Bounded top-k BM25 via MaxScore over the compact serving snapshot.
    ///
    /// Produces exactly the first `k` rows of [`rank_reference`] for the same
    /// query, ids and order and scores identical. `k` is assumed positive
    /// (`Some(0)` is handled in [`rank`]).
    fn rank_maxscore(
        &self,
        query: &str,
        k: usize,
        allowed_indices: Option<&BTreeSet<usize>>,
    ) -> Vec<(String, f64)> {
        if self.n == 0 {
            return Vec::new();
        }
        let query_terms = tokenize(query).into_iter().collect::<BTreeSet<_>>();
        if query_terms.is_empty() {
            return Vec::new();
        }
        self.ensure_serving();
        let serving = self.serving.borrow();
        let serving = serving
            .as_ref()
            .expect("serving snapshot is built by ensure_serving");

        // Query terms present in the snapshot, in ascending term order. The
        // BTreeSet iterates sorted, so `terms[..]` is the ascending-term-string
        // order the exhaustive path folds each doc's contributions in; keeping
        // that fold order is what makes the accumulated scores bit-identical.
        let terms: Vec<&ServingTerm> = query_terms
            .iter()
            .filter_map(|term| serving.terms.get(term))
            .collect();
        let m = terms.len();
        if m == 0 {
            return Vec::new();
        }

        // Pruning order: term indices sorted by `(max_score asc, term-order asc)`.
        // MaxScore grows the non-essential prefix (cheapest upper bounds first)
        // as the threshold rises; the fold-index tie-break keeps it deterministic.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| terms[a].max_score.total_cmp(&terms[b].max_score).then(a.cmp(&b)));

        let mut cursor = vec![0usize; m];
        // The heap never holds more than `min(k, live docs)` entries, so cap the
        // preallocation there: a caller may pass a `k` far larger than the
        // corpus, and `k + 1` would otherwise over-allocate (or overflow).
        let mut heap: BinaryHeap<HeapEntry> =
            BinaryHeap::with_capacity(self.n.min(k).saturating_add(1));

        loop {
            let heap_full = heap.len() >= k;
            // theta = current k-th best score (the heap root is the worst kept).
            let theta = if heap_full {
                heap.peek().map_or(f64::NEG_INFINITY, |entry| entry.score)
            } else {
                f64::NEG_INFINITY
            };

            // Non-essential = longest `order` prefix whose max-score sum is
            // STRICTLY below theta. A doc containing only those terms cannot
            // reach theta, so candidates come from the essential suffix only.
            // Strict `<` keeps ties: a doc whose bound equals theta could still
            // win on the unit_id tie-break, so it must stay essential.
            let mut split = 0usize;
            let mut cumulative = 0.0_f64;
            for (index, &term_index) in order.iter().enumerate() {
                cumulative += terms[term_index].max_score;
                if cumulative < theta {
                    split = index + 1;
                } else {
                    break;
                }
            }

            // Next candidate: smallest position among the essential cursors.
            let mut candidate: Option<usize> = None;
            for &term_index in &order[split..] {
                if let Some(posting) = terms[term_index].postings.get(cursor[term_index]) {
                    candidate = Some(candidate.map_or(posting.pos, |current| current.min(posting.pos)));
                }
            }
            let Some(candidate) = candidate else {
                break;
            };

            // Advance every cursor to the first posting at or past the
            // candidate, so each term's membership can be read off `cursor`.
            for term_index in 0..m {
                let postings = &terms[term_index].postings;
                while cursor[term_index] < postings.len()
                    && postings[cursor[term_index]].pos < candidate
                {
                    cursor[term_index] += 1;
                }
            }

            if allowed_indices.map_or(true, |allowed| allowed.contains(&candidate)) {
                // Upper bound the candidate: essential contributions it actually
                // has, plus every non-essential term's max score. Prune only on a
                // strict miss so equal-bound ties are still resolved by unit_id.
                let mut pruned = false;
                if heap_full {
                    let mut upper_bound = 0.0_f64;
                    for &term_index in &order[..split] {
                        upper_bound += terms[term_index].max_score;
                    }
                    for &term_index in &order[split..] {
                        if let Some(posting) = terms[term_index].postings.get(cursor[term_index]) {
                            if posting.pos == candidate {
                                upper_bound += posting.score;
                            }
                        }
                    }
                    if upper_bound < theta {
                        pruned = true;
                    }
                }

                if !pruned {
                    // Exact score: fold this doc's contributions in ascending
                    // term order (the `terms` index order), matching the
                    // reference summation order bit for bit.
                    let mut score = 0.0_f64;
                    for (term_index, term) in terms.iter().enumerate() {
                        if let Some(posting) = term.postings.get(cursor[term_index]) {
                            if posting.pos == candidate {
                                score += posting.score;
                            }
                        }
                    }
                    if let Some(unit_id) = self.unit_id_by_pos.get(&candidate) {
                        let entry = HeapEntry {
                            score,
                            unit_id: unit_id.clone(),
                        };
                        if heap.len() < k {
                            heap.push(entry);
                        } else if heap.peek().is_some_and(|worst| entry < *worst) {
                            heap.pop();
                            heap.push(entry);
                        }
                    }
                }
            }

            // Consume the candidate on every term that holds it.
            for term_index in 0..m {
                let postings = &terms[term_index].postings;
                if postings
                    .get(cursor[term_index])
                    .is_some_and(|posting| posting.pos == candidate)
                {
                    cursor[term_index] += 1;
                }
            }
        }

        let mut ranked: Vec<(String, f64)> = heap
            .into_iter()
            .map(|entry| (entry.unit_id, entry.score))
            .collect();
        ranked.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.0.cmp(&right.0))
        });
        ranked
    }

    /// Average document length, guarded so an all-empty corpus scores as `1.0`.
    fn avgdl(&self) -> f64 {
        if self.n == 0 {
            return 1.0;
        }
        let value = self.total_len as f64 / self.n as f64;
        if value == 0.0 {
            1.0
        } else {
            value
        }
    }

    /// Drops the compact serving snapshot; the next query rebuilds it.
    fn invalidate_serving(&mut self) {
        self.serving.get_mut().take();
    }

    /// Builds the serving snapshot if the cache is dirty.
    fn ensure_serving(&self) {
        if self.serving.borrow().is_some() {
            return;
        }
        let snapshot = self.build_serving();
        *self.serving.borrow_mut() = Some(snapshot);
    }

    /// Compacts the mutable postings into the scoring snapshot.
    ///
    /// Each surviving term gets its postings as a position-sorted array of
    /// precomputed BM25 contributions plus the max contribution (its MaxScore
    /// upper bound). The contribution uses the exact expression and guards of
    /// [`rank_reference`], so folding these values reproduces its scores.
    fn build_serving(&self) -> ServingIndex {
        let avgdl = self.avgdl();
        let mut terms: HashMap<String, ServingTerm> = HashMap::with_capacity(self.postings.len());
        for (term, posting) in &self.postings {
            let idf = self.idf(term);
            if idf <= 0.0 {
                continue;
            }
            let mut entries: Vec<ServingPosting> = Vec::with_capacity(posting.len());
            let mut max_score = 0.0_f64;
            for (pos, term_frequency) in posting {
                let length = *self.doc_len.get(pos).unwrap_or(&0) as f64;
                let term_frequency = *term_frequency as f64;
                let denominator =
                    term_frequency + self.k1 * (1.0 - self.b + self.b * (length / avgdl));
                if denominator <= 0.0 {
                    continue;
                }
                let score = idf * (term_frequency * (self.k1 + 1.0)) / denominator;
                if score > max_score {
                    max_score = score;
                }
                entries.push(ServingPosting { pos: *pos, score });
            }
            if entries.is_empty() {
                continue;
            }
            entries.sort_by_key(|posting| posting.pos);
            terms.insert(term.clone(), ServingTerm { postings: entries, max_score });
        }
        ServingIndex { terms }
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
        self.invalidate_serving();
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
        self.invalidate_serving();
    }

    fn idf(&self, term: &str) -> f64 {
        let df = *self.doc_freq.get(term).unwrap_or(&0);
        if df == 0 {
            return 0.0;
        }
        (1.0 + (self.n as f64 - df as f64 + 0.5) / (df as f64 + 0.5)).ln()
    }
}

/// Compact, query-only snapshot of the mutable postings.
///
/// Rebuilt lazily after any mutation. It holds only what scoring needs: for
/// every term that contributes (positive idf, non-empty postings) a
/// position-sorted array of precomputed BM25 contributions and the per-term
/// maximum contribution used as a MaxScore upper bound.
#[derive(Debug, Clone)]
struct ServingIndex {
    terms: HashMap<String, ServingTerm>,
}

/// One term's serving postings plus its MaxScore upper bound.
#[derive(Debug, Clone)]
struct ServingTerm {
    /// Postings sorted ascending by `pos`, so cursors advance monotonically.
    postings: Vec<ServingPosting>,
    /// Largest contribution any doc gets from this term; the term's score cap.
    max_score: f64,
}

/// One posting: a document position and its precomputed BM25 contribution.
#[derive(Debug, Clone)]
struct ServingPosting {
    pos: usize,
    score: f64,
}

/// A candidate in the bounded top-k heap.
///
/// Ordered so the [`BinaryHeap`] root (the "greatest") is the *worst*-ranked
/// kept candidate: lowest score, and among equal scores the largest unit id.
/// That inverts the final `(score desc, unit_id asc)` order, so the root is
/// exactly the entry to beat, and `entry < root` means the entry outranks it.
/// `total_cmp` gives a panic-free total order (BM25 scores are always finite).
#[derive(Debug, Clone)]
struct HeapEntry {
    score: f64,
    unit_id: String,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.score.total_cmp(&other.score) {
            // Lower score ranks worse, so it is the "greater" heap element.
            Ordering::Less => Ordering::Greater,
            Ordering::Greater => Ordering::Less,
            // Equal score: the larger unit id ranks worse (sorts later).
            Ordering::Equal => self.unit_id.cmp(&other.unit_id),
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for HeapEntry {}

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

/// Differential tests pinning the MaxScore path to the exhaustive reference.
///
/// The exhaustive `rank_reference` is the specification; `rank(_, Some(k), _)`
/// (the MaxScore path) must return the identical rows, ids AND order AND scores
/// bit-for-bit, for every corpus, query, `k`, and filter. These tests generate
/// randomized corpora with mutation scripts (so the serving snapshot is
/// invalidated and rebuilt), plus the named boundary cases, and compare with
/// `f64::to_bits` so any last-ULP drift fails loudly.
#[cfg(test)]
mod maxscore_parity {
    use super::Bm25Index;
    use std::collections::BTreeSet;

    /// Deterministic SplitMix64, so a failing seed reproduces exactly. Keeping
    /// the PRNG in-test avoids a dev-dependency the workspace does not carry.
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// A value in `[0, n)`; `0` for an empty range.
        fn below(&mut self, n: usize) -> usize {
            if n == 0 {
                0
            } else {
                (self.next_u64() % n as u64) as usize
            }
        }

        /// A value in `[lo, hi]` inclusive.
        fn between(&mut self, lo: usize, hi: usize) -> usize {
            lo + self.below(hi - lo + 1)
        }
    }

    /// A skewed vocabulary: a couple of near-ubiquitous tokens (low idf, long
    /// postings that MaxScore should push non-essential) and several rarer ones,
    /// so pruning has something to bite on and ties are plausible.
    const VOCAB: &[&str] = &[
        "common", "common", "the", "alpha", "beta", "gamma", "delta", "epsilon", "e1234", "abc-123",
        "rare", "zzz",
    ];

    fn random_tokens(rng: &mut Rng) -> Vec<String> {
        let length = rng.between(0, 8);
        (0..length)
            .map(|_| VOCAB[rng.below(VOCAB.len())].to_string())
            .collect()
    }

    /// Builds a random corpus, then folds in a random mutation script so the
    /// serving snapshot is repeatedly invalidated and rebuilt against the delta.
    fn random_index(rng: &mut Rng) -> Bm25Index {
        let unit_count = rng.between(0, 24);
        let group_count = rng.between(1, 4);
        let unit_ids: Vec<String> = (0..unit_count).map(|i| format!("u{i:03}")).collect();
        let token_lists: Vec<Vec<String>> = (0..unit_count).map(|_| random_tokens(rng)).collect();
        let group_ids: Vec<String> = (0..unit_count)
            .map(|i| format!("g{}", i % group_count))
            .collect();
        let mut index =
            Bm25Index::from_token_lists(&unit_ids, &token_lists, Some(&group_ids)).unwrap();

        let mut next_id = unit_count;
        for _ in 0..rng.between(0, 16) {
            match rng.below(4) {
                0 => {
                    // Upsert a fresh unit (also exercises re-add of a live id).
                    let id = if rng.below(3) == 0 && next_id > 0 {
                        format!("u{:03}", rng.below(next_id))
                    } else {
                        let id = format!("u{next_id:03}");
                        next_id += 1;
                        id
                    };
                    let group = format!("g{}", rng.below(group_count));
                    index.add_unit(&id, &random_tokens(rng), Some(&group));
                }
                1 => {
                    if next_id > 0 {
                        index.remove_unit(&format!("u{:03}", rng.below(next_id)));
                    }
                }
                2 => {
                    let group = format!("g{}", rng.below(group_count));
                    let units: Vec<(String, Vec<String>)> = (0..rng.between(0, 3))
                        .map(|_| {
                            let id = format!("u{next_id:03}");
                            next_id += 1;
                            (id, random_tokens(rng))
                        })
                        .collect();
                    index.replace_group(&group, &units);
                }
                _ => {
                    index.remove_group(&format!("g{}", rng.below(group_count)));
                }
            }
        }
        index
    }

    fn bits(rows: &[(String, f64)]) -> Vec<(String, u64)> {
        rows.iter()
            .map(|(id, score)| (id.clone(), score.to_bits()))
            .collect()
    }

    /// Asserts the MaxScore top-k equals the exhaustive reference for this query
    /// and filter across a spread of `k`, including the tie-sensitive boundaries.
    fn assert_parity(index: &Bm25Index, query: &str, allowed: Option<&BTreeSet<usize>>, ctx: &str) {
        let full = index.rank_reference(query, None, allowed);
        let matches = full.len();
        let mut ks = vec![1usize, 2, 3];
        if matches > 0 {
            ks.push(matches - 1);
            ks.push(matches);
        }
        ks.push(matches + 1);
        ks.push(matches + 5);
        for k in ks {
            let maxscore = index.rank(query, Some(k), allowed);
            let reference = index.rank_reference(query, Some(k), allowed);
            assert_eq!(
                bits(&maxscore),
                bits(&reference),
                "maxscore != reference ({ctx}, k={k})"
            );
            let head = &full[..full.len().min(k)];
            assert_eq!(
                bits(&maxscore),
                bits(head),
                "maxscore != full-sort head ({ctx}, k={k})"
            );
        }
        // A zero-length request is empty, matching the reference truncation.
        assert!(index.rank(query, Some(0), allowed).is_empty(), "{ctx}, k=0");
    }

    /// Filters over a mix of empty / partial / full / all position sets.
    fn allowed_variants(index: &Bm25Index, rng: &mut Rng) -> Vec<Option<BTreeSet<usize>>> {
        let live: Vec<usize> = index
            .unit_ids()
            .iter()
            .filter_map(|id| index.position_of(id))
            .collect();
        let mut subset = BTreeSet::new();
        for &pos in &live {
            if rng.below(2) == 0 {
                subset.insert(pos);
            }
        }
        vec![
            None,
            Some(BTreeSet::new()),
            Some(subset),
            Some(live.into_iter().collect()),
        ]
    }

    #[test]
    fn maxscore_matches_reference_over_random_corpora() {
        let queries = [
            "",
            "common",
            "rare",
            "zzz",
            "common rare",
            "alpha beta gamma",
            "e1234 abc-123",
            "common the alpha delta epsilon rare",
            "missingtoken",
            "common missingtoken rare",
        ];
        for seed in 0..80u64 {
            let mut rng = Rng::new(seed.wrapping_mul(0x1234_5678).wrapping_add(1));
            let index = random_index(&mut rng);
            for query in queries {
                for allowed in allowed_variants(&index, &mut rng) {
                    assert_parity(
                        &index,
                        query,
                        allowed.as_ref(),
                        &format!("seed={seed}, query={query:?}"),
                    );
                }
            }
        }
    }

    #[test]
    fn ties_break_on_unit_id_in_ascending_order() {
        // Identical documents all score identically, so the whole ranking is the
        // pure tie-break: ascending unit id, truncated to k.
        let unit_ids: Vec<String> = (0..8).map(|i| format!("u{i}")).collect();
        let token_lists: Vec<Vec<String>> = (0..8)
            .map(|_| vec!["alpha".to_string(), "beta".to_string()])
            .collect();
        let index = Bm25Index::from_token_lists(&unit_ids, &token_lists, None).unwrap();
        let ranked = index.rank("alpha", Some(3), None);
        let ids: Vec<&str> = ranked.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["u0", "u1", "u2"]);
        assert_parity(&index, "alpha", None, "all-same-score");
    }

    #[test]
    fn ties_spanning_the_k_boundary_keep_smallest_ids() {
        // One clear winner, then a block of documents tied at the boundary score.
        // The survivors past the winner must be the smallest-id tied documents.
        let mut unit_ids = vec!["top".to_string()];
        let mut token_lists = vec![vec!["alpha".to_string(), "alpha".to_string()]];
        for i in 0..6 {
            unit_ids.push(format!("d{i}"));
            token_lists.push(vec!["alpha".to_string()]);
        }
        let index = Bm25Index::from_token_lists(&unit_ids, &token_lists, None).unwrap();
        let ranked = index.rank("alpha", Some(3), None);
        let ids: Vec<&str> = ranked.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(ids, ["top", "d0", "d1"]);
        assert_parity(&index, "alpha", None, "ties-across-k");
    }

    #[test]
    fn idf_zero_and_absent_terms_do_not_score() {
        // "zzz" is absent; a doc sharing only an absent term must not appear.
        let index = Bm25Index::from_token_lists(
            &["a".to_string(), "b".to_string(), "c".to_string()],
            &[
                vec!["alpha".to_string()],
                vec!["beta".to_string()],
                vec!["alpha".to_string(), "beta".to_string()],
            ],
            None,
        )
        .unwrap();
        assert!(index.rank("zzz", Some(5), None).is_empty());
        assert_parity(&index, "alpha zzz", None, "absent-term");
    }

    #[test]
    fn filtered_to_empty_returns_nothing() {
        let index = Bm25Index::from_token_lists(
            &["a".to_string(), "b".to_string()],
            &[vec!["alpha".to_string()], vec!["alpha".to_string()]],
            None,
        )
        .unwrap();
        // A filter with no live positions removes every candidate.
        let empty = BTreeSet::new();
        assert!(index.rank("alpha", Some(5), Some(&empty)).is_empty());
    }

    #[test]
    fn huge_limit_does_not_overallocate_or_panic() {
        // A limit far larger than the corpus must not preallocate for `k`; it
        // returns every match, identical to the exhaustive path.
        let index = Bm25Index::from_token_lists(
            &["a".to_string(), "b".to_string(), "c".to_string()],
            &[
                vec!["alpha".to_string()],
                vec!["alpha".to_string(), "beta".to_string()],
                vec!["beta".to_string()],
            ],
            None,
        )
        .unwrap();
        let huge = index.rank("alpha beta", Some(1_000_000_000), None);
        let reference = index.rank_reference("alpha beta", Some(1_000_000_000), None);
        assert_eq!(bits(&huge), bits(&reference));
        // usize::MAX exercises the saturating capacity guard.
        assert_eq!(
            bits(&index.rank("alpha beta", Some(usize::MAX), None)),
            bits(&reference)
        );
    }

    #[test]
    fn single_document_and_single_term() {
        let index =
            Bm25Index::from_token_lists(&["only".to_string()], &[vec!["alpha".to_string()]], None)
                .unwrap();
        assert_parity(&index, "alpha", None, "single-doc");
        // Empty corpus scores nothing.
        let empty = Bm25Index::empty();
        assert!(empty.rank("alpha", Some(3), None).is_empty());
    }
}
