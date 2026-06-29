//! Per-field metadata value indexes used by the filter planner.

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::filter::doc_set::DocSet;
use crate::filter::predicate::{as_number, compare_ordered};

/// Per-key value index for one metadata field within a generation.
///
/// Exact-match lookups use `value_docs` directly and stay incremental on every
/// write. Ordered range operators (`$gt`/`$gte`/`$lt`/`$lte`) need the distinct
/// values partitioned into a sorted numeric run plus a lexicographic remainder;
/// that partition is derived lazily from `value_docs` and cached in `ordered`,
/// so writes never rebuild it (they only invalidate the cache in O(1)).
#[derive(Debug, Default)]
pub struct FieldIndex {
    /// Documents carrying this metadata key.
    pub docs: DocSet,
    /// Exact stored value -> documents carrying that value.
    pub value_docs: BTreeMap<String, DocSet>,
    /// Lazily materialized ordered partition; `None` means "rebuild on next read".
    ordered: RefCell<Option<OrderedPartitions>>,
}

impl FieldIndex {
    /// Adds one stored field value for a document.
    pub fn insert(&mut self, document_id: String, value: String) {
        self.docs.insert(document_id.clone());
        self.value_docs
            .entry(value)
            .or_default()
            .insert(document_id);
        self.invalidate_ordered();
    }

    /// Removes one stored field value for a document.
    pub fn remove(&mut self, document_id: &str, value: &str) {
        self.docs.remove(document_id);
        if let Some(value_docs) = self.value_docs.get_mut(value) {
            value_docs.remove(document_id);
            if value_docs.is_empty() {
                self.value_docs.remove(value);
            }
        }
        self.invalidate_ordered();
    }

    /// Returns whether this field has no indexed documents.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Resolves an ordered operator (`$gt`/`$gte`/`$lt`/`$lte`) to its doc set.
    ///
    /// When the operand parses as a number we compare against the cached numeric
    /// partition (binary search) plus a lexicographic pass over the non-numeric
    /// remainder, matching the per-document semantics in
    /// [`compare_ordered`](crate::filter::predicate). Otherwise we fall back to a
    /// lexicographic scan of every stored value.
    pub fn resolve_ordered(&self, op: &str, operand: &str) -> DocSet {
        let mut docs = DocSet::new();
        if let Some(operand_number) = as_number(operand) {
            self.ensure_ordered();
            let ordered = self.ordered.borrow();
            let partitions = ordered
                .as_ref()
                .expect("ordered partitions populated by ensure_ordered");
            for value in partitions.numeric_values_satisfying(op, operand_number) {
                if let Some(value_docs) = self.value_docs.get(value) {
                    docs.extend(value_docs.iter().cloned());
                }
            }
            for value in &partitions.nonnumeric_values {
                if compare_ordered(value, op, operand) {
                    if let Some(value_docs) = self.value_docs.get(value) {
                        docs.extend(value_docs.iter().cloned());
                    }
                }
            }
            return docs;
        }

        for (value, value_docs) in &self.value_docs {
            if compare_ordered(value, op, operand) {
                docs.extend(value_docs.iter().cloned());
            }
        }
        docs
    }

    /// Drops the cached ordered partition so the next ordered read rebuilds it.
    fn invalidate_ordered(&mut self) {
        *self.ordered.get_mut() = None;
    }

    /// Builds the ordered partition from `value_docs` if it is not already cached.
    fn ensure_ordered(&self) {
        if self.ordered.borrow().is_some() {
            return;
        }
        let mut numeric = Vec::new();
        let mut nonnumeric = Vec::new();
        for value in self.value_docs.keys() {
            if let Some(number) = as_number(value) {
                numeric.push(NumericValue {
                    number,
                    value: value.clone(),
                });
            } else {
                nonnumeric.push(value.clone());
            }
        }
        numeric.sort_by(|left, right| {
            left.number
                .partial_cmp(&right.number)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.value.cmp(&right.value))
        });
        *self.ordered.borrow_mut() = Some(OrderedPartitions {
            numeric_values: numeric,
            nonnumeric_values: nonnumeric,
        });
    }
}

impl Clone for FieldIndex {
    fn clone(&self) -> Self {
        // The ordered partition is a derived cache; clones start cold and rebuild
        // lazily, which also avoids touching the source's `RefCell` borrow state.
        Self {
            docs: self.docs.clone(),
            value_docs: self.value_docs.clone(),
            ordered: RefCell::new(None),
        }
    }
}

impl PartialEq for FieldIndex {
    fn eq(&self, other: &Self) -> bool {
        // Equality is over the authoritative data only; the ordered cache is derived.
        self.docs == other.docs && self.value_docs == other.value_docs
    }
}

/// Distinct field values split into a sorted numeric run and the remainder.
#[derive(Debug, Clone, Default, PartialEq)]
struct OrderedPartitions {
    numeric_values: Vec<NumericValue>,
    nonnumeric_values: Vec<String>,
}

impl OrderedPartitions {
    /// Returns stored value strings whose numeric value satisfies `op operand`.
    fn numeric_values_satisfying(&self, op: &str, operand: f64) -> Vec<&str> {
        let range: Box<dyn Iterator<Item = &NumericValue> + '_> = match op {
            "$gt" => Box::new(self.numeric_values[self.upper_partition(operand)..].iter()),
            "$gte" => Box::new(self.numeric_values[self.lower_partition(operand)..].iter()),
            "$lt" => Box::new(self.numeric_values[..self.lower_partition(operand)].iter()),
            "$lte" => Box::new(self.numeric_values[..self.upper_partition(operand)].iter()),
            _ => Box::new([].iter()),
        };
        range.map(|entry| entry.value.as_str()).collect()
    }

    fn lower_partition(&self, operand: f64) -> usize {
        self.numeric_values
            .partition_point(|entry| entry.number < operand)
    }

    fn upper_partition(&self, operand: f64) -> usize {
        self.numeric_values
            .partition_point(|entry| entry.number <= operand)
    }
}

#[derive(Debug, Clone, PartialEq)]
struct NumericValue {
    number: f64,
    value: String,
}

/// Builds per-field value indexes and the full document-id set for a generation.
pub fn build_field_indexes(
    document_metadata: &BTreeMap<String, BTreeMap<String, String>>,
) -> (BTreeMap<String, FieldIndex>, DocSet) {
    let mut fields: BTreeMap<String, FieldIndex> = BTreeMap::new();
    let mut all_docs = DocSet::new();

    for (document_id, metadata) in document_metadata {
        all_docs.insert(document_id.clone());
        for (key, value) in metadata {
            fields
                .entry(key.clone())
                .or_default()
                .insert(document_id.clone(), value.clone());
        }
    }
    (fields, all_docs)
}

#[cfg(test)]
mod tests {
    use super::FieldIndex;

    fn sorted_docs(index: &FieldIndex, op: &str, operand: &str) -> Vec<String> {
        let mut docs: Vec<String> = index.resolve_ordered(op, operand).iter().cloned().collect();
        docs.sort();
        docs
    }

    #[test]
    fn resolves_numeric_ordered_values_via_partition() {
        let mut index = FieldIndex::default();
        for (id, value) in [("a", "5"), ("b", "10"), ("c", "15"), ("d", "20")] {
            index.insert(id.to_string(), value.to_string());
        }

        assert_eq!(sorted_docs(&index, "$gte", "12"), vec!["c", "d"]);
        assert_eq!(sorted_docs(&index, "$gt", "20"), Vec::<String>::new());
        assert_eq!(sorted_docs(&index, "$lt", "10"), vec!["a"]);
        assert_eq!(sorted_docs(&index, "$lte", "10"), vec!["a", "b"]);
    }

    #[test]
    fn nonnumeric_values_match_lexicographic_predicate_semantics() {
        // Mirrors `predicate::compare_ordered`: a numeric operand still compares
        // lexicographically against non-numeric stored values, so "free"/"nan"
        // (both > "10.0" lexically) satisfy `$gte "10.0"` alongside the numerics.
        let mut index = FieldIndex::default();
        for value in ["9.5", "free", "nan", "inf", "1e9"] {
            index.insert(format!("doc-{value}"), value.to_string());
        }

        assert_eq!(
            sorted_docs(&index, "$gte", "10.0"),
            vec!["doc-1e9", "doc-free", "doc-inf", "doc-nan"]
        );
    }

    #[test]
    fn ordered_cache_survives_and_invalidates_on_write() {
        let mut index = FieldIndex::default();
        index.insert("doc-a".to_string(), "10".to_string());
        index.insert("doc-b".to_string(), "20".to_string());

        // First read builds the cache; a repeat read serves from it.
        assert_eq!(index.resolve_ordered("$gt", "15").len(), 1);
        assert_eq!(index.resolve_ordered("$gt", "15").len(), 1);

        // A subsequent write must invalidate the cache so the new value is seen.
        index.insert("doc-c".to_string(), "30".to_string());
        let mut gt_fifteen: Vec<String> =
            index.resolve_ordered("$gt", "15").iter().cloned().collect();
        gt_fifteen.sort();
        assert_eq!(gt_fifteen, vec!["doc-b".to_string(), "doc-c".to_string()]);
    }
}
