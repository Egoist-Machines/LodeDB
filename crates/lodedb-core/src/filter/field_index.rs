//! Per-field metadata value indexes used by the filter planner.

use std::cmp::Ordering;
use std::collections::BTreeMap;

use crate::filter::doc_set::DocSet;
use crate::filter::predicate::as_number;

/// Per-key value index for one metadata field within a generation.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FieldIndex {
    /// Documents carrying this metadata key.
    pub docs: DocSet,
    /// Exact stored value -> documents carrying that value.
    pub value_docs: BTreeMap<String, DocSet>,
    numeric_values: Vec<NumericValue>,
    nonnumeric_values: Vec<String>,
}

impl FieldIndex {
    /// Adds one stored field value for a document.
    pub fn insert(&mut self, document_id: String, value: String) {
        self.docs.insert(document_id.clone());
        self.value_docs
            .entry(value)
            .or_default()
            .insert(document_id);
    }

    /// Partitions distinct values into sorted numeric values and non-numeric values.
    pub fn finalize(&mut self) {
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
        self.numeric_values = numeric;
        self.nonnumeric_values = nonnumeric;
    }

    /// Returns stored value strings whose numeric value satisfies `op operand`.
    pub fn numeric_values_satisfying(&self, op: &str, operand: f64) -> Vec<&str> {
        let range: Box<dyn Iterator<Item = &NumericValue> + '_> = match op {
            "$gt" => Box::new(self.numeric_values[self.upper_partition(operand)..].iter()),
            "$gte" => Box::new(self.numeric_values[self.lower_partition(operand)..].iter()),
            "$lt" => Box::new(self.numeric_values[..self.lower_partition(operand)].iter()),
            "$lte" => Box::new(self.numeric_values[..self.upper_partition(operand)].iter()),
            _ => Box::new([].iter()),
        };
        range.map(|entry| entry.value.as_str()).collect()
    }

    /// Returns non-numeric stored values for lexicographic ordered fallback.
    pub fn nonnumeric_values(&self) -> &[String] {
        &self.nonnumeric_values
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
    for field in fields.values_mut() {
        field.finalize();
    }
    (fields, all_docs)
}

#[cfg(test)]
mod tests {
    use super::FieldIndex;

    #[test]
    fn partitions_numeric_values_with_nan_as_nonnumeric() {
        let mut index = FieldIndex::default();
        for value in ["9.5", "free", "nan", "inf", "1e9"] {
            index.insert(format!("doc-{value}"), value.to_string());
        }
        index.finalize();

        assert_eq!(
            index.numeric_values_satisfying("$gte", 10.0),
            vec!["1e9", "inf"]
        );
        assert_eq!(
            index.nonnumeric_values(),
            &["free".to_string(), "nan".to_string()]
        );
    }
}
