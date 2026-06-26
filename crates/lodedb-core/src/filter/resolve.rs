//! Set-based metadata filter resolution over per-field indexes.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::{CoreError, CoreErrorCode};
use crate::filter::ast::is_ordered_operator;
use crate::filter::doc_set::DocSet;
use crate::filter::field_index::FieldIndex;
use crate::filter::predicate::{as_number, compare_ordered};

/// Resolves a validated metadata filter to the matching document-id set.
///
/// Entries of a node are AND-ed. `$and`/`$or`/`$not` recurse. The returned set is
/// fresh and may be mutated by callers.
pub fn resolve_filter(
    metadata_filter: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    let node = metadata_filter.as_object().ok_or_else(|| {
        CoreError::new(
            CoreErrorCode::InvalidArgument,
            "validated filter must be an object",
        )
    })?;
    let mut result: Option<DocSet> = None;
    for (key, spec) in node {
        let docs = match key.as_str() {
            "$and" => resolve_and(spec, fields, all_docs)?,
            "$or" => resolve_or(spec, fields, all_docs)?,
            "$not" => difference(all_docs, &resolve_filter(spec, fields, all_docs)?),
            field => resolve_field(field, spec, fields, all_docs)?,
        };
        result = Some(match result {
            Some(previous) => intersection(&previous, &docs),
            None => docs,
        });
        if result.as_ref().is_some_and(DocSet::is_empty) {
            return Ok(DocSet::new());
        }
    }
    Ok(result.unwrap_or_else(|| all_docs.clone()))
}

fn resolve_and(
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    let subs = spec
        .as_array()
        .ok_or_else(|| invalid("validated $and must be a list"))?;
    let mut result: Option<DocSet> = None;
    for sub in subs {
        let docs = resolve_filter(sub, fields, all_docs)?;
        result = Some(match result {
            Some(previous) => intersection(&previous, &docs),
            None => docs,
        });
        if result.as_ref().is_some_and(DocSet::is_empty) {
            return Ok(DocSet::new());
        }
    }
    Ok(result.unwrap_or_else(|| all_docs.clone()))
}

fn resolve_or(
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    let subs = spec
        .as_array()
        .ok_or_else(|| invalid("validated $or must be a list"))?;
    let mut result = DocSet::new();
    for sub in subs {
        result.extend(resolve_filter(sub, fields, all_docs)?);
    }
    Ok(result)
}

fn resolve_field(
    field: &str,
    spec: &Value,
    fields: &BTreeMap<String, FieldIndex>,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    let empty = FieldIndex::default();
    let index = fields.get(field).unwrap_or(&empty);
    if let Some(expected) = spec.as_str() {
        return Ok(index.value_docs.get(expected).cloned().unwrap_or_default());
    }
    let operators = spec
        .as_object()
        .ok_or_else(|| invalid("validated field spec must be a string or operator map"))?;
    let mut result: Option<DocSet> = None;
    for (op, operand) in operators {
        let docs = resolve_operator(op, operand, index, all_docs)?;
        result = Some(match result {
            Some(previous) => intersection(&previous, &docs),
            None => docs,
        });
        if result.as_ref().is_some_and(DocSet::is_empty) {
            return Ok(DocSet::new());
        }
    }
    Ok(result.unwrap_or_else(|| all_docs.clone()))
}

fn resolve_operator(
    op: &str,
    operand: &Value,
    index: &FieldIndex,
    all_docs: &DocSet,
) -> Result<DocSet, CoreError> {
    match op {
        "$eq" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $eq operand must be a string"))?;
            Ok(index.value_docs.get(value).cloned().unwrap_or_default())
        }
        "$ne" => {
            let value = operand
                .as_str()
                .ok_or_else(|| invalid("validated $ne operand must be a string"))?;
            Ok(difference(
                all_docs,
                index.value_docs.get(value).unwrap_or(&DocSet::new()),
            ))
        }
        "$in" => {
            let values = operand
                .as_array()
                .ok_or_else(|| invalid("validated $in operand must be a list"))?;
            let mut docs = DocSet::new();
            for value in values {
                let value = value
                    .as_str()
                    .ok_or_else(|| invalid("validated $in values must be strings"))?;
                if let Some(value_docs) = index.value_docs.get(value) {
                    docs.extend(value_docs.iter().cloned());
                }
            }
            Ok(docs)
        }
        "$nin" => {
            let values = operand
                .as_array()
                .ok_or_else(|| invalid("validated $nin operand must be a list"))?;
            let mut excluded = DocSet::new();
            for value in values {
                let value = value
                    .as_str()
                    .ok_or_else(|| invalid("validated $nin values must be strings"))?;
                if let Some(value_docs) = index.value_docs.get(value) {
                    excluded.extend(value_docs.iter().cloned());
                }
            }
            Ok(difference(all_docs, &excluded))
        }
        "$exists" => {
            let exists = operand
                .as_bool()
                .ok_or_else(|| invalid("validated $exists operand must be a boolean"))?;
            if exists {
                Ok(index.docs.clone())
            } else {
                Ok(difference(all_docs, &index.docs))
            }
        }
        ordered if is_ordered_operator(ordered) => {
            let operand = operand
                .as_str()
                .ok_or_else(|| invalid("validated ordered operand must be a string"))?;
            Ok(resolve_ordered(ordered, operand, index))
        }
        _ => Err(invalid("unsupported validated operator")),
    }
}

fn resolve_ordered(op: &str, operand: &str, index: &FieldIndex) -> DocSet {
    let mut docs = DocSet::new();
    if let Some(operand_number) = as_number(operand) {
        for value in index.numeric_values_satisfying(op, operand_number) {
            if let Some(value_docs) = index.value_docs.get(value) {
                docs.extend(value_docs.iter().cloned());
            }
        }
        for value in index.nonnumeric_values() {
            if compare_ordered(value, op, operand) {
                if let Some(value_docs) = index.value_docs.get(value) {
                    docs.extend(value_docs.iter().cloned());
                }
            }
        }
        return docs;
    }

    for (value, value_docs) in &index.value_docs {
        if compare_ordered(value, op, operand) {
            docs.extend(value_docs.iter().cloned());
        }
    }
    docs
}

fn intersection(left: &DocSet, right: &DocSet) -> DocSet {
    left.intersection(right).cloned().collect()
}

fn difference(left: &DocSet, right: &DocSet) -> DocSet {
    left.difference(right).cloned().collect()
}

fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::InvalidArgument, message)
}
