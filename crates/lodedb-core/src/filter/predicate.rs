//! Metadata predicate evaluation for validated filters.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::error::{CoreError, CoreErrorCode};
use crate::filter::ast::is_ordered_operator;

/// Returns whether a document's stored metadata satisfies a validated filter.
pub fn matches_metadata_filter(
    document_metadata: &BTreeMap<String, String>,
    metadata_filter: &Value,
) -> Result<bool, CoreError> {
    let node = metadata_filter.as_object().ok_or_else(|| {
        CoreError::new(
            CoreErrorCode::InvalidArgument,
            "validated filter must be an object",
        )
    })?;
    for (key, spec) in node {
        let matched = match key.as_str() {
            "$and" => spec
                .as_array()
                .ok_or_else(|| invalid("validated $and must be a list"))?
                .iter()
                .map(|sub| matches_metadata_filter(document_metadata, sub))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .all(|value| value),
            "$or" => spec
                .as_array()
                .ok_or_else(|| invalid("validated $or must be a list"))?
                .iter()
                .map(|sub| matches_metadata_filter(document_metadata, sub))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .any(|value| value),
            "$not" => !matches_metadata_filter(document_metadata, spec)?,
            field => matches_field(document_metadata, field, spec)?,
        };
        if !matched {
            return Ok(false);
        }
    }
    Ok(true)
}

fn matches_field(
    metadata: &BTreeMap<String, String>,
    field: &str,
    spec: &Value,
) -> Result<bool, CoreError> {
    if let Some(expected) = spec.as_str() {
        return Ok(metadata.get(field).is_some_and(|stored| stored == expected));
    }
    let operators = spec
        .as_object()
        .ok_or_else(|| invalid("validated field spec must be a string or operator map"))?;
    for (op, operand) in operators {
        if !matches_operator(metadata, field, op, operand)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn matches_operator(
    metadata: &BTreeMap<String, String>,
    field: &str,
    op: &str,
    operand: &Value,
) -> Result<bool, CoreError> {
    let stored = metadata.get(field);
    match op {
        "$eq" => Ok(stored.is_some_and(|value| operand.as_str() == Some(value.as_str()))),
        "$ne" => Ok(!stored.is_some_and(|value| operand.as_str() == Some(value.as_str()))),
        "$in" => {
            let values = operand
                .as_array()
                .ok_or_else(|| invalid("validated $in operand must be a list"))?;
            Ok(stored.is_some_and(|value| contains_string(values, value)))
        }
        "$nin" => {
            let values = operand
                .as_array()
                .ok_or_else(|| invalid("validated $nin operand must be a list"))?;
            Ok(!stored.is_some_and(|value| contains_string(values, value)))
        }
        "$exists" => {
            let exists = operand
                .as_bool()
                .ok_or_else(|| invalid("validated $exists operand must be a boolean"))?;
            Ok(metadata.contains_key(field) == exists)
        }
        ordered if is_ordered_operator(ordered) => {
            let operand = operand
                .as_str()
                .ok_or_else(|| invalid("validated ordered operand must be a string"))?;
            Ok(stored.is_some_and(|value| compare_ordered(value, ordered, operand)))
        }
        _ => Err(invalid("unsupported validated operator")),
    }
}

fn contains_string(values: &[Value], needle: &str) -> bool {
    values.iter().any(|value| value.as_str() == Some(needle))
}

fn compare_ordered(stored: &str, op: &str, operand: &str) -> bool {
    let operand_number = as_number(operand);
    if let Some(operand_number) = operand_number {
        if let Some(stored_number) = as_number(stored) {
            return compare_f64(stored_number, op, operand_number);
        }
    }
    compare_str(stored, op, operand)
}

fn as_number(text: &str) -> Option<f64> {
    let value = text.parse::<f64>().ok()?;
    if value.is_nan() {
        None
    } else {
        Some(value)
    }
}

fn compare_f64(left: f64, op: &str, right: f64) -> bool {
    match op {
        "$gt" => left > right,
        "$gte" => left >= right,
        "$lt" => left < right,
        "$lte" => left <= right,
        _ => false,
    }
}

fn compare_str(left: &str, op: &str, right: &str) -> bool {
    match op {
        "$gt" => left > right,
        "$gte" => left >= right,
        "$lt" => left < right,
        "$lte" => left <= right,
        _ => false,
    }
}

fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::InvalidArgument, message)
}

#[cfg(test)]
mod tests {
    use super::matches_metadata_filter;
    use crate::filter::validate::validate_metadata_filter;

    fn matches(doc: serde_json::Value, filter: serde_json::Value) -> bool {
        let doc = serde_json::from_value(doc).expect("document metadata");
        let filter = validate_metadata_filter(&filter).expect("valid filter");
        matches_metadata_filter(&doc, &filter).expect("match filter")
    }

    #[test]
    fn evaluates_missing_field_semantics() {
        assert!(matches(
            serde_json::json!({"b": "1"}),
            serde_json::json!({"a": {"$ne": "1"}})
        ));
        assert!(matches(
            serde_json::json!({"b": "1"}),
            serde_json::json!({"a": {"$nin": ["1"]}})
        ));
        assert!(matches(
            serde_json::json!({"b": "1"}),
            serde_json::json!({"a": {"$exists": false}})
        ));
        assert!(!matches(
            serde_json::json!({"b": "1"}),
            serde_json::json!({"a": "1"})
        ));
    }

    #[test]
    fn evaluates_numeric_and_lexicographic_ordering() {
        assert!(matches(
            serde_json::json!({"year": "2020"}),
            serde_json::json!({"year": {"$gte": 2020}})
        ));
        assert!(matches(
            serde_json::json!({"x": "abc"}),
            serde_json::json!({"x": {"$gt": 1}})
        ));
        assert!(matches(
            serde_json::json!({"v": "nan"}),
            serde_json::json!({"v": {"$gt": "0"}})
        ));
        assert!(matches(
            serde_json::json!({"v": "inf"}),
            serde_json::json!({"v": {"$gte": "1e9"}})
        ));
    }
}
