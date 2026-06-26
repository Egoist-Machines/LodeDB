//! Metadata filter validation and operand stringification.

use serde_json::{Map, Value};

use crate::error::{CoreError, CoreErrorCode};
use crate::filter::ast::{is_comparison_operator, is_logical_operator, MAX_DEPTH};

/// Validates a metadata filter at the engine boundary using engine stringification.
pub fn validate_metadata_filter(metadata: &Value) -> Result<Value, CoreError> {
    walk(metadata, StringifyMode::Engine, 0)
}

/// Coerces a metadata filter from SDK-style operands.
pub fn coerce_sdk_filter(metadata: &Value) -> Result<Value, CoreError> {
    walk(metadata, StringifyMode::Sdk, 0)
}

#[derive(Debug, Clone, Copy)]
enum StringifyMode {
    Engine,
    Sdk,
}

fn walk(node: &Value, stringify: StringifyMode, depth: usize) -> Result<Value, CoreError> {
    if depth > MAX_DEPTH {
        return invalid("filter is nested too deeply");
    }
    let object = node
        .as_object()
        .ok_or_else(|| invalid_err("filter must be an object"))?;
    if object.is_empty() {
        return invalid("filter must be a non-empty object");
    }
    let mut result = Map::new();
    for (key, spec) in object {
        if key.trim().is_empty() {
            return invalid("filter keys must be non-blank strings");
        }
        if is_logical_operator(key) {
            result.insert(key.clone(), walk_logical(key, spec, stringify, depth)?);
        } else if key.starts_with('$') {
            return invalid(format!(
                "unsupported filter operator {key:?} at field level"
            ));
        } else {
            result.insert(key.clone(), walk_field(key, spec, stringify, depth)?);
        }
    }
    Ok(Value::Object(result))
}

fn walk_logical(
    op: &str,
    spec: &Value,
    stringify: StringifyMode,
    depth: usize,
) -> Result<Value, CoreError> {
    if op == "$not" {
        return walk(spec, stringify, depth + 1);
    }
    let items = spec
        .as_array()
        .ok_or_else(|| invalid_err(format!("{op} requires a non-empty list of filters")))?;
    if items.is_empty() {
        return invalid(format!("{op} requires a non-empty list of filters"));
    }
    let walked = items
        .iter()
        .map(|item| walk(item, stringify, depth + 1))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Value::Array(walked))
}

fn walk_field(
    field: &str,
    spec: &Value,
    stringify: StringifyMode,
    depth: usize,
) -> Result<Value, CoreError> {
    if let Some(object) = spec.as_object() {
        if object.is_empty() {
            return invalid(format!(
                "operator map for field {field:?} must be non-empty"
            ));
        }
        let mut operators = Map::new();
        for (op, operand) in object {
            if !is_comparison_operator(op) {
                return invalid(format!("unsupported operator {op:?} for field {field:?}"));
            }
            operators.insert(op.clone(), validate_operand(field, op, operand, stringify)?);
        }
        return Ok(Value::Object(operators));
    }
    scalar(field, spec, stringify)
        .map(Value::String)
        .map_err(|error| {
            if depth > MAX_DEPTH {
                invalid_err("filter is nested too deeply")
            } else {
                error
            }
        })
}

fn validate_operand(
    field: &str,
    op: &str,
    operand: &Value,
    stringify: StringifyMode,
) -> Result<Value, CoreError> {
    if op == "$exists" {
        return operand
            .as_bool()
            .map(Value::Bool)
            .ok_or_else(|| invalid_err(format!("$exists for field {field:?} requires a boolean")));
    }
    if matches!(op, "$in" | "$nin") {
        let items = operand.as_array().ok_or_else(|| {
            invalid_err(format!(
                "{op} for field {field:?} requires a non-empty list"
            ))
        })?;
        if items.is_empty() {
            return invalid(format!(
                "{op} for field {field:?} requires a non-empty list"
            ));
        }
        let values = items
            .iter()
            .map(|item| scalar(field, item, stringify).map(Value::String))
            .collect::<Result<Vec<_>, _>>()?;
        return Ok(Value::Array(values));
    }
    scalar(field, operand, stringify).map(Value::String)
}

fn scalar(field: &str, value: &Value, stringify: StringifyMode) -> Result<String, CoreError> {
    match value {
        Value::Null => Ok(String::new()),
        Value::String(text) => Ok(text.clone()),
        Value::Bool(flag) => match stringify {
            StringifyMode::Engine => Ok(if *flag { "True" } else { "False" }.to_string()),
            StringifyMode::Sdk => Ok(if *flag { "true" } else { "false" }.to_string()),
        },
        Value::Number(number) => Ok(number.to_string()),
        Value::Array(_) | Value::Object(_) => Err(invalid_err(format!(
            "filter operand for field {field:?} must be a scalar value"
        ))),
    }
}

fn invalid(message: impl Into<String>) -> Result<Value, CoreError> {
    Err(invalid_err(message))
}

fn invalid_err(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::InvalidArgument, message)
}

#[cfg(test)]
mod tests {
    use super::{coerce_sdk_filter, validate_metadata_filter};

    #[test]
    fn engine_and_sdk_bool_stringification_differ() {
        assert_eq!(
            validate_metadata_filter(&serde_json::json!({"fresh": {"$eq": true}})).unwrap(),
            serde_json::json!({"fresh": {"$eq": "True"}})
        );
        assert_eq!(
            coerce_sdk_filter(&serde_json::json!({"fresh": true})).unwrap(),
            serde_json::json!({"fresh": "true"})
        );
    }

    #[test]
    fn rejects_invalid_grammar() {
        assert!(validate_metadata_filter(&serde_json::json!({})).is_err());
        assert!(validate_metadata_filter(&serde_json::json!({"$foo": 1})).is_err());
        assert!(validate_metadata_filter(&serde_json::json!({"a": {"$in": []}})).is_err());
        assert!(validate_metadata_filter(&serde_json::json!({"a": {"$exists": "yes"}})).is_err());
    }
}
