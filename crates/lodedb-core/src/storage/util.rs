use crate::error::{CoreError, CoreErrorCode};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;

pub(crate) type CoreResult<T> = Result<T, CoreError>;

pub(crate) fn corrupt(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::CorruptStore, message)
}

pub(crate) fn invalid(message: impl Into<String>) -> CoreError {
    CoreError::new(CoreErrorCode::InvalidArgument, message)
}

pub(crate) fn read_json(path: &Path, context: &str) -> CoreResult<Value> {
    let data = fs::read_to_string(path)
        .map_err(|error| corrupt(format!("{context} could not be read: {error}")))?;
    serde_json::from_str(&data).map_err(|error| corrupt(format!("{context} is corrupt: {error}")))
}

pub(crate) fn read_json_object(path: &Path, context: &str) -> CoreResult<Map<String, Value>> {
    match read_json(path, context)? {
        Value::Object(object) => Ok(object),
        _ => Err(corrupt(format!("{context} is not a JSON object"))),
    }
}

pub(crate) fn sha256_bytes_hex(bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(bytes);
    format!("{:x}", digest.finalize())
}

pub(crate) fn sha256_file_hex(path: &Path) -> CoreResult<String> {
    let data = fs::read(path)
        .map_err(|error| corrupt(format!("{} could not be read: {error}", path.display())))?;
    Ok(sha256_bytes_hex(&data))
}

pub(crate) fn verify_file_sha256(path: &Path, expected: &str, context: &str) -> CoreResult<()> {
    if expected.is_empty() {
        return Ok(());
    }
    let actual = sha256_file_hex(path)?;
    if actual != expected {
        return Err(corrupt(format!("{context} failed manifest checksum")));
    }
    Ok(())
}

pub(crate) fn py_canonical_json(value: &Value) -> CoreResult<String> {
    match value {
        Value::Null => Ok("null".to_string()),
        Value::Bool(true) => Ok("true".to_string()),
        Value::Bool(false) => Ok("false".to_string()),
        Value::Number(number) => Ok(number.to_string()),
        Value::String(text) => serde_json::to_string(text)
            .map_err(|error| corrupt(format!("could not encode JSON string: {error}"))),
        Value::Array(items) => {
            let rendered = items
                .iter()
                .map(py_canonical_json)
                .collect::<CoreResult<Vec<_>>>()?;
            Ok(format!("[{}]", rendered.join(", ")))
        }
        Value::Object(object) => {
            let mut parts = Vec::with_capacity(object.len());
            for (key, item) in object.iter() {
                let key_json = serde_json::to_string(key)
                    .map_err(|error| corrupt(format!("could not encode JSON key: {error}")))?;
                parts.push(format!("{key_json}: {}", py_canonical_json(item)?));
            }
            Ok(format!("{{{}}}", parts.join(", ")))
        }
    }
}

pub(crate) fn body_sha256(body: &Value) -> CoreResult<String> {
    Ok(sha256_bytes_hex(py_canonical_json(body)?.as_bytes()))
}

pub(crate) fn get_i64(object: &Map<String, Value>, key: &str, default: i64) -> i64 {
    object.get(key).and_then(Value::as_i64).unwrap_or(default)
}

pub(crate) fn get_str<'a>(object: &'a Map<String, Value>, key: &str) -> &'a str {
    object.get(key).and_then(Value::as_str).unwrap_or("")
}

pub(crate) fn value_object<'a>(
    value: &'a Value,
    context: &str,
) -> CoreResult<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| corrupt(format!("{context} must be a JSON object")))
}
