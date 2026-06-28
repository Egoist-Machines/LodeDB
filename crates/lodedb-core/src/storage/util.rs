use crate::error::{CoreError, CoreErrorCode};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::Write;
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

/// Encodes a string as a JSON string literal byte-for-byte identical to
/// CPython's ``json.dumps`` default (``ensure_ascii=True``): ``"``, ``\`` and the
/// five short control escapes map to their named forms, ``0x20..=0x7E`` pass
/// through raw, and every other scalar (other control chars, ``0x7F`` and all
/// non-ASCII) becomes a lowercase ``\uXXXX`` escape, with a UTF-16 surrogate pair
/// above the BMP. The native engine and the Python engine both checksum a
/// canonical body and verify each other's writes across the FFI boundary, so the
/// two encoders must agree to the byte. ``serde_json::to_string`` emits raw UTF-8
/// for non-ASCII instead, which silently diverges on real document text.
fn encode_json_string_ascii(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{08}' => out.push_str("\\b"),
            '\u{09}' => out.push_str("\\t"),
            '\u{0a}' => out.push_str("\\n"),
            '\u{0c}' => out.push_str("\\f"),
            '\u{0d}' => out.push_str("\\r"),
            ' '..='~' => out.push(ch),
            _ => {
                let cp = ch as u32;
                if cp <= 0xffff {
                    out.push_str(&format!("\\u{cp:04x}"));
                } else {
                    let value = cp - 0x1_0000;
                    let high = 0xd800 + (value >> 10);
                    let low = 0xdc00 + (value & 0x3ff);
                    out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

pub(crate) fn py_canonical_json(value: &Value) -> CoreResult<String> {
    match value {
        Value::Null => Ok("null".to_string()),
        Value::Bool(true) => Ok("true".to_string()),
        Value::Bool(false) => Ok("false".to_string()),
        Value::Number(number) => Ok(number.to_string()),
        Value::String(text) => Ok(encode_json_string_ascii(text)),
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
                let key_json = encode_json_string_ascii(key);
                parts.push(format!("{key_json}: {}", py_canonical_json(item)?));
            }
            Ok(format!("{{{}}}", parts.join(", ")))
        }
    }
}

pub(crate) fn body_sha256(body: &Value) -> CoreResult<String> {
    Ok(sha256_bytes_hex(py_canonical_json(body)?.as_bytes()))
}

pub(crate) fn write_py_json(path: &Path, value: &Value, fsync: bool) -> CoreResult<usize> {
    let text = py_canonical_json(value)?;
    write_text_atomic(path, &text, fsync)
}

pub(crate) fn write_text_atomic(path: &Path, text: &str, fsync: bool) -> CoreResult<usize> {
    write_bytes_atomic(path, text.as_bytes(), fsync)
}

pub(crate) fn write_bytes_atomic(path: &Path, bytes: &[u8], fsync: bool) -> CoreResult<usize> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            corrupt(format!(
                "could not create directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let temporary = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    {
        let mut handle = File::create(&temporary)
            .map_err(|error| corrupt(format!("could not create temp file: {error}")))?;
        handle
            .write_all(bytes)
            .map_err(|error| corrupt(format!("could not write temp file: {error}")))?;
        if fsync {
            handle
                .sync_all()
                .map_err(|error| corrupt(format!("could not fsync temp file: {error}")))?;
        }
    }
    fs::rename(&temporary, path)
        .map_err(|error| corrupt(format!("could not replace {}: {error}", path.display())))?;
    if fsync {
        fsync_dir(path.parent().unwrap_or_else(|| Path::new(".")))?;
    }
    Ok(bytes.len())
}

pub(crate) fn write_pretty_json_atomic(
    path: &Path,
    value: &Value,
    fsync: bool,
) -> CoreResult<usize> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| corrupt(format!("could not encode manifest JSON: {error}")))?;
    write_text_atomic(path, &text, fsync)
}

pub(crate) fn fsync_dir(path: &Path) -> CoreResult<()> {
    let Ok(handle) = File::open(path) else {
        return Ok(());
    };
    handle.sync_all().map_err(|error| {
        corrupt(format!(
            "could not fsync directory {}: {error}",
            path.display()
        ))
    })
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

#[cfg(test)]
mod canonical_json_tests {
    use super::*;
    use serde_json::json;

    // py_canonical_json must equal CPython's json.dumps(value, sort_keys=True)
    // byte-for-byte: the native and Python engines checksum a canonical body and
    // verify each other's persisted writes across the FFI boundary.
    #[test]
    fn matches_python_ensure_ascii_escaping() {
        let cases = [
            ("plain ascii", "\"plain ascii\""),
            ("é", "\"\\u00e9\""),
            ("§", "\"\\u00a7\""),
            ("—", "\"\\u2014\""),
            ("\u{201c}", "\"\\u201c\""),
            ("\u{1F600}", "\"\\ud83d\\ude00\""),
            ("\u{7f}", "\"\\u007f\""),
            ("\u{2028}", "\"\\u2028\""),
            ("\t", "\"\\t\""),
            ("\u{0}", "\"\\u0000\""),
            ("\u{0b}", "\"\\u000b\""),
            ("\\", "\"\\\\\""),
            ("/", "\"/\""),
        ];
        for (input, expected) in cases {
            assert_eq!(
                py_canonical_json(&Value::String(input.to_string())).unwrap(),
                expected,
                "escaping mismatch for {input:?}"
            );
        }
    }

    // The body checksum survives serialize -> parse -> re-serialize for non-ASCII
    // document text (the real-corpus reopen path), and the on-disk form is ASCII.
    #[test]
    fn canonical_form_is_idempotent_for_non_ascii() {
        let body = json!({
            "schema_version": 2,
            "documents": {"doc-1": "café — naïve résumé § 12 \u{1F600}"},
        });
        let written = py_canonical_json(&body).unwrap();
        let parsed: Value = serde_json::from_str(&written).unwrap();
        assert_eq!(written, py_canonical_json(&parsed).unwrap());
        assert!(written.is_ascii(), "canonical body must be ASCII: {written}");
    }
}
