use crate::error::{CoreError, CoreErrorCode};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
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
    let mut file = File::open(path)
        .map_err(|error| corrupt(format!("{} could not be read: {error}", path.display())))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| corrupt(format!("{} could not be read: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// FNV-1a-64, shared by compact binary sidecars for lightweight section and
/// record checksums. This matches TurboVec's calibration-fingerprint hash.
pub(crate) fn fnv1a64(bytes: &[u8]) -> u64 {
    fnv1a64_update(0xcbf2_9ce4_8422_2325, bytes)
}

pub(crate) fn fnv1a64_update(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// IEEE 754 half-precision bits to f32, matching numpy's float16 conversion.
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = if (bits >> 15) & 1 == 1 { -1.0_f32 } else { 1.0 };
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = (bits & 0x3ff) as f32;
    let magnitude = match exponent {
        0 => mantissa * 2.0_f32.powi(-24),
        0x1f => {
            if mantissa == 0.0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + mantissa / 1024.0) * 2.0_f32.powi(exponent as i32 - 15),
    };
    sign * magnitude
}

/// Converts f32 to IEEE 754 binary16 with round-to-nearest, ties-to-even.
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        let payload = ((mantissa >> 13) as u16) & 0x03ff;
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { payload.max(1) };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half = (significand >> shift) as u16;
        let remainder = significand & ((1_u32 << shift) - 1);
        let halfway = 1_u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && half & 1 == 1) {
            half += 1;
        }
        return sign | half;
    }

    let mut half = ((half_exponent as u16) << 10) | ((mantissa >> 13) as u16);
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && half & 1 == 1) {
        half += 1;
    }
    sign | half
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

/// One non-blocking attempt to open `path` (creating it) and take an OS lock hold.
///
/// `Ok(Some(file))` holds the lock; `Ok(None)` means another holder currently
/// blocks it and the caller should retry; `Err` is a real open/lock failure the
/// caller wraps in its own message. Unix opens read+write+create and takes a
/// non-blocking advisory `flock`, exclusive by default or shared when `exclusive`
/// is false (a shared `flock` interoperates with the Python writer's `fcntl.flock`).
/// Windows has no advisory `flock` that the Python `msvcrt` byte lock can see, so
/// it opens with an exclusive share mode (`share_mode(0)`); there is no shared
/// form, so a shared request degrades to exclusive there. Other platforms open
/// without locking (not a multi-writer target).
///
/// This is the single primitive under both the writer lock (`PersistentLock`) and
/// the LSN counter lock (`lsn::open_exclusive`); each wraps it in its own
/// retry-until-deadline loop with its own contention message.
pub(crate) fn try_lock_file(path: &Path, exclusive: bool) -> std::io::Result<Option<File>> {
    #[cfg(unix)]
    {
        use rustix::fs::{flock, FlockOperation};
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        let operation = if exclusive {
            FlockOperation::NonBlockingLockExclusive
        } else {
            FlockOperation::NonBlockingLockShared
        };
        match flock(&file, operation) {
            Ok(()) => Ok(Some(file)),
            Err(error)
                if error == rustix::io::Errno::WOULDBLOCK || error == rustix::io::Errno::AGAIN =>
            {
                Ok(None)
            }
            Err(error) => Err(std::io::Error::from(error)),
        }
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const ERROR_SHARING_VIOLATION: i32 = 32;
        const ERROR_LOCK_VIOLATION: i32 = 33;
        let _ = exclusive;
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .share_mode(0)
            .open(path)
        {
            Ok(file) => Ok(Some(file)),
            Err(error)
                if matches!(
                    error.raw_os_error(),
                    Some(ERROR_SHARING_VIOLATION) | Some(ERROR_LOCK_VIOLATION)
                ) =>
            {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = exclusive;
        Ok(Some(
            OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(path)?,
        ))
    }
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

/// Encodes a JSON number byte-for-byte identical to CPython's ``json.dumps``.
///
/// Integers (``i64``/``u64``) already render identically in serde_json and
/// CPython, so they pass through ``Number::to_string``. Floats do not: serde_json
/// formats with Ryu (``1e20``, ``1e-7``) while CPython uses ``repr(float)``
/// (``1e+20``, ``1e-07``) -- a different exponent sign/padding and a different
/// fixed-vs-scientific threshold. Both engines checksum a canonical body and
/// verify each other's writes across the FFI boundary, so a float in any
/// re-checksummed body would diverge the shared hash and fail the peer's reopen,
/// exactly like the non-ASCII string bug did (see [`encode_json_string_ascii`]).
/// ``serde_json::Number`` can only hold a finite ``f64`` (``Number::from_f64``
/// rejects NaN/inf), so no non-finite case reaches the float branch.
fn encode_json_number(number: &serde_json::Number) -> String {
    if number.is_f64() {
        if let Some(value) = number.as_f64() {
            return format_python_float(value);
        }
    }
    number.to_string()
}

/// Renders a finite ``f64`` exactly like CPython's ``repr`` / ``json.dumps``.
///
/// Rust's ``{:e}`` yields the shortest round-tripping digits with one leading
/// digit and a base-10 exponent (``[-]d[.ddd]e[-]EXP``). CPython's dtoa produces
/// the same (unique) shortest digits, so only the layout differs: CPython uses
/// scientific notation iff the decimal point falls at ``decpt <= -4`` or
/// ``decpt > 16`` (with ``decpt == EXP + 1``), always writes the exponent sign,
/// zero-pads the exponent magnitude to at least two digits, and appends ``.0`` to
/// an integral value written in fixed notation.
fn format_python_float(value: f64) -> String {
    // Shortest digits + exponent, e.g. "-1.5e20", "1e-7", "0e0", "-0e0".
    let exponential = format!("{value:e}");
    let (negative, rest) = match exponential.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, exponential.as_str()),
    };
    let (mantissa, exponent) = rest
        .split_once('e')
        .expect("Rust `{:e}` always emits an exponent");
    let exponent: i32 = exponent
        .parse()
        .expect("Rust `{:e}` emits an integer exponent");
    // Significant digits with the point removed: "1.5" -> "15", "1" -> "1".
    let digits: String = mantissa.chars().filter(|&ch| ch != '.').collect();
    let digit_count = digits.len() as i32;

    let mut out = String::new();
    if negative {
        out.push('-');
    }
    // CPython 'r' format uses scientific notation iff decpt <= -4 or decpt > 16,
    // with decpt == exponent + 1, i.e. exponent < -4 or exponent >= 16.
    if !(-4..16).contains(&exponent) {
        // Scientific: leading digit, optional fraction, signed >=2-digit exponent.
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            out.push('0');
        }
        out.push_str(&magnitude.to_string());
    } else {
        // Fixed: place the decimal point `exponent + 1` digits from the left.
        let point = exponent + 1;
        if point <= 0 {
            out.push_str("0.");
            out.push_str(&"0".repeat((-point) as usize));
            out.push_str(&digits);
        } else if point >= digit_count {
            out.push_str(&digits);
            out.push_str(&"0".repeat((point - digit_count) as usize));
            out.push_str(".0");
        } else {
            let split = point as usize;
            out.push_str(&digits[..split]);
            out.push('.');
            out.push_str(&digits[split..]);
        }
    }
    out
}

pub(crate) fn py_canonical_json(value: &Value) -> CoreResult<String> {
    match value {
        Value::Null => Ok("null".to_string()),
        Value::Bool(true) => Ok("true".to_string()),
        Value::Bool(false) => Ok("false".to_string()),
        Value::Number(number) => Ok(encode_json_number(number)),
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

/// Wraps a sidecar `body` in the checksummed base payload every JSON sidecar
/// store (`.tvtext`, `.tvlex`, `.tvann`) writes: `{schema_version, body_sha256,
/// body}`. The body carries its own `schema_version` too, mirroring the shape the
/// Python writer produces and cross-verifies. Extracted so the stores share one
/// definition of the wrap rather than re-rolling it each.
pub(crate) fn checksummed_body_payload(schema_version: i64, body: Value) -> CoreResult<Value> {
    Ok(serde_json::json!({
        "schema_version": schema_version,
        "body_sha256": body_sha256(&body)?,
        "body": body,
    }))
}

/// Reads a checksummed base payload written by [`checksummed_body_payload`],
/// verifying the outer schema version and the body checksum, and returns the inner
/// `body` value for the store to parse. `context` names the store for error
/// messages (e.g. `"tvann index"`).
pub(crate) fn read_checksummed_body(
    base_path: &Path,
    schema_version: i64,
    context: &str,
) -> CoreResult<Value> {
    let payload = read_json(base_path, &format!("{context} base"))?;
    let payload = value_object(&payload, &format!("{context} base"))?;
    if get_i64(payload, "schema_version", -1) != schema_version {
        return Err(corrupt(format!(
            "unsupported {context} base schema version"
        )));
    }
    let body = payload
        .get("body")
        .ok_or_else(|| corrupt(format!("{context} base body is missing")))?;
    if body_sha256(body)? != get_str(payload, "body_sha256") {
        return Err(corrupt(format!("{context} base failed checksum")));
    }
    Ok(body.clone())
}

/// Validates a sidecar's commit-manifest block: the schema-version gate and, when
/// a `base` object is present, the base file's checksum. Shared by every sidecar
/// store, which all embed a `{schema_version, base: {sha256, ...}}` block in the
/// commit manifest. `context` names the store for error messages.
pub(crate) fn validate_sidecar_manifest(
    base_path: &Path,
    manifest: &Value,
    schema_version: i64,
    context: &str,
) -> CoreResult<()> {
    let manifest = value_object(manifest, &format!("{context} manifest"))?;
    if get_i64(manifest, "schema_version", -1) != schema_version {
        return Err(corrupt(format!(
            "unsupported {context} manifest schema version"
        )));
    }
    if let Some(base) = manifest.get("base").and_then(Value::as_object) {
        verify_file_sha256(
            base_path,
            get_str(base, "sha256"),
            &format!("{context} base"),
        )?;
    }
    Ok(())
}

/// Builds the `base` object every sidecar store embeds in its commit manifest:
/// `{file_name, sha256, file_bytes, ...extra}`. `extra` carries the store-specific
/// counters (`document_count`, `cluster_count`, `calibration_fingerprint`). The
/// caller wraps this in `{schema_version, base, ...}` (base-only stores add
/// nothing more; delta-journaled stores also add `deltas`/`next_seq`). `context`
/// names the store for the (rare) metadata-read error.
pub(crate) fn sidecar_base_block(
    base_path: &Path,
    context: &str,
    extra: impl IntoIterator<Item = (&'static str, Value)>,
) -> CoreResult<Value> {
    let file_bytes = base_path
        .metadata()
        .map_err(|error| corrupt(format!("{context} base metadata failed: {error}")))?
        .len();
    let mut base = Map::new();
    base.insert(
        "file_name".to_string(),
        Value::from(
            base_path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned(),
        ),
    );
    base.insert(
        "sha256".to_string(),
        Value::from(sha256_file_hex(base_path)?),
    );
    base.insert("file_bytes".to_string(), Value::from(file_bytes));
    for (key, value) in extra {
        base.insert(key.to_string(), value);
    }
    Ok(Value::Object(base))
}

pub(crate) fn write_py_json(path: &Path, value: &Value, fsync: bool) -> CoreResult<usize> {
    let text = py_canonical_json(value)?;
    write_text_atomic(path, &text, fsync)
}

/// The first bytes of a zstd frame (little-endian magic ``0xFD2FB528``). Read
/// paths use it to tell a compressed payload from legacy plain JSON.
const ZSTD_MAGIC: [u8; 4] = [0x28, 0xb5, 0x2f, 0xfd];

/// Compression level for the document-text store. The base is rewritten only at
/// checkpoint and the segments are off the per-add WAL hot path, so a high-ratio
/// level is worth the CPU; retained payload text (e.g. mem0 memories) compresses
/// several-fold.
const TEXT_ZSTD_LEVEL: i32 = 19;

/// Writes ``value`` as canonical (CPython-identical) JSON, zstd-compressed, via
/// the same atomic temp-and-rename as [`write_py_json`]. The on-disk bytes are a
/// zstd frame; callers checksum the file bytes for integrity, while the logical
/// ``body_sha256`` is taken over the JSON value and is unaffected by compression.
pub(crate) fn write_py_json_zstd(path: &Path, value: &Value, fsync: bool) -> CoreResult<usize> {
    let text = py_canonical_json(value)?;
    let compressed = zstd::encode_all(text.as_bytes(), TEXT_ZSTD_LEVEL).map_err(|error| {
        corrupt(format!("could not compress {}: {error}", path.display()))
    })?;
    write_bytes_atomic(path, &compressed, fsync)
}

/// Reads a JSON value that may be zstd-compressed (written by
/// [`write_py_json_zstd`]) or plain UTF-8 JSON (a legacy/uncompressed store). The
/// zstd frame magic distinguishes the two, so existing uncompressed stores keep
/// loading unchanged.
pub(crate) fn read_maybe_zstd_json(path: &Path, context: &str) -> CoreResult<Value> {
    let data =
        fs::read(path).map_err(|error| corrupt(format!("{context} could not be read: {error}")))?;
    let json = if data.starts_with(&ZSTD_MAGIC) {
        zstd::decode_all(&data[..])
            .map_err(|error| corrupt(format!("{context} could not be decompressed: {error}")))?
    } else {
        data
    };
    serde_json::from_slice(&json).map_err(|error| corrupt(format!("{context} is corrupt: {error}")))
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

    // The number branch must match CPython's json.dumps(float) byte-for-byte;
    // serde_json's Ryu output (1e20, 1e-7) diverges from CPython's repr (1e+20,
    // 1e-07) on exponent layout and the fixed/scientific threshold.
    #[test]
    fn numbers_match_python_json_dumps() {
        // (value, json.dumps(value)) pairs captured from CPython 3.
        let floats: &[(f64, &str)] = &[
            (1e+20, "1e+20"),
            (1e-07, "1e-07"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e+16, "1e+16"),
            (1000000000000000.0, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (1e-05, "1e-05"),
            (-0.0, "-0.0"),
            (0.0, "0.0"),
            (123.456, "123.456"),
            (1e+100, "1e+100"),
            (1e-100, "1e-100"),
            (2.5, "2.5"),
            (100.0, "100.0"),
            (300.0, "300.0"),
            (9999999999999998.0, "9999999999999998.0"),
            (1.5e-10, "1.5e-10"),
            (45.125, "45.125"),
            (6.022e+23, "6.022e+23"),
            (5.0, "5.0"),
            (0.5, "0.5"),
            (-2.5e-08, "-2.5e-08"),
            (1234567890123456.0, "1234567890123456.0"),
            (1.2345678901234568e+16, "1.2345678901234568e+16"),
        ];
        for &(value, expected) in floats {
            assert_eq!(
                py_canonical_json(&json!(value)).unwrap(),
                expected,
                "float repr mismatch for {value}"
            );
        }
        // Integers carry no decimal point, matching json.dumps(int).
        for (value, expected) in [(0_i64, "0"), (42, "42"), (-7, "-7")] {
            assert_eq!(py_canonical_json(&json!(value)).unwrap(), expected);
        }
        assert_eq!(
            py_canonical_json(&json!(9007199254740993_i64)).unwrap(),
            "9007199254740993"
        );
    }

    // A float-bearing body survives serialize -> parse -> re-serialize and equals
    // CPython's json.dumps(body, sort_keys=True), so a native write of float
    // metadata re-checksums identically under the Python reader.
    #[test]
    fn canonical_form_is_idempotent_for_floats() {
        let body = json!({
            "schema_version": 2,
            "metadata": {"score": 0.1, "scale": 1e20, "tiny": 1e-7, "whole": 5.0},
        });
        let written = py_canonical_json(&body).unwrap();
        assert_eq!(
            written,
            "{\"metadata\": {\"scale\": 1e+20, \"score\": 0.1, \"tiny\": 1e-07, \"whole\": 5.0}, \"schema_version\": 2}"
        );
        let parsed: Value = serde_json::from_str(&written).unwrap();
        assert_eq!(written, py_canonical_json(&parsed).unwrap());
    }
}
