use crate::storage::util::{
    body_sha256, checksummed_body_from_text, checksummed_body_payload, corrupt, get_i64, get_str,
    parse_json, read_json, read_text, sha256_file_hex, sidecar_base_block,
    validate_sidecar_manifest, value_object, write_pretty_json_atomic, write_py_json, CoreResult,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

pub const LEXICAL_INDEX_DELTA_DIR_SUFFIX: &str = ".tvlex-delta";
pub const LEXICAL_INDEX_MANIFEST_NAME: &str = "manifest.json";
pub const LEXICAL_INDEX_SCHEMA_VERSION: i64 = 1;

pub type TokenLists = Vec<Vec<String>>;

pub fn manifest_path(base_path: &Path) -> PathBuf {
    base_path
        .with_file_name(format!(
            "{}{}",
            base_path.file_name().unwrap().to_string_lossy(),
            LEXICAL_INDEX_DELTA_DIR_SUFFIX
        ))
        .join(LEXICAL_INDEX_MANIFEST_NAME)
}

pub fn load(
    base_path: &Path,
    manifest: Option<&Value>,
) -> CoreResult<BTreeMap<String, TokenLists>> {
    match read_bytes(base_path, manifest)? {
        Some(bytes) => bytes.parse(),
        None => Ok(BTreeMap::new()),
    }
}

/// The text of a committed lexical store, base and delta segments, read and
/// checked against the commit manifest but not parsed. A read-only open keeps
/// this instead of the token lists (a fraction of their in-memory size) and
/// parses it the first time something needs the tokens; holding the text
/// rather than the paths keeps that working after a writer collects the epoch.
#[derive(Debug, Clone)]
pub struct LexicalBytes {
    base: String,
    segments: Vec<String>,
}

/// Reads what [`load`] parses, with the same manifest, ordering and segment
/// checksum checks. `None` when the base file is absent.
pub fn read_bytes(base_path: &Path, manifest: Option<&Value>) -> CoreResult<Option<LexicalBytes>> {
    if !base_path.is_file() {
        return Ok(None);
    }
    if let Some(manifest) = manifest {
        validate_sidecar_manifest(
            base_path,
            manifest,
            LEXICAL_INDEX_SCHEMA_VERSION,
            "lexical index",
        )?;
    }
    let base = read_text(base_path, "lexical index base")?;
    let mut segments = Vec::new();
    if let Some(manifest) = manifest {
        let manifest = value_object(manifest, "lexical index manifest")?;
        let delta_dir = base_path.with_file_name(format!(
            "{}{}",
            base_path.file_name().unwrap().to_string_lossy(),
            LEXICAL_INDEX_DELTA_DIR_SUFFIX
        ));
        let mut previous_seq = -1_i64;
        for entry in manifest
            .get("deltas")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let entry = value_object(entry, "lexical index delta manifest entry")?;
            let sequence = get_i64(entry, "seq", -1);
            if sequence <= previous_seq {
                return Err(corrupt("lexical index manifest has out-of-order segments"));
            }
            previous_seq = sequence;
            let file_name = get_str(entry, "file_name");
            let path = delta_dir.join(file_name);
            if file_name.is_empty() || !path.is_file() {
                return Err(corrupt(format!(
                    "lexical index segment is missing: {file_name}"
                )));
            }
            if sha256_file_hex(&path)? != get_str(entry, "sha256") {
                return Err(corrupt(format!(
                    "lexical index segment failed checksum: {file_name}"
                )));
            }
            segments.push(read_text(&path, "lexical index segment")?);
        }
    }
    Ok(Some(LexicalBytes { base, segments }))
}

impl LexicalBytes {
    /// Visits the effective committed documents without a corpus-sized token
    /// tree. Slices resolve delta precedence first; each document is decoded
    /// once, sharing that decode between checksum validation and the builder.
    /// Callers must discard partial work on error. Visit order is unspecified.
    pub(crate) fn visit_documents(
        &self,
        mut visit: impl FnMut(&str, TokenLists),
    ) -> CoreResult<()> {
        let mut frames = vec![DocumentFrame::parse(&self.base, "documents", false)?];
        for segment in &self.segments {
            frames.push(DocumentFrame::parse(segment, "upserted", true)?);
        }
        let mut effective = BTreeMap::new();
        for frame in &frames {
            for id in &frame.deleted {
                effective.remove(id.as_str());
            }
            for (id, raw) in &frame.documents {
                effective.insert(id.as_str(), *raw);
            }
        }
        for frame in &frames {
            frame.verify_and_visit(&effective, &mut visit)?;
        }
        Ok(())
    }

    /// Parses the base and replays the segments in manifest order.
    pub fn parse(&self) -> CoreResult<BTreeMap<String, TokenLists>> {
        let mut documents = parse_base(&self.base)?;
        for segment in &self.segments {
            let body = segment_body(segment)?;
            for deleted in body
                .get("deleted")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                documents.remove(deleted);
            }
            if let Some(upserted) = body.get("upserted").and_then(Value::as_object) {
                for (document_id, tokens) in upserted {
                    documents.insert(document_id.clone(), normalize_token_lists(tokens)?);
                }
            }
        }
        Ok(documents)
    }
}

type DocumentSlices<'a> = BTreeMap<String, &'a serde_json::value::RawValue>;

/// A lexical frame retains only ids and borrowed JSON slices, never tokens.
/// Keeping key order here accepts noncanonical input while hashing exactly the
/// same canonical Python JSON as the eager loader (including unknown fields).
struct DocumentFrame<'a> {
    fields: DocumentSlices<'a>,
    documents: DocumentSlices<'a>,
    deleted: Vec<String>,
    expected_hash: String,
    collection: &'static str,
}

impl<'a> DocumentFrame<'a> {
    fn parse(text: &'a str, collection: &'static str, segment: bool) -> CoreResult<Self> {
        let parse_fields = |raw: &'a str| -> CoreResult<DocumentSlices<'a>> {
            serde_json::from_str(raw).map_err(|error| corrupt(format!("lexical index: {error}")))
        };
        let payload = parse_fields(text)?;
        let schema = payload
            .get("schema_version")
            .and_then(|raw| serde_json::from_str::<i64>(raw.get()).ok());
        if schema != Some(LEXICAL_INDEX_SCHEMA_VERSION) {
            return Err(corrupt("unsupported lexical index schema version"));
        }
        let body = payload
            .get("body")
            .ok_or_else(|| corrupt("lexical index body is missing"))?;
        let fields = parse_fields(body.get())?;
        let expected_hash = payload
            .get("body_sha256")
            .and_then(|raw| serde_json::from_str::<String>(raw.get()).ok())
            .unwrap_or_default();
        let documents = match fields.get(collection) {
            Some(raw) if raw.get().trim_start().starts_with('{') => parse_fields(raw.get())?,
            _ if segment => BTreeMap::new(),
            _ => return Err(corrupt("lexical index base documents must be an object")),
        };
        let deleted = if segment {
            fields
                .get("deleted")
                .map(|raw| parse_json(raw.get(), "lexical index deleted ids"))
                .transpose()?
                .as_ref()
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        } else {
            Vec::new()
        };
        Ok(Self {
            fields,
            documents,
            deleted,
            expected_hash,
            collection,
        })
    }

    fn verify_and_visit(
        &self,
        effective: &BTreeMap<&str, &serde_json::value::RawValue>,
        visit: &mut impl FnMut(&str, TokenLists),
    ) -> CoreResult<()> {
        use crate::storage::util::py_canonical_json;
        use sha2::{Digest, Sha256};

        let mut hasher = Sha256::new();
        hasher.update(b"{");
        for (offset, (key, raw)) in self.fields.iter().enumerate() {
            if offset > 0 {
                hasher.update(b", ");
            }
            hasher.update(py_canonical_json(&Value::String(key.clone()))?.as_bytes());
            hasher.update(b": ");
            if key == self.collection && raw.get().trim_start().starts_with('{') {
                hasher.update(b"{");
                for (offset, (id, tokens)) in self.documents.iter().enumerate() {
                    if offset > 0 {
                        hasher.update(b", ");
                    }
                    hasher.update(py_canonical_json(&Value::String(id.clone()))?.as_bytes());
                    hasher.update(b": ");
                    let value = parse_json(tokens.get(), "lexical index token lists")?;
                    hasher.update(py_canonical_json(&value)?.as_bytes());
                    // Validate even entries later replaced/deleted, as the eager
                    // loader does, but build only the final version of a unit.
                    let lists = normalize_token_lists(&value)?;
                    if effective
                        .get(id.as_str())
                        .is_some_and(|current| std::ptr::eq(*current, *tokens))
                    {
                        visit(id, lists);
                    }
                }
                hasher.update(b"}");
            } else {
                let value = parse_json(raw.get(), "lexical index body field")?;
                hasher.update(py_canonical_json(&value)?.as_bytes());
            }
        }
        hasher.update(b"}");
        if format!("{:x}", hasher.finalize()) != self.expected_hash {
            return Err(corrupt("lexical index failed checksum"));
        }
        Ok(())
    }
}
fn parse_base(text: &str) -> CoreResult<BTreeMap<String, TokenLists>> {
    let body = checksummed_body_from_text(text, LEXICAL_INDEX_SCHEMA_VERSION, "lexical index")?;
    let body_object = value_object(&body, "lexical index base body")?;
    let documents = body_object
        .get("documents")
        .and_then(Value::as_object)
        .ok_or_else(|| corrupt("lexical index base documents must be an object"))?;
    documents
        .iter()
        .map(|(key, value)| Ok((key.clone(), normalize_token_lists(value)?)))
        .collect()
}

pub fn record_base(
    base_path: &Path,
    documents: &BTreeMap<String, TokenLists>,
    fsync: bool,
) -> CoreResult<Value> {
    let body = serde_json::json!({
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "documents": documents,
    });
    let payload = checksummed_body_payload(LEXICAL_INDEX_SCHEMA_VERSION, body)?;
    write_py_json(base_path, &payload, fsync)?;
    let manifest_path = manifest_path(base_path);
    let previous = if manifest_path.is_file() {
        Some(read_json(&manifest_path, "lexical index manifest")?)
    } else {
        None
    };
    let next_seq = previous
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|object| object.get("next_seq"))
        .and_then(Value::as_u64)
        .unwrap_or(0)
        + 1;
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            corrupt(format!(
                "lexical index delta directory could not be created: {error}"
            ))
        })?;
    }
    let manifest = serde_json::json!({
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "base": sidecar_base_block(
            base_path,
            "lexical index",
            [("document_count", Value::from(documents.len()))],
        )?,
        "deltas": [],
        "next_seq": next_seq,
    });
    write_pretty_json_atomic(&manifest_path, &manifest, fsync)?;
    Ok(manifest)
}

pub fn append_delta(
    base_path: &Path,
    upserted: &BTreeMap<String, TokenLists>,
    deleted: &[String],
    document_count_after: usize,
    fsync: bool,
) -> CoreResult<Value> {
    let manifest_path = manifest_path(base_path);
    let mut manifest = read_json(&manifest_path, "lexical index manifest")?;
    let manifest_object = manifest
        .as_object_mut()
        .ok_or_else(|| corrupt("lexical index manifest must be a JSON object"))?;
    let sequence = manifest_object
        .get("next_seq")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let body = serde_json::json!({
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "upserted": upserted,
        "deleted": deleted,
    });
    let segment = serde_json::json!({
        "schema_version": LEXICAL_INDEX_SCHEMA_VERSION,
        "seq": sequence,
        "document_count_after": document_count_after,
        "body_sha256": body_sha256(&body)?,
        "body": body,
    });
    let segment_name = format!("lexical-{sequence:08}.lxd");
    let delta_dir = base_path.with_file_name(format!(
        "{}{}",
        base_path.file_name().unwrap().to_string_lossy(),
        LEXICAL_INDEX_DELTA_DIR_SUFFIX
    ));
    let segment_path = delta_dir.join(&segment_name);
    write_py_json(&segment_path, &segment, fsync)?;
    let deltas = manifest_object
        .entry("deltas")
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| corrupt("lexical index manifest deltas must be a list"))?;
    deltas.push(serde_json::json!({
        "file_name": segment_name,
        "sha256": sha256_file_hex(&segment_path)?,
        "file_bytes": segment_path.metadata().map_err(|error| corrupt(format!("lexical index segment metadata failed: {error}")))?.len(),
        "seq": sequence,
        "upserted": upserted.len(),
        "deleted": deleted.len(),
    }));
    manifest_object.insert("next_seq".to_string(), Value::from(sequence + 1));
    write_pretty_json_atomic(&manifest_path, &manifest, fsync)?;
    Ok(manifest)
}

fn segment_body(text: &str) -> CoreResult<Value> {
    let segment = parse_json(text, "lexical index segment")?;
    let segment = value_object(&segment, "lexical index segment")?;
    if get_i64(segment, "schema_version", -1) != LEXICAL_INDEX_SCHEMA_VERSION {
        return Err(corrupt("unsupported lexical index segment schema version"));
    }
    let body = segment
        .get("body")
        .ok_or_else(|| corrupt("lexical index segment body is missing"))?;
    if !body.is_object() {
        return Err(corrupt("lexical index segment body is missing"));
    }
    if body_sha256(body)? != get_str(segment, "body_sha256") {
        return Err(corrupt("lexical index segment failed checksum"));
    }
    Ok(body.clone())
}

fn normalize_token_lists(value: &Value) -> CoreResult<TokenLists> {
    let chunks = value
        .as_array()
        .ok_or_else(|| corrupt("lexical index token lists must be a list of chunks"))?;
    chunks
        .iter()
        .map(|chunk| {
            let tokens = chunk
                .as_array()
                .ok_or_else(|| corrupt("lexical index chunk must be a list of tokens"))?;
            Ok(tokens
                .iter()
                .map(|token| token.as_str().unwrap_or("").to_string())
                .collect())
        })
        .collect()
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use crate::storage::util::{checksummed_body_payload, py_canonical_json};
    use serde_json::json;

    fn bytes(body: Value) -> String {
        // Pretty output exercises whitespace, non-ASCII and escaped keys;
        // checksums still use the canonical Python encoding.
        serde_json::to_string_pretty(&checksummed_body_payload(1, body).unwrap()).unwrap()
    }

    #[test]
    fn document_visitor_matches_eager_parse_with_deltas_and_escapes() {
        let lexical = LexicalBytes {
            base: bytes(json!({"schema_version": 1, "documents": {
                "z": [["old"]], "a\\\"é": [["café", "😀", "", 9]], "empty": [], "chunks": [[], ["alpha"]]
            }})),
            segments: vec![
                bytes(
                    json!({"schema_version": 1, "deleted": ["z", "missing"], "upserted": {"chunks": [["new"]], "z": [["restored"]]}}),
                ),
                bytes(json!({"schema_version": 1, "deleted": ["z"], "upserted": {"final": [[]]}})),
            ],
        };
        let mut visited = BTreeMap::new();
        lexical
            .visit_documents(|id, lists| {
                visited.insert(id.to_string(), lists);
            })
            .unwrap();
        assert_eq!(visited, lexical.parse().unwrap());
        // Also accept compact canonical input, not just pretty JSON.
        let lexical = LexicalBytes {
            base: py_canonical_json(&serde_json::from_str::<Value>(&lexical.base).unwrap())
                .unwrap(),
            segments: lexical.segments,
        };
        let mut canonical = BTreeMap::new();
        lexical
            .visit_documents(|id, lists| {
                canonical.insert(id.to_string(), lists);
            })
            .unwrap();
        assert_eq!(canonical, visited);
    }

    #[test]
    fn document_visitor_rejects_corruption_and_invalid_replaced_entries() {
        let good = bytes(json!({"schema_version": 1, "documents": {"d": [["alpha"]]}}));
        let bad_tokens = bytes(json!({"schema_version": 1, "documents": {"d": ["not a chunk"]}}));
        let bad_documents = bytes(json!({"schema_version": 1, "documents": []}));
        let bad_schema = good.replacen("\"schema_version\": 1", "\"schema_version\": 2", 2);
        for base in [
            good.replace("alpha", "beta"),
            bad_tokens,
            bad_documents,
            bad_schema,
        ] {
            let lexical = LexicalBytes {
                base,
                segments: vec![bytes(
                    json!({"schema_version": 1, "deleted": ["d"], "upserted": {}}),
                )],
            };
            assert!(lexical.parse().is_err());
            assert!(lexical.visit_documents(|_, _| {}).is_err());
        }
        let lexical = LexicalBytes {
            base: good,
            segments: vec![
                bytes(json!({"schema_version": 1, "upserted": {"d": [["alpha"]]}}))
                    .replace("alpha", "beta"),
            ],
        };
        assert!(lexical.visit_documents(|_, _| {}).is_err());
    }
}
