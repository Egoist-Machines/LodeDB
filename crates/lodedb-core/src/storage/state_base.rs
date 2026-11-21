//! Streaming writer for the generation state base (`g<epoch>.json`).
//!
//! The state base is canonical (CPython `json.dumps(sort_keys=True)`) JSON holding
//! the scalar state header plus four per-document collections. Building it as a
//! `serde_json::Value` tree, rendering that to one `String`, writing it, and then
//! re-reading the file to checksum it costs several transient copies of the whole
//! document table, which is what a base rewrite at tens of millions of documents
//! spends its time and memory on (issue #89).
//!
//! [`write_state_base`] emits the same bytes straight from borrowed document
//! records: it never materializes the tree or the text, renders the per-chunk
//! rows (one SHA-256 each) on all cores, and hashes the output as it is written,
//! so the file digest the state-journal manifest records needs no second read.
//! The output is byte-identical to `py_canonical_json` over the equivalent
//! payload, so stores, the Python engine, and the cloud transfer core all see
//! the format they already verify.

use crate::storage::util::{
    corrupt, encode_json_string_ascii, fsync_dir, invalid, write_py_canonical_json, CanonicalSink,
    CoreResult,
};
use crate::types::CoreMetadata;
use rayon::prelude::*;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// One document's borrowed state row. `chunk_ids` yields the document's chunk
/// ids in stored order.
pub struct StateDocument<'a, C> {
    pub document_id: &'a str,
    pub content_hash: &'a str,
    pub metadata: &'a CoreMetadata,
    pub chunk_ids: C,
}

/// What a streamed state base wrote: the counts and the file digest the
/// state-journal manifest records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateBaseSummary {
    pub document_count: usize,
    pub chunk_count: usize,
    pub sha256: String,
    pub file_bytes: u64,
}

/// Output is hashed and written in blocks of this size.
const FLUSH_BYTES: usize = 4 * 1024 * 1024;
/// Chunk rows rendered per parallel task.
const CHUNK_ROWS_PER_TASK: usize = 4096;
/// Chunk rows rendered between writes, which bounds the rendered text held in
/// memory to a few tens of megabytes whatever the corpus size.
const CHUNK_ROWS_PER_WINDOW: usize = 1 << 18;

const CHUNKS_KEY: &str = "chunks";
const DOCUMENT_CHUNK_IDS_KEY: &str = "document_chunk_ids";
const DOCUMENT_HASHES_KEY: &str = "document_hashes";
const DOCUMENT_METADATA_KEY: &str = "document_metadata";

enum Section<'h> {
    Header(&'h Value),
    Chunks,
    DocumentChunkIds,
    DocumentHashes,
    DocumentMetadata,
}

/// Writes canonical JSON to a file while hashing it. Hashing and the write of
/// each block run concurrently. I/O errors are held and surfaced by `finish`,
/// since `CanonicalSink` is infallible.
struct HashingFileSink {
    file: File,
    hasher: Sha256,
    buffer: Vec<u8>,
    bytes: u64,
    error: Option<std::io::Error>,
}

impl HashingFileSink {
    fn new(file: File) -> Self {
        Self {
            file,
            hasher: Sha256::new(),
            buffer: Vec::with_capacity(FLUSH_BYTES + 64 * 1024),
            bytes: 0,
            error: None,
        }
    }

    fn flush_block(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        self.bytes += self.buffer.len() as u64;
        let (hasher, file, buffer) = (&mut self.hasher, &mut self.file, &self.buffer);
        let failed = self.error.is_some();
        let ((), written) = rayon::join(
            || hasher.update(buffer),
            || {
                if failed {
                    Ok(())
                } else {
                    file.write_all(buffer)
                }
            },
        );
        if let Err(error) = written {
            self.error = Some(error);
        }
        self.buffer.clear();
    }

    fn finish(mut self, fsync: bool) -> std::io::Result<(String, u64)> {
        self.flush_block();
        if let Some(error) = self.error {
            return Err(error);
        }
        if fsync {
            self.file.sync_all()?;
        }
        Ok((format!("{:x}", self.hasher.finalize()), self.bytes))
    }
}

impl CanonicalSink for HashingFileSink {
    fn push_str(&mut self, text: &str) {
        self.buffer.extend_from_slice(text.as_bytes());
        if self.buffer.len() >= FLUSH_BYTES {
            self.flush_block();
        }
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Appends the lowercase SHA-256 hex digest of `text`, the same digest as
/// `text::hash::sha256_text`, without allocating.
fn push_sha256_hex(text: &str, out: &mut String) {
    for byte in Sha256::digest(text.as_bytes()) {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
}

/// Renders a run of `chunks` rows, each preceded by the `", "` item separator.
fn render_chunk_rows(rows: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(rows.len() * 160);
    for (chunk_id, document_id) in rows {
        out.push_str(", {\"chunk_id\": ");
        encode_json_string_ascii(chunk_id, &mut out);
        out.push_str(", \"content_hash\": \"");
        push_sha256_hex(chunk_id, &mut out);
        out.push_str("\", \"document_id\": ");
        encode_json_string_ascii(document_id, &mut out);
        out.push('}');
    }
    out
}

fn write_chunks_section<'a, I, C>(documents: I, sink: &mut impl CanonicalSink) -> usize
where
    I: Iterator<Item = StateDocument<'a, C>>,
    C: Iterator<Item = &'a str>,
{
    let mut rows: Vec<(&str, &str)> = Vec::new();
    for document in documents {
        let document_id = document.document_id;
        rows.extend(document.chunk_ids.map(|chunk_id| (chunk_id, document_id)));
    }
    // Stable, like the sort the tree-building writer applied, so equal chunk ids
    // keep document order and the output stays byte-identical.
    rows.par_sort_by(|left, right| left.0.cmp(right.0));
    sink.push_char('[');
    let mut first = true;
    for window in rows.chunks(CHUNK_ROWS_PER_WINDOW) {
        let rendered: Vec<String> = window
            .par_chunks(CHUNK_ROWS_PER_TASK)
            .map(render_chunk_rows)
            .collect();
        for piece in &rendered {
            // Every row carries a leading separator; drop it for the first row.
            sink.push_str(if first { &piece[2..] } else { piece.as_str() });
            first = false;
        }
    }
    sink.push_char(']');
    rows.len()
}

/// Writes one `{"<document id>": <value>, ...}` section, rendering each value
/// with `value`.
fn write_document_map<'a, I, C, S>(
    documents: I,
    sink: &mut S,
    mut value: impl FnMut(StateDocument<'a, C>, &mut S),
) where
    I: Iterator<Item = StateDocument<'a, C>>,
    C: Iterator<Item = &'a str>,
    S: CanonicalSink,
{
    sink.push_char('{');
    for (position, document) in documents.enumerate() {
        if position > 0 {
            sink.push_str(", ");
        }
        encode_json_string_ascii(document.document_id, sink);
        sink.push_str(": ");
        value(document, sink);
    }
    sink.push_char('}');
}

fn write_payload<'a, D, I, C>(
    header: &Map<String, Value>,
    documents: &D,
    sink: &mut impl CanonicalSink,
) -> CoreResult<(usize, usize)>
where
    D: Fn() -> I,
    I: Iterator<Item = StateDocument<'a, C>>,
    C: Iterator<Item = &'a str>,
{
    // The collections are object keys in sorted order, so every document id must
    // be strictly ascending for the output to be canonical.
    let mut document_count = 0usize;
    let mut previous: Option<&str> = None;
    for document in documents() {
        if previous.is_some_and(|previous| previous >= document.document_id) {
            return Err(invalid(
                "state base documents must be in strictly ascending id order",
            ));
        }
        previous = Some(document.document_id);
        document_count += 1;
    }

    // The document collections sort in among the header's scalar keys; a
    // collection key replaces a header key of the same name.
    let mut sections: BTreeMap<&str, Section<'_>> = header
        .iter()
        .map(|(key, value)| (key.as_str(), Section::Header(value)))
        .collect();
    sections.insert(CHUNKS_KEY, Section::Chunks);
    sections.insert(DOCUMENT_CHUNK_IDS_KEY, Section::DocumentChunkIds);
    sections.insert(DOCUMENT_HASHES_KEY, Section::DocumentHashes);
    sections.insert(DOCUMENT_METADATA_KEY, Section::DocumentMetadata);

    let mut chunk_count = 0usize;
    sink.push_char('{');
    for (position, (key, section)) in sections.iter().enumerate() {
        if position > 0 {
            sink.push_str(", ");
        }
        encode_json_string_ascii(key, sink);
        sink.push_str(": ");
        match section {
            Section::Header(value) => write_py_canonical_json(value, sink),
            Section::Chunks => chunk_count = write_chunks_section(documents(), sink),
            Section::DocumentChunkIds => write_document_map(documents(), sink, |document, sink| {
                sink.push_char('[');
                for (position, chunk_id) in document.chunk_ids.enumerate() {
                    if position > 0 {
                        sink.push_str(", ");
                    }
                    encode_json_string_ascii(chunk_id, sink);
                }
                sink.push_char(']');
            }),
            Section::DocumentHashes => write_document_map(documents(), sink, |document, sink| {
                encode_json_string_ascii(document.content_hash, sink);
            }),
            Section::DocumentMetadata => write_document_map(documents(), sink, |document, sink| {
                sink.push_char('{');
                for (position, (key, value)) in document.metadata.iter().enumerate() {
                    if position > 0 {
                        sink.push_str(", ");
                    }
                    encode_json_string_ascii(key, sink);
                    sink.push_str(": ");
                    encode_json_string_ascii(value, sink);
                }
                sink.push_char('}');
            }),
        }
    }
    sink.push_char('}');
    Ok((document_count, chunk_count))
}

/// Streams the state base for `header` plus `documents` to `path` (temp file and
/// rename, like every other persisted artifact) and returns its counts and file
/// digest.
///
/// `documents` is called once per pass and must yield the same documents each
/// time, in strictly ascending document-id order (a `BTreeMap` iteration).
pub fn write_state_base<'a, D, I, C>(
    path: &Path,
    header: &Map<String, Value>,
    documents: D,
    fsync: bool,
) -> CoreResult<StateBaseSummary>
where
    D: Fn() -> I,
    I: Iterator<Item = StateDocument<'a, C>>,
    C: Iterator<Item = &'a str>,
{
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
    let file = File::create(&temporary)
        .map_err(|error| corrupt(format!("could not create temp file: {error}")))?;
    let mut sink = HashingFileSink::new(file);
    let written = write_payload(header, &documents, &mut sink).and_then(|counts| {
        let (sha256, file_bytes) = sink
            .finish(fsync)
            .map_err(|error| corrupt(format!("could not write temp file: {error}")))?;
        Ok((counts, sha256, file_bytes))
    });
    let ((document_count, chunk_count), sha256, file_bytes) = match written {
        Ok(written) => written,
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
    };
    fs::rename(&temporary, path)
        .map_err(|error| corrupt(format!("could not replace {}: {error}", path.display())))?;
    if fsync {
        fsync_dir(path.parent().unwrap_or_else(|| Path::new(".")))?;
    }
    Ok(StateBaseSummary {
        document_count,
        chunk_count,
        sha256,
        file_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::{write_state_base, StateDocument, CHUNK_ROWS_PER_TASK};
    use crate::storage::util::{py_canonical_json, sha256_bytes_hex};
    use crate::types::CoreMetadata;
    use serde_json::{json, Map, Value};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    struct Doc {
        hash: String,
        metadata: CoreMetadata,
        chunk_ids: Vec<String>,
    }

    fn temp_path(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("lodedb-state-base-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir.join("g1.json")
    }

    /// The tree-building writer this module replaces: the reference bytes.
    fn reference(header: &Map<String, Value>, documents: &BTreeMap<String, Doc>) -> String {
        let mut chunks = Vec::new();
        let mut hashes = Map::new();
        let mut chunk_ids = Map::new();
        let mut metadata = Map::new();
        for (document_id, doc) in documents {
            hashes.insert(document_id.clone(), json!(doc.hash));
            chunk_ids.insert(document_id.clone(), json!(doc.chunk_ids));
            metadata.insert(document_id.clone(), json!(doc.metadata));
            for chunk_id in &doc.chunk_ids {
                chunks.push(json!({
                    "chunk_id": chunk_id,
                    "content_hash": crate::text::hash::sha256_text(chunk_id),
                    "document_id": document_id,
                }));
            }
        }
        chunks.sort_by(|left, right| {
            left.get("chunk_id")
                .and_then(Value::as_str)
                .cmp(&right.get("chunk_id").and_then(Value::as_str))
        });
        let mut payload = header.clone();
        payload.insert("chunks".to_string(), Value::Array(chunks));
        payload.insert("document_hashes".to_string(), Value::Object(hashes));
        payload.insert("document_chunk_ids".to_string(), Value::Object(chunk_ids));
        payload.insert("document_metadata".to_string(), Value::Object(metadata));
        py_canonical_json(&Value::Object(payload)).unwrap()
    }

    fn assert_matches_reference(name: &str, header: Value, documents: BTreeMap<String, Doc>) {
        let Value::Object(header) = header else {
            panic!("header must be an object");
        };
        let path = temp_path(name);
        let summary = write_state_base(
            &path,
            &header,
            || {
                documents.iter().map(|(document_id, doc)| StateDocument {
                    document_id,
                    content_hash: &doc.hash,
                    metadata: &doc.metadata,
                    chunk_ids: doc.chunk_ids.iter().map(String::as_str),
                })
            },
            false,
        )
        .unwrap();
        let written = std::fs::read(&path).unwrap();
        let expected = reference(&header, &documents);
        assert!(written == expected.as_bytes(), "{name}: bytes differ");
        assert_eq!(summary.sha256, sha256_bytes_hex(&written));
        assert_eq!(summary.file_bytes, written.len() as u64);
        assert_eq!(summary.document_count, documents.len());
        assert_eq!(
            summary.chunk_count,
            documents
                .values()
                .map(|doc| doc.chunk_ids.len())
                .sum::<usize>()
        );
        assert!(!path.with_file_name("g1.json.tmp").exists());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    fn header() -> Value {
        // Keys on both sides of every collection key, plus nested and float
        // values, so the section merge order and header encoding are exercised.
        json!({
            "ann": {"algorithm": "cluster", "nprobe": 16},
            "client_id_hash": "h\u{e9}",
            "delete_count": 3,
            "document_count_hint": 1.5e20,
            "embedded_chunk_count": 7,
            "fallback_reasons": {},
            "zz_last": null,
        })
    }

    #[test]
    fn empty_store_matches_reference() {
        assert_matches_reference("empty", header(), BTreeMap::new());
    }

    #[test]
    fn awkward_ids_and_metadata_match_reference() {
        let mut documents = BTreeMap::new();
        let ids = [
            "plain",
            "quote\"and\\slash",
            "control\u{01}\n\t\u{7f}",
            "caf\u{e9}",
            "\u{1f600} astral",
            "",
            " leading space",
        ];
        for (position, id) in ids.iter().enumerate() {
            let chunk_ids = (0..position % 4)
                .map(|chunk| format!("{id}#{chunk}"))
                .collect();
            let metadata = (0..position % 3)
                .map(|key| (format!("k\u{fc}{key}"), format!("v\"{position}\u{2028}")))
                .collect();
            documents.insert(
                (*id).to_string(),
                Doc {
                    hash: format!("hash-{position}"),
                    metadata,
                    chunk_ids,
                },
            );
        }
        assert_matches_reference("awkward", header(), documents);
    }

    #[test]
    fn chunk_order_is_global_across_parallel_tasks() {
        // More rows than one parallel task, with chunk ids that sort differently
        // from their owning documents, so the global sort and the piece joins are
        // both exercised.
        let mut documents = BTreeMap::new();
        let count = CHUNK_ROWS_PER_TASK * 2 + 37;
        for row in 0..count {
            documents.insert(
                format!("doc-{row:06}"),
                Doc {
                    hash: format!("{row:x}"),
                    metadata: CoreMetadata::new(),
                    chunk_ids: vec![
                        format!("chunk-{:06}", (row * 7919) % count),
                        format!("z-{row}"),
                    ],
                },
            );
        }
        assert_matches_reference("parallel", json!({"native_dim": 8}), documents);
    }

    #[test]
    fn header_collection_keys_are_replaced() {
        let mut documents = BTreeMap::new();
        documents.insert(
            "a".to_string(),
            Doc {
                hash: "h".to_string(),
                metadata: CoreMetadata::new(),
                chunk_ids: vec!["a".to_string()],
            },
        );
        assert_matches_reference(
            "override",
            json!({"chunks": "stale", "document_hashes": 1, "name": "n"}),
            documents,
        );
    }

    #[test]
    fn unsorted_documents_are_rejected_and_leave_no_file() {
        let path = temp_path("unsorted");
        let metadata = CoreMetadata::new();
        let ids = ["b", "a"];
        let result = write_state_base(
            &path,
            &Map::new(),
            || {
                ids.iter().map(|id| StateDocument {
                    document_id: id,
                    content_hash: "h",
                    metadata: &metadata,
                    chunk_ids: std::iter::empty(),
                })
            },
            false,
        );
        assert!(result.is_err());
        assert!(!path.exists());
        assert!(!path.with_file_name("g1.json.tmp").exists());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
