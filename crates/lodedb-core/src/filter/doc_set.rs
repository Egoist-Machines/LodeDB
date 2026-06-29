//! Deterministic document-id and chunk-id set helpers for filter planning.

use std::collections::{BTreeMap, BTreeSet};

/// A deterministic set of document ids or chunk ids.
pub type DocSet = BTreeSet<String>;

/// Expands matching document ids to chunk ids using a document -> chunk-id index.
///
/// Documents absent from `document_chunks` have no indexed chunks and therefore add
/// nothing to the result. The returned set is sorted for deterministic bindings and
/// tests.
pub fn expand_doc_ids_to_chunk_ids(
    document_ids: &DocSet,
    document_chunks: &BTreeMap<String, Vec<String>>,
) -> DocSet {
    let mut chunk_ids = DocSet::new();
    for document_id in document_ids {
        if let Some(chunks) = document_chunks.get(document_id) {
            chunk_ids.extend(chunks.iter().cloned());
        }
    }
    chunk_ids
}
