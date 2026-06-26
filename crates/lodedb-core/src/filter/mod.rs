//! Metadata filter validation, predicate evaluation, and indexed planning.

pub mod ast;
pub mod doc_set;
pub mod field_index;
pub mod predicate;
pub mod resolve;
pub mod validate;

pub use doc_set::{expand_doc_ids_to_chunk_ids, DocSet};
pub use field_index::{build_field_indexes, FieldIndex};
pub use predicate::matches_metadata_filter;
pub use resolve::resolve_filter;
pub use validate::{coerce_sdk_filter, validate_metadata_filter};
