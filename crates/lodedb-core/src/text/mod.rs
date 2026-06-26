//! Text identity helpers shared by text ingest and lexical indexing.

pub mod chunk;
pub mod hash;

pub use chunk::{chunk_id_for_hash, chunk_text};
pub use hash::{normalized_chunk_hash, sha256_text};
