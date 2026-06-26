//! Chunking and chunk-id helpers that mirror the Python oracle.

use crate::error::{CoreError, CoreErrorCode};

/// Splits text into deterministic non-empty chunks bounded by character count.
pub fn chunk_text(text: &str, character_limit: usize) -> Result<Vec<String>, CoreError> {
    if character_limit == 0 {
        return Err(CoreError::new(
            CoreErrorCode::InvalidArgument,
            "character_limit must be positive",
        ));
    }
    let stripped = text.trim();
    if stripped.is_empty() {
        return Ok(Vec::new());
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    for character in stripped.chars() {
        current.push(character);
        if current.chars().count() == character_limit {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    Ok(chunks)
}

/// Builds a stable chunk ID from document ID, normalized chunk hash, and occurrence.
pub fn chunk_id_for_hash(document_id: &str, chunk_hash: &str, occurrence: usize) -> String {
    let prefix: String = chunk_hash.chars().take(12).collect();
    format!("{document_id}:{prefix}:{occurrence:04}")
}

#[cfg(test)]
mod tests {
    use super::{chunk_id_for_hash, chunk_text};

    #[test]
    fn chunks_trimmed_text_by_character_count() {
        assert_eq!(
            chunk_text("  abcdefg  ", 3).expect("chunk"),
            vec!["abc".to_string(), "def".to_string(), "g".to_string()]
        );
    }

    #[test]
    fn chunk_ids_include_hash_prefix_and_occurrence() {
        assert_eq!(
            chunk_id_for_hash(
                "doc-alpha",
                "88d4741101bdc046b9e5eecbd45bdeac103ffb3e68a51b157d2d2f304999c461",
                7,
            ),
            "doc-alpha:88d4741101bd:0007"
        );
    }
}
