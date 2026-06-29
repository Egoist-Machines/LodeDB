use lodedb_core::text::chunk::{chunk_id_for_hash, chunk_text};
use lodedb_core::text::hash::{normalized_chunk_hash, sha256_text};
use lodedb_core::vector::stable_id::stable_uint64_ids_for_chunk_ids;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct IdentityFixtures {
    hash_cases: Vec<HashCase>,
    chunk_cases: Vec<ChunkCase>,
    chunk_id_cases: Vec<ChunkIdCase>,
    document_cases: Vec<DocumentCase>,
    stable_id_cases: Vec<StableIdCase>,
}

#[derive(Debug, Deserialize)]
struct HashCase {
    text: String,
    sha256: String,
    normalized_chunk_hash: String,
}

#[derive(Debug, Deserialize)]
struct ChunkCase {
    text: String,
    limit: usize,
    chunks: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ChunkIdCase {
    document_id: String,
    chunk_hash: String,
    occurrence: usize,
    chunk_id: String,
}

#[derive(Debug, Deserialize)]
struct DocumentCase {
    document_id: String,
    text: String,
    limit: usize,
    chunks: Vec<String>,
    chunk_hashes: Vec<String>,
    chunk_ids: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StableIdCase {
    chunk_ids: Vec<String>,
    stable_ids: Vec<u64>,
}

fn fixtures() -> IdentityFixtures {
    serde_json::from_str(include_str!(
        "../../../tests/fixtures/native_core_identity/identity.json"
    ))
    .expect("identity fixture must parse")
}

#[test]
fn hash_fixtures_match_python_oracle() {
    for case in fixtures().hash_cases {
        assert_eq!(sha256_text(&case.text), case.sha256);
        assert_eq!(
            normalized_chunk_hash(&case.text),
            case.normalized_chunk_hash
        );
    }
}

#[test]
fn chunk_fixtures_match_python_oracle() {
    for case in fixtures().chunk_cases {
        assert_eq!(
            chunk_text(&case.text, case.limit).expect("chunk"),
            case.chunks
        );
    }
}

#[test]
fn chunk_id_fixtures_match_python_oracle() {
    for case in fixtures().chunk_id_cases {
        assert_eq!(
            chunk_id_for_hash(&case.document_id, &case.chunk_hash, case.occurrence),
            case.chunk_id
        );
    }
}

#[test]
fn document_identity_fixtures_match_python_oracle() {
    for case in fixtures().document_cases {
        let chunks = chunk_text(&case.text, case.limit).expect("chunk document");
        assert_eq!(chunks, case.chunks);
        let chunk_hashes = chunks
            .iter()
            .map(|chunk| normalized_chunk_hash(chunk))
            .collect::<Vec<_>>();
        assert_eq!(chunk_hashes, case.chunk_hashes);

        let mut seen = std::collections::BTreeMap::<String, usize>::new();
        let chunk_ids = chunk_hashes
            .iter()
            .map(|chunk_hash| {
                let occurrence = *seen.get(chunk_hash).unwrap_or(&0);
                seen.insert(chunk_hash.clone(), occurrence + 1);
                chunk_id_for_hash(&case.document_id, chunk_hash, occurrence)
            })
            .collect::<Vec<_>>();
        assert_eq!(chunk_ids, case.chunk_ids);
    }
}

#[test]
fn stable_id_fixtures_match_python_oracle() {
    for case in fixtures().stable_id_cases {
        assert_eq!(
            stable_uint64_ids_for_chunk_ids(&case.chunk_ids),
            case.stable_ids
        );
    }
}
