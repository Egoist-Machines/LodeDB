use std::path::PathBuf;

use lodedb_core::storage::wal;
use lodedb_core::storage::{
    write_generation_commit, GenerationCommitInput, GenerationWriteOptions,
};
use serde_json::json;

const INDEX_KEY: &str = "6f78dec251fa5e544784ac1af95b0ae6530cad714a2d34f8c4615740ecbf8205";

fn main() {
    let Some(target) = std::env::args_os().nth(1) else {
        eprintln!("usage: write_wal_fixture <output-dir>");
        std::process::exit(2);
    };
    let target = PathBuf::from(target);
    std::fs::create_dir_all(&target).expect("create fixture directory");
    let state = json!({
        "cache_reuse_count": 0,
        "chunks": [],
        "client_id_hash": INDEX_KEY,
        "columnar_generation": 1,
        "created_at": "2026-06-26T00:00:00+00:00",
        "delete_count": 0,
        "deleted_chunk_count": 0,
        "document_chunk_ids": {},
        "document_hashes": {},
        "document_metadata": {},
        "embedded_chunk_count": 0,
        "fallback_count": 0,
        "fallback_reasons": {},
        "index_id": "default",
        "index_key": INDEX_KEY,
        "metadata": {},
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "name": "lodedb-local",
        "native_dim": 384,
        "provider": "local_open",
        "query_count": 0,
        "route_profile": "minilm-turbovec",
        "schema_version": 1,
        "status": "ready",
        "storage_profile": "turbovec_direct",
        "task": "direct-turbovec",
        "turbovec_bit_width": 4,
        "updated_at": "2026-06-26T00:00:00+00:00"
    });
    write_generation_commit(
        &target,
        GenerationCommitInput {
            index_key: INDEX_KEY,
            generation: 1,
            base_epoch: 1,
            state: &state,
            tvim: None,
            raw_text: None,
            lexical_tokens: None,
            multivec: None,
            compress_text: true,
        },
        GenerationWriteOptions::default(),
    )
    .expect("write base fixture");
    wal::append_record(
        &wal::wal_path(&target, INDEX_KEY),
        "upsert_documents",
        &json!({
            "client_id": "lodedb-local",
            "index_id": "default",
            "documents": [{
                "document_id": "rust-wal-alpha",
                "text": "Rust authored WAL payload",
                "metadata": {"kind": "fixture", "tenant": "rust"}
            }]
        }),
        false,
    )
    .expect("append WAL");
    std::fs::write(
        target.join("fixture_manifest.txt"),
        "{\n  \"fixture_schema_version\": 1,\n  \"lodedb_version\": \"0.4.0\",\n  \"metadata\": {\n    \"commit_mode\": \"wal\",\n    \"mode\": \"rust_wal\"\n  }\n}\n",
    )
    .expect("write fixture manifest");
}
