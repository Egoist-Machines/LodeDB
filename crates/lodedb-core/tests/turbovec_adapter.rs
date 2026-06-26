use lodedb_core::vector::index::CoreVectorChunk;
use lodedb_core::vector::turbovec::TurboVecNativeIndex;

const DIM: usize = 64;

fn one_hot(axis: usize) -> Vec<f32> {
    let mut row = vec![0.0; DIM];
    row[axis] = 1.0;
    row
}

fn chunks() -> Vec<CoreVectorChunk> {
    (0..8)
        .map(|i| CoreVectorChunk::new(format!("doc-{i}:chunk-0"), format!("doc-{i}"), one_hot(i)))
        .collect()
}

fn build() -> TurboVecNativeIndex {
    TurboVecNativeIndex::build(&chunks(), DIM, 4, 7).expect("adapter must build")
}

#[test]
fn unfiltered_search_resolves_chunk_and_document_ids() {
    let index = build();
    let hits = index.search(&one_hot(3), 3, &[]).expect("search");

    assert_eq!(hits[0].chunk_id, "doc-3:chunk-0");
    assert_eq!(hits[0].document_id, "doc-3");
    assert!(hits.windows(2).all(|pair| pair[0].score >= pair[1].score));
    assert_eq!(index.backend_metadata().vector_count, 8);
}

#[test]
fn filtered_search_applies_chunk_allowlist_by_stable_id() {
    let index = build();
    let allowlist = vec![
        "doc-1:chunk-0".to_string(),
        "doc-3:chunk-0".to_string(),
        "missing".to_string(),
    ];
    let hits = index.search(&one_hot(3), 5, &allowlist).expect("search");

    assert_eq!(
        hits.iter()
            .map(|hit| hit.chunk_id.as_str())
            .collect::<Vec<_>>(),
        ["doc-3:chunk-0", "doc-1:chunk-0"]
    );

    let empty = index
        .search(&one_hot(3), 5, &["missing".to_string()])
        .expect("empty allowlist");
    assert!(empty.is_empty());
}

#[test]
fn batch_search_preserves_query_rows() {
    let index = build();
    let rows = index
        .search_batch(&[one_hot(0), one_hot(5)], 2, &[])
        .expect("batch search");

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0].chunk_id, "doc-0:chunk-0");
    assert_eq!(rows[1][0].chunk_id, "doc-5:chunk-0");
}

#[test]
fn write_and_load_tvim_round_trips_search() {
    let chunks = chunks();
    let index = TurboVecNativeIndex::build(&chunks, DIM, 4, 11).expect("build");
    let path = std::env::temp_dir().join(format!(
        "lodedb_core_turbovec_adapter_{}.tvim",
        std::process::id()
    ));

    let metrics = index.write(&path).expect("write");
    assert_eq!(metrics.compact_backend, "turbovec_idmap");
    assert!(metrics.snapshot_bytes > 0);
    assert!(!metrics.raw_payload_text_present);

    let loaded = TurboVecNativeIndex::load(&path, &chunks, 11).expect("load");
    std::fs::remove_file(&path).ok();

    assert_eq!(
        loaded.search(&one_hot(6), 1, &[]).expect("search")[0].chunk_id,
        "doc-6:chunk-0"
    );
    assert_eq!(loaded.backend_metadata(), index.backend_metadata());
}
