use lodedb_core::vector::index::CoreVectorChunk;
use lodedb_core::vector::turbovec::TurboVecNativeIndex;
use turbovec::search;

// This test lives alone to isolate the process-global block-skip counter.
fn axis_query(axis: usize) -> Vec<f32> {
    let mut query = vec![0.0f32; 8];
    query[axis] = 1.0;
    query
}

#[test]
fn cluster_ordered_allowlist_skips_more_simd_blocks() {
    let dim = 8;
    let rows_per_cluster = 32;
    let mut chunks = Vec::with_capacity(rows_per_cluster * 4);
    for row in 0..rows_per_cluster {
        for cluster in 0..4 {
            let mut vector = vec![0.0; dim];
            vector[cluster * 2] = 1.0;
            vector[cluster * 2 + 1] = (row + 1) as f32 * 0.001;
            let chunk_id = format!("cluster-{cluster}-row-{row}");
            chunks.push(CoreVectorChunk::new(chunk_id.clone(), chunk_id, vector));
        }
    }
    let mut index = TurboVecNativeIndex::build(&chunks, dim, 4, 1).unwrap();
    let cluster_zero: Vec<String> = (0..rows_per_cluster)
        .map(|row| format!("cluster-0-row-{row}"))
        .collect();
    let allowlist = index.stable_ids_for_chunks(&cluster_zero);
    let mut ordered = Vec::with_capacity(chunks.len());
    for cluster in 0..4 {
        let chunk_ids: Vec<String> = (0..rows_per_cluster)
            .map(|row| format!("cluster-{cluster}-row-{row}"))
            .collect();
        ordered.extend(index.stable_ids_for_chunks(&chunk_ids));
    }

    search::reset_blocks_skipped_by_mask();
    index
        .search_with_stable_allowlist(&axis_query(0), 5, &allowlist)
        .unwrap();
    let insertion_order_skips = search::blocks_skipped_by_mask();

    index.reorder_slots(&ordered).unwrap();
    search::reset_blocks_skipped_by_mask();
    index
        .search_with_stable_allowlist(&axis_query(0), 5, &allowlist)
        .unwrap();
    let cluster_order_skips = search::blocks_skipped_by_mask();
    assert!(
        cluster_order_skips > insertion_order_skips,
        "cluster order skipped {cluster_order_skips} blocks, insertion order skipped {insertion_order_skips}"
    );
}
