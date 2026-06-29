//! Stable uint64 id mapping for TurboVec rows.

use crate::text::hash::sha256_digest;

/// Maps chunk IDs to deterministic nonzero uint64 IDs with collision repair.
pub fn stable_uint64_ids_for_chunk_ids(chunk_ids: &[String]) -> Vec<u64> {
    let candidates = chunk_ids
        .iter()
        .map(|chunk_id| stable_uint64_for_text(chunk_id))
        .collect::<Vec<_>>();
    repair_stable_uint64_candidates(&candidates)
}

/// Returns the first eight SHA-256 bytes as a stable little-endian uint64.
pub fn stable_uint64_for_text(value: &str) -> u64 {
    let digest = sha256_digest(value.as_bytes());
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 digest always has at least eight bytes"),
    )
}

pub(crate) fn repair_stable_uint64_candidates(candidates: &[u64]) -> Vec<u64> {
    let mut used = std::collections::BTreeSet::new();
    let mut ids = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let mut repaired = *candidate;
        while repaired == 0 || used.contains(&repaired) {
            repaired = repaired.wrapping_add(1);
        }
        used.insert(repaired);
        ids.push(repaired);
    }
    ids
}

#[cfg(test)]
mod tests {
    use super::{
        repair_stable_uint64_candidates, stable_uint64_for_text, stable_uint64_ids_for_chunk_ids,
    };

    #[test]
    fn stable_id_uses_little_endian_sha256_prefix() {
        assert_eq!(
            stable_uint64_for_text("doc-alpha:88d4741101bd:0000"),
            9_230_259_498_505_691_355
        );
    }

    #[test]
    fn repairs_zero_and_in_batch_collisions() {
        assert_eq!(
            repair_stable_uint64_candidates(&[0, 1, 1, u64::MAX, u64::MAX]),
            vec![1, 2, 3, u64::MAX, 4]
        );
    }

    #[test]
    fn stable_ids_preserve_input_order() {
        let chunk_ids = vec![
            "doc-alpha:88d4741101bd:0000".to_string(),
            "doc-beta:fbccf9cc33a3:0000".to_string(),
        ];
        assert_eq!(
            stable_uint64_ids_for_chunk_ids(&chunk_ids),
            vec![9_230_259_498_505_691_355, 4_357_769_671_793_707_343]
        );
    }
}
