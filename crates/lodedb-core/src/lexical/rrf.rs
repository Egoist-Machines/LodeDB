//! Reciprocal Rank Fusion matching the Python lexical oracle.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{CoreError, CoreErrorCode};

/// Reciprocal Rank Fusion smoothing constant.
pub const RRF_C: f64 = 60.0;

/// Fuses ranked id lists with Reciprocal Rank Fusion.
pub fn reciprocal_rank_fusion(
    rankings: &[Vec<String>],
    c: f64,
    weights: Option<&[f64]>,
) -> Result<Vec<(String, f64)>, CoreError> {
    if weights.is_some_and(|weights| weights.len() != rankings.len()) {
        return Err(CoreError::new(
            CoreErrorCode::InvalidArgument,
            "weights must align with rankings",
        ));
    }
    let mut fused: BTreeMap<String, f64> = BTreeMap::new();
    for (ranker_index, ranking) in rankings.iter().enumerate() {
        let weight = weights
            .and_then(|values| values.get(ranker_index))
            .copied()
            .unwrap_or(1.0);
        let mut seen = BTreeSet::new();
        for (position, raw_id) in ranking.iter().enumerate() {
            if !seen.insert(raw_id) {
                continue;
            }
            let rank = position as f64 + 1.0;
            *fused.entry(raw_id.clone()).or_insert(0.0) += weight / (c + rank);
        }
    }
    let mut rows = fused.into_iter().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    Ok(rows)
}

/// Returns only the fused unit-id order for the two-ranker hybrid case.
pub fn fuse_unit_rankings(
    vector_unit_ids: &[String],
    lexical_unit_ids: &[String],
    c: f64,
) -> Vec<String> {
    reciprocal_rank_fusion(
        &[vector_unit_ids.to_vec(), lexical_unit_ids.to_vec()],
        c,
        None,
    )
    .unwrap_or_default()
    .into_iter()
    .map(|(unit_id, _score)| unit_id)
    .collect()
}

#[cfg(test)]
mod tests {
    use super::{fuse_unit_rankings, reciprocal_rank_fusion, RRF_C};

    #[test]
    fn fuses_with_stable_tie_breaks() {
        let fused = reciprocal_rank_fusion(
            &[
                vec!["A".to_string(), "B".to_string(), "C".to_string()],
                vec!["B".to_string(), "C".to_string(), "D".to_string()],
            ],
            RRF_C,
            None,
        )
        .unwrap();
        assert_eq!(
            fused
                .iter()
                .map(|(unit_id, _)| unit_id.as_str())
                .collect::<Vec<_>>(),
            ["B", "C", "A", "D"]
        );
        assert_eq!(
            fuse_unit_rankings(
                &["A".to_string(), "B".to_string()],
                &["B".to_string(), "C".to_string()],
                RRF_C,
            ),
            ["B".to_string(), "A".to_string(), "C".to_string()]
        );
    }
}
