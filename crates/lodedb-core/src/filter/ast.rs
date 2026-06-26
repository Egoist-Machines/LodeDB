//! Filter grammar constants.

pub(crate) const MAX_DEPTH: usize = 32;

pub(crate) fn is_logical_operator(key: &str) -> bool {
    matches!(key, "$and" | "$or" | "$not")
}

pub(crate) fn is_comparison_operator(key: &str) -> bool {
    matches!(
        key,
        "$eq" | "$ne" | "$gt" | "$gte" | "$lt" | "$lte" | "$in" | "$nin" | "$exists"
    )
}

pub(crate) fn is_ordered_operator(key: &str) -> bool {
    matches!(key, "$gt" | "$gte" | "$lt" | "$lte")
}
