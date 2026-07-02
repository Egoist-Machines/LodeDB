//! Small shared vector kernels used by both the exact-scan query rotation and the
//! ANN cluster index. Kept in one place so the two never drift (the ANN centroid
//! scoring must share the exact scan's coordinate space and metric).

/// Dot product of two slices, stopping at the shorter (the two call sites always
/// pass equal-length slices in the same dimension).
pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

/// Rotates `query` by a row-major `dim * dim` matrix: `out[o] = Σ q[i]·R[o·dim+i]`,
/// accumulating in f32 in ascending `i` so the result is bit-identical across
/// callers. Callers that accept untrusted input must check `rotation.len() ==
/// dim * dim` first (see the engine's `rotate_query`); this indexes
/// `rotation[o*dim + i]` for `i in 0..min(dim, query.len())`, so a short matrix
/// panics and a short query silently truncates.
pub(crate) fn rotate(query: &[f32], rotation: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for (o, slot) in out.iter_mut().enumerate() {
        let base = o * dim;
        let mut acc = 0.0f32;
        for (i, &value) in query.iter().enumerate().take(dim) {
            acc += value * rotation[base + i];
        }
        *slot = acc;
    }
    out
}
