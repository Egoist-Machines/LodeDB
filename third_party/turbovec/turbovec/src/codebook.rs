//! Lloyd-Max scalar quantizer for the Beta distribution.
//!
//! After orthogonal rotation, each coordinate of a unit vector on S^{d-1}
//! follows Beta((d-1)/2, (d-1)/2) on [-1, 1]. This module computes optimal
//! quantization boundaries and centroids for that distribution.

use statrs::distribution::{Beta, ContinuousCDF, Continuous};

/// Returns (boundaries, centroids) for the given bit width and dimension.
pub fn codebook(bits: usize, dim: usize) -> (Vec<f32>, Vec<f32>) {
    lloyd_max(bits, dim, 200, 1e-12)
}

/// Returns only the centroids, which is all the search and reconstruct paths
/// need. The Lloyd-Max iteration costs about 50 ms per (bits, dim) on an
/// Apple M-series core, paid on the first search of every process, so common
/// embedding dims read the result from [`PRECOMPUTED_CENTROIDS`] instead.
/// Other dims compute it as before.
pub fn centroids(bits: usize, dim: usize) -> Vec<f32> {
    PRECOMPUTED_CENTROIDS
        .iter()
        .find(|(b, d, _)| *b == bits && *d == dim)
        .map(|(_, _, table)| table.iter().map(|&value| f32::from_bits(value)).collect())
        .unwrap_or_else(|| codebook(bits, dim).1)
}

/// `codebook(bits, dim).1` as f32 bit patterns, for common embedding dims.
/// The unit test below recomputes every entry and requires the same bits.
const PRECOMPUTED_CENTROIDS: &[(usize, usize, &[u32])] = &[
    (2, 128, &[0xbe083c09, 0xbd23ce3a, 0x3d23ce3a, 0x3e083c09]),
    (
        3,
        128,
        &[
            0xbe40eb31, 0xbdf1f343, 0xbd885e09, 0xbcb0fb86, 0x3cb0fb86, 0x3d885e09, 0x3df1f343,
            0x3e40eb31,
        ],
    ),
    (
        4,
        128,
        &[
            0xbe735deb, 0xbe392ccc, 0xbe11351c, 0xbde1de5a, 0xbda9a184, 0xbd6ca1f3, 0xbd0be214,
            0xbc392d81, 0x3c392d81, 0x3d0be214, 0x3d6ca1f3, 0x3da9a184, 0x3de1de5a, 0x3e11351c,
            0x3e392ccc, 0x3e735deb,
        ],
    ),
    (2, 256, &[0xbdc0ffbd, 0xbce7bd7e, 0x3ce7bd7e, 0x3dc0ffbd]),
    (
        3,
        256,
        &[
            0xbe091119, 0xbdab8d39, 0xbd4131e1, 0xbc7aa250, 0x3c7aa250, 0x3d4131e1, 0x3dab8d39,
            0x3e091119,
        ],
    ),
    (
        4,
        256,
        &[
            0xbe2d79b2, 0xbe03ac5b, 0xbdce3a35, 0xbda040a8, 0xbd70902d, 0xbd27b9a5, 0xbcc64009,
            0xbc0334f4, 0x3c0334f4, 0x3cc64009, 0x3d27b9a5, 0x3d70902d, 0x3da040a8, 0x3dce3a35,
            0x3e03ac5b, 0x3e2d79b2,
        ],
    ),
    (2, 384, &[0xbd9dac8d, 0xbcbd3ce1, 0x3cbd3ce1, 0x3d9dac8d]),
    (
        3,
        384,
        &[
            0xbde02fad, 0xbd8c32e2, 0xbd1dd60d, 0xbc4cbc2e, 0x3c4cbc2e, 0x3d1dd60d, 0x3d8c32e2,
            0x3de02fad,
        ],
    ),
    (
        4,
        384,
        &[
            0xbe0e0613, 0xbdd76c9e, 0xbda89f6d, 0xbd82fe36, 0xbd4499fc, 0xbd090e67, 0xbca1fc8a,
            0xbbd667ac, 0x3bd667ac, 0x3ca1fc8a, 0x3d090e67, 0x3d4499fc, 0x3d82fe36, 0x3da89f6d,
            0x3dd76c9e, 0x3e0e0613,
        ],
    ),
    (2, 512, &[0xbd8896d1, 0xbca3e502, 0x3ca3e502, 0x3d8896d1]),
    (
        3,
        512,
        &[
            0xbdc24e50, 0xbd72f11b, 0xbd08bb00, 0xbc31589c, 0x3c31589c, 0x3d08bb00, 0x3d72f11b,
            0x3dc24e50,
        ],
    ),
    (
        4,
        512,
        &[
            0xbdf652f5, 0xbdbabcdd, 0xbd92229d, 0xbd6303ab, 0xbd2a574c, 0xbced7ba8, 0xbc8c55aa,
            0xbbb9be34, 0x3bb9be34, 0x3c8c55aa, 0x3ced7ba8, 0x3d2a574c, 0x3d6303ab, 0x3d92229d,
            0x3dbabcdd, 0x3df652f5,
        ],
    ),
    (2, 768, &[0xbd5f1d0d, 0xbc85d3cd, 0x3c85d3cd, 0x3d5f1d0d]),
    (
        3,
        768,
        &[
            0xbd9ec6da, 0xbd46738f, 0xbcdf5888, 0xbc10d5e0, 0x3c10d5e0, 0x3cdf5888, 0x3d46738f,
            0x3d9ec6da,
        ],
    ),
    (
        4,
        768,
        &[
            0xbdc964ee, 0xbd989d1e, 0xbd6ecea6, 0xbd3975fa, 0xbd0b25c0, 0xbcc1fb3c, 0xbc653f5c,
            0xbb97b5b7, 0x3b97b5b7, 0x3c653f5c, 0x3cc1fb3c, 0x3d0b25c0, 0x3d3975fa, 0x3d6ecea6,
            0x3d989d1e, 0x3dc964ee,
        ],
    ),
    (2, 1024, &[0xbd413ff4, 0xbc67cd7c, 0x3c67cd7c, 0x3d413ff4]),
    (
        3,
        1024,
        &[
            0xbd898f43, 0xbd2be72e, 0xbcc173a3, 0xbbfae412, 0x3bfae412, 0x3cc173a3, 0x3d2be72e,
            0x3d898f43,
        ],
    ),
    (
        4,
        1024,
        &[
            0xbdae87c3, 0xbd843ab9, 0xbd4ee2f4, 0xbd20a8b8, 0xbcf11104, 0xbca806bb, 0xbc4691d6,
            0xbb836823, 0x3b836823, 0x3c4691d6, 0x3ca806bb, 0x3cf11104, 0x3d20a8b8, 0x3d4ee2f4,
            0x3d843ab9, 0x3dae87c3,
        ],
    ),
    (2, 1536, &[0xbd1dcf82, 0xbc3d4590, 0x3c3d4590, 0x3d1dcf82]),
    (
        3,
        1536,
        &[
            0xbd60b93f, 0xbd0c63e2, 0xbc9df9d9, 0xbbccdff6, 0x3bccdff6, 0x3c9df9d9, 0x3d0c63e2,
            0x3d60b93f,
        ],
    ),
    (
        4,
        1536,
        &[
            0xbd8e9985, 0xbd5807f1, 0xbd28fb6e, 0xbd0336ec, 0xbcc4e02d, 0xbc893862, 0xbc222914,
            0xbb569f67, 0x3b569f67, 0x3c222914, 0x3c893862, 0x3cc4e02d, 0x3d0336ec, 0x3d28fb6e,
            0x3d5807f1, 0x3d8e9985,
        ],
    ),
    (2, 3072, &[0xbcdf35c6, 0xbc05d6dd, 0x3c05d6dd, 0x3cdf35c6]),
    (
        3,
        3072,
        &[
            0xbd1ef792, 0xbcc6963c, 0xbc5f71d9, 0xbb90e286, 0x3b90e286, 0x3c5f71d9, 0x3cc6963c,
            0x3d1ef792,
        ],
    ),
    (
        4,
        3072,
        &[
            0xbd49cd81, 0xbd18d427, 0xbcef0fce, 0xbcb99e20, 0xbc8b3e96, 0xbc4218ef, 0xbbe55edc,
            0xbb17c96c, 0x3b17c96c, 0x3be55edc, 0x3c4218ef, 0x3c8b3e96, 0x3cb99e20, 0x3cef0fce,
            0x3d18d427, 0x3d49cd81,
        ],
    ),
];

fn lloyd_max(bits: usize, dim: usize, max_iter: usize, tol: f64) -> (Vec<f32>, Vec<f32>) {
    let a = (dim as f64 - 1.0) / 2.0;
    // Beta(a, a) on [0, 1], shifted to [-1, 1] via loc=-1, scale=2
    let beta = Beta::new(a, a).unwrap();

    let n_levels = 1usize << bits;

    // Initialize centroids within +/- 3 std devs
    let std_dev = (2.0 * a / ((2.0 * a + 1.0) * 4.0 * a)).sqrt(); // std of Beta on [-1,1]
    let spread = 3.0 * std_dev;
    let mut centroids: Vec<f64> = (0..n_levels)
        .map(|i| -spread + 2.0 * spread * i as f64 / (n_levels as f64 - 1.0))
        .collect();

    for _ in 0..max_iter {
        // Boundaries = midpoints between consecutive centroids
        let boundaries: Vec<f64> = (0..n_levels - 1)
            .map(|i| (centroids[i] + centroids[i + 1]) / 2.0)
            .collect();

        let mut edges = Vec::with_capacity(n_levels + 1);
        edges.push(-1.0);
        edges.extend_from_slice(&boundaries);
        edges.push(1.0);

        let mut new_centroids = vec![0.0f64; n_levels];

        for i in 0..n_levels {
            let lo = edges[i];
            let hi = edges[i + 1];

            // CDF on [-1, 1]: transform to [0, 1] for Beta
            let cdf_lo = beta.cdf((lo + 1.0) / 2.0);
            let cdf_hi = beta.cdf((hi + 1.0) / 2.0);
            let prob = cdf_hi - cdf_lo;

            if prob < 1e-15 {
                new_centroids[i] = centroids[i];
            } else {
                // Conditional mean = integral(x * pdf(x), lo, hi) / prob
                // where pdf is on [-1, 1]: pdf_shifted(x) = beta.pdf((x+1)/2) / 2
                let mean = adaptive_simpson(
                    |x| {
                        let t = (x + 1.0) / 2.0;
                        x * beta.pdf(t) / 2.0
                    },
                    lo,
                    hi,
                    1e-14,
                    50,
                );
                new_centroids[i] = mean / prob;
            }
        }

        let max_change = centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);

        centroids = new_centroids;

        if max_change < tol {
            break;
        }
    }

    let boundaries: Vec<f32> = (0..n_levels - 1)
        .map(|i| ((centroids[i] + centroids[i + 1]) / 2.0) as f32)
        .collect();
    let centroids_f32: Vec<f32> = centroids.iter().map(|&c| c as f32).collect();

    (boundaries, centroids_f32)
}

/// Adaptive Simpson's rule for numerical integration.
fn adaptive_simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64, max_depth: usize) -> f64 {
    let mid = (a + b) / 2.0;
    let fa = f(a);
    let fb = f(b);
    let fm = f(mid);
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
    adaptive_simpson_rec(&f, a, b, fa, fb, fm, whole, tol, max_depth)
}

fn adaptive_simpson_rec<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> f64 {
    let mid = (a + b) / 2.0;
    let m1 = (a + mid) / 2.0;
    let m2 = (mid + b) / 2.0;
    let fm1 = f(m1);
    let fm2 = f(m2);
    let left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
    let right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
    let refined = left + right;

    if depth == 0 || (refined - whole).abs() < 15.0 * tol {
        refined + (refined - whole) / 15.0
    } else {
        adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol / 2.0, depth - 1)
            + adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol / 2.0, depth - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::{centroids, codebook, PRECOMPUTED_CENTROIDS};

    #[test]
    fn precomputed_centroids_match_lloyd_max() {
        for &(bits, dim, table) in PRECOMPUTED_CENTROIDS {
            let computed: Vec<u32> = codebook(bits, dim).1.iter().map(|c| c.to_bits()).collect();
            assert_eq!(computed, table, "bits={bits} dim={dim}");
        }
    }

    #[test]
    fn dims_outside_the_table_still_compute() {
        assert!(PRECOMPUTED_CENTROIDS.iter().all(|(_, dim, _)| *dim != 200));
        let bits_of = |values: Vec<f32>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits_of(centroids(4, 200)), bits_of(codebook(4, 200).1));
    }
}
