//! Correctness tests for the rotation matrix generator.
//!
//! The rotation is a deterministic orthogonal matrix derived from a
//! fixed crate-wide seed via QR decomposition. These tests verify the
//! mathematical invariants the downstream quantizer depends on:
//!
//! - The matrix is orthogonal: `R^T R = I`.
//! - Rotation preserves vector L2 norms.
//! - Generation is deterministic across calls.
//! - `R^T` recovers the input (round-trip via the transpose).
//!
//! Tolerances here are generous enough for f32 + faer QR roundoff.

use turbovec::rotation::make_rotation_matrix;

fn mat_mul(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            let mut acc = 0.0f32;
            for k in 0..dim {
                acc += a[i * dim + k] * b[k * dim + j];
            }
            out[i * dim + j] = acc;
        }
    }
    out
}

fn transpose(m: &[f32], dim: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            t[j * dim + i] = m[i * dim + j];
        }
    }
    t
}

fn mat_vec(m: &[f32], v: &[f32], dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dim];
    for i in 0..dim {
        let mut acc = 0.0f32;
        for j in 0..dim {
            acc += m[i * dim + j] * v[j];
        }
        out[i] = acc;
    }
    out
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn rand_vec(dim: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut out = Vec::with_capacity(dim);
    for _ in 0..dim {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (((state >> 32) as u32) & 0x007FFFFF) | 0x3F800000;
        let uniform = f32::from_bits(bits) - 1.0;
        out.push(uniform * 2.0 - 1.0);
    }
    out
}

#[test]
fn orthogonal_across_dims() {
    for &dim in &[32usize, 64, 128, 256] {
        let r = make_rotation_matrix(dim);
        let rt = transpose(&r, dim);
        let product = mat_mul(&rt, &r, dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                let got = product[i * dim + j];
                assert!(
                    (got - expected).abs() < 1e-3,
                    "R^T R[{}][{}] = {}, expected {} (dim={})",
                    i,
                    j,
                    got,
                    expected,
                    dim
                );
            }
        }
    }
}

#[test]
fn preserves_norm() {
    for &dim in &[32usize, 64, 128] {
        let r = make_rotation_matrix(dim);
        for seed in 0..5u64 {
            let x = rand_vec(dim, seed);
            let y = mat_vec(&r, &x, dim);
            let nx = l2_norm(&x);
            let ny = l2_norm(&y);
            assert!(
                (nx - ny).abs() / nx < 1e-3,
                "norm changed: |Rx|={} vs |x|={} (dim={}, seed={})",
                ny,
                nx,
                dim,
                seed
            );
        }
    }
}

#[test]
fn deterministic_for_same_dim() {
    let r1 = make_rotation_matrix(128);
    let r2 = make_rotation_matrix(128);
    assert_eq!(r1.len(), r2.len());
    for (i, (a, b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "rotation[{}] differs across calls: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn inverse_round_trip_via_transpose() {
    let dim = 128;
    let r = make_rotation_matrix(dim);
    let rt = transpose(&r, dim);
    for seed in 0..5u64 {
        let x = rand_vec(dim, seed);
        let rx = mat_vec(&r, &x, dim);
        let x_hat = mat_vec(&rt, &rx, dim);
        for (j, (orig, back)) in x.iter().zip(x_hat.iter()).enumerate() {
            assert!(
                (orig - back).abs() < 1e-3,
                "R^T R x [{}] = {} vs original {} (seed={})",
                j,
                back,
                orig,
                seed
            );
        }
    }
}

#[test]
fn size_matches_dim_squared() {
    for &dim in &[16usize, 64, 256] {
        let r = make_rotation_matrix(dim);
        assert_eq!(r.len(), dim * dim);
    }
}

/// The pre-serial generator: faer's high-level QR on the rayon pool. Kept
/// here as the reference the single-threaded build must match bit for bit,
/// since every persisted code was encoded against it.
fn reference_rotation_matrix(dim: usize) -> Vec<f32> {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;
    use rand_distr::StandardNormal;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut g = faer::Mat::<f64>::zeros(dim, dim);
    for j in 0..dim {
        for i in 0..dim {
            g.write(i, j, rng.sample(StandardNormal));
        }
    }
    let qr = g.qr();
    let mut q = qr.compute_thin_q();
    let r = qr.compute_thin_r();
    for j in 0..dim {
        if r.read(j, j) < 0.0 {
            for i in 0..dim {
                q.write(i, j, -q.read(i, j));
            }
        }
    }
    let mut out = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            out[i * dim + j] = q.read(i, j) as f32;
        }
    }
    out
}

#[test]
fn serial_qr_matches_the_parallel_reference_bit_for_bit() {
    faer::set_global_parallelism(faer::Parallelism::Rayon(4));
    // 768 is the last serial dim; 1024 takes the pooled path and must equal
    // the reference under the same pool.
    assert_eq!(turbovec::rotation::SERIAL_QR_MAX_DIM, 768);
    for &dim in &[8usize, 16, 24, 48, 64, 128, 200, 256, 384, 512, 768, 1024] {
        let serial = make_rotation_matrix(dim);
        let reference = reference_rotation_matrix(dim);
        let differing = serial
            .iter()
            .zip(&reference)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(differing, 0, "dim={dim}: {differing} entries differ");
    }
}
