//! Random orthogonal rotation matrix generation.
//!
//! Generates a deterministic orthogonal matrix via QR decomposition of
//! a seeded Gaussian random matrix. The rotation makes each coordinate
//! of a unit vector follow a known Beta distribution.
//!
//! Above 768 dimensions the QR split degree is pinned: the result depends
//! on the split degree, not the thread count. A fixed degree remains
//! bit-identical on any machine while still using the rayon pool. It keeps
//! the matrix previously produced on every machine with four or more cores;
//! machines with fewer than four cores previously computed a different
//! matrix at some dimensions above 768.

use faer::dyn_stack::{GlobalPodBuffer, PodStack};
use faer::linalg::householder;
use faer::linalg::qr::no_pivoting::compute as qr;
use faer::{Conj, Mat, Parallelism};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::ROTATION_SEED;

/// Largest dim whose QR runs single-threaded. Up to here the serial QR
/// equals the pooled one bit for bit (`tests/rotation.rs`) and takes a few
/// ms. Above it the pool is 2 to 3.5 times faster and uses the fixed split
/// degree [`POOLED_QR_DEGREE`] to remove dependence on the core count.
pub const SERIAL_QR_MAX_DIM: usize = 768;

/// Fixed QR split degree for dims above [`SERIAL_QR_MAX_DIM`]. Pinning this
/// degree removes dependence on the core count. Whether different CPU
/// architectures agree bit for bit is a separate, pre-existing question.
pub const POOLED_QR_DEGREE: usize = 8;

/// Generate a dim x dim orthogonal matrix (deterministic, seeded).
/// Returns row-major flat Vec<f32> of length dim*dim.
///
/// Up to [`SERIAL_QR_MAX_DIM`] the QR runs single-threaded. faer's default
/// would use the rayon global pool, which spawns one worker per core and
/// gives each a GEMM packing slab; for a one-off factorization at embedding
/// dims that costs more memory than the matrix itself. The blocked
/// Householder steps split GEMM work by output tile, never along the
/// reduction, so at these dims the result is bit-identical to the pooled QR.
/// Larger dims use [`POOLED_QR_DEGREE`] regardless of the global setting.
pub fn make_rotation_matrix(dim: usize) -> Vec<f32> {
    let parallelism = if dim <= SERIAL_QR_MAX_DIM {
        Parallelism::None
    } else {
        Parallelism::Rayon(POOLED_QR_DEGREE)
    };
    rotation_matrix(dim, parallelism)
}

fn rotation_matrix(dim: usize, parallelism: Parallelism) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(ROTATION_SEED);

    // Generate random Gaussian matrix
    let mut factors = Mat::<f64>::zeros(dim, dim);
    for j in 0..dim {
        for i in 0..dim {
            factors.write(i, j, rng.sample(StandardNormal));
        }
    }
    // Non-pivoted QR decomposition (deterministic), in place: `factors` ends
    // up holding R on and above the diagonal and the Householder basis below.
    let blocksize = qr::recommended_blocksize::<f64>(dim, dim);
    let mut householder = Mat::<f64>::zeros(blocksize, dim);
    let params = qr::QrComputeParams::default();
    qr::qr_in_place(
        factors.as_mut(),
        householder.as_mut(),
        parallelism,
        PodStack::new(&mut GlobalPodBuffer::new(
            qr::qr_in_place_req::<f64>(dim, dim, blocksize, parallelism, params)
                .expect("QR workspace size overflow"),
        )),
        params,
    );

    // Thin Q: apply the Householder sequence to the identity.
    let mut q = Mat::<f64>::zeros(dim, dim);
    q.as_mut().diagonal_mut().column_vector_mut().fill(1.0);
    householder::apply_block_householder_sequence_on_the_left_in_place_with_conj(
        factors.as_ref(),
        householder.as_ref(),
        Conj::No,
        q.as_mut(),
        parallelism,
        PodStack::new(&mut GlobalPodBuffer::new(
            householder::apply_block_householder_sequence_on_the_left_in_place_req::<f64>(
                dim, blocksize, dim,
            )
            .expect("Householder workspace size overflow"),
        )),
    );

    // Sign correction: Q = Q * diag(sign(diag(R)))
    for j in 0..dim {
        if !(factors.read(j, j) >= 0.0) {
            for i in 0..dim {
                q.write(i, j, -q.read(i, j));
            }
        }
    }

    row_major(&q)
}

fn row_major(q: &Mat<f64>) -> Vec<f32> {
    let dim = q.nrows();
    let mut result = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i * dim + j] = q.read(i, j) as f32;
        }
    }
    result
}
