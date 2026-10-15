//! Random orthogonal rotation matrix generation.
//!
//! Generates a deterministic orthogonal matrix via QR decomposition of
//! a seeded Gaussian random matrix. The rotation makes each coordinate
//! of a unit vector follow a known Beta distribution.

use faer::dyn_stack::{GlobalPodBuffer, PodStack};
use faer::linalg::householder;
use faer::linalg::qr::no_pivoting::compute as qr;
use faer::{Conj, Mat, Parallelism};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::ROTATION_SEED;

/// Generate a dim x dim orthogonal matrix (deterministic, seeded).
/// Returns row-major flat Vec<f32> of length dim*dim.
///
/// The QR runs single-threaded on purpose. faer's default would use the
/// rayon global pool, which spawns one worker per core and gives each a
/// GEMM packing slab; for a one-off dim*dim factorization that costs more
/// memory than the matrix itself and is no faster at common embedding dims.
/// The blocked Householder steps split GEMM work by output tile, never along
/// the reduction, so the result is bit-identical to the parallel QR
/// (`tests/rotation.rs` pins this against `faer::Mat::qr`).
pub fn make_rotation_matrix(dim: usize) -> Vec<f32> {
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
        Parallelism::None,
        PodStack::new(&mut GlobalPodBuffer::new(
            qr::qr_in_place_req::<f64>(dim, dim, blocksize, Parallelism::None, params)
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
        Parallelism::None,
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

    // Convert to row-major f32
    let mut result = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i * dim + j] = q.read(i, j) as f32;
        }
    }

    result
}
