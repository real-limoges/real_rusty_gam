//! Linear algebra backend abstraction layer.
//!
//! This module provides a unified interface for linear algebra operations,
//! supporting multiple backends:
//! - `openblas`: Uses ndarray-linalg with OpenBLAS (default, highest performance)
//! - `pure-rust`: Uses faer for pure Rust implementation (WASM-compatible)
//!
//! The backend is selected at compile time via feature flags.

use crate::GamlssError;
use ndarray::{Array1, Array2};

/// Result type for linear algebra operations.
pub type Result<T> = std::result::Result<T, GamlssError>;

// =============================================================================
// OpenBLAS Backend (default)
// =============================================================================

#[cfg(feature = "openblas")]
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    use ndarray_linalg::Solve;
    Ok(a.solve(b)?)
}

#[cfg(feature = "openblas")]
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray_linalg::Inverse;
    Ok(a.inv()?)
}

#[cfg(feature = "openblas")]
pub fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray_linalg::{Cholesky, UPLO};
    Ok(a.cholesky(UPLO::Lower)?)
}

/// Solves the lower triangular system L * x = b for x.
#[cfg(feature = "openblas")]
pub fn solve_triangular_lower(l: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    use ndarray_linalg::triangular::{Diag, SolveTriangular};
    use ndarray_linalg::UPLO;
    Ok(l.solve_triangular(UPLO::Lower, Diag::NonUnit, b)?)
}

/// Solves the upper triangular system U * x = b for x.
#[cfg(feature = "openblas")]
pub fn solve_triangular_upper(u: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    use ndarray_linalg::triangular::{Diag, SolveTriangular};
    use ndarray_linalg::UPLO;
    Ok(u.solve_triangular(UPLO::Upper, Diag::NonUnit, b)?)
}

/// Computes the inverse of a lower triangular matrix L.
///
/// The inverse is computed by solving L * X = I for X.
#[cfg(feature = "openblas")]
pub fn inv_lower_triangular(l: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray_linalg::triangular::{Diag, SolveTriangular};
    use ndarray_linalg::UPLO;
    let n = l.nrows();
    let identity = Array2::<f64>::eye(n);
    Ok(l.solve_triangular(UPLO::Lower, Diag::NonUnit, &identity)?)
}

// =============================================================================
// Pure Rust Backend (faer) - WASM compatible
// =============================================================================

#[cfg(feature = "pure-rust")]
pub fn solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    use faer::prelude::*;

    // Convert ndarray -> faer
    let a_faer = ndarray_to_faer_mat(a);
    let b_faer = ndarray_to_faer_col(b);

    // Solve using LU decomposition with partial pivoting
    let plu = a_faer.partial_piv_lu();
    let x_faer = plu.solve(&b_faer);

    // Convert back to ndarray
    faer_col_to_ndarray(&x_faer)
}

#[cfg(feature = "pure-rust")]
pub fn inv(a: &Array2<f64>) -> Result<Array2<f64>> {
    use faer::prelude::*;

    // Convert ndarray -> faer
    let a_faer = ndarray_to_faer_mat(a);

    // Compute inverse using LU decomposition
    let plu = a_faer.partial_piv_lu();
    let inv_faer = plu.inverse();

    // Convert back to ndarray
    faer_mat_to_ndarray(&inv_faer)
}

#[cfg(feature = "pure-rust")]
pub fn cholesky_lower(a: &Array2<f64>) -> Result<Array2<f64>> {
    // Convert ndarray -> faer
    let a_faer = ndarray_to_faer_mat(a);

    // Compute Cholesky decomposition
    let chol = a_faer.cholesky(faer::Side::Lower).map_err(|_| {
        GamlssError::Linalg(
            "Cholesky decomposition failed (matrix not positive definite)".to_string(),
        )
    })?;

    // The Cholesky object itself represents L, we can use it directly
    // Extract the lower triangular matrix
    let l_faer = chol.compute_l();

    // Convert back to ndarray
    faer_mat_to_ndarray(&l_faer)
}

/// Solves the lower triangular system L * x = b for x.
#[cfg(feature = "pure-rust")]
pub fn solve_triangular_lower(l: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let l_faer = ndarray_to_faer_mat(l);
    let mut x_faer = ndarray_to_faer_col(b);
    faer::linalg::triangular_solve::solve_lower_triangular_in_place(
        l_faer.as_ref(),
        x_faer.as_mut().as_2d_mut(),
        faer::Parallelism::None,
    );
    faer_col_to_ndarray(&x_faer)
}

/// Solves the upper triangular system U * x = b for x.
#[cfg(feature = "pure-rust")]
pub fn solve_triangular_upper(u: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>> {
    let u_faer = ndarray_to_faer_mat(u);
    let mut x_faer = ndarray_to_faer_col(b);
    faer::linalg::triangular_solve::solve_upper_triangular_in_place(
        u_faer.as_ref(),
        x_faer.as_mut().as_2d_mut(),
        faer::Parallelism::None,
    );
    faer_col_to_ndarray(&x_faer)
}

/// Computes the inverse of a lower triangular matrix L.
///
/// The inverse is computed by solving L * X = I for X.
#[cfg(feature = "pure-rust")]
pub fn inv_lower_triangular(l: &Array2<f64>) -> Result<Array2<f64>> {
    let l_faer = ndarray_to_faer_mat(l);
    let n = l_faer.nrows();
    let mut dst = faer::Mat::<f64>::zeros(n, n);
    faer::linalg::triangular_inverse::invert_lower_triangular(
        dst.as_mut(),
        l_faer.as_ref(),
        faer::Parallelism::None,
    );
    faer_mat_to_ndarray(&dst)
}

// =============================================================================
// Conversion Helpers: ndarray <-> faer
// =============================================================================

#[cfg(feature = "pure-rust")]
fn ndarray_to_faer_mat(arr: &Array2<f64>) -> faer::Mat<f64> {
    use faer::Mat;

    let (nrows, ncols) = arr.dim();
    Mat::from_fn(nrows, ncols, |i, j| arr[[i, j]])
}

#[cfg(feature = "pure-rust")]
fn ndarray_to_faer_col(arr: &Array1<f64>) -> faer::Col<f64> {
    use faer::Col;

    let n = arr.len();
    Col::from_fn(n, |i| arr[i])
}

#[cfg(feature = "pure-rust")]
fn faer_mat_to_ndarray(mat: &faer::Mat<f64>) -> Result<Array2<f64>> {
    let (nrows, ncols) = (mat.nrows(), mat.ncols());
    let result = Array2::from_shape_fn((nrows, ncols), |(i, j)| mat.read(i, j));
    Ok(result)
}

#[cfg(feature = "pure-rust")]
fn faer_col_to_ndarray(col: &faer::Col<f64>) -> Result<Array1<f64>> {
    let n = col.nrows();
    let result = Array1::from_shape_fn(n, |i| col.read(i));
    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_solve() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let b = array![8.0, 7.0];

        let x = solve(&a, &b).unwrap();

        // Check Ax ≈ b
        let ax = a.dot(&x);
        assert!((ax[0] - b[0]).abs() < 1e-10);
        assert!((ax[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_inv() {
        let a = array![[4.0, 2.0], [2.0, 3.0]];

        let a_inv = inv(&a).unwrap();

        // Check A * A^-1 ≈ I
        let identity = a.dot(&a_inv);
        assert!((identity[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((identity[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(identity[[0, 1]].abs() < 1e-10);
        assert!(identity[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_cholesky() {
        // Positive definite matrix
        let a = array![[4.0, 2.0], [2.0, 3.0]];

        let l = cholesky_lower(&a).unwrap();

        // Check L * L^T ≈ A
        let lt = l.t().to_owned();
        let reconstructed = l.dot(&lt);
        assert!((reconstructed[[0, 0]] - a[[0, 0]]).abs() < 1e-10);
        assert!((reconstructed[[0, 1]] - a[[0, 1]]).abs() < 1e-10);
        assert!((reconstructed[[1, 0]] - a[[1, 0]]).abs() < 1e-10);
        assert!((reconstructed[[1, 1]] - a[[1, 1]]).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_not_positive_definite() {
        // Not positive definite (negative eigenvalue)
        let a = array![[1.0, 2.0], [2.0, 1.0]];

        let result = cholesky_lower(&a);
        assert!(result.is_err());
    }

    #[test]
    fn test_solve_triangular_lower() {
        // L = [[2, 0], [1, 3]], b = [4, 7]
        // L*x = b => x = [2, 5/3]
        let l = array![[2.0, 0.0], [1.0, 3.0]];
        let b = array![4.0, 7.0];

        let x = solve_triangular_lower(&l, &b).unwrap();
        let lx = l.dot(&x);
        assert!((lx[0] - b[0]).abs() < 1e-10);
        assert!((lx[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_solve_triangular_upper() {
        let u = array![[2.0, 1.0], [0.0, 3.0]];
        let b = array![5.0, 6.0];

        let x = solve_triangular_upper(&u, &b).unwrap();
        let ux = u.dot(&x);
        assert!((ux[0] - b[0]).abs() < 1e-10);
        assert!((ux[1] - b[1]).abs() < 1e-10);
    }

    #[test]
    fn test_inv_lower_triangular() {
        let l = array![[2.0, 0.0], [1.0, 3.0]];

        let l_inv = inv_lower_triangular(&l).unwrap();

        // L * L^{-1} should be identity
        let product = l.dot(&l_inv);
        assert!((product[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((product[[1, 1]] - 1.0).abs() < 1e-10);
        assert!(product[[0, 1]].abs() < 1e-10);
        assert!(product[[1, 0]].abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_triangular_roundtrip() {
        // Verify that Cholesky + triangular solve gives same result as direct solve
        let a = array![[4.0, 2.0], [2.0, 3.0]];
        let b = array![8.0, 7.0];

        // Direct solve
        let x_direct = solve(&a, &b).unwrap();

        // Cholesky-based solve: A = L L^T, so L y = b, L^T x = y
        let l = cholesky_lower(&a).unwrap();
        let y = solve_triangular_lower(&l, &b).unwrap();
        let lt = l.t().to_owned();
        let x_chol = solve_triangular_upper(&lt, &y).unwrap();

        assert!((x_direct[0] - x_chol[0]).abs() < 1e-10);
        assert!((x_direct[1] - x_chol[1]).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_covariance_via_triangular_inv() {
        // Verify V = L^{-T} L^{-1} equals A^{-1}
        let a = array![[4.0, 2.0], [2.0, 3.0]];

        let v_direct = inv(&a).unwrap();

        let l = cholesky_lower(&a).unwrap();
        let l_inv = inv_lower_triangular(&l).unwrap();
        let v_chol = l_inv.t().dot(&l_inv);

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (v_direct[[i, j]] - v_chol[[i, j]]).abs() < 1e-10,
                    "Mismatch at [{}, {}]: {} vs {}",
                    i,
                    j,
                    v_direct[[i, j]],
                    v_chol[[i, j]]
                );
            }
        }
    }
}
