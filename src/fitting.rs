#![allow(unused_imports, unused_variables)]

use rand::rand_core;
use rand::rng;
use argmin::core::{CostFunction, Error, Executor, Gradient};
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use super::error::GamError;
use super::families;
use super::families::{Family, Link};
use super::splines::{create_basis_matrix, create_penalty_matrix, kronecker_product};
use super::terms::{Term, Smooth};
use super::types::*;
use ndarray::{s, concatenate, Axis, Array1, Array2, ShapeError};
use ndarray_linalg::{Cholesky, Inverse, Lapack, Solve, UPLO};
use rand_distr::{Distribution, StandardNormal};
use rand::Rng;
use polars::datatypes::DataType;
use polars::prelude::*;
use std::sync::Arc;
use std::iter;

// some constants around the PIRLS algorithm
const MAX_PIRLS_ITER: usize = 25;
const PIRLS_TOLERANCE: f64 = 1e-6;


// Here's the good stuff

pub(crate) fn fit_model<F: Family + 'static>(
    data: &DataFrame,
    y: &Vector,
    terms: &[Term],
    family: &F,
) -> Result<(Coefficients, CovarianceMatrix), GamError> {
    let n_obs = y.len();

    let mut model_matrix_parts = Vec::new();
    let mut penalty_blocks = Vec::new();
    let mut total_coeffs = 0;

    for term in terms.iter() {
        match term {
            Term::Intercept => {
                let part = Array1::ones(n_obs).into_shape((n_obs, 1))
                    .map_err(|err| GamError::ComputationError(err.to_string()))?;
                model_matrix_parts.push(part);
                total_coeffs += 1;
            }
            Term::Linear { col_name } => {
                let x_col_vec = get_col_as_f64(data, col_name, n_obs)?;
                let part = x_col_vec.into_shape((n_obs, 1))
                    .map_err(|err| GamError::ComputationError(err.to_string()))?;
                model_matrix_parts.push(part);
                total_coeffs += 1;
            }
            Term::Smooth(smooth) => {
                let (basis, penalties) = assemble_smooth(data, n_obs, smooth)?;
                let n_coeffs = basis.ncols();
                model_matrix_parts.push(basis);

                for penalty_block in penalties {
                    penalty_blocks.push((total_coeffs, penalty_block));
                }
                total_coeffs += n_coeffs;
            }
        }
    }

    let x_model = ModelMatrix(concatenate(
        Axis(1),
        &model_matrix_parts
            .iter()
            .map(|m| m.view())
            .collect::<Vec<_>>(),
    )?);

    let penalty_matrices = penalty_blocks
        .into_iter()
        .map(|(start_index, block)| {
            let mut s_j = PenaltyMatrix(Matrix::zeros((total_coeffs, total_coeffs)));
            let n = block.ncols();
            s_j.slice_mut(s![start_index..start_index + n, start_index..start_index + n])
                .assign(&block);
            s_j
        })
        .collect::<Vec<_>>();

    let cost_function = GamCost {
        x_matrix: &x_model,
        y_vector: y,
        penalty_matrices: &penalty_matrices,
        family
    };

    let initial_log_lambdas = LogLambdas(Vector::zeros(penalty_matrices.len()));
    let linesearch = MoreThuenteLineSearch::new();
    let m = 7;
    let solver = LBFGS::new(linesearch, m);

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.param(initial_log_lambdas).max_iters(100))
        .run()?;

    let best_log_lambdas = res.state.best_param.unwrap();
    let best_lambdas = best_log_lambdas.mapv(f64::exp);

    let (beta, _, v_beta_unscaled, edf) = run_pirls(
        &x_model,
        y,
        family,
        &penalty_matrices,
        &best_lambdas,
    )?;

    let phi = if std::any::TypeId::of::<F>() == std::any::TypeId::of::<families::Gaussian>() {
        let n = y.len() as f64;
        let eta = x_model.dot(&beta.0);
        let mu = eta.mapv(|e| family.link().inv_link(e));
        let variance = mu.mapv(|m| family.variance(m));

        let residuals = y - &mu;
        let pearson_residuals = &residuals / variance.mapv(f64::sqrt);
        let res = pearson_residuals.mapv(|r| r.powi(2)).sum();

        (res / (n - edf)).max(1e-10)
    } else {
        1.0
    };
    let v_beta = CovarianceMatrix(v_beta_unscaled.0 * phi);

    Ok((beta, v_beta))
}

fn get_col_as_f64(data: &DataFrame, name: &str, n_obs: usize) -> Result<Vector, GamError> {
    let series = data.column(name)
        .map_err(|e| GamError::Input(format!("Column '{}' not found: {}", name, e)))?;

    let f64_series = if series.dtype() != &DataType::Float64 {
        series.cast(&DataType::Float64)?
    } else {
        series.clone()
    };
    // todo: open a PR about the private ndarray::error::ShapeError issue
    let f64_chunked_array = f64_series.f64()?;

    let ndarray_data = f64_chunked_array.to_ndarray()?;
    let arr = ndarray_data
        .to_shape(n_obs)
        .map_err(|e| GamError::Shape(e.to_string()))?;

    Ok(Vector::from(arr.to_vec()))

}

fn assemble_smooth(data: &DataFrame, n_obs: usize, smooth: &Smooth
) -> Result<(Matrix, Vec<PenaltyMatrix>), GamError> {
    // each smooth has its own arm of the match

    match smooth {
        Smooth::PSpline1D {
            col_name, n_splines, degree, penalty_order
        } => {
            // super straight forward flow
            let x_col = get_col_as_f64(data, col_name, n_obs)?;
            let basis = create_basis_matrix(&x_col, *n_splines, *degree);
            let penalty = create_penalty_matrix(*n_splines, *penalty_order);

            Ok((basis, vec![PenaltyMatrix(penalty)]))
        }

        Smooth::TensorProduct {
            col_name_1, n_splines_1, penalty_order_1,
            col_name_2, n_splines_2, penalty_order_2,
            degree
        } => {

            //  First set up both sidees of the product
            let x1 = get_col_as_f64(data, col_name_1, n_obs)?;
            let b1 = create_basis_matrix(&x1, *n_splines_1, *degree);
            let s1 = create_penalty_matrix(*n_splines_1, *penalty_order_1);

            let x2 = get_col_as_f64(data, col_name_2, n_obs)?;
            let b2 = create_basis_matrix(&x2, *n_splines_2, *degree);
            let s2 = create_penalty_matrix(*n_splines_2, *penalty_order_2);

            let n_coeffs_total = *n_splines_1 * *n_splines_2;

            let mut basis = Matrix::zeros((n_obs, n_coeffs_total));

            // send the basis vectors into the blender
            for i in 0..n_obs {
                let row1 = b1.row(i);
                let row2 = b2.row(i);
                let row_out = kronecker_product(
                    &row1.insert_axis(Axis(0)).to_owned(),
                    &row2.insert_axis(Axis(0)).to_owned(),
                );
                basis.row_mut(i).assign(&row_out);
            }

            // this pushes them out the penalties into matrices
            let i_k1 = Matrix::eye(*n_splines_1);
            let i_k2 = Matrix::eye(*n_splines_2);

            let penalty_1 = kronecker_product(&s1, &i_k2);
            let penalty_2 = kronecker_product(&i_k1, &s2);
            Ok((basis, vec![PenaltyMatrix(penalty_1), PenaltyMatrix(penalty_2)]))
        },
        Smooth::RandomEffect { col_name } => {
            let series = data.column(col_name)?;
            let cat_series = series.categorical()?;
            let id_codes = cat_series.physical();

            let n_groups = id_codes.n_unique()?;
            let mut basis = Matrix::zeros((n_obs, n_groups));

            let id_col_ndarray = id_codes.to_ndarray()?
                .into_shape_with_order(n_obs)
                .map_err(|err| {
                    GamError::ComputationError(err.to_string())
                })?;

            for i in 0..n_obs {
                let group_id = id_col_ndarray[i] as usize;
                if group_id < n_groups {
                    basis[[i, group_id]] = 1.0;
                }
            }
            let penalty = Matrix::eye(n_groups);

            Ok((basis, vec![PenaltyMatrix(penalty)]))
        }
    }
}

struct GamCost<'a, F: Family> {
    x_matrix: &'a ModelMatrix,
    y_vector: &'a Vector,
    penalty_matrices: &'a Vec<PenaltyMatrix>,
    family: &'a F,
}
impl<'a, F: Family> CostFunction for GamCost<'a, F> {
    type Param = LogLambdas;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let lambdas = param.mapv(f64::exp);

        let (beta, _, _, edf) = run_pirls(
            self.x_matrix,
            self.y_vector,
            self.family,
            self.penalty_matrices,
            &lambdas,
        ).map_err(|e| Error::new(e))?;

        let n = self.y_vector.len() as f64;
        let eta = self.x_matrix.dot(&beta.0);
        let mu = eta.mapv(|e| self.family.link().inv_link(e));
        let variance = mu.mapv(|m| self.family.variance(m));

        let residuals = self.y_vector - &mu;
        let pearson_residuals = &residuals / variance.mapv(f64::sqrt);
        let rss = pearson_residuals.mapv(|r| r.powi(2)).sum();

        let denominator = (n - edf).powi(2);
        if denominator.abs() < 1e-10 {
            return Ok(f64::MAX);
        }
        let gcv_score = (n * rss) / denominator;

        Ok(gcv_score)
    }
}

impl<'a, F: Family> Gradient for GamCost<'a, F> {
    type Param = LogLambdas;
    type Gradient = LogLambdas;
    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        const H: f64 = 1.4901161193847656e-8; // This seemingly random choice is (f64::EPSILON.sqrt())
        let n = param.0.len();
        let mut grad_vec = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut param_plus_h_vec = param.0.clone();
            param_plus_h_vec += H;

            let mut param_minus_h_vec = param.0.clone();
            param_minus_h_vec -= H;

            let cost_plus = self.cost(&LogLambdas(param_plus_h_vec))?;
            let cost_minus = self.cost(&LogLambdas(param_minus_h_vec))?;

            grad_vec[i] = (cost_plus - cost_minus) / (2.0 * H);
        }
        Ok(LogLambdas(grad_vec))
    }
}
//

fn run_pirls<F: Family>(
    x_matrix: &Matrix,
    y_vector: &Vector,
    family: &F,
    penalty_matrices: &Vec<PenaltyMatrix>,
    lambdas: &Vector
) -> Result<(Coefficients, Vector, CovarianceMatrix, f64), GamError> {
    // lifted some pirls code out of a numerical analysis book

    let (n_obs, n_coeffs) = x_matrix.dim();

    let mut s_lambda = Matrix::zeros((n_coeffs, n_coeffs));
    for (i, s_j) in penalty_matrices.iter().enumerate() {
        s_lambda.scaled_add(lambdas[i], s_j);
    }

    let mut beta = Coefficients(Array1::zeros(n_coeffs));
    let y_mean = y_vector.mean().unwrap_or(0.5).max(0.01);

    let mut eta = Array1::from_elem(n_obs, family.link().link(y_mean));
    let mut w_diag = Array1::zeros(n_obs);
    let mut z = Array1::zeros(n_obs);

    // the actual PIRLS
    for _iter in 0..MAX_PIRLS_ITER {
        let mu = eta.mapv(|e| family.link().inv_link(e));
        for i in 0..n_obs {
            let (z_i, w_ii) = family.working_response_and_weights(y_vector[i], eta[i], mu[i]);
            z[i] = z_i;
            w_diag[i] = w_ii;
        }
        let w = Array2::from_diag(&w_diag);

        let x_t_w = x_matrix.t().dot(&w);
        let lhs = x_t_w.dot(x_matrix) + &s_lambda;
        let rhs = x_t_w.dot(&z);

        let new_beta = Coefficients(lhs.solve(&rhs)?);

        let diff = (&new_beta.0 - &beta.0).mapv(f64::abs).sum();

        if diff < PIRLS_TOLERANCE {
            let v_beta_unscaled = CovarianceMatrix(lhs.inv()?);
            let x_t_w_x_final = x_t_w.dot(x_matrix);
            let edf = v_beta_unscaled.dot(&x_t_w_x_final).diag().sum();

            return Ok((new_beta, w_diag, v_beta_unscaled, edf));
        }
        beta = new_beta;
        eta = x_matrix.dot(&beta.0);
    }
    Err(GamError::Convergence(MAX_PIRLS_ITER))
}

pub(crate) fn sample_posterior(
    beta_hat: &Coefficients,
    v_beta: &CovarianceMatrix,
    n_samples: usize,
) -> Vec<Vector> {

    let l_factor = match &v_beta.0.cholesky(UPLO::Lower) {
        Ok(cholesky) => cholesky,
        Err(_) => return vec![],
    };

    let mut rng_rs = rng();

    sample_from_cholesky(&beta_hat.0, &v_beta.0, n_samples, &mut rng_rs)
}


pub(crate) fn sample_from_cholesky(
    mean: &Array1<f64>,
    l_factor: &Array2<f64>,
    n_samples: usize,
    rng: &mut (impl Rng + rand_core::RngCore + rand_core::RngCore + rand_core::RngCore)
) -> Vec<Array1<f64>> {

    let dim = mean.len();

    (0..n_samples)
        .map(|_| {
            let z = Array1::from_shape_fn(dim, |_| StandardNormal.sample(rng));
            mean + l_factor.dot(&z)
        })
        .collect()
}