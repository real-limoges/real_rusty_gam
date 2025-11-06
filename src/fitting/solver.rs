use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;
use crate::Term;
use crate::families::{Link, Distribution, Gaussian};
use argmin::core::Gradient;
use super::{GamlssError, ModelMatrix, PenaltyMatrix, Coefficients, CovarianceMatrix, LogLambdas};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Inverse, Solve, UPLO};
use argmin::core::{CostFunction, Error, Executor};
use polars::prelude::{DataFrame};

pub(crate) const MAX_PIRLS_ITER: usize = 25;
pub(crate) const PIRLS_TOLERANCE: f64 = 1e-6;

pub(crate) struct GamlssCost<'a, D: Distribution> {
    pub(crate) x_matrix: &'a ModelMatrix,
    pub(crate) y_vector: &'a Array1<f64>,
    pub(crate) penalty_matrices: &'a Vec<PenaltyMatrix>,
    pub(crate) distribution: &'a D,
}

impl<'a, D: Distribution> CostFunction for GamlssCost<'a, D> {
    type Param = LogLambdas;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let lambdas = param.mapv(f64::exp);

        let (beta, _, _, edf) = run_pirls(
            self.x_matrix,
            self.y_vector,
            self.distribution,
            self.penalty_matrices,
            &lambdas,
        ).map_err(Error::new)?;

        let n = self.y_vector.len() as f64;
        let eta = self.x_matrix.dot(&beta.0);
        let mu = eta.mapv(|e| self.distribution.link().inv_link(e));
        let variance = mu.mapv(|m| self.distribution.variance(m));

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

impl<'a, D: Distribution> Gradient for GamlssCost<'a, D> {
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

pub(crate) fn run_optimization<D>(
    x_model: &ModelMatrix,
    y: &Array1<f64>,
    penalty_matrices: &Vec<PenaltyMatrix>,
    distributional: &D,
) -> Result<Array1<f64>, GamlssError> {
    let cost_function = GamlssCost {
        x_matrix: &x_model,
        y_vector: y,
        penalty_matrices: &penalty_matrices,
        distribution: &D
    };

    let initial_log_lambdas = LogLambdas(Array1::<f64>::zeros(penalty_matrices.len()));
    let linesearch = MoreThuenteLineSearch::new();
    let m = 7;
    let solver = LBFGS::new(linesearch, m);

    let res = Executor::new(cost_function, solver)
        .configure(|state| state.param(initial_log_lambdas).max_iters(100))
        .run()?;

    let best_log_lambdas = res.state.best_param.unwrap();
    let best_lambdas = best_log_lambdas.mapv(f64::exp);

    Ok(best_lambdas)
}

pub(crate) fn run_pirls<D: Distribution>(
    x_matrix: &Array2<f64>,
    y_vector: &Array1<f64>,
    distribution: &D,
    penalty_matrices: &[PenaltyMatrix],
    lambdas: &Array1<f64>
) -> Result<(Coefficients, Array1<f64>, CovarianceMatrix, f64), GamlssError> {
    // lifted some pirls code out of a numerical analysis book

    let (n_obs, n_coeffs) = x_matrix.dim();

    let mut s_lambda = Array2::<f64>::zeros((n_coeffs, n_coeffs));
    for (i, s_j) in penalty_matrices.iter().enumerate() {
        s_lambda.scaled_add(lambdas[i], s_j);
    }

    let mut beta = Coefficients(Array1::<f64>::zeros(n_coeffs));
    let y_mean = y_vector.mean().unwrap_or(0.5).max(0.01);

    let mut eta = Array1::<f64>::from_elem(n_obs, distribution.link().link(y_mean));
    let mut w_diag = Array1::<f64>::zeros(n_obs);
    let mut z = Array1::<f64>::zeros(n_obs);

    // the actual PIRLS
    for _iter in 0..MAX_PIRLS_ITER {
        let mu = eta.mapv(|e| distribution.link().inv_link(e));
        for i in 0..n_obs {
            let (z_i, w_ii) = distribution.working_response_and_weights(y_vector[i], eta[i], mu[i]);
            z[i] = z_i;
            w_diag[i] = w_ii;
        }
        let w = Array2::<f64>::from_diag(&w_diag);

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
    Err(GamlssError::Convergence(MAX_PIRLS_ITER))
}