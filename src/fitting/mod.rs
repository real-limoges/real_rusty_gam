use std::collections::HashMap;
use crate::families::{Distribution, Link};
mod assembler;
mod solver;
pub(crate) mod inference;

use assembler::assemble_model_matrices;
use solver::{run_optimization, run_pirls};
use inference::sample_posterior;

use super::error::GamlssError;
use super::terms::{Term, Smooth};
use super::types::*;
use ndarray::{Array1, Array2, Axis};
use polars::prelude::DataFrame;
use crate::families;


#[derive(Debug)]
pub struct ModelParameter{
    beta: Coefficients,
    x_matrix: ModelMatrix,
    penalty_matrices: Vec<PenaltyMatrix>,
    lambda: Array1<f64>,
    eta: Array1<f64>,
}
pub(crate) fn fit_gamlss<D: Distribution>(
    data: &DataFrame,
    y: &Array1<f64>,
    formula: &HashMap<String, Vec<Term>>,
    family: &D
) -> Result<HashMap<String, ModelParameter>, GamlssError> {
    let mut models: HashMap<String, ModelParameter> = HashMap::new();

    for param_name in family.parameters() {
        let terms = formula.get(param_name).unwrap();
        let (x_model, penalty_matrices, _) = assemble_model_matrices(data, y.len(), terms)?;
    }

    let n_obs = y.len();

    let (x_model, penalty_matrices, total_coeffs) = assemble_model_matrices(data, n_obs, terms)?;

    let best_lambdas = run_optimization(
        &x_model,
        y,
        &penalty_matrices,
        family
    )?;

    let (beta, _, v_beta_unscaled, edf) = run_pirls(
        &x_model,
        y,
        family,
        &penalty_matrices,
        &best_lambdas
    );

    let phi = if std::any::TypeId::of::<D>() == std::any::TypeId::of::<families::Gaussian>() {
        let n = y.len() as f64;
        let eta = x_model.dot(&beta.0);
        let mu = eta.mapv(|e| family.link().inv_link(e));
        let variance = mu.mapv(|m| family.variance(m));

        let residuals = y - &mu;
        let pearson_residuals = &residuals / variance.mapv(f64::sqrt);
        let rss = pearson_residuals.mapv(|r| r.powi(2)).sum();

        (rss / (n - edf)).max(1e-10)
    } else {
        1.0
    };

    let v_beta = CovarianceMatrix(v_beta_unscaled.0 * phi);

    Ok((beta, v_beta))
}
