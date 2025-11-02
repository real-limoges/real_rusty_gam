#![allow(unused_imports, unused_variables)]
use super::error::GamError;
use super::families::Family;
use super::splines::{create_basis_matrix, create_penalty_matrix, kronecker_product};
use super::terms::{Term, Smooth};
use super::types::*;
use ndarray::{s, concatenate, Axis};


use rand_distr::Distribution;

use polars::datatypes::DataType;
use polars::prelude::{ DataFrame, NamedFrom};

// some constants around the PIRLS algorithm
const MAX_PIRLS_ITER: usize = 25;
const PIRLS_TOLERANCE: f64 = 1e-6;




// Here's the good stuff

pub(crate) fn fit_model<F: Family>(
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
                let part = Vector::ones(n_obs).into_shape((n_obs, 1))?;
                model_matrix_parts.push(part);
                total_coeffs += 1;
            }
            Term::Linear { col_name } => {
                let x_col_vec = get_col_as_f64(data, col_name, n_obs)?;
                let part = x_col_vec.into_shape((n_obs, 1))?;
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
    todo!()
}

fn get_col_as_f64(data: &DataFrame, name: &str, n_obs: usize) -> Result<Vector, GamError> {
    let series = data.column(name)
        .map_err(|e| GamError::Input(format!("Column '{}' not found: {}", name, e)))?;

    let f64_series = if series.dtype() != &DataType::Float64 {
        series.cast(&DataType::Float64)?
    } else {
        series.clone()
    };
    let arr = f64_series.f64()?.to_ndarray()?.into_shape(n_obs)?;

    Ok(Vector::from_vec(arr.to_vec()))
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

            todo!()
        }
    }
}