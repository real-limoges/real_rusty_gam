use std::arch::aarch64::float64x1_t;
use ndarray::{s, Array1, Array2};


pub fn create_basis_matrix(x: &Array1<f64>, n_splines: usize, degree: usize) -> Array2<f64> {
    let n_obs = x.len();
    let knots = select_knots(x, n_splines, degree);

    let mut basis_matrix = Array2::<f64>::zeros((n_obs, n_splines));

    for (i, &x_i) in x.iter().enumerate() {
        let basis_evals = eval_all_bases(&knots, n_splines, degree, x_i);

        for (j, &val)  in basis_evals.iter().enumerate() {
            if j < n_splines {
                basis_matrix[[i, j]] = val;
            }
        }
    }

    basis_matrix
}

fn eval_all_bases(knots: &[f64], n_splines: usize, degree: usize, x: f64) -> Vec<f64> {
    let n_knots = knots.len();
    let mut basis = vec![0.0; n_knots - 1];

    // go all the way up to the second to last knot
    for i in 0..(n_knots - 2) {
        if knots[i] <= x && x < knots[i + 1] {
            basis[i] = 1.0
        }
    }
    // annoying edge case with the end
    let last_i = n_knots - 2;
    if knots[last_i] <= x && x <= knots[last_i + 1] {
        basis[last_i] = 1.0
    }

    for p in 1..=degree {
        let num_funcs_current = degree - p - 1;
        let mut current_basis = vec![0.0; num_funcs_current];

        for i in 0..=num_funcs_current {
            let mut term1 = 0.0;
            let denominator1 = knots[i + p] - knots[i];
            if denominator1.abs() > 1e-9 {
                term1 = ((x - knots[i]) / denominator1) * basis[i];
            }

            let mut term2 = 0.0;
            let denominator2 = knots[i + p + 1] - knots[i + 1];
            if denominator2.abs() > 1e-9 {
                term2 = ((knots[i + p + 1] - x) / denominator2) * basis[i + 1];
            }
            current_basis[i] = term1 + term2;
        }
    }

    basis
}


fn select_knots(x: &Array1<f64>, n_splines: usize, degree: usize) -> Vec<f64> {
    //  selects the correct number of knots for smooth terms

    let min_val = x.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    let total_knots_needed = n_splines + degree + 1;
    let n_interior_knots = total_knots_needed.saturating_sub(2 * (degree + 1));
    let mut interior_knots = Vec::with_capacity(n_interior_knots);

    if n_interior_knots > 0 {
        let mut sorted_x = x.to_vec();
        sorted_x.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n_segments = (n_interior_knots) as f64;
        for i in 1..n_interior_knots {
            let quantile = (i as f64) / n_segments;
            let index = (quantile * (sorted_x.len() - 1) as f64).round() as usize;
            interior_knots.push(sorted_x[index]);
        }
    }

    let mut knots = Vec::with_capacity(total_knots_needed);

    knots.extend_from_slice(&vec![min_val; degree + 1]);
    knots.append(&mut interior_knots);
    knots.extend_from_slice(&vec![max_val; degree + 1]);

    knots
}

// P----- Add Penalty Matrices

pub fn create_penalty_matrix(n_splines: usize, order: usize) -> Array2<f64> {
    // I only implemented two types of penalties. First and Second Order.

    let n_rows_d = n_splines.saturating_sub(order);
    if n_rows_d == 0 {
        return Array2::<f64>::zeros((n_splines, n_splines));
    }

    let mut d_matrix = Array2::<f64>::zeros((n_rows_d, n_splines));
    match order {
        1 =>
            for i in 0..n_rows_d {
                d_matrix[[i,i]] = 1.0;
                d_matrix[[i, i+1]] = -1.0;
            }
        _ =>
            // this is the second order penalty. I gave it to everything that isn't 1
            for i in 0..n_rows_d {
                d_matrix[[i,i]] = 1.0;
                d_matrix[[i,i+1]] = -2.0;
                d_matrix[[i,i+2]] = 1.0;
            }
    }
    d_matrix.t().dot(&d_matrix)
}

pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    // A little bit of linear algebra magic
    let (m,n) = a.dim();
    let (p,q) = b.dim();
    let mut c = Array2::<f64>::zeros((m*p, p*q));

    for i in 0..m {
        for j in 0..n {
            let a_scalar = a[[i,j]];
            let mut block = c.slice_mut(s![i*p..(i+1)*p, j*q..(j+1)*q]);
            block.assign(&(b*a_scalar));
        }
    }

    c
}