#[derive(Debug, Clone)]
pub enum Term {
    Intercept,
    // non-penalized linear term
    Linear {
        col_index: usize,
    },
    // smooth portion for one term
    Smooth(Smooth),
}

#[derive(Debug, Clone)]
pub enum Smooth {
    PSpline1D {
        col_index: usize,
        n_splines: usize,
        degree: usize,
        penalty_order: usize,
    },
    TensorProduct {
        col_index_1: usize,
        n_splines_1: usize,
        penalty_order_1: usize,

        col_index_2: usize,
        n_splines_2: usize,
        penalty_order_2: usize,

        // I'm just forcing them to have the same degree
        degree: usize,
    },
    RandomEffect {
        col_index: usize,
    },
}