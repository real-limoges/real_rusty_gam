#[derive(Debug, Clone)]
pub enum Term {
    Intercept,

    // non-penalized linear term
    Linear {
        col_index: usize,
    },

    // smooth portion for one term
    Smooth {
        col_index: usize,
        n_splines: usize,
        degree: usize,
        penalty_order: usize,
    },
}