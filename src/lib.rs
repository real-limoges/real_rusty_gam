mod error;
mod fitting;
mod splines;
mod types;
mod terms;

pub use error::GamError;
pub mod families;
pub use terms::Term;
pub use types::*;

use families::Family;

#[derive(Debug)]
pub struct GeneralizedAdditiveModel {
    pub coefficients: Coefficients,
    pub covariance: CovarianceMatrix,
    terms: Vec<Term>,
}

impl GeneralizedAdditiveModel {
    pub fn fit<F: Family> (
        x: &Matrix,
        y: &Vector,
        terms: &[Term],
        family: &F,
    ) -> Result<Self, GamError> {
        todo!()
    }
}