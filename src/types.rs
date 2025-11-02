use ndarray::{Array1, Array2};
use std::ops::{Deref, DerefMut};

// ----- Mnemonics because the ndarray names stink
pub type Vector = Array1<f64>;
pub type Matrix = Array2<f64>;

// ----- Newtypes for Safety
#[derive(Debug, Clone)]
pub struct Coefficients(pub Vector);

#[derive(Debug, Clone)]
pub struct LogLambdas(pub Vector);

#[derive(Debug, Clone)]
pub struct ModelMatrix(pub Matrix);

#[derive(Debug, Clone)]
pub struct PenaltyMatrix(pub Matrix);

#[derive(Debug, Clone)]
pub struct CovarianceMatrix(pub Matrix);

// ----- Impls for Newtypes

impl Deref for Coefficients {
    type Target = Vector;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Coefficients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for LogLambdas {
    type Target = Vector;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for LogLambdas {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for ModelMatrix {
    type Target = Matrix;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ModelMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for PenaltyMatrix {
    type Target = Matrix;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for PenaltyMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Deref for CovarianceMatrix {
    type Target = Matrix;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for CovarianceMatrix {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}