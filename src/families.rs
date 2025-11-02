use std::fmt::Debug;
use std::future::join;

// ----- These traits help make sure the actual distributions are implemented correctly
pub trait Link: Debug + Send + Sync {
    fn link(&self, mu:f64) -> f64;
    fn inv_link(&self, eta:f64) -> f64;
    fn d_link(&self, mu:f64) -> f64;
}

pub trait Family: Debug + Sync + Send {
    type Link: Link;

    fn link(&self) -> &Self::Link;
    fn variance(&self, mu:f64) -> f64;
    fn working_response_and_weights(&self, y: f64, eta: f64, mu: f64) -> (f64, f64) {
        let d_link = self.link().d_link(mu);
        let variance = self.variance(mu);
        let weight = 1.0 / (d_link.powi(2) * variance);
        let working_response = eta + (y - mu) * d_link;

        (working_response, weight)
    }
}

// ----- These are the actual families
#[derive(Debug, Clone, Copy, Default)]
pub struct LogLink;

impl Link for LogLink {
    fn link(&self, mu:f64) -> f64 {
        mu.ln().max(-30.0)
    }
    fn inv_link(&self, eta:f64) -> f64 {
        eta.min(30.0).exp()
    }
    fn d_link(&self, mu:f64) -> f64 {
        1.0 / mu.max(1e-10)  // this will make sure no div/0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Poisson {
    link: LogLink,
}

impl Poisson {
    pub fn new() -> Self {
        Self { link: LogLink }
    }
}

impl Family for Poisson {
    type Link = LogLink;

    fn link(&self) -> &Self::Link {
        &self.link
    }
    fn variance(&self, mu:f64) -> f64 {
        mu
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityLink;

impl Link for IdentityLink {
    fn link(&self, mu:f64) -> f64 {
        mu
    }
    fn inv_link(&self, eta:f64) -> f64 {
        eta
    }
    fn d_link(&self, mu: f64) -> f64 {
        1.0
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Gaussian {
    link: IdentityLink,
}

impl Gaussian {
    pub fn new() -> Self {
        Self { link: IdentityLink }
    }
}

impl Family for Gaussian {
    type Link = IdentityLink;

    fn link(&self) -> &Self::Link {
        &self.link
    }
    fn variance(&self, mu:f64) -> f64 {
        todo!()
    }
}