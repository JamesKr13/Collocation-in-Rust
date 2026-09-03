use nalgebra::{DMatrix, DVector, Scalar};
use std::ops::{Add, Div, Mul, Neg, Sub, AddAssign};
use num_traits::{FromPrimitive, Zero};


use super::chebyshev::Chebyshev;

pub trait PolynomialTrait<T>: Sync + Send {
    fn collocation_nodes(&self) -> DVector<f64>;
    fn dmatrix(&self, c: DVector<f64>) -> DMatrix<T>;

    fn scale_dmatrix(&self, d_ref: &DMatrix<T>, a: f64, b: f64) -> DMatrix<T>;
}

#[derive(Clone, Copy)]
pub enum PolynomialKind {
    Chebyshev,
    Legendre,
    Monomial,
}

impl PolynomialKind {
    pub fn create<T>(&self, degree: usize) -> Box<dyn PolynomialTrait<T>>
    where
        T: FromPrimitive
            + Zero
            + Scalar
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Div<Output = T>
            + AddAssign
            + Neg<Output = T>
            + 'static,
    {
        match self {
            PolynomialKind::Chebyshev => Box::new(Chebyshev::new(degree)) as Box<dyn PolynomialTrait<T>>,
            PolynomialKind::Legendre => todo!(),
            PolynomialKind::Monomial => todo!(),
        }
    }
}