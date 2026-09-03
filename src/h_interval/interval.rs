use nalgebra::{DMatrix, DVector, Matrix, Vector2, Scalar};
use num_traits::{FromPrimitive, Zero, ToPrimitive};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use super::polynomial::{chebyshev, polynomial_trait};

use chebyshev::Chebyshev;
use polynomial_trait::{PolynomialKind, PolynomialTrait};


pub struct Interval<T>
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
        + Sync
        + 'static,
{
    pub polynomial: Box<dyn PolynomialTrait<T>>,
    pub domain: (f64, f64),
    pub local_collocation_nodes: DVector<f64>,
    pub n_nodes: usize,
}

impl<T> Interval<T>
where
    T: FromPrimitive
        + ToPrimitive
        + Zero
        + Scalar
        + Clone
        + Sync
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + Neg<Output = T>
        + 'static,
{
    pub fn new(kind: PolynomialKind, degree: usize, domain: (f64, f64)) -> Self {
        let poly: Box<dyn PolynomialTrait<T>> = kind.create::<T>(degree);
        let local_collocation_nodes = poly.collocation_nodes();
        let n_nodes: usize = local_collocation_nodes.len();
        Interval {
            polynomial: kind.create::<T>(degree),
            domain,
            local_collocation_nodes,
            n_nodes

        }
    }

    pub fn collocation_nodes(&self) -> &DVector<f64> {
        &self.local_collocation_nodes
    }

    pub fn dmatrix(&self, c: DVector<f64>) -> DMatrix<T> {
        self.polynomial.dmatrix(c)
    }

    pub fn get_local_collocation_point(&self, n: usize) -> &f64 {
        &self.local_collocation_nodes[n]
    }

    pub fn scale_dmatrix(&self) -> DMatrix<T> {
        let nodes = self.polynomial.collocation_nodes();
        let d_ref = self.polynomial.dmatrix(nodes);
        let (a, b) = self.domain.clone();
        self.polynomial.scale_dmatrix(&d_ref, a, b)
    }
    pub fn map_to_physical(&self) -> DVector<f64> {
        self.local_collocation_nodes.map(|x|{
            self.domain.0 + 0.5 *(self.domain.1- self.domain.0)*(x + 1.0)
        })
    }

    pub fn get_total_nodes(&self) -> usize {
        return self.n_nodes;
    }
}
