use num_traits::{Zero, FromPrimitive};
use nalgebra::{DVector, DMatrix, Scalar};
use std::ops::{Add, Mul, Sub};

pub enum BoundaryCondition {
    Dirichlet(f64),
    Neumann(f64),
}
use crate::matrix_op::Operations;

impl BoundaryCondition {
    pub fn apply<T>(&self, u: &DVector<T>, d: &DMatrix<f64>, endpoint: Endpoint) -> T
    where
        T: Scalar + Copy + Zero + FromPrimitive + Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
    {
        let idx = match endpoint {
            Endpoint::Left  => 0,
            Endpoint::Right => u.len() - 1,
        };
        match self {
            BoundaryCondition::Dirichlet(val) => {
                u[idx] - T::from_f64(*val).expect("boundary value conversion failed")
            }
            BoundaryCondition::Neumann(val) => {
                let du = Operations::apply_matrix(d, u);
                du[idx] - T::from_f64(*val).expect("boundary value conversion failed")
            }
        }
    }
}


pub enum Endpoint {
    Left,
    Right,
}
