use num_traits::{Zero, FromPrimitive};
use nalgebra::{DVector, DMatrix, Scalar};
use std::ops::{Add, AddAssign, Mul, Sub};

use crate::toolbox;
use toolbox::dual_matrix_operations::Operations;

pub enum BoundaryCondition<T> {
    Dirichlet(T),
    Neumann(T),
    Robin { alpha: T, beta: T, value: T },
    RobinQuadratic { alpha: T, beta: T, gamma: T, value: T },
    CharacteristicEq { sum: T, product: T, value: T },
}

pub enum Endpoint {
    Left,
    Right,
}

impl<T> BoundaryCondition<T>
where
    T: Scalar + Copy + Zero + FromPrimitive + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + AddAssign,
{
    pub fn apply(&self, u: &DVector<T>, d: &DMatrix<T>, endpoint: Endpoint) -> T {
        let idx = match endpoint {
            Endpoint::Left => 0,
            Endpoint::Right => u.len() - 1,
        };

        match self {
            BoundaryCondition::Dirichlet(val) => u[idx] - *val,

            BoundaryCondition::Neumann(val) => {
                let du = Operations::apply_matrix(d, u);
                du[idx] - *val
            }

            BoundaryCondition::CharacteristicEq { sum, product, value } => {
                let du  = Operations::apply_matrix(d, u);
                let d2u = Operations::apply_matrix(d, &du);

                d2u[idx] - *sum * du[idx] + *product * (u[idx] - *value)
            }

            BoundaryCondition::Robin { alpha, beta, value } => {
                let du = Operations::apply_matrix(d, u);
                *alpha * u[idx] + *beta * du[idx] - *value
            }

            BoundaryCondition::RobinQuadratic { alpha, beta, gamma, value } => {
                let du = Operations::apply_matrix(d, u);
                let u_sq = u[idx] * u[idx];
                *alpha * u[idx] + *beta * du[idx] + *gamma * u_sq - *value
            }
        }
    }
    
}
