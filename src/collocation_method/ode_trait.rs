use nalgebra::{DMatrix, DVector, RealField};
use num_traits::{FromPrimitive, Zero};
use std::ops::{Add, Mul, Sub};

use super::boundary_conditions::BoundaryCondition;

pub trait ODE<S, T> {
    fn eval(row: usize, x: S, u: &DVector<S>, d: &DMatrix<T>) -> S;
    fn forcing(xi: S) -> S;
    fn bc_left() -> BoundaryCondition<S>;
    fn bc_right() -> BoundaryCondition<S>;
}