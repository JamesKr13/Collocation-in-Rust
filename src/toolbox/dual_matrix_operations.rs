
use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::Zero;
use std::ops::{AddAssign, Mul};

pub struct Operations;

impl Operations {
    /// Multiply a real-valued matrix (`T`) against a working-scalar
    /// vector (`S`), e.g. D (f64/T) times u (Dual<T>/S).
    pub fn apply_matrix<S, T>(d: &DMatrix<T>, u: &DVector<S>) -> DVector<S>
    where
        T: Scalar + Copy,
        S: Scalar + Copy + Zero + AddAssign + Mul<Output = S> + From<T>,
    {
        let (rows, cols) = d.shape();
        debug_assert_eq!(cols, u.len(), "apply_matrix: dimension mismatch");

        DVector::from_fn(rows, |i, _| {
            let mut acc = S::zero();
            for j in 0..cols {
                acc += S::from(d[(i, j)]) * u[j];
            }
            acc
        })
    }
}