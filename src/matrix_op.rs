
pub mod Operations {
    use num_traits::{Zero, FromPrimitive};
    use nalgebra::{DVector, DMatrix, Scalar};
    use std::ops::{Add, Mul};

    pub fn apply_matrix<T>(d: &DMatrix<f64>, u: &DVector<T>) -> DVector<T>
    where
        T: Scalar + Copy + Zero + FromPrimitive + Mul<Output = T> + Add<Output = T>,
    {
        DVector::from_iterator(
            d.nrows(),
            d.row_iter().map(|row| {
                row.iter()
                    .zip(u.iter())
                    .fold(T::zero(), |acc, (&a, &b)| {
                        acc + b * T::from_f64(a).expect("matrix element conversion failed")
                    })
            }),
        )
    }
}