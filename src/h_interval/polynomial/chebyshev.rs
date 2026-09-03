use  nalgebra::{DMatrix, DVector, Scalar};
use num_traits::{FromPrimitive, Zero};
use core::f64;
use std::ops::{Add, Div, Mul, Neg, Sub, AddAssign};

use super::polynomial_trait::PolynomialTrait;

#[derive(Copy, Clone)]
pub struct Chebyshev {
    pub n: usize, // number of collocation points
}

impl Chebyshev {
    pub fn new(n: usize) -> Self {
        assert!(n>= 2, "Collocation points must be greater then 2");
        Self { n }
    }
}

impl<T> PolynomialTrait<T> for Chebyshev
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
    fn collocation_nodes(&self) -> DVector<f64> {
        let n = self.n;
        let mut nodes = DVector::<f64>::zeros(n);
        for j in 0..n {
            let theta = std::f64::consts::PI * (j as f64) / ((n - 1) as f64);
            nodes[j] = -theta.cos();
        }
        nodes
    }

    fn dmatrix(&self, c: DVector<f64>) -> DMatrix<T> {
        let n = c.len();
        let mut d = DMatrix::<T>::zeros(n, n);

        // Compute the coefficient weights c_i
        // c_0 = c_{N-1} = 2, all others are 1
        let mut weights = vec![T::from_f64(1.0).expect("Incorrect Type match: dmatrix-Chebyshev"); n];
        weights[0] = T::from_f64(2.0).expect("Incorrect Type match: dmatrix-Chebyshev");
        weights[n - 1] = T::from_f64(2.0).expect("Incorrect Type match: dmatrix-Chebyshev");

        // Compute off-diagonal entries
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let sign = T::from_f64(if (i + j) % 2 == 0 { 1.0 } else { -1.0 })
                        .expect("Incorrect Type match: dmatrix-Chebyshev");
                    d[(i, j)] = (weights[i].clone() / weights[j].clone()) * sign
                        / T::from_f64(c[i].clone() - c[j].clone()).unwrap();
                }
            }
        }

        // Compute diagonal entries
        // D_ii = -sum of off-diagonal entries in row i
        for i in 0..n {
            let mut sum = T::zero();
            for j in 0..n {
                if i != j {
                    sum += d[(i, j)].clone();
                }
            }
            d[(i, i)] = -sum;
        }

        d
    }

    fn scale_dmatrix(&self, d_ref: &DMatrix<T>, a: f64, b: f64) -> DMatrix<T> {
        let a_T = T::from_f64(a).unwrap();
        let b_T = T::from_f64(b).unwrap();
        let two = T::from_f64(2.0).expect("Incorrect type match: scale_dmatrix");
        let scale = two / (b_T - a_T);

        d_ref.map(|entry| entry * scale.clone())
    }
}

impl Chebyshev {
    pub fn clenshaw_curtis_weights(n: usize) -> DVector<f64> {
        let big_n = (n - 1) as f64; // N = n - 1
        let mut w = DVector::<f64>::zeros(n);

        for j in 0..n {
            // Use j' = N - j so this lines up with the -cos(theta) node ordering (ascending -1..1)
            let jp = (n - 1) - j;
            let theta = std::f64::consts::PI * (jp as f64) / big_n;

            let mut sum = 0.0;
            let kmax = ((n - 1) / 2) as i64; // floor(N/2)
            for k in 1..=kmax {
                let bk = if 2 * k == n as i64 - 1 { 1.0 } else { 2.0 };
                sum += bk / ((4 * k * k - 1) as f64) * (2.0 * k as f64 * theta).cos();
            }

            let c_end = if jp == 0 || jp == n - 1 { 1.0 } else { 2.0 };
            w[j] = (c_end / big_n) * (1.0 - sum);
        }

        w
    }
}