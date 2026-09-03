use nalgebra::{DMatrix, DVector, Scalar};
use std::ops::{Add, Div, Mul, Neg, Sub};
extern crate lapack_src;
extern crate openblas_src;
use lapack::{dgbsv,sgbsv};
// extern "C" {
//     fn dgbsv(n: i32, kl: i32, ku: i32, nrhs: i32, ab: &mut [f64], ldab: i32,
//               ipiv: &mut [i32], b: &mut [f64], ldb: i32, info: &mut i32);
//     fn sgbsv(n: i32, kl: i32, ku: i32, nrhs: i32, ab: &mut [f32], ldab: i32,
//               ipiv: &mut [i32], b: &mut [f32], ldb: i32, info: &mut i32);
// }

pub trait BandedLinSolve:
    Scalar + Copy + PartialOrd + num_traits::Zero
    + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self>
    + Div<Output = Self> + Neg<Output = Self>
{
    fn banded_solve(a: &DMatrix<Self>, b: &DVector<Self>, kl: usize, ku: usize) -> DVector<Self> {
        banded_solve_generic(a, b, kl, ku)
    }
}

pub fn banded_solve_generic<T>(a_in: &DMatrix<T>, b_in: &DVector<T>, kl: usize, _ku: usize) -> DVector<T>
where
    T: BandedLinSolve,
{
    let n = b_in.len();
    assert_eq!(a_in.nrows(), n);
    assert_eq!(a_in.ncols(), n);

    let mut a = a_in.clone();
    let mut b = b_in.clone();
    let abs_t = |x: T| if x < T::zero() { -x } else { x };

    for k in 0..n {
        let mut piv_row = k;
        let mut piv_val = abs_t(a[(k, k)]);
        let row_limit = (k + kl).min(n - 1);
        for i in (k + 1)..=row_limit {
            let v = abs_t(a[(i, k)]);
            if v > piv_val { piv_row = i; piv_val = v; }
        }
        assert!(piv_val != T::zero(), "banded_solve: singular matrix at column {k}");
        if piv_row != k {
            for j in 0..n { a.swap((k, j), (piv_row, j)); }
            b.swap_rows(k, piv_row);
        }
        let pivot = a[(k, k)];
        for i in (k + 1)..=row_limit {
            let factor = a[(i, k)] / pivot;
            if factor == T::zero() { continue; }
            for j in k..n {
                let delta = factor * a[(k, j)];
                a[(i, j)] = a[(i, j)] - delta;
            }
            let delta_b = factor * b[k];
            b[i] = b[i] - delta_b;
        }
    }

    let mut x = DVector::from_element(n, T::zero());
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n { sum = sum - a[(i, j)] * x[j]; }
        x[i] = sum / a[(i, i)];
    }
    x
}

impl BandedLinSolve for f64 {
    fn banded_solve(a: &DMatrix<f64>, b: &DVector<f64>, kl: usize, ku: usize) -> DVector<f64> {
        let n = b.len();
        let ldab = 2 * kl + ku + 1;
        let mut ab = vec![0.0f64; ldab * n];
        for j in 0..n {
            for i in (j.saturating_sub(ku))..=(j + kl).min(n - 1) {
                let row = kl + ku + i - j;
                ab[row + j * ldab] = a[(i, j)];
            }
        }
        let mut ipiv = vec![0i32; n];
        let mut bv = b.as_slice().to_vec();
        let mut info = 0i32;
        unsafe {
            dgbsv(n as i32, kl as i32, ku as i32, 1, &mut ab, ldab as i32, &mut ipiv, &mut bv, n as i32, &mut info);
        }
        assert_eq!(info, 0, "dgbsv failed with info={info}");
        DVector::from_vec(bv)
    }
}

impl BandedLinSolve for f32 {
    fn banded_solve(a: &DMatrix<f32>, b: &DVector<f32>, kl: usize, ku: usize) -> DVector<f32> {
        let n = b.len();
        let ldab = 2 * kl + ku + 1;
        let mut ab = vec![0.0f32; ldab * n];
        for j in 0..n {
            for i in (j.saturating_sub(ku))..=(j + kl).min(n - 1) {
                let row = kl + ku + i - j;
                ab[row + j * ldab] = a[(i, j)];
            }
        }
        let mut ipiv = vec![0i32; n];
        let mut bv = b.as_slice().to_vec();
        let mut info = 0i32;
        unsafe {
            sgbsv(n as i32, kl as i32, ku as i32, 1, &mut ab, ldab as i32, &mut ipiv, &mut bv, n as i32, &mut info);
        }
        assert_eq!(info, 0, "sgbsv failed with info={info}");
        DVector::from_vec(bv)
    }
}