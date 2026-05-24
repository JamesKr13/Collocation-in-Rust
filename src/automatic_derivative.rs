use nalgebra::{DMatrix, DVector, RealField};
use num_traits::{FromPrimitive, Zero};
use std::ops::{Add, Mul, Sub};
use rayon::prelude::*;

use crate::dual::Dual;
use crate::multi_interval_domain::MultiIntervalDomain;
use crate::boundary_conditions::{BoundaryCondition, Endpoint};
use crate::matrix_op::Operations;

pub trait ODE<T> where T: Zero {
    fn eval(row: usize, x: T, u: &DVector<T>, d: &DMatrix<f64>) -> T
    where T: nalgebra::Scalar + Mul<Output=T> + Add<Output=T> + Copy;

    fn forcing(xi: T) -> T {
        T::zero()
    } 
    fn bc_right() -> BoundaryCondition;
    fn bc_left() -> BoundaryCondition;
}

fn residual<T, O: ODE<T>>(
    ode: &O,
    u: &DVector<T>,
    multi_interval_domain: &MultiIntervalDomain,
    non_homogeneous: Option<impl Fn(T) -> T>,
    bc_left: &BoundaryCondition,
    bc_right: &BoundaryCondition,
) -> DVector<T>
where
    T: nalgebra::Scalar + Mul<Output=T> + Add<Output=T> + Copy + Zero + Sub<Output=T> + FromPrimitive,
{
    let n_intervals = multi_interval_domain.interval.len();
    let n_per       = multi_interval_domain.interval[0].collocation_points().len();
    let n_total     = n_intervals * n_per;
    let mut F = DVector::from_vec(vec![T::zero(); n_total]);

    for (i, interval) in multi_interval_domain.interval.iter().enumerate() {
        let global_start = i * n_per;
        let d_loc   = interval.scaled_dmatrix();
        let u_local = local_u(u, i, n_per);
        let xi_local = interval.map_to_physical();
        for local_row in 1..=(n_per - 2) {
            let global_row = global_start + local_row;
            let x_i = T::from_f64(xi_local[local_row])
                .expect("collocation point conversion failed");
            let f_val = non_homogeneous.as_ref().map(|f| f(x_i)).unwrap_or(T::zero());
            F[global_row] = O::eval(local_row, x_i, &u_local, &d_loc) - f_val;
        }
    }

    for i in 0..n_intervals - 1 {
        let u_left  = local_u(u, i,     n_per);
        let u_right = local_u(u, i + 1, n_per);

        let du_left  = Operations::apply_matrix(
            &multi_interval_domain.interval[i].scaled_dmatrix(), &u_left);
        let du_right = Operations::apply_matrix(
            &multi_interval_domain.interval[i + 1].scaled_dmatrix(), &u_right);

        F[i * n_per + n_per - 1] = du_left[n_per - 1] - du_right[0];

        F[(i + 1) * n_per] = u_left[n_per - 1] - u_right[0];
    }

    F[0] = bc_left.apply(
        &local_u(u, 0, n_per),
        &multi_interval_domain.get_interval(0).scaled_dmatrix(),
        Endpoint::Left,
    );
    F[n_total - 1] = bc_right.apply(
        &local_u(u, n_intervals - 1, n_per),
        &multi_interval_domain.get_interval(n_intervals - 1).scaled_dmatrix(),
        Endpoint::Right,
    );
    F
}

fn local_u<T: nalgebra::Scalar + Copy>(u: &DVector<T>, i: usize, n_per: usize) -> DVector<T> {
    // println!("here 1");
    let start = i * n_per;
    DVector::from_iterator(n_per, u.rows(start, n_per).iter().cloned())
}

pub fn build_J_R<T, O: ODE<Dual<T>> + Sync>(
    ode: &O,
    u: &DVector<T>,
    multi_interval_domain: &MultiIntervalDomain,
    non_homogeneous: Option<impl Fn(Dual<T>) -> Dual<T> + Sync + Send>,
    bc_left: BoundaryCondition,
    bc_right: BoundaryCondition,
) -> (DMatrix<T>, DVector<T>)
where
    T: nalgebra::Scalar + Mul<Output=T> + Add<Output=T> + Copy + RealField + FromPrimitive + Send + Sync
{
    let n = u.len();

    // Residual pass (no perturbation)
    let u_real = DVector::from_fn(n, |i, _| Dual { real: u[i], eps: T::zero() });
    let f_r = residual::<Dual<T>, O>(ode, &u_real, multi_interval_domain,
                                      non_homogeneous.as_ref().map(|f| f),
                                      &bc_left, &bc_right);
    let R: DVector<T> = DVector::from_fn(n, |i, _| f_r[i].real);

    // Jacobian: compute all columns in parallel
    let cols: Vec<DVector<T>> = (0..n).into_par_iter().map(|k| {
        let u_seed = DVector::from_fn(n, |i, _| Dual {
            real: u[i],
            eps: if i == k { T::one() } else { T::zero() },
        });
        let f_dual = residual::<Dual<T>, O>(ode, &u_seed, multi_interval_domain,
                                             non_homogeneous.as_ref().map(|f| f),
                                             &bc_left, &bc_right);
        DVector::from_fn(n, |i, _| f_dual[i].eps)
    }).collect();

    let mut J = DMatrix::zeros(n, n);
    for (k, col) in cols.into_iter().enumerate() {
        J.set_column(k, &col);
    }
    (J, R)
}


