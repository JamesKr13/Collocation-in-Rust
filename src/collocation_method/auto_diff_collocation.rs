use log::info;
use nalgebra::{DMatrix, DVector, RealField, Scalar};
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use crate::h_interval;
use h_interval::interval::Interval;
use h_interval::multi_interval::MultiIntervalDomain;

use crate::toolbox;
use toolbox::{auto_diff_dual::Dual, dual_matrix_operations::Operations};

use super::boundary_conditions::{BoundaryCondition, Endpoint};
use super::ode_trait::ODE;

pub fn residual<S, T, O>(
    _ode: &O,
    u: &DVector<S>,
    multi_interval_domain: &MultiIntervalDomain<T>,
    non_homogeneous: Option<impl Fn(S) -> S>,
    _bc_left: &BoundaryCondition<S>,
    _bc_right: &BoundaryCondition<S>,
    phase_condition: Option<&(dyn Fn(&DVector<S>) -> S + Sync)>,
) -> DVector<S>
where
    O: ODE<S, T>,
    T: FromPrimitive
        + ToPrimitive
        + Zero
        + Scalar
        + Clone
        + Copy
        + Sync
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + Neg<Output = T>
        + 'static,
    S: FromPrimitive
        + ToPrimitive
        + Zero
        + Scalar
        + Clone
        + Copy
        + Sync
        + Add<Output = S>
        + Sub<Output = S>
        + Mul<Output = S>
        + Div<Output = S>
        + AddAssign
        + Neg<Output = S>
        + From<T>
        + 'static,
{
    let n_total: usize = multi_interval_domain
        .intervals
        .iter()
        .map(|interval| interval.n_nodes)
        .sum();

    assert_eq!(n_total, u.len());

    let mut r = DVector::from_element(n_total, S::zero());
    let bc_rows = [0, n_total - 1];

    for (interval_idx, interval) in multi_interval_domain.intervals.iter().enumerate() {
        let g_start: usize = multi_interval_domain.intervals[..interval_idx]
            .iter()
            .map(|a| a.n_nodes)
            .sum();

        let d_loc: DMatrix<T> = interval.scale_dmatrix();
        let n_nodes = interval.n_nodes;
        let u_loc = u.rows(g_start, n_nodes).clone_owned();
        let xi_loc = interval.map_to_physical();

        for (loc_row, xi) in xi_loc.iter().enumerate() {
            let x_s = S::from(T::from_f64(*xi).unwrap());
            let forcing = non_homogeneous
                .as_ref()
                .map(|f| f(x_s))
                .unwrap_or_else(S::zero);

            r[g_start + loc_row] =
                O::eval(loc_row, x_s, &u_loc, &d_loc) - forcing;
        }
    }

    for i in 0..multi_interval_domain.intervals.len().saturating_sub(1) {
        let interval_left = &multi_interval_domain.intervals[i];
        let interval_right = &multi_interval_domain.intervals[i + 1];

        let g_start_left: usize = multi_interval_domain.intervals[..i]
            .iter()
            .map(|a| a.n_nodes)
            .sum();

        let g_start_right: usize = multi_interval_domain.intervals[..i + 1]
            .iter()
            .map(|a| a.n_nodes)
            .sum();

        let n_left = interval_left.n_nodes;
        let n_right = interval_right.n_nodes;

        let u_left = u.rows(g_start_left, n_left).clone_owned();
        let u_right = u.rows(g_start_right, n_right).clone_owned();

        let d_left_s = interval_left.scale_dmatrix().map(S::from);
        let d_right_s = interval_right.scale_dmatrix().map(S::from);

        let du_left = Operations::apply_matrix(&d_left_s, &u_left);
        let du_right = Operations::apply_matrix(&d_right_s, &u_right);

        let x_left = interval_left.map_to_physical();
        let x_right = interval_right.map_to_physical();

        let row_deriv = g_start_left + n_left - 1;
        let row_val = g_start_right;

        let i_left = if x_left[0] > x_left[n_left - 1] {
            0
        } else {
            n_left - 1
        };

        let i_right = if x_right[0] < x_right[n_right - 1] {
            0
        } else {
            n_right - 1
        };

        let row_deriv = g_start_left + i_left;
        let row_value = g_start_right + i_right;

        r[row_deriv] =
            du_left[i_left] - du_right[i_right];

        r[row_value] =
            u_left[i_left] - u_right[i_right];
    }

    let first_interval = &multi_interval_domain.intervals[0];
    let u_first = u.rows(0, first_interval.n_nodes).clone_owned();
    let d_first_s = first_interval.scale_dmatrix().map(S::from);

    let last_interval = multi_interval_domain.intervals.last().unwrap();

    let last_start: usize = multi_interval_domain.intervals
        [..multi_interval_domain.intervals.len() - 1]
        .iter()
        .map(|a| a.n_nodes)
        .sum();

    let u_last = u
        .rows(last_start, last_interval.n_nodes)
        .clone_owned();

    let d_last_s = last_interval.scale_dmatrix().map(S::from);

    r[0] = _bc_left.apply(&u_first, &d_first_s, Endpoint::Left);
    r[n_total - 1] = _bc_right.apply(&u_last, &d_last_s, Endpoint::Right);

    // if let Some(phase) = phase_condition {
    //     r[1] =phase(u);
    // } else {
    //     let (interval_idx, n_idx) = multi_interval_domain.node_near_zero();

    //     let g_start: usize = multi_interval_domain.intervals[..interval_idx]
    //         .iter()
    //         .map(|a| a.n_nodes)
    //         .sum();

    //     let interval = &multi_interval_domain.intervals[interval_idx];
    //     let u_loc = u.rows(g_start, interval.n_nodes).clone_owned();
    //     let d_s = interval.scale_dmatrix().map(S::from);

    //     let du = Operations::apply_matrix(&d_s, &u_loc);
    //     let d2u = Operations::apply_matrix(&d_s, &du);

    //     r[1] = d2u[n_idx];
    // }


    if let Some(phase) = phase_condition {
        r[1] = phase(u);
    } else {
        // Find the interval containing physical x = 0.
        let interval_idx = multi_interval_domain
            .intervals
            .iter()
            .position(|interval| {
                let (a, b) = interval.domain;
                a <= 0.0 && 0.0 <= b
            })
            .expect("The domain must contain physical x = 0");

        let interval = &multi_interval_domain.intervals[interval_idx];

        let offset: usize = multi_interval_domain.intervals[..interval_idx]
            .iter()
            .map(|interval| interval.n_nodes)
            .sum();

        let n = interval.n_nodes;
        let x = interval.map_to_physical();
        // Interpolation weights: weights[j] = l_j(0).
        let mut weights = vec![0.0_f64; n];
        let x_zero = x.iter().map(|&x| x.abs()).min_by(|a, b| a.total_cmp(b)).unwrap();
        // info!("{x_zero}-{n}");

        if let Some(j) = x.iter().position(|&xj| xj == 0.0) {
            // Zero is already a node.
            weights[j] = 1.0;
        } else {
            for j in 0..n {
                let mut weight = 1.0;

                for k in 0..n {
                    if k != j {
                        weight *= -x[k] / (x[j] - x[k]);
                    }
                }

                weights[j] = weight;
            }
        }

        // Nodal second derivatives, preserving Dual components.
        let u_loc = u.rows(offset, n).clone_owned();
        let d_s = interval.scale_dmatrix().map(S::from);

        let du = Operations::apply_matrix(&d_s, &u_loc);
        let d2u = Operations::apply_matrix(&d_s, &du);

        // Evaluate the second-derivative polynomial at physical zero.
        let mut phase_residual = S::zero();

        for j in 0..n {
            let weight = S::from(T::from_f64(weights[j]).unwrap());
            phase_residual += weight * d2u[j];
        }

        r[1] = phase_residual;
    }

    r
}

pub fn build_j_r<T, O>(
    ode: &O,
    u: &DVector<T>,
    multi_interval_domain: &MultiIntervalDomain<T>,
    non_homogeneous: Option<impl Fn(Dual<T>) -> Dual<T> + Sync + Send + Clone>,
    bc_left: &BoundaryCondition<Dual<T>>,
    bc_right: &BoundaryCondition<Dual<T>>,
    phase_condition: Option<
        &(dyn Fn(&DVector<Dual<T>>) -> Dual<T> + Sync)
    >,
) -> (DMatrix<T>, DVector<T>)
where
    O: ODE<Dual<T>, T> + Sync,
    T: Scalar
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + AddAssign
        + Neg<Output = T>
        + Copy
        + RealField
        + FromPrimitive
        + ToPrimitive
        + Zero
        + One
        + Send
        + Sync
        + 'static,
{
    let n = u.len();

    let u_zero_seed = DVector::from_fn(n, |i, _| Dual {
        real: u[i],
        eps: T::zero(),
    });

    let f_r = residual::<Dual<T>, T, O>(
        ode,
        &u_zero_seed,
        multi_interval_domain,
        non_homogeneous.clone(),
        bc_left,
        bc_right,
        phase_condition,
    );

    let r = DVector::from_fn(n, |i, _| f_r[i].real);

    let cols: Vec<DVector<T>> = (0..n)
        .into_par_iter()
        .map(|k| {
            let u_seed = DVector::from_fn(n, |i, _| Dual {
                real: u[i],
                eps: if i == k {
                    T::one()
                } else {
                    T::zero()
                },
            });

            let f_dual = residual::<Dual<T>, T, O>(
                ode,
                &u_seed,
                multi_interval_domain,
                non_homogeneous.clone(),
                bc_left,
                bc_right,
                phase_condition,
            );

            DVector::from_fn(n, |i, _| f_dual[i].eps)
        })
        .collect();

    let mut j = DMatrix::zeros(n, n);

    for (k, col) in cols.into_iter().enumerate() {
        j.set_column(k, &col);
    }

    (j, r)
}