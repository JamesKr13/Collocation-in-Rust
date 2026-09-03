    use nalgebra::{DMatrix, DVector, RealField, Scalar};
    use num_traits::{FromPrimitive, One, Zero, ToPrimitive};
    use std::ops::{Add, Mul, Sub, Div, AddAssign, Neg};
    use rayon::prelude::*;

    use crate::h_interval;
    use h_interval::multi_interval::MultiIntervalDomain;
    use h_interval::interval::Interval;

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
    ) -> DVector<S>
    where
        O: ODE<S, T>,
        T: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
            + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
            + AddAssign + Neg<Output = T> + 'static,
        S: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
            + Add<Output = S> + Sub<Output = S> + Mul<Output = S> + Div<Output = S>
            + AddAssign + Neg<Output = S> + From<T> + 'static,
    {
        let n_total: usize = multi_interval_domain.intervals.iter()
            .map(|interval: &Interval<T>| interval.get_total_nodes())
            .sum();

        let mut r = DVector::from_vec(vec![S::zero(); n_total]);
        let bc_rows = [0, n_total - 1];

        for (interval_idx, _) in multi_interval_domain.intervals.iter().enumerate() {
            let g_start: usize = multi_interval_domain.intervals[..interval_idx]
                .iter()
                .map(|a| a.n_nodes)
                .sum();
            let interval: &Interval<T> = &multi_interval_domain.intervals[interval_idx];
            let d_loc: DMatrix<T> = interval.scale_dmatrix();
            let n_nodes = interval.n_nodes;
            let u_loc = u.rows(g_start, n_nodes).clone_owned();
            let xi_loc = interval.map_to_physical(); // Vec<T>

            for (loc_row, xi) in xi_loc.iter().enumerate() {
                let x_s: S = S::from(T::from_f64(*xi).unwrap());
                let forcing = non_homogeneous.as_ref().map(|f| f(x_s)).unwrap_or_else(S::zero);
                r[g_start + loc_row] = O::eval(loc_row, x_s, &u_loc, &d_loc) - forcing;
            }
        }
        for i in 0..multi_interval_domain.intervals.len() - 1 {
            let interval_left = &multi_interval_domain.intervals[i];
            let interval_right = &multi_interval_domain.intervals[i + 1];
            let g_start_left: usize = multi_interval_domain.intervals[..i].iter().map(|a| a.n_nodes).sum();
            let g_start_right: usize = multi_interval_domain.intervals[..i + 1].iter().map(|a| a.n_nodes).sum();

            let n_left = interval_left.n_nodes;
            let n_right = interval_right.n_nodes;

            let u_left = u.rows(g_start_left, n_left).clone_owned();
            let u_right = u.rows(g_start_right, n_right).clone_owned();

            let d_left_s: DMatrix<S> = interval_left.scale_dmatrix().map(|t| S::from(t));
            let d_right_s: DMatrix<S> = interval_right.scale_dmatrix().map(|t| S::from(t));
            let du_left = Operations::apply_matrix(&d_left_s, &u_left);
            let du_right = Operations::apply_matrix(&d_right_s, &u_right);
            let row_deriv = g_start_left + n_left - 1;
            let row_val = g_start_right;

            if !bc_rows.contains(&row_deriv) {
                r[row_deriv] = du_left[n_left - 1] - du_right[0];
            }
            if !bc_rows.contains(&row_val) {
                r[row_val] = u_left[n_left - 1] - u_right[0];
            }
            // println!("{:?} + {:?}: interval is {:?}", row_deriv, row_val, i);
        }

        let first_interval = &multi_interval_domain.intervals[0];
        let u_first = u.rows(0, first_interval.n_nodes).clone_owned();
        let d_first = first_interval.scale_dmatrix();

        let last_interval = multi_interval_domain.intervals.last().unwrap();
        let last_start: usize = multi_interval_domain.intervals[..multi_interval_domain.intervals.len() - 1]
            .iter()
            .map(|a| a.n_nodes)
            .sum();
        let u_last = u.rows(last_start, last_interval.n_nodes).clone_owned();
        let d_last = last_interval.scale_dmatrix();
        let d_first_s: DMatrix<S> = d_first.map(|t| S::from(t));
        let d_last_s: DMatrix<S> = d_last.map(|t| S::from(t));

        r[0] = _bc_left.apply(&u_first, &d_first_s, Endpoint::Left);
        r[n_total - 1] = _bc_right.apply(&u_last, &d_last_s, Endpoint::Right);
        // println!("{:?}", r[0]);
        let (interval_idx, n_idx) = multi_interval_domain.node_near_zero();
        let g_start: usize = multi_interval_domain.intervals[..interval_idx]
        .iter()
        .map(|a| a.n_nodes)
        .sum();
        let n_nodes = multi_interval_domain.intervals[interval_idx].n_nodes;
        let u_loc = u.rows(g_start, n_nodes).clone_owned();

        // let d_0 = multi_interval_domain.intervals[interval_idx].scale_dmatrix();
        // let d_0_s: DMatrix<S> = d_0.map(|t| S::from(t));   // match the S/T conversion pattern used elsewhere
        // let du = Operations::apply_matrix(&d_0_s, &u_loc);
        // let d2u = Operations::apply_matrix(&d_0_s, &du);
        // r[multi_interval_domain.global_index((interval_idx, n_idx))] = d2u[n_idx];
        r
    }

    pub fn build_j_r<T, O>(
        ode: &O,
        u: &DVector<T>,
        multi_interval_domain: &MultiIntervalDomain<T>,
        non_homogeneous: Option<impl Fn(Dual<T>) -> Dual<T> + Sync + Send + Clone>,
        bc_left: &BoundaryCondition<Dual<T>>,
        bc_right: &BoundaryCondition<Dual<T>>,
    ) -> (DMatrix<T>, DVector<T>)
    where
        O: ODE<Dual<T>, T> + Sync,
        T: Scalar + Mul<Output = T> + Add<Output = T> + Sub<Output = T> + Div<Output = T>
            + AddAssign + Neg<Output = T> + Copy + RealField + FromPrimitive + ToPrimitive
            + Zero + One + Send + Sync + 'static,
    {
        let n = u.len();

        // Zero-seeded pass gives the residual for free (eps stays 0 throughout,
        // so f_r[i].real is just F(u)), no separate O: ODE<T, T> impl needed.
        let u_zero_seed = DVector::from_fn(n, |i, _| Dual { real: u[i], eps: T::zero() });
        let f_r = residual::<Dual<T>, T, O>(
            ode, &u_zero_seed, multi_interval_domain, non_homogeneous.clone(), bc_left, bc_right,
        );
        let r: DVector<T> = DVector::from_fn(n, |i, _| f_r[i].real);

        let cols: Vec<DVector<T>> = (0..n)
            .into_par_iter()
            .map(|k| {
                let u_seed = DVector::from_fn(n, |i, _| Dual {
                    real: u[i],
                    eps: if i == k { T::one() } else { T::zero() },
                });
                let f_dual: nalgebra::Matrix<Dual<T>, nalgebra::Dyn, nalgebra::Const<1>, nalgebra::VecStorage<Dual<T>, nalgebra::Dyn, nalgebra::Const<1>>> = residual::<Dual<T>, T, O>(
                    ode, &u_seed, multi_interval_domain, non_homogeneous.clone(), bc_left, bc_right,
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