#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::Dual;
    use crate::multi_interval_domain::MultiIntervalDomain;
    use crate::boundary_conditions::BoundaryCondition;
    use nalgebra::{DMatrix, DVector};

    struct QuadraticODE;

    impl ODE<f64> for QuadraticODE {
        fn eval(x: f64, u: f64, _d: &DMatrix<f64>) -> f64 {
            u * u + u - x - x * x
        }
    }

    impl ODE<Dual<f64>> for QuadraticODE {
        fn eval(x: Dual<f64>, u: Dual<f64>, _d: &DMatrix<f64>) -> Dual<f64> {
            u * u + u - x - x * x
        }
    }

    macro_rules! dirichlet {
        ($a:expr, $b:expr) => {
            (BoundaryCondition::Dirichlet($a), BoundaryCondition::Dirichlet($b))
        };
    }

    fn newton<O: ODE<Dual<f64>>>(
        ode: &O,
        domain: &MultiIntervalDomain,
        bc_a: f64,
        bc_b: f64,
        tol: f64,
        max_iter: usize,
    ) -> (DVector<f64>, f64) {
        let n = domain.global_collocation_points().len();
        let mut u = DVector::zeros(n);
        let mut norm = f64::MAX;

        for _ in 0..max_iter {
            let (bc_left, bc_right) = dirichlet!(bc_a, bc_b);
            let (J, R) = build_J_R(ode, &u, domain,
                None::<fn(Dual<f64>) -> Dual<f64>>,
                bc_left, bc_right);

            norm = R.norm();
            if norm < tol { break; }

            let delta = J.lu().solve(&(-&R)).expect("singular Jacobian in Newton");
            u += delta;
        }
        (u, norm)
    }

    #[test]
    fn test_newton_converges_to_exact_solution() {
        let (a, b) = (-1.0_f64, 1.0_f64);
        let domain = MultiIntervalDomain::new(4, a, b, 4);
        let x_nodes = domain.global_collocation_points();

        let tol = 1e-10;
        let (u, residual_norm) = newton(&QuadraticODE, &domain, a, b, tol, 30);

        assert!(
            residual_norm < tol,
            "Newton did not converge: |R| = {residual_norm:.3e}"
        );

        // Pointwise comparison against exact solution u(x) = x.
        let max_err = x_nodes.iter().zip(u.iter())
            .map(|(xi, ui)| (ui - xi).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_err < 1e-9,
            "Solution deviates from u(x)=x: max pointwise error = {max_err:.3e}"
        );
    }

    #[test]
    fn test_ad_jacobian_matches_finite_differences() {
        let (a, b) = (-1.0_f64, 1.0_f64);
        let domain = MultiIntervalDomain::new(3, a, b, 3);
        let n = domain.global_collocation_points().len();

        // Evaluate away from the solution so the Jacobian is non-trivial.
        let u = DVector::from_fn(n, |i, _| {
            0.3 * (i as f64 - n as f64 / 2.0) / n as f64
        });

        // AD Jacobian + base residual.
        let (J_ad, R_base) = {
            let (bc_left, bc_right) = dirichlet!(a, b);
            build_J_R(&QuadraticODE, &u, &domain,
                None::<fn(Dual<f64>) -> Dual<f64>>,
                bc_left, bc_right)
        };

        // Forward-difference Jacobian.
        let h = 1e-6;
        let mut J_fd = DMatrix::zeros(n, n);
        for k in 0..n {
            let mut u_pert = u.clone();
            u_pert[k] += h;
            let (bc_left, bc_right) = dirichlet!(a, b);
            let (_, R_pert) = build_J_R(&QuadraticODE, &u_pert, &domain,
                None::<fn(Dual<f64>) -> Dual<f64>>,
                bc_left, bc_right);
            for i in 0..n {
                J_fd[(i, k)] = (R_pert[i] - R_base[i]) / h;
            }
        }

        // Relative Frobenius-norm difference; FD error is O(h) ≈ 1e-6.
        let abs_err = (&J_ad - &J_fd).norm();
        let rel_err = abs_err / J_fd.norm().max(1e-15);
        assert!(
            rel_err < 1e-4,
            "AD Jacobian disagrees with FD: relative error = {rel_err:.3e}\n\
             AD:\n{J_ad}\nFD:\n{J_fd}"
        );
    }

    #[test]
    fn test_collocation_error_decreases_with_refinement() {
        let (a, b) = (-1.0_f64, 1.0_f64);
        let tol = 1e-10;

        let mut prev_err = f64::MAX;
        for n_intervals in [2, 4, 8] {
            let domain = MultiIntervalDomain::new(n_intervals, a, b, 4);
            let x_nodes = domain.global_collocation_points();
            let (u, _) = newton(&QuadraticODE, &domain, a, b, tol, 30);

            let max_err = x_nodes.iter().zip(u.iter())
                .map(|(xi, ui)| (ui - xi).abs())
                .fold(0.0_f64, f64::max);

            assert!(
                max_err <= prev_err + 1e-14,
                "Error did not decrease with refinement at n_intervals={n_intervals}: \
                 {max_err:.3e} > prev {prev_err:.3e}"
            );
            prev_err = max_err;
        }
    }
}