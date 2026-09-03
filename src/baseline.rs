fn run_baseline<T>()
where
    T: num_traits::Float + Scalar
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
        + Send
        + Sync
        + LowerExp
        + Trig
        + 'static,
{
    use crate::toolbox::test_odes::SineBaselineODE;

    let d = (0.0, 1.0);
    let domain = MultiIntervalDomain::<T>::new(
        polynomial_trait::PolynomialKind::Chebyshev, 13, d, 1, IntervalDivision::Uniform,
    );

    // Zero initial guess. Since the operator is linear in u, the
    // Jacobian is exact and constant, so a single linear solve gives
    // the collocation solution directly: J * delta = -r, u = u0 + delta.
    let u0 = DVector::from_vec(
        domain.intervals.iter()
            .flat_map(|a| a.map_to_physical().map(|_| T::zero()).iter().copied().collect::<Vec<T>>())
            .collect(),
    );

    let (j_mat, r) = build_j_r::<T, SineBaselineODE>(
        &SineBaselineODE, &u0, &domain,
        Some(SineBaselineODE::forcing),
        &SineBaselineODE::bc_left(), &SineBaselineODE::bc_right(),
    );

    // Direct dense linear solve via nalgebra's LU decomposition,
    // rather than Newton iteration or the banded solver.
    let rhs = -&r;
    let delta = j_mat
        .clone()
        .lu()
        .solve(&rhs)
        .expect("linear system should be nonsingular for the baseline problem");

    let u = u0 + delta;

    // Verify against the exact solution u(x) = sin(x): assemble the
    // exact values at the same physical nodes, and report L2/L-infinity
    // discretization error.
    let x_phys: Vec<T> = domain.intervals.iter()
        .flat_map(|a| a.map_to_physical().iter().copied().collect::<Vec<T>>())
        .collect();
    let u_exact: DVector<T> = DVector::from_vec(
        x_phys.iter().map(|&x| x.sin()).collect()
    );

    let err = &u - &u_exact;
    let l2_err = err.norm().to_f64().unwrap();
    let linf_err = err.iter()
        .map(|v| v.abs().to_f64().unwrap())
        .fold(0.0_f64, f64::max);

    println!("Baseline (u'' + u = 0, u(0)=0, u(1)=sin(1)):");
    println!("  L2 error   = {l2_err:.3e}");
    println!("  Linf error = {linf_err:.3e}");

    // Residual at the solution, purely as a sanity check that the
    // linear solve actually zeroed the collocation equations.
    let (_, r_final) = build_j_r::<T, SineBaselineODE>(
        &SineBaselineODE, &u, &domain,
        Some(SineBaselineODE::forcing),
        &SineBaselineODE::bc_left(), &SineBaselineODE::bc_right(),
    );
    println!("  Residual after solve = {:.3e}", r_final.norm());

    plot_solutions(
        &[("Collocation approx.", u.clone()), ("Exact sin(x)", u_exact)],
        &domain,
        "Baseline: u'' + u = 0, exact u = sin(x)",
        "baseline_sine.html",
    );
}