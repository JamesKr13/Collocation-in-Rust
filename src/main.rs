use nalgebra::{DMatrix, DVector, Dyn, RealField, Scalar};
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use std::f32::consts::PI;
use std::fmt::LowerExp;
use std::iter::Product;
use std::ops::{Add, Mul, Sub, Div, AddAssign, Neg};
use std::process::Output;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use plotly::{Plot, Scatter};
use plotly::common::{Mode, Line, Marker, MarkerSymbol};
use plotly::layout::{Layout, Axis};

mod toolbox;
mod collocation_method;
mod h_interval;

use h_interval::multi_interval::{MultiIntervalDomain, IntervalDivision};
use h_interval::polynomial::{polynomial_trait, chebyshev::Chebyshev};
use toolbox::banded_solver::BandedLinSolve;
use toolbox::auto_diff_dual::Dual;
use collocation_method::ode_trait::ODE;
use collocation_method::auto_diff_collocation::build_j_r;
use toolbox::grapher::{plot_solutions, plot_convergence, plot_error_vs_degree, plot_convergence_multi, plot_error_vs_intervals_multi};
use crate::h_interval::multi_interval::IntervalDivision::ChebyshevInterval;
use crate::toolbox::test_odes::{KdvSolitonODE, SineBaselineODE, ManufacturedSolutionODE, CubicReactionManufacturedODE, HeteroclinicOrbit};
use crate::toolbox::trig_trait::Trig;

fn iteration_loop<T, O>(ode: &O, running: Arc<AtomicBool>)
where
    T: BandedLinSolve + num_traits::Float + Scalar
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
    O: ODE<Dual<T>, T> + Sync,
{
    let d = (-0.0, 1.0);
    let domain = MultiIntervalDomain::<T>::new(polynomial_trait::PolynomialKind::Chebyshev, 4, d, 8, IntervalDivision::Uniform);
    println!("{:?}", domain.break_points);
    let mut iter_u = Vec::new();
    let mut norms: Vec<f64> = Vec::new();
    let mut u = DVector::from_vec(domain.intervals.iter().flat_map(|a| {
        let v = a.map_to_physical().map(|x|T::from_f64(10.0*x.sinh()/x.cosh()).unwrap());
        v.iter().copied().collect::<Vec<T>>()
        }).collect());

    for iter in 0..=500 {
        iter_u.push(u.clone());
        if !running.load(Ordering::SeqCst) {
            println!("\nCtrl-C received at iter {iter}, plotting current approximation...");
            break;
        }

        let (j_mat, r) = build_j_r::<T, O>(
            ode, &u, &domain, Some(O::forcing), &O::bc_left(), &O::bc_right(),
        );
        let norm = r.norm();
        norms.push(norm.to_f64().expect("residual norm should convert to f64"));
        println!("Iter {iter:>2}: |F| = {norm:.3e}");
        if norm < T::from(1e-16).unwrap() { println!("Converged."); break; }

        let (kl, ku) = collocation_bandwidth(&domain);
        let delta = T::banded_solve(&j_mat, &(-&r), kl, ku);
        u += delta;
        println!("{:?}", u[1]);
    }
    println!("{:?}", u[1]);
    // plot_solutions(
    //     &[("Approximation", u.clone())],
    //     &domain,
    //     "Pendulum ODE solution",
    //     "solution1.html",
    // );
    plot_convergence(&norms, "convergence.html");
}


fn collocation_bandwidth<T>(domain: &MultiIntervalDomain<T>) -> (usize, usize)
where
    T: Scalar
        + Mul<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Div<Output = T>
        + AddAssign
        + Neg<Output = T>
        + Copy
        + FromPrimitive
        + ToPrimitive
        + Zero
        + Send
        + Sync
        + 'static
{
    let mut max_bw = 0usize;
    for pair in domain.intervals.windows(2) {
        let n_left = pair[0].collocation_nodes().len();
        let n_right = pair[1].collocation_nodes().len();
        // coupling rows at the interface span roughly n_left + n_right - 1
        max_bw = max_bw.max(n_left.max(n_right));
    }
    // fall back to the single-interval case
    if domain.intervals.len() == 1 {
        max_bw = domain.intervals[0].collocation_nodes().len();
    }
    (max_bw, max_bw) // kl, ku usually symmetric for these schemes
}

fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    // iteration_loop::<f64, SineBaselineODE>(&SineBaselineODE, running);
    // let degrees: Vec<usize> = (3..=24).step_by(2).collect();

    // let beta = ManufacturedSolutionODE::BETA;
    // let rho_star = 1.0 / beta.sqrt() + (1.0 + 1.0 / beta).sqrt();
    // let c_leading = 1.0; // placeholder; see note below on fitting this

    // linear_solve_convergence::<f64, ManufacturedSolutionODE, _>(
    //     &ManufacturedSolutionODE,
    //     move |x: f64| 1.0 / (1.0 + beta * x * x),
    //     &degrees,
    //     (-1.0, 1.0),
    //     Some((rho_star, c_leading)),
    // );

    // let beta = CubicReactionManufacturedODE::BETA;
    // let exact = move |x: f64| -> f64 {
    //     (std::f64::consts::PI * x).sin() / (1.0 + beta * x * x)
    // };

    // let initial_guess = move |x: f64| -> f64 { (std::f64::consts::PI * x).sin() };

    // let c = 1.0 / beta.sqrt();
    // let rho_star = c + (c * c + 1.0).sqrt();
    // let theory = Some((rho_star, 1.0));
    // let degrees: Vec<usize> = (4..=20).step_by(2).collect();

    // nonlinear_solve_convergence::<f64, CubicReactionManufacturedODE, _, _>(
    //     &CubicReactionManufacturedODE,
    //     exact,
    //     initial_guess,
    //     &degrees,
    //     (-1.0, 1.0),
    //     1,
    //     theory,
    //     running,
    // );

    // let c = KdvSolitonODE::SPEED;
    // let lambda = c.sqrt(); // decay rate, matches Beyn eigenvalue √c

    // let exact = move |x: f64| -> f64 {
    //     c / (0.5 * lambda * x).cosh().powi(2)
    // };

    // let initial_guess = move |x: f64| -> f64 {
    //     3.0*c / (0.5 * lambda * x).cosh().powi(2)
    //     };
    // let l_domain = 10.0;
    // let b = std::f64::consts::PI / (lambda * l_domain);
    // let rho_star = b + (b * b + 1.0).sqrt();
    // let theory = Some((rho_star, 1.0));

    // let degrees: Vec<usize> = (3..=13).step_by(2).collect();

    // nonlinear_solve_convergence::<f64, KdvSolitonODE, _, _>(
    //     &KdvSolitonODE,
    //     exact,
    //     initial_guess,
    //     &degrees,
    //     (-l_domain, l_domain),
    //     5,
    //     theory,
    //     running,
    // );

    // let interval_counts: Vec<usize> = (1..=20).collect();
    // let theory: Option<(f64, f64)> = None;

    let u_minus = HeteroclinicOrbit::u_minus();
    let lambda_left = HeteroclinicOrbit::lambda_val(u_minus, 1.0);  // unstable rate at x -> -inf
    let lambda_right = HeteroclinicOrbit::lambda_val(0.0, -1.0);    // stable rate at x -> +inf (negative)

    // Transition rate for the initial-guess logistic profile: average of the
    // two asymptotic |rate| magnitudes, so both tails decay at roughly the
    // right speed even though there's no closed-form exact solution to match.
    let k_guess = 0.5 * (lambda_left - lambda_right); // lambda_right < 0, so this is positive

    // No exact solution available for this problem -- use the same matched
    // asymptotic profile as both "exact" (for plotting/reference only, NOT
    // a true convergence benchmark) and as the initial guess. Error norms
    // against this "exact" are not meaningful; only use plot_solutions to
    // visually inspect the converged shape, and rely on Newton residual /
    // self-convergence between degrees for accuracy checks instead.
    let exact = move |x: f64| -> f64 {
        u_minus * 0.5 * (1.0 - (k_guess * x).tanh())
    };

    let initial_guess = move |x: f64| -> f64 {
        u_minus * 0.5 * (1.0 - (k_guess * x).tanh())
    };

    let l_domain = 10.0;

    // Bernstein-rate theory line: nearest singularity distance depends on
    // which asymptotic rate is smaller in magnitude (slower decay controls
    // the dominant analyticity strip), i.e. min(|lambda_left|, |lambda_right|).
    let lambda_min = lambda_left.min(lambda_right.abs());
    let b = std::f64::consts::PI / (lambda_min * l_domain);
    let rho_star = b + (b * b + 1.0).sqrt();
    let theory = Some((rho_star, 1.0));

    let degrees: Vec<usize> = (3..=13).step_by(2).collect();

    nonlinear_solve_convergence::<f64, HeteroclinicOrbit, _, _>(
        &HeteroclinicOrbit,
        exact,
        initial_guess,
        &degrees,
        (-l_domain, l_domain),
        5,
        theory,
        running,
    );

//     h_refinement_convergence::<f64, KdvSolitonODE, _, _>(
//     &KdvSolitonODE,
//     exact,
//     initial_guess,
//     &degrees,
//     (-l_domain, l_domain),
//     &interval_counts,
//     theory,
//     running,
// );


}



fn barycentric_weights<T>(values: DVector<T>, j: usize) -> T where T: Sub<Output = T> + FromPrimitive + Product + Clone {
    values.iter().enumerate().map(|(k, x)| if j != k {(values[j].clone() - x.clone())} else {T::from_i8(0).unwrap()}).product()
}

fn linear_solve_convergence<T, O, F>(
    ode: &O,
    exact: F,
    degrees: &[usize],
    domain_bounds: (f64, f64),
    theory: Option<(f64, f64)>, // (rho_star, leading constant C)
)
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
    O: ODE<Dual<T>, T> + Sync,
    F: Fn(T) -> T + Copy,
{
    let d = domain_bounds;
    let mut l2_errors = Vec::new();
    let mut linf_errors = Vec::new();

    let mut all_domains: Vec<MultiIntervalDomain<T>> = Vec::new();
    let mut all_solutions: Vec<DVector<T>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();

    for &n in degrees {
        let domain = MultiIntervalDomain::<T>::new(
            polynomial_trait::PolynomialKind::Chebyshev, 1, d, n, IntervalDivision::Uniform,
        );

        let u0 = DVector::from_vec(domain.intervals.iter().flat_map(|a| {
            a.map_to_physical().map(|_| T::zero()).iter().copied().collect::<Vec<T>>()
        }).collect());

        let (j_mat, r) = build_j_r::<T, O>(
            ode, &u0, &domain, Some(O::forcing), &O::bc_left(), &O::bc_right(),
        );

        let rhs = -&r;
        let delta = j_mat
            .clone()
            .lu()
            .solve(&rhs)
            .expect("linear system should be nonsingular");
        let u = u0 + &delta;
        let resid_check = &j_mat * &delta + &r;
        println!("post-solve residual: {:.3e}", resid_check.norm());

        let x_phys: Vec<T> = domain.intervals.iter()
            .flat_map(|a| a.map_to_physical().iter().copied().map(|a| T::from_f64(a).unwrap()).collect::<Vec<T>>())
            .collect();
        let u_exact = DVector::from_vec(
            x_phys.iter().map(|&x| exact(x)).collect::<Vec<T>>()
        );

        let err = &u - &u_exact;
        let l2_err = err.norm().to_f64().unwrap();
        let linf_err = err.iter()
            .map(|v| v.abs().to_f64().unwrap())
            .fold(0.0_f64, f64::max);

        println!("n = {n:>3}: L2 = {l2_err:.3e}, Linf = {linf_err:.3e}");

        l2_errors.push(l2_err);
        linf_errors.push(linf_err);

        labels.push(format!("n = {n}"));
        all_solutions.push(u);
        all_domains.push(domain);
    }

    plot_error_vs_degree(degrees, &l2_errors, &linf_errors, theory, "baseline_convergence.html");

    let solution_entries: Vec<(&str, DVector<T>, &MultiIntervalDomain<T>)> = labels
        .iter()
        .zip(all_solutions.into_iter())
        .zip(all_domains.iter())
        .map(|((label, u), domain)| (label.as_str(), u, domain))
        .collect();

    let exact_f64 = move |x: f64| -> f64 {
        exact(T::from_f64(x).unwrap()).to_f64().unwrap()
    };

    plot_solutions(
        &solution_entries,
        Some(("Exact", &exact_f64 as &dyn Fn(f64) -> f64)),
        "Solutions vs exact, across degrees",
        "baseline_solutions.html",
    );
}


fn nonlinear_solve_convergence<T, O, F, G>(
    ode: &O,
    exact: F,
    initial_guess: G,
    degrees: &[usize],
    domain_bounds: (f64, f64),
    n_intervals: usize,
    theory: Option<(f64, f64)>,
    running: Arc<AtomicBool>,
)
where
    T: BandedLinSolve + num_traits::Float + Scalar
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
    O: ODE<Dual<T>, T> + Sync,
    F: Fn(T) -> T + Copy,
    G: Fn(T) -> T + Copy,
{
    let d = domain_bounds;
    let mut l2_errors = Vec::new();
    let mut linf_errors = Vec::new();

    let mut all_domains: Vec<MultiIntervalDomain<T>> = Vec::new();
    let mut all_solutions: Vec<DVector<T>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut all_norms: Vec<(String, Vec<f64>)> = Vec::new();
    for &n in degrees {
        if !running.load(Ordering::SeqCst) {
            println!("\nCtrl-C received before degree {n}, stopping study.");
            break;
        }

        let domain = MultiIntervalDomain::<T>::new(
            polynomial_trait::PolynomialKind::Chebyshev, n_intervals, d, n, IntervalDivision::Uniform,
        );

        let mut u = DVector::from_vec(domain.intervals.iter().flat_map(|a| {
            a.map_to_physical().map(|x| initial_guess(T::from_f64(x).unwrap())).iter().copied().collect::<Vec<T>>()
        }).collect());

        let mut norms: Vec<f64> = Vec::new();
        let (kl, ku) = collocation_bandwidth(&domain);

        for iter in 0..=100 {
            if !running.load(Ordering::SeqCst) {
                println!("\nCtrl-C received at iter {iter} (n = {n}), using current approximation.");
                break;
            }

            let (j_mat, r) = build_j_r::<T, O>(
                ode, &u, &domain, Some(O::forcing), &O::bc_left(), &O::bc_right(),
            );

            let norm = r.norm();
            norms.push(norm.to_f64().expect("residual norm should convert to f64"));
            println!("n = {n:>3}, iter {iter:>2}: |F| = {norm:.3e}");

            if norm < T::from_f64(1e-10).unwrap() {
                println!("Converged for n = {n}.");
                break;
            }

            let delta = T::banded_solve(&j_mat, &(-&r), kl, ku);
            u += delta;
        }
        println!("{:?}",empirical_convergence_order(&norms));
        all_norms.push((format!("n = {n}"), norms.clone()));

        let x_phys: Vec<T> = domain.intervals.iter()
            .flat_map(|a| a.map_to_physical().iter().copied().map(|x| T::from_f64(x).unwrap()).collect::<Vec<T>>())
            .collect();
        let u_exact = DVector::from_vec(
            x_phys.iter().map(|&x| exact(x)).collect::<Vec<T>>()
        );

        let err = &u - &u_exact;
        let w_ref = Chebyshev::clenshaw_curtis_weights(n);       // weights on [-1,1]
        let jacobian = (domain_bounds.1 - domain_bounds.0) / 2.0; // = L for your [-L, L] domain
        let weights: DVector<f64> = w_ref * jacobian;
        // println!("{:?}", weights);
        let l2_err = err.iter()
            .zip(weights.iter())
            .map(|(e, w)| {
                let e64 = e.to_f64().unwrap();
                w * e64 * e64
            })
            .sum::<f64>()
            .sqrt();
        let linf_err = err.iter()
            .map(|v| v.abs().to_f64().unwrap())
            .fold(0.0_f64, f64::max);

        println!("n = {n:>3}: L2 = {l2_err:.3e}, Linf = {linf_err:.3e}");

        l2_errors.push(l2_err);
        linf_errors.push(linf_err);

        labels.push(format!("n = {n}"));
        all_solutions.push(u);
        all_domains.push(domain);

        plot_convergence(&norms, &format!("newton_convergence_n{n}.html"));
    }
    plot_convergence_multi(&all_norms, "newton_convergence_all_degrees.html");

    plot_error_vs_degree(degrees, &l2_errors, &linf_errors, theory, "nonlinear_convergence.html");

    let solution_entries: Vec<(&str, DVector<T>, &MultiIntervalDomain<T>)> = labels
        .iter()
        .zip(all_solutions.into_iter())
        .zip(all_domains.iter())
        .map(|((label, u), domain)| (label.as_str(), u, domain))
        .collect();

    let exact_f64 = move |x: f64| -> f64 {
        exact(T::from_f64(x).unwrap()).to_f64().unwrap()
    };

    plot_solutions(
        &solution_entries,
        Some(("Exact", &exact_f64 as &dyn Fn(f64) -> f64)),
        "Solutions vs exact, across degrees",
        "nonlinear_solutions.html",
    );
}

fn empirical_convergence_order(norms: &[f64]) -> Vec<f64> {
    let mut orders = Vec::new();
    for k in 1..norms.len().saturating_sub(1) {
        let (r_prev, r_curr, r_next) = (norms[k - 1], norms[k], norms[k + 1]);
        if r_prev > 1e-15 && r_curr > 1e-15 && r_next > 1e-15 {
            let p = (r_next / r_curr).ln() / (r_curr / r_prev).ln();
            orders.push(p);
        }
    }
    orders
}

fn h_refinement_convergence<T, O, F, G>(
    ode: &O,
    exact: F,
    initial_guess: G,
    degrees: &[usize],
    domain_bounds: (f64, f64),
    interval_counts: &[usize],
    theory: Option<(f64, f64)>,
    running: Arc<AtomicBool>,
)
where
    T: BandedLinSolve + num_traits::Float + Scalar
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
    O: ODE<Dual<T>, T> + Sync,
    F: Fn(T) -> T + Copy,
    G: Fn(T) -> T + Copy,
{
    let d = domain_bounds;

    // One (label, errors-per-K) series per fixed polynomial degree.
    let mut l2_series: Vec<(String, Vec<f64>)> = Vec::new();
    let mut linf_series: Vec<(String, Vec<f64>)> = Vec::new();

    for &n in degrees {
        if !running.load(Ordering::SeqCst) {
            println!("\nCtrl-C received before degree {n}, stopping h-refinement study.");
            break;
        }

        let mut l2_errors = Vec::new();
        let mut linf_errors = Vec::new();

        for &k_intervals in interval_counts {
            if !running.load(Ordering::SeqCst) {
                println!("\nCtrl-C received before K = {k_intervals} (n = {n}), stopping.");
                break;
            }

            let domain = MultiIntervalDomain::<T>::new(
                polynomial_trait::PolynomialKind::Chebyshev,
                k_intervals,
                d,
                n,
                IntervalDivision::Uniform,
            );

            let mut u = DVector::from_vec(
                domain.intervals.iter()
                    .flat_map(|a| {
                        a.map_to_physical()
                            .map(|x| initial_guess(T::from_f64(x).unwrap()))
                            .iter().copied().collect::<Vec<T>>()
                    })
                    .collect(),
            );

            let (kl, ku) = collocation_bandwidth(&domain);

            for iter in 0..=100 {
                if !running.load(Ordering::SeqCst) {
                    println!("\nCtrl-C received at iter {iter} (n = {n}, K = {k_intervals}).");
                    break;
                }

                let (j_mat, r) = build_j_r::<T, O>(
                    ode, &u, &domain, Some(O::forcing), &O::bc_left(), &O::bc_right(),
                );

                let norm = r.norm();

                if norm < T::from_f64(1e-10).unwrap() {
                    break;
                }

                let delta = T::banded_solve(&j_mat, &(-&r), kl, ku);
                u += delta;
            }

            // Per-interval weighted L2 error, summed across intervals.
            let mut l2_sq_sum = 0.0_f64;
            let mut linf_err = 0.0_f64;
            let mut offset = 0usize;

            for interval in domain.intervals.iter() {
                let x_phys = interval.map_to_physical();
                let len = x_phys.len();

                let w_ref = Chebyshev::clenshaw_curtis_weights(n);
                // half-width of THIS subinterval, for the correct local Jacobian
                let (a, b) = domain_bounds; // adjust to your actual accessor
                let jacobian = (b - a) / 2.0;
                let weights: DVector<f64> = w_ref * jacobian;

                for (i, &xp) in x_phys.iter().enumerate() {
                    let x_t = T::from_f64(xp).unwrap();
                    let e64 = (u[offset + i] - exact(x_t)).to_f64().unwrap();
                    l2_sq_sum += weights[i] * e64 * e64;
                    linf_err = linf_err.max(e64.abs());
                }

                offset += len;
            }

            let l2_err = l2_sq_sum.sqrt();

            println!("n = {n:>3}, K = {k_intervals:>3}: L2 = {l2_err:.3e}, Linf = {linf_err:.3e}");

            l2_errors.push(l2_err);
            linf_errors.push(linf_err);
        }

        l2_series.push((format!("n = {n}"), l2_errors));
        linf_series.push((format!("n = {n}"), linf_errors));
    }

    plot_error_vs_intervals_multi(
    interval_counts,
    &l2_series,
    &linf_series,
    theory,
    true, // loglog: algebraic h-refinement usually reads best log-log
    "h_refinement_error.html",
);
}