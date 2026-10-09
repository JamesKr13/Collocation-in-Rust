use nalgebra::{DMatrix, DVector, Dyn, RealField, Scalar, zero};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use plotly::common::{Line, Marker, MarkerSymbol, Mode};
use plotly::contour::Operation;
use plotly::layout::ThetaUnit::Degrees;
use plotly::layout::{Axis, Layout};
use plotly::{Configuration, Plot, Scatter};
use std::f32::consts::PI;
use std::fmt::LowerExp;
use std::iter::Product;
use std::net::Shutdown::Write;
use std::ops::{Add, AddAssign, Div, Index, Mul, Neg, Sub};
use std::process::Output;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

mod collocation_method;
mod h_interval;
mod toolbox;

use collocation_method::auto_diff_collocation::build_j_r;
use collocation_method::ode_trait::ODE;
use h_interval::multi_interval::{IntervalDivision, MultiIntervalDomain};
use h_interval::polynomial::{chebyshev::Chebyshev, polynomial_trait};
use toolbox::auto_diff_dual::Dual;
use toolbox::banded_solver::BandedLinSolve;
use toolbox::grapher::{
    plot_continuation_error_grid, plot_convergence, plot_convergence_multi, plot_error_vs_degree,
    plot_error_vs_intervals_multi, plot_iterates, plot_solutions,
};

mod baseline;
use crate::h_interval::multi_interval;
use crate::h_interval::multi_interval::IntervalDivision::{ChebyshevInterval, Uniform};
use crate::toolbox::asymptotic_guess::{asymptotic_guess, sanity_check};
use crate::toolbox::dual_matrix_operations::Operations;
use crate::toolbox::grapher::plot_all_h_step_solutions;
use crate::toolbox::pb::{EigenPair, HeteroclinicOrbit};
use crate::toolbox::test_odes::{
    CubicReactionManufacturedODE, KdvSolitonODE, ManufacturedSolutionODE, SineBaselineODE,
};
use crate::toolbox::trig_trait::Trig;
use baseline::{
    run_cubic_manufactured, run_kdv, run_manufactured, run_pendulum, run_sine_baseline,
};

use crate::toolbox::asymptotic_guess;

use log::{LevelFilter, debug, info, warn};
use simplelog::{Config, WriteLogger};
use std::fs::File;

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
        + 'static,
{
    let mut max_bw = 0usize;
    for pair in domain.intervals.windows(2) {
        let n_left = pair[0].collocation_nodes().len();
        let n_right = pair[1].collocation_nodes().len();
        max_bw = max_bw.max(n_left.max(n_right));
    }
    if domain.intervals.len() == 1 {
        max_bw = domain.intervals[0].collocation_nodes().len();
    }
    (max_bw, max_bw)
}

fn main() {
    let log_file = File::create("collocation.log").expect("failed to create log");
    WriteLogger::init(LevelFilter::Trace, Config::default(), log_file)
        .expect("failt to create logger");
    // env_logger::init();

    let running = Arc::new(AtomicBool::new(true));
    let running_handler = running.clone();

    ctrlc::set_handler(move || {
        running_handler.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let mut values: Vec<usize> = (5..=55).step_by(1).collect();

    if std::env::var("BASELINE").is_ok() {
        let mut values: Vec<usize> = (5..=18).step_by(1).collect();
        run_sine_baseline(&values, running.clone());
        run_pendulum(&values, running.clone());
        let mut values: Vec<usize> = (5..=80).step_by(2).collect();
        run_manufactured(&values, running.clone());
        run_cubic_manufactured(&values, running.clone());
        let mut values: Vec<usize> = (5..=140).step_by(4).collect();
        run_kdv(&values, running.clone());
    }
    let mut values: Vec<usize> = (1..=200).step_by(1).collect();
    // Degrees below 5 have too few nodes to resolve the orbit.
    let delta_bar = 1.8;
    let s0 = 1.5; // speed the asymptotic guess is built at
    let s1 = 2.0; // end of the continuation

    // if let Some((_domain, u)) = solve_single_approximation_h_step(
    //     13,13,
    //     (-1.0, 0.5),
    //     2.5, delta_bar,
    //     0.5,
    //     100,
    //     running.clone(),
    // ) {
    //     info!("single collocation solve done: {} unknowns", u.len());
    // }

    let h_intervals = vec![1,2,4,8,16,32,64,128,256,512];
    // heteroclinic_continuation_interval_convergence(
    //     &h_intervals, 5,
    //     25,50,
    //     (-30.0, 10.0),
    //     s0, s1, 2,
    //     delta_bar,
    //     5,
    //     "heteroclinic_continuation_errors.html",
    //     running.clone()
    // );

    delta_sweep_3d(
        13,
        33,
        (-30.0, 10.0),
        2.5,
        1.8,
        20.0,
        50,
        "delta_sweep_3d.html",
        running.clone(),
    );
    // speed_sweep_3d(
    //     13, 33,
    //     (-30.0, 10.0),
    //     1.5, 2.5,
    //     delta_bar,
    //     40,
    //     "speed_sweep_3d.html",
    //     running.clone(),
    // );
}

fn barycentric_weights<T>(values: DVector<T>, j: usize) -> T
where
    T: Sub<Output = T> + FromPrimitive + Product + Clone,
{
    values
        .iter()
        .enumerate()
        .map(|(k, x)| {
            if j != k {
                (values[j].clone() - x.clone())
            } else {
                T::from_i8(0).unwrap()
            }
        })
        .product()
}

// fn linear_solve_convergence<T, O, F>(
//     ode: &O,
//     exact: F,
//     degrees: &[usize],
//     domain_bounds: (f64, f64),
//     theory: Option<(f64, f64)>, // (rho_star, leading constant C)
// ) where
//     T: num_traits::Float
//         + Scalar
//         + Mul<Output = T>
//         + Add<Output = T>
//         + Sub<Output = T>
//         + Div<Output = T>
//         + AddAssign
//         + Neg<Output = T>
//         + Copy
//         + RealField
//         + FromPrimitive
//         + ToPrimitive
//         + Zero
//         + Send
//         + Sync
//         + LowerExp
//         + Trig
//         + 'static,
//     O: ODE<Dual<T>, T> + Sync,
//     F: Fn(T) -> T + Copy,
// {
//     let d = domain_bounds;
//     let mut l2_errors = Vec::new();
//     let mut linf_errors = Vec::new();

//     let mut all_domains: Vec<MultiIntervalDomain<T>> = Vec::new();
//     let mut all_solutions: Vec<DVector<T>> = Vec::new();
//     let mut labels: Vec<String> = Vec::new();

//     for &n in degrees {
//         let domain = MultiIntervalDomain::<T>::new(
//             polynomial_trait::PolynomialKind::Chebyshev,
//             1,
//             d,
//             n,
//             IntervalDivision::Uniform,
//         );

//         let x_phys: Vec<T> = domain
//             .intervals
//             .iter()
//             .flat_map(|a| {
//                 a.map_to_physical()
//                     .iter()
//                     .copied()
//                     .map(|a| T::from_f64(a).unwrap())
//                     .collect::<Vec<T>>()
//             })
//             .collect();
//         let u_exact = DVector::from_vec(x_phys.iter().map(|&x| exact(x)).collect::<Vec<T>>());

//         let exact_f64 = move |x: f64| -> f64 { exact(T::from_f64(x).unwrap()).to_f64().unwrap() };

//         let u0 = DVector::from_vec(
//             domain
//                 .intervals
//                 .iter()
//                 .flat_map(|a| {
//                     a.map_to_physical()
//                         .map(|_| T::zero())
//                         .iter()
//                         .copied()
//                         .collect::<Vec<T>>()
//                 })
//                 .collect(),
//         );
//         let u_start = u0.clone();
//         let (j_mat, r) = build_j_r::<T, O>(
//             ode,
//             &u0,
//             &domain,
//             Some(O::forcing),
//             &O::bc_left(),
//             &O::bc_right(),
//             None,
//         );

//         let rhs = -&r;
//         let delta = j_mat
//             .clone()
//             .lu()
//             .solve(&rhs)
//             .expect("linear system should be nonsingular");
//         let u = u0 + &delta;
//         let resid_check = &j_mat * &delta + &r;
//         info!("post-solve residual: {:.3e}", resid_check.norm());

//         if cfg!(debug_assertions) {
//             let iterates = vec![
//                 ("initial guess (0)".to_string(), u_start),
//                 ("after linear solve".to_string(), u.clone()),
//             ];
//             plot_iterates(
//                 &iterates,
//                 &domain,
//                 &exact_f64,
//                 &format!("Linear solve, n = {n}"),
//                 &format!("linear_iterates_n{n}.html"),
//                 "obsolete",
//             );
//         }
//         let err = &u - &u_exact;
//         let (l2_err, linf_err) = compute_error_norms(&err, domain_bounds);

//         info!("n = {n:>3}: L2 = {l2_err:.3e}, Linf = {linf_err:.3e}");
//         l2_errors.push(l2_err);
//         linf_errors.push(linf_err);
//         labels.push(format!("n = {n}"));
//         all_solutions.push(u);
//         all_domains.push(domain);
//     }

//     plot_error_vs_degree(
//         degrees,
//         &l2_errors,
//         &linf_errors,
//         theory,
//         "baseline_convergence.html",
//         "",
//     );

//     let solution_entries: Vec<(&str, DVector<T>, &MultiIntervalDomain<T>)> = labels
//         .iter()
//         .zip(all_solutions.into_iter())
//         .zip(all_domains.iter())
//         .map(|((label, u), domain)| (label.as_str(), u, domain))
//         .collect();

//     let exact_f64 = move |x: f64| -> f64 { exact(T::from_f64(x).unwrap()).to_f64().unwrap() };

//     plot_solutions(
//         &solution_entries,
//         Some(("Exact", &exact_f64 as &dyn Fn(f64) -> f64)),
//         "Solutions vs exact, across degrees",
//         "baseline_solutions.html",
//         "test",
//     );
// }
fn nonlinear_solve_convergence<T, O, F, G>(
    ode: &O,
    exact: F,
    initial_guess: G,
    degrees: &[usize],
    domain_bounds: (f64, f64),
    n_intervals: usize,
    theory: Option<(f64, f64)>,
    running: Arc<AtomicBool>,
    name: &str,
) where
    T: BandedLinSolve
        + num_traits::Float
        + Scalar
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
    let tol = 1e-9_f64;
    let mut all_domains: Vec<MultiIntervalDomain<T>> = Vec::new();
    let mut all_solutions: Vec<DVector<T>> = Vec::new();
    let mut labels: Vec<String> = Vec::new();
    let mut all_norms: Vec<(String, Vec<f64>)> = Vec::new();

    let exact_f64 = move |x: f64| -> f64 { exact(T::from_f64(x).unwrap()).to_f64().unwrap() };

    for &n in degrees {
        if !running.load(Ordering::SeqCst) {
            info!("\nCtrl-C received before degree {n}, stopping study.");
            break;
        }

        let domain = MultiIntervalDomain::new(
            polynomial_trait::PolynomialKind::Chebyshev,
            n_intervals,
            d,
            n,
            IntervalDivision::Uniform,
        );

        let mut u = DVector::from_vec(
            domain
                .intervals
                .iter()
                .flat_map(|a| {
                    a.map_to_physical()
                        .map(|x| initial_guess(T::from_f64(x).unwrap()))
                        .iter()
                        .copied()
                        .collect::<Vec<T>>()
                })
                .collect(),
        );

        let mut norms: Vec<f64> = Vec::new();
        let (kl, ku) = collocation_bandwidth(&domain);

        let debug_plots = std::env::var("DEBUG_PLOTS").is_ok();
        let mut iterates: Vec<(String, DVector<T>)> = Vec::new();
        if debug_plots {
            iterates.push(("initial guess".to_string(), u.clone()));
        }

        for iter in 0..=100 {
            if !running.load(Ordering::SeqCst) {
                info!("\nCtrl-C received at iter {iter} (n = {n}), using current approximation.");
                break;
            }

            let (j_mat, r) = build_j_r::<T, O>(
                ode,
                &u,
                &domain,
                Some(O::forcing),
                &O::bc_left(),
                &O::bc_right(),
                None,
            );

            let norm = r
                .iter()
                .map(|v| v.abs().to_f64().unwrap())
                .fold(0.0_f64, f64::max);
            norms.push(norm.to_f64().expect("residual norm should convert to f64"));
            info!("n = {n:>3}, iter {iter:>2}: |F| = {norm:.3e}");

            if norm < tol {
                info!("Converged for n = {n}.");
                break;
            }

            let delta = T::banded_solve(&j_mat, &(-&r), kl, ku);
            u += &delta;
            if debug_plots {
                iterates.push((format!("iter {:>2}", iter + 1), u.clone()));
            }

            let step_norm = delta
                .iter()
                .map(|v| v.abs().to_f64().unwrap())
                .fold(0.0_f64, f64::max);
            if step_norm < tol {
                info!("Newton step converged for n = {n}.");
                break;
            }
        }
        info!("{:?}", empirical_convergence_order(&norms));
        all_norms.push((format!("n = {n}"), norms.clone()));

        let (l2_err, linf_err) = total_error_norm(&u, &exact_f64, &domain, &domain_bounds, 1001);

        info!("n = {n:>3}: L2 = {l2_err:.3e}, Linf = {linf_err:.3e}");
        labels.push(format!("n = {n}"));

        if debug_plots {
            plot_iterates(
                &iterates,
                &domain,
                &exact_f64,
                &format!("Newton iterates, n = {n}"),
                &format!("{name}/newton_iterates_n{n}.html"),
                &format!("{name}_iter_n{n}"),
            );
        }

        l2_errors.push(l2_err);
        linf_errors.push(linf_err);
        all_solutions.push(u);
        all_domains.push(domain);
        plot_convergence(
            &norms,
            &format!("{name}/newton_convergence_n{n}.html"),
            &name,
        );
    }
    plot_convergence_multi(
        &all_norms,
        &format!("{name}/newton_convergence_all_degrees.html"),
        &format!("{name}_convergence"),
    );

    plot_error_vs_degree(
        degrees,
        &l2_errors,
        &linf_errors,
        theory,
        &format!("{name}/nonlinear_convergence.html"),
        name,
    );

    let solution_entries: Vec<(&str, DVector<T>, &MultiIntervalDomain<T>)> = labels
        .iter()
        .zip(all_solutions.into_iter())
        .zip(all_domains.iter())
        .map(|((label, u), domain)| (label.as_str(), u, domain))
        .collect();

    plot_solutions(
        &solution_entries,
        Some(("Exact", &exact_f64 as &dyn Fn(f64) -> f64)),
        "Solutions vs exact, across degrees",
        &format!("{name}/nonlinear_solutions.html"),
        name,
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
) where
    T: BandedLinSolve
        + num_traits::Float
        + Scalar
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
            info!("\nCtrl-C received before degree {n}, stopping h-refinement study.");
            break;
        }

        let mut l2_errors = Vec::new();
        let mut linf_errors = Vec::new();

        for &k_intervals in interval_counts {
            if !running.load(Ordering::SeqCst) {
                info!("\nCtrl-C received before K = {k_intervals} (n = {n}), stopping.");
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
                domain
                    .intervals
                    .iter()
                    .flat_map(|a| {
                        a.map_to_physical()
                            .map(|x| initial_guess(T::from_f64(x).unwrap()))
                            .iter()
                            .copied()
                            .collect::<Vec<T>>()
                    })
                    .collect(),
            );

            let (kl, ku) = collocation_bandwidth(&domain);

            for iter in 0..=100 {
                if !running.load(Ordering::SeqCst) {
                    info!("\nCtrl-C received at iter {iter} (n = {n}, K = {k_intervals}).");
                    break;
                }

                let (j_mat, r) = build_j_r::<T, O>(
                    ode,
                    &u,
                    &domain,
                    Some(O::forcing),
                    &O::bc_left(),
                    &O::bc_right(),
                    None,
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

            info!("n = {n:>3}, K = {k_intervals:>3}: L2 = {l2_err:.3e}, Linf = {linf_err:.3e}");

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

fn continuation_in_speed<G>(
    degree: usize,
    n_intervals: usize,
    domain_bounds: (f64, f64),
    s_start: f64,
    s_end: f64,
    n_steps: usize,
    delta_bar: f64,
    initial_guess: G,
    running: Arc<AtomicBool>,
) -> Option<(MultiIntervalDomain<f64>, Vec<(f64, DVector<f64>)>)>
where
    G: Fn(f64) -> f64 + Copy,
    HeteroclinicOrbit: ODE<Dual<f64>, f64>,
{
    let domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        n_intervals,
        domain_bounds,
        degree,
        IntervalDivision::Uniform,
    );

    let u0: DVector<f64> = DVector::from_vec(
        domain
            .intervals
            .iter()
            .flat_map(|interval| {
                interval
                    .map_to_physical()
                    .iter()
                    .map(|&x| initial_guess(x))
                    .collect::<Vec<f64>>()
            })
            .collect(),
    );

    let s_values = speed_grid(s_start, s_end, n_steps);

    // Uses the damped Newton solver, so a bad step stalls instead of exploding.
    let all_steps = continue_in_speed_from(&domain, u0, &s_values, delta_bar, running)?;

    plot_all_continuations(
        &domain,
        &all_steps,
        "heteroclinic_continuation_solutions.html",
    );

    Some((domain, all_steps))
}

fn heteroclinic_phase_condition<T>(
    u: &DVector<Dual<T>>,
    target: T,
    domain: &MultiIntervalDomain<T>,
) -> Dual<T>
where
    T: Scalar
        + RealField
        + FromPrimitive
        + ToPrimitive
        + Copy
        + Zero
        + Send
        + Sync
        + 'static,
{
    let interval_idx = domain
        .intervals
        .iter()
        .position(|interval| {
            let (a, b) = interval.domain;
            a <= 0.0 && 0.0 <= b
        })
        .expect("Domain must contain physical zero");

    let offset: usize = domain.intervals[..interval_idx]
        .iter()
        .map(|interval| interval.n_nodes)
        .sum();

    let interval = &domain.intervals[interval_idx];
    let x = interval.map_to_physical();
    let n = interval.n_nodes;

    // Zero is already a node.
    if let Some(j) = x.iter().position(|&xj| xj == 0.0) {
        return Dual {
            real: u[offset + j].real - target,
            eps: u[offset + j].eps,
        };
    }

    // Otherwise interpolate the current polynomial at zero.
    let mut result = Dual {
        real: -target,
        eps: T::zero(),
    };

    for j in 0..n {
        let mut weight = 1.0_f64;

        for k in 0..n {
            if k != j {
                weight *= -x[k] / (x[j] - x[k]);
            }
        }

        let weight = T::from_f64(weight).unwrap();

        result.real += weight * u[offset + j].real;
        result.eps += weight * u[offset + j].eps;
    }

    result
}

fn barycentric_interpolate(xn: &[f64], yn: &[f64], x: f64) -> f64 {
    if let Some((_, &y)) = xn.iter().zip(yn).find(|(xi, _)| (x - *xi).abs() < 1e-13) {
        return y;
    }

    let (num, den) = (0..xn.len()).fold((0.0, 0.0), |(num, den), j| {
        let w = 1.0
            / (0..xn.len())
                .filter(|&k| k != j)
                .map(|k| xn[j] - xn[k])
                .product::<f64>();

        let t = w / (x - xn[j]);
        (num + t * yn[j], den + t)
    });

    num / den
}

fn heteroclinic_degree_convergence(
    error_degrees: &[usize],
    ref_degree: usize,
    ref_intervals: usize,
    domain: (f64, f64),
    error_steps: usize,
    s_start: f64,
    s_end: f64,
    delta_bar: f64,
    initial_guess: &dyn Fn(f64) -> f64,
    theory: Option<(f64, f64)>,
    running: Arc<AtomicBool>,
) {
    // Sweeps inital high referance solution
    let result = continuation_in_speed(
        ref_degree,
        ref_intervals,
        domain,
        s_start,
        s_end,
        error_steps,
        delta_bar,
        initial_guess,
        running.clone(),
    )
    .unwrap();
    let mut low_degree_approx = Vec::new();
    for degree in error_degrees {
        let (c_domain, values) = continuation_in_speed(
            *degree,
            1,
            domain,
            s_start,
            s_end,
            error_steps,
            delta_bar,
            initial_guess,
            running.clone(),
        )
        .unwrap();

        low_degree_approx.push(
            values
                .iter()
                .map(|(_, u)| interpolate_solution_to_reference(u, &c_domain, &result.0))
                .collect::<Vec<_>>(),
        );
    }
    let mut l2_errors: Vec<Vec<f64>> = Vec::new();
    let mut linf_errors: Vec<Vec<f64>> = Vec::new();

    for approx in &low_degree_approx {
        let mut l2 = Vec::new();
        let mut linf = Vec::new();

        for (u, (_, u_ref)) in approx.iter().zip(result.1.iter()) {
            let err = u - u_ref;
            let mut l2_sq = 0.0;
            let mut off = 0;

            for int in &result.0.intervals {
                let n = int.n_nodes;
                let x = int.map_to_physical();
                let w = Chebyshev::clenshaw_curtis_weights(n);
                let j = (x.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                    - x.iter().cloned().fold(f64::INFINITY, f64::min))
                    / 2.0;

                l2_sq += (0..n).map(|i| w[i] * j * err[off + i].powi(2)).sum::<f64>();
                off += n;
            }

            l2.push(l2_sq.sqrt());
            linf.push(err.iter().map(|e| e.abs()).fold(0.0, f64::max));
        }

        l2_errors.push(l2);
        linf_errors.push(linf);
    }

    for (step, (s, _)) in result.1.iter().enumerate() {
        let valid: Vec<_> = error_degrees
            .iter()
            .zip(l2_errors.iter())
            .zip(linf_errors.iter())
            .filter(|((_, l2), linf)| step < l2.len() && step < linf.len())
            .collect();

        let degrees: Vec<usize> = valid.iter().map(|((n, _), _)| **n).collect();
        let l2: Vec<f64> = valid.iter().map(|((_, e), _)| e[step]).collect();
        let linf: Vec<f64> = valid.iter().map(|(_, e)| e[step]).collect();

        plot_error_vs_degree(
            &degrees,
            &l2,
            &linf,
            None,
            &format!("heteroclinic_error_s_{s:.6}.html"),
            "PB",
        );
    }
}

fn interpolate_solution_to_reference(
    u: &DVector<f64>,
    domain: &MultiIntervalDomain<f64>,
    ref_domain: &MultiIntervalDomain<f64>,
) -> DVector<f64> {
    let mut source = Vec::new();
    let mut offset = 0usize;

    for interval in &domain.intervals {
        let x = interval.map_to_physical();
        let n = x.len();
        let values = u.rows(offset, n).iter().copied().collect::<Vec<_>>();

        let a = x.iter().copied().fold(f64::INFINITY, f64::min);
        let b = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        source.push((a, b, x, values));
        offset += n;
    }

    let mut interpolated = Vec::new();

    for ref_interval in &ref_domain.intervals {
        for &xr in ref_interval.map_to_physical().iter() {
            let (_, _, x, values) = source
                .iter()
                .find(|(a, b, _, _)| xr >= *a - 1e-12 && xr <= *b + 1e-12)
                .expect("reference node outside source domain");

            interpolated.push(barycentric_interpolate(x.as_slice(), values, xr));
        }
    }

    DVector::from_vec(interpolated)
}

fn heteroclinic_singular_parameter_error(
    low_degree: usize,
    low_intervals: usize,
    ref_degree: usize,
    ref_intervals: usize,
    domain: (f64, f64),
    s: f64,
    delta_bar: f64,
    initial_guess: &dyn Fn(f64) -> f64,
    running: Arc<AtomicBool>,
) -> (f64, f64) {
    let (ref_domain, ref_steps) = continuation_in_speed(
        ref_degree,
        ref_intervals,
        domain,
        s,
        s,
        1,
        delta_bar,
        initial_guess,
        running.clone(),
    )
    .unwrap();

    let (_, u_ref) = ref_steps.last().unwrap();

    let (low_domain, low_steps) = continuation_in_speed(
        low_degree,
        low_intervals,
        domain,
        s,
        s,
        1,
        delta_bar,
        initial_guess,
        running,
    )
    .unwrap();

    let (_, u_low) = low_steps.last().unwrap();

    // Same interval partition as the reference, twice the degree.
    let eval_domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        ref_intervals,
        domain,
        2 * ref_degree,
        IntervalDivision::Uniform,
    );

    // Evaluate BOTH reconstructed solutions on the common fine grid.
    let low_eval =
        interpolate_solution_to_reference(u_low, &low_domain, &eval_domain);

    let ref_eval =
        interpolate_solution_to_reference(u_ref, &ref_domain, &eval_domain);

    let err = low_eval - ref_eval;

    let mut l2_sq = 0.0;
    let mut offset = 0;

    for interval in &eval_domain.intervals {
        let n = interval.n_nodes;
        let x = interval.map_to_physical();
        let w = Chebyshev::clenshaw_curtis_weights(n);

        let a = x.iter().copied().fold(f64::INFINITY, f64::min);
        let b = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let jac = (b - a) / 2.0;

        l2_sq += (0..n)
            .map(|i| w[i] * jac * err[offset + i].powi(2))
            .sum::<f64>();

        offset += n;
    }

    let l2 = l2_sq.sqrt();
    let linf = err.iter().map(|e| e.abs()).fold(0.0, f64::max);

    info!(
        "s={s:.6}, n={low_degree}, K={low_intervals}: \
         L2={l2:.3e}, Linf={linf:.3e}"
    );

    (l2, linf)
}

fn heteroclinic_singular_degree_convergence(
    degrees: &[usize],
    ref_degree: usize,
    ref_intervals: usize,
    domain: (f64, f64),
    s: f64,
    delta_bar: f64,
    initial_guess: &dyn Fn(f64) -> f64,
    running: Arc<AtomicBool>,
) {
    let mut l2 = Vec::new();
    let mut linf = Vec::new();

    for &n in degrees {
        let (e2, einf) = heteroclinic_singular_parameter_error(
            n,
            1,
            ref_degree,
            ref_intervals,
            domain,
            s,
            delta_bar,
            initial_guess,
            running.clone(),
        );

        l2.push(e2);
        linf.push(einf);
    }

    plot_error_vs_degree(
        degrees,
        &l2,
        &linf,
        None,
        &format!("singular_parameter_error_s.html"),
        "PB",
    );
}

fn total_error_norm<T>(
    u: &DVector<T>,
    exact: &dyn Fn(f64) -> f64,
    domain: &MultiIntervalDomain<T>,
    domain_bounds: &(f64, f64),
    fine_degree: usize,
) -> (f64, f64)
where
    T: FromPrimitive
        + ToPrimitive
        + Zero
        + Scalar
        + Copy
        + Sync
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + Neg<Output = T>
        + Trig
        + 'static,
{
    let u_f64: Vec<f64> = u.iter().map(|v| v.to_f64().unwrap()).collect();

    // Source pieces: (a, b, nodes, values) per interval of the solution.
    let mut source: Vec<(f64, f64, Vec<f64>, Vec<f64>)> = Vec::new();
    let mut offset = 0usize;
    for interval in &domain.intervals {
        let x: Vec<f64> = interval
            .map_to_physical()
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();
        let n = x.len();
        let a = x.iter().copied().fold(f64::INFINITY, f64::min);
        let b = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        source.push((a, b, x, u_f64[offset..offset + n].to_vec()));
        offset += n;
    }
    assert_eq!(offset, u_f64.len(), "u length doesn't match domain");

    // Fine grid with the same breakpoints, higher degree.
    let fine = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        domain.intervals.len(),
        *domain_bounds,
        fine_degree,
        IntervalDivision::Uniform,
    );

    let (mut l2_sq, mut linf) = (0.0_f64, 0.0_f64);

    for int in &fine.intervals {
        let n = int.n_nodes;
        let x = int.map_to_physical();
        let w = Chebyshev::clenshaw_curtis_weights(n);
        let a = x.iter().copied().fold(f64::INFINITY, f64::min);
        let b = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let jac = (b - a) / 2.0;

        for i in 0..n {
            let xr = x[i];
            let (_, _, nodes, vals) = source
                .iter()
                .find(|(sa, sb, _, _)| xr >= *sa - 1e-12 && xr <= *sb + 1e-12)
                .expect("fine node outside solution domain");

            let e = barycentric_interpolate(nodes, vals, xr) - exact(xr);
            l2_sq += w[i] * jac * e * e;
            linf = linf.max(e.abs());
        }
    }

    (l2_sq.sqrt(), linf)
}

pub fn plot_all_continuations(
    domain: &MultiIntervalDomain<f64>,
    steps: &[(f64, DVector<f64>)],
    filename: &str,
) {
    let labels: Vec<String> = steps.iter().map(|(s, _)| format!("s={s:.4}")).collect();

    let entries: Vec<(&str, DVector<f64>, &MultiIntervalDomain<f64>)> = steps
        .iter()
        .enumerate()
        .map(|(i, (_, u))| (labels[i].as_str(), u.clone(), domain))
        .collect();

    plot_solutions(
        &entries,
        None,
        "Heteroclinic continuation",
        filename,
        filename.trim_end_matches(".html"),
    );
}

fn speed_grid(s0: f64, s1: f64, n_steps: usize) -> Vec<f64> {
    if n_steps <= 1 {
        vec![s0]
    } else {
        (0..n_steps)
            .map(|i| s0 + (s1 - s0) * i as f64 / (n_steps - 1) as f64)
            .collect()
    }
}

fn guess_vector(
    domain: &MultiIntervalDomain<f64>,
    s: f64,
    delta_bar: f64,
    tag: Option<&str>,
) -> DVector<f64> {
    HeteroclinicOrbit::set_params(s, delta_bar);
    let u_minus = HeteroclinicOrbit::u_minus();
    let left = HeteroclinicOrbit::eigen_pair(u_minus);
    let right = HeteroclinicOrbit::eigen_pair(0.0);
    let guess = asymptotic_guess(u_minus, s, delta_bar);

    if let Some(tag) = tag {
        sanity_check(u_minus, s, delta_bar, left, right, tag.to_string());
        info!("Initial guess plot written to: {tag}.html");
    }

    DVector::from_vec(
        domain
            .intervals
            .iter()
            .flat_map(|it| {
                it.map_to_physical()
                    .iter()
                    .map(|&x| guess(x, left, right))
                    .collect::<Vec<_>>()
            })
            .collect(),
    )
}

fn is_plausible_orbit(u: &DVector<f64>, u_minus: f64) -> bool {
    let bound = 5.0 * u_minus.abs().max(1.0);

    for (i, x) in u.iter().enumerate() {
        if !x.is_finite() || x.abs() > bound {
            debug!(
                "Orbit sanity check failed:\n\
                 index = {i}\n\
                 value = {x:?}\n\
                 u_minus = {u_minus:?}\n\
                 bound = {bound:?}"
            );
            return false;
        }
    }

    debug!(
        "Orbit sanity check passed:\n\
         u_minus = {u_minus:?}\n\
         bound = {bound:?}"
    );
    true
}

fn solve_heteroclinic_step(
    u: &mut DVector<f64>,
    domain: &MultiIntervalDomain<f64>,
    s: f64,
    delta_bar: f64,
    running: Arc<AtomicBool>,
) -> Option<()>
where
    HeteroclinicOrbit: ODE<Dual<f64>, f64>,
{
    const MAX_ITERS: usize = 100;
    const TOL: f64 = 1e-9;
    const MIN_STEP: f64 = 1.0 / 65536.0;

    if !running.load(Ordering::SeqCst) {
        return None;
    }
    HeteroclinicOrbit::set_params(s, delta_bar);

    // BCs depend on the parameters, so build them after set_params.
    let bc_l = <HeteroclinicOrbit as ODE<Dual<f64>, f64>>::bc_left();
    let bc_r = <HeteroclinicOrbit as ODE<Dual<f64>, f64>>::bc_right();

    let u_seed = u.clone();
    let phase_target = HeteroclinicOrbit::u_minus() * 0.5;
    let phase = |ud: &DVector<Dual<f64>>| heteroclinic_phase_condition(ud, phase_target, domain);
    let phase_ref: &(dyn Fn(&DVector<Dual<f64>>) -> Dual<f64> + Sync) = &phase;

    let eval = |v: &DVector<f64>| {
        build_j_r::<f64, HeteroclinicOrbit>(
            &HeteroclinicOrbit,
            v,
            domain,
            Some(<HeteroclinicOrbit as ODE<Dual<f64>, f64>>::forcing),
            &bc_l,
            &bc_r,
            None,
        )
    };

    info!("solving at s = {s:.6}");
    let (mut j, mut r) = eval(u);
    for iter in 0..=MAX_ITERS {
        let rn = r.norm();
        info!("  iter {iter:>3}: |r| = {rn:.6e}");

        if !rn.is_finite() {
            info!("  residual not finite, abandoning");
            return None;
        }
        if rn < TOL {
            return Some(());
        }
        if !running.load(Ordering::SeqCst) {
            return None;
        }

        let delta = match j.clone().lu().solve(&(-&r)) {
            Some(d) => d,
            None => {
                info!("  LU solve failed (singular Jacobian)");
                return None;
            }
        };

        // Backtracking line search on |r|.
        let mut t = 1.0_f64;
        let mut accepted = false;
        while t >= MIN_STEP {
            let trial = &*u + &delta * t;
            let (jt, rt) = eval(&trial);
            let rtn = rt.norm();
            if rtn.is_finite() && rtn < (1.0 - 1e-4 * t) * rn {
                *u = trial;
                j = jt;
                r = rt;
                accepted = true;
                if t < 1.0 {
                    info!("  damped step t = {t:.4}");
                }
                break;
            }
            t *= 0.5;
        }
        if !accepted {
            info!("  line search stalled, abandoning");
            return None;
        }
        // Full Newton step, without damping.
        // *u += &delta;

        // let (j_next, r_next) = eval(u);
        // j = j_next;
        // r = r_next;
    }
    info!("  max iterations reached");
    None
}

fn continue_in_speed_from(
    domain: &MultiIntervalDomain<f64>,
    u0: DVector<f64>,
    speeds: &[f64],
    delta_bar: f64,
    running: Arc<AtomicBool>,
) -> Option<Vec<(f64, DVector<f64>)>>
where
    HeteroclinicOrbit: ODE<Dual<f64>, f64>,
{
    let mut u = u0;
    let mut out = Vec::with_capacity(speeds.len());

    for &s in speeds {
        solve_heteroclinic_step(&mut u, domain, s, delta_bar, running.clone())?;

        // set_params was called inside the step, so u_minus is for this s.
        debug!("{}", is_plausible_orbit(&u, HeteroclinicOrbit::u_minus()));
        if !is_plausible_orbit(&u, HeteroclinicOrbit::u_minus()) {
            info!("  solution at s = {s:.6} is not a plausible orbit, rejecting");
            return None;
        }
        out.push((s, u.clone()));
    }
    debug!("\nout::{:?}\n", out);
    Some(out)
}

fn sample_pair_error_on_interval(
    lower_solution: &DVector<f64>,
    lower_domain: &MultiIntervalDomain<f64>,
    upper_solution: &DVector<f64>,
    upper_domain: &MultiIntervalDomain<f64>,
    interval: (f64, f64),
    sample_number: usize,
) -> Vec<(f64, f64)> {
    if interval.1 <= interval.0 || sample_number < 2 {
        return Vec::new();
    }
    let sample_degree = sample_number - 1;

    let sample_domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        1,
        interval,
        sample_degree,
        IntervalDivision::Uniform,
    );

    let lower_sampled = interpolate_solution_to_reference(
        lower_solution,
        lower_domain,
        &sample_domain,
    );

    let upper_sampled = interpolate_solution_to_reference(
        upper_solution,
        upper_domain,
        &sample_domain,
    );

    let x = sample_domain.intervals[0].map_to_physical();

    x.iter()
        .zip(lower_sampled.iter())
        .zip(upper_sampled.iter())
        .map(|((&x, &lower), &upper)| {
            // Absolute pointwise interpolation error.
            (x, (upper - lower).abs())
        })
        .collect()
}

pub fn solve_single_approximation_h_step(
    degree: usize,
    n_intervals: usize,
    bounds: (f64, f64),
    s: f64,
    delta_bar: f64,
    h_steps: f64,
    sample_number: usize,
    running: Arc<AtomicBool>,
) -> Option<(MultiIntervalDomain<f64>, DVector<f64>)>
where
    HeteroclinicOrbit: ODE<Dual<f64>, f64>,
{
    let mut current_bounds = bounds;

    // Actual collocation solutions.
    let mut solutions: Vec<(MultiIntervalDomain<f64>, DVector<f64>)> =
        Vec::new();

    // Each entry is a separate, disconnected error trace.
    //
    // errors[0] = one error interval
    // errors[1] = another error interval
    // ...
    let mut errors: Vec<Vec<(f64, f64)>> = Vec::new();

    for n in 0..=5 {
        let domain = MultiIntervalDomain::<f64>::new(
            polynomial_trait::PolynomialKind::Chebyshev,
            n_intervals,
            current_bounds,
            degree,
            IntervalDivision::Uniform,
        );

        let mut u = guess_vector(
            &domain,
            s,
            delta_bar,
            Some(&format!("initial_guess_{s}_hstep_{n}")),
        );

        solve_heteroclinic_step(
            &mut u,
            &domain,
            s,
            delta_bar,
            running.clone(),
        )?;

        if let Some((prior_domain, prior_solution)) = solutions.last() {
            if solutions.len() == 1 {
                let mut error =
                    sample_pair_error_on_interval(
                        prior_solution,
                        prior_domain,
                        &u,
                        &domain,
                        prior_domain.domain,
                        sample_number,
                    );

                error.sort_by(|a, b| a.0.total_cmp(&b.0));

                if !error.is_empty() {
                    errors.push(error);
                }
            } else {
                let (inner_domain, _) =
                    &solutions[solutions.len() - 2];

                let prior_left = prior_domain.domain.0;
                let prior_right = prior_domain.domain.1;

                let inner_left = inner_domain.domain.0;
                let inner_right = inner_domain.domain.1;

                if prior_left < inner_left {
                    let mut left_error =
                        sample_pair_error_on_interval(
                            prior_solution,
                            prior_domain,
                            &u,
                            &domain,
                            (prior_left, inner_left),
                            sample_number,
                        );

                    left_error.retain(|(x, _)| {
                        (*x - inner_left).abs() > 1e-12
                    });

                    left_error.sort_by(|a, b| {
                        a.0.total_cmp(&b.0)
                    });

                    if !left_error.is_empty() {
                        errors.push(left_error);
                    }
                }

                if inner_right < prior_right {
                    let mut right_error =
                        sample_pair_error_on_interval(
                            prior_solution,
                            prior_domain,
                            &u,
                            &domain,
                            (inner_right, prior_right),
                            sample_number,
                        );

                    // Avoid duplicating the interface point.
                    right_error.retain(|(x, _)| {
                        (*x - inner_right).abs() > 1e-12
                    });

                    right_error.sort_by(|a, b| {
                        a.0.total_cmp(&b.0)
                    });

                    if !right_error.is_empty() {
                        errors.push(right_error);
                    }
                }
            }
        }

        solutions.push((domain, u));
        current_bounds = (
            current_bounds.0 - h_steps,
            current_bounds.1 + h_steps,
        );
    }

    let single_file = "single_collocation.html";

    solutions.reverse();

    plot_all_h_step_solutions(
        &solutions,
        &errors,
        single_file,
    );

    info!(
        "Single collocation h-step solutions written to: {single_file}"
    );
    solutions.into_iter().next()
}


pub fn solve_single_approximation(
    degree: usize,
    n_intervals: usize,
    bounds: (f64, f64),
    s: f64,
    delta_bar: f64,
    running: Arc<AtomicBool>,
) -> Option<(MultiIntervalDomain<f64>, DVector<f64>)>
where
    HeteroclinicOrbit: ODE<Dual<f64>, f64>,
{
    let domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        n_intervals,
        bounds,
        degree,
        IntervalDivision::Uniform,
    );

    let u0 = guess_vector(&domain, s, delta_bar, Some(&format!("initial_guess_{s}")));
    let steps = continue_in_speed_from(&domain, u0, &[s], delta_bar, running)?;

    let single_file = "single_collocation.html";
    plot_all_continuations(&domain, &steps, single_file);
    info!("Single collocation solution plot written to: {single_file}");

    let (_, u) = steps.into_iter().last()?;
    Some((domain, u))
}

fn heteroclinic_error_against_reference(
    u: &DVector<f64>,
    domain: &MultiIntervalDomain<f64>,
    u_ref: &DVector<f64>,
    ref_domain: &MultiIntervalDomain<f64>,
) -> (f64, f64) {
    let err = interpolate_solution_to_reference(u, domain, ref_domain) - u_ref;
    let (mut l2, mut off) = (0.0, 0);

    for int in &ref_domain.intervals {
        let n = int.n_nodes;
        let x = int.map_to_physical();
        let w = Chebyshev::clenshaw_curtis_weights(n);
        let a = x.iter().copied().fold(f64::INFINITY, f64::min);
        let b = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let jac = (b - a) / 2.0;

        l2 += (0..n)
            .map(|i| w[i] * jac * err[off + i].powi(2))
            .sum::<f64>();
        off += n;
    }

    (l2.sqrt(), err.iter().map(|e| e.abs()).fold(0.0, f64::max))
}

#[derive(Debug, Clone)]
pub struct ContinuationErrorPanel {
    pub speed: f64,
    pub degrees: Vec<usize>,
    pub l2_errors: Vec<f64>,
    pub linf_errors: Vec<f64>,
}

fn heteroclinic_continuation_interval_convergence(
    interval_counts: &[usize],
    degree: usize,
    ref_degree: usize,
    ref_intervals: usize,
    domain: (f64, f64),
    s0: f64,
    s1: f64,
    n_steps: usize,
    delta_bar: f64,
    grid_columns: usize,
    filename: &str,
    running: Arc<AtomicBool>,
) -> Option<Vec<ContinuationErrorPanel>> {
    if n_steps == 0
        || interval_counts.is_empty()
        || interval_counts.contains(&0)
        || ref_intervals == 0
    {
        warn!("Empty speed grid or invalid interval counts; aborting study.");
        return None;
    }

    let speeds = speed_grid(s0, s1, n_steps);

    let reference_domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        ref_intervals,
        domain,
        ref_degree,
        IntervalDivision::Uniform,
    );

    // Vary the interval count while keeping polynomial degree fixed.
    let low_domains: Vec<MultiIntervalDomain<f64>> = interval_counts
        .iter()
        .map(|&k| {
            MultiIntervalDomain::<f64>::new(
                polynomial_trait::PolynomialKind::Chebyshev,
                k,
                domain,
                degree,
                IntervalDivision::Uniform,
            )
        })
        .collect();

    // One common fine grid for all error measurements and solution plots.
    let evaluation_domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        ref_intervals,
        domain,
        2 * ref_degree,
        IntervalDivision::Uniform,
    );

    // Construct the asymptotic guess only once.
    let mut u_high = guess_vector(
        &reference_domain,
        s0,
        delta_bar,
        Some(&format!("initial_guess_s{s0}")),
    );

    let mut high_ref_us: Vec<(f64, DVector<f64>)> = Vec::new();
    let mut panels: Vec<ContinuationErrorPanel> = Vec::new();

    for speed in speeds {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        // First speed: asymptotic guess.
        // Later speeds: previous converged reference solution.
        let mut high_trial = u_high.clone();

        if solve_heteroclinic_step(
            &mut high_trial,
            &reference_domain,
            speed,
            delta_bar,
            running.clone(),
        )
        .is_none()
        {
            warn!("Reference continuation failed at s = {speed:.6}.");
            break;
        }

        let u_minus = HeteroclinicOrbit::u_minus();

        if !is_plausible_orbit(&high_trial, u_minus) {
            warn!("Reference sanity check failed at s = {speed:.6}.");
            break;
        }

        u_high = high_trial;
        high_ref_us.push((speed, u_high.clone()));

        let high_eval = interpolate_solution_to_reference(
            &u_high,
            &reference_domain,
            &evaluation_domain,
        );

        let mut panel = ContinuationErrorPanel {
            speed,
            // Compatibility with the existing panel type:
            // this field now records interval counts K.
            degrees: Vec::new(),
            l2_errors: Vec::new(),
            linf_errors: Vec::new(),
        };

        let mut solution_labels =
            vec![format!("Reference, n = {ref_degree}, K = {ref_intervals}")];

        let mut solution_values = vec![high_eval.clone()];

        for (&k, low_domain) in interval_counts.iter().zip(low_domains.iter()) {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            // Seed from the converged reference at the current speed.
            let mut u_low = interpolate_solution_to_reference(
                &u_high,
                &reference_domain,
                low_domain,
            );

            let (seed_l2, seed_linf) = heteroclinic_error_against_reference(
                &u_low,
                low_domain,
                &high_eval,
                &evaluation_domain,
            );

            info!(
                "s={speed:.6}, n={degree}, K={k:>3}: BEFORE Newton: \
                 L2={seed_l2:.6e}, Linf={seed_linf:.6e}"
            );

            if solve_heteroclinic_step(
                &mut u_low,
                low_domain,
                speed,
                delta_bar,
                running.clone(),
            )
            .is_none()
            {
                warn!(
                    "K = {k}, n = {degree}, s = {speed:.6}: \
                     Newton solve failed; skipping."
                );
                continue;
            }

            if !is_plausible_orbit(&u_low, u_minus) {
                warn!(
                    "K = {k}, n = {degree}, s = {speed:.6}: \
                     sanity check failed; skipping."
                );
                continue;
            }

            let (e2, einf) = heteroclinic_error_against_reference(
                &u_low,
                low_domain,
                &high_eval,
                &evaluation_domain,
            );

            info!(
                "s={speed:.6}, n={degree}, K={k:>3}: \
                 BEFORE Newton: L2={seed_l2:.6e}, Linf={seed_linf:.6e}; \
                 AFTER Newton: L2={e2:.6e}, Linf={einf:.6e}"
            );

            panel.degrees.push(k);
            panel.l2_errors.push(e2);
            panel.linf_errors.push(einf);

            let low_eval = interpolate_solution_to_reference(
                &u_low,
                low_domain,
                &evaluation_domain,
            );

            solution_labels.push(format!("n = {degree}, K = {k}"));
            solution_values.push(low_eval);
        }

        let solution_entries: Vec<(
            &str,
            DVector<f64>,
            &MultiIntervalDomain<f64>,
        )> = solution_labels
            .iter()
            .zip(solution_values.iter())
            .map(|(label, values)| {
                (label.as_str(), values.clone(), &evaluation_domain)
            })
            .collect();

        let plot_stem = format!(
            "{}_solutions_s{speed:.6}",
            filename.trim_end_matches(".html"),
        );

        plot_solutions(
            &solution_entries,
            None,
            &format!(
                "Interval refinement and reference at s = {speed:.6}, \
                 fixed degree n = {degree}"
            ),
            &format!("{plot_stem}.html"),
            &plot_stem,
        );

        if !panel.degrees.is_empty() {
            panels.push(panel);
        }

        if !running.load(Ordering::SeqCst) {
            break;
        }
    }

    if !high_ref_us.is_empty() {
        plot_all_continuations(
            &reference_domain,
            &high_ref_us,
            "high_resolution_continuation.html",
        );
    }

    if panels.is_empty() {
        warn!("No successful error comparisons.");
        return None;
    }

    // Update this plotter's axis label to "Number of intervals, K".
    plot_continuation_error_grid(&panels, grid_columns, filename);
    info!("Error plot written to: {filename}");

    Some(panels)
}

fn heteroclinic_continuation_degree_convergence(
    degrees: &[usize],
    low_intervals: usize,
    ref_degree: usize,
    ref_intervals: usize,
    domain: (f64, f64),
    s0: f64,
    s1: f64,
    n_steps: usize,
    delta_bar: f64,
    grid_columns: usize,
    filename: &str,
    running: Arc<AtomicBool>,
) -> Option<Vec<ContinuationErrorPanel>> {
    if n_steps == 0 || degrees.is_empty() {
        warn!("Empty speed or degree grid; aborting study.");
        return None;
    }

    let speeds = speed_grid(s0, s1, n_steps);

    let reference_domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        ref_intervals,
        domain,
        ref_degree,
        IntervalDivision::Uniform,
    );

    let low_domains: Vec<MultiIntervalDomain<f64>> = degrees
        .iter()
        .map(|&n| {
            MultiIntervalDomain::<f64>::new(
                polynomial_trait::PolynomialKind::Chebyshev,
                low_intervals,
                domain,
                n,
                IntervalDivision::Uniform,
            )
        })
        .collect();

    // One common fine grid for all error measurements and solution plots.
    let evaluation_domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        ref_intervals,
        domain,
        2 * ref_degree,
        IntervalDivision::Uniform,
    );

    // Construct the asymptotic guess only once.
    let mut u_high = guess_vector(
        &reference_domain,
        s0,
        delta_bar,
        Some(&format!("initial_guess_s{s0}")),
    );

    let mut high_ref_us: Vec<(f64, DVector<f64>)> = Vec::new();
    let mut panels: Vec<ContinuationErrorPanel> = Vec::new();

    for speed in speeds {
        if !running.load(Ordering::SeqCst) {
            break;
        }

        let mut u_high = guess_vector(
            &reference_domain,
            speed,
            delta_bar,
            Some(&format!("initial_guess_s{speed}")),
        );

        // First speed: asymptotic guess.
        // Later speeds: previous converged reference solution.
        let mut high_trial = u_high.clone();

        if solve_heteroclinic_step(
            &mut high_trial,
            &reference_domain,
            speed,
            delta_bar,
            running.clone(),
        )
        .is_none()
        {
            warn!("Reference continuation failed at s = {speed:.6}.");
            break;
        }

        let u_minus = HeteroclinicOrbit::u_minus();

        if !is_plausible_orbit(&high_trial, u_minus) {
            warn!("Reference sanity check failed at s = {speed:.6}.");
            break;
        }

        u_high = high_trial;
        high_ref_us.push((speed, u_high.clone()));

        // Reference values on the common evaluation grid.
        let high_eval = interpolate_solution_to_reference(
            &u_high,
            &reference_domain,
            &evaluation_domain,
        );

        let mut panel = ContinuationErrorPanel {
            speed,
            degrees: Vec::new(),
            l2_errors: Vec::new(),
            linf_errors: Vec::new(),
        };

        let mut solution_labels =
            vec![format!("Reference, n = {ref_degree}, K = {ref_intervals}")];

        let mut solution_values = vec![high_eval.clone()];

        for (&n, low_domain) in degrees.iter().zip(low_domains.iter()) {
            if !running.load(Ordering::SeqCst) {
                break;
            }

            // Start from the CONVERGED reference at the CURRENT speed.
            let mut u_low = interpolate_solution_to_reference(
                &u_high,
                &reference_domain,
                low_domain,
            );

            // BEFORE Newton: error from representing the reference
            // on the lower-degree grid.
            let (seed_l2, seed_linf) = heteroclinic_error_against_reference(
                &u_low,
                low_domain,
                &high_eval,
                &evaluation_domain,
            );

            info!(
                "s={speed:.6}, n={n:>3}: BEFORE Newton: \
                 L2={seed_l2:.6e}, Linf={seed_linf:.6e}"
            );

            if solve_heteroclinic_step(
                &mut u_low,
                low_domain,
                speed,
                delta_bar,
                running.clone(),
            )
            .is_none()
            {
                warn!("n = {n}, s = {speed:.6}: Newton solve failed; skipping.");
                continue;
            }

            if !is_plausible_orbit(&u_low, u_minus) {
                warn!("n = {n}, s = {speed:.6}: sanity check failed; skipping.");
                continue;
            }

            // AFTER Newton: use exactly the same reference and evaluation grid.
            // The helper already returns sqrt(integral(error^2)).
            let (e2, einf) = heteroclinic_error_against_reference(
                &u_low,
                low_domain,
                &high_eval,
                &evaluation_domain,
            );

            info!(
                "s={speed:.6}, n={n:>3}: \
                 BEFORE Newton: L2={seed_l2:.6e}, Linf={seed_linf:.6e}; \
                 AFTER Newton: L2={e2:.6e}, Linf={einf:.6e}\n\n\n"
            );

            panel.degrees.push(n);
            panel.l2_errors.push(e2);
            panel.linf_errors.push(einf);

            // Evaluate the converged low-degree solution for plotting.
            let low_eval = interpolate_solution_to_reference(
                &u_low,
                low_domain,
                &evaluation_domain,
            );

            solution_labels.push(format!("n = {n}, K = {low_intervals}"));
            solution_values.push(low_eval);
        }

        let solution_entries: Vec<(
            &str,
            DVector<f64>,
            &MultiIntervalDomain<f64>,
        )> = solution_labels
            .iter()
            .zip(solution_values.iter())
            .map(|(label, values)| {
                (label.as_str(), values.clone(), &evaluation_domain)
            })
            .collect();

        let plot_stem = format!(
            "{}_solutions_s{speed:.6}",
            filename.trim_end_matches(".html"),
        );

        plot_solutions(
            &solution_entries,
            None,
            &format!("All polynomial degrees and reference at s = {speed:.6}"),
            &format!("{plot_stem}.html"),
            &plot_stem,
        );

        if !panel.degrees.is_empty() {
            panels.push(panel);
        }

        if !running.load(Ordering::SeqCst) {
            break;
        }
    }

    if !high_ref_us.is_empty() {
        plot_all_continuations(
            &reference_domain,
            &high_ref_us,
            "high_resolution_continuation.html",
        );
    }

    if panels.is_empty() {
        warn!("No successful error comparisons.");
        return None;
    }

    plot_continuation_error_grid(&panels, grid_columns, filename);
    info!("Error plot written to: {filename}");

    Some(panels)
}

fn continue_in_delta_bar(
    domain: &MultiIntervalDomain<f64>,
    s: f64,
    delta_start: f64,
    delta_end: f64,
    max_step: f64,
    min_step: f64,
    running: Arc<AtomicBool>,
) -> Option<Vec<(f64, DVector<f64>)>>
where
    HeteroclinicOrbit: ODE<Dual<f64>, f64>,
{
    let mut u = guess_vector(
        domain,
        s,
        delta_start,
        Some(&format!("initial_guess_s{s}_delta{delta_start}")),
    );

    solve_heteroclinic_step(&mut u, domain, s, delta_start, running.clone())?;
    if !is_plausible_orbit(&u, HeteroclinicOrbit::u_minus()) {
        info!("Starting solution at delta_bar = {delta_start} is not a plausible orbit.");
        return None;
    }

    let mut out = vec![(delta_start, u.clone())];
    let mut delta = delta_start;
    let mut h = max_step;

    while delta < delta_end - 1e-12 {
        if !running.load(Ordering::SeqCst) {
            info!("Interrupted at delta_bar = {delta:.4}.");
            break;
        }

        let target = (delta + h).min(delta_end);
        let mut trial = u.clone();

        let ok = solve_heteroclinic_step(&mut trial, domain, s, target, running.clone()).is_some()
            && is_plausible_orbit(&trial, HeteroclinicOrbit::u_minus());

        if ok {
            u = trial;
            delta = target;
            out.push((delta, u.clone()));
            h = (h * 1.5).min(max_step);
        } else {
            h *= 0.5;
            info!("  step failed, reducing to h = {h:.4e}");
            if h < min_step {
                info!("Step size below {min_step:.1e}; stopping at delta_bar = {delta:.4}.");
                break;
            }
        }
    }

    Some(out)
}

fn delta_sweep_3d(
    degree: usize,
    n_intervals: usize,
    bounds: (f64, f64),
    s: f64,
    delta_start: f64,
    delta_end: f64,
    n_points: usize,
    filename: &str,
    running: Arc<AtomicBool>,
) -> Option<(MultiIntervalDomain<f64>, Vec<(f64, DVector<f64>)>)> {
    let domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        n_intervals,
        bounds,
        degree,
        IntervalDivision::Uniform,
    );

    let max_step = (delta_end - delta_start) / (n_points.max(2) - 1) as f64;
    let min_step = max_step / 1024.0;

    let steps = continue_in_delta_bar(
        &domain,
        s,
        delta_start,
        delta_end,
        max_step,
        min_step,
        running,
    )?;

    info!(
        "delta sweep: {} solutions, delta_bar {:.4} -> {:.4}",
        steps.len(),
        steps.first().map(|(d, _)| *d).unwrap_or(delta_start),
        steps.last().map(|(d, _)| *d).unwrap_or(delta_start),
    );

    if steps.len() >= 2 {
        plot_delta_surface(&domain, &steps, s, filename);
        println!("3D surface written to: {filename}");
    } else {
        info!("Fewer than 2 solutions; no surface drawn.");
    }

    Some((domain, steps))
}

/// Surface of u(ζ) against a continuation parameter (y axis).
/// `steps` holds (parameter value, solution) pairs, in continuation order.
fn plot_parameter_surface(
    domain: &MultiIntervalDomain<f64>,
    steps: &[(f64, DVector<f64>)],
    param_label: &str,
    title: String,
    filename: &str,
) {
    let x_all: Vec<f64> = domain
        .intervals
        .iter()
        .flat_map(|it| it.map_to_physical().iter().copied().collect::<Vec<f64>>())
        .collect();

    let mut order: Vec<usize> = (0..x_all.len()).collect();
    order.sort_by(|&a, &b| x_all[a].partial_cmp(&x_all[b]).unwrap());

    // Drop duplicated interface nodes.
    let mut keep: Vec<usize> = Vec::with_capacity(order.len());
    for i in order {
        match keep.last() {
            Some(&last) if (x_all[i] - x_all[last]).abs() < 1e-10 => {}
            _ => keep.push(i),
        }
    }

    let x: Vec<f64> = keep.iter().map(|&i| x_all[i]).collect();
    let y: Vec<f64> = steps.iter().map(|(p, _)| *p).collect();
    let z: Vec<Vec<f64>> = steps
        .iter()
        .map(|(_, u)| keep.iter().map(|&i| u[i]).collect())
        .collect();

    let surface = plotly::Surface::new(z).x(x).y(y);

    let mut plot = Plot::new();
    plot.add_trace(surface);
    plot.set_layout(
        Layout::new()
            .title(title)
            .scene(
                plotly::layout::LayoutScene::new()
                    .x_axis(Axis::new().title("ζ"))
                    .y_axis(Axis::new().title(param_label))
                    .z_axis(Axis::new().title("u")),
            )
            .height(750)
            .auto_size(true),
    );
    plot.set_configuration(Configuration::new().responsive(true));
    plot.write_html(filename);
}

fn plot_delta_surface(
    domain: &MultiIntervalDomain<f64>,
    steps: &[(f64, DVector<f64>)],
    s: f64,
    filename: &str,
) {
    plot_parameter_surface(
        domain,
        steps,
        "δ",
        format!("Heteroclinic orbit u(ζ), at s = {s}"),
        filename,
    );
}

fn plot_speed_surface(
    domain: &MultiIntervalDomain<f64>,
    steps: &[(f64, DVector<f64>)],
    delta_bar: f64,
    filename: &str,
) {
    plot_parameter_surface(
        domain,
        steps,
        "s",
        format!("Heteroclinic orbit u(ζ), at δ = {delta_bar}"),
        filename,
    );
}

fn continue_in_speed_adaptive(
    domain: &MultiIntervalDomain<f64>,
    delta_bar: f64,
    s_start: f64,
    s_end: f64,
    max_step: f64,
    min_step: f64,
    running: Arc<AtomicBool>,
) -> Option<Vec<(f64, DVector<f64>)>>
where
    HeteroclinicOrbit: ODE<Dual<f64>, f64>,
{
    let mut u = guess_vector(
        domain,
        s_start,
        delta_bar,
        Some(&format!("initial_guess_s{s_start}_delta{delta_bar}")),
    );

    solve_heteroclinic_step(&mut u, domain, s_start, delta_bar, running.clone())?;
    if !is_plausible_orbit(&u, HeteroclinicOrbit::u_minus()) {
        info!("Starting solution at s = {s_start} is not a plausible orbit.");
        return None;
    }

    let dir = if s_end >= s_start { 1.0 } else { -1.0 };
    let mut out = vec![(s_start, u.clone())];
    let mut s = s_start;
    let mut h = max_step;

    while (s_end - s) * dir > 1e-12 {
        if !running.load(Ordering::SeqCst) {
            info!("Interrupted at s = {s:.4}.");
            break;
        }

        let target = if dir > 0.0 {
            (s + h).min(s_end)
        } else {
            (s - h).max(s_end)
        };
        let mut trial = u.clone();

        let ok = solve_heteroclinic_step(&mut trial, domain, target, delta_bar, running.clone())
            .is_some()
            && is_plausible_orbit(&trial, HeteroclinicOrbit::u_minus());

        if ok {
            u = trial;
            s = target;
            out.push((s, u.clone()));
            h = (h * 1.5).min(max_step);
        } else {
            h *= 0.5;
            info!("  step failed, reducing to h = {h:.4e}");
            if h < min_step {
                info!("Step size below {min_step:.1e}; stopping at s = {s:.4}.");
                break;
            }
        }
    }

    Some(out)
}

fn speed_sweep_3d(
    degree: usize,
    n_intervals: usize,
    bounds: (f64, f64),
    s_start: f64,
    s_end: f64,
    delta_bar: f64,
    n_points: usize,
    filename: &str,
    running: Arc<AtomicBool>,
) -> Option<(MultiIntervalDomain<f64>, Vec<(f64, DVector<f64>)>)> {
    let domain = MultiIntervalDomain::<f64>::new(
        polynomial_trait::PolynomialKind::Chebyshev,
        n_intervals,
        bounds,
        degree,
        IntervalDivision::Uniform,
    );

    let max_step = (s_end - s_start).abs() / (n_points.max(2) - 1) as f64;
    let min_step = max_step / 1024.0;

    let steps = continue_in_speed_adaptive(
        &domain, delta_bar, s_start, s_end, max_step, min_step, running,
    )?;

    info!(
        "speed sweep: {} solutions, s {:.4} -> {:.4}",
        steps.len(),
        steps.first().map(|(s, _)| *s).unwrap_or(s_start),
        steps.last().map(|(s, _)| *s).unwrap_or(s_start),
    );

    if steps.len() >= 2 {
        plot_speed_surface(&domain, &steps, delta_bar, filename);
        println!("3D speed surface written to: {filename}");
    } else {
        info!("Fewer than 2 solutions; no surface drawn.");
    }

    Some((domain, steps))
}
