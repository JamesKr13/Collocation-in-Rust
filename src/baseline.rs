use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::toolbox::test_odes::{
    CubicReactionManufacturedODE,
    KdvSolitonODE,
    ManufacturedSolutionODE,
    PendulumODE,
    SineBaselineODE,
};
use crate::nonlinear_solve_convergence;

const TEST_INTERVALS: usize = 1;

pub fn run_pendulum(degrees: &[usize], running: Arc<AtomicBool>) {
    let exact = |x: f64| 0.5 * (x - 1.0) * x.sin();
    let initial_guess = |_x: f64| 0.0;
    let name = "Pendulm_ODE";

    nonlinear_solve_convergence::<f64, PendulumODE, _, _>(
        &PendulumODE,
        exact,
        initial_guess,
        degrees,
        (0.0, 1.0),
        TEST_INTERVALS,
        None,
        running,
        name,
    );
}

pub fn run_sine_baseline(degrees: &[usize], running: Arc<AtomicBool>) {
    let exact = |x: f64| x.sin();
    let initial_guess = |_x: f64| 0.0;
    let name = "SineBaselineODE";

    nonlinear_solve_convergence::<f64, SineBaselineODE, _, _>(
        &SineBaselineODE,
        exact,
        initial_guess,
        degrees,
        (0.0, 1.0),
        TEST_INTERVALS,
        None,
        running,
        name,
    );
}

pub fn run_manufactured(degrees: &[usize], running: Arc<AtomicBool>) {
    let beta = ManufacturedSolutionODE::BETA;
    let exact = move |x: f64| 1.0 / (1.0 + beta * x * x);
    let initial_guess = |_x: f64| 0.0;

    let b = 1.0 / beta.sqrt();
    let rho_star = b + (1.0 + 1.0/beta).sqrt();
    let name = "ManufacturedSolutionODE";

    nonlinear_solve_convergence::<f64, ManufacturedSolutionODE, _, _>(
        &ManufacturedSolutionODE,
        exact,
        initial_guess,
        degrees,
        (-1.0, 1.0),
        TEST_INTERVALS,
        Some((rho_star, 1.0)),
        running,
        name,
    );
}

pub fn run_cubic_manufactured(degrees: &[usize], running: Arc<AtomicBool>) {
    let beta = CubicReactionManufacturedODE::BETA;

    let exact = move |x: f64| {
        (std::f64::consts::PI * x).sin() / (1.0 + beta * x * x)
    };

    let initial_guess = move |x: f64| {
        0.2 * (std::f64::consts::PI * x).sin() / (1.0 + beta * x * x)
    };

    let b = 1.0 / beta.sqrt();
    let rho_star = b + (1.0 + b * b).sqrt();
    let name = "CubicReactionManufacturedODE";

    nonlinear_solve_convergence::<f64, CubicReactionManufacturedODE, _, _>(
        &CubicReactionManufacturedODE,
        exact,
        initial_guess,
        degrees,
        (-1.0, 1.0),
        TEST_INTERVALS,
        Some((rho_star, 1.0)),
        running,
        name,
    );
}

pub fn run_kdv(degrees: &[usize], running: Arc<AtomicBool>) {
    let c = KdvSolitonODE::SPEED;
    let lambda = c.sqrt();
    let l_domain = 10.0;

    let exact = move |x: f64| {
        c / (0.5 * lambda * x).cosh().powi(2)
    };

    let initial_guess = move |x: f64| {
        3.0 * c / (0.5 * lambda * x).cosh().powi(2)
    };

    let b = std::f64::consts::PI / (lambda * l_domain);
    let z_star = std::f64::consts::PI /(l_domain*KdvSolitonODE::SPEED.sqrt());
    let rho_star = (z_star + (z_star.powi(2)+1.0).sqrt()).abs();
    let name = "KdvSolitonODE";

    nonlinear_solve_convergence::<f64, KdvSolitonODE, _, _>(
        &KdvSolitonODE,
        exact,
        initial_guess,
        degrees,
        (-l_domain, l_domain),
        TEST_INTERVALS,
        Some((rho_star, 1.0)),
        running,
        name,
    );
}


