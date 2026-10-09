use nalgebra::{DMatrix, DVector, Dyn, RealField, Scalar, zero};
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use plotly::contour::Operation;
use plotly::layout::ThetaUnit::Degrees;
use std::f32::consts::PI;
use std::fmt::{LowerExp};
use std::iter::Product;
use std::ops::{Add, AddAssign, Div, Index, Mul, Neg, Sub};
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
use toolbox::grapher::{plot_continuation_error_grid, plot_solutions, plot_convergence, plot_error_vs_degree, plot_convergence_multi, plot_error_vs_intervals_multi};

mod baseline;
use baseline::{run_sine_baseline, run_cubic_manufactured, run_kdv, run_manufactured, run_pendulum};
use crate::h_interval::multi_interval;
use crate::h_interval::multi_interval::IntervalDivision::ChebyshevInterval;
use crate::toolbox::test_odes::{KdvSolitonODE, SineBaselineODE, ManufacturedSolutionODE, CubicReactionManufacturedODE};
use crate::toolbox::trig_trait::Trig;
use crate::toolbox::pb::{EigenPair, HeteroclinicOrbit};
use crate::toolbox::dual_matrix_operations::Operations;
use crate::toolbox::asymptotic_guess::{asymptotic_guess, sanity_check};

use crate::toolbox::asymptotic_guess;

fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let running_handler = running.clone();

    ctrlc::set_handler(move || {
        running_handler.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    let mut degrees: Vec<usize> = (5..=143).step_by(4).collect();
    println!("{:.?}", degrees);

    // run_pendulum(&degrees, running.clone());
    // run_sine_baseline(&degrees, running.clone());
    // run_manufactured(&degrees, running.clone());
    // run_cubic_manufactured(&degrees, running.clone());
    // run_kdv(&degrees, running.clone());
    let values: Vec<usize> = (3..=6)
        .step_by(2)
        .filter(|x| x % 2 == 1)
        .collect();

    let s0=1.21;
    let delta_bar=1.8;

    HeteroclinicOrbit::set_params(s0,delta_bar);

    let u_minus=HeteroclinicOrbit::u_minus();
    let left_eigen_pair=HeteroclinicOrbit::eigen_pair(u_minus);
    let right_eigen_pair=HeteroclinicOrbit::eigen_pair(0.0);

    sanity_check(
        u_minus,
        s0,
        delta_bar,
        left_eigen_pair,
        right_eigen_pair,
        "initial_guess".into(),
    );
    
    heteroclinic_continuation_degree_convergence(
        &values,1,
        13,33,
        (-30.0,10.0),
        1.6,1.6,0,
        1.8,
        4,
        "heteroclinic_continuation_errors.html",
        running.clone()
    );
}
