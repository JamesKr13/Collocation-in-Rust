use nalgebra::{DMatrix, DVector,};

use plotly::{Plot, Scatter};
use plotly::common::{Mode, Line, Marker, MarkerSymbol};
use plotly::layout::{Layout, Axis};

mod dual;
use dual::Dual;

mod multi_interval_domain;
use multi_interval_domain::MultiIntervalDomain;

mod automatic_derivative;

mod boundary_conditions;
use boundary_conditions::BoundaryCondition;
mod matrix_op;
extern crate lapack_src;
extern crate openblas_src;
use lapack::dgbsv;
mod test_odes;
use test_odes::*;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

use crate::automatic_derivative::ODE;


fn plot_solution(u: &DVector<f64>, domain: &MultiIntervalDomain, title: &str, filename: &str) {
    let n_intervals = domain.interval.len();
    let n_per = domain.interval[0].collocation_points().len();

    let mut x_vals: Vec<f64> = Vec::new();
    let mut u_vals: Vec<f64> = Vec::new();

    for i in 0..n_intervals {
        let physical_pts = domain.interval[i].map_to_physical();
        let start = if i == 0 { 0 } else { 1 };
        for local_row in start..n_per {
            let global_idx = i * n_per + local_row;
            x_vals.push(physical_pts[local_row]);
            u_vals.push(u[global_idx]);
        }
    }

    let solution_line = Scatter::new(x_vals.clone(), u_vals.clone())
        .name("u(x)")
        .mode(Mode::Lines)
        .line(Line::new().color("steelblue").width(2.5));

    let nodes = Scatter::new(x_vals.clone(), u_vals.clone())
        .name("collocation nodes")
        .mode(Mode::Markers)
        .marker(
            Marker::new()
                .size(8)
                .color("steelblue")
                .symbol(MarkerSymbol::CircleOpen)
        );

    let mut breakpoint_shapes = vec![];
    for &bp in domain.breakpoints.iter().skip(1).take(domain.breakpoints.len() - 2) {
        breakpoint_shapes.push(
            plotly::layout::Shape::new()
                .shape_type(plotly::layout::ShapeType::Line)
                .x0(bp).x1(bp)
                .y0(0.0).y1(1.0)
                .y_ref("paper")
                .line(plotly::layout::ShapeLine::new()
                    .color("rgba(150,150,150,0.5)")
                    .width(1.0)
                    .dash(plotly::common::DashType::Dash)),
        );
    }

    let layout = Layout::new()
        .title(plotly::common::Title::new().text(title))
        .x_axis(Axis::new().title(plotly::common::Title::new().text("x")))
        .y_axis(Axis::new().title(plotly::common::Title::new().text("u(x)")))
        .shapes(breakpoint_shapes)
        .show_legend(true);

    let mut plot = Plot::new();
    plot.add_trace(solution_line);
    plot.add_trace(nodes);
    plot.set_layout(layout);
    plot.write_html(filename);
    println!("Saved plot to {filename}");
}

fn banded_solve(J: &DMatrix<f64>, R: &DVector<f64>, kl: usize, ku:usize) -> DVector<f64> {
    let n = R.len();
    let ldab = 2*kl + ku +1;
    let mut ab = vec![0.0f64; ldab * n];

    for j in 0..n {
        for i in (j.saturating_sub(ku))..=(j+kl).min(n-1) {
            let row = kl + ku + i -j;
            ab[row + j * ldab] = J[(i,j)];
        }
    }

    let mut ipiv = vec![0i32; n];
    let mut b = R.as_slice().to_vec();
    let mut info = 0i32;
    unsafe {
        dgbsv(n as i32, kl as i32, ku as i32,1,&mut ab, ldab as i32, &mut ipiv, &mut b,n as i32,&mut info);
    }
    let mut actual_kl = 0usize;
    let mut actual_ku = 0usize;
    for i in 0..n {
        for j in 0..n {
            if J[(i,j)].abs() > 1e-10 {
                if i > j { actual_kl = actual_kl.max(i - j); }
                if j > i { actual_ku = actual_ku.max(j - i); }
            }
        }
    }
    println!("actual kl={actual_kl} ku={actual_ku}");
    assert_eq!(info, 0, "dgbsv failed with info={}", info);
    DVector::from_vec(b)
}


fn main() {
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    iteration_loop(&NonlinearDiffusionODE, running); 
}

fn iteration_loop<O>(ode: &O, running: Arc<AtomicBool>) where O: ODE<Dual<f64>> + Sync{
    let (a, b) = (-1.0_f64, 1.0_f64);
    let domain = MultiIntervalDomain::new(8, a, b, 16);
    let n_total = domain.interval.len() * domain.get_interval(0).collocation_points().len();
    let n_per = domain.get_interval(0).collocation_points().len();
    let pi = std::f64::consts::PI;

    let mut u = DVector::from_vec(vec![1.0; n_total]);
    for iter in 0..=50 {
        if !running.load(Ordering::SeqCst) {
            println!("\nCtrl-C received at iter {iter}, plotting current approximation...");
            break;
        }

        let (J, R) = automatic_derivative::build_J_R(
            ode, &u, &domain,
            Some(O::forcing),
            O::bc_left(),
            O::bc_right(),
        );
        let norm = R.norm();
        println!("Iter {iter:>2}: |F| = {norm:.3e}");
        if norm < 1e-10 { println!("Converged."); break; }
        // let delta = J.lu().solve(&(-&R)).expect("singular");
        let kl = n_per;
        let ku = n_per;
        let delta = banded_solve(&J, &(-&R), kl, ku);
        u += delta;
    }
    plot_solution(&u, &domain, "Burgers ODE solution", "solution.html");
}

