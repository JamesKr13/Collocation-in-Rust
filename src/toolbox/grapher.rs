// grapher.rs
use nalgebra::{DVector, Scalar};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use plotly::{Plot, Scatter};
use plotly::common::{Mode, Line, Marker, MarkerSymbol, DashType};
use plotly::layout::{Layout, Axis, AxisType};

use crate::h_interval::multi_interval::MultiIntervalDomain;
use super::trig_trait::Trig;

/// Generic barycentric weights for an arbitrary node set, computed
/// directly from the node positions via the product formula
/// w_j = 1 / prod_{k != j} (x_j - x_k). Works for any nodes (Chebyshev
/// Gauss or Gauss-Lobatto), since it makes no assumption about which
/// convention produced them.
fn barycentric_weights(nodes: &[f64]) -> Vec<f64> {
    let n = nodes.len();
    let mut w = vec![1.0; n];
    for j in 0..n {
        for k in 0..n {
            if k != j {
                w[j] *= nodes[j] - nodes[k];
            }
        }
        w[j] = 1.0 / w[j];
    }
    w
}

/// Evaluate the degree-(n-1) interpolant through (nodes[j], values[j])
/// at the point x, via the barycentric formula. Falls back to an exact
/// node value when x coincides (to machine precision) with a node, to
/// avoid division by zero.
fn barycentric_eval(nodes: &[f64], weights: &[f64], values: &[f64], x: f64) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for j in 0..nodes.len() {
        let diff = x - nodes[j];
        if diff.abs() < 1e-14 {
            return values[j];
        }
        let t = weights[j] / diff;
        num += t * values[j];
        den += t;
    }
    num / den
}

/// Per-interval node positions and precomputed barycentric weights,
/// plus the index range this interval occupies in the flat solution
/// vector (matching the flat_map ordering used to build `u` in main.rs).
struct IntervalInfo {
    nodes: Vec<f64>,
    weights: Vec<f64>,
    start: usize,
    end: usize,
}

pub fn plot_solutions<T>(
    solutions: &[(&str, DVector<T>, &MultiIntervalDomain<T>)],
    exact: Option<(&str, &dyn Fn(f64) -> f64)>,
    title: &str,
    filename: &str,
) where
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
        + Trig
        + 'static,
{
    const SAMPLES_PER_INTERVAL: usize = 200;

    let colors = [
        "steelblue", "firebrick", "seagreen", "darkorange",
        "purple", "teal", "goldenrod", "slategray",
    ];

    let mut plot = Plot::new();

    // Track the overall x-range across every solution's own domain so the
    // optional exact-solution curve can be sampled over the full extent.
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;

    for (idx, (label, u, domain)) in solutions.iter().enumerate() {
        // Build node/weight data fresh for THIS solution's domain — degrees
        // or interval counts may differ between entries.
        let mut interval_infos = Vec::new();
        let mut offset = 0usize;
        for interval in domain.intervals.iter() {
            let physical_pts = interval.map_to_physical();
            let nodes: Vec<f64> = physical_pts
                .iter()
                .map(|xi| xi.to_f64().expect("failed to convert node to f64"))
                .collect();
            let n = nodes.len();
            let weights = barycentric_weights(&nodes);
            interval_infos.push(IntervalInfo {
                nodes,
                weights,
                start: offset,
                end: offset + n,
            });
            offset += n;
        }

        let u_vals: Vec<f64> = u
            .iter()
            .map(|v| v.to_f64().expect("failed to convert value to f64"))
            .collect();

        assert_eq!(
            offset,
            u_vals.len(),
            "solution length doesn't match its domain's grid length for '{label}'"
        );

        let color = colors[idx % colors.len()];

        let mut fine_x = Vec::new();
        let mut fine_u = Vec::new();
        for info in interval_infos.iter() {
            let local_vals = &u_vals[info.start..info.end];
            let a = info.nodes.iter().cloned().fold(f64::INFINITY, f64::min);
            let b = info.nodes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            x_min = x_min.min(a);
            x_max = x_max.max(b);

            for s in 0..SAMPLES_PER_INTERVAL {
                let t = s as f64 / (SAMPLES_PER_INTERVAL - 1) as f64;
                let x = a + t * (b - a);
                let ux = barycentric_eval(&info.nodes, &info.weights, local_vals, x);
                fine_x.push(x);
                fine_u.push(ux);
            }
        }

        let line = Scatter::new(fine_x, fine_u)
            .name(*label)
            .mode(Mode::Lines)
            .line(Line::new().color(color).width(2.5));

        let node_x: Vec<f64> = interval_infos
            .iter()
            .flat_map(|info| info.nodes.clone())
            .collect();
        let nodes_trace = Scatter::new(node_x, u_vals.clone())
            .mode(Mode::Markers)
            .marker(
                Marker::new()
                    .size(6)
                    .color(color)
                    .symbol(MarkerSymbol::CircleOpen),
            )
            .show_legend(false);

        plot.add_trace(line);
        plot.add_trace(nodes_trace);
    }

    // Overlay the true/exact solution as a smooth dashed reference curve,
    // sampled directly (no barycentric interpolation needed — we have the
    // closed form).
    if let Some((label, f)) = exact {
        const EXACT_SAMPLES: usize = 400;
        let mut ex_x = Vec::with_capacity(EXACT_SAMPLES);
        let mut ex_u = Vec::with_capacity(EXACT_SAMPLES);
        for s in 0..EXACT_SAMPLES {
            let t = s as f64 / (EXACT_SAMPLES - 1) as f64;
            let x = x_min + t * (x_max - x_min);
            ex_x.push(x);
            ex_u.push(f(x));
        }
        let exact_line = Scatter::new(ex_x, ex_u)
            .name(label)
            .mode(Mode::Lines)
            .line(
                Line::new()
                    .color("black")
                    .width(2.0)
                    .dash(plotly::common::DashType::Dash),
            );
        plot.add_trace(exact_line);
    }

    let layout = Layout::new().width(800)
        .height(600)
        .title(plotly::common::Title::new().text(title))
        .x_axis(Axis::new().title(plotly::common::Title::new().text("x")))
        .y_axis(Axis::new().title(plotly::common::Title::new().text("u(x)")))
        .show_legend(true);

    plot.set_layout(layout);
    plot.write_html(filename);
    println!("Saved plot to {filename}");
}

/// Plot residual norm vs iteration on a log-y scale to visualize
/// Newton convergence.
pub fn plot_convergence(norms: &[f64], filename: &str) {
    let iters: Vec<f64> = (0..norms.len()).map(|i| i as f64).collect();

    let trace = Scatter::new(iters, norms.to_vec())
        .name("|R|")
        .mode(Mode::LinesMarkers)
        .line(Line::new().color("crimson").width(2.0));

    let mut plot = Plot::new();
    plot.add_trace(trace);

    let layout = Layout::new()
        .title(plotly::common::Title::new().text("Newton convergence"))
        .x_axis(Axis::new().title(plotly::common::Title::new().text("iteration")))
        .y_axis(
            Axis::new()
                .title(plotly::common::Title::new().text("residual norm"))
                .type_(plotly::layout::AxisType::Log),
        );

    plot.set_layout(layout);
    plot.write_html(filename);
    println!("Saved plot to {filename}");
}

/// Plot L2/L-infinity discretization error vs. polynomial degree n,
/// on a log-y scale, for verification against Jackson's/Bernstein's
/// predicted convergence rates (Section 5.1).
/// Plot L2/L-infinity discretization error vs. polynomial degree n,
/// on a log-y scale, optionally overlaid with the theoretical
/// convergence rate predicted by Bernstein's theorem (Theorem 2.1.7),
/// for direct visual verification against theory.
pub fn plot_error_vs_degree(
    degrees: &[usize],
    l2_errors: &[f64],
    linf_errors: &[f64],
    theory: Option<(f64, f64)>, // (rho_star, leading constant C)
    filename: &str,
) {
    let n_vals: Vec<f64> = degrees.iter().map(|&n| n as f64).collect();

    let l2_trace = Scatter::new(n_vals.clone(), l2_errors.to_vec())
        .name("L2 error (observed)")
        .mode(Mode::LinesMarkers)
        .line(Line::new().color("steelblue").width(2.0));

    let linf_trace = Scatter::new(n_vals.clone(), linf_errors.to_vec())
        .name("L-infinity error (observed)")
        .mode(Mode::LinesMarkers)
        .line(Line::new().color("firebrick").width(2.0));

    let mut plot = Plot::new();
    plot.add_trace(l2_trace);
    plot.add_trace(linf_trace);

    if let Some((rho_star, c)) = theory {
        let theory_vals: Vec<f64> = degrees.iter()
            .map(|&n| c * rho_star.powi(-(n as i32)))
            .collect();
        let theory_trace = Scatter::new(n_vals, theory_vals)
            .name(format!("Theory: O(ρ*^-n), ρ* = {rho_star:.3}"))
            .mode(Mode::Lines)
            .line(Line::new().color("gray").width(1.5).dash(plotly::common::DashType::Dash));
        plot.add_trace(theory_trace);
    }

    let layout = Layout::new().width(800)
        .height(600)
        .title(plotly::common::Title::new().text("Discretization error vs. polynomial degree"))
        .x_axis(Axis::new().title(plotly::common::Title::new().text("n")))
        .y_axis(
            Axis::new()
                .title(plotly::common::Title::new().text("error"))
                .type_(plotly::layout::AxisType::Log),
        );

    plot.set_layout(layout);
    plot.write_html(filename);
    println!("Saved plot to {filename}");
}

pub fn plot_error_vs_degree_loglog(
    degrees: &[usize],
    l2_errors: &[f64],
    linf_errors: &[f64],
    theory: Option<(f64, f64)>, // (rate k, leading constant C), error ~ C * n^{-k}
    filename: &str,
) {
    let n_vals: Vec<f64> = degrees.iter().map(|&n| n as f64).collect();

    let l2_trace = Scatter::new(n_vals.clone(), l2_errors.to_vec())
        .name("L2 error (observed)")
        .mode(Mode::LinesMarkers)
        .line(Line::new().color("steelblue").width(2.0));

    let linf_trace = Scatter::new(n_vals.clone(), linf_errors.to_vec())
        .name("L-infinity error (observed)")
        .mode(Mode::LinesMarkers)
        .line(Line::new().color("firebrick").width(2.0));

    let mut plot = Plot::new();
    plot.add_trace(l2_trace);
    plot.add_trace(linf_trace);

    if let Some((k, c)) = theory {
        let theory_vals: Vec<f64> = degrees.iter()
            .map(|&n| c * (n as f64).powf(-k))
            .collect();
        let theory_trace = Scatter::new(n_vals.clone(), theory_vals)
            .name(format!("Theory: O(n^-{k})"))
            .mode(Mode::Lines)
            .line(Line::new().color("gray").width(1.5).dash(DashType::Dash));
        plot.add_trace(theory_trace);
    }

    let layout = Layout::new()
        .title(plotly::common::Title::new().text("Discretization error vs. polynomial degree (log-log)"))
        .x_axis(
            Axis::new()
                .title(plotly::common::Title::new().text("n"))
                .type_(AxisType::Log),
        )
        .y_axis(
            Axis::new()
                .title(plotly::common::Title::new().text("error"))
                .type_(AxisType::Log),
        );

    plot.set_layout(layout);
    plot.write_html(filename);
    println!("Saved plot to {filename}");
}

pub fn plot_convergence_multi(all_norms: &[(String, Vec<f64>)], filename: &str) {
    let mut plot = Plot::new();

    for (label, norms) in all_norms {
        // x = iteration index (0-based), y = residual norm.
        let x: Vec<usize> = (0..norms.len()).collect();

        let trace = Scatter::new(x, norms.clone())
            .mode(Mode::LinesMarkers)
            .name(label);

        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title("Newton convergence across degrees")
        .x_axis(Axis::new().title("Iteration"))
        .y_axis(
            Axis::new()
                .title("Residual norm |F|")
                .type_(AxisType::Log),
        );

    plot.set_layout(layout);
    plot.write_html(filename);
}

pub fn plot_error_vs_intervals_multi(
    interval_counts: &[usize],
    l2_series: &[(String, Vec<f64>)],   // (label = "n = ...", L2 errors per K)
    linf_series: &[(String, Vec<f64>)], // (label = "n = ...", Linf errors per K)
    theory: Option<(f64, f64)>,         // (rate p, leading constant C), error ~ C * K^-p
    loglog: bool,
    filename: &str,
) {
    let colors = [
        "steelblue", "firebrick", "seagreen", "darkorange",
        "purple", "teal", "goldenrod", "slategray",
    ];

    let k_vals: Vec<f64> = interval_counts.iter().map(|&k| k as f64).collect();

    let mut plot = Plot::new();

    for (idx, ((l2_label, l2_errors), (_, linf_errors))) in
        l2_series.iter().zip(linf_series.iter()).enumerate()
    {
        let color = colors[idx % colors.len()];
        let n_pts = l2_errors.len().min(k_vals.len());

        let l2_trace = Scatter::new(k_vals[..n_pts].to_vec(), l2_errors[..n_pts].to_vec())
            .name(format!("{l2_label} (L2)"))
            .mode(Mode::LinesMarkers)
            .line(Line::new().color(color).width(2.0));

        let linf_trace = Scatter::new(k_vals[..n_pts].to_vec(), linf_errors[..n_pts].to_vec())
            .name(format!("{l2_label} (Linf)"))
            .mode(Mode::LinesMarkers)
            .line(Line::new().color(color).width(2.0).dash(DashType::Dot));

        plot.add_trace(l2_trace);
        plot.add_trace(linf_trace);
    }

    if let Some((p, c)) = theory {
        let theory_vals: Vec<f64> = interval_counts.iter()
            .map(|&k| c * (k as f64).powf(-p))
            .collect();
        let theory_trace = Scatter::new(k_vals.clone(), theory_vals)
            .name(format!("Theory: O(K^-{p:.2})"))
            .mode(Mode::Lines)
            .line(Line::new().color("gray").width(1.5).dash(DashType::Dash));
        plot.add_trace(theory_trace);
    }

    let mut layout = Layout::new().width(800)
        .height(600)
        .title(plotly::common::Title::new().text("Discretization error vs. number of intervals (h-refinement)"))
        .x_axis(
            Axis::new()
                .title(plotly::common::Title::new().text("K (intervals)"))
                .type_(if loglog { AxisType::Log } else { AxisType::Linear }),
        )
        .y_axis(
            Axis::new()
                .title(plotly::common::Title::new().text("error"))
                .type_(AxisType::Log),
        )
        .show_legend(true);

    plot.set_layout(layout);
    plot.write_html(filename);
    println!("Saved plot to {filename}");
}