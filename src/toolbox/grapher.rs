use log::Level::Trace;
use log::{debug, info};
// grapher.rs
use nalgebra::{DVector, Scalar};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use plotly::configuration::{ImageButtonFormats, ToImageButtonOptions};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use plotly::{Configuration, Plot, Scatter};
use plotly::common::{DashType, Font, Line, Marker, MarkerSymbol, Mode, ExponentFormat};
use plotly::layout::{Layout, Axis, AxisType};

use crate::h_interval::multi_interval::MultiIntervalDomain;
use super::trig_trait::Trig;
use super::super::ContinuationErrorPanel;


/// Creates the parent folder of `filename` if it doesn't exist yet.
/// Works for bare filenames ("a.html") and nested paths ("study/a.html").
fn ensure_parent(filename: &str) {
    if let Some(parent) = std::path::Path::new(filename).parent() {
        // A bare filename has an empty parent, which needs no folder.
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("failed to create output folder");
        }
    }
}

/// Writes a plot to `filename`, creating any missing folders first.
fn write_plot(plot: &Plot, filename: &str) {
    ensure_parent(filename);
    plot.write_html(filename);
}

/// Writes raw HTML text to `filename`, creating any missing folders first.
fn write_html_text(filename: &str, html: &str) {
    ensure_parent(filename);
    std::fs::write(filename, html).expect("failed to write html file");
}


/// Log axis that always shows ticks in scientific notation (1e-5, 2e-5, ...).
fn log_axis(title: &str) -> Axis {
    Axis::new()
        .title(plotly::common::Title::new().text(title))
        .type_(AxisType::Log)
        .exponent_format(ExponentFormat::CapitalE)
        .tick_format(".0e")
}

/// Sets the filename used by the camera button in the HTML plot.
fn set_png_name(plot: &mut Plot, camera_name: &str) {
    plot.set_configuration(
        Configuration::new().to_image_button_options(
            ToImageButtonOptions::new()
                .format(ImageButtonFormats::Png)
                .filename(camera_name),
        ),
    );
}

/// File name without folder or extension ("a/b/c.html" -> "c").
fn stem(filename: &str) -> &str {
    std::path::Path::new(filename)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(filename)
}


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
    camera_name: &str,
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
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;

    for (idx, (label, u, domain)) in solutions.iter().enumerate() {
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
                    .dash(DashType::Dash),
            );
        plot.add_trace(exact_line);
    }

    let layout = Layout::new().width(800)
        .height(600)
        .title(plotly::common::Title::new().text(title))
        .x_axis(Axis::new().title(plotly::common::Title::new().text("x")))
        .y_axis(Axis::new().title(plotly::common::Title::new().text("u(x)")))
        .show_legend(true).font(Font::new().size(18));

    set_png_name(&mut plot, camera_name);
    plot.set_layout(layout);
    write_plot(&plot, filename);
    info!("Saved plot to {filename}");
}

/// Plot residual norm vs iteration on a log-y scale to visualize
/// Newton convergence.
pub fn plot_convergence(norms: &[f64], filename: &str, camera_name: &str) {
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
        .y_axis(log_axis("residual norm"))
        .font(Font::new().size(18));

    let new_name = stem(filename);
    set_png_name(&mut plot, &format!("{new_name}_{camera_name}"));
    plot.set_layout(layout);
    write_plot(&plot, filename);
    info!("Saved plot to {filename}");
}

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
    camera_name: &str,
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
            .line(Line::new().color("gray").width(1.5).dash(DashType::Dash));
        plot.add_trace(theory_trace);
    }

    let layout = Layout::new().width(800)
        .height(600)
        .title(plotly::common::Title::new().text("Collocation error vs. polynomial degree"))
        .x_axis(Axis::new().title(plotly::common::Title::new().text("n")))
        .y_axis(log_axis("error"))
        .font(Font::new().size(18));

    let new_name = stem(filename);
    set_png_name(&mut plot, &format!("{new_name}_{camera_name}"));
    plot.set_layout(layout);
    write_plot(&plot, filename);
    info!("Saved plot to {filename}");
}

pub fn plot_all_h_step_solutions(
    solutions: &[(MultiIntervalDomain<f64>, DVector<f64>)],
    errors: &[Vec<(f64, f64)>],
    filename: &str,
) {
    let mut plot = Plot::new();

    /*
     * Solution traces
     */
    for (domain, u) in solutions.iter() {
        let mut points: Vec<(f64, f64)> = Vec::new();

        let mut offset = 0usize;

        for interval in &domain.intervals {
            let n = interval.n_nodes;
            let x_nodes = interval.map_to_physical();

            for j in 0..n {
                points.push((
                    x_nodes[j],
                    u[offset + j],
                ));
            }

            offset += n;
        }

        points.sort_by(|a, b| a.0.total_cmp(&b.0));

        let (x_all, y_all): (Vec<f64>, Vec<f64>) =
            points.into_iter().unzip();

        let trace = Scatter::new(x_all, y_all)
            .mode(Mode::Lines)
            .name(format!("Domain={:?}", domain.domain))
            .line(Line::new().width(3.0));

        plot.add_trace(trace);
    }

    for (i, error) in errors.iter().enumerate() {
        if error.is_empty() {
            continue;
        }

        let mut error_points = error.clone();

        error_points.sort_by(|a, b| a.0.total_cmp(&b.0));

        let (error_x, error_y): (Vec<f64>, Vec<f64>) =
            error_points.into_iter().unzip();

        let error_trace = Scatter::new(error_x, error_y)
            .mode(Mode::Lines)
            .x_axis("x2")
            .y_axis("y2")
            .line(Line::new().width(3.0)).show_legend(false);

        plot.add_trace(error_trace);
    }

    let layout = Layout::new()
        .title("Collocation solutions under domain expansion")
        .height(900)
        .width(1200)
        .x_axis(
            Axis::new()
                .anchor("y")
                .dtick(0.5)
                .show_grid(true),
        )
        .y_axis(
            Axis::new()
                .title("u(ζ)")
                .domain(&[0.38, 1.0]),
        )
        .x_axis2(
            Axis::new()
                .title("ζ")
                .anchor("y2")
                .matches("x"),
        )
        .y_axis2(
            Axis::new()
                .title("Domain Error")
                .domain(&[0.0, 0.28]),
        )
        .font(Font::new().size(20));

    plot.set_layout(layout);

    plot.write_html(filename);
}

pub fn plot_error_vs_degree_loglog(
    degrees: &[usize],
    l2_errors: &[f64],
    linf_errors: &[f64],
    theory: Option<(f64, f64)>,
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
        .x_axis(log_axis("n"))
        .y_axis(log_axis("error"))
        .font(Font::new().size(18));

    set_png_name(&mut plot, stem(filename));
    plot.set_layout(layout);
    write_plot(&plot, filename);
    info!("Saved plot to {filename}");
}

pub fn plot_convergence_multi(all_norms: &[(String, Vec<f64>)], filename: &str, camera_name: &str) {
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
        .y_axis(log_axis("Residual norm |F|"))
        .font(Font::new().size(18));

    set_png_name(&mut plot, camera_name);
    plot.set_layout(layout);
    write_plot(&plot, filename);
    info!("Saved plot to {filename}");
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

    let layout = Layout::new().width(800)
        .height(600)
        .title(plotly::common::Title::new().text("Discretization error vs. number of intervals (h-refinement)"))
        .x_axis(if loglog {
            log_axis("K (intervals)")
        } else {
            Axis::new().title(plotly::common::Title::new().text("K (intervals)"))
        })
        .y_axis(log_axis("error"))
        .show_legend(true).font(Font::new().size(18));

    set_png_name(&mut plot, stem(filename));
    plot.set_layout(layout);
    write_plot(&plot, filename);
    info!("Saved plot to {filename}");
}



pub fn plot_continuation_error_grid(panels:&[ContinuationErrorPanel],cols:usize,filename:&str){
    if panels.is_empty(){return;}
    let cols=cols.max(1);

    let mut html=format!(r#"<!DOCTYPE html><html><head><meta charset="utf-8"><script src="https://cdn.plot.ly/plotly-3.0.1.min.js"></script><style>body{{font-family:sans-serif;padding:20px}}.grid{{display:grid;grid-template-columns:repeat({},minmax(400px,1fr));gap:15px}}</style></head><body><div class="grid">"#,cols);

    for(i,p)in panels.iter().enumerate(){
        let x:Vec<f64>=p.degrees.iter().map(|&n|n as f64).collect();
        let mut plot=Plot::new();

        plot.add_trace(Scatter::new(x.clone(),p.l2_errors.clone()).name("L2").mode(Mode::LinesMarkers));
        plot.add_trace(Scatter::new(x,p.linf_errors.clone()).name("Linf").mode(Mode::LinesMarkers));

        plot.set_layout(Layout::new()
            .height(400)
            .title(format!("s = {:.5}",p.speed))
            .x_axis(Axis::new().title("Polynomial n"))
            .y_axis(log_axis("Error")));

        html.push_str(&plot.to_inline_html(Some(&format!("plot-{i}"))));
    }

    html.push_str("</div></body></html>");
    write_html_text(filename, &html);
    info!("Saved plot to {filename}");
}



pub fn plot_guess<F>(
    mut guess: F,
    bounds: (f64, f64),
    samples: usize,
    title: &str,
    filename: &str,
) where
    F: FnMut(f64) -> f64,
{
    let (a, b) = bounds;
    assert!(a.is_finite() && b.is_finite() && a < b);
    assert!(samples >= 2);

    let mut xs: Vec<f64> = (0..samples)
        .map(|i| {
            let t = i as f64 / (samples - 1) as f64;
            (1.0 - t) * a + t * b
        })
        .collect();

    // Include the point where the two asymptotic pieces join.
    if a < 0.0 && b > 0.0 {
        xs.push(0.0);
        xs.sort_by(f64::total_cmp);
        xs.dedup();
    }

    let ys: Vec<f64> = xs.iter()
        .map(|&x| {
            let y = guess(x);
            assert!(y.is_finite(), "Non-finite initial guess at x = {x}");
            y
        })
        .collect();

    let trace = Scatter::new(xs, ys)
        .name("Initial guess")
        .mode(Mode::Lines)
        .line(Line::new().color("steelblue").width(2.5));

    let mut plot = Plot::new();
    plot.add_trace(trace);

    set_png_name(&mut plot, stem(filename));
    plot.set_layout(
        Layout::new()
            .width(800)
            .height(600)
            .title(plotly::common::Title::new().text(title))
            .x_axis(
                Axis::new().title(plotly::common::Title::new().text("x"))
            )
            .y_axis(
                Axis::new().title(plotly::common::Title::new().text("u(x)"))
            )
            .font(Font::new().size(18)),
    );

    write_plot(&plot, filename);
    info!("Saved plot to {filename}");
}


pub fn plot_iterates<T>(
    iterates: &[(String, DVector<T>)],
    domain: &MultiIntervalDomain<T>,
    exact: &dyn Fn(f64) -> f64,
    title: &str,
    filename: &str,
    camera_name: &str,
) where
    T: FromPrimitive + ToPrimitive + Zero + Scalar + Copy + Sync
        + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
        + AddAssign + Neg<Output = T> + Trig + 'static,
{
    let entries: Vec<(&str, DVector<T>, &MultiIntervalDomain<T>)> = iterates
        .iter()
        .map(|(label, u)| (label.as_str(), u.clone(), domain))
        .collect();

    plot_solutions(&entries, Some(("Exact", exact)), title, filename, camera_name);
    debug!("Iterations saved to {:?}", filename);
}