use std::f64::consts::PI;
use std::ops::Mul;

use nalgebra::{DMatrix, DVector, Matrix, Vector2};

use plotly::{Plot, Scatter};
use plotly::common::{Mode, Line, Marker, MarkerSymbol};
use plotly::layout::{Layout, Axis};


trait PolynomialClass {
    fn collocation_nodes(&self) -> DVector<f64>;
    fn dmatrix(&self, c: DVector<f64>) -> DMatrix<f64>;
}

// Interval struct handles a single interval existing on [-1,1]
struct Interval<P: PolynomialClass> {
    local_collocation_points: DVector<f64>,
    local_dmatrix: DMatrix<f64>,
    poly: P,
    global_a: f64,
    global_b: f64,
}

impl<P: PolynomialClass> Interval<P> {
    fn new(poly: P, a: f64, b: f64) -> Self {
        let local_collocation_points = poly.collocation_nodes();
        let clone_lcp = local_collocation_points.clone();
        Self {
            local_collocation_points,
            local_dmatrix: poly.dmatrix(clone_lcp),
            poly,
            global_a: a,
            global_b: b
        }
    }

    fn test(&self) {
        let n = self.local_collocation_points.len();
        let ones: DVector<f64> = DVector::from_element(n, 1.0);
        
        let du_exact = &ones - 2.0 * &self.local_collocation_points;
        let ddu_exact = -2.0 * &ones;
        
        // Compute (1 - x) first, then multiply
        let one_minus_x = &ones - &self.local_collocation_points;
        let u_test = self.local_collocation_points.component_mul(&one_minus_x);
        let ddu_approx = &self.local_dmatrix * &u_test;
        println!("{0}",if (&ddu_approx - &du_exact).amax() < 1.0e-12 {"Diff Matrix is accurate"} else {"Scream and cry"}); 
        println!("{0}",if (&self.local_dmatrix * &ddu_approx - &ddu_exact).amax() < 1.0e-12 {"2nd Order Diff Matrix is accurate"} else {"Weep like a baby"}); 

    }

    fn map_to_physical(&self) -> DVector<f64> {
        self.local_collocation_points.map(|x|{
            &self.global_a + 0.5 *(&self.global_b-&self.global_a)*(x+1.0)
        })
    }
    // When the local -> physical, the chain rule introduces this factor
    fn scaled_dmatrix(&self) -> DMatrix<f64> {
        &self.local_dmatrix * (2.0/(&self.global_b-self.global_a))
    }
}

#[derive(Copy, Clone)]
struct Chebyshev {
    n: usize, // number of collocation points
}

impl Chebyshev {
    fn new(n: usize) -> Self {
        assert!(n >= 2, "Need at least 2 collocation points");
        Self { n }
    }
}

impl PolynomialClass for Chebyshev {
    fn collocation_nodes(&self) -> DVector<f64> {
        let n = self.n;
        let mut nodes = DVector::zeros(n);
        
        for j in 0..n {
            let theta = std::f64::consts::PI * (j as f64) / ((n - 1) as f64);
            nodes[j] = -theta.cos();
        }
        
        nodes
    }
    
    fn dmatrix(&self, c: DVector<f64>) -> DMatrix<f64> {
        let n = c.len();
        let mut d = DMatrix::zeros(n, n);
        
        // Compute the coefficient weights c_i
        // c_0 = c_{N-1} = 2, all others are 1
        let mut weights = vec![1.0; n];
        weights[0] = 2.0;
        weights[n - 1] = 2.0;
        
        // Compute off-diagonal entries
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                    d[(i, j)] = (weights[i] / weights[j]) * sign / (c[i] - c[j]);
                }
            }
        }
        
        // Compute diagonal entries
        // D_ii = -sum of off-diagonal entries in row i
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += d[(i, j)];
                }
            }
            d[(i, i)] = -sum;
        }
        
        d
    }
}


struct MultiIntervalDomain {
    interval: Vec<Box<dyn IntervalTrait>>,
    breakpoints: Vec<f64>
}

// This purely exist because I though what if interval and different distribution
// and more general code is muy bueno
trait IntervalTrait {
    fn collocation_points(&self) -> &DVector<f64>;
    fn dmatrix(&self) -> &DMatrix<f64>;
    fn map_to_physical(&self) -> DVector<f64>;
    fn scaled_dmatrix(&self) -> DMatrix<f64>;
}

impl<P: PolynomialClass> IntervalTrait for Interval<P> {
    fn collocation_points(&self) -> &DVector<f64> {
        &self.local_collocation_points
    }

    fn dmatrix(&self) -> &DMatrix<f64> {
        &self.dmatrix()
    }
    fn map_to_physical(&self) -> DVector<f64> {
        self.map_to_physical()
    }

    fn scaled_dmatrix(&self) -> DMatrix<f64> {
        self.scaled_dmatrix()
    }
}
// For playing around, for the problem at hand, we do actually have
// Neumann bc, as u -> u_0 or u -> 0 as u' -> 0, decays
enum BoundaryCondition {
    Dirichlet(f64),Neumann(f64)
}


impl MultiIntervalDomain {
    fn new(n_interval: usize,truncate_a: f64, truncate_b: f64, n_collocate: usize) -> Self {
        // equispaced intervals
        let h = (truncate_b-truncate_a)/ n_interval as f64;
        let breakpoints: Vec<f64> =  (0..=n_interval).map(|i| truncate_a + i as f64 * h).collect();
        // Default we will use the cheb, any other is not implmented
        let cheb: Chebyshev = Chebyshev::new(n_collocate);
        Self {
            interval: (0..n_interval) // This should not break, but shrug
                .map(|i| {
                    Box::new(Interval::new(cheb.clone(), breakpoints[i], breakpoints[i+1]))
                        as Box<dyn IntervalTrait>
                })
                .collect(),
        breakpoints,
        }
    }

    fn global_collocation_points(&self) -> DVector<f64> {
        let mut pts: Vec<f64> = Vec::new();
        for (i, iv) in self.interval.iter().enumerate() {
            let local_pts = iv.map_to_physical();
            let start = if i == 0 { 0 } else { 1 };
            pts.extend_from_slice(&local_pts.as_slice()[start..]);
        }
        DVector::from_vec(pts)
    }


    fn diff_matrix(&self) -> DMatrix<f64> {
        let n_intervals = self.interval.len();
        let n_per = self.interval[0].collocation_points().len();
        let n_total = n_per + (n_intervals - 1) * (n_per - 1);
        let mut D = DMatrix::zeros(n_total, n_total);

        for (i, interval) in self.interval.iter().enumerate() {
            let d_local = interval.scaled_dmatrix();
            let global_start = i * (n_per - 1);
            for row in 0..n_per {
                for col in 0..n_per {
                    D[(global_start + row, global_start + col)] += d_local[(row, col)];
                }
            }
        }
        D
    }
    
}

trait ODE {

    // Residua is incorrect
    fn build_resiudal(&self,multi_interval_domain: &MultiIntervalDomain, u: &DVector<f64>, x: &DVector<f64>, bc_left: &BoundaryCondition, bc_right: &BoundaryCondition, residual_fn: impl Fn(Vec<f64>) -> f64) -> DVector<f64> {
        let n_intervals = multi_interval_domain.interval.len();
        let n_per = multi_interval_domain.interval[0].collocation_points().len();
        let n_total: usize = u.len();
        let mut F: DVector<f64> = DVector::zeros(n_total); // Stores the interfaces

        for (i, interval) in multi_interval_domain.interval.iter().enumerate() {
            let global_start = i* (n_per-1);
            let d = interval.scaled_dmatrix();
            let u_local = self.local_u(u, i, n_per);
            let du_local = &d * &u_local;
            let d2u__local = &d * &du_local;

            let row_start = 1;
            let row_end = if i == n_intervals -1 {n_per-1} else {n_per -1};

            for local_row in row_start..row_end {
                let global_row = global_start + local_row;
                F[global_row] = residual_fn(vec![x[global_row], u_local[local_row], du_local[local_row], d2u__local[local_row]]);
            }
        }

        // Continuity
        for i in 0..(n_intervals-1) {
            let interface_idx = (i + 1) * (n_per-1);

            let d_left =multi_interval_domain.interval[i].scaled_dmatrix();
            let d_right = multi_interval_domain.interval[i + 1].scaled_dmatrix();

            let u_left = self.local_u(u, i, n_per);
            let u_right = self.local_u(u, i + 1, n_per);

            let du_left = &d_left * &u_left;
            let du_right = &d_right * &u_right;
            println!("{}", interface_idx);
            F[interface_idx] = du_left[n_per-1] - du_right[0];
        }

        self.apply_vec(&mut F, multi_interval_domain, &u,bc_left, true, n_total, n_per, n_intervals);
        self.apply_vec(&mut F, multi_interval_domain,&u,bc_right, false, n_total, n_per, n_intervals);
        
        println!("{}", F);

        F
    }
    fn local_u<'b>(&self, u: &DVector<f64>, i: usize, n_per: usize) -> DVector<f64> {
        let start = i * (n_per - 1);
        u.rows(start, n_per).clone_owned()
    }

    // Jacobian is incorrect
    fn build_jacobian(&self, u: &DVector<f64>, bc_left: &BoundaryCondition, bc_right: &BoundaryCondition, ode_jac: impl Fn(f64, f64, f64, &DMatrix<f64>) -> DVector<f64>) -> DMatrix<f64>;

    fn apply_matx(&self, A: &mut DMatrix<f64>,multi_interval_domain: &MultiIntervalDomain, bc: &BoundaryCondition, is_left: bool, n_total: usize, n_per: usize, u_intervals: usize) {
        let row = if is_left { 0 } else { n_total - 1 };
        for col in 0..n_total { A[(row, col)] = 0.0; }
        match bc {
            BoundaryCondition::Dirichlet(_) => A[(row, row)] = 1.0,
            BoundaryCondition::Neumann(_) => {
                let i = if is_left { 0 } else { u_intervals - 1 };
                let d = multi_interval_domain.interval[i].scaled_dmatrix();
                let local_row = if is_left { 0 } else { n_per - 1 };
                let global_start = i * (n_per - 1);
                for j in 0..n_per {
                    A[(row, global_start + j)] = d[(local_row, j)];
                }
            }
        }
    }

    fn apply_vec(&self, A: &mut DVector<f64>,multi_interval_domain: &MultiIntervalDomain, u: &DVector<f64>,  bc: &BoundaryCondition, is_left: bool, n_total: usize, n_per: usize, u_intervals: usize) {
        let row = if is_left { 0 } else { n_total - 1 };
            let i = if is_left { 0 } else { u_intervals - 1};
            let d = multi_interval_domain.interval[i].scaled_dmatrix();
            let u_loc = self.local_u(u, i, n_per);
            match bc {
                BoundaryCondition::Dirichlet(val) => A[row] = u[row] - val,
                BoundaryCondition::Neumann(val) => {
                    let local_row = if is_left { 0 } else { n_per - 1 };
                    let du_loc = &d * &u_loc;
                    A[row] = du_loc[local_row] - val;
                }
            }
    }

}

struct NonLinearODE<'a> {
    multi_interval_domain: &'a MultiIntervalDomain
}

impl<'a> ODE for NonLinearODE<'a> {
    fn build_jacobian(&self, u: &DVector<f64>, bc_left: &BoundaryCondition, bc_right: &BoundaryCondition, ode_jac: impl Fn(f64, f64, f64, &DMatrix<f64>) -> DVector<f64>) -> DMatrix<f64> {
        let n_intervals = self.multi_interval_domain.interval.len();
        let n_per = self.multi_interval_domain.interval[0].collocation_points().len();
        let n_total = u.len();
        let mut J = DMatrix::zeros(n_total, n_total);

        for (i, interval) in self.multi_interval_domain.interval.iter().enumerate() {
            let global_start = i * (n_per - 1);
            let d  = interval.scaled_dmatrix();
            let d2 = &d * &d;
            let u_loc = self.local_u(u, i, n_per);

            for local_row in 1..(n_per - 1) {
                let global_row = global_start + local_row;

                for local_col in 0..n_per {
                    let global_col = global_start + local_col;

                    let jac_val = d2[(local_row, local_col)] + if local_col == local_row { 3.0 * u_loc[local_row].powi(2) } else { 0.0 };

                    J[(global_row, global_col)] += jac_val;
                }
            }
        }

        // continuity
        for i in 0..(n_intervals - 1) {
            let interface_idx = (i + 1) * (n_per - 1);
            let d_left  = self.multi_interval_domain.interval[i].scaled_dmatrix();
            let d_right = self.multi_interval_domain.interval[i + 1].scaled_dmatrix();
            let left_start  = i * (n_per - 1);
            let right_start = (i + 1) * (n_per - 1);

            for j in 0..n_per {
                J[(interface_idx, left_start  + j)] += d_left [(n_per - 1, j)];
                J[(interface_idx, right_start + j)] -= d_right[(0,         j)];
            }
        }

        self.apply_matx(&mut J, self.multi_interval_domain, bc_left,  true,  n_total, n_per, n_intervals);
        self.apply_matx(&mut J, self.multi_interval_domain, bc_right, false, n_total, n_per, n_intervals);

        J
    }

    // test function 
}
struct LinearODE<'a>{
    multi_interval_domain: &'a MultiIntervalDomain
}

impl<'a> ODE for LinearODE<'a> {

    fn build_jacobian(&self, u: &DVector<f64>, bc_left: &BoundaryCondition, bc_right: &BoundaryCondition, ode_jac: impl Fn(f64, f64, f64, &DMatrix<f64>) -> DVector<f64>) -> DMatrix<f64> {
        let n_intervals = self.multi_interval_domain.interval.len();
        let n_per = self.multi_interval_domain.interval[0].collocation_points().len();
        let n_total = u.len();
        let mut J = DMatrix::zeros(n_total, n_total);

        for (i, interval) in self.multi_interval_domain.interval.iter().enumerate() {
            let global_start = i * (n_per - 1);
            let d = interval.scaled_dmatrix();
            let u_loc = self.local_u(u, i, n_per);
            let du_loc = &d * &u_loc;

            let row_start = 1;
            let row_end = n_per - 1;

            for local_row in row_start..row_end {
                let global_row = global_start + local_row;
               
               
                for local_col in 0..n_per {
                    let global_col = global_start + local_col;
                    let d2: f64 = (&d * &d)[(local_row, local_col)];
                    let nonlin = du_loc[local_row] * (local_col == local_row) as u8 as f64
                        + u_loc[local_row] * d[(local_row, local_col)];
                    J[(global_row, global_col)] += d2 + nonlin;
                }
            }
        }

        for i in 0..(n_intervals - 1) {
            let interface_idx = (i + 1) * (n_per - 1);
            let d_left = self.multi_interval_domain.interval[i].scaled_dmatrix();
            let d_right = self.multi_interval_domain.interval[i + 1].scaled_dmatrix();

            let left_start = i * (n_per - 1);
            let right_start = (i + 1) * (n_per - 1);

            for j in 0..n_per {
                J[(interface_idx, left_start + j)] += d_left[(n_per - 1, j)];
                J[(interface_idx, right_start + j)] -= d_right[(0, j)];
            }
        }

        // BC rows
        self.apply_matx(&mut J, self.multi_interval_domain, bc_left, true, n_total, n_per, n_intervals);
        self.apply_matx(&mut J, self.multi_interval_domain,bc_right, false, n_total, n_per, n_intervals);

        J
    }
    
}

fn plot_solution(x: &DVector<f64>, u: &DVector<f64>, domain: &MultiIntervalDomain, title: &str, filename: &str) {
    let x_vals: Vec<f64> = x.iter().cloned().collect();
    let u_vals: Vec<f64> = u.iter().cloned().collect();

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
    };

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


fn main() {
    let domain = MultiIntervalDomain::new(3, -5.0, 5.0, 3);
    let x = domain.global_collocation_points();
    let n = x.len();
    // let mut u: DVector<f64> = x.map(|xi| (2.0*std::f64::consts::PI*xi/5.0).sin());
    let mut u: DVector<f64> = DVector::zeros(n);

    let bc_left = BoundaryCondition::Dirichlet(0.0);
    let bc_right = BoundaryCondition::Dirichlet(10.0);
// (2.0*std::f64::consts::PI/5.0).powi(2)*p[0]*p[0]*((2.0*std::f64::consts::PI*p[0]/5.0)).sin()
    let ode = NonLinearODE {multi_interval_domain: &domain};
    for iter in 0..=100 {
        let F = ode.build_resiudal(&domain, &u, &x, &bc_left, &bc_right,
            |p| p[3] + p[1].powi(3) - p[0].sin(),
        );
        let J = ode.build_jacobian(&u, &bc_left, &bc_right, |_, _, _, _| unreachable!());
        println!("{}X{} To {}", J.nrows(), J.ncols(), F.nrows());
        println!("{}", J);
        println!("{}", F);
        let norm = F.norm();
        println!("Iter {iter:>2}: |F| = {norm:.3e}");
        if norm < 1e-10 { println!("Converged."); break; }

        let delta = J.lu().solve(&(-&F)).expect("singular");

        u+= delta;
    }

    plot_solution(&x, &u, &domain, "Solution", "solution.html");
}