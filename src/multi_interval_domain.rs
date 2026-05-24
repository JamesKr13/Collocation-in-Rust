use nalgebra::{DMatrix, DVector, Matrix, Vector2};


trait PolynomialClass: Sync + Send {
    fn collocation_nodes(&self) -> DVector<f64>;
    fn dmatrix(&self, c: DVector<f64>) -> DMatrix<f64>;
}

// Interval struct handles a single interval existing on [-1,1]
pub struct Interval<P: PolynomialClass> {
    local_collocation_points: DVector<f64>,
    local_dmatrix: DMatrix<f64>,
    poly: P,
    global_a: f64,
    global_b: f64,
}

impl<P: PolynomialClass> Interval<P> {
    pub fn new(poly: P, a: f64, b: f64) -> Self {
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
    pub fn get_local_collocation_point(&self, n: usize) -> f64 {
        self.local_collocation_points[n]
    }

    pub fn test(&self) {
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

    pub fn map_to_physical(&self) -> DVector<f64> {
        self.local_collocation_points.map(|x|{
            &self.global_a + 0.5 *(&self.global_b-&self.global_a)*(x+1.0)
        })
    }
    // When the local -> physical, the chain rule introduces this factor
    pub fn scaled_dmatrix(&self) -> DMatrix<f64> {
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


pub struct MultiIntervalDomain {
    pub interval: Vec<Box<dyn IntervalTrait + Send + Sync>>,
    pub breakpoints: Vec<f64>
}

// This purely exist because I though what if interval and different distribution
// and more general code is muy bueno
pub trait IntervalTrait: Send + Sync {
    fn collocation_points(&self) -> &DVector<f64>;
    fn dmatrix(&self) -> &DMatrix<f64>;
    fn map_to_physical(&self) -> DVector<f64>;
    fn scaled_dmatrix(&self) -> DMatrix<f64>;
    fn get_local_collocation_point(&self, n: usize) -> f64;
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
    fn get_local_collocation_point(&self, n: usize) -> f64 {
        self.get_local_collocation_point(n)
    }
}

impl MultiIntervalDomain {
    pub fn new(n_interval: usize,truncate_a: f64, truncate_b: f64, n_collocate: usize) -> Self {
        // equispaced intervals
        let h = (truncate_b-truncate_a)/ n_interval as f64;
        let breakpoints: Vec<f64> =  (0..=n_interval).map(|i| truncate_a + i as f64 * h).collect();
        // Default we will use the cheb, any other is not implmented
        let cheb: Chebyshev = Chebyshev::new(n_collocate);
        Self {
            interval: (0..n_interval) // This should not break, but shrug
                .map(|i| {
                    Box::new(Interval::new(cheb.clone(), breakpoints[i], breakpoints[i+1]))
                    as Box<dyn IntervalTrait + Send + Sync>
                })
                .collect(),
        breakpoints,
        }
    }

    pub fn global_collocation_points_raw(&self) -> DVector<f64> {
        let mut pts: Vec<f64> = Vec::new();
        for iv in self.interval.iter() {
            let local_pts = iv.map_to_physical();
            pts.extend_from_slice(local_pts.as_slice()); // no skipping
        }
        DVector::from_vec(pts)
    }

    pub fn get_interval(&self, n: usize) -> &dyn IntervalTrait {
        self.interval[n].as_ref()
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