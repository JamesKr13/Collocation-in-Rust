use core::f64;
use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::h_interval::{interval::Interval, polynomial::polynomial_trait::PolynomialKind};

pub enum IntervalDivision {
    Uniform,
    Geometric,
    Exponential,
    ChebyshevInterval,
    Adaptive,
    Custom(Arc<dyn Fn(usize, f64, f64) -> Vec<(f64, f64)> + Send + Sync>),
}

impl Clone for IntervalDivision {
    fn clone(&self) -> Self {
        match self {
            IntervalDivision::Uniform => IntervalDivision::Uniform,
            IntervalDivision::Geometric => IntervalDivision::Geometric,
            IntervalDivision::Exponential => IntervalDivision::Exponential,
            IntervalDivision::ChebyshevInterval => IntervalDivision::ChebyshevInterval,
            IntervalDivision::Adaptive => IntervalDivision::Adaptive,
            IntervalDivision::Custom(f) => IntervalDivision::Custom(Arc::clone(f)),
        }
    }
}

pub struct MultiIntervalDomain<T>
where
    T: FromPrimitive
        + ToPrimitive
        + Zero
        + Scalar
        + Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + Neg<Output = T>
        + 'static
        + Sync,
{
    pub domain: (f64, f64),
    pub intervals: Vec<Interval<T>>,
    pub division_type: IntervalDivision,
    pub poly_kind: PolynomialKind,
    pub break_points: Vec<(f64, f64)>,
}

impl<T> MultiIntervalDomain<T>
where
    T: FromPrimitive
        + ToPrimitive
        + Zero
        + Scalar
        + Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Div<Output = T>
        + AddAssign
        + Neg<Output = T>
        + Sync
        + 'static,
{
    pub fn new(
        kind: PolynomialKind,
        n_interval: usize,
        domain: (f64, f64),
        n_collocation: usize,
        division: IntervalDivision,
    ) -> Self {
        let (a, b) = domain.clone();

        // Convert domain endpoints down to f64 once, since breakpoint
        // geometry is computed purely in f64 (see earlier discussion).
        let a_f64 = a.to_f64().expect("Incorrect type match: domain conversion");
        let b_f64 = b.to_f64().expect("Incorrect type match: domain conversion");

        let breakpoints = match &division {
            IntervalDivision::Uniform => Self::uniform_split(n_interval, &(a_f64, b_f64)),
            IntervalDivision::Geometric => Self::geometric_split(n_interval, &(a_f64, b_f64)),
            IntervalDivision::Exponential => Self::exponential_split(n_interval, &(a_f64, b_f64)),
            IntervalDivision::ChebyshevInterval => Self::chebyshev_split(n_interval, &(a_f64, b_f64)),
            IntervalDivision::Adaptive => todo!(),
            IntervalDivision::Custom(f) => f(n_interval, a_f64, b_f64),
        };

        let intervals: Vec<Interval<T>> =
            Self::multi_interval_create(breakpoints.clone(), kind, n_collocation);

        Self {
            domain,
            intervals,
            division_type: division,
            poly_kind: kind,
            break_points: breakpoints,
        }
    }

    fn multi_interval_create(breaks: Vec<(f64, f64)>, kind: PolynomialKind, degree: usize) -> Vec<Interval<T>> {
        breaks.iter().map(|a| Interval::new(kind.clone(), degree, *a)).collect()
    }

    fn uniform_split(n_interval: usize, (a, b): &(f64, f64)) -> Vec<(f64, f64)> {
        let width = (b - a) / n_interval as f64;

        (0..n_interval)
            .map(|i| {
                let lo = a + width * i as f64;
                let hi = a + width * (i + 1) as f64;
                (lo, hi)
            })
            .collect()
    }

    fn geometric_split(n_interval: usize, (a, b): &(f64, f64)) -> Vec<(f64, f64)> {
        let ratio = 0.7_f64;

        // Compute raw widths first: w, w*r, w*r^2, ... summing to (b - a)
        let mut raw_widths = Vec::with_capacity(n_interval);
        let mut acc = 1.0;
        let mut total = 0.0;
        for _ in 0..n_interval {
            raw_widths.push(acc);
            total += acc;
            acc *= ratio;
        }

        let mut breakpoints = Vec::with_capacity(n_interval);
        let mut cursor = 0.0;

        for w in raw_widths.iter() {
            let frac = w / total;
            let width = (b - a) * frac;
            let lo = a + cursor;
            cursor += width;
            let hi = a + cursor;
            breakpoints.push((lo, hi));
        }

        breakpoints
    }

    fn exponential_split(n_interval: usize, (a, b): &(f64, f64)) -> Vec<(f64, f64)> {
        let k = 3.0_f64;
        let n = n_interval as f64;
        let denom = k.exp() - 1.0;

        let mut points = Vec::with_capacity(n_interval + 1);
        for i in 0..=n_interval {
            let frac = ((k * (i as f64) / n).exp() - 1.0) / denom;
            points.push(frac);
        }

        let width = b - a;

        (0..n_interval)
            .map(|i| {
                let lo = a + width * points[i];
                let hi = a + width * points[i + 1];
                (lo, hi)
            })
            .collect()
    }

    fn chebyshev_split(n_interval: usize, (a, b): &(f64, f64)) -> Vec<(f64, f64)> {
        let n = n_interval;
        // n_interval+1 breakpoints on [-1, 1] via Chebyshev-Gauss-Lobatto spacing
        let mut points = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let theta = std::f64::consts::PI * (i as f64) / (n as f64);
            points.push(-theta.cos()); // ranges from -1 to 1, denser near ends
        }

        let half_width = (b - a) / 2.0;

        (0..n)
            .map(|i| {
                // map [-1, 1] -> [a, b]
                let lo = a + half_width * (points[i] + 1.0);
                let hi = a + half_width * (points[i + 1] + 1.0);
                (lo, hi)
            })
            .collect()
    }

    pub fn cover_of_i(&self, i: f64) -> usize {
        self.break_points
            .iter()
            .position(|&(lo, hi)| lo <= i && hi >= i)
            .expect("Intervals do not cover {i}")
    }

    pub fn node_near_zero(&self) -> (usize, usize) {
        let interval_idx = self.cover_of_i(0.0);

        let nodes = self.intervals[interval_idx].polynomial.collocation_nodes();

        let node_idx = nodes
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let a_f64 = a.to_f64().expect("Incorrect type match: node_near_zero");
                let b_f64 = b.to_f64().expect("Incorrect type match: node_near_zero");
                a_f64.abs().partial_cmp(&b_f64.abs()).unwrap()
            })
            .map(|(i, _)| i)
            .expect("Interval has no collocation nodes");

        (interval_idx, node_idx)
    }

    pub fn global_index(&self, (interval_idx, node_idx): (usize, usize)) -> usize {
        let preceding_nodes: usize = self.intervals[..interval_idx]
            .iter()
            .map(|interval| interval.n_nodes)
            .sum();

        preceding_nodes + node_idx
    }
}