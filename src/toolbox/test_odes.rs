use nalgebra::{DMatrix, DVector, Scalar};
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use super::dual_matrix_operations::Operations;
use super::auto_diff_dual::Dual;
use super::trig_trait::Trig;

use crate::collocation_method;
use collocation_method::ode_trait::ODE;
use collocation_method::boundary_conditions::BoundaryCondition;

pub struct PendulumODE;

impl<S, T> ODE<S, T> for PendulumODE
where
    S: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = S> + Sub<Output = S> + Mul<Output = S> + Div<Output = S>
        + AddAssign + Neg<Output = S> + Trig + From<T> + 'static,
    T: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
        + AddAssign + Neg<Output = T> + 'static,
{
    fn eval(row: usize, _x: S, u: &DVector<S>, d: &DMatrix<T>) -> S {
        let du = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        d2u[row] + u[row]
    }

    fn forcing(xi: S) -> S {
        xi.cos()
    }

    fn bc_left() -> BoundaryCondition<S> {
        BoundaryCondition::Dirichlet(S::zero())
    }

    fn bc_right() -> BoundaryCondition<S> {
        BoundaryCondition::Dirichlet(S::zero())
    }
}

pub struct KdvSolitonODE;

impl KdvSolitonODE {
    pub const SPEED: f64 = 2.0; // wave speed c
}

impl<S, T> ODE<S, T> for KdvSolitonODE
where
    S: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = S> + Sub<Output = S> + Mul<Output = S> + Div<Output = S>
        + AddAssign + Neg<Output = S> + Trig + From<T> + 'static,
    T: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
        + AddAssign + Neg<Output = T> + 'static,
{
    fn eval(row: usize, _x: S, u: &DVector<S>, d: &DMatrix<T>) -> S {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        let ur  = u[row];

        let c: S       = S::from(T::from_f64(Self::SPEED).unwrap());
        let oneonehalf: S = S::from(T::from_f64(1.5).unwrap());

        d2u[row] - c * ur + oneonehalf * ur * ur
}

    fn forcing(xi: S) -> S {
        S::zero()
    }

    fn bc_left() -> BoundaryCondition<S> {
        BoundaryCondition::Robin {
            alpha: S::from_f64(-Self::SPEED.sqrt()).unwrap(),
            beta:  S::from_f64(1.0).unwrap(),
            value: S::from_f64(0.0).unwrap(),
        }
    }

    fn bc_right() -> BoundaryCondition<S> {
        BoundaryCondition::Robin {
            alpha: S::from_f64(Self::SPEED.sqrt()).unwrap(),
            beta:  S::from_f64(1.0).unwrap(),
            value: S::from_f64(0.0).unwrap(),
        }
    }
    // fn bc_left() -> BoundaryCondition<S> {
    //     BoundaryCondition::RobinQuadratic {
    //         alpha: S::from_f64(-Self::SPEED.sqrt()).unwrap(),
    //         beta:  S::from_f64(1.0).unwrap(),
    //         gamma: S::from_f64(1.0 / (2.0 * Self::SPEED.sqrt())).unwrap(),
    //         value: S::from_f64(0.0).unwrap(),
    //     }
    // }

    // fn bc_right() -> BoundaryCondition<S> {
    //     BoundaryCondition::RobinQuadratic {
    //         alpha: S::from_f64(Self::SPEED.sqrt()).unwrap(),
    //         beta:  S::from_f64(1.0).unwrap(),
    //         gamma: S::from_f64(-1.0 / (2.0 * Self::SPEED.sqrt())).unwrap(),
    //         value: S::from_f64(0.0).unwrap(),
    //     }
    // }
}





/// Baseline verification problem: u'' + u = 0 on [0,1],
/// u(0) = 0, u(1) = sin(1), exact solution u(x) = sin(x).
///
/// Purely a sanity check on the collocation machinery: since u = sin(x)
/// is entire, Bernstein's theorem predicts convergence to machine
/// precision at moderate n (Section 5.2.1). The operator is linear in
/// u, so Newton's method should converge in a single step.
pub struct SineBaselineODE;

impl<S, T> ODE<S, T> for SineBaselineODE
where
    S: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = S> + Sub<Output = S> + Mul<Output = S> + Div<Output = S>
        + AddAssign + Neg<Output = S> + Trig + From<T> + 'static,
    T: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
        + AddAssign + Neg<Output = T> + 'static,
{
    fn eval(row: usize, _x: S, u: &DVector<S>, d: &DMatrix<T>) -> S {
        let du = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        d2u[row] + u[row]
    }

    fn forcing(_xi: S) -> S {
        S::zero()
    }

    fn bc_left() -> BoundaryCondition<S> {
        // u(0) = 0
        BoundaryCondition::Dirichlet(S::zero())
    }

    fn bc_right() -> BoundaryCondition<S> {
        // u(1) = sin(1)
        BoundaryCondition::Dirichlet(S::from_f64(1.0_f64.sin()).unwrap())
    }
}

/// Manufactured solution with a controlled singularity in the complex plane.
///
/// Exact solution u(x) = 1/(1+beta*x^2) on [-1,1], for the linear operator
/// L[u] = u'' + u = f(x). The poles at x = ±i/sqrt(beta) set the size of
/// the Bernstein ellipse of analyticity, so `BETA` directly controls the
/// expected spectral convergence rate (Section 5.2.1-style test problem).
pub struct ManufacturedSolutionODE;

impl ManufacturedSolutionODE {
    pub const BETA: f64 = 4.0; // controls distance of poles x = ±i/sqrt(beta) from [-1,1]
}

impl<S, T> ODE<S, T> for ManufacturedSolutionODE
where
    S: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = S> + Sub<Output = S> + Mul<Output = S> + Div<Output = S>
        + AddAssign + Neg<Output = S> + Trig + From<T> + 'static,
    T: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
        + AddAssign + Neg<Output = T> + 'static,
{
    fn eval(row: usize, _x: S, u: &DVector<S>, d: &DMatrix<T>) -> S {
        let du = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        d2u[row] + u[row]
    }

    fn forcing(xi: S) -> S {
        // f(x) = [beta^2 x^4 + (6 beta^2 + 2 beta) x^2 + (1 - 2 beta)] / (1 + beta x^2)^3
        let beta: S = S::from_f64(Self::BETA).unwrap();
        let one: S = S::from_f64(1.0).unwrap();
        let two: S = S::from_f64(2.0).unwrap();
        let six: S = S::from_f64(6.0).unwrap();

        let x2 = xi * xi;
        let x4 = x2 * x2;
        let beta2 = beta * beta;

        let numer = beta2 * x4 + (six * beta2 + two * beta) * x2 + (one - two * beta);
        let denom_base = one + beta * x2;
        let denom = denom_base * denom_base * denom_base;

        numer / denom
    }

    fn bc_left() -> BoundaryCondition<S> {
        // u(-1) = 1/(1+beta)
        BoundaryCondition::Dirichlet(S::from_f64(1.0 / (1.0 + Self::BETA)).unwrap())
    }

    fn bc_right() -> BoundaryCondition<S> {
        // u(1) = 1/(1+beta)
        BoundaryCondition::Dirichlet(S::from_f64(1.0 / (1.0 + Self::BETA)).unwrap())
    }
}

pub struct CubicReactionManufacturedODE;

impl CubicReactionManufacturedODE {
    pub const BETA: f64 = 4.0; // controls distance of poles x = ±i/sqrt(beta) from [-1,1]
}

impl<S, T> ODE<S, T> for CubicReactionManufacturedODE
where
    S: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = S> + Sub<Output = S> + Mul<Output = S> + Div<Output = S>
        + AddAssign + Neg<Output = S> + Trig + From<T> + 'static,
    T: FromPrimitive + ToPrimitive + Zero + Scalar + Clone + Copy + Sync
        + Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>
        + AddAssign + Neg<Output = T> + 'static,
{
    fn eval(row: usize, _x: S, u: &DVector<S>, d: &DMatrix<T>) -> S {
        let du = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        let ur = u[row];

        // u'' + u^3
        d2u[row] + ur * ur * ur
    }

    fn forcing(xi: S) -> S {
        // Manufactured f(x) = u''(x) + u(x)^3 for u = N/D,
        // N = sin(pi x), D = 1 + beta x^2, via the quotient rule --
        // computed term-by-term rather than hand-simplified, to avoid
        // an algebra slip in a messy closed form.
        let one: S  = S::from_f64(1.0).unwrap();
        let two: S  = S::from_f64(2.0).unwrap();
        let beta: S = S::from_f64(Self::BETA).unwrap();
        let pi: S   = S::from_f64(std::f64::consts::PI).unwrap();

        let n  = (pi * xi).sin();               // N
        let np = pi * (pi * xi).cos();           // N'
        let npp = -(pi * pi) * n;                // N'' = -pi^2 N

        let dd  = one + beta * xi * xi;          // D
        let dp  = two * beta * xi;               // D'
        let dpp = two * beta;                    // D''

        let q  = np * dd - n * dp;               // Q  = N'D - ND'
        let qp = npp * dd - n * dpp;              // Q' = N''D - ND''

        let u_exact  = n / dd;
        let d2u_exact = (qp * dd - two * q * dp) / (dd * dd * dd);

        d2u_exact + u_exact * u_exact * u_exact
    }

    fn bc_left() -> BoundaryCondition<S> {
        // u(-1) = sin(-pi)/(1+beta) = 0
        BoundaryCondition::Dirichlet(S::zero())
    }

    fn bc_right() -> BoundaryCondition<S> {
        // u(1) = sin(pi)/(1+beta) = 0
        BoundaryCondition::Dirichlet(S::zero())
    }
}

