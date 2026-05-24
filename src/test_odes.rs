use crate::automatic_derivative::ODE;
use crate::dual::Dual;
use nalgebra::{DVector, DMatrix};
use crate::matrix_op::Operations;
use crate::boundary_conditions::BoundaryCondition;

pub struct BurgersODE;

impl ODE<f64> for BurgersODE {
    // u'' + u *u' - f(x) = 0
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] + u[row] * du[row]
    }

    fn forcing(xi: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let sinpx = (pi * xi).sin();
        let cospx = (pi * xi).cos();
        -1.0 * pi * pi * sinpx + sinpx * pi * cospx
    }

    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

}

impl ODE<Dual<f64>> for BurgersODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        if row == 1 {
        }
        d2u[row] + u[row] * du[row]
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        let pi  = Dual { real: std::f64::consts::PI, eps: 0.0 };
        let pi2 = Dual { real: std::f64::consts::PI * std::f64::consts::PI, eps: 0.0 };
        let sinpx = (pi * xi).sin();
        let cospx = (pi * xi).cos();
        pi2 * sinpx * Dual { real: -1.0, eps: 0.0 } + sinpx * pi * cospx
    }

    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}

pub struct NonlinearDiffusionODE;
impl ODE<f64> for NonlinearDiffusionODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] + u[row] * u[row]
    }
    fn forcing(xi: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let half_pi = pi / 2.0;
        let c = (half_pi * xi).cos();
        -(half_pi * half_pi) * c + c * c
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}
impl ODE<Dual<f64>> for NonlinearDiffusionODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        d2u[row] + u[row] * u[row]
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        let pi      = Dual { real: std::f64::consts::PI, eps: 0.0 };
        let half    = Dual { real: 0.5, eps: 0.0 };
        let half_pi = half * pi;
        let c       = (half_pi * xi).cos();
        Dual { real: -1.0, eps: 0.0 } * half_pi * half_pi * c + c * c
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}
// Fisher-KPP type: u'' + u*(1 - u) = f(x)
// f(x) = x^2
// BCs: u(-1) = 0, u(1) = 0
pub struct FisherODE;
impl ODE<f64> for FisherODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] + u[row] * (1.0 - u[row])
    }
    fn forcing(xi: f64) -> f64 {
        xi * xi
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}
impl ODE<Dual<f64>> for FisherODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        let one = Dual { real: 1.0, eps: 0.0 };
        d2u[row] + u[row] * (one - u[row])
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        xi * xi
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}

// u'' + sin(u) = f(x)
// f(x) = cos(x) 
// BCs: u(-1) = 0, u(1) = 0
pub struct PendulumODE;
impl ODE<f64> for PendulumODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] + u[row].sin()
    }
    fn forcing(xi: f64) -> f64 {
        xi.cos()
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}
impl ODE<Dual<f64>> for PendulumODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        d2u[row] + u[row].sin()
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        xi.cos()
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}

// u'' - u*u'' + u' = f(x)
// f(x) = sin(x)
// BCs: u(-1) = 0, u(1) = 0
pub struct ConvectionDiffusionODE;
impl ODE<f64> for ConvectionDiffusionODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] - u[row] * d2u[row] + du[row]
    }
    fn forcing(xi: f64) -> f64 {
        xi.sin()
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}
impl ODE<Dual<f64>> for ConvectionDiffusionODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        let one = Dual { real: 1.0, eps: 0.0 };
        d2u[row] * (one - u[row]) + du[row]
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        xi.sin()
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}

//Poisson: u'' = f(x)
// Exact solution: u(x) = sin(pi x)
// f(x) = -pi^2 sin(pi*x)
// BCs: u(-1) = 0, u(1) = 0
pub struct PoissonODE;
impl ODE<f64> for PoissonODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row]
    }
    fn forcing(xi: f64) -> f64 {
        let pi = std::f64::consts::PI;
        -pi * pi * (pi * xi).sin()
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}
impl ODE<Dual<f64>> for PoissonODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        d2u[row]
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        let pi  = Dual { real: std::f64::consts::PI, eps: 0.0 };
        let pi2 = Dual { real: std::f64::consts::PI * std::f64::consts::PI, eps: 0.0 };
        Dual { real: -1.0, eps: 0.0 } * pi2 * (pi * xi).sin()
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}