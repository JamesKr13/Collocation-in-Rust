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


pub struct ImportantOde;

impl ODE<f64> for ImportantOde {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        let s = 1.5;
        let delta_bar = 5.0;
        -s * u[row] * (u[row] -s) - u[row] + 0.5 * u[row] * u[row] * (u[row] - s) + delta_bar * (u[row]-s) * d2u[row] - delta_bar * (u[row]-s) * du[row]
    }
    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(1.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}

impl ODE<Dual<f64>> for ImportantOde {
    fn eval(row: usize, x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64>
    where Dual<f64>: nalgebra::Scalar + std::ops::Mul<Output=Dual<f64>> + std::ops::Add<Output=Dual<f64>> + Copy
    {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        let s = 1.5;
        let delta_bar = 5.0;
         u[row] * (u[row] -s) * (-s) - u[row] +  u[row] * u[row] * (u[row] - s) * (0.5 as f64) + (u[row]-s) * d2u[row] * delta_bar - (u[row]-s) * du[row] * delta_bar
    }

    fn bc_left() ->  BoundaryCondition {
        BoundaryCondition::Dirichlet(1.0)
    }

    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(0.0)
    }
}

// Allen–Cahn  (bistable phase-field ODE)
// u'' = u^3 – u
// Exact: u(x) = tanh(x / sqrt(2))
// Verify: u'' = –sech^2(x/√2)·tanh(x/sqrt(2)) = u^3 – u
// f(x)  = 0
// BCs:   u(–5) = tanh(–5/sqrt(2)),  u(5) = tanh(5/sqrt(2))
pub struct AllenCahnODE;

impl ODE<f64> for AllenCahnODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] - (u[row].powi(3) - u[row])   // u'' – (u^3 – u)
    }
    fn forcing(_xi: f64) -> f64 { 0.0 }
    fn bc_left() -> BoundaryCondition {
        BoundaryCondition::Dirichlet((-5.0_f64 / 2.0_f64.sqrt()).tanh())
    }
    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(( 5.0_f64 / 2.0_f64.sqrt()).tanh())
    }
}

impl ODE<Dual<f64>> for AllenCahnODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        let ur  = u[row];
        d2u[row] - (ur * ur * ur - ur)
    }
    fn forcing(_xi: Dual<f64>) -> Dual<f64> { Dual { real: 0.0, eps: 0.0 } }
    fn bc_left() -> BoundaryCondition {
        BoundaryCondition::Dirichlet((-5.0_f64 / 2.0_f64.sqrt()).tanh())
    }
    fn bc_right() -> BoundaryCondition {
        BoundaryCondition::Dirichlet(( 5.0_f64 / 2.0_f64.sqrt()).tanh())
    }
}

// Manufactured Quadratic Nonlinearity
// u'' – u^2 = f(x)
// Exact: u(x) = cos(pi *x/2)  on [–1, 1]
// BCs:   u(–1) = 0,  u(1) = 0
// f(x) = u_xx – u^2 = –(pi^2/4)cos(pi *x/2) – cos^2(pi *x/2)
pub struct QuadraticNonlinearODE;

impl ODE<f64> for QuadraticNonlinearODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] - u[row].powi(2)
    }
    fn forcing(xi: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let c  = (pi * xi / 2.0).cos();
        -(pi * pi / 4.0) * c - c * c
    }
    fn bc_left()  -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
    fn bc_right() -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
}

impl ODE<Dual<f64>> for QuadraticNonlinearODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        let ur  = u[row];
        d2u[row] - ur * ur
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        let pi       = Dual { real: std::f64::consts::PI,                    eps: 0.0 };
        let pi2_on4  = Dual { real: std::f64::consts::PI * std::f64::consts::PI / 4.0, eps: 0.0 };
        let two      = Dual { real: 2.0, eps: 0.0 };
        let neg_one  = Dual { real: -1.0, eps: 0.0 };
        let c = (pi * xi / two).cos();
        neg_one * pi2_on4 * c - c * c
    }
    fn bc_left()  -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
    fn bc_right() -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
}


// Duffing-type  (hardening cubic spring)
// u'' + u + u^3 = f(x)
// Exact: u(x) = sin(pi *x)  on [–1, 1]
// BCs:   u(–1) = 0,  u(1) = 0
// f(x) = –pi^2sin(pi *x) + sin(pi *x) + sin^3(pi *x)
//      = (1 – pi^2)sin(pi *x) + sin^3(pi *x)
pub struct DuffingODE;

impl ODE<f64> for DuffingODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] + u[row] + u[row].powi(3)
    }
    fn forcing(xi: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let s  = (pi * xi).sin();
        (1.0 - pi * pi) * s + s.powi(3)
    }
    fn bc_left()  -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
    fn bc_right() -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
}

impl ODE<Dual<f64>> for DuffingODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        let ur  = u[row];
        d2u[row] + ur + ur * ur * ur
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        let pi   = Dual { real: std::f64::consts::PI,                    eps: 0.0 };
        let pi2  = Dual { real: std::f64::consts::PI * std::f64::consts::PI, eps: 0.0 };
        let one  = Dual { real:  1.0, eps: 0.0 };
        let s    = (pi * xi).sin();
        (one - pi2) * s + s * s * s
    }
    fn bc_left()  -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
    fn bc_right() -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
}

// Bratu-type  (exponential nonlinearity)
// u'' – e^u = f(x)
// Exact: u(x) = sin(pi * x)  on [–1, 1]
// BCs:   u(–1) = 0,  u(1) = 0
// f(x) = –pi^2sin(pi x) – exp(sin(pi * x))=
pub struct BratuODE;

impl ODE<f64> for BratuODE {
    fn eval(row: usize, _x: f64, u: &DVector<f64>, d: &DMatrix<f64>) -> f64 {
        let du  = d * u;
        let d2u = d * &du;
        d2u[row] - u[row].exp()
    }
    fn forcing(xi: f64) -> f64 {
        let pi = std::f64::consts::PI;
        let s  = (pi * xi).sin();
        -pi * pi * s - s.exp()
    }
    fn bc_left()  -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
    fn bc_right() -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
}

impl ODE<Dual<f64>> for BratuODE {
    fn eval(row: usize, _x: Dual<f64>, u: &DVector<Dual<f64>>, d: &DMatrix<f64>) -> Dual<f64> {
        let du  = Operations::apply_matrix(d, u);
        let d2u = Operations::apply_matrix(d, &du);
        d2u[row] - u[row].exp()
    }
    fn forcing(xi: Dual<f64>) -> Dual<f64> {
        let pi  = Dual { real: std::f64::consts::PI,                    eps: 0.0 };
        let pi2 = Dual { real: std::f64::consts::PI * std::f64::consts::PI, eps: 0.0 };
        let neg = Dual { real: -1.0, eps: 0.0 };
        let s   = (pi * xi).sin();
        neg * pi2 * s - s.exp()
    }
    fn bc_left()  -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
    fn bc_right() -> BoundaryCondition { BoundaryCondition::Dirichlet(0.0) }
}