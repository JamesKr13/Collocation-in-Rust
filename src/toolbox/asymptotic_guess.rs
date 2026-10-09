use log::debug;
use nalgebra::coordinates::X;

use super::pb::{HeteroclinicOrbit, EigenPair};
use super::grapher::plot_guess;
use std::cmp::Ordering;

fn critical_curve(u_minus:f64,s_param:f64,_bar_delta:f64,_pm:bool)->f64{
    let dg=1.5*u_minus.powi(2)-3.0*s_param*u_minus+s_param.powi(2)-1.0;
    debug!(
        "u_minus = {u_minus:?}\n\
        s_param = {s_param:?}\n\
        dg = {dg:?}"
    );
    4.0*s_param*dg/(u_minus-s_param)
}

pub fn asymptotic_guess(u_minus: f64, s_param: f64, bar_delta: f64) -> Box<dyn Fn(f64, EigenPair, EigenPair) -> f64> {
    let deg_crit = critical_curve(u_minus, s_param, bar_delta, false);


    match bar_delta.total_cmp(&deg_crit) {
        Ordering::Less => Box::new(move |x: f64, left_eigen_pair: EigenPair, right_eigen_pair: EigenPair| -> f64 {
            let (alpha, beta) = match left_eigen_pair {
                EigenPair::Complex { alpha, beta } => (alpha, beta),
                _ => panic!("Expecting Complex Eigenpair"),
            };

            let (lambda_r1, lambda_r2) = match right_eigen_pair {
                EigenPair::Real { plus: x, minus: y } => (x, y),
                _ => panic!("Expecting Real Eigenpair"),
            };

            let stable_lambda_r = if lambda_r1 < 0.0 { lambda_r1 } else { lambda_r2 };

            let p = alpha.powi(2) + beta.powi(2);
            let d = p - 2.0 * alpha * stable_lambda_r;
            let a = 2.0 * alpha * stable_lambda_r * u_minus / d;
            let b = stable_lambda_r * u_minus * (beta.powi(2) - alpha.powi(2))
                / (beta * d);
            let c = u_minus * p / d;

            if x <= 0.0 {
                u_minus + (alpha * x).exp()
                    * (a * (beta * x).cos() + b * (beta * x).sin())
            } else {
                c * (stable_lambda_r * x).exp()
            }
        }),
        Ordering::Equal => Box::new(move |x: f64, left_eigen_pair: EigenPair, right_eigen_pair: EigenPair| -> f64 {
            let lambda = match left_eigen_pair {
                EigenPair::Real { plus: x, minus: y } => (x + y) / 2.0,
                EigenPair::Complex { alpha, beta } if beta == 0.0 => alpha,
                _ => panic!("Expecting Repeated Real Eigenpair"),
            };

            let (lambda_r1, lambda_r2) = match right_eigen_pair {
                EigenPair::Real { plus: x, minus: y } => (x, y),
                _ => panic!("Expecting Real Eigenpair"),
            };

            let stable_lambda_r = if lambda_r1 < 0.0 { lambda_r1 } else { lambda_r2 };

            let d = lambda.powi(2) - 2.0 * lambda * stable_lambda_r;
            let a = 2.0 * lambda * stable_lambda_r * u_minus / d;
            let b = -stable_lambda_r * u_minus * lambda.powi(2) / d;
            let c = u_minus * lambda.powi(2) / d;

            if x <= 0.0 {
                u_minus + (a + b * x) * (lambda * x).exp()
            } else {
                c * (stable_lambda_r * x).exp()
            }
        }),
        Ordering::Greater => Box::new(move |x: f64, left_eigen_pair: EigenPair, right_eigen_pair: EigenPair| -> f64 {
            let (lambda_l1,lambda_l2)=match left_eigen_pair{
                EigenPair::Real{plus,minus}=>(plus,minus),
                pair=>panic!("Expected real LEFT eigenvalues, got {pair:?}"),
            };

            let (lambda_r1,lambda_r2)=match right_eigen_pair{
                EigenPair::Real{plus,minus}=>(plus,minus),
                pair=>panic!("Expected real RIGHT eigenvalues, got {pair:?}"),
            };

            let stable_lambda_r = if lambda_r1 < 0.0 {lambda_r1} else {lambda_r2};

            let d = lambda_l1 * lambda_l2 - stable_lambda_r * (lambda_l1 + lambda_l2);
            let a = -stable_lambda_r * u_minus * lambda_l2.powi(2) / ((lambda_l1 - lambda_l2) * d);
            let b = stable_lambda_r * u_minus * lambda_l1.powi(2) / ((lambda_l1 - lambda_l2) * d);
            let c = u_minus * lambda_l1 * lambda_l2 / d;

            if x <= 0.0 {
                u_minus + a * (lambda_l1 * x).exp() + b * (lambda_l2 * x).exp()
            } else {
                c * (stable_lambda_r * x).exp()
            }
        })
    }
}


pub fn sanity_check(
    u_minus: f64,
    s_param: f64,
    bar_delta: f64,
    left_eigen_pair: EigenPair,
    right_eigen_pair: EigenPair,
    name: String,
) {

    debug!(
    "u_minus = {u_minus:?}\n\
     s_param = {s_param:?}\n\
     bar_delta = {bar_delta:?}\n\
     critical_curve = {:?}\n\
     left_eigen_pair = {left_eigen_pair:?}\n\
     right_eigen_pair = {right_eigen_pair:?}",
    critical_curve(u_minus, s_param, bar_delta, false),
);

    let guess = asymptotic_guess(u_minus, s_param, bar_delta);

    plot_guess(
        |x| guess(x, left_eigen_pair, right_eigen_pair),
        (-30.0, 5.0),
        2001,
        "Asymptotic initial guess",
        &format!("{}.html", name),
    );
}