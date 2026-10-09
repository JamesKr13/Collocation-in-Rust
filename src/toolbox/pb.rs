use nalgebra::{DMatrix,DVector,Scalar};
use num_traits::{FromPrimitive,ToPrimitive,Zero};
use std::ops::{Add,AddAssign,Div,Mul,Neg,Sub};
use std::sync::RwLock;

use super::dual_matrix_operations::Operations;
use super::auto_diff_dual::Dual;
use super::trig_trait::Trig;

use crate::collocation_method;
use collocation_method::ode_trait::ODE;
use collocation_method::boundary_conditions::BoundaryCondition;

static PARAMS:RwLock<(f64,f64)>=RwLock::new((1.01,1.4));

pub struct HeteroclinicOrbit;

#[derive(Debug,Clone,Copy)]
pub enum EigenPair{
    Real{plus:f64,minus:f64},
    Complex{alpha:f64,beta:f64},
}

impl HeteroclinicOrbit{
    pub fn set_params(s:f64,delta:f64){*PARAMS.write().unwrap()=(s,delta);}
    fn s_param()->f64{PARAMS.read().unwrap().0}
    fn delta_bar()->f64{PARAMS.read().unwrap().1}

    pub fn n_val(u:f64)->f64{
        let s=Self::s_param();
        1.0-u/s-1.0/(u-s).powi(2)
    }

    pub fn c_val(u:f64)->f64{
        let s=Self::s_param();
        2.0/(u-s).powi(3)-1.0/s
    }

    pub fn eigen_pair(u:f64)->EigenPair{
        let s=Self::s_param();
        let g_u=Self::n_val(u)/Self::delta_bar();
        let disc=(1.0/s).powi(2)+4.0*g_u;
        let alpha=1.0/(2.0*s);

        if disc>=0.0{
            let r=disc.sqrt()/2.0;
            EigenPair::Real{plus:alpha+r,minus:alpha-r}
        }else{
            EigenPair::Complex{
                alpha,
                beta:(-disc).sqrt()/2.0,
            }
        }
    }

    pub fn lambda_val(u:f64,branch:f64)->Option<f64>{
        match Self::eigen_pair(u){
            EigenPair::Real{plus,minus}=>
                Some(if branch>=0.0{plus}else{minus}),
            EigenPair::Complex{..}=>None,
        }
    }

    pub fn k_val(u:f64,lambda:f64)->f64{
        let s=Self::s_param();
        let delta=Self::delta_bar();
        let g_u=Self::n_val(u)/delta;
        let g_uu=Self::c_val(u)/delta;
        lambda*g_uu/(2.0*lambda/s+3.0*g_u)
    }

    pub fn u_minus()->f64{
        let s=Self::s_param();
        (3.0*s-(s*s+8.0).sqrt())/2.0
    }

    /*
        Characteristic equation

            lambda^2 - sum lambda + product = 0.

        For a complex pair lambda=alpha +- i beta,

            sum     = 2 alpha,
            product = alpha^2 + beta^2,

        so this remains entirely real.
    */
    pub fn char_eq_coeffs(u:f64)->(f64,f64){
        match Self::eigen_pair(u){
            EigenPair::Real{plus,minus}=>
                (plus+minus,plus*minus),

            EigenPair::Complex{alpha,beta}=>
                (2.0*alpha,alpha*alpha+beta*beta),
        }
    }
}

impl<S,T> ODE<S,T> for HeteroclinicOrbit
where
    S:FromPrimitive+ToPrimitive+Zero+Scalar+Clone+Copy+Sync
        +Add<Output=S>+Sub<Output=S>+Mul<Output=S>+Div<Output=S>
        +AddAssign+Neg<Output=S>+Trig+From<T>+'static,
    T:FromPrimitive+ToPrimitive+Zero+Scalar+Clone+Copy+Sync
        +Add<Output=T>+Sub<Output=T>+Mul<Output=T>+Div<Output=T>
        +AddAssign+Neg<Output=T>+'static,
{
    fn eval(row:usize,_x:S,u:&DVector<S>,d:&DMatrix<T>)->S{
        let du=Operations::apply_matrix(d,u);
        let d2u=Operations::apply_matrix(d,&du);
        let ur=u[row];
        let dur=du[row];
        let d2ur=d2u[row];

        let s:S=<S as From<T>>::from(T::from_f64(Self::s_param()).unwrap());
        let delta:S=<S as From<T>>::from(T::from_f64(Self::delta_bar()).unwrap());
        let half=S::from_f64(0.5).unwrap();

        let ums=ur-s;
        let algebraic=-s*ur*ums-ur+half*ur*ur*ums;
        let derivative_terms=delta*s*ums*d2ur-delta*ums*dur;

        algebraic+derivative_terms
    }

    fn forcing(_xi:S)->S{S::zero()}

    fn bc_left()->BoundaryCondition<S>{
        let u_star=Self::u_minus();
        let(sum,product)=Self::char_eq_coeffs(u_star);

        BoundaryCondition::CharacteristicEq{
            sum:S::from_f64(sum).unwrap(),
            product:S::from_f64(product).unwrap(),
            value:S::from_f64(u_star).unwrap(),
        }
    }

    fn bc_right()->BoundaryCondition<S>{
        /*
            The right Robin condition selects a single real
            eigendirection. If the right equilibrium ever develops
            a complex pair, this BC is no longer the correct form.
        */
        let lambda=match Self::eigen_pair(0.0){
            EigenPair::Real{minus,..}=>minus,

            EigenPair::Complex{alpha,beta}=>panic!(
                "Right equilibrium has complex eigenvalues \
                 lambda={alpha} +/- {beta}i; a single Robin \
                 eigendirection condition is not applicable."
            ),
        };

        BoundaryCondition::Robin{
            alpha:-S::from_f64(lambda).unwrap(),
            beta:S::from_f64(1.0).unwrap(),
            value:S::zero(),
        }
    }
}