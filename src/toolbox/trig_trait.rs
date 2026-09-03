use nalgebra::Scalar;
use num_traits::{FromPrimitive, ToPrimitive, Zero};
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

use super::auto_diff_dual::Dual;

/// Trait for the trig ops eval/forcing need, since none of the
/// numeric bounds elsewhere imply `.sin()`/`.cos()`.
pub trait Trig {
    fn sin(self) -> Self;
    fn cos(self) -> Self;
}

// Base case: real scalars.
impl Trig for f64 {
    fn sin(self) -> Self { f64::sin(self) }
    fn cos(self) -> Self { f64::cos(self) }
}

impl Trig for f32 {
    fn sin(self) -> Self { f32::sin(self) }
    fn cos(self) -> Self { f32::cos(self) }
}

// Generic case: Dual<T> for any T that supports Trig + the arithmetic
// needed for the chain rule (sin(a+bε) = sin(a) + b·cos(a)ε, etc).
impl<T> Trig for Dual<T>
where
    T: Trig
        + Scalar
        + Copy
        + Clone
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + Neg<Output = T>,
{
    fn sin(self) -> Self {
        // adjust field names (`re`/`eps`, `val`/`dual`, etc.) to your Dual struct
        Dual {
            real: self.real.sin(),
            eps: self.eps * self.real.cos(),
        }
    }

    fn cos(self) -> Self {
        Dual {
            real: self.real.cos(),
            eps: -self.eps * self.real.sin(),
        }
    }
}