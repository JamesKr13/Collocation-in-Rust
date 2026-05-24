use std::fmt;
use std::ops::{Add, Div, Mul, Sub};
use num_traits::{Float, FromPrimitive, Zero};
use nalgebra::Scalar;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Dual<T> {
    pub real: T,
    pub eps: T,
}

impl<T: Mul<Output=T> + Add<Output=T> + Copy> Mul for Dual<T> {
    type Output = Dual<T>;
    fn mul(self, rhs: Self) -> Dual<T> {
        Dual {
            real: self.real * rhs.real,
            eps: self.real * rhs.eps + self.eps * rhs.real,
        }
    }
}

impl<T: Add<Output=T>> Add for Dual<T> {
    type Output = Dual<T>;
    fn add(self, rhs: Self) -> Dual<T> {
        Dual {
            real: self.real + rhs.real,
            eps: self.eps + rhs.eps,
        }
    }
}

impl<T: Mul<Output=T> + Div<Output=T> + Sub<Output=T> + Copy> Div for Dual<T> {
    type Output = Dual<T>;
    fn div(self, rhs: Dual<T>) -> Dual<T> {
        Dual {
            real: self.real / rhs.real,
            eps: (self.eps * rhs.real - self.real * rhs.eps) / (rhs.real * rhs.real),
        }
    }
}

impl<T: Sub<Output=T>> Sub for Dual<T> {
    type Output = Dual<T>;
    fn sub(self, rhs: Self) -> Dual<T> {
        Dual {
            real: self.real - rhs.real,
            eps: self.eps - rhs.eps,
        }
    }
}

impl<T: Float + FromPrimitive> Dual<T> {
    pub fn sin(self) -> Dual<T> {
        Dual { real: self.real.sin(), eps: self.eps * self.real.cos() }
    }

    pub fn cos(self) -> Dual<T> {
        Dual { real: self.real.cos(), eps: self.eps * (-self.real.sin()) }
    }

    pub fn ln(self) -> Dual<T> {
        Dual { real: self.real.ln(), eps: self.eps / self.real }
    }

    pub fn powi(self, n: i32) -> Dual<T> {
        let n_t = T::from_i32(n).expect("i32 conversion failed");
        Dual {
            real: self.real.powi(n),
            eps: n_t * self.real.powi(n - 1) * self.eps,
        }
    }

    pub fn powf(self, n: T) -> Dual<T> {
        Dual {
            real: self.real.powf(n),
            eps: n * self.real.powf(n - T::one()) * self.eps,
        }
    }
    pub fn exp(self) -> Dual<T> {
        Dual {
            real: T::exp(self.real),
            eps:  T::exp(self.real)* self.eps,
        }
    }
}

impl<T> Mul<f64> for Dual<T>
where
    T: Mul<f64, Output = T>,
{
    type Output = Dual<T>;

    fn mul(self, rhs: f64) -> Self::Output {
        Dual {
            real: self.real * rhs,
            eps: self.eps * rhs,
        }
    }
}

impl<T> Sub<f64> for Dual<T>
where
    T: Sub<f64, Output = T> + Copy,
{
    type Output = Dual<T>;

    fn sub(self, rhs: f64) -> Self::Output {
        Dual {
            real: self.real - rhs,
            eps: self.eps, // derivative unchanged for constant shift
        }
    }
}


impl<T> Zero for Dual<T>
where
    T: Zero + Copy,
{
    fn zero() -> Self {
        Dual {
            real: T::zero(),
            eps: T::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        self.real.is_zero() && self.eps.is_zero()
    }
}

impl<T: FromPrimitive + Zero> FromPrimitive for Dual<T> {
    fn from_i8(n: i8)     -> Option<Self> { T::from_i8(n)    .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_i16(n: i16)   -> Option<Self> { T::from_i16(n)   .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_i32(n: i32)   -> Option<Self> { T::from_i32(n)   .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_i64(n: i64)   -> Option<Self> { T::from_i64(n)   .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_i128(n: i128) -> Option<Self> { T::from_i128(n)  .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_isize(n: isize) -> Option<Self> { T::from_isize(n).map(|r| Dual { real: r, eps: T::zero() }) }

    fn from_u8(n: u8)     -> Option<Self> { T::from_u8(n)    .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_u16(n: u16)   -> Option<Self> { T::from_u16(n)   .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_u32(n: u32)   -> Option<Self> { T::from_u32(n)   .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_u64(n: u64)   -> Option<Self> { T::from_u64(n)   .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_u128(n: u128) -> Option<Self> { T::from_u128(n)  .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_usize(n: usize) -> Option<Self> { T::from_usize(n).map(|r| Dual { real: r, eps: T::zero() }) }

    fn from_f32(n: f32)   -> Option<Self> { T::from_f32(n)   .map(|r| Dual { real: r, eps: T::zero() }) }
    fn from_f64(n: f64)   -> Option<Self> { T::from_f64(n)   .map(|r| Dual { real: r, eps: T::zero() }) }
}

