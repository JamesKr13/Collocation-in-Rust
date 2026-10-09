use std::ops::{Add, Div, Mul, Sub, AddAssign,Neg};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use nalgebra::Scalar;
use std::num::FpCategory;
use crate::toolbox::banded_solver::BandedLinSolve;

#[warn(dead_code)]
#[derive(Clone, Copy, Debug)]
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

impl<T: Float> Float for Dual<T> {
    fn nan() -> Self { Dual { real: T::nan(), eps: T::zero() } }
    fn infinity() -> Self { Dual { real: T::infinity(), eps: T::zero() } }
    fn neg_infinity() -> Self { Dual { real: T::neg_infinity(), eps: T::zero() } }
    fn neg_zero() -> Self { Dual { real: T::neg_zero(), eps: T::zero() } }
    fn min_value() -> Self { Dual { real: T::min_value(), eps: T::zero() } }
    fn min_positive_value() -> Self { Dual { real: T::min_positive_value(), eps: T::zero() } }
    fn max_value() -> Self { Dual { real: T::max_value(), eps: T::zero() } }
    fn epsilon() -> Self { Dual { real: T::epsilon(), eps: T::zero() } }

    // Classification only depends on the value, never the derivative.
    fn is_nan(self) -> bool { self.real.is_nan() }
    fn is_infinite(self) -> bool { self.real.is_infinite() }
    fn is_finite(self) -> bool { self.real.is_finite() }
    fn is_normal(self) -> bool { self.real.is_normal() }
    fn classify(self) -> FpCategory { self.real.classify() }
    fn is_sign_positive(self) -> bool { self.real.is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.real.is_sign_negative() }
    fn integer_decode(self) -> (u64, i16, i8) { self.real.integer_decode() }

    // Piecewise-constant ops: derivative is zero almost everywhere.
    fn floor(self) -> Self { Dual { real: self.real.floor(), eps: T::zero() } }
    fn ceil(self) -> Self { Dual { real: self.real.ceil(), eps: T::zero() } }
    fn round(self) -> Self { Dual { real: self.real.round(), eps: T::zero() } }
    fn trunc(self) -> Self { Dual { real: self.real.trunc(), eps: T::zero() } }
    fn signum(self) -> Self { Dual { real: self.real.signum(), eps: T::zero() } }

    // fract(x) = x - floor(x); slope 1 a.e.
    fn fract(self) -> Self { Dual { real: self.real.fract(), eps: self.eps } }

    fn abs(self) -> Self {
        if self.real >= T::zero() { self } else { Dual { real: -self.real, eps: -self.eps } }
    }

    fn max(self, other: Self) -> Self { if self.real >= other.real { self } else { other } }
    fn min(self, other: Self) -> Self { if self.real <= other.real { self } else { other } }

    fn abs_sub(self, other: Self) -> Self {
        if self.real > other.real {
            Dual { real: self.real - other.real, eps: self.eps - other.eps }
        } else {
            Dual::zero()
        }
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        // (self*a)+b: let the existing Mul/Add propagate the derivative.
        Dual {
            real: self.real * a.real + b.real,
            eps: self.eps * a.real + self.real * a.eps + b.eps,
        }
    }

    fn recip(self) -> Self {
        let r = self.real.recip();
        Dual { real: r, eps: -self.eps * r * r }
    }

    fn powi(self, n: i32) -> Self {
        let n_t = T::from(n).unwrap();
        Dual { real: self.real.powi(n), eps: n_t * self.real.powi(n - 1) * self.eps }
    }

    fn powf(self, n: Self) -> Self {
        // x^y ; d(x^y) = y*x^(y-1)*dx + x^y*ln(x)*dy
        let xy = self.real.powf(n.real);
        let dx_term = n.real * self.real.powf(n.real - T::one()) * self.eps;
        let dy_term = if self.real > T::zero() {
            xy * self.real.ln() * n.eps
        } else {
            T::zero()
        };
        Dual { real: xy, eps: dx_term + dy_term }
    }

    fn sqrt(self) -> Self {
        let r = self.real.sqrt();
        Dual { real: r, eps: self.eps / (T::from(2.0).unwrap() * r) }
    }

    fn cbrt(self) -> Self {
        let r = self.real.cbrt();
        Dual { real: r, eps: self.eps / (T::from(3.0).unwrap() * r * r) }
    }

    fn exp(self) -> Self {
        let e = self.real.exp();
        Dual { real: e, eps: self.eps * e }
    }
    fn exp2(self) -> Self {
        let e = self.real.exp2();
        Dual { real: e, eps: self.eps * e * T::from(std::f64::consts::LN_2).unwrap() }
    }
    fn exp_m1(self) -> Self {
        let e = self.real.exp();
        Dual { real: self.real.exp_m1(), eps: self.eps * e }
    }

    fn ln(self) -> Self { Dual { real: self.real.ln(), eps: self.eps / self.real } }
    fn ln_1p(self) -> Self {
        Dual { real: self.real.ln_1p(), eps: self.eps / (T::one() + self.real) }
    }
    fn log(self, base: Self) -> Self { self.ln() / base.ln() }
    fn log2(self) -> Self {
        let ln2 = T::from(std::f64::consts::LN_2).unwrap();
        Dual { real: self.real.log2(), eps: self.eps / (self.real * ln2) }
    }
    fn log10(self) -> Self {
        let ln10 = T::from(std::f64::consts::LN_10).unwrap();
        Dual { real: self.real.log10(), eps: self.eps / (self.real * ln10) }
    }

    fn to_degrees(self) -> Self {
        let f = T::from(180.0 / std::f64::consts::PI).unwrap();
        Dual { real: self.real.to_degrees(), eps: self.eps * f }
    }
    fn to_radians(self) -> Self {
        let f = T::from(std::f64::consts::PI / 180.0).unwrap();
        Dual { real: self.real.to_radians(), eps: self.eps * f }
    }

    fn hypot(self, other: Self) -> Self {
        let h = self.real.hypot(other.real);
        Dual { real: h, eps: (self.real * self.eps + other.real * other.eps) / h }
    }

    fn sin(self) -> Self { Dual { real: self.real.sin(), eps: self.eps * self.real.cos() } }
    fn cos(self) -> Self { Dual { real: self.real.cos(), eps: -self.eps * self.real.sin() } }
    fn tan(self) -> Self {
        let t = self.real.tan();
        Dual { real: t, eps: self.eps * (T::one() + t * t) }
    }
    fn asin(self) -> Self {
        let d = (T::one() - self.real * self.real).sqrt();
        Dual { real: self.real.asin(), eps: self.eps / d }
    }
    fn acos(self) -> Self {
        let d = (T::one() - self.real * self.real).sqrt();
        Dual { real: self.real.acos(), eps: -self.eps / d }
    }
    fn atan(self) -> Self {
        Dual { real: self.real.atan(), eps: self.eps / (T::one() + self.real * self.real) }
    }
    fn atan2(self, other: Self) -> Self {
        let d = self.real * self.real + other.real * other.real;
        Dual {
            real: self.real.atan2(other.real),
            eps: (other.real * self.eps - self.real * other.eps) / d,
        }
    }
    fn sin_cos(self) -> (Self, Self) { (self.sin(), self.cos()) }

    fn sinh(self) -> Self { Dual { real: self.real.sinh(), eps: self.eps * self.real.cosh() } }
    fn cosh(self) -> Self { Dual { real: self.real.cosh(), eps: self.eps * self.real.sinh() } }
    fn tanh(self) -> Self {
        let t = self.real.tanh();
        Dual { real: t, eps: self.eps * (T::one() - t * t) }
    }
    fn asinh(self) -> Self {
        let d = (self.real * self.real + T::one()).sqrt();
        Dual { real: self.real.asinh(), eps: self.eps / d }
    }
    fn acosh(self) -> Self {
        let d = (self.real * self.real - T::one()).sqrt();
        Dual { real: self.real.acosh(), eps: self.eps / d }
    }
    fn atanh(self) -> Self {
        Dual { real: self.real.atanh(), eps: self.eps / (T::one() - self.real * self.real) }
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

impl<T: ToPrimitive> ToPrimitive for Dual<T> {
    fn to_i64(&self) -> Option<i64> { self.real.to_i64() }
    fn to_u64(&self) -> Option<u64> { self.real.to_u64() }

    fn to_i8(&self) -> Option<i8> { self.real.to_i8() }
    fn to_i16(&self) -> Option<i16> { self.real.to_i16() }
    fn to_i32(&self) -> Option<i32> { self.real.to_i32() }
    fn to_i128(&self) -> Option<i128> { self.real.to_i128() }
    fn to_isize(&self) -> Option<isize> { self.real.to_isize() }

    fn to_u8(&self) -> Option<u8> { self.real.to_u8() }
    fn to_u16(&self) -> Option<u16> { self.real.to_u16() }
    fn to_u32(&self) -> Option<u32> { self.real.to_u32() }
    fn to_u128(&self) -> Option<u128> { self.real.to_u128() }
    fn to_usize(&self) -> Option<usize> { self.real.to_usize() }

    fn to_f32(&self) -> Option<f32> { self.real.to_f32() }
    fn to_f64(&self) -> Option<f64> { self.real.to_f64() }
}

impl<T: AddAssign> AddAssign for Dual<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.real += rhs.real;
        self.eps += rhs.eps;
    }
}

impl<T: Neg<Output = T>> Neg for Dual<T> {
    type Output = Dual<T>;

    fn neg(self) -> Self::Output {
        Dual {
            real: -self.real,
            eps: -self.eps,
        }
    }
}

impl<T: num_traits::Zero> From<T> for Dual<T> {
    fn from(real: T) -> Self {
        Dual { real, eps: T::zero() }
    }
}

impl<T: PartialEq> PartialEq for Dual<T> {
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real
    }
}

impl<T: PartialOrd> PartialOrd for Dual<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.real.partial_cmp(&other.real)
    }
}

impl<T> BandedLinSolve for Dual<T>
where
    T: num_traits::Zero + Scalar + Copy + PartialOrd
        + Add<Output = T> + Sub<Output = T> + Mul<Output = T>
        + Div<Output = T> + Neg<Output = T>,
{}

impl<T: Float> num_traits::One for Dual<T> {
    fn one() -> Self {
        Dual { real: T::one(), eps: T::zero() }
    }
}

impl<T: Float> std::ops::Rem for Dual<T> {
    type Output = Self;
    fn rem(self, other: Self) -> Self {       
        Dual { real: self.real % other.real, eps: self.eps }
    }
}

impl<T: Float> num_traits::Num for Dual<T> {
    type FromStrRadixErr = T::FromStrRadixErr;
    fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(s, radix).map(|r| Dual { real: r, eps: T::zero() })
    }
}

impl<T: Float> num_traits::NumCast for Dual<T> {
    fn from<U: num_traits::ToPrimitive>(n: U) -> Option<Self> {
        T::from(n).map(|r| Dual { real: r, eps: T::zero() })
    }
}

