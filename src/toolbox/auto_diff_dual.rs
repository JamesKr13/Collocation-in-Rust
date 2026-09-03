use std::ops::{Add, Div, Mul, Sub, AddAssign,Neg};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};


#[warn(dead_code)]
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