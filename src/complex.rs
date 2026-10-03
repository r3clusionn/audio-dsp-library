//! A minimal complex number: just what the FFT and the filter responses need.

use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };

    #[inline(always)]
    pub const fn new(re: f64, im: f64) -> Complex {
        Complex { re, im }
    }

    /// `r * (cos theta + i sin theta)`.
    #[inline]
    pub fn from_polar(r: f64, theta: f64) -> Complex {
        let (s, c) = theta.sin_cos();
        Complex { re: r * c, im: r * s }
    }

    /// `e^(i theta)`.
    #[inline]
    pub fn cis(theta: f64) -> Complex {
        Complex::from_polar(1.0, theta)
    }

    #[inline(always)]
    pub fn conj(self) -> Complex {
        Complex { re: self.re, im: -self.im }
    }

    /// The squared magnitude, without the square root.
    #[inline(always)]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    #[inline]
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// The angle in radians, in `(-pi, pi]`.
    #[inline]
    pub fn arg(self) -> f64 {
        self.im.atan2(self.re)
    }

    #[inline(always)]
    pub fn scale(self, k: f64) -> Complex {
        Complex { re: self.re * k, im: self.im * k }
    }

    /// `e^self`.
    pub fn exp(self) -> Complex {
        Complex::from_polar(self.re.exp(), self.im)
    }

    /// The principal square root.
    pub fn sqrt(self) -> Complex {
        let r = self.abs();
        if r == 0.0 {
            return Complex::ZERO;
        }
        let re = ((r + self.re) / 2.0).sqrt();
        let im = ((r - self.re) / 2.0).sqrt();
        Complex { re, im: if self.im < 0.0 { -im } else { im } }
    }

    /// `1 / self`.
    pub fn recip(self) -> Complex {
        let d = self.norm_sqr();
        Complex { re: self.re / d, im: -self.im / d }
    }

    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl From<f64> for Complex {
    fn from(re: f64) -> Complex {
        Complex { re, im: 0.0 }
    }
}

impl Add for Complex {
    type Output = Complex;
    #[inline(always)]
    fn add(self, o: Complex) -> Complex {
        Complex { re: self.re + o.re, im: self.im + o.im }
    }
}

impl Sub for Complex {
    type Output = Complex;
    #[inline(always)]
    fn sub(self, o: Complex) -> Complex {
        Complex { re: self.re - o.re, im: self.im - o.im }
    }
}

impl Mul for Complex {
    type Output = Complex;
    #[inline(always)]
    fn mul(self, o: Complex) -> Complex {
        Complex { re: self.re * o.re - self.im * o.im, im: self.re * o.im + self.im * o.re }
    }
}

impl Mul<f64> for Complex {
    type Output = Complex;
    #[inline(always)]
    fn mul(self, k: f64) -> Complex {
        self.scale(k)
    }
}

// Division by multiplying with the reciprocal is the definition, not a typo.
#[allow(clippy::suspicious_arithmetic_impl)]
impl Div for Complex {
    type Output = Complex;
    fn div(self, o: Complex) -> Complex {
        self * o.recip()
    }
}

impl Neg for Complex {
    type Output = Complex;
    #[inline(always)]
    fn neg(self) -> Complex {
        Complex { re: -self.re, im: -self.im }
    }
}

impl AddAssign for Complex {
    #[inline(always)]
    fn add_assign(&mut self, o: Complex) {
        *self = *self + o;
    }
}

impl SubAssign for Complex {
    #[inline(always)]
    fn sub_assign(&mut self, o: Complex) {
        *self = *self - o;
    }
}

impl MulAssign for Complex {
    #[inline(always)]
    fn mul_assign(&mut self, o: Complex) {
        *self = *self * o;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Complex, b: Complex) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, -2.0);
        assert_eq!(a + b, Complex::new(4.0, 2.0));
        assert_eq!(a - b, Complex::new(2.0, 6.0));
        assert_eq!(a * b, Complex::new(11.0, -2.0));
        assert!(close((a * b) / b, a));
        assert_eq!(a.conj(), Complex::new(3.0, -4.0));
        assert_eq!(a.abs(), 5.0);
        assert_eq!(a.norm_sqr(), 25.0);
        assert_eq!(-a, Complex::new(-3.0, -4.0));
    }

    #[test]
    fn polar_and_exp() {
        let z = Complex::from_polar(2.0, std::f64::consts::FRAC_PI_3);
        assert!((z.abs() - 2.0).abs() < 1e-15);
        assert!((z.arg() - std::f64::consts::FRAC_PI_3).abs() < 1e-15);
        assert!(close(Complex::new(0.0, std::f64::consts::PI).exp(), Complex::new(-1.0, 0.0)));
        assert!(close(Complex::cis(0.5) * Complex::cis(0.25), Complex::cis(0.75)));
    }

    #[test]
    fn square_roots_cover_every_quadrant() {
        for z in [
            Complex::new(3.0, 4.0),
            Complex::new(-3.0, 4.0),
            Complex::new(-3.0, -4.0),
            Complex::new(3.0, -4.0),
            Complex::new(-9.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(2.0, 0.0),
        ] {
            let r = z.sqrt();
            assert!(close(r * r, z), "{z:?}");
            assert!(r.re >= 0.0, "principal root has a non-negative real part: {z:?} -> {r:?}");
        }
    }
}
