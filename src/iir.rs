//! IIR filters: biquads and cascades of them.
//!
//! * [`Biquad`] has the eight standard audio designs from Robert Bristow-Johnson's *Audio EQ
//!   Cookbook* (low-pass, high-pass, band-pass, notch, all-pass, peaking, low shelf, high shelf) and
//!   runs in transposed direct form II.
//! * [`butter`] and [`cheby1`] design Butterworth and Chebyshev type I filters of any order the way
//!   SciPy does: an analog prototype's poles, a frequency transformation (low-pass, high-pass,
//!   band-pass or band-stop), the bilinear transform with pre-warping, then pairing the poles and
//!   zeros into second-order sections ([`Sos`]). Sections keep high orders numerically stable,
//!   where a single high-order polynomial would not be.

use std::f64::consts::PI;

use crate::complex::Complex;
use crate::fir::DesignError;

/// `y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] - a1 y[n-1] - a2 y[n-2]` (coefficients normalised so
/// that `a0 = 1`), with its state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Biquad {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
    s1: f64,
    s2: f64,
}

struct Rbj {
    cos: f64,
    alpha: f64,
}

fn rbj(f0: f64, q: f64, fs: f64) -> Rbj {
    let w0 = 2.0 * PI * f0 / fs;
    Rbj { cos: w0.cos(), alpha: w0.sin() / (2.0 * q) }
}

impl Biquad {
    /// From un-normalised coefficients `b0 b1 b2` over `a0 a1 a2`.
    pub fn new(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Biquad {
        Biquad { b0: b0 / a0, b1: b1 / a0, b2: b2 / a0, a1: a1 / a0, a2: a2 / a0, s1: 0.0, s2: 0.0 }
    }

    pub fn lowpass(f0: f64, q: f64, fs: f64) -> Biquad {
        let r = rbj(f0, q, fs);
        let b = (1.0 - r.cos) / 2.0;
        Biquad::new(b, 1.0 - r.cos, b, 1.0 + r.alpha, -2.0 * r.cos, 1.0 - r.alpha)
    }

    pub fn highpass(f0: f64, q: f64, fs: f64) -> Biquad {
        let r = rbj(f0, q, fs);
        let b = (1.0 + r.cos) / 2.0;
        Biquad::new(b, -(1.0 + r.cos), b, 1.0 + r.alpha, -2.0 * r.cos, 1.0 - r.alpha)
    }

    /// Band-pass with 0 dB gain at the centre.
    pub fn bandpass(f0: f64, q: f64, fs: f64) -> Biquad {
        let r = rbj(f0, q, fs);
        Biquad::new(r.alpha, 0.0, -r.alpha, 1.0 + r.alpha, -2.0 * r.cos, 1.0 - r.alpha)
    }

    pub fn notch(f0: f64, q: f64, fs: f64) -> Biquad {
        let r = rbj(f0, q, fs);
        Biquad::new(1.0, -2.0 * r.cos, 1.0, 1.0 + r.alpha, -2.0 * r.cos, 1.0 - r.alpha)
    }

    pub fn allpass(f0: f64, q: f64, fs: f64) -> Biquad {
        let r = rbj(f0, q, fs);
        Biquad::new(1.0 - r.alpha, -2.0 * r.cos, 1.0 + r.alpha, 1.0 + r.alpha, -2.0 * r.cos, 1.0 - r.alpha)
    }

    /// A bell: `gain_db` at `f0`, unity far away.
    pub fn peaking(f0: f64, q: f64, gain_db: f64, fs: f64) -> Biquad {
        let r = rbj(f0, q, fs);
        let a = 10f64.powf(gain_db / 40.0);
        Biquad::new(1.0 + r.alpha * a, -2.0 * r.cos, 1.0 - r.alpha * a, 1.0 + r.alpha / a, -2.0 * r.cos, 1.0 - r.alpha / a)
    }

    /// `gain_db` below `f0`, unity above.
    pub fn low_shelf(f0: f64, q: f64, gain_db: f64, fs: f64) -> Biquad {
        let r = rbj(f0, q, fs);
        let a = 10f64.powf(gain_db / 40.0);
        let t = 2.0 * a.sqrt() * r.alpha;
        Biquad::new(
            a * ((a + 1.0) - (a - 1.0) * r.cos + t),
            2.0 * a * ((a - 1.0) - (a + 1.0) * r.cos),
            a * ((a + 1.0) - (a - 1.0) * r.cos - t),
            (a + 1.0) + (a - 1.0) * r.cos + t,
            -2.0 * ((a - 1.0) + (a + 1.0) * r.cos),
            (a + 1.0) + (a - 1.0) * r.cos - t,
        )
    }

    /// `gain_db` above `f0`, unity below.
    pub fn high_shelf(f0: f64, q: f64, gain_db: f64, fs: f64) -> Biquad {
        let r = rbj(f0, q, fs);
        let a = 10f64.powf(gain_db / 40.0);
        let t = 2.0 * a.sqrt() * r.alpha;
        Biquad::new(
            a * ((a + 1.0) + (a - 1.0) * r.cos + t),
            -2.0 * a * ((a - 1.0) + (a + 1.0) * r.cos),
            a * ((a + 1.0) + (a - 1.0) * r.cos - t),
            (a + 1.0) - (a - 1.0) * r.cos + t,
            2.0 * ((a - 1.0) - (a + 1.0) * r.cos),
            (a + 1.0) - (a - 1.0) * r.cos - t,
        )
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        y
    }

    pub fn process_block(&mut self, buf: &mut [f64]) {
        for x in buf {
            *x = self.process(*x);
        }
    }

    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }

    /// The frequency response at `freq_hz`.
    pub fn response(&self, freq_hz: f64, fs: f64) -> Complex {
        let z1 = Complex::cis(-2.0 * PI * freq_hz / fs);
        let z2 = z1 * z1;
        let num = Complex::from(self.b0) + z1.scale(self.b1) + z2.scale(self.b2);
        let den = Complex::ONE + z1.scale(self.a1) + z2.scale(self.a2);
        num / den
    }

    /// True if both poles are inside the unit circle.
    pub fn is_stable(&self) -> bool {
        // The stability triangle for z^2 + a1 z + a2.
        self.a2.abs() < 1.0 && self.a1.abs() < 1.0 + self.a2
    }
}

/// A cascade of biquads ("second-order sections").
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sos {
    pub sections: Vec<Biquad>,
}

impl Sos {
    pub fn process(&mut self, x: f64) -> f64 {
        self.sections.iter_mut().fold(x, |v, s| s.process(v))
    }

    pub fn process_block(&mut self, buf: &mut [f64]) {
        for s in &mut self.sections {
            s.process_block(buf);
        }
    }

    pub fn reset(&mut self) {
        for s in &mut self.sections {
            s.reset();
        }
    }

    pub fn response(&self, freq_hz: f64, fs: f64) -> Complex {
        self.sections.iter().fold(Complex::ONE, |acc, s| acc * s.response(freq_hz, fs))
    }

    pub fn is_stable(&self) -> bool {
        self.sections.iter().all(Biquad::is_stable)
    }
}

/// What a designed filter passes. Frequencies in hertz.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Band {
    Lowpass(f64),
    Highpass(f64),
    Bandpass(f64, f64),
    Bandstop(f64, f64),
}

/// Zeros, poles and gain.
struct Zpk {
    z: Vec<Complex>,
    p: Vec<Complex>,
    k: f64,
}

fn butter_prototype(n: usize) -> Zpk {
    // p = -exp(i pi m / 2n) for m = -n+1, -n+3, ..., n-1
    let p = (0..n).map(|i| -Complex::cis(PI * (2 * i as i64 - n as i64 + 1) as f64 / (2 * n) as f64)).collect();
    Zpk { z: Vec::new(), p, k: 1.0 }
}

fn cheby1_prototype(n: usize, ripple_db: f64) -> Zpk {
    let eps = (10f64.powf(0.1 * ripple_db) - 1.0).sqrt();
    let mu = (1.0 / eps).asinh() / n as f64;
    let p: Vec<Complex> = (0..n)
        .map(|i| {
            let theta = PI * (2 * i as i64 - n as i64 + 1) as f64 / (2 * n) as f64;
            // -sinh(mu + i theta)
            -Complex::new(mu.sinh() * theta.cos(), mu.cosh() * theta.sin())
        })
        .collect();
    let mut k = p.iter().fold(Complex::ONE, |acc, &x| acc * -x).re;
    if n.is_multiple_of(2) {
        k /= (1.0 + eps * eps).sqrt();
    }
    Zpk { z: Vec::new(), p, k }
}

fn prod(v: &[Complex]) -> Complex {
    v.iter().fold(Complex::ONE, |acc, &x| acc * x)
}

fn neg(v: &[Complex]) -> Vec<Complex> {
    v.iter().map(|&x| -x).collect()
}

fn transform(proto: Zpk, band: Band, warp: impl Fn(f64) -> f64) -> Zpk {
    let degree = proto.p.len() - proto.z.len();
    match band {
        Band::Lowpass(f) => {
            let wo = warp(f);
            Zpk {
                z: proto.z.iter().map(|&z| z.scale(wo)).collect(),
                p: proto.p.iter().map(|&p| p.scale(wo)).collect(),
                k: proto.k * wo.powi(degree as i32),
            }
        }
        Band::Highpass(f) => {
            let wo = Complex::from(warp(f));
            let mut z: Vec<Complex> = proto.z.iter().map(|&z| wo / z).collect();
            z.extend(std::iter::repeat_n(Complex::ZERO, degree));
            let k = proto.k * (prod(&neg(&proto.z)) / prod(&neg(&proto.p))).re;
            Zpk { z, p: proto.p.iter().map(|&p| wo / p).collect(), k }
        }
        Band::Bandpass(lo, hi) => {
            let (w1, w2) = (warp(lo), warp(hi));
            let (wo, bw) = ((w1 * w2).sqrt(), w2 - w1);
            let split = |v: &[Complex]| -> Vec<Complex> {
                let lp: Vec<Complex> = v.iter().map(|&x| x.scale(bw / 2.0)).collect();
                let mut out: Vec<Complex> = lp.iter().map(|&x| x + (x * x - Complex::from(wo * wo)).sqrt()).collect();
                out.extend(lp.iter().map(|&x| x - (x * x - Complex::from(wo * wo)).sqrt()));
                out
            };
            let mut z = split(&proto.z);
            z.extend(std::iter::repeat_n(Complex::ZERO, degree));
            Zpk { z, p: split(&proto.p), k: proto.k * bw.powi(degree as i32) }
        }
        Band::Bandstop(lo, hi) => {
            let (w1, w2) = (warp(lo), warp(hi));
            let (wo, bw) = ((w1 * w2).sqrt(), w2 - w1);
            let half = Complex::from(bw / 2.0);
            let mut zh: Vec<Complex> = proto.z.iter().map(|&z| half / z).collect();
            let ph: Vec<Complex> = proto.p.iter().map(|&p| half / p).collect();
            for _ in 0..degree {
                zh.push(Complex::new(0.0, wo));
                zh.push(Complex::new(0.0, -wo));
            }
            // The added zeros are already at +-i wo; only the transformed prototype zeros split.
            let split = |v: &[Complex]| -> Vec<Complex> {
                let mut out: Vec<Complex> = v.iter().map(|&x| x + (x * x - Complex::from(wo * wo)).sqrt()).collect();
                out.extend(v.iter().map(|&x| x - (x * x - Complex::from(wo * wo)).sqrt()));
                out
            };
            let nz = proto.z.len();
            let mut z = split(&zh[..nz]);
            z.extend_from_slice(&zh[nz..]);
            let k = proto.k * (prod(&neg(&proto.z)) / prod(&neg(&proto.p))).re;
            Zpk { z, p: split(&ph), k }
        }
    }
}

/// The bilinear transform with `fs = 2` (frequencies already pre-warped for it).
fn bilinear(a: Zpk) -> Zpk {
    let fs2 = Complex::from(4.0);
    let degree = a.p.len() - a.z.len();
    let mut z: Vec<Complex> = a.z.iter().map(|&z| (fs2 + z) / (fs2 - z)).collect();
    z.extend(std::iter::repeat_n(Complex::from(-1.0), degree));
    let p = a.p.iter().map(|&p| (fs2 + p) / (fs2 - p)).collect();
    let num = prod(&a.z.iter().map(|&z| fs2 - z).collect::<Vec<_>>());
    let den = prod(&a.p.iter().map(|&p| fs2 - p).collect::<Vec<_>>());
    Zpk { z, p, k: a.k * (num / den).re }
}

/// Splits roots into conjugate pairs (by the member with positive imaginary part) and real roots.
fn pairs(v: &[Complex]) -> (Vec<Complex>, Vec<f64>) {
    let tol = 1e-9;
    let complex: Vec<Complex> = v.iter().copied().filter(|z| z.im > tol).collect();
    let mut real: Vec<f64> = v.iter().filter(|z| z.im.abs() <= tol).map(|z| z.re).collect();
    real.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (complex, real)
}

/// Pairs poles with nearby zeros into sections. The product of the sections is the same however
/// they are paired; pairing nearby roots keeps each section's gain moderate.
fn to_sos(d: Zpk) -> Sos {
    let (cp, rp) = pairs(&d.p);
    let (cz, rz) = pairs(&d.z);
    // Each section: two poles (or one) as (quadratic) coefficients.
    let mut pole_secs: Vec<(Complex, [f64; 2])> = cp.iter().map(|&p| (p, [-2.0 * p.re, p.norm_sqr()])).collect();
    for ch in rp.chunks(2) {
        let c = if ch.len() == 2 { [-(ch[0] + ch[1]), ch[0] * ch[1]] } else { [-ch[0], 0.0] };
        pole_secs.push((Complex::from(ch[0]), c));
    }
    let mut zero_secs: Vec<(Complex, [f64; 2], bool)> = cz.iter().map(|&z| (z, [-2.0 * z.re, z.norm_sqr()], true)).collect();
    for ch in rz.chunks(2) {
        let c = if ch.len() == 2 { [-(ch[0] + ch[1]), ch[0] * ch[1]] } else { [-ch[0], 0.0] };
        zero_secs.push((Complex::from(ch[0]), c, ch.len() == 2));
    }
    // Poles closest to the unit circle first; each takes the nearest remaining zero pair.
    pole_secs.sort_by(|a, b| b.0.abs().partial_cmp(&a.0.abs()).unwrap());
    let mut sections = Vec::with_capacity(pole_secs.len());
    for (p, a) in &pole_secs {
        let b = if zero_secs.is_empty() {
            [1.0, 0.0, 0.0]
        } else {
            let (i, _) = zero_secs
                .iter()
                .enumerate()
                .min_by(|x, y| (x.1 .0 - *p).abs().partial_cmp(&(y.1 .0 - *p).abs()).unwrap())
                .unwrap();
            let (_, c, _) = zero_secs.remove(i);
            [1.0, c[0], c[1]]
        };
        sections.push(Biquad::new(b[0], b[1], b[2], 1.0, a[0], a[1]));
    }
    // The overall gain goes on the first section.
    if let Some(s) = sections.first_mut() {
        s.b0 *= d.k;
        s.b1 *= d.k;
        s.b2 *= d.k;
    }
    Sos { sections }
}

fn check(order: usize, band: Band, fs: f64) -> Result<(), DesignError> {
    if order == 0 || order > 64 {
        return Err(DesignError::BadOrder(order));
    }
    let ok = |f: f64| f > 0.0 && f < fs / 2.0;
    let good = match band {
        Band::Lowpass(f) | Band::Highpass(f) => ok(f),
        Band::Bandpass(a, b) | Band::Bandstop(a, b) => ok(a) && ok(b) && a < b,
    };
    if good {
        Ok(())
    } else {
        Err(DesignError::BadCutoff(format!("{band:?} must lie strictly between 0 and {} Hz", fs / 2.0)))
    }
}

fn design(proto: Zpk, band: Band, fs: f64) -> Sos {
    // Pre-warp: the analog frequency that the bilinear transform (at fs = 2) maps to f.
    let warp = |f: f64| 4.0 * (PI * f / fs).tan();
    to_sos(bilinear(transform(proto, band, warp)))
}

/// A Butterworth filter of `order` (band filters have twice as many poles), like
/// `scipy.signal.butter(order, ..., output='sos')`.
pub fn butter(order: usize, band: Band, fs: f64) -> Result<Sos, DesignError> {
    check(order, band, fs)?;
    Ok(design(butter_prototype(order), band, fs))
}

/// A Chebyshev type I filter with `ripple_db` of pass-band ripple, like `scipy.signal.cheby1`.
pub fn cheby1(order: usize, ripple_db: f64, band: Band, fs: f64) -> Result<Sos, DesignError> {
    check(order, band, fs)?;
    // Written this way round so that NaN is refused too.
    if ripple_db.is_nan() || ripple_db <= 0.0 {
        return Err(DesignError::BadCutoff(format!("ripple must be positive, got {ripple_db} dB")));
    }
    Ok(design(cheby1_prototype(order, ripple_db), band, fs))
}

#[cfg(test)]
mod tests;
