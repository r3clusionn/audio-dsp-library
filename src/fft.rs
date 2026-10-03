//! Fast Fourier transforms of any length.
//!
//! [`Fft::new`] picks an algorithm from the length:
//!
//! | Length | Algorithm | Cost |
//! |---|---|---|
//! | power of two | iterative radix-2, decimation in time | `n log n` |
//! | prime factors all 13 or less (44,100, 48,000, 1,000, ...) | recursive mixed radix | `n (sum of factors)` |
//! | anything else | Bluestein: a chirp turns the transform into a power-of-two convolution | about 3 power-of-two transforms of length `2n` or more |
//!
//! The forward transform is `X[k] = sum x[j] e^(-2 pi i jk/n)`, unscaled; [`Fft::inverse`] divides by
//! `n`, so `inverse(forward(x)) == x`. This is NumPy's convention.
//!
//! [`RealFft`] transforms real input into the `n / 2 + 1` non-redundant bins, using a complex
//! transform of half the length when `n` is even.

use std::f64::consts::PI;

use crate::complex::Complex;

/// Largest prime factor the mixed-radix path handles; bigger ones go to Bluestein.
const MAX_RADIX: usize = 13;

enum Plan {
    /// Lengths 0 and 1.
    Trivial,
    Pow2(Pow2),
    Mixed(Mixed),
    Bluestein(Box<Bluestein>),
}

/// A transform of one fixed length. Building it precomputes twiddle factors; reuse it.
pub struct Fft {
    n: usize,
    plan: Plan,
}

struct Pow2 {
    n: usize,
    rev: Vec<u32>,
    /// For each stage with half-size `h`, `e^(-2 pi i k / 2h)` for `k < h`, stored at `h - 1 + k`.
    tw: Vec<Complex>,
}

struct Mixed {
    n: usize,
    factors: Vec<usize>,
    /// `e^(-2 pi i k / n)` for every `k < n`.
    tw: Vec<Complex>,
}

struct Bluestein {
    n: usize,
    inner: Pow2,
    /// `e^(-i pi k^2 / n)`.
    chirp: Vec<Complex>,
    /// The transform of the conjugate chirp, laid out for circular convolution.
    kernel: Vec<Complex>,
}

impl Pow2 {
    fn new(n: usize) -> Pow2 {
        debug_assert!(n.is_power_of_two());
        let bits = n.trailing_zeros();
        let rev = (0..n as u32).map(|i| if bits == 0 { 0 } else { i.reverse_bits() >> (32 - bits) }).collect();
        let mut tw = Vec::with_capacity(n.max(1));
        let mut h = 1;
        while h < n {
            for k in 0..h {
                tw.push(Complex::cis(-PI * k as f64 / h as f64));
            }
            h *= 2;
        }
        Pow2 { n, rev, tw }
    }

    fn forward(&self, a: &mut [Complex]) {
        let n = self.n;
        for i in 0..n {
            let j = self.rev[i] as usize;
            if i < j {
                a.swap(i, j);
            }
        }
        if n >= 2 {
            for p in a.as_chunks_mut::<2>().0 {
                let (x, y) = (p[0], p[1]);
                p[0] = x + y;
                p[1] = x - y;
            }
        }
        let mut h = 2;
        while h < n {
            let tw = &self.tw[h - 1..2 * h - 1];
            for block in a.chunks_exact_mut(2 * h) {
                let (lo, hi) = block.split_at_mut(h);
                for ((x, y), w) in lo.iter_mut().zip(hi.iter_mut()).zip(tw) {
                    let t = *y * *w;
                    *y = *x - t;
                    *x += t;
                }
            }
            h *= 2;
        }
    }
}

/// The prime factors of `n` (with 4 used for pairs of 2s), if all are at most `MAX_RADIX`.
fn factor(mut n: usize) -> Option<Vec<usize>> {
    let mut f = Vec::new();
    while n.is_multiple_of(4) {
        f.push(4);
        n /= 4;
    }
    let mut p = 2;
    while n > 1 {
        if p > MAX_RADIX {
            return None;
        }
        while n.is_multiple_of(p) {
            f.push(p);
            n /= p;
        }
        p += 1;
    }
    Some(f)
}

impl Mixed {
    fn new(n: usize, factors: Vec<usize>) -> Mixed {
        let tw = (0..n).map(|k| Complex::cis(-2.0 * PI * k as f64 / n as f64)).collect();
        Mixed { n, factors, tw }
    }

    /// Writes the transform of `input[0], input[stride], ...` (`out.len()` points) into `out`.
    /// `tws` is how far apart this sub-transform's twiddles are in the full table.
    fn rec(&self, input: &[Complex], stride: usize, out: &mut [Complex], factors: &[usize], tws: usize) {
        let len = out.len();
        let p = factors[0];
        let mut t = [Complex::ZERO; MAX_RADIX];
        if factors.len() == 1 {
            // The last factor: one small DFT straight from the strided input.
            for (q, tq) in t.iter_mut().enumerate().take(p) {
                *tq = input[q * stride];
            }
            self.butterfly(&mut t, p);
            out[..p].copy_from_slice(&t[..p]);
            return;
        }
        let m = len / p;
        for q in 0..p {
            self.rec(&input[q * stride..], stride * p, &mut out[q * m..(q + 1) * m], &factors[1..], tws * p);
        }
        // Combine the p sub-transforms with radix-p butterflies.
        for k in 0..m {
            t[0] = out[k];
            for (q, tq) in t.iter_mut().enumerate().take(p).skip(1) {
                *tq = out[q * m + k] * self.tw[q * k * tws];
            }
            self.butterfly(&mut t, p);
            for (r, tr) in t.iter().enumerate().take(p) {
                out[r * m + k] = *tr;
            }
        }
    }

    /// A length-`p` DFT of `t[..p]`, in place.
    #[inline]
    fn butterfly(&self, t: &mut [Complex; MAX_RADIX], p: usize) {
        // -i z
        let mi = |z: Complex| Complex::new(z.im, -z.re);
        match p {
            2 => {
                let (a, b) = (t[0], t[1]);
                t[0] = a + b;
                t[1] = a - b;
            }
            3 => {
                const H: f64 = 0.866_025_403_784_438_6; // sin(2 pi / 3)
                let (s, d) = (t[1] + t[2], t[1] - t[2]);
                let m = t[0] - s.scale(0.5);
                let rot = mi(d).scale(H);
                t[0] += s;
                t[1] = m + rot;
                t[2] = m - rot;
            }
            4 => {
                let (a, b) = (t[0] + t[2], t[0] - t[2]);
                let (c, d) = (t[1] + t[3], t[1] - t[3]);
                let d = mi(d);
                t[0] = a + c;
                t[1] = b + d;
                t[2] = a - c;
                t[3] = b - d;
            }
            5 => {
                const C1: f64 = 0.309_016_994_374_947_45; // cos(2 pi / 5)
                const C2: f64 = -0.809_016_994_374_947_5; // cos(4 pi / 5)
                const S1: f64 = 0.951_056_516_295_153_5; // sin(2 pi / 5)
                const S2: f64 = 0.587_785_252_292_473_1; // sin(4 pi / 5)
                let (a1, b1) = (t[1] + t[4], t[1] - t[4]);
                let (a2, b2) = (t[2] + t[3], t[2] - t[3]);
                let m1 = t[0] + a1.scale(C1) + a2.scale(C2);
                let m2 = t[0] + a1.scale(C2) + a2.scale(C1);
                let n1 = mi(b1.scale(S1) + b2.scale(S2));
                let n2 = mi(b1.scale(S2) - b2.scale(S1));
                t[0] = t[0] + a1 + a2;
                t[1] = m1 + n1;
                t[4] = m1 - n1;
                t[2] = m2 + n2;
                t[3] = m2 - n2;
            }
            _ => {
                let root = self.n / p; // e^(-2 pi i / p) is tw[root]
                let x = *t;
                for (r, tr) in t.iter_mut().enumerate().take(p) {
                    let mut acc = Complex::ZERO;
                    for (q, xq) in x.iter().enumerate().take(p) {
                        acc += *xq * self.tw[((q * r) % p) * root];
                    }
                    *tr = acc;
                }
            }
        }
    }
}

impl Bluestein {
    fn new(n: usize) -> Bluestein {
        let m = (2 * n - 1).next_power_of_two();
        let inner = Pow2::new(m);
        // k^2 mod 2n keeps the angle small, so it stays accurate for large k.
        let chirp: Vec<Complex> = (0..n)
            .map(|k| {
                let k2 = (k as u128 * k as u128 % (2 * n as u128)) as f64;
                Complex::cis(-PI * k2 / n as f64)
            })
            .collect();
        let mut kernel = vec![Complex::ZERO; m];
        kernel[0] = chirp[0].conj();
        for k in 1..n {
            kernel[k] = chirp[k].conj();
            kernel[m - k] = chirp[k].conj();
        }
        inner.forward(&mut kernel);
        Bluestein { n, inner, chirp, kernel }
    }

    fn forward(&self, a: &mut [Complex], scratch: &mut [Complex]) {
        let m = self.inner.n;
        let buf = &mut scratch[..m];
        for (k, b) in buf.iter_mut().enumerate() {
            *b = if k < self.n { a[k] * self.chirp[k] } else { Complex::ZERO };
        }
        self.inner.forward(buf);
        for (b, k) in buf.iter_mut().zip(&self.kernel) {
            *b *= *k;
        }
        // Inverse transform by swapping real and imaginary parts around a forward one.
        swap_parts(buf);
        self.inner.forward(buf);
        swap_parts(buf);
        let scale = 1.0 / m as f64;
        for k in 0..self.n {
            a[k] = buf[k].scale(scale) * self.chirp[k];
        }
    }
}

fn swap_parts(a: &mut [Complex]) {
    for z in a {
        *z = Complex::new(z.im, z.re);
    }
}

impl Fft {
    pub fn new(n: usize) -> Fft {
        let plan = if n <= 1 {
            Plan::Trivial
        } else if n.is_power_of_two() {
            Plan::Pow2(Pow2::new(n))
        } else if let Some(f) = factor(n) {
            Plan::Mixed(Mixed::new(n, f))
        } else {
            Plan::Bluestein(Box::new(Bluestein::new(n)))
        };
        Fft { n, plan }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Which algorithm the plan uses: `"radix-2"`, `"mixed radix"`, `"Bluestein"` or `"trivial"`.
    pub fn algorithm(&self) -> &'static str {
        match self.plan {
            Plan::Trivial => "trivial",
            Plan::Pow2(_) => "radix-2",
            Plan::Mixed(_) => "mixed radix",
            Plan::Bluestein(_) => "Bluestein",
        }
    }

    /// Scratch space, in elements, that `forward_with_scratch` needs.
    pub fn scratch_len(&self) -> usize {
        match &self.plan {
            Plan::Trivial | Plan::Pow2(_) => 0,
            Plan::Mixed(m) => m.n,
            Plan::Bluestein(b) => b.inner.n,
        }
    }

    /// The forward transform, in place. Panics if `buf.len() != self.len()`.
    pub fn forward(&self, buf: &mut [Complex]) {
        let mut scratch = vec![Complex::ZERO; self.scratch_len()];
        self.forward_with_scratch(buf, &mut scratch);
    }

    /// The forward transform with caller-provided scratch (at least `scratch_len()` elements), so
    /// repeated transforms do not allocate.
    pub fn forward_with_scratch(&self, buf: &mut [Complex], scratch: &mut [Complex]) {
        assert_eq!(buf.len(), self.n, "buffer length does not match the plan");
        match &self.plan {
            Plan::Trivial => {}
            Plan::Pow2(p) => p.forward(buf),
            Plan::Mixed(m) => {
                let out = &mut scratch[..m.n];
                m.rec(buf, 1, out, &m.factors, 1);
                buf.copy_from_slice(out);
            }
            Plan::Bluestein(b) => b.forward(buf, scratch),
        }
    }

    /// The inverse transform, scaled by `1 / n`.
    pub fn inverse(&self, buf: &mut [Complex]) {
        self.inverse_unscaled(buf);
        let s = 1.0 / self.n.max(1) as f64;
        for z in buf.iter_mut() {
            *z = z.scale(s);
        }
    }

    /// The inverse transform without the `1 / n` scaling.
    pub fn inverse_unscaled(&self, buf: &mut [Complex]) {
        // ifft(x) = swap(fft(swap(x))), where swap exchanges real and imaginary parts.
        swap_parts(buf);
        self.forward(buf);
        swap_parts(buf);
    }
}

/// A transform of real input of one fixed length, giving the `n / 2 + 1` bins from 0 Hz to Nyquist.
pub struct RealFft {
    n: usize,
    half: Option<Fft>,
    full: Option<Fft>,
    /// `e^(-2 pi i k / n)` for `k <= n / 2`.
    tw: Vec<Complex>,
}

impl RealFft {
    pub fn new(n: usize) -> RealFft {
        if n >= 2 && n.is_multiple_of(2) {
            let tw = (0..=n / 2).map(|k| Complex::cis(-2.0 * PI * k as f64 / n as f64)).collect();
            RealFft { n, half: Some(Fft::new(n / 2)), full: None, tw }
        } else {
            RealFft { n, half: None, full: Some(Fft::new(n)), tw: Vec::new() }
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// The number of output bins: `n / 2 + 1`.
    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }

    /// Transforms `input` (`n` samples) into `n / 2 + 1` bins.
    pub fn forward(&self, input: &[f64]) -> Vec<Complex> {
        assert_eq!(input.len(), self.n, "input length does not match the plan");
        if let Some(full) = &self.full {
            let mut buf: Vec<Complex> = input.iter().map(|&x| Complex::from(x)).collect();
            full.forward(&mut buf);
            buf.truncate(self.bins());
            return buf;
        }
        let h = self.n / 2;
        let half = self.half.as_ref().unwrap();
        // Pack even samples into the real part and odd ones into the imaginary part.
        let mut z: Vec<Complex> = (0..h).map(|k| Complex::new(input[2 * k], input[2 * k + 1])).collect();
        half.forward(&mut z);
        let mut out = vec![Complex::ZERO; h + 1];
        for k in 0..=h {
            // z[k mod h] and z[(h - k) mod h], without a division per bin.
            let zk = if k == h { z[0] } else { z[k] };
            let zr = if k == 0 { z[0] } else { z[h - k] }.conj();
            let even = (zk + zr).scale(0.5);
            // (zk - zr) / 2i
            let d = zk - zr;
            let odd = Complex::new(d.im, -d.re).scale(0.5);
            out[k] = even + self.tw[k] * odd;
        }
        out
    }

    /// Turns `n / 2 + 1` bins back into `n` real samples (scaled, so it inverts `forward`). The
    /// imaginary parts of the 0 Hz and Nyquist bins are ignored, as they must be zero for real input.
    pub fn inverse(&self, spec: &[Complex]) -> Vec<f64> {
        assert_eq!(spec.len(), self.bins(), "spectrum length does not match the plan");
        let n = self.n;
        if let Some(full) = &self.full {
            let mut buf = vec![Complex::ZERO; n];
            for k in 0..n {
                buf[k] = if k < spec.len() { spec[k] } else { spec[n - k].conj() };
            }
            if n > 0 {
                buf[0].im = 0.0;
            }
            full.inverse(&mut buf);
            return buf.iter().map(|z| z.re).collect();
        }
        let h = n / 2;
        let half = self.half.as_ref().unwrap();
        let mut z = vec![Complex::ZERO; h];
        for (k, zk) in z.iter_mut().enumerate() {
            let a = spec[k];
            let b = spec[h - k].conj(); // X[k + h] for a real signal
            let even = (a + b).scale(0.5);
            let odd = ((a - b) * self.tw[k].conj()).scale(0.5);
            *zk = even + Complex::new(-odd.im, odd.re);
        }
        half.inverse(&mut z);
        let mut out = vec![0.0; n];
        for k in 0..h {
            out[2 * k] = z[k].re;
            out[2 * k + 1] = z[k].im;
        }
        out
    }
}

/// The direct `O(n^2)` transform, for testing.
pub fn dft(x: &[Complex]) -> Vec<Complex> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let mut acc = Complex::ZERO;
            for (j, xj) in x.iter().enumerate() {
                let a = ((j as u128 * k as u128) % n as u128) as f64;
                acc += *xj * Complex::cis(-2.0 * PI * a / n as f64);
            }
            acc
        })
        .collect()
}

#[cfg(test)]
mod tests;
