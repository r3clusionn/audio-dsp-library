//! Window functions.
//!
//! Each window comes in two forms, as in SciPy: *symmetric* (the first and last points mirror each
//! other; right for designing FIR filters) and *periodic* (one point of the symmetric window of
//! length `n + 1` is dropped, so the window repeats seamlessly; right for spectral analysis).

use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Window {
    Rectangular,
    Hann,
    Hamming,
    Blackman,
    BlackmanHarris,
    Nuttall,
    FlatTop,
    /// A triangle that reaches zero at both ends.
    Bartlett,
    /// `beta` trades main-lobe width for side-lobe level; see [`kaiser_beta`].
    Kaiser(f64),
}

/// The modified Bessel function of the first kind, order zero, by its power series.
pub fn bessel_i0(x: f64) -> f64 {
    let q = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = 1.0;
    let mut k = 1.0;
    while term > sum * 1e-17 {
        term *= q / (k * k);
        sum += term;
        k += 1.0;
        if k > 500.0 {
            break;
        }
    }
    sum
}

/// Kaiser's formula: the `beta` that gives `atten_db` decibels of stop-band attenuation.
pub fn kaiser_beta(atten_db: f64) -> f64 {
    if atten_db > 50.0 {
        0.1102 * (atten_db - 8.7)
    } else if atten_db >= 21.0 {
        0.5842 * (atten_db - 21.0).powf(0.4) + 0.07886 * (atten_db - 21.0)
    } else {
        0.0
    }
}

/// A sum of cosines `a0 - a1 cos(x) + a2 cos(2x) - ...` over `x in [0, 2 pi]`.
fn general_cosine(m: usize, coeffs: &[f64]) -> Vec<f64> {
    if m == 1 {
        return vec![1.0];
    }
    let denom = (m - 1) as f64;
    (0..m)
        .map(|i| {
            let x = 2.0 * PI * i as f64 / denom;
            coeffs.iter().enumerate().map(|(k, a)| if k % 2 == 0 { *a } else { -*a } * (k as f64 * x).cos()).sum()
        })
        .collect()
}

impl Window {
    /// The window as `n` samples. `periodic` selects the spectral-analysis form.
    pub fn generate(self, n: usize, periodic: bool) -> Vec<f64> {
        if n == 0 {
            return Vec::new();
        }
        // The symmetric window of one more point, with the last point dropped, is the periodic one.
        let m = if periodic && n > 1 { n + 1 } else { n };
        let mut w = self.symmetric(m);
        w.truncate(n);
        w
    }

    fn symmetric(self, m: usize) -> Vec<f64> {
        match self {
            Window::Rectangular => vec![1.0; m],
            Window::Hann => general_cosine(m, &[0.5, 0.5]),
            Window::Hamming => general_cosine(m, &[0.54, 0.46]),
            Window::Blackman => general_cosine(m, &[0.42, 0.5, 0.08]),
            Window::BlackmanHarris => general_cosine(m, &[0.35875, 0.48829, 0.14128, 0.01168]),
            Window::Nuttall => general_cosine(m, &[0.3635819, 0.4891775, 0.1365995, 0.0106411]),
            Window::FlatTop => general_cosine(m, &[0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]),
            Window::Bartlett => {
                if m == 1 {
                    return vec![1.0];
                }
                let half = (m - 1) as f64 / 2.0;
                (0..m).map(|i| 1.0 - ((i as f64 - half) / half).abs()).collect()
            }
            Window::Kaiser(beta) => {
                if m == 1 {
                    return vec![1.0];
                }
                let alpha = (m - 1) as f64 / 2.0;
                let d = bessel_i0(beta);
                (0..m)
                    .map(|i| {
                        let r = (i as f64 - alpha) / alpha;
                        bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / d
                    })
                    .collect()
            }
        }
    }
}

/// The sum of a window's samples divided by its length: how much a steady tone's amplitude is
/// reduced. Divide a windowed spectrum's magnitudes by `n * coherent_gain` to read amplitudes.
pub fn coherent_gain(w: &[f64]) -> f64 {
    w.iter().sum::<f64>() / w.len() as f64
}

/// The equivalent noise bandwidth in bins: how much wider than one bin the window's noise filter is.
pub fn enbw(w: &[f64]) -> f64 {
    let s: f64 = w.iter().sum();
    let s2: f64 = w.iter().map(|x| x * x).sum();
    w.len() as f64 * s2 / (s * s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetric_windows_mirror_and_peak_in_the_middle() {
        for w in [
            Window::Hann,
            Window::Hamming,
            Window::Blackman,
            Window::BlackmanHarris,
            Window::Nuttall,
            Window::FlatTop,
            Window::Bartlett,
            Window::Kaiser(8.6),
        ] {
            for n in [5usize, 8, 33, 64] {
                let v = w.generate(n, false);
                assert_eq!(v.len(), n);
                for i in 0..n {
                    assert!((v[i] - v[n - 1 - i]).abs() < 1e-12, "{w:?} n={n} i={i}");
                }
            }
        }
    }

    #[test]
    fn known_values() {
        assert_eq!(Window::Rectangular.generate(4, false), vec![1.0; 4]);
        let h = Window::Hann.generate(5, false);
        for (a, b) in h.iter().zip([0.0, 0.5, 1.0, 0.5, 0.0]) {
            assert!((a - b).abs() < 1e-15);
        }
        // Hamming's end points are 0.08.
        assert!((Window::Hamming.generate(9, false)[0] - 0.08).abs() < 1e-15);
        // Bartlett: 0, .5, 1, .5, 0
        let b = Window::Bartlett.generate(5, false);
        for (a, e) in b.iter().zip([0.0, 0.5, 1.0, 0.5, 0.0]) {
            assert!((a - e).abs() < 1e-15);
        }
        // A Kaiser window with beta 0 is rectangular.
        assert!(Window::Kaiser(0.0).generate(7, false).iter().all(|&x| (x - 1.0).abs() < 1e-15));
    }

    #[test]
    fn periodic_is_the_symmetric_window_of_one_more_point_without_its_last() {
        for w in [Window::Hann, Window::Blackman, Window::Kaiser(5.0), Window::Bartlett] {
            let p = w.generate(16, true);
            let s = w.generate(17, false);
            assert_eq!(p.len(), 16);
            for i in 0..16 {
                assert!((p[i] - s[i]).abs() < 1e-15, "{w:?} {i}");
            }
        }
    }

    #[test]
    fn tiny_lengths() {
        for w in [Window::Hann, Window::Kaiser(5.0), Window::Bartlett, Window::FlatTop] {
            assert!(w.generate(0, false).is_empty());
            assert_eq!(w.generate(1, false), vec![1.0]);
            assert_eq!(w.generate(1, true), vec![1.0]);
            assert_eq!(w.generate(2, true).len(), 2);
        }
    }

    #[test]
    fn bessel_matches_known_values() {
        // From tables of I0.
        for (x, e) in [
            (0.0, 1.0),
            (1.0, 1.2660658777520082),
            (2.0, 2.279585302336067),
            (5.0, 27.239871823604442),
            (10.0, 2815.716628466254),
            (20.0, 43558282.55959),
        ] {
            assert!((bessel_i0(x) / e - 1.0).abs() < 1e-12, "I0({x}) = {}", bessel_i0(x));
        }
    }

    #[test]
    fn kaiser_beta_formula() {
        assert_eq!(kaiser_beta(10.0), 0.0);
        assert!((kaiser_beta(60.0) - 5.65326).abs() < 1e-4);
        assert!((kaiser_beta(40.0) - 3.395321).abs() < 1e-4);
    }

    #[test]
    fn gains() {
        let h = Window::Hann.generate(1024, true);
        assert!((coherent_gain(&h) - 0.5).abs() < 1e-12);
        assert!((enbw(&h) - 1.5).abs() < 1e-9);
        let r = Window::Rectangular.generate(100, true);
        assert_eq!((coherent_gain(&r), enbw(&r)), (1.0, 1.0));
    }
}
