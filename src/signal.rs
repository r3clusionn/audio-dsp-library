//! Test signals.

use std::f64::consts::PI;

/// `amplitude * sin(2 pi f t)`, `n` samples.
pub fn sine(n: usize, freq_hz: f64, fs: f64, amplitude: f64) -> Vec<f64> {
    (0..n).map(|i| amplitude * (2.0 * PI * freq_hz * i as f64 / fs).sin()).collect()
}

/// An exponential sine sweep from `f0` to `f1` over `n` samples (Farina's sweep): equal time per
/// octave, the usual signal for measuring an audio system.
pub fn log_sweep(n: usize, f0: f64, f1: f64, fs: f64, amplitude: f64) -> Vec<f64> {
    let t = n as f64 / fs;
    let k = (f1 / f0).ln();
    (0..n)
        .map(|i| {
            let ti = i as f64 / fs;
            amplitude * (2.0 * PI * f0 * t / k * ((ti / t * k).exp() - 1.0)).sin()
        })
        .collect()
}

/// A unit impulse at sample 0.
pub fn impulse(n: usize) -> Vec<f64> {
    let mut v = vec![0.0; n];
    if n > 0 {
        v[0] = 1.0;
    }
    v
}

/// Deterministic white noise, uniform in `[-amplitude, amplitude]`, from a 64-bit xorshift
/// generator seeded with `seed`.
pub fn white_noise(n: usize, amplitude: f64, seed: u64) -> Vec<f64> {
    let mut s = seed.max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            amplitude * ((s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators() {
        let s = sine(48, 1000.0, 48_000.0, 0.5);
        assert_eq!(s[0], 0.0);
        assert!((s[12] - 0.5).abs() < 1e-12);
        assert_eq!(impulse(3), vec![1.0, 0.0, 0.0]);
        assert!(impulse(0).is_empty());
        let n = white_noise(10_000, 0.25, 7);
        assert!(n.iter().all(|v| v.abs() <= 0.25));
        assert_eq!(n, white_noise(10_000, 0.25, 7), "deterministic");
        let mean = n.iter().sum::<f64>() / n.len() as f64;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn the_sweep_starts_and_ends_at_its_frequencies() {
        let fs = 48_000.0;
        let x = log_sweep(96_000, 20.0, 20_000.0, fs, 1.0);
        assert!(x.iter().all(|v| v.abs() <= 1.0));
        let start = crate::spectrum::peak_frequency(&x[..9600], fs);
        let near_end = crate::spectrum::peak_frequency(&x[94_000..96_000], fs);
        assert!(start < 40.0, "{start}");
        assert!(near_end > 15_000.0 && near_end < 20_000.0, "{near_end}");
    }
}
