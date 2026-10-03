//! Spectral analysis: Welch's averaged periodogram, the Goertzel algorithm, and finding a tone's
//! frequency more precisely than one FFT bin.

use std::f64::consts::PI;

use crate::complex::Complex;
use crate::fft::RealFft;
use crate::window::Window;

/// Power spectral density by Welch's method, as `scipy.signal.welch(x, fs, window, nperseg,
/// noverlap)` with its defaults: each segment has its mean removed and is windowed, the
/// periodograms are averaged, the result is one-sided and in units squared per hertz.
///
/// Returns `(frequencies, psd)`, both `nperseg / 2 + 1` long. Panics if `x` is shorter than one
/// segment or `noverlap >= nperseg`.
pub fn welch(x: &[f64], fs: f64, window: Window, nperseg: usize, noverlap: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(nperseg > 0 && noverlap < nperseg, "need 0 <= noverlap < nperseg");
    assert!(x.len() >= nperseg, "the signal is shorter than one segment");
    let w = window.generate(nperseg, true);
    let scale = 1.0 / (fs * w.iter().map(|v| v * v).sum::<f64>());
    let fft = RealFft::new(nperseg);
    let step = nperseg - noverlap;
    let segments = (x.len() - nperseg) / step + 1;
    let bins = nperseg / 2 + 1;
    let mut psd = vec![0.0; bins];
    let mut seg = vec![0.0; nperseg];
    for s in 0..segments {
        let chunk = &x[s * step..s * step + nperseg];
        let mean = chunk.iter().sum::<f64>() / nperseg as f64;
        for ((d, c), wi) in seg.iter_mut().zip(chunk).zip(&w) {
            *d = (c - mean) * wi;
        }
        for (p, z) in psd.iter_mut().zip(fft.forward(&seg)) {
            *p += z.norm_sqr();
        }
    }
    for (k, p) in psd.iter_mut().enumerate() {
        *p *= scale / segments as f64;
        // One-sided: fold the negative frequencies in, except at 0 Hz and Nyquist.
        let nyquist = nperseg.is_multiple_of(2) && k == bins - 1;
        if k != 0 && !nyquist {
            *p *= 2.0;
        }
    }
    let freqs = (0..bins).map(|k| k as f64 * fs / nperseg as f64).collect();
    (freqs, psd)
}

/// The DFT of `x` at one frequency, which need not be a bin centre, by Goertzel's recursion: one
/// multiply-add per sample and no stored spectrum. Equal to `sum x[n] e^(-2 pi i f n / fs)`.
pub fn goertzel(x: &[f64], freq_hz: f64, fs: f64) -> Complex {
    let w = 2.0 * PI * freq_hz / fs;
    let c = 2.0 * w.cos();
    let (mut s1, mut s2) = (0.0, 0.0);
    for &v in x {
        let s = v + c * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    // X = e^(i w (N-1)) (s1 - e^(-i w) s2), rotated so the phase refers to sample 0.
    let n = x.len() as f64;
    let y = Complex::new(s1 - s2 * w.cos(), s2 * w.sin());
    y * Complex::cis(-w * (n - 1.0))
}

/// The frequency of the strongest component of `x`, from a Hann-windowed FFT refined by fitting a
/// parabola through the logarithm of the peak bin and its neighbours. Accurate to a small fraction
/// of a bin for a clean tone.
pub fn peak_frequency(x: &[f64], fs: f64) -> f64 {
    let n = x.len();
    assert!(n >= 4, "need at least four samples");
    let w = Window::Hann.generate(n, true);
    let xs: Vec<f64> = x.iter().zip(&w).map(|(a, b)| a * b).collect();
    let spec = RealFft::new(n).forward(&xs);
    let mag: Vec<f64> = spec.iter().map(|z| z.abs()).collect();
    let (k, _) = mag.iter().enumerate().skip(1).fold((1, 0.0), |best, (i, &m)| if m > best.1 { (i, m) } else { best });
    if k == 0 || k + 1 >= mag.len() {
        return k as f64 * fs / n as f64;
    }
    let (a, b, c) = (mag[k - 1].max(1e-300).ln(), mag[k].ln(), mag[k + 1].max(1e-300).ln());
    let d = a - 2.0 * b + c;
    let offset = if d.abs() < 1e-300 { 0.0 } else { 0.5 * (a - c) / d };
    (k as f64 + offset) * fs / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(n: usize, f: f64, fs: f64, a: f64) -> Vec<f64> {
        (0..n).map(|i| a * (2.0 * PI * f * i as f64 / fs).sin()).collect()
    }

    #[test]
    fn welch_puts_a_tones_power_in_the_right_place() {
        let fs = 8000.0;
        let x = tone(16_000, 1000.0, fs, 2.0);
        let (f, p) = welch(&x, fs, Window::Hann, 256, 128);
        assert_eq!(f.len(), 129);
        assert_eq!(f[32], 1000.0);
        let k = p.iter().enumerate().fold((0, 0.0), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0;
        assert_eq!(k, 32);
        // Total power (integral of the density) equals the tone's power, a^2 / 2 = 2.
        let df = fs / 256.0;
        let total: f64 = p.iter().sum::<f64>() * df;
        assert!((total - 2.0).abs() < 0.01, "{total}");
    }

    #[test]
    fn welch_of_white_noise_is_flat_at_its_variance_over_half_the_rate() {
        let mut s = 5u64;
        let x: Vec<f64> = (0..200_000)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 1_000_000) as f64 / 500_000.0 - 1.0
            })
            .collect();
        let var = 1.0 / 3.0; // uniform on [-1, 1]
        let (_, p) = welch(&x, 1000.0, Window::Hann, 512, 256);
        let mid = &p[10..240];
        let mean = mid.iter().sum::<f64>() / mid.len() as f64;
        assert!((mean - var / 500.0).abs() / (var / 500.0) < 0.02, "{mean}");
    }

    #[test]
    fn goertzel_equals_the_dft_sum_anywhere() {
        let x: Vec<f64> = (0..300).map(|i| ((i * 37) % 23) as f64 - 11.0).collect();
        for f in [0.0, 13.0, 1234.567, 4000.0] {
            let g = goertzel(&x, f, 8000.0);
            let d = x
                .iter()
                .enumerate()
                .fold(Complex::ZERO, |acc, (n, &v)| acc + Complex::cis(-2.0 * PI * f * n as f64 / 8000.0).scale(v));
            assert!((g - d).abs() < 1e-8 * d.abs().max(1.0), "{f}: {g:?} vs {d:?}");
        }
    }

    #[test]
    fn peak_frequency_is_accurate_between_bins() {
        let fs = 48_000.0;
        for f in [440.0, 997.3, 1000.0, 12_345.6] {
            let x = tone(4096, f, fs, 0.8);
            let est = peak_frequency(&x, fs);
            let bin = fs / 4096.0;
            assert!((est - f).abs() < 0.05 * bin, "{f}: {est} (bin {bin} Hz)");
        }
    }
}
