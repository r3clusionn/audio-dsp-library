use super::*;
use std::f64::consts::PI;

fn sine(n: usize, f: f64, fs: f64) -> Vec<f64> {
    (0..n).map(|i| (2.0 * PI * f * i as f64 / fs).sin()).collect()
}

/// Amplitude of the component at `f` (by correlation over a whole number of cycles is not needed:
/// a long Hann-free least-squares fit of sin and cos is accurate to well under 0.1 percent).
fn amplitude(x: &[f64], f: f64, fs: f64) -> f64 {
    let (mut s, mut c) = (0.0, 0.0);
    for (i, v) in x.iter().enumerate() {
        let a = 2.0 * PI * f * i as f64 / fs;
        s += v * a.sin();
        c += v * a.cos();
    }
    2.0 * (s * s + c * c).sqrt() / x.len() as f64
}

#[test]
fn output_lengths_follow_the_ceiling_rule() {
    for (up, down, n, want) in
        [(3usize, 2usize, 10usize, 15usize), (2, 3, 10, 7), (160, 147, 44_100, 48_000), (1, 1, 5, 5), (4, 6, 9, 6)]
    {
        let mut r = Resampler::new(up, down);
        assert_eq!(r.run(&vec![0.5; n]).len(), want, "{up}/{down} of {n}");
    }
    assert_eq!(Resampler::new(4, 6).ratio(), (2, 3));
    assert_eq!(Resampler::between(44_100, 48_000).ratio(), (160, 147));
}

#[test]
fn the_identity_ratio_copies() {
    let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
    assert_eq!(resample_poly(&x, 7, 7), x);
}

#[test]
fn a_tone_keeps_its_frequency_and_amplitude() {
    let x = sine(44_100, 1000.0, 44_100.0);
    let y = resample_poly(&x, 160, 147);
    assert_eq!(y.len(), 48_000);
    // Away from the ends (the filter's edges), the 1 kHz tone at 48 kHz is all there is.
    let mid = &y[2000..46_000];
    let a = amplitude(mid, 1000.0, 48_000.0);
    assert!((a - 1.0).abs() < 1e-3, "amplitude {a}");
    let ideal: Vec<f64> = (2000..46_000).map(|i| (2.0 * PI * 1000.0 * i as f64 / 48_000.0).sin()).collect();
    let err = mid.iter().zip(&ideal).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / mid.len() as f64;
    let snr = 10.0 * (0.5 / err).log10();
    // SciPy's default filter (Kaiser, beta 5) gives 63.95 dB on this exact input; so does this.
    assert!((snr - 63.95).abs() < 0.01, "SNR {snr} dB");
    // A stronger window trades a wider transition for much less ripple (SciPy: 107.4 dB).
    let y = Resampler::with_window(160, 147, Window::Kaiser(10.0), 10).run(&x);
    let err = y[2000..46_000].iter().zip(&ideal).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / ideal.len() as f64;
    let snr = 10.0 * (0.5 / err).log10();
    assert!((snr - 107.4).abs() < 0.1, "SNR with beta 10: {snr} dB");
}

#[test]
fn downsampling_removes_what_would_alias() {
    // A 20 kHz tone cannot exist at 22.05 kHz; it must be filtered out, not folded down to 2.05 kHz.
    let x = sine(44_100, 20_000.0, 44_100.0);
    let y = resample_poly(&x, 1, 2);
    let leak = amplitude(&y[1000..21_000], 2050.0, 22_050.0);
    assert!(20.0 * leak.log10() < -40.0, "alias at {} dB", 20.0 * leak.log10());
    // Linear interpolation does let it through.
    let lin = linear(&x, 0.5);
    let leak_lin = amplitude(&lin[1000..21_000], 2050.0, 22_050.0);
    assert!(leak_lin > 0.5, "linear interpolation aliases: {leak_lin}");
}

#[test]
fn streaming_gives_the_same_samples_in_any_chunking() {
    let mut s = 7u64;
    let x: Vec<f64> = (0..3001)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f64 / 1000.0 - 1.0
        })
        .collect();
    for (up, down) in [(160usize, 147usize), (147, 160), (3, 1), (1, 4), (2, 3)] {
        let whole = resample_poly(&x, up, down);
        for chunk in [1usize, 2, 7, 64, 1000, 5000] {
            let mut r = Resampler::new(up, down);
            let mut out = Vec::new();
            for c in x.chunks(chunk) {
                r.process(c, &mut out);
            }
            r.finish(&mut out);
            assert_eq!(out, whole, "{up}/{down} in chunks of {chunk}");
            // Reusable after finish.
            assert_eq!(r.run(&x), whole);
        }
    }
}

#[test]
fn interpolators_hit_the_input_samples_and_bend_smoothly() {
    let x = vec![0.0, 1.0, 0.0, -1.0, 0.0];
    let l = linear(&x, 2.0);
    assert_eq!(l.len(), 10);
    assert_eq!((l[0], l[1], l[2], l[3]), (0.0, 0.5, 1.0, 0.5));
    let c = cubic(&x, 2.0);
    assert_eq!((c[0], c[2], c[4], c[6]), (0.0, 1.0, 0.0, -1.0));
    // Catmull-Rom between 0 and 1 with neighbours 0 and 0 overshoots linear: 0.5625.
    assert!((c[1] - 0.5625).abs() < 1e-12, "{}", c[1]);
}
