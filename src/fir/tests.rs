use super::*;

fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

fn gain_db(taps: &[f64], f: f64, fs: f64) -> f64 {
    20.0 * response(taps, f, fs).abs().log10()
}

#[test]
fn a_lowpass_passes_low_and_stops_high() {
    let fs = 48_000.0;
    let h = lowpass(101, 4000.0, fs, Window::Hamming).unwrap();
    assert!((response(&h, 0.0, fs).abs() - 1.0).abs() < 1e-12, "unit gain at 0 Hz");
    assert!(gain_db(&h, 2000.0, fs).abs() < 0.05);
    assert!(gain_db(&h, 8000.0, fs) < -50.0, "{}", gain_db(&h, 8000.0, fs));
    // -6 dB at the cutoff, as for any windowed-sinc design.
    assert!((gain_db(&h, 4000.0, fs) + 6.02).abs() < 0.1);
    // Linear phase: symmetric taps.
    for i in 0..h.len() {
        assert!((h[i] - h[h.len() - 1 - i]).abs() < 1e-15);
    }
}

#[test]
fn highpass_bandpass_bandstop_scale_their_pass_band_to_unity() {
    let fs = 48_000.0;
    let hp = highpass(101, 4000.0, fs, Window::Blackman).unwrap();
    assert!((response(&hp, 24_000.0, fs).abs() - 1.0).abs() < 1e-12);
    assert!(gain_db(&hp, 500.0, fs) < -60.0);
    let bp = bandpass(201, 1000.0, 3000.0, fs, Window::Hann).unwrap();
    assert!((response(&bp, 2000.0, fs).abs() - 1.0).abs() < 1e-3);
    assert!(gain_db(&bp, 100.0, fs) < -40.0 && gain_db(&bp, 8000.0, fs) < -40.0);
    let bs = bandstop(201, 1000.0, 3000.0, fs, Window::Hann).unwrap();
    assert!((response(&bs, 0.0, fs).abs() - 1.0).abs() < 1e-12);
    assert!(gain_db(&bs, 2000.0, fs) < -40.0);
}

#[test]
fn design_errors_are_reported() {
    assert_eq!(firwin(0, &[0.5], true, Window::Hann), Err(DesignError::NoTaps));
    assert_eq!(firwin(10, &[0.5], false, Window::Hann), Err(DesignError::EvenTapsPassNyquist));
    assert!(matches!(firwin(11, &[1.0], true, Window::Hann), Err(DesignError::BadCutoff(_))));
    assert!(matches!(firwin(11, &[0.0], true, Window::Hann), Err(DesignError::BadCutoff(_))));
    assert!(matches!(firwin(11, &[0.5, 0.3], true, Window::Hann), Err(DesignError::BadCutoff(_))));
    assert!(matches!(firwin(11, &[], true, Window::Hann), Err(DesignError::BadCutoff(_))));
    // An even-length low-pass is fine (it does not pass Nyquist).
    assert!(firwin(10, &[0.5], true, Window::Hann).is_ok());
}

#[test]
fn kaiser_rule_meets_its_attenuation() {
    let fs = 48_000.0;
    // 80 dB, transition 1 kHz wide (from 4 to 5 kHz).
    let (n, beta) = kaiserord(80.0, 1000.0 / (fs / 2.0));
    let n = n | 1;
    let h = lowpass(n, 4500.0, fs, Window::Kaiser(beta)).unwrap();
    let worst = (0..200).map(|i| gain_db(&h, 5000.0 + i as f64 * 95.0, fs)).fold(f64::MIN, f64::max);
    assert!(worst < -78.0, "stop band reaches {worst} dB with {n} taps");
    let ripple = (0..200).map(|i| gain_db(&h, i as f64 * 20.0, fs).abs()).fold(0.0, f64::max);
    assert!(ripple < 0.01, "pass-band ripple {ripple} dB");
}

#[test]
fn the_streaming_filter_equals_convolution() {
    let mut r = rng(4);
    let taps: Vec<f64> = (0..37).map(|_| r()).collect();
    let x: Vec<f64> = (0..500).map(|_| r()).collect();
    let full = convolve(&x, &taps);
    let mut f = Fir::new(taps.clone());
    let mut y = vec![0.0; x.len()];
    f.process_block(&x, &mut y);
    for i in 0..x.len() {
        assert!((y[i] - full[i]).abs() < 1e-12, "sample {i}");
    }
    f.reset();
    assert!((f.process(x[0]) - full[0]).abs() < 1e-15);
    let mut one = Fir::new(vec![0.5]);
    assert_eq!(one.process(4.0), 2.0);
}

#[test]
fn fft_convolution_equals_direct_convolution() {
    let mut r = rng(8);
    for (na, nb) in [(1usize, 1usize), (1, 9), (17, 3), (100, 100), (1000, 37), (513, 512)] {
        let a: Vec<f64> = (0..na).map(|_| r()).collect();
        let b: Vec<f64> = (0..nb).map(|_| r()).collect();
        let d = convolve(&a, &b);
        let f = fft_convolve(&a, &b);
        assert_eq!(d.len(), f.len());
        let e = d.iter().zip(&f).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
        assert!(e < 1e-11, "{na}x{nb}: {e}");
    }
    assert!(convolve(&[], &[1.0]).is_empty());
    assert!(fft_convolve(&[1.0], &[]).is_empty());
}

#[test]
fn overlap_add_produces_exactly_the_convolution_in_any_chunking() {
    let mut r = rng(12);
    for (taps_n, block, n) in [(64usize, 64usize, 1000usize), (300, 128, 777), (5, 256, 3), (1, 16, 50), (129, 100, 100)] {
        let taps: Vec<f64> = (0..taps_n).map(|_| r()).collect();
        let x: Vec<f64> = (0..n).map(|_| r()).collect();
        let want = convolve(&x, &taps);
        for chunk in [1usize, 7, 64, 1000] {
            let mut ola = OverlapAdd::new(&taps, block);
            let mut out = Vec::new();
            for c in x.chunks(chunk) {
                ola.process(c, &mut out);
            }
            ola.finish(&mut out);
            assert_eq!(out.len(), want.len(), "taps {taps_n} block {block} n {n} chunk {chunk}");
            let e = out.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(e < 1e-11, "taps {taps_n} block {block} n {n} chunk {chunk}: {e}");
            // Reusable after finish.
            let mut again = Vec::new();
            ola.process(&x, &mut again);
            ola.finish(&mut again);
            assert_eq!(again.len(), want.len());
        }
    }
}
