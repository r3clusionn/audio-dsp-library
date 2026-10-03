//! Every result compared with NumPy and SciPy on the same inputs (`tests/fixtures/scipy.json`,
//! written by `scripts/make_fixtures.py`).

use adsp::fir::{convolve, fft_convolve, firwin};
use adsp::iir::{butter, cheby1, Band, Biquad};
use adsp::resample::resample_poly;
use adsp::spectrum::welch;
use adsp::{Complex, Fft, RealFft, Window};
use serde_json::Value;

fn fixtures() -> Value {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/scipy.json");
    serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
}

fn reals(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn complexes(v: &Value) -> Vec<Complex> {
    v.as_array().unwrap().iter().map(|p| Complex::new(p[0].as_f64().unwrap(), p[1].as_f64().unwrap())).collect()
}

/// Largest difference relative to the largest magnitude in `want`.
fn rel_err(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "lengths differ");
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / scale
}

fn window_named(name: &str, beta: Option<f64>) -> Window {
    match name {
        "Hann" => Window::Hann,
        "Hamming" => Window::Hamming,
        "Blackman" => Window::Blackman,
        "BlackmanHarris" => Window::BlackmanHarris,
        "Nuttall" => Window::Nuttall,
        "FlatTop" => Window::FlatTop,
        "Bartlett" => Window::Bartlett,
        "Kaiser" => Window::Kaiser(beta.unwrap_or(8.6)),
        other => panic!("unknown window {other}"),
    }
}

#[test]
fn fft_matches_numpy_for_every_algorithm() {
    let f = fixtures();
    for case in f["fft"].as_array().unwrap() {
        let x = complexes(&case["x"]);
        let want = complexes(&case["y"]);
        let plan = Fft::new(x.len());
        let mut y = x.clone();
        plan.forward(&mut y);
        let scale = want.iter().fold(0.0f64, |m, z| m.max(z.abs()));
        let err = y.iter().zip(&want).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max) / scale;
        assert!(err < 1e-13, "n = {} ({}): relative error {err:e}", x.len(), plan.algorithm());
        // And the real-input transform on the real parts.
        let re: Vec<f64> = x.iter().map(|z| z.re).collect();
        let mut full: Vec<Complex> = re.iter().map(|&v| Complex::from(v)).collect();
        plan.forward(&mut full);
        let half = RealFft::new(re.len()).forward(&re);
        let err = half.iter().zip(&full).map(|(a, b)| (*a - *b).abs()).fold(0.0, f64::max) / scale;
        assert!(err < 1e-13, "real n = {}: {err:e}", x.len());
    }
}

#[test]
fn windows_match_scipy() {
    let f = fixtures();
    for case in f["windows"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n = case["n"].as_u64().unwrap() as usize;
        let periodic = case["periodic"].as_bool().unwrap();
        let w = window_named(name, None).generate(n, periodic);
        let err = rel_err(&w, &reals(&case["w"]));
        assert!(err < 1e-14, "{name} n={n} periodic={periodic}: {err:e}");
    }
}

#[test]
fn firwin_matches_scipy_coefficient_for_coefficient() {
    let f = fixtures();
    for case in f["firwin"].as_array().unwrap() {
        let numtaps = case["numtaps"].as_u64().unwrap() as usize;
        let cutoff = reals(&case["cutoff"]);
        let pass_zero = case["pass_zero"].as_bool().unwrap();
        let w = window_named(case["window"].as_str().unwrap(), case["beta"].as_f64());
        let h = firwin(numtaps, &cutoff, pass_zero, w).unwrap();
        let err = rel_err(&h, &reals(&case["h"]));
        assert!(err < 1e-13, "firwin({numtaps}, {cutoff:?}, pass_zero={pass_zero}, {w:?}): {err:e}");
    }
}

#[test]
fn butterworth_and_chebyshev_filters_match_scipy() {
    let f = fixtures();
    let x = reals(&f["iir_input"]);
    for case in f["iir"].as_array().unwrap() {
        let order = case["order"].as_u64().unwrap() as usize;
        let wn = reals(&case["wn"]);
        let band = match case["btype"].as_str().unwrap() {
            "lowpass" => Band::Lowpass(wn[0]),
            "highpass" => Band::Highpass(wn[0]),
            "bandpass" => Band::Bandpass(wn[0], wn[1]),
            _ => Band::Bandstop(wn[0], wn[1]),
        };
        let mut sos = match case["kind"].as_str().unwrap() {
            "butter" => butter(order, band, 48_000.0).unwrap(),
            _ => cheby1(order, case["ripple"].as_f64().unwrap(), band, 48_000.0).unwrap(),
        };
        let label = format!("{} {} {:?}", case["kind"], order, band);
        // The response at every probe frequency.
        for (fq, want) in reals(&case["freqs"]).iter().zip(complexes(&case["h"])) {
            let got = sos.response(*fq, 48_000.0);
            assert!((got - want).abs() < 1e-9 * want.abs().max(1e-6), "{label} at {fq} Hz: {got:?} vs {want:?}");
        }
        // The filtered signal.
        let mut y = x.clone();
        sos.process_block(&mut y);
        let err = rel_err(&y, &reals(&case["y"]));
        assert!(err < 1e-9, "{label}: output differs by {err:e}");
    }
}

#[test]
fn cookbook_biquads_match_an_independent_transcription() {
    let f = fixtures();
    let x = reals(&f["iir_input"]);
    for case in f["biquads"].as_array().unwrap() {
        let (f0, q, g) = (case["f0"].as_f64().unwrap(), case["q"].as_f64().unwrap(), case["gain"].as_f64().unwrap());
        let mut bq = match case["kind"].as_str().unwrap() {
            "lowpass" => Biquad::lowpass(f0, q, 48_000.0),
            "peaking" => Biquad::peaking(f0, q, g, 48_000.0),
            "low_shelf" => Biquad::low_shelf(f0, q, g, 48_000.0),
            "high_shelf" => Biquad::high_shelf(f0, q, g, 48_000.0),
            _ => Biquad::notch(f0, q, 48_000.0),
        };
        let b = reals(&case["b"]);
        let a = reals(&case["a"]);
        let mine = [bq.b0, bq.b1, bq.b2, 1.0, bq.a1, bq.a2];
        let theirs = [b[0], b[1], b[2], a[0], a[1], a[2]];
        assert!(rel_err(&mine, &theirs) < 1e-14, "{}: {mine:?} vs {theirs:?}", case["kind"]);
        let mut y = x.clone();
        bq.process_block(&mut y);
        assert!(rel_err(&y, &reals(&case["y"])) < 1e-12, "{} output", case["kind"]);
    }
}

#[test]
fn resample_poly_matches_scipy_sample_for_sample() {
    let f = fixtures();
    let x = reals(&f["resample_input"]);
    for case in f["resample"].as_array().unwrap() {
        let (up, down) = (case["up"].as_u64().unwrap() as usize, case["down"].as_u64().unwrap() as usize);
        let y = resample_poly(&x, up, down);
        let err = rel_err(&y, &reals(&case["y"]));
        assert!(err < 1e-13, "{up}/{down}: {err:e}");
    }
}

#[test]
fn convolution_matches_numpy() {
    let f = fixtures();
    let c = &f["convolve"];
    let (a, b, want) = (reals(&c["a"]), reals(&c["b"]), reals(&c["y"]));
    assert!(rel_err(&convolve(&a, &b), &want) < 1e-14);
    assert!(rel_err(&fft_convolve(&a, &b), &want) < 1e-13);
}

#[test]
fn welch_matches_scipy() {
    let f = fixtures();
    let x = reals(&f["welch_input"]);
    for case in f["welch"].as_array().unwrap() {
        let w = window_named(case["window"].as_str().unwrap(), None);
        let (nper, nover) = (case["nperseg"].as_u64().unwrap() as usize, case["noverlap"].as_u64().unwrap() as usize);
        let (freqs, psd) = welch(&x, 1000.0, w, nper, nover);
        assert!(rel_err(&freqs, &reals(&case["f"])) < 1e-15);
        let err = rel_err(&psd, &reals(&case["p"]));
        assert!(err < 1e-12, "{w:?} {nper}/{nover}: {err:e}");
    }
}
