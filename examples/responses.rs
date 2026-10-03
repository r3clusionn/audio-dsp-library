//! Writes the library's own responses as CSV for `scripts/plots.py`:
//! `cargo run --release --example responses -- OUT_DIR`.

use std::fmt::Write as _;

use adsp::dynamics::gain_to_db;
use adsp::fir::{self, kaiserord};
use adsp::iir::{butter, cheby1, Band, Biquad};
use adsp::{RealFft, Window};

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| "target/plots".into());
    std::fs::create_dir_all(&dir).unwrap();
    let fs = 48_000.0;

    // Filter magnitude responses on a log frequency axis.
    let (n, beta) = kaiserord(80.0, 600.0 / (fs / 2.0));
    let fir_taps = fir::lowpass(n | 1, 1300.0, fs, Window::Kaiser(beta)).unwrap();
    type Curve = Box<dyn Fn(f64) -> f64>;
    let designs: Vec<(String, Curve)> = vec![
        (
            "Butterworth order 2".into(),
            Box::new({
                let s = butter(2, Band::Lowpass(1000.0), fs).unwrap();
                move |f| gain_to_db(s.response(f, fs).abs())
            }),
        ),
        (
            "Butterworth order 8".into(),
            Box::new({
                let s = butter(8, Band::Lowpass(1000.0), fs).unwrap();
                move |f| gain_to_db(s.response(f, fs).abs())
            }),
        ),
        (
            "Chebyshev I order 6 (1 dB ripple)".into(),
            Box::new({
                let s = cheby1(6, 1.0, Band::Lowpass(1000.0), fs).unwrap();
                move |f| gain_to_db(s.response(f, fs).abs())
            }),
        ),
        (format!("FIR (Kaiser window) {} taps", n | 1), Box::new(move |f| gain_to_db(fir::response(&fir_taps, f, fs).abs()))),
    ];
    let mut csv = String::from("freq");
    for (name, _) in &designs {
        write!(csv, ",\"{name}\"").unwrap();
    }
    csv.push('\n');
    let mut f = 10.0f64;
    while f < fs / 2.0 {
        write!(csv, "{f}").unwrap();
        for (_, d) in &designs {
            write!(csv, ",{}", d(f).max(-160.0)).unwrap();
        }
        csv.push('\n');
        f *= 1.01;
    }
    std::fs::write(format!("{dir}/filters.csv"), csv).unwrap();

    // Equaliser curves.
    let eq: Vec<(&str, Biquad)> = vec![
        ("low shelf +6 dB at 100 Hz", Biquad::low_shelf(100.0, 0.707, 6.0, fs)),
        ("peaking -9 dB at 1 kHz (Q 2)", Biquad::peaking(1000.0, 2.0, -9.0, fs)),
        ("peaking +4 dB at 4 kHz (Q 1)", Biquad::peaking(4000.0, 1.0, 4.0, fs)),
        ("high shelf -4 dB at 10 kHz", Biquad::high_shelf(10_000.0, 0.707, -4.0, fs)),
    ];
    let mut csv = String::from("freq");
    for (name, _) in &eq {
        write!(csv, ",\"{name}\"").unwrap();
    }
    csv.push_str(",sum\n");
    let mut f = 10.0f64;
    while f < fs / 2.0 {
        write!(csv, "{f}").unwrap();
        let mut total = 0.0;
        for (_, b) in &eq {
            let g = gain_to_db(b.response(f, fs).abs());
            total += g;
            write!(csv, ",{g}").unwrap();
        }
        writeln!(csv, ",{total}").unwrap();
        f *= 1.01;
    }
    std::fs::write(format!("{dir}/eq.csv"), csv).unwrap();

    // Window spectra: 64-point windows zero-padded to 8192, normalised to 0 dB.
    let windows = [
        ("Rectangular", Window::Rectangular),
        ("Hann", Window::Hann),
        ("Blackman-Harris", Window::BlackmanHarris),
        ("Kaiser beta 10", Window::Kaiser(10.0)),
    ];
    let n = 8192;
    let r = RealFft::new(n);
    let spectra: Vec<Vec<f64>> = windows
        .iter()
        .map(|(_, w)| {
            let mut x = w.generate(64, false);
            x.resize(n, 0.0);
            let s = r.forward(&x);
            let peak = s[0].abs();
            s.iter().map(|z| gain_to_db(z.abs() / peak).max(-200.0)).collect()
        })
        .collect();
    let mut csv = String::from("bin");
    for (name, _) in &windows {
        write!(csv, ",\"{name}\"").unwrap();
    }
    csv.push('\n');
    for k in 0..=n / 2 {
        write!(csv, "{}", k as f64 * 64.0 / n as f64).unwrap();
        for s in &spectra {
            write!(csv, ",{}", s[k]).unwrap();
        }
        csv.push('\n');
    }
    std::fs::write(format!("{dir}/windows.csv"), csv).unwrap();
    println!("wrote filters.csv, eq.csv and windows.csv to {dir}");
}
