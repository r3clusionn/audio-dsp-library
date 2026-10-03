//! Throughput: `cargo run --release --example bench`.
//!
//! Every figure is the median of 7 runs. FFT rows compare with `rustfft` (version 6) on the same
//! data, both planned once outside the timed loop.

use std::hint::black_box;
use std::time::Instant;

use adsp::fir::{fft_convolve, lowpass, Fir, OverlapAdd};
use adsp::iir::{butter, Band, Biquad};
use adsp::resample::Resampler;
use adsp::spectrum::welch;
use adsp::{signal, Complex, Fft, RealFft, Window};
use rustfft::num_complex::Complex64;
use rustfft::FftPlanner;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Seconds per call of `f`, repeating it enough times to run for about 50 ms per sample.
fn time(mut f: impl FnMut()) -> f64 {
    let t = Instant::now();
    f();
    let once = t.elapsed().as_secs_f64().max(1e-7);
    let reps = ((0.05 / once) as usize).clamp(1, 1_000_000);
    median(
        (0..7)
            .map(|_| {
                let t = Instant::now();
                for _ in 0..reps {
                    f();
                }
                t.elapsed().as_secs_f64() / reps as f64
            })
            .collect(),
    )
}

fn fmt_time(s: f64) -> String {
    if s < 1e-3 {
        format!("{:.1} us", s * 1e6)
    } else {
        format!("{:.2} ms", s * 1e3)
    }
}

fn main() {
    println!("FFT, complex, forward (median of 7)\n");
    println!("| Length | adsp algorithm | adsp | rustfft | adsp / rustfft |");
    println!("|---|---|---|---|---|");
    let mut planner = FftPlanner::<f64>::new();
    for n in [1024usize, 4096, 65_536, 1 << 20, 44_100, 48_000, 1009, 65_537] {
        let x: Vec<Complex> = signal::white_noise(2 * n, 1.0, 3).chunks(2).map(|p| Complex::new(p[0], p[1])).collect();
        let plan = Fft::new(n);
        let mut scratch = vec![Complex::ZERO; plan.scratch_len()];
        let mut buf = x.clone();
        let ours = time(|| {
            buf.copy_from_slice(&x);
            plan.forward_with_scratch(&mut buf, &mut scratch);
            black_box(&buf);
        });
        let rf = planner.plan_fft_forward(n);
        let xr: Vec<Complex64> = x.iter().map(|z| Complex64::new(z.re, z.im)).collect();
        let mut br = xr.clone();
        let mut sr = vec![Complex64::new(0.0, 0.0); rf.get_inplace_scratch_len()];
        let theirs = time(|| {
            br.copy_from_slice(&xr);
            rf.process_with_scratch(&mut br, &mut sr);
            black_box(&br);
        });
        println!("| {n} | {} | {} | {} | {:.1}x |", plan.algorithm(), fmt_time(ours), fmt_time(theirs), ours / theirs);
    }

    println!("\nReal-input FFT (adsp only)\n");
    println!("| Length | time |");
    println!("|---|---|");
    for n in [4096usize, 65_536] {
        let x = signal::white_noise(n, 1.0, 4);
        let r = RealFft::new(n);
        println!(
            "| {n} | {} |",
            fmt_time(time(|| {
                black_box(r.forward(&x));
            }))
        );
    }

    let fs = 48_000.0;
    let one_second = signal::white_noise(48_000, 0.5, 5);
    let rate = |secs: f64| format!("{:.0}x real time", 1.0 / secs);
    println!("\nOne second of 48 kHz audio\n");
    println!("| Operation | time | speed |");
    println!("|---|---|---|");
    let taps = lowpass(255, 4000.0, fs, Window::Kaiser(8.0)).unwrap();
    let t = time(|| {
        let mut f = Fir::new(taps.clone());
        let mut y = vec![0.0; one_second.len()];
        f.process_block(&one_second, &mut y);
        black_box(y);
    });
    println!("| FIR, 255 taps, sample by sample | {} | {} |", fmt_time(t), rate(t));
    let t = time(|| {
        let mut o = OverlapAdd::new(&taps, 256);
        let mut y = Vec::with_capacity(48_400);
        o.process(&one_second, &mut y);
        o.finish(&mut y);
        black_box(y);
    });
    println!("| FIR, 255 taps, overlap-add (block 256) | {} | {} |", fmt_time(t), rate(t));
    let long = signal::white_noise(48_000, 0.1, 6);
    let t = time(|| {
        let mut o = OverlapAdd::new(&long, 4096);
        let mut y = Vec::with_capacity(96_000);
        o.process(&one_second, &mut y);
        o.finish(&mut y);
        black_box(y);
    });
    println!("| Convolution with a 1-second impulse response, overlap-add (block 4096) | {} | {} |", fmt_time(t), rate(t));
    let t = time(|| {
        black_box(fft_convolve(&one_second, &long));
    });
    println!("| The same, one FFT convolution | {} | {} |", fmt_time(t), rate(t));
    let t = time(|| {
        let mut b = Biquad::peaking(1000.0, 1.0, 6.0, fs);
        let mut y = one_second.clone();
        b.process_block(&mut y);
        black_box(y);
    });
    println!("| One biquad | {} | {} |", fmt_time(t), rate(t));
    let sos = butter(8, Band::Lowpass(1000.0), fs).unwrap();
    let t = time(|| {
        let mut s = sos.clone();
        let mut y = one_second.clone();
        s.process_block(&mut y);
        black_box(y);
    });
    println!("| Butterworth, order 8 (4 sections) | {} | {} |", fmt_time(t), rate(t));
    let in441 = signal::white_noise(44_100, 0.5, 7);
    let t = time(|| {
        let mut r = Resampler::between(44_100, 48_000);
        black_box(r.run(&in441));
    });
    println!("| Resample 44.1 to 48 kHz (one second of input) | {} | {} |", fmt_time(t), rate(t));
    let ten = signal::white_noise(480_000, 0.5, 8);
    let t = time(|| {
        black_box(welch(&ten, fs, Window::Hann, 4096, 2048));
    });
    println!("| Welch PSD of 10 seconds, 4096-point segments | {} | {} |", fmt_time(t), rate(t / 10.0));
}
