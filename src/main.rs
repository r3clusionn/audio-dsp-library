//! `adsp`: the library's building blocks on WAV files.

use std::path::PathBuf;
use std::process::ExitCode;

use adsp::dynamics::{gain_to_db, normalize, peak, rms, Compressor, Limiter};
use adsp::fir::{self, Fir};
use adsp::iir::{butter, Band, Biquad, Sos};
use adsp::resample::Resampler;
use adsp::signal;
use adsp::spectrum::{peak_frequency, welch};
use adsp::wav::{self, Audio, Format};
use adsp::Window;
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser)]
#[command(
    name = "adsp",
    version,
    about = "Audio DSP on WAV files: inspect, generate, filter, resample, process dynamics, analyse spectra."
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum OutFormat {
    #[value(name = "16")]
    Pcm16,
    #[value(name = "24")]
    Pcm24,
    #[value(name = "f32")]
    Float32,
}

impl From<OutFormat> for Format {
    fn from(f: OutFormat) -> Format {
        match f {
            OutFormat::Pcm16 => Format::Pcm16,
            OutFormat::Pcm24 => Format::Pcm24,
            OutFormat::Float32 => Format::Float32,
        }
    }
}

#[derive(Clone, Copy, ValueEnum)]
enum Kind {
    Sine,
    Sweep,
    Noise,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum FilterType {
    Lowpass,
    Highpass,
    Bandpass,
    Bandstop,
    Peaking,
    LowShelf,
    HighShelf,
    Notch,
}

#[derive(Subcommand)]
enum Cmd {
    /// Format, length and levels of each channel.
    Info { file: PathBuf },
    /// Write a test signal.
    Gen {
        out: PathBuf,
        #[arg(value_enum)]
        kind: Kind,
        /// Frequency in Hz (the start frequency of a sweep).
        #[arg(long, default_value_t = 1000.0)]
        freq: f64,
        /// End frequency of a sweep.
        #[arg(long, default_value_t = 20_000.0)]
        to: f64,
        #[arg(long, default_value_t = 1.0)]
        seconds: f64,
        #[arg(long, default_value_t = 48_000)]
        rate: u32,
        /// Peak level in dBFS.
        #[arg(long, default_value_t = -6.0, allow_negative_numbers = true)]
        level: f64,
        #[arg(long, value_enum, default_value = "24")]
        format: OutFormat,
    },
    /// Filter every channel. Butterworth IIR by default; --fir for a linear-phase FIR.
    Filter {
        input: PathBuf,
        out: PathBuf,
        #[arg(value_enum)]
        kind: FilterType,
        /// Frequency in Hz (the lower edge for band filters).
        freq: f64,
        /// Upper edge for band filters.
        freq2: Option<f64>,
        /// Butterworth order (lowpass, highpass, bandpass, bandstop).
        #[arg(long, default_value_t = 4)]
        order: usize,
        /// Use a Kaiser-window FIR with this many taps instead (lowpass, highpass, bandpass, bandstop).
        #[arg(long)]
        fir: Option<usize>,
        /// Gain in dB for peaking and shelving filters.
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gain: f64,
        #[arg(long, default_value_t = std::f64::consts::FRAC_1_SQRT_2)]
        q: f64,
        #[arg(long, value_enum, default_value = "24")]
        format: OutFormat,
    },
    /// Change the sample rate (polyphase, as scipy.signal.resample_poly).
    Resample {
        input: PathBuf,
        out: PathBuf,
        rate: u32,
        #[arg(long, value_enum, default_value = "24")]
        format: OutFormat,
    },
    /// Normalise, compress or limit.
    Gain {
        input: PathBuf,
        out: PathBuf,
        /// Scale so the peak is this many dBFS.
        #[arg(long, allow_negative_numbers = true)]
        normalize: Option<f64>,
        /// Compress above this threshold (dBFS) ...
        #[arg(long, allow_negative_numbers = true)]
        compress: Option<f64>,
        /// ... at this ratio.
        #[arg(long, default_value_t = 4.0)]
        ratio: f64,
        /// Limit to this ceiling (dBFS) with 5 ms of look-ahead.
        #[arg(long, allow_negative_numbers = true)]
        limit: Option<f64>,
        #[arg(long, value_enum, default_value = "24")]
        format: OutFormat,
    },
    /// Power spectrum (Welch) of the first channel in bands, as a bar chart.
    Spectrum {
        file: PathBuf,
        /// Bands per octave.
        #[arg(long, default_value_t = 3)]
        per_octave: usize,
    },
    /// Print a filter's response without touching audio.
    Response {
        #[arg(value_enum)]
        kind: FilterType,
        freq: f64,
        freq2: Option<f64>,
        #[arg(long, default_value_t = 4)]
        order: usize,
        #[arg(long)]
        fir: Option<usize>,
        #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
        gain: f64,
        #[arg(long, default_value_t = std::f64::consts::FRAC_1_SQRT_2)]
        q: f64,
        #[arg(long, default_value_t = 48_000.0)]
        rate: f64,
    },
}

type Res<T> = Result<T, String>;

fn load(p: &PathBuf) -> Res<Audio> {
    wav::read_file(p).map_err(|e| format!("{}: {e}", p.display()))
}

fn save(p: &PathBuf, a: &Audio, f: OutFormat) -> Res<()> {
    wav::write_file(p, a, f.into()).map_err(|e| format!("{}: {e}", p.display()))
}

fn db_text(v: f64) -> String {
    if v == 0.0 {
        "-inf".into()
    } else {
        format!("{:.2}", gain_to_db(v))
    }
}

enum AnyFilter {
    Iir(Sos),
    Bq(Biquad),
    Fir(Fir, Vec<f64>),
}

impl AnyFilter {
    fn process(&mut self, x: f64) -> f64 {
        match self {
            AnyFilter::Iir(s) => s.process(x),
            AnyFilter::Bq(b) => b.process(x),
            AnyFilter::Fir(f, _) => f.process(x),
        }
    }

    fn response(&self, f: f64, fs: f64) -> adsp::Complex {
        match self {
            AnyFilter::Iir(s) => s.response(f, fs),
            AnyFilter::Bq(b) => b.response(f, fs),
            AnyFilter::Fir(_, taps) => fir::response(taps, f, fs),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build(
    kind: FilterType,
    freq: f64,
    freq2: Option<f64>,
    order: usize,
    fir_taps: Option<usize>,
    gain: f64,
    q: f64,
    fs: f64,
) -> Res<AnyFilter> {
    let band = || -> Res<Band> {
        Ok(match kind {
            FilterType::Lowpass => Band::Lowpass(freq),
            FilterType::Highpass => Band::Highpass(freq),
            FilterType::Bandpass | FilterType::Bandstop => {
                let hi = freq2.ok_or("band filters need a second frequency")?;
                if kind == FilterType::Bandpass {
                    Band::Bandpass(freq, hi)
                } else {
                    Band::Bandstop(freq, hi)
                }
            }
            _ => unreachable!(),
        })
    };
    let e = |e: adsp::fir::DesignError| e.to_string();
    Ok(match kind {
        FilterType::Peaking => AnyFilter::Bq(Biquad::peaking(freq, q, gain, fs)),
        FilterType::LowShelf => AnyFilter::Bq(Biquad::low_shelf(freq, q, gain, fs)),
        FilterType::HighShelf => AnyFilter::Bq(Biquad::high_shelf(freq, q, gain, fs)),
        FilterType::Notch => AnyFilter::Bq(Biquad::notch(freq, q, fs)),
        _ => match fir_taps {
            Some(n) => {
                let w = Window::Kaiser(8.0);
                let taps = match band()? {
                    Band::Lowpass(f) => fir::lowpass(n, f, fs, w),
                    Band::Highpass(f) => fir::highpass(n | 1, f, fs, w),
                    Band::Bandpass(a, b) => fir::bandpass(n, a, b, fs, w),
                    Band::Bandstop(a, b) => fir::bandstop(n | 1, a, b, fs, w),
                }
                .map_err(e)?;
                AnyFilter::Fir(Fir::new(taps.clone()), taps)
            }
            None => AnyFilter::Iir(butter(order, band()?, fs).map_err(e)?),
        },
    })
}

fn run(cli: Cli) -> Res<()> {
    match cli.cmd {
        Cmd::Info { file } => {
            let a = load(&file)?;
            println!("{}", file.display());
            println!(
                "  {} Hz, {} channel(s), {} frames, {:.3} s",
                a.sample_rate,
                a.channels.len(),
                a.frames(),
                a.duration_secs()
            );
            println!("  {:<8} {:>11} {:>10} {:>12} {:>14}", "channel", "peak dBFS", "RMS dBFS", "DC offset", "strongest Hz");
            for (i, ch) in a.channels.iter().enumerate() {
                let dc = ch.iter().sum::<f64>() / ch.len().max(1) as f64;
                let probe = &ch[..ch.len().min(a.sample_rate as usize * 4)];
                let f = if probe.len() >= 4 && peak(probe) > 0.0 {
                    format!("{:.1}", peak_frequency(probe, a.sample_rate as f64))
                } else {
                    "-".into()
                };
                println!("  {:<8} {:>11} {:>10} {:>12.6} {:>14}", i, db_text(peak(ch)), db_text(rms(ch)), dc, f);
            }
        }
        Cmd::Gen { out, kind, freq, to, seconds, rate, level, format } => {
            let n = (seconds * rate as f64).round() as usize;
            let amp = adsp::dynamics::db_to_gain(level);
            let x = match kind {
                Kind::Sine => signal::sine(n, freq, rate as f64, amp),
                Kind::Sweep => signal::log_sweep(n, freq, to, rate as f64, amp),
                Kind::Noise => signal::white_noise(n, amp, 1),
            };
            save(&out, &Audio { sample_rate: rate, channels: vec![x] }, format)?;
            println!("wrote {} ({} samples at {} Hz)", out.display(), n, rate);
        }
        Cmd::Filter { input, out, kind, freq, freq2, order, fir, gain, q, format } => {
            let mut a = load(&input)?;
            let fs = a.sample_rate as f64;
            for ch in a.channels.iter_mut() {
                let mut f = build(kind, freq, freq2, order, fir, gain, q, fs)?;
                for v in ch.iter_mut() {
                    *v = f.process(*v);
                }
            }
            save(&out, &a, format)?;
            println!("wrote {}", out.display());
        }
        Cmd::Resample { input, out, rate, format } => {
            let a = load(&input)?;
            let mut r = Resampler::between(a.sample_rate as usize, rate as usize);
            let channels = a.channels.iter().map(|c| r.run(c)).collect();
            let b = Audio { sample_rate: rate, channels };
            save(&out, &b, format)?;
            let (up, down) = r.ratio();
            println!(
                "wrote {} ({} -> {} Hz, ratio {up}/{down}, {} -> {} frames)",
                out.display(),
                a.sample_rate,
                rate,
                a.frames(),
                b.frames()
            );
        }
        Cmd::Gain { input, out, normalize: norm, compress, ratio, limit, format } => {
            let mut a = load(&input)?;
            let fs = a.sample_rate as f64;
            if norm.is_none() && compress.is_none() && limit.is_none() {
                return Err("nothing to do: give --normalize, --compress or --limit".into());
            }
            let before: Vec<f64> = a.channels.iter().map(|c| peak(c)).collect();
            for ch in a.channels.iter_mut() {
                if let Some(t) = compress {
                    let mut c = Compressor::new(t, ratio, 6.0, 5.0, 100.0, 0.0, fs);
                    c.process_block(ch);
                }
                if let Some(c) = limit {
                    let look = (0.005 * fs).round() as usize;
                    let mut l = Limiter::new(c, look, 50.0, fs);
                    // Feed silence after the end to flush the look-ahead delay, then drop the
                    // leading delay so the output lines up with the input.
                    let n = ch.len();
                    let mut y: Vec<f64> = ch.iter().map(|&v| l.process(v)).collect();
                    y.extend((0..l.latency()).map(|_| l.process(0.0)));
                    ch.copy_from_slice(&y[l.latency()..l.latency() + n]);
                }
                if let Some(t) = norm {
                    normalize(ch, t);
                }
            }
            save(&out, &a, format)?;
            for (i, (b, ch)) in before.iter().zip(&a.channels).enumerate() {
                println!("channel {i}: peak {} -> {} dBFS", db_text(*b), db_text(peak(ch)));
            }
        }
        Cmd::Spectrum { file, per_octave } => {
            let a = load(&file)?;
            let fs = a.sample_rate as f64;
            let x = &a.channels[0];
            let nper = 4096.min(x.len().next_power_of_two() / 2).max(16);
            if x.len() < nper {
                return Err("file too short for a spectrum".into());
            }
            let (freqs, psd) = welch(x, fs, Window::Hann, nper, nper / 2);
            let df = fs / nper as f64;
            let per = per_octave.max(1) as f64;
            // Start where a band is at least two bins wide, so no band is empty.
            let ratio = 2f64.powf(1.0 / per);
            let mut lo = (2.0 * df / (ratio - 1.0)).max(20.0);
            let mut rows = Vec::new();
            while lo < fs / 2.0 {
                let hi = (lo * 2f64.powf(1.0 / per)).min(fs / 2.0);
                let p: f64 = freqs.iter().zip(&psd).filter(|(f, _)| **f >= lo && **f < hi).map(|(_, p)| p * df).sum();
                rows.push((lo, hi, p));
                lo = hi;
            }
            let top = rows.iter().fold(0.0f64, |m, r| m.max(r.2)).max(1e-300);
            println!(
                "{} (channel 0, Welch, {} Hz resolution), band power relative to the strongest band",
                file.display(),
                df.round()
            );
            for (lo, hi, p) in rows {
                let db = 10.0 * (p / top).log10();
                let bar = ((db + 60.0).max(0.0) / 60.0 * 40.0).round() as usize;
                let shown = if db < -150.0 { "  < -150".to_string() } else { format!("{db:>8.1}") };
                println!("  {:>7.0} - {:<7.0} Hz {} dB  {}", lo, hi, shown, "#".repeat(bar));
            }
        }
        Cmd::Response { kind, freq, freq2, order, fir, gain, q, rate } => {
            let f = build(kind, freq, freq2, order, fir, gain, q, rate)?;
            println!("{:>8}  {:>9}  {:>9}", "Hz", "gain dB", "phase deg");
            let mut fq = 20.0f64;
            while fq < rate / 2.0 {
                let h = f.response(fq, rate);
                println!("{:>8.0}  {:>9.2}  {:>9.1}", fq, gain_to_db(h.abs()), h.arg().to_degrees());
                fq *= 2f64.powf(1.0 / 3.0);
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("adsp: {e}");
            ExitCode::from(2)
        }
    }
}
