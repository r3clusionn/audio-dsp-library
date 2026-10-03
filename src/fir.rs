//! FIR filters: windowed-sinc design, a streaming filter, and convolution.
//!
//! [`firwin`] follows SciPy's `scipy.signal.firwin` exactly (same band layout, same scaling), so a
//! design can be checked against it coefficient for coefficient. Frequencies are given as fractions
//! of the Nyquist frequency (0 to 1); the helpers [`lowpass`], [`highpass`], [`bandpass`] and
//! [`bandstop`] take hertz.

use std::f64::consts::PI;
use std::fmt;

use crate::complex::Complex;
use crate::fft::RealFft;
use crate::window::{kaiser_beta, Window};

#[derive(Clone, Debug, PartialEq)]
pub enum DesignError {
    /// A cutoff is outside (0, 1) of Nyquist, or the cutoffs are not strictly increasing.
    BadCutoff(String),
    /// A filter that passes the Nyquist frequency needs an odd number of taps.
    EvenTapsPassNyquist,
    NoTaps,
    /// An order of zero, or above what the design supports.
    BadOrder(usize),
}

impl fmt::Display for DesignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DesignError::BadCutoff(s) => write!(f, "bad cutoff: {s}"),
            DesignError::EvenTapsPassNyquist => {
                write!(f, "a filter that passes the Nyquist frequency needs an odd number of taps")
            }
            DesignError::NoTaps => write!(f, "a filter needs at least one tap"),
            DesignError::BadOrder(n) => write!(f, "unsupported filter order {n}"),
        }
    }
}

impl std::error::Error for DesignError {}

/// `sin(pi x) / (pi x)`.
pub fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// Windowed-sinc design, as `scipy.signal.firwin(numtaps, cutoffs, window=..., pass_zero=...)`.
///
/// `cutoffs` are band edges as fractions of Nyquist. With `pass_zero` the first band (from 0 Hz)
/// passes: one cutoff gives a low-pass, two a band-stop. Without it: a high-pass or band-pass.
/// The result is scaled to unit gain at the centre of the first pass band (0 Hz, Nyquist, or the
/// middle of the band).
pub fn firwin(numtaps: usize, cutoffs: &[f64], pass_zero: bool, window: Window) -> Result<Vec<f64>, DesignError> {
    if numtaps == 0 {
        return Err(DesignError::NoTaps);
    }
    if cutoffs.is_empty() {
        return Err(DesignError::BadCutoff("no cutoff given".into()));
    }
    for (i, &c) in cutoffs.iter().enumerate() {
        if !(c > 0.0 && c < 1.0) {
            return Err(DesignError::BadCutoff(format!("{c} is not strictly between 0 and 1 (fractions of Nyquist)")));
        }
        if i > 0 && c <= cutoffs[i - 1] {
            return Err(DesignError::BadCutoff("cutoffs must be strictly increasing".into()));
        }
    }
    let pass_nyquist = (cutoffs.len() % 2 == 1) ^ pass_zero;
    if pass_nyquist && numtaps.is_multiple_of(2) {
        return Err(DesignError::EvenTapsPassNyquist);
    }
    let mut edges = Vec::with_capacity(cutoffs.len() + 2);
    if pass_zero {
        edges.push(0.0);
    }
    edges.extend_from_slice(cutoffs);
    if pass_nyquist {
        edges.push(1.0);
    }
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let m: Vec<f64> = (0..numtaps).map(|i| i as f64 - alpha).collect();
    let mut h = vec![0.0; numtaps];
    for band in edges.as_chunks::<2>().0 {
        let (left, right) = (band[0], band[1]);
        for (hi, &mi) in h.iter_mut().zip(&m) {
            *hi += right * sinc(right * mi) - left * sinc(left * mi);
        }
    }
    let w = window.generate(numtaps, false);
    for (hi, wi) in h.iter_mut().zip(&w) {
        *hi *= wi;
    }
    let (left, right) = (edges[0], edges[1]);
    let scale_freq = if left == 0.0 {
        0.0
    } else if right == 1.0 {
        1.0
    } else {
        0.5 * (left + right)
    };
    let s: f64 = h.iter().zip(&m).map(|(hi, mi)| hi * (PI * mi * scale_freq).cos()).sum();
    for hi in h.iter_mut() {
        *hi /= s;
    }
    Ok(h)
}

/// The Kaiser-window design rule (`scipy.signal.kaiserord`): the number of taps and the `beta` that
/// give `atten_db` of stop-band attenuation with a transition `width` (fraction of Nyquist).
pub fn kaiserord(atten_db: f64, width: f64) -> (usize, f64) {
    let a = atten_db.abs();
    let taps = ((a - 7.95) / 2.285 / (PI * width) + 1.0).ceil().max(1.0) as usize;
    (taps, kaiser_beta(a))
}

fn nyq(f: f64, fs: f64) -> f64 {
    f / (fs / 2.0)
}

pub fn lowpass(numtaps: usize, cutoff_hz: f64, fs: f64, window: Window) -> Result<Vec<f64>, DesignError> {
    firwin(numtaps, &[nyq(cutoff_hz, fs)], true, window)
}

pub fn highpass(numtaps: usize, cutoff_hz: f64, fs: f64, window: Window) -> Result<Vec<f64>, DesignError> {
    firwin(numtaps, &[nyq(cutoff_hz, fs)], false, window)
}

pub fn bandpass(numtaps: usize, low_hz: f64, high_hz: f64, fs: f64, window: Window) -> Result<Vec<f64>, DesignError> {
    firwin(numtaps, &[nyq(low_hz, fs), nyq(high_hz, fs)], false, window)
}

pub fn bandstop(numtaps: usize, low_hz: f64, high_hz: f64, fs: f64, window: Window) -> Result<Vec<f64>, DesignError> {
    firwin(numtaps, &[nyq(low_hz, fs), nyq(high_hz, fs)], true, window)
}

/// The frequency response of `taps` at `freq_hz`.
pub fn response(taps: &[f64], freq_hz: f64, fs: f64) -> Complex {
    let w = 2.0 * PI * freq_hz / fs;
    taps.iter().enumerate().fold(Complex::ZERO, |acc, (k, &h)| acc + Complex::cis(-w * k as f64).scale(h))
}

/// A streaming FIR filter. Each input sample produces one output sample.
#[derive(Clone, Debug)]
pub struct Fir {
    taps: Vec<f64>,
    /// The last `n` inputs, stored twice so the newest `n` are always one contiguous slice.
    hist: Vec<f64>,
    pos: usize,
}

impl Fir {
    pub fn new(taps: Vec<f64>) -> Fir {
        assert!(!taps.is_empty(), "a filter needs at least one tap");
        let n = taps.len();
        // Reversed, so the dot product runs forward over the history.
        let taps: Vec<f64> = taps.into_iter().rev().collect();
        Fir { taps, hist: vec![0.0; 2 * n], pos: 0 }
    }

    pub fn len(&self) -> usize {
        self.taps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.taps.is_empty()
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let n = self.taps.len();
        self.hist[self.pos] = x;
        self.hist[self.pos + n] = x;
        self.pos = (self.pos + 1) % n;
        // hist[pos .. pos + n] is oldest to newest.
        let window = &self.hist[self.pos..self.pos + n];
        window.iter().zip(&self.taps).map(|(a, b)| a * b).sum()
    }

    pub fn process_block(&mut self, input: &[f64], output: &mut [f64]) {
        for (y, &x) in output.iter_mut().zip(input) {
            *y = self.process(x);
        }
    }

    pub fn reset(&mut self) {
        self.hist.fill(0.0);
        self.pos = 0;
    }
}

/// Full linear convolution, directly: `a.len() + b.len() - 1` samples.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut y = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            y[i + j] += ai * bj;
        }
    }
    y
}

/// Full linear convolution through the FFT: the same result as [`convolve`], in `O(n log n)`.
pub fn fft_convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let len = a.len() + b.len() - 1;
    let n = len.next_power_of_two().max(2);
    let fft = RealFft::new(n);
    let mut pa = a.to_vec();
    pa.resize(n, 0.0);
    let mut pb = b.to_vec();
    pb.resize(n, 0.0);
    let fa = fft.forward(&pa);
    let fb = fft.forward(&pb);
    let prod: Vec<Complex> = fa.iter().zip(&fb).map(|(x, y)| *x * *y).collect();
    let mut y = fft.inverse(&prod);
    y.truncate(len);
    y
}

/// Streaming FFT convolution by overlap-add: long filters (reverbs, measured responses) at a cost
/// per sample that grows with `log` of the block size instead of with the filter length.
///
/// Output is produced a block at a time, so it lags the input by up to `block - 1` samples; the
/// samples themselves are exactly those of [`Fir`] with the same taps.
pub struct OverlapAdd {
    block: usize,
    fft: RealFft,
    kernel: Vec<Complex>,
    input: Vec<f64>,
    tail: Vec<f64>,
    taps: usize,
    /// Input samples taken and output samples emitted by `process` since the last `finish`.
    consumed: usize,
    emitted: usize,
}

impl OverlapAdd {
    /// `block` input samples are transformed at a time (a power of two around the filter length is
    /// usually fastest).
    pub fn new(taps: &[f64], block: usize) -> OverlapAdd {
        assert!(!taps.is_empty() && block > 0);
        let n = (block + taps.len() - 1).next_power_of_two().max(2);
        let fft = RealFft::new(n);
        let mut k = taps.to_vec();
        k.resize(n, 0.0);
        let kernel = fft.forward(&k);
        OverlapAdd {
            block,
            fft,
            kernel,
            input: Vec::with_capacity(block),
            tail: vec![0.0; taps.len() - 1],
            taps: taps.len(),
            consumed: 0,
            emitted: 0,
        }
    }

    /// Feeds input and appends every output sample that is complete to `out`.
    pub fn process(&mut self, mut input: &[f64], out: &mut Vec<f64>) {
        while !input.is_empty() {
            let take = (self.block - self.input.len()).min(input.len());
            self.input.extend_from_slice(&input[..take]);
            self.consumed += take;
            input = &input[take..];
            if self.input.len() == self.block {
                self.run_block(out);
                self.emitted += self.block;
            }
        }
    }

    /// Processes what is buffered (padded with silence) and emits the filter's ring-out: after
    /// this, exactly the samples [`convolve`] would produce have been emitted, and the convolver is
    /// ready for a new signal.
    pub fn finish(&mut self, out: &mut Vec<f64>) {
        let target = self.consumed + self.taps - 1;
        let start = out.len();
        if !self.input.is_empty() {
            self.input.resize(self.block, 0.0);
            self.run_block(out);
        }
        out.extend_from_slice(&self.tail);
        // The padded block and the tail run past the end of the convolution; drop the zeros.
        let emitted = self.emitted + (out.len() - start);
        out.truncate(out.len() - (emitted - target));
        self.tail.fill(0.0);
        self.consumed = 0;
        self.emitted = 0;
    }

    fn run_block(&mut self, out: &mut Vec<f64>) {
        let n = self.fft.len();
        let mut buf = std::mem::take(&mut self.input);
        buf.resize(n, 0.0);
        let spec: Vec<Complex> = self.fft.forward(&buf).iter().zip(&self.kernel).map(|(a, b)| *a * *b).collect();
        let y = self.fft.inverse(&spec);
        let t = self.taps - 1;
        for (i, v) in y[..self.block].iter().enumerate() {
            out.push(v + if i < t { self.tail[i] } else { 0.0 });
        }
        // New tail: what spills past this block, plus the old tail's part beyond this block.
        let mut tail = vec![0.0; t];
        for (i, slot) in tail.iter_mut().enumerate() {
            let old = if self.block + i < t { self.tail[self.block + i] } else { 0.0 };
            *slot = y[self.block + i] + old;
        }
        self.tail = tail;
        buf.clear();
        self.input = buf;
    }
}

#[cfg(test)]
mod tests;
