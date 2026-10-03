//! Sample-rate conversion.
//!
//! [`Resampler`] converts by a rational factor `up / down` with a polyphase windowed-sinc filter.
//! With the default settings it computes exactly what `scipy.signal.resample_poly(x, up, down)`
//! does: the same Kaiser-windowed filter (`beta = 5`, `10 * max(up, down)` taps either side of the
//! centre, cutoff at the lower of the two Nyquist frequencies), the same alignment, the same output
//! length. Unlike that function it also works on a stream, block by block, with identical output.
//!
//! [`linear`] and [`cubic`] interpolate at any ratio, cheaply and with aliasing; they are here for
//! comparison and for uses where that does not matter.

use crate::fir::firwin;
use crate::window::Window;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Converts by `up / down` (reduced to lowest terms). Feed input with [`process`](Self::process)
/// and end with [`finish`](Self::finish).
#[derive(Clone, Debug)]
pub struct Resampler {
    up: usize,
    down: usize,
    /// The filter, scaled by `up`, with SciPy's zero padding in front.
    h: Vec<f64>,
    /// Output index 0 corresponds to this index into the upsampled-and-filtered signal.
    pre: usize,
    /// Inputs received so far, and the most recent of them (enough to cover the filter).
    received: usize,
    hist: Vec<f64>,
    /// The next output sample to produce.
    next_out: usize,
}

impl Resampler {
    /// SciPy's defaults: a Kaiser window with `beta = 5` and `10 * max(up, down)` taps per side.
    pub fn new(up: usize, down: usize) -> Resampler {
        Resampler::with_window(up, down, Window::Kaiser(5.0), 10)
    }

    /// Converts between two sample rates, e.g. 44,100 Hz to 48,000 Hz (147 to 160).
    pub fn between(from_hz: usize, to_hz: usize) -> Resampler {
        Resampler::new(to_hz, from_hz)
    }

    /// A custom filter: `window` and `half_taps_per_rate` taps either side of the centre for each unit
    /// of `max(up, down)`. More taps give a steeper transition at more cost.
    pub fn with_window(up: usize, down: usize, window: Window, half_taps_per_rate: usize) -> Resampler {
        assert!(up > 0 && down > 0, "rates must be positive");
        let g = gcd(up, down);
        let (up, down) = (up / g, down / g);
        let max_rate = up.max(down);
        let half_len = half_taps_per_rate * max_rate;
        let mut taps = if up == 1 && down == 1 {
            vec![1.0]
        } else {
            firwin(2 * half_len + 1, &[1.0 / max_rate as f64], true, window).expect("valid resampling filter")
        };
        for t in taps.iter_mut() {
            *t *= up as f64;
        }
        // Pad the front so that output samples land on the filter's centre (as SciPy does).
        let (pre_pad, half) = if up == 1 && down == 1 { (0, 0) } else { (down - half_len % down, half_len) };
        let mut h = vec![0.0; pre_pad];
        h.extend(taps);
        let pre = (half + pre_pad) / down;
        let keep = h.len().div_ceil(up) + 1;
        Resampler { up, down, h, pre, received: 0, hist: vec![0.0; keep], next_out: 0 }
    }

    /// The reduced ratio `(up, down)`.
    pub fn ratio(&self) -> (usize, usize) {
        (self.up, self.down)
    }

    /// The number of output samples for `n` input samples: `ceil(n * up / down)`.
    pub fn output_len(&self, n: usize) -> usize {
        (n * self.up).div_ceil(self.down)
    }

    /// Input at index `i`, or zero before the start; only valid for recent indices.
    fn x(&self, i: usize) -> f64 {
        let k = self.hist.len();
        if i >= self.received || i + k < self.received {
            0.0
        } else {
            self.hist[i % k]
        }
    }

    /// The output sample `m`, using inputs up to (but not including) `limit`.
    fn compute(&self, m: usize) -> f64 {
        let t = (m + self.pre) * self.down;
        let len = self.h.len();
        // Inputs i with 0 <= t - i*up < len.
        let i_max = t / self.up;
        let i_min = if t + 1 > len { (t + 1 - len).div_ceil(self.up) } else { 0 };
        let mut acc = 0.0;
        for i in i_min..=i_max {
            acc += self.x(i) * self.h[t - i * self.up];
        }
        acc
    }

    /// The last input index output `m` depends on.
    fn needs(&self, m: usize) -> usize {
        (m + self.pre) * self.down / self.up
    }

    /// Takes more input and appends every output sample that can now be computed.
    pub fn process(&mut self, input: &[f64], out: &mut Vec<f64>) {
        let k = self.hist.len();
        for &v in input {
            self.hist[self.received % k] = v;
            self.received += 1;
            while self.needs(self.next_out) < self.received {
                out.push(self.compute(self.next_out));
                self.next_out += 1;
            }
        }
    }

    /// Emits the remaining output (treating input after the end as silence) and resets.
    pub fn finish(&mut self, out: &mut Vec<f64>) {
        let total = self.output_len(self.received);
        // Pretend silence keeps arriving without overwriting what is still needed.
        while self.next_out < total {
            let need = self.needs(self.next_out);
            while self.received <= need {
                let k = self.hist.len();
                self.hist[self.received % k] = 0.0;
                self.received += 1;
            }
            out.push(self.compute(self.next_out));
            self.next_out += 1;
        }
        self.received = 0;
        self.next_out = 0;
        self.hist.fill(0.0);
    }

    /// Converts a whole signal at once.
    pub fn run(&mut self, input: &[f64]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.output_len(input.len()));
        self.process(input, &mut out);
        self.finish(&mut out);
        out
    }
}

/// `scipy.signal.resample_poly(x, up, down)`.
pub fn resample_poly(x: &[f64], up: usize, down: usize) -> Vec<f64> {
    Resampler::new(up, down).run(x)
}

/// Linear interpolation to `ratio` times as many samples (`ratio = to_hz / from_hz`).
pub fn linear(x: &[f64], ratio: f64) -> Vec<f64> {
    let n = ((x.len() as f64) * ratio).round() as usize;
    (0..n)
        .map(|m| {
            let pos = m as f64 / ratio;
            let i = pos.floor() as usize;
            let f = pos - i as f64;
            let a = x.get(i).copied().unwrap_or(0.0);
            let b = x.get(i + 1).copied().unwrap_or(0.0);
            a + (b - a) * f
        })
        .collect()
}

/// Cubic (Catmull-Rom) interpolation to `ratio` times as many samples.
pub fn cubic(x: &[f64], ratio: f64) -> Vec<f64> {
    let n = ((x.len() as f64) * ratio).round() as usize;
    let at = |i: isize| if i < 0 { 0.0 } else { x.get(i as usize).copied().unwrap_or(0.0) };
    (0..n)
        .map(|m| {
            let pos = m as f64 / ratio;
            let i = pos.floor() as isize;
            let t = pos - i as f64;
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
        })
        .collect()
}

#[cfg(test)]
mod tests;
