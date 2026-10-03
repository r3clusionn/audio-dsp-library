//! Gain and dynamics: level measurement, normalisation, fades, a compressor, a look-ahead limiter
//! and a DC blocker.

use std::collections::VecDeque;

/// `10^(db / 20)`.
pub fn db_to_gain(db: f64) -> f64 {
    10f64.powf(db / 20.0)
}

/// `20 log10(gain)`; minus infinity for zero.
pub fn gain_to_db(gain: f64) -> f64 {
    20.0 * gain.abs().log10()
}

/// The largest absolute sample.
pub fn peak(x: &[f64]) -> f64 {
    x.iter().fold(0.0, |m, v| m.max(v.abs()))
}

/// Root mean square; zero for an empty signal.
pub fn rms(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt()
}

/// Scales `x` so its peak is `target_db` (dBFS). Silence is left alone. Returns the gain applied.
pub fn normalize(x: &mut [f64], target_db: f64) -> f64 {
    let p = peak(x);
    if p == 0.0 {
        return 1.0;
    }
    let g = db_to_gain(target_db) / p;
    for v in x.iter_mut() {
        *v *= g;
    }
    g
}

/// Fade curve shapes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Fade {
    Linear,
    /// Sine-shaped: a fade-out and a fade-in of the same length sum to constant power.
    EqualPower,
}

fn fade_gain(shape: Fade, t: f64) -> f64 {
    match shape {
        Fade::Linear => t,
        Fade::EqualPower => (t * std::f64::consts::FRAC_PI_2).sin(),
    }
}

/// Fades the first `len` samples in from silence.
pub fn fade_in(x: &mut [f64], len: usize, shape: Fade) {
    let len = len.min(x.len());
    for (i, v) in x[..len].iter_mut().enumerate() {
        *v *= fade_gain(shape, i as f64 / len as f64);
    }
}

/// Fades the last `len` samples out to silence.
pub fn fade_out(x: &mut [f64], len: usize, shape: Fade) {
    let n = x.len();
    let len = len.min(n);
    for i in 0..len {
        x[n - 1 - i] *= fade_gain(shape, i as f64 / len as f64);
    }
}

/// The coefficient of a one-pole smoother with time constant `ms`.
fn coeff(ms: f64, fs: f64) -> f64 {
    if ms <= 0.0 {
        0.0
    } else {
        (-1.0 / (ms * 0.001 * fs)).exp()
    }
}

/// A feed-forward compressor with a soft knee (the gain computer and smoothing of Giannoulis,
/// Massberg and Reiss, 2012). Levels are per-sample peak levels in dBFS.
#[derive(Clone, Debug)]
pub struct Compressor {
    pub threshold_db: f64,
    pub ratio: f64,
    pub knee_db: f64,
    pub makeup_db: f64,
    attack: f64,
    release: f64,
    /// Current gain reduction in dB (zero or negative).
    gr: f64,
}

impl Compressor {
    pub fn new(
        threshold_db: f64,
        ratio: f64,
        knee_db: f64,
        attack_ms: f64,
        release_ms: f64,
        makeup_db: f64,
        fs: f64,
    ) -> Compressor {
        assert!(ratio >= 1.0, "ratio must be at least 1");
        Compressor {
            threshold_db,
            ratio,
            knee_db: knee_db.max(0.0),
            makeup_db,
            attack: coeff(attack_ms, fs),
            release: coeff(release_ms, fs),
            gr: 0.0,
        }
    }

    /// The static curve: output level for a steady input level, both in dB.
    pub fn curve(&self, level_db: f64) -> f64 {
        let (t, r, w) = (self.threshold_db, self.ratio, self.knee_db);
        let over = level_db - t;
        if 2.0 * over < -w {
            level_db
        } else if 2.0 * over.abs() <= w && w > 0.0 {
            level_db + (1.0 / r - 1.0) * (over + w / 2.0).powi(2) / (2.0 * w)
        } else {
            t + over / r
        }
    }

    /// The gain reduction being applied right now, in dB (zero or negative).
    pub fn reduction_db(&self) -> f64 {
        self.gr
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        let level = if x == 0.0 { -200.0 } else { gain_to_db(x) };
        let target = self.curve(level) - level;
        let a = if target < self.gr { self.attack } else { self.release };
        self.gr = a * self.gr + (1.0 - a) * target;
        x * db_to_gain(self.gr + self.makeup_db)
    }

    pub fn process_block(&mut self, buf: &mut [f64]) {
        for v in buf {
            *v = self.process(*v);
        }
    }
}

/// A look-ahead brickwall limiter: the output never exceeds the ceiling, and the gain changes
/// smoothly because it starts falling `lookahead` samples before a peak arrives.
///
/// The required gain for each sample (`ceiling / |x|`, at most 1) is held at its minimum over the
/// look-ahead window and then averaged over the same window. Every value in that average is at
/// most the gain the delayed sample needs, so the average is too: the ceiling holds exactly, with
/// no clipping stage. Output is delayed by `lookahead - 1` samples.
#[derive(Clone, Debug)]
pub struct Limiter {
    ceiling: f64,
    len: usize,
    release: f64,
    delay: VecDeque<f64>,
    /// (index, required gain), increasing gains: the sliding minimum.
    mins: VecDeque<(u64, f64)>,
    held: VecDeque<f64>,
    held_sum: f64,
    released: f64,
    n: u64,
}

impl Limiter {
    pub fn new(ceiling_db: f64, lookahead: usize, release_ms: f64, fs: f64) -> Limiter {
        let len = lookahead.max(1);
        Limiter {
            ceiling: db_to_gain(ceiling_db),
            len,
            release: coeff(release_ms, fs),
            delay: VecDeque::from(vec![0.0; len - 1]),
            mins: VecDeque::new(),
            held: VecDeque::from(vec![1.0; len]),
            held_sum: len as f64,
            released: 1.0,
            n: 0,
        }
    }

    /// Samples of delay between input and output.
    pub fn latency(&self) -> usize {
        self.len - 1
    }

    pub fn process(&mut self, x: f64) -> f64 {
        let need = if x.abs() > self.ceiling { self.ceiling / x.abs() } else { 1.0 };
        let i = self.n;
        self.n += 1;
        while self.mins.back().is_some_and(|&(_, g)| g >= need) {
            self.mins.pop_back();
        }
        self.mins.push_back((i, need));
        while self.mins.front().is_some_and(|&(j, _)| j + (self.len as u64) <= i) {
            self.mins.pop_front();
        }
        let window_min = self.mins.front().unwrap().1;
        // Release: recover towards 1 smoothly, but never above what the window requires.
        self.released = (self.release * self.released + (1.0 - self.release)).min(window_min);
        let old = self.held.pop_front().unwrap();
        self.held.push_back(self.released);
        self.held_sum += self.released - old;
        if i.is_multiple_of(4096) {
            // Recompute now and then so rounding cannot accumulate.
            self.held_sum = self.held.iter().sum();
        }
        let gain = self.held_sum / self.len as f64;
        self.delay.push_back(x);
        let out = self.delay.pop_front().unwrap();
        out * gain
    }

    pub fn process_block(&mut self, buf: &mut [f64]) {
        for v in buf {
            *v = self.process(*v);
        }
    }
}

/// Removes DC: `y[n] = x[n] - x[n-1] + r y[n-1]`, a high-pass at about `(1 - r) fs / 2 pi`.
#[derive(Clone, Debug)]
pub struct DcBlocker {
    r: f64,
    x1: f64,
    y1: f64,
}

impl DcBlocker {
    /// `cutoff_hz` is the approximate -3 dB point.
    pub fn new(cutoff_hz: f64, fs: f64) -> DcBlocker {
        DcBlocker { r: 1.0 - 2.0 * std::f64::consts::PI * cutoff_hz / fs, x1: 0.0, y1: 0.0 }
    }

    pub fn process(&mut self, x: f64) -> f64 {
        let y = x - self.x1 + self.r * self.y1;
        self.x1 = x;
        self.y1 = y;
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_and_conversions() {
        assert!((db_to_gain(-6.0206) - 0.5).abs() < 1e-5);
        assert!((gain_to_db(0.1) + 20.0).abs() < 1e-12);
        assert_eq!(gain_to_db(0.0), f64::NEG_INFINITY);
        assert_eq!(peak(&[0.1, -0.7, 0.3]), 0.7);
        assert!((rms(&[1.0, -1.0, 1.0, -1.0]) - 1.0).abs() < 1e-15);
        assert_eq!(rms(&[]), 0.0);
        let mut x = vec![0.25, -0.5];
        let g = normalize(&mut x, 0.0);
        assert_eq!((g, x[1]), (2.0, -1.0));
        let mut silent = vec![0.0; 4];
        assert_eq!(normalize(&mut silent, -1.0), 1.0);
    }

    #[test]
    fn fades() {
        let mut x = vec![1.0; 10];
        fade_in(&mut x, 4, Fade::Linear);
        assert_eq!(&x[..5], &[0.0, 0.25, 0.5, 0.75, 1.0]);
        let mut y = vec![1.0; 10];
        fade_out(&mut y, 4, Fade::Linear);
        assert_eq!(&y[5..], &[1.0, 0.75, 0.5, 0.25, 0.0]);
        // Equal power: half-way through, the gain is sin(pi/4), so power is halved, not quartered.
        let mut e = vec![1.0; 100];
        fade_in(&mut e, 100, Fade::EqualPower);
        assert_eq!(e[0], 0.0);
        assert!((e[50] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((e[50] * e[50] - 0.5).abs() < 1e-12);
        // Lengths beyond the signal are clipped, not a panic.
        let mut z = vec![1.0; 3];
        fade_out(&mut z, 10, Fade::Linear);
    }

    #[test]
    fn the_compressor_curve() {
        let c = Compressor::new(-20.0, 4.0, 0.0, 1.0, 50.0, 0.0, 48_000.0);
        assert_eq!(c.curve(-30.0), -30.0);
        assert_eq!(c.curve(-20.0), -20.0);
        assert_eq!(c.curve(-8.0), -17.0);
        let soft = Compressor::new(-20.0, 4.0, 10.0, 1.0, 50.0, 0.0, 48_000.0);
        // Outside the knee the soft curve equals the hard one; inside it lies between.
        assert_eq!(soft.curve(-30.0), -30.0);
        assert_eq!(soft.curve(-8.0), -17.0);
        let k = soft.curve(-20.0);
        assert!(k < -20.0 && k > -20.0 - 10.0, "{k}");
        // Continuous at both knee edges.
        assert!((soft.curve(-25.0 + 1e-9) - soft.curve(-25.0 - 1e-9)).abs() < 1e-6);
        assert!((soft.curve(-15.0 + 1e-9) - soft.curve(-15.0 - 1e-9)).abs() < 1e-6);
    }

    #[test]
    fn the_compressor_settles_on_its_curve_with_its_time_constants() {
        let fs = 48_000.0;
        let mut c = Compressor::new(-20.0, 4.0, 0.0, 10.0, 100.0, 0.0, fs);
        // A steady level of 0.5 (-6.02 dBFS) settles at -20 + 13.98 / 4 = -16.51 dBFS.
        let level = gain_to_db(0.5);
        let want = -20.0 + (level + 20.0) / 4.0;
        let mut y = 0.0;
        for _ in 0..48_000 {
            y = c.process(0.5);
        }
        assert!((gain_to_db(y) - want).abs() < 1e-6, "{}", gain_to_db(y));
        let full = want - level;
        // Attack: after one time constant 63 percent of the 10.5 dB reduction is applied.
        let mut c = Compressor::new(-20.0, 4.0, 0.0, 10.0, 100.0, 0.0, fs);
        for _ in 0..480 {
            c.process(0.5);
        }
        let frac = c.reduction_db() / full;
        assert!((frac - (1.0 - (-1.0f64).exp())).abs() < 0.01, "{frac}");
        // Release is slower than attack.
        for _ in 0..480 {
            c.process(0.001);
        }
        // Release is slower than attack: after the same time, most of the reduction is still there.
        assert!(c.reduction_db() < 0.8 * frac * full, "{}", c.reduction_db());
    }

    #[test]
    fn the_limiter_never_exceeds_its_ceiling() {
        let fs = 48_000.0;
        let mut s = 99u64;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 20_000) as f64 / 10_000.0 - 1.0
        };
        for (ceiling, look) in [(-1.0, 64usize), (-6.0, 1), (-0.1, 480), (-20.0, 7)] {
            let mut l = Limiter::new(ceiling, look, 50.0, fs);
            let c = db_to_gain(ceiling);
            let mut worst: f64 = 0.0;
            for i in 0..200_000 {
                // Bursts up to +26 dB, quiet passages, single-sample spikes.
                let x = match (i / 5000) % 4 {
                    0 => rnd() * 20.0,
                    1 => rnd() * 0.05,
                    2 => {
                        if i % 997 == 0 {
                            15.0
                        } else {
                            rnd() * 0.3
                        }
                    }
                    _ => rnd() * 3.0,
                };
                worst = worst.max(l.process(x).abs());
            }
            assert!(worst <= c * (1.0 + 1e-12), "ceiling {ceiling} dB look-ahead {look}: peak {worst}");
            assert!(worst > c * 0.9, "it is a limiter, not a mute");
        }
    }

    #[test]
    fn the_limiter_passes_quiet_audio_unchanged_after_its_delay() {
        let mut l = Limiter::new(-1.0, 32, 50.0, 48_000.0);
        let x: Vec<f64> = (0..1000).map(|i| 0.5 * (i as f64 * 0.01).sin()).collect();
        let y: Vec<f64> = x.iter().map(|&v| l.process(v)).collect();
        assert_eq!(l.latency(), 31);
        for i in 31..1000 {
            assert!((y[i] - x[i - 31]).abs() < 1e-12);
        }
    }

    #[test]
    fn the_dc_blocker_removes_offset_and_keeps_audio() {
        let fs = 48_000.0;
        let mut d = DcBlocker::new(10.0, fs);
        let mut last = 0.0;
        for i in 0..96_000 {
            last = d.process(0.5 + 0.1 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / fs).sin());
        }
        // After two seconds the offset is gone and the 1 kHz tone remains at its level.
        let mut tail = Vec::new();
        for i in 96_000..96_480 {
            tail.push(d.process(0.5 + 0.1 * (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / fs).sin()));
        }
        let mean: f64 = tail.iter().sum::<f64>() / tail.len() as f64;
        assert!(mean.abs() < 1e-3, "{mean} {last}");
        assert!((rms(&tail) - 0.1 / 2f64.sqrt()).abs() < 1e-3);
    }
}
