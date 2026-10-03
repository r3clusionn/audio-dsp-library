use super::*;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    fn complex(&mut self, n: usize) -> Vec<Complex> {
        (0..n).map(|_| Complex::new(self.next(), self.next())).collect()
    }
}

fn max_err(a: &[Complex], b: &[Complex]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max)
}

#[test]
fn every_length_up_to_200_matches_the_direct_transform() {
    let mut rng = Rng(1);
    for n in 0..=200 {
        let x = rng.complex(n);
        let mut y = x.clone();
        let f = Fft::new(n);
        f.forward(&mut y);
        let e = max_err(&y, &dft(&x));
        assert!(e < 1e-11 * (n.max(1) as f64), "n = {n} ({}): error {e}", f.algorithm());
        f.inverse(&mut y);
        assert!(max_err(&y, &x) < 1e-13 * (n.max(2) as f64).log2().max(1.0) * 10.0, "n = {n}: round trip");
    }
}

#[test]
fn the_algorithm_follows_the_length() {
    assert_eq!(Fft::new(1).algorithm(), "trivial");
    assert_eq!(Fft::new(1024).algorithm(), "radix-2");
    assert_eq!(Fft::new(44_100).algorithm(), "mixed radix");
    assert_eq!(Fft::new(48_000).algorithm(), "mixed radix");
    assert_eq!(Fft::new(1009).algorithm(), "Bluestein");
    assert_eq!(Fft::new(2 * 17).algorithm(), "Bluestein");
    assert_eq!(Fft::new(13 * 13).algorithm(), "mixed radix");
}

/// Tones whose spectrum is known exactly: a cosine at bin `b` with amplitude `a` puts `a * n / 2`
/// in bins `b` and `n - b`, and nothing anywhere else.
fn tone_check(n: usize) {
    let tones = [(3usize, 1.0, 0.0), (n / 7, 0.5, 0.3), (n / 2 - 1, 0.25, -1.1)];
    let mut x = vec![Complex::ZERO; n];
    for &(b, a, ph) in &tones {
        for (j, xj) in x.iter_mut().enumerate() {
            let ang = 2.0 * PI * ((b * j) % n) as f64 / n as f64 + ph;
            xj.re += a * ang.cos();
        }
    }
    let f = Fft::new(n);
    f.forward(&mut x);
    let mut expect = vec![Complex::ZERO; n];
    for &(b, a, ph) in &tones {
        expect[b] += Complex::from_polar(a * n as f64 / 2.0, ph);
        expect[n - b] += Complex::from_polar(a * n as f64 / 2.0, -ph);
    }
    let e = max_err(&x, &expect);
    assert!(e < 1e-8 * n as f64, "n = {n} ({}): error {e}", f.algorithm());
}

#[test]
fn large_transforms_of_known_tones() {
    for n in [1 << 16, 1 << 20, 44_100, 48_000, 96_000, 65_537, 100_003, 3 * 3 * 5 * 7 * 11 * 13] {
        tone_check(n);
    }
}

#[test]
fn parseval_and_linearity() {
    let mut rng = Rng(9);
    for n in [64usize, 100, 127, 4096, 6000] {
        let f = Fft::new(n);
        let x = rng.complex(n);
        let y = rng.complex(n);
        let mut fx = x.clone();
        let mut fy = y.clone();
        f.forward(&mut fx);
        f.forward(&mut fy);
        let e_time: f64 = x.iter().map(|z| z.norm_sqr()).sum();
        let e_freq: f64 = fx.iter().map(|z| z.norm_sqr()).sum::<f64>() / n as f64;
        assert!((e_time - e_freq).abs() < 1e-9 * e_time, "Parseval n = {n}");
        let mut s: Vec<Complex> = x.iter().zip(&y).map(|(a, b)| *a * 2.0 + *b * -3.0).collect();
        f.forward(&mut s);
        let lin: Vec<Complex> = fx.iter().zip(&fy).map(|(a, b)| *a * 2.0 + *b * -3.0).collect();
        assert!(max_err(&s, &lin) < 1e-9 * n as f64, "linearity n = {n}");
    }
}

#[test]
fn an_impulse_is_flat_and_a_shift_is_a_phase_ramp() {
    for n in [8usize, 30, 31] {
        let f = Fft::new(n);
        let mut x = vec![Complex::ZERO; n];
        x[0] = Complex::ONE;
        f.forward(&mut x);
        assert!(x.iter().all(|z| (*z - Complex::ONE).abs() < 1e-12));
        let mut d = vec![Complex::ZERO; n];
        d[3] = Complex::ONE;
        f.forward(&mut d);
        for (k, z) in d.iter().enumerate() {
            let want = Complex::cis(-2.0 * PI * (3 * k % n) as f64 / n as f64);
            assert!((*z - want).abs() < 1e-12, "n = {n} k = {k}");
        }
    }
}

#[test]
fn real_transform_matches_the_complex_one() {
    let mut rng = Rng(3);
    for n in (1..=130).chain([1024, 4410, 1009 * 2, 1009]) {
        let x: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        let r = RealFft::new(n);
        let spec = r.forward(&x);
        assert_eq!(spec.len(), n / 2 + 1);
        let mut c: Vec<Complex> = x.iter().map(|&v| Complex::from(v)).collect();
        Fft::new(n).forward(&mut c);
        let e = max_err(&spec, &c[..n / 2 + 1]);
        assert!(e < 1e-10 * n as f64, "n = {n}: error {e}");
        let back = r.inverse(&spec);
        let e = back.iter().zip(&x).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!(e < 1e-12 * n as f64, "n = {n}: round trip error {e}");
    }
}

#[test]
fn scratch_lets_repeated_transforms_reuse_memory() {
    let f = Fft::new(1000);
    let mut scratch = vec![Complex::ZERO; f.scratch_len()];
    let mut rng = Rng(5);
    let x = rng.complex(1000);
    let mut a = x.clone();
    let mut b = x.clone();
    f.forward(&mut a);
    f.forward_with_scratch(&mut b, &mut scratch);
    assert_eq!(a, b);
}

#[test]
#[should_panic(expected = "does not match the plan")]
fn a_wrong_length_is_refused() {
    Fft::new(8).forward(&mut [Complex::ZERO; 7]);
}
