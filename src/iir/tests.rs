use super::*;

const FS: f64 = 48_000.0;

fn db(z: Complex) -> f64 {
    20.0 * z.abs().log10()
}

#[test]
fn cookbook_designs_hit_their_defining_points() {
    let q = std::f64::consts::FRAC_1_SQRT_2;
    let lp = Biquad::lowpass(1000.0, q, FS);
    assert!((lp.response(0.0, FS).abs() - 1.0).abs() < 1e-12);
    assert!(lp.response(FS / 2.0, FS).abs() < 1e-12);
    assert!((db(lp.response(1000.0, FS)) + 3.0103).abs() < 1e-3, "Butterworth Q gives -3 dB at f0");
    let hp = Biquad::highpass(1000.0, q, FS);
    assert!(hp.response(0.0, FS).abs() < 1e-12);
    assert!((hp.response(FS / 2.0, FS).abs() - 1.0).abs() < 1e-12);
    let bp = Biquad::bandpass(2000.0, 2.0, FS);
    assert!((bp.response(2000.0, FS).abs() - 1.0).abs() < 1e-12);
    let notch = Biquad::notch(2000.0, 5.0, FS);
    assert!(notch.response(2000.0, FS).abs() < 1e-12);
    assert!((notch.response(0.0, FS).abs() - 1.0).abs() < 1e-12);
    let ap = Biquad::allpass(2000.0, 1.0, FS);
    for f in [0.0, 100.0, 2000.0, 15_000.0] {
        assert!((ap.response(f, FS).abs() - 1.0).abs() < 1e-12);
    }
    for g in [-12.0, -3.0, 6.0, 18.0] {
        let pk = Biquad::peaking(3000.0, 1.5, g, FS);
        assert!((db(pk.response(3000.0, FS)) - g).abs() < 1e-9, "peaking {g}");
        assert!(db(pk.response(0.0, FS)).abs() < 1e-9);
        let ls = Biquad::low_shelf(200.0, q, g, FS);
        assert!((db(ls.response(0.0, FS)) - g).abs() < 1e-9 && db(ls.response(FS / 2.0, FS)).abs() < 1e-9);
        let hs = Biquad::high_shelf(5000.0, q, g, FS);
        assert!((db(hs.response(FS / 2.0, FS)) - g).abs() < 1e-9 && db(hs.response(0.0, FS)).abs() < 1e-9);
    }
}

#[test]
fn processing_matches_the_difference_equation() {
    let mut f = Biquad::peaking(1000.0, 1.0, 6.0, FS);
    let c = f;
    let x: Vec<f64> = (0..200).map(|i| ((i * 7919) % 101) as f64 / 50.0 - 1.0).collect();
    let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
    for &xn in &x {
        let want = c.b0 * xn + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
        let got = f.process(xn);
        assert!((got - want).abs() < 1e-12);
        (x2, x1, y2, y1) = (x1, xn, y1, want);
    }
    f.reset();
    assert_eq!(f.process(0.0), 0.0);
}

#[test]
fn butterworth_is_3_db_down_at_the_cutoff_and_falls_at_6_db_per_order_per_octave() {
    for order in 1..=12 {
        let s = butter(order, Band::Lowpass(1000.0), FS).unwrap();
        assert!(s.is_stable());
        assert_eq!(s.sections.len(), order.div_ceil(2));
        assert!((s.response(0.0, FS).abs() - 1.0).abs() < 1e-9, "order {order}");
        assert!((db(s.response(1000.0, FS)) + 3.0103).abs() < 1e-3, "order {order}: {}", db(s.response(1000.0, FS)));
        // Maximally flat: monotonic decrease.
        let mut last = f64::MAX;
        for i in 0..100 {
            let g = s.response(i as f64 * 230.0, FS).abs();
            assert!(g <= last + 1e-12, "order {order} not monotonic");
            last = g;
        }
        // Far above the cutoff (but far from Nyquist, where the bilinear warp dominates).
        let slope = db(s.response(4000.0, FS)) - db(s.response(2000.0, FS));
        assert!((slope + 6.02 * order as f64).abs() < 0.6 * order as f64, "order {order}: {slope} dB per octave");
        let hp = butter(order, Band::Highpass(1000.0), FS).unwrap();
        assert!((hp.response(FS / 2.0, FS).abs() - 1.0).abs() < 1e-9);
        assert!((db(hp.response(1000.0, FS)) + 3.0103).abs() < 1e-3);
    }
}

#[test]
fn band_designs() {
    let bp = butter(4, Band::Bandpass(500.0, 2000.0), FS).unwrap();
    assert_eq!(bp.sections.len(), 4);
    assert!(bp.is_stable());
    let centre = (500.0f64 * 2000.0).sqrt();
    assert!((bp.response(centre, FS).abs() - 1.0).abs() < 1e-6);
    assert!((db(bp.response(500.0, FS)) + 3.0103).abs() < 1e-3);
    assert!((db(bp.response(2000.0, FS)) + 3.0103).abs() < 1e-3);
    assert!(bp.response(0.0, FS).abs() < 1e-9 && bp.response(FS / 2.0, FS).abs() < 1e-9);
    let bs = butter(3, Band::Bandstop(500.0, 2000.0), FS).unwrap();
    assert!(bs.is_stable());
    assert!(bs.response(centre, FS).abs() < 1e-6);
    assert!((bs.response(0.0, FS).abs() - 1.0).abs() < 1e-9 && (bs.response(FS / 2.0, FS).abs() - 1.0).abs() < 1e-9);
}

#[test]
fn chebyshev_ripple_stays_inside_its_bound_and_ends_at_the_bound() {
    for (order, ripple) in [(2usize, 1.0), (5, 0.5), (8, 3.0)] {
        let s = cheby1(order, ripple, Band::Lowpass(2000.0), FS).unwrap();
        assert!(s.is_stable());
        let mut lo = f64::MAX;
        let mut hi = f64::MIN;
        for i in 0..=400 {
            let g = db(s.response(i as f64 * 5.0, FS));
            lo = lo.min(g);
            hi = hi.max(g);
        }
        assert!(hi < 1e-9 && lo > -ripple - 1e-9, "order {order}: pass band {lo}..{hi}");
        assert!(lo < -ripple + 0.05, "order {order}: the ripple really reaches its depth");
        assert!((db(s.response(2000.0, FS)) + ripple).abs() < 1e-6, "edge of the pass band is at -ripple");
    }
}

#[test]
fn bad_designs_are_refused() {
    assert!(butter(0, Band::Lowpass(100.0), FS).is_err());
    assert!(butter(65, Band::Lowpass(100.0), FS).is_err());
    assert!(butter(2, Band::Lowpass(24_000.0), FS).is_err());
    assert!(butter(2, Band::Lowpass(0.0), FS).is_err());
    assert!(butter(2, Band::Bandpass(2000.0, 1000.0), FS).is_err());
    assert!(cheby1(2, 0.0, Band::Lowpass(100.0), FS).is_err());
    assert!(cheby1(2, f64::NAN, Band::Lowpass(100.0), FS).is_err());
}

#[test]
fn high_orders_stay_stable_and_accurate_thanks_to_sections() {
    let s = butter(40, Band::Lowpass(100.0), FS).unwrap();
    assert!(s.is_stable());
    assert!((db(s.response(100.0, FS)) + 3.0103).abs() < 1e-3);
    let mut f = s.clone();
    let mut buf = vec![0.0; 20_000];
    buf[0] = 1.0;
    f.process_block(&mut buf);
    assert!(buf.iter().all(|v| v.is_finite()));
    assert!(buf[19_000..].iter().all(|v| v.abs() < 1e-6), "the impulse response decays");
}

#[test]
fn the_stability_triangle() {
    let bq = |a1: f64, a2: f64| Biquad::new(1.0, 0.0, 0.0, 1.0, a1, a2);
    // Poles at 0.5 +- 0.5i (|p| = 0.71) and at 0.9 and 0.8: stable.
    assert!(bq(-1.0, 0.5).is_stable());
    assert!(bq(-1.7, 0.72).is_stable());
    // a2 inside the band but a real pole outside the circle: poles at 1.1 and 0.5.
    assert!(!bq(-1.6, 0.55).is_stable());
    // A pole pair outside the circle: |p|^2 = 1.2.
    assert!(!bq(-1.0, 1.2).is_stable());
    // The impulse response of the unstable one really grows.
    let mut f = bq(-1.6, 0.55);
    let mut last = f.process(1.0);
    for _ in 0..200 {
        last = f.process(0.0);
    }
    assert!(last.abs() > 1e6);
}
