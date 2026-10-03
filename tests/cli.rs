//! Runs the `adsp` binary on generated files.

use std::path::Path;
use std::process::{Command, Output};

use adsp::dynamics::{gain_to_db, peak, rms};
use adsp::wav;

fn adsp(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_adsp")).args(args).output().expect("run adsp")
}

fn ok(args: &[&str]) -> String {
    let o = adsp(args);
    assert!(o.status.success(), "{args:?}: {}", String::from_utf8_lossy(&o.stderr));
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn p(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

#[test]
fn generate_inspect_and_resample() {
    let d = tempdir();
    let tone = p(&d, "tone.wav");
    ok(&["gen", &tone, "sine", "--freq", "997", "--seconds", "2", "--rate", "44100", "--level", "-6"]);
    let info = ok(&["info", &tone]);
    assert!(info.contains("44100 Hz, 1 channel(s), 88200 frames, 2.000 s"), "{info}");
    assert!(info.contains("-6.00") && info.contains("997.0"), "{info}");
    let out = p(&d, "tone48.wav");
    let msg = ok(&["resample", &tone, &out, "48000"]);
    assert!(msg.contains("ratio 160/147") && msg.contains("88200 -> 96000 frames"), "{msg}");
    let a = wav::read_file(&out).unwrap();
    assert_eq!((a.sample_rate, a.frames()), (48_000, 96_000));
    assert!((gain_to_db(peak(&a.channels[0][1000..95_000])) + 6.0).abs() < 0.05);
}

#[test]
fn filters_do_what_they_say() {
    let d = tempdir();
    let fs = 48_000.0;
    // A file holding 200 Hz and 5 kHz at equal levels.
    let x: Vec<f64> = (0..48_000)
        .map(|i| {
            let t = i as f64 / fs;
            0.4 * (2.0 * std::f64::consts::PI * 200.0 * t).sin() + 0.4 * (2.0 * std::f64::consts::PI * 5000.0 * t).sin()
        })
        .collect();
    let input = p(&d, "mix.wav");
    wav::write_file(&input, &wav::Audio { sample_rate: 48_000, channels: vec![x] }, wav::Format::Float32).unwrap();
    let level = |file: &str, f: f64| {
        let a = wav::read_file(file).unwrap();
        adsp::spectrum::goertzel(&a.channels[0][4800..], f, fs).abs() / (a.frames() - 4800) as f64 * 2.0
    };
    for (args, keep, kill) in [
        (vec!["lowpass", "1000"], 200.0, 5000.0),
        (vec!["highpass", "1000", "--order", "8"], 5000.0, 200.0),
        (vec!["lowpass", "1000", "--fir", "255"], 200.0, 5000.0),
        (vec!["bandstop", "3000", "7000"], 200.0, 5000.0),
        (vec!["notch", "5000", "--q", "2"], 200.0, 5000.0),
    ] {
        let out = p(&d, "f.wav");
        let mut a = vec!["filter", input.as_str(), out.as_str()];
        a.extend(args.iter().copied());
        a.extend(["--format", "f32"]);
        ok(&a);
        let (k, x) = (level(&out, keep), level(&out, kill));
        assert!((k - 0.4).abs() < 0.02, "{args:?}: kept tone at {k}");
        assert!(x < 0.4 * 0.01, "{args:?}: removed tone still at {x}");
    }
    // A peaking filter adds its gain at its centre.
    let out = p(&d, "eq.wav");
    ok(&["filter", &input, &out, "peaking", "5000", "--gain", "6", "--q", "1", "--format", "f32"]);
    assert!((gain_to_db(level(&out, 5000.0) / 0.4) - 6.0).abs() < 0.1);
}

#[test]
fn gain_commands() {
    let d = tempdir();
    let n = p(&d, "noise.wav");
    ok(&["gen", &n, "noise", "--level", "-1", "--seconds", "1"]);
    let out = p(&d, "lim.wav");
    let msg = ok(&["gain", &n, &out, "--limit", "-12", "--format", "f32"]);
    assert!(msg.contains("-> -12.00 dBFS") || msg.contains("-> -12"), "{msg}");
    let a = wav::read_file(&out).unwrap();
    assert!(peak(&a.channels[0]) <= adsp::dynamics::db_to_gain(-12.0) * (1.0 + 1e-6));
    assert_eq!(a.frames(), 48_000, "the look-ahead delay is removed, not appended");
    let out2 = p(&d, "norm.wav");
    ok(&["gain", &n, &out2, "--normalize", "-3", "--format", "f32"]);
    assert!((gain_to_db(peak(&wav::read_file(&out2).unwrap().channels[0])) + 3.0).abs() < 1e-4);
    let out3 = p(&d, "comp.wav");
    ok(&["gain", &n, &out3, "--compress", "-20", "--ratio", "8", "--format", "f32"]);
    let c = wav::read_file(&out3).unwrap();
    let before = wav::read_file(&n).unwrap();
    assert!(rms(&c.channels[0]) < 0.6 * rms(&before.channels[0]));
}

#[test]
fn spectrum_and_response() {
    let d = tempdir();
    let s = p(&d, "sweep.wav");
    ok(&["gen", &s, "sweep", "--freq", "20", "--to", "20000", "--seconds", "3"]);
    let text = ok(&["spectrum", &s]);
    assert!(text.lines().count() > 20, "{text}");
    assert!(!text.contains("-120"), "no empty bands: {text}");
    let r = ok(&["response", "lowpass", "1000", "--order", "2"]);
    // Rows are a third of an octave apart from 20 Hz: the 18th is 20 * 2^(17/3) = 1016 Hz.
    let f = 20.0 * 2f64.powf(17.0 / 3.0);
    let row = r.lines().find(|l| l.trim_start().starts_with("1016")).unwrap_or_else(|| panic!("{r}"));
    let gain: f64 = row.split_whitespace().nth(1).unwrap().parse().unwrap();
    let want = gain_to_db(adsp::iir::butter(2, adsp::iir::Band::Lowpass(1000.0), 48_000.0).unwrap().response(f, 48_000.0).abs());
    assert!((gain - want).abs() < 0.006, "{row} vs {want}");
}

#[test]
fn errors_are_clear() {
    let d = tempdir();
    let o = adsp(&["info", &p(&d, "missing.wav")]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("missing.wav"));
    let junk = p(&d, "junk.wav");
    std::fs::write(&junk, b"not audio").unwrap();
    let o = adsp(&["info", &junk]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("not a valid WAV file"));
    let s = p(&d, "s.wav");
    ok(&["gen", &s, "sine"]);
    let o = adsp(&["filter", &s, &p(&d, "o.wav"), "bandpass", "100"]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("second frequency"));
    let o = adsp(&["filter", &s, &p(&d, "o.wav"), "lowpass", "30000"]);
    assert_eq!(o.status.code(), Some(2));
    let o = adsp(&["gain", &s, &p(&d, "o.wav")]);
    assert!(String::from_utf8_lossy(&o.stderr).contains("nothing to do"));
}

fn tempdir() -> std::path::PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!("adsp-cli-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&d).unwrap();
    d
}
