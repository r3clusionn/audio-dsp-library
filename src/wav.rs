//! Reading and writing WAV files.
//!
//! Reads PCM (8, 16, 24 and 32 bit), IEEE float (32 and 64 bit) and the `WAVE_FORMAT_EXTENSIBLE`
//! header that wraps either. Samples are converted to `f64` in `[-1, 1)` and split into channels.
//! Writes 16-bit, 24-bit or 32-bit float. Chunks it does not need (`LIST`, `fact`, ...) are skipped.

use std::fmt;
use std::io::{self, Read, Write};

#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    pub sample_rate: u32,
    /// One vector per channel, all the same length.
    pub channels: Vec<Vec<f64>>,
}

impl Audio {
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    pub fn duration_secs(&self) -> f64 {
        self.frames() as f64 / self.sample_rate as f64
    }
}

/// How samples are stored when writing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Pcm16,
    Pcm24,
    Float32,
}

#[derive(Debug)]
pub enum WavError {
    Io(io::Error),
    /// Not a RIFF/WAVE file, or its structure is damaged.
    Invalid(String),
    /// A valid file in a format this reader does not handle.
    Unsupported(String),
}

impl fmt::Display for WavError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WavError::Io(e) => write!(f, "{e}"),
            WavError::Invalid(s) => write!(f, "not a valid WAV file: {s}"),
            WavError::Unsupported(s) => write!(f, "unsupported WAV format: {s}"),
        }
    }
}

impl std::error::Error for WavError {}

impl From<io::Error> for WavError {
    fn from(e: io::Error) -> Self {
        WavError::Io(e)
    }
}

fn u16le(b: &[u8]) -> u16 {
    u16::from_le_bytes([b[0], b[1]])
}

fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Parses a whole WAV file.
pub fn parse(data: &[u8]) -> Result<Audio, WavError> {
    let bad = |s: &str| WavError::Invalid(s.to_string());
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(bad("missing RIFF/WAVE header"));
    }
    let mut pos = 12;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    let mut samples: Option<&[u8]> = None;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32le(&data[pos + 4..pos + 8]) as usize;
        let body_start = pos + 8;
        // A truncated last chunk is common for data written by crashed recorders: take what exists.
        let body_end = body_start.saturating_add(size).min(data.len());
        let body = &data[body_start..body_end];
        match id {
            b"fmt " => {
                if body.len() < 16 {
                    return Err(bad("fmt chunk too short"));
                }
                let mut tag = u16le(&body[0..2]);
                let channels = u16le(&body[2..4]);
                let rate = u32le(&body[4..8]);
                let bits = u16le(&body[14..16]);
                if tag == 0xFFFE {
                    if body.len() < 26 {
                        return Err(bad("extensible fmt chunk too short"));
                    }
                    // The sub-format GUID starts with the real format tag.
                    tag = u16le(&body[24..26]);
                }
                fmt = Some((tag, channels, rate, bits));
            }
            b"data" => samples = Some(body),
            _ => {}
        }
        // Chunks are padded to an even length.
        pos = body_start.saturating_add(size).saturating_add(size & 1);
    }
    let (tag, channels, rate, bits) = fmt.ok_or_else(|| bad("no fmt chunk"))?;
    let raw = samples.ok_or_else(|| bad("no data chunk"))?;
    if channels == 0 || rate == 0 {
        return Err(bad("zero channels or zero sample rate"));
    }
    let width = match (tag, bits) {
        (1, 8 | 16 | 24 | 32) | (3, 32 | 64) => bits as usize / 8,
        _ => return Err(WavError::Unsupported(format!("format tag {tag} with {bits} bits"))),
    };
    let nch = channels as usize;
    let frames = raw.len() / (width * nch);
    let mut out = vec![Vec::with_capacity(frames); nch];
    for f in 0..frames {
        for (c, ch) in out.iter_mut().enumerate() {
            let s = &raw[(f * nch + c) * width..][..width];
            let v = match (tag, bits) {
                (1, 8) => (s[0] as f64 - 128.0) / 128.0,
                (1, 16) => i16::from_le_bytes([s[0], s[1]]) as f64 / 32768.0,
                (1, 24) => (i32::from_le_bytes([0, s[0], s[1], s[2]]) >> 8) as f64 / 8_388_608.0,
                (1, 32) => i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64 / 2_147_483_648.0,
                (3, 32) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f64,
                _ => f64::from_le_bytes(s.try_into().unwrap()),
            };
            ch.push(v);
        }
    }
    Ok(Audio { sample_rate: rate, channels: out })
}

pub fn read(mut r: impl Read) -> Result<Audio, WavError> {
    let mut data = Vec::new();
    r.read_to_end(&mut data)?;
    parse(&data)
}

pub fn read_file(path: impl AsRef<std::path::Path>) -> Result<Audio, WavError> {
    parse(&std::fs::read(path)?)
}

/// Encodes `audio` as a WAV file. PCM samples are rounded and clipped to the format's range.
pub fn encode(audio: &Audio, format: Format) -> Vec<u8> {
    let nch = audio.channels.len() as u16;
    let frames = audio.frames();
    let (tag, bits): (u16, u16) = match format {
        Format::Pcm16 => (1, 16),
        Format::Pcm24 => (1, 24),
        Format::Float32 => (3, 32),
    };
    let width = bits as usize / 8;
    let data_len = frames * nch as usize * width;
    let mut v = Vec::with_capacity(44 + data_len);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&tag.to_le_bytes());
    v.extend_from_slice(&nch.to_le_bytes());
    v.extend_from_slice(&audio.sample_rate.to_le_bytes());
    v.extend_from_slice(&(audio.sample_rate * nch as u32 * width as u32).to_le_bytes());
    v.extend_from_slice(&(nch * width as u16).to_le_bytes());
    v.extend_from_slice(&bits.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&(data_len as u32).to_le_bytes());
    for f in 0..frames {
        for ch in &audio.channels {
            let x = ch[f];
            match format {
                Format::Pcm16 => v.extend_from_slice(&((x * 32768.0).round().clamp(-32768.0, 32767.0) as i16).to_le_bytes()),
                Format::Pcm24 => {
                    let i = (x * 8_388_608.0).round().clamp(-8_388_608.0, 8_388_607.0) as i32;
                    v.extend_from_slice(&i.to_le_bytes()[..3]);
                }
                Format::Float32 => v.extend_from_slice(&(x as f32).to_le_bytes()),
            }
        }
    }
    if data_len % 2 == 1 {
        v.push(0);
    }
    v
}

pub fn write(mut w: impl Write, audio: &Audio, format: Format) -> io::Result<()> {
    w.write_all(&encode(audio, format))
}

pub fn write_file(path: impl AsRef<std::path::Path>, audio: &Audio, format: Format) -> io::Result<()> {
    std::fs::write(path, encode(audio, format))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo() -> Audio {
        let l: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.05).sin() * 0.9).collect();
        let r: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.11).cos() * -0.5).collect();
        Audio { sample_rate: 44_100, channels: vec![l, r] }
    }

    #[test]
    fn round_trips_within_the_format_precision() {
        let a = stereo();
        for (fmt, tol) in [(Format::Pcm16, 1.0 / 32768.0), (Format::Pcm24, 1.0 / 8_388_608.0), (Format::Float32, 1e-7)] {
            let b = parse(&encode(&a, fmt)).unwrap();
            assert_eq!(b.sample_rate, 44_100);
            assert_eq!(b.channels.len(), 2);
            assert_eq!(b.frames(), 1000);
            for (ca, cb) in a.channels.iter().zip(&b.channels) {
                for (x, y) in ca.iter().zip(cb) {
                    assert!((x - y).abs() <= tol, "{fmt:?}: {x} vs {y}");
                }
            }
        }
    }

    #[test]
    fn clipping_and_extremes() {
        let a = Audio { sample_rate: 8000, channels: vec![vec![-1.0, 1.0, 2.0, -3.0, 0.0]] };
        let b = parse(&encode(&a, Format::Pcm16)).unwrap();
        assert_eq!(b.channels[0], vec![-1.0, 32767.0 / 32768.0, 32767.0 / 32768.0, -1.0, 0.0]);
        let c = parse(&encode(&a, Format::Pcm24)).unwrap();
        assert_eq!(c.channels[0][0], -1.0);
        assert_eq!(c.channels[0][2], 8_388_607.0 / 8_388_608.0);
        // An odd-length data chunk is padded, and the file still parses.
        let odd = Audio { sample_rate: 8000, channels: vec![vec![0.5]] };
        let e = encode(&odd, Format::Pcm24);
        assert_eq!(e.len() % 2, 0);
        assert!((parse(&e).unwrap().channels[0][0] - 0.5).abs() < 1e-6);
    }

    /// A hand-made file: 8-bit PCM, an extra LIST chunk with odd length before the data.
    #[test]
    fn foreign_layouts() {
        let mut f = Vec::new();
        f.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        f.extend_from_slice(b"fmt \x10\0\0\0\x01\0\x01\0\x40\x1f\0\0\x40\x1f\0\0\x01\0\x08\0");
        f.extend_from_slice(b"LIST\x03\0\0\0abc\0");
        f.extend_from_slice(b"data\x03\0\0\0\x00\x80\xff\0");
        let a = parse(&f).unwrap();
        assert_eq!(a.sample_rate, 8000);
        assert_eq!(a.channels[0], vec![-1.0, 0.0, 127.0 / 128.0]);
        // Extensible header wrapping 32-bit float.
        let mut g = Vec::new();
        g.extend_from_slice(b"RIFF\0\0\0\0WAVE");
        g.extend_from_slice(b"fmt \x28\0\0\0\xfe\xff\x01\0\x80\xbb\0\0\0\xee\x02\0\x04\0\x20\0");
        g.extend_from_slice(&[22, 0, 32, 0, 4, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0x10, 0, 0x80, 0, 0, 0xaa, 0, 0x38, 0x9b, 0x71]);
        g.extend_from_slice(b"data\x04\0\0\0");
        g.extend_from_slice(&0.25f32.to_le_bytes());
        let b = parse(&g).unwrap();
        assert_eq!((b.sample_rate, b.channels[0][0]), (48_000, 0.25));
    }

    #[test]
    fn damaged_and_unsupported_files_are_errors_not_panics() {
        assert!(matches!(parse(b"hello"), Err(WavError::Invalid(_))));
        assert!(matches!(parse(b"RIFF\0\0\0\0WAVE"), Err(WavError::Invalid(_))));
        let mut no_data = Vec::new();
        no_data.extend_from_slice(b"RIFF\0\0\0\0WAVEfmt \x10\0\0\0\x01\0\x01\0\x40\x1f\0\0\x40\x1f\0\0\x02\0\x10\0");
        assert!(matches!(parse(&no_data), Err(WavError::Invalid(_))));
        let mut adpcm = no_data.clone();
        adpcm[20] = 2;
        adpcm.extend_from_slice(b"data\0\0\0\0");
        assert!(matches!(parse(&adpcm), Err(WavError::Unsupported(_))));
        // Every truncation of a valid file is handled.
        let good = encode(&stereo(), Format::Pcm16);
        for cut in 0..good.len() {
            let _ = parse(&good[..cut]);
        }
        // A data chunk that claims more than the file holds yields the frames that exist.
        let short = &good[..good.len() - 10];
        assert_eq!(parse(short).unwrap().frames(), 997);
    }
}
