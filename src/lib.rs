//! Audio DSP building blocks with no dependencies.
//!
//! | Module | What it has |
//! |---|---|
//! | [`fft`] | complex FFT of any length (radix-2, mixed radix, Bluestein) and a real-input FFT |
//! | [`window`] | Hann, Hamming, Blackman, Blackman-Harris, Nuttall, flat-top, Bartlett, Kaiser |
//! | [`fir`] | windowed-sinc design (as SciPy's `firwin`), a streaming filter, direct, FFT and overlap-add convolution |
//! | [`iir`] | cookbook biquads, Butterworth and Chebyshev type I designs of any order as second-order sections |
//! | [`resample`] | a polyphase resampler matching `scipy.signal.resample_poly`, streaming; linear and cubic interpolation |
//! | [`dynamics`] | levels, normalisation, fades, a compressor, a look-ahead limiter, a DC blocker |
//! | [`spectrum`] | Welch's method (as SciPy's `welch`), Goertzel, sub-bin peak frequency |
//! | [`signal`] | test signals: sine, exponential sweep, impulse, white noise |
//! | [`wav`] | WAV reading and writing |
//!
//! ```
//! use adsp::iir::{butter, Band};
//! use adsp::resample::Resampler;
//! use adsp::signal::sine;
//! use adsp::spectrum::peak_frequency;
//!
//! // A 1 kHz tone at 44.1 kHz, low-passed and converted to 48 kHz.
//! let mut x = sine(44_100, 1000.0, 44_100.0, 0.5);
//! let mut lp = butter(4, Band::Lowpass(5000.0), 44_100.0).unwrap();
//! lp.process_block(&mut x);
//! let y = Resampler::between(44_100, 48_000).run(&x);
//! assert_eq!(y.len(), 48_000);
//! assert!((peak_frequency(&y, 48_000.0) - 1000.0).abs() < 0.5);
//! ```

pub mod complex;
pub mod dynamics;
pub mod fft;
pub mod fir;
pub mod iir;
pub mod resample;
pub mod signal;
pub mod spectrum;
pub mod wav;
pub mod window;

pub use complex::Complex;
pub use fft::{Fft, RealFft};
pub use window::Window;
