"""The same operations as examples/bench.rs, in NumPy and SciPy, for context.

    python scripts/bench_scipy.py

Each figure is the median of 7 timings, each repeating the call for about 50 ms.
"""
import statistics
import time

import numpy as np
import scipy.signal as sig


def bench(f):
    t = time.perf_counter()
    f()
    once = max(time.perf_counter() - t, 1e-7)
    reps = max(1, min(1_000_000, int(0.05 / once)))
    runs = []
    for _ in range(7):
        t = time.perf_counter()
        for _ in range(reps):
            f()
        runs.append((time.perf_counter() - t) / reps)
    return statistics.median(runs)


def fmt(s):
    return f"{s * 1e6:.1f} us" if s < 1e-3 else f"{s * 1e3:.2f} ms"


rng = np.random.default_rng(1)
print("| Operation | NumPy / SciPy |")
print("|---|---|")
for n in [1024, 4096, 65536, 1 << 20, 44100, 48000, 1009, 65537]:
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    print(f"| FFT {n} | {fmt(bench(lambda: np.fft.fft(x)))} |")
fs = 48000.0
x = rng.standard_normal(48000) * 0.5
taps = sig.firwin(255, 4000.0, fs=fs, window=("kaiser", 8.0))
print(f"| FIR, 255 taps, lfilter, 1 s | {fmt(bench(lambda: sig.lfilter(taps, [1.0], x)))} |")
long = rng.standard_normal(48000) * 0.1
print(f"| fftconvolve with 1 s response | {fmt(bench(lambda: sig.fftconvolve(x, long)))} |")
sos = sig.butter(8, 1000.0, fs=fs, output="sos")
print(f"| Butterworth order 8, sosfilt, 1 s | {fmt(bench(lambda: sig.sosfilt(sos, x)))} |")
x441 = rng.standard_normal(44100) * 0.5
print(f"| resample_poly 160/147, 1 s | {fmt(bench(lambda: sig.resample_poly(x441, 160, 147)))} |")
ten = rng.standard_normal(480000) * 0.5
print(f"| welch 10 s, 4096 | {fmt(bench(lambda: sig.welch(ten, fs, nperseg=4096, noverlap=2048)))} |")
