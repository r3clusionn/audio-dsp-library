"""Writes tests/fixtures/scipy.json: inputs and what NumPy and SciPy compute from them.

The Rust tests compare against this file, so SciPy is only needed to regenerate it:

    python scripts/make_fixtures.py

Generated with NumPy 2.5.3 and SciPy 1.18.1.
"""
import json
import math
import os

import numpy as np
import scipy
import scipy.signal as sig

rng = np.random.default_rng(20261002)
out = {"numpy": np.__version__, "scipy": scipy.__version__}


def c(z):
    return [[float(v.real), float(v.imag)] for v in np.asarray(z).ravel()]


def r(v):
    return [float(x) for x in np.asarray(v).ravel()]


# FFT of complex input, every algorithm path.
out["fft"] = []
for n in [1, 2, 3, 5, 7, 8, 12, 16, 17, 31, 32, 60, 64, 97, 100, 128, 243, 256, 1000, 1009, 1024, 2310]:
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    out["fft"].append({"x": c(x), "y": c(np.fft.fft(x))})

# Windows.
names = {"hann": "Hann", "hamming": "Hamming", "blackman": "Blackman", "blackmanharris": "BlackmanHarris",
         "nuttall": "Nuttall", "flattop": "FlatTop", "bartlett": "Bartlett"}
out["windows"] = []
for name in list(names) + [("kaiser", 8.6)]:
    for n in [7, 32, 33]:
        for periodic in [True, False]:
            w = sig.get_window(name, n, fftbins=periodic)
            label = "Kaiser" if isinstance(name, tuple) else names[name]
            out["windows"].append({"name": label, "n": n, "periodic": periodic, "w": r(w)})

# firwin.
out["firwin"] = []
for numtaps, cutoff, pass_zero, window in [
    (101, [0.2], True, "hamming"),
    (51, [0.3], False, "hann"),
    (64, [0.25], True, "blackman"),
    (201, [0.1, 0.3], False, ("kaiser", 8.0)),
    (151, [0.2, 0.5], True, "hamming"),
    (31, [0.05, 0.2, 0.5, 0.8], False, "hann"),
]:
    h = sig.firwin(numtaps, cutoff, window=window, pass_zero=pass_zero)
    wname = "Kaiser" if isinstance(window, tuple) else names[window]
    beta = window[1] if isinstance(window, tuple) else None
    out["firwin"].append({"numtaps": numtaps, "cutoff": cutoff, "pass_zero": pass_zero, "window": wname, "beta": beta, "h": r(h)})

# IIR designs: SciPy's sections, its filtered output and response; pairing may differ from ours,
# so outputs and responses are compared, not coefficients.
fs = 48000.0
x_iir = rng.standard_normal(600)
out["iir_input"] = r(x_iir)
out["iir"] = []
for kind, order, btype, wn, ripple in [
    ("butter", 1, "lowpass", [1000.0], None),
    ("butter", 4, "lowpass", [1000.0], None),
    ("butter", 7, "highpass", [3000.0], None),
    ("butter", 12, "lowpass", [200.0], None),
    ("butter", 3, "bandpass", [500.0, 2000.0], None),
    ("butter", 4, "bandstop", [1000.0, 4000.0], None),
    ("cheby1", 5, "lowpass", [2000.0], 0.5),
    ("cheby1", 4, "highpass", [500.0], 1.0),
    ("cheby1", 3, "bandpass", [300.0, 3400.0], 2.0),
    ("cheby1", 2, "bandstop", [900.0, 1100.0], 0.1),
]:
    w = wn[0] if len(wn) == 1 else wn
    if kind == "butter":
        sos = sig.butter(order, w, btype=btype, fs=fs, output="sos")
    else:
        sos = sig.cheby1(order, ripple, w, btype=btype, fs=fs, output="sos")
    freqs = np.array([0.0, 50.0, 300.0, 999.0, 1500.0, 5000.0, 12000.0, 23000.0])
    _, h = sig.sosfreqz(sos, worN=freqs, fs=fs)
    out["iir"].append({"kind": kind, "order": order, "btype": btype, "wn": wn, "ripple": ripple,
                       "y": r(sig.sosfilt(sos, x_iir)), "freqs": r(freqs), "h": c(h)})

# Cookbook biquads, transcribed independently from the Audio EQ Cookbook, run through lfilter.
def cookbook(kind, f0, q, gain, fs):
    A = 10 ** (gain / 40)
    w0 = 2 * math.pi * f0 / fs
    cs, al = math.cos(w0), math.sin(w0) / (2 * q)
    if kind == "lowpass":
        b, a = [(1 - cs) / 2, 1 - cs, (1 - cs) / 2], [1 + al, -2 * cs, 1 - al]
    elif kind == "peaking":
        b, a = [1 + al * A, -2 * cs, 1 - al * A], [1 + al / A, -2 * cs, 1 - al / A]
    elif kind == "low_shelf":
        sq = 2 * math.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cs + sq), 2 * A * ((A - 1) - (A + 1) * cs), A * ((A + 1) - (A - 1) * cs - sq)]
        a = [(A + 1) + (A - 1) * cs + sq, -2 * ((A - 1) + (A + 1) * cs), (A + 1) + (A - 1) * cs - sq]
    elif kind == "high_shelf":
        sq = 2 * math.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cs + sq), -2 * A * ((A - 1) + (A + 1) * cs), A * ((A + 1) + (A - 1) * cs - sq)]
        a = [(A + 1) - (A - 1) * cs + sq, 2 * ((A - 1) - (A + 1) * cs), (A + 1) - (A - 1) * cs - sq]
    elif kind == "notch":
        b, a = [1, -2 * cs, 1], [1 + al, -2 * cs, 1 - al]
    return b, a


out["biquads"] = []
for kind, f0, q, gain in [("lowpass", 1200.0, 0.8, 0.0), ("peaking", 3000.0, 2.0, 7.5), ("low_shelf", 150.0, 0.707, -6.0),
                          ("high_shelf", 8000.0, 0.9, 4.0), ("notch", 60.0, 10.0, 0.0)]:
    b, a = cookbook(kind, f0, q, gain, fs)
    out["biquads"].append({"kind": kind, "f0": f0, "q": q, "gain": gain, "b": r(np.array(b) / a[0]),
                           "a": r(np.array(a) / a[0]), "y": r(sig.lfilter(b, a, x_iir))})

# resample_poly.
x_rs = rng.standard_normal(500)
out["resample_input"] = r(x_rs)
out["resample"] = [{"up": u, "down": d, "y": r(sig.resample_poly(x_rs, u, d))}
                   for u, d in [(3, 2), (2, 3), (160, 147), (147, 160), (1, 4), (5, 1), (6, 4)]]

# convolve.
a1, b1 = rng.standard_normal(300), rng.standard_normal(77)
out["convolve"] = {"a": r(a1), "b": r(b1), "y": r(np.convolve(a1, b1))}

# welch.
x_w = rng.standard_normal(5000) + 0.5 * np.sin(2 * np.pi * 123.0 * np.arange(5000) / 1000.0)
out["welch_input"] = r(x_w)
out["welch"] = []
for window, nper, nover in [("hann", 256, 128), ("hamming", 300, 100), ("blackman", 1024, 0)]:
    f, p = sig.welch(x_w, fs=1000.0, window=window, nperseg=nper, noverlap=nover)
    out["welch"].append({"window": names[window], "nperseg": nper, "noverlap": nover, "f": r(f), "p": r(p)})

path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "scipy.json")
with open(path, "w") as fh:
    json.dump(out, fh)
print("wrote", os.path.getsize(path), "bytes")
