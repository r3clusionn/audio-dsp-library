"""Plots the CSV files from `cargo run --release --example responses` into docs/images.

    cargo run --release --example responses -- target/plots
    python scripts/plots.py target/plots

Needs matplotlib. Every curve is the library's own output; nothing is computed here.
"""
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "target/plots"
dst = os.path.join(os.path.dirname(__file__), "..", "docs", "images")


def load(name):
    with open(os.path.join(src, name)) as f:
        rows = list(csv.reader(f))
    head, data = rows[0], [[float(v) for v in r] for r in rows[1:]]
    cols = list(zip(*data))
    return head, cols


plt.rcParams.update({"font.size": 10, "figure.dpi": 110})


def finish(ax, title, xlabel, ylabel, path, logx=True):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(dst, path))
    plt.close()


head, cols = load("filters.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, ys in zip(head[1:], cols[1:]):
    ax.plot(cols[0], ys, label=name)
ax.set_ylim(-140, 5)
ax.set_xlim(10, 24000)
finish(ax, "Low-pass designs at 1 kHz, 48 kHz sample rate", "Hz", "gain (dB)", "filters.png")

head, cols = load("eq.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, ys in zip(head[1:-1], cols[1:-1]):
    ax.plot(cols[0], ys, label=name, alpha=0.7)
ax.plot(cols[0], cols[-1], label="all four in series", color="black", linewidth=2)
ax.set_xlim(10, 24000)
ax.set_ylim(-12, 9)
finish(ax, "Cookbook equaliser biquads", "Hz", "gain (dB)", "eq.png")

head, cols = load("windows.csv")
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, ys in zip(head[1:], cols[1:]):
    ax.plot(cols[0], ys, label=name)
ax.set_xlim(0, 32)
ax.set_ylim(-160, 5)
finish(ax, "Spectra of 64-point windows", "frequency (bins of the 64-point transform)", "dB", "windows.png", logx=False)
print("wrote filters.png, eq.png and windows.png")
