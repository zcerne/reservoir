#!/usr/bin/env python
"""Two-tone intermodulation spectrum — nonlinearity by the frequency method.

Reads snap_point's time trace (NOT the 1Ddft bins) and FFTs it offline with a
Hann window. That choice is the whole point of the script: a rectangular DFT of
a finite record leaks -13 dB into the neighbouring bins and falls off only as
1/f, so a percent-level intermodulation sideband sitting ~4 resolution elements
from a fundamental would be buried in the fundamental's own sidelobes. A Hann
window's first sidelobe is -31 dB and decays as f^-3, which puts the leakage
floor well under the sidebands we are trying to see.

    python plotting/plot_twotone_spectrum.py <design_dir> <out.png> [peak ...]

`peak` values are the beat-peak amplitudes whose runs were saved with
--suffix p<peak> (the slurm script's convention). With none given the plain
unsuffixed run is read. With several, the last panel adds IMD3 vs drive — the
curve that says WHERE the medium turns nonlinear, not just that it does.

Reported per drive:
    P(f1), P(f2)                     fundamentals
    P(2f1-f2), P(2f2-f1)             third-order intermodulation
    P(3f1-2f2), P(3f2-2f1)           fifth-order
    IMD3 = (3rd order) / (fundamentals)
A linear medium gives IMD3 at the numerical floor; a saturating one gives
percent-level or more, growing as drive^2 until it compresses.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DESIGN = sys.argv[1]
OUT = sys.argv[2]
PEAKS = sys.argv[3:]

cfg = json.load(open(os.path.join(DESIGN, "simulation_data.json")))
f1 = 1.0 / float(cfg["source_1"]["lam"])
f2 = 1.0 / float(cfg["source_2"]["lam"])
df = abs(f2 - f1)
# label -> (frequency, order); f1 < f2 by construction of the design
LINES = [("f1", f1, 1), ("f2", f2, 1),
         ("2f1-f2", 2 * f1 - f2, 3), ("2f2-f1", 2 * f2 - f1, 3),
         ("3f1-2f2", 3 * f1 - 2 * f2, 5), ("3f2-2f1", 3 * f2 - 2 * f1, 5)]

SIMDIR = os.path.join(DESIGN, "simulation_gpumeep")
if not os.path.isdir(SIMDIR):
    SIMDIR = os.path.join(DESIGN, "simulation")


def spectrum(suffix):
    """Hann-windowed power spectrum of the exit Ey trace. Returns (f, P, t, E)."""
    tag = f"_{suffix}" if suffix else ""
    d = np.load(os.path.join(SIMDIR, f"snap_point{tag}.npz"))
    t = np.asarray(d["t"], float).reshape(-1)
    # 0Dsnap stores (N, 1), not (N,) — the point monitor returns a 1-element
    # array per step. Left as-is, `E * w` broadcasts (N,1) against (N,) into an
    # N x N matrix: 6.4 GB at N = 80000, which OOM-kills the process rather
    # than erroring. Flatten to the point's own series first.
    E = np.asarray(d["Ey"], float).reshape(len(t), -1)[:, 0]
    dt = float(np.median(np.diff(t)))
    w = np.hanning(len(E))
    # coherent gain of the Hann window, so bin heights stay comparable to the
    # rectangular case (0.5 for hanning); power scales as its square.
    F = np.fft.rfft(E * w) / (len(E) * 0.5)
    return np.fft.rfftfreq(len(E), d=dt), np.abs(F) ** 2, t, E


def line_powers(f, P):
    """Integrate power within +-1 resolution element of each expected line."""
    res = f[1] - f[0]
    half = max(2.0 * res, 0.25 * df)          # never wider than half the spacing
    out = {}
    for name, nu, order in LINES:
        m = np.abs(f - nu) <= half
        out[name] = (float(P[m].sum()), nu, order)
    return out


runs = PEAKS or [""]
n = len(runs)
extra = 1 if n > 1 else 0
fig, axes = plt.subplots(n + extra, 2 if not extra else 2,
                         figsize=(13, 3.4 * (n + extra)), squeeze=False)

summary = []
for row, pk in enumerate(runs):
    suffix = f"p{pk}" if pk else ""
    f, P, t, E = spectrum(suffix)
    pw = line_powers(f, P)
    fund = pw["f1"][0] + pw["f2"][0]
    imd3 = (pw["2f1-f2"][0] + pw["2f2-f1"][0]) / (fund + 1e-300)
    imd5 = (pw["3f1-2f2"][0] + pw["3f2-2f1"][0]) / (fund + 1e-300)
    summary.append((float(pk) if pk else np.nan, fund, imd3, imd5, pw))

    ax = axes[row][0]                                  # beat envelope in time
    ax.plot(t, E, lw=0.4, color="tab:blue")
    ax.set_ylabel(f"peak {pk or '-'}\nEy at exit")
    ax.set_xlim(t[0], t[0] + 3.0 / df)                 # three beat periods
    ax.grid(alpha=0.25)
    if row == 0:
        ax.set_title(f"exit trace — beat period 1/df = {1/df:.0f} t.u.")

    ax = axes[row][1]                                  # spectrum
    band = (f > f1 - 6 * df) & (f < f2 + 6 * df)
    floor = max(P[band].max() * 1e-10, 1e-300)
    ax.semilogy(f[band], np.maximum(P[band], floor), lw=1.0, color="k")
    for name, nu, order in LINES:
        c = {1: "tab:green", 3: "tab:red", 5: "tab:orange"}[order]
        ax.axvline(nu, color=c, ls="--", lw=1.0, alpha=0.8)
        ax.text(nu, P[band].max(), name, rotation=90, fontsize=7, color=c,
                ha="right", va="top")
    ax.set_ylabel("power (Hann)")
    ax.grid(alpha=0.25)
    ax.text(0.015, 0.06, f"IMD3 {imd3:.3e}   IMD5 {imd5:.3e}",
            transform=ax.transAxes, fontsize=9)
    if row == 0:
        ax.set_title("Hann-windowed exit spectrum — fundamentals, 3rd, 5th order")

axes[len(runs) - 1 + 0][0].set_xlabel("time [MEEP units]")
axes[len(runs) - 1 + 0][1].set_xlabel("frequency [c/um]")

if extra:                                              # IMD3 vs drive
    for ax in axes[-1]:
        ax.remove()
    ax = fig.add_subplot(len(runs) + 1, 1, len(runs) + 1)
    a = np.array([s[0] for s in summary])
    ax.loglog(a, [s[2] for s in summary], "o-", color="tab:red", label="IMD3 (3rd order)")
    ax.loglog(a, [s[3] for s in summary], "s-", color="tab:orange", label="IMD5 (5th order)")
    ref = np.array([s[2] for s in summary])[0] * (a / a[0]) ** 2
    ax.loglog(a, ref, "k:", lw=1, label="drive^2 (weakly-nonlinear slope)")
    ax.set_xlabel("beat-peak drive amplitude")
    ax.set_ylabel("power / fundamentals")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    ax.set_title("intermodulation vs drive — where the medium turns nonlinear")

fig.suptitle(f"{os.path.basename(DESIGN)} — two-tone intermodulation "
             f"(f1 {f1:.6f}, f2 {f2:.6f}, df {df:.3e})", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)

print(f"{'peak':>8} {'fundamentals':>14} {'IMD3':>12} {'IMD5':>12}")
for pk, fund, imd3, imd5, pw in summary:
    print(f"{pk:>8g} {fund:>14.4e} {imd3:>12.4e} {imd5:>12.4e}")
    for name, (p, nu, order) in pw.items():
        print(f"         {name:>8s}  f={nu:.6f}  P={p:.4e}  (order {order})")
