#!/usr/bin/env python
"""signal_modulation/design01: point traces at the crystal exit for three
drives. The DFT window starts at 3000, so the sensors only record the SETTLED
state — for a CW experiment that is the useful part: a flat envelope whose
LEVEL is the observable, and whose growth with drive is (or isn't) linear."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else (
    "/home/ziga/Lips/resevoir/data/signal_modulation/design01/simulation_gpumeep")
OUT = sys.argv[2] if len(sys.argv) > 2 else (
    "/home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode/data/"
    "signal_modulation/design01/figures/traces_3amp.png")
CASES = [(1, "small", "tab:blue"), (22, "medium", "tab:green"),
         (46, "large", "tab:red")]

fig, axes = plt.subplots(3, 2, figsize=(14, 8.5))
levels = []
for row, (amp, label, color) in enumerate(CASES):
    d = np.load(os.path.join(RES, f"snap_point_a{amp}.npz"))
    t = d["t"]
    E = np.sqrt(d["Ey"].ravel() ** 2 + d["Ez"].ravel() ** 2)
    env = np.convolve(E, np.ones(201) / 201, "same")
    lvl = float(np.median(env[(t > 3500) & (t < 5500)]))
    levels.append((amp, lvl))

    ax = axes[row, 0]           # a few optical cycles, to show it is CW
    m = (t > 4000) & (t < 4040)
    ax.plot(t[m], d["Ey"].ravel()[m], lw=1.0, color=color, label="Ey")
    ax.plot(t[m], d["Ez"].ravel()[m], lw=1.0, color="0.5", label="Ez")
    ax.set_ylabel(f"drive {amp} ({label})\nE at exit")
    ax.grid(alpha=0.25)
    if row == 0:
        ax.set_title("40 t.u. of the carrier — steady CW, not a pulse")
        ax.legend(fontsize=7)

    ax = axes[row, 1]           # the whole recorded window
    ax.plot(t, E, lw=0.4, color=color, alpha=0.5)
    ax.plot(t, env, lw=1.6, color="k")
    ax.axhline(lvl, color="0.4", ls=":", lw=1)
    ax.set_ylabel("|E| envelope")
    ax.grid(alpha=0.25)
    ax.text(0.015, 0.9, f"steady level {lvl:.4g}", transform=ax.transAxes,
            fontsize=9, va="top")
    if row == 0:
        ax.set_title("whole recorded window (3000-6000) — flat = settled")
for ax in axes[-1]:
    ax.set_xlabel("time [MEEP units]")

base = levels[0]
note = "  |  ".join(f"drive {a}: level {l:.3g} ({l/base[1]:.2f}x vs drive 1, "
                    f"linear would be {a/base[0]:.0f}x)" for a, l in levels)
fig.suptitle("signal_modulation/design01 — Nd:YAG exit traces at three drives\n"
             + note, fontsize=10)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
for a, l in levels:
    print(f"  drive {a:4d}: steady |E| {l:.4g}   ({l/base[1]:6.2f}x vs drive 1, linear = {a/base[0]:4.0f}x)")
