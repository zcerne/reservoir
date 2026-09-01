#!/usr/bin/env python
"""signal_modulation/design01: the amplitude sweep — does the Nd:YAG crystal
saturate? Steady-state DFT window (3000-6000), 1064 nm.

Two readings of the same data, because the ratio and the increment say
different things: GAIN (out/in) tends to 1 as the drive grows, while ADDED
energy (out - in) is what the pump can actually supply per unit time."""
import glob
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = sys.argv[1]
OUT = sys.argv[2]


def power(mon, a, band=False):
    d = np.load(os.path.join(RES, f"{mon}_a{a}.npz"))
    lam = 1 / d["freqs"]
    P = np.abs(d["Ey"]) ** 2 + np.abs(d["Ez"]) ** 2
    P = P.sum(axis=1) if P.ndim > 1 else P
    if not band:
        return float(P[np.argmin(abs(lam - 1.064))])
    m = (lam > 1.02) & (lam < 1.11)
    i = np.argsort((1 / lam[m]))
    return float(np.trapezoid(P[m][i], (1 / lam[m])[i]))


amps = sorted(int(os.path.basename(f).split("_a")[-1][:-4])
              for f in glob.glob(os.path.join(RES, "monitor_2_a*.npz")))
amps = [a for a in amps if a != 10]          # 10 is the old pilot, different window
ein = np.array([power("monitor_1", a) for a in amps])
eout = np.array([power("monitor_2", a) for a in amps])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

ax = axes[0]
ax.loglog(ein, eout, "o-", color="tab:red", lw=1.8, label="measured")
lo, hi = ein.min() * 0.5, ein.max() * 2
ax.loglog([lo, hi], [lo, hi], color="0.6", ls="--", lw=1.2, label="out = in (transparent)")
k = eout[0] / ein[0]
ax.loglog([lo, hi], [lo * k, hi * k], color="tab:blue", ls=":", lw=1.4,
          label=f"small-signal gain {k:.2f} extrapolated")
for a, x, y in zip(amps, ein, eout):
    ax.annotate(str(a), (x, y), textcoords="offset points", xytext=(5, -9), fontsize=7)
ax.set_xlabel("input at 1064 nm"); ax.set_ylabel("output at 1064 nm")
ax.set_title("transfer — the curve peels off the\nsmall-signal line and joins out = in")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)

ax = axes[1]
ax.semilogx(ein, eout / ein, "o-", color="tab:red", lw=1.8)
ax.axhline(1.0, color="0.6", ls="--", lw=1.2)
ax.set_xlabel("input at 1064 nm"); ax.set_ylabel("gain  out / in")
ax.set_title("gain falls 1.35 -> 1.0:\nthe amplifier runs out of pump")
ax.grid(alpha=0.3, which="both")

ax = axes[2]
added = eout - ein
ax.semilogx(ein, added / max(added.max(), 1), "o-", color="tab:green", lw=1.8)
ax.axhline(0, color="0.6", ls="--", lw=1.2)
ax.set_xlabel("input at 1064 nm")
ax.set_ylabel("added energy (out - in), normalised")
ax.set_title("energy the crystal ADDS — plateaus, then\nis lost in the noise of a big difference")
ax.grid(alpha=0.3, which="both")

fig.suptitle("signal_modulation/design01 — Nd:YAG amplitude sweep, steady state "
             "(DFT 3000-6000, 1064 nm)", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)
for a, i_, o_ in zip(amps, ein, eout):
    print(f"  amp {a:4d}: in {i_:10.4g}  out {o_:10.4g}  gain {o_/i_:6.3f}  added {o_-i_:+10.3g}")
