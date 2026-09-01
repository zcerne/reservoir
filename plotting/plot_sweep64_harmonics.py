#!/usr/bin/env python
"""64-pair amplitude sweep — nonlinearity by the established harmonics method.

Reads <design>/datasets/harmonics.npz (64 static CW runs, source_1 driven with
cos(3t_j) and source_2 with cos(5t_j)), selects ONE optical wavelength bin from
monitor_2's comb, and DFTs the 64 outputs over the sweep index. A linear
amplitude map puts power only in bins 3 and 5; harmonics (6, 9, 10) and
intermodulation products (2, 8, 1, 7) are the nonlinearity.

    python plotting/plot_sweep64_harmonics.py <design_dir> <out.png> [lam]

`lam` defaults to 1.064 (the signal line). The full 61-bin comb is stored per
run, so the readout wavelength is a post-processing choice, not a rerun.

Note the classification comes from characterization/n4_harmonics_distortion, the
same code the reservoir designs were scored with — that shared path is the whole
point of running this method here.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, "characterization")]
import n4_harmonics_distortion as n4          # noqa: E402

DESIGN = sys.argv[1]
OUT = sys.argv[2]
LAM = float(sys.argv[3]) if len(sys.argv) > 3 else 1.064

cfg = json.load(open(os.path.join(DESIGN, "simulation_data.json")))
mon = cfg["monitor_2"]
lam_lo, lam_hi = mon["lam_range"]
n_lam = int(mon["n_lam"])
# SimpleSim builds the comb linearly in FREQUENCY between 1/lam_hi and 1/lam_lo
# (sensor.py add_flux), not linearly in wavelength — picking the bin on a
# wavelength grid lands on the wrong line.
freqs = np.linspace(1.0 / lam_hi, 1.0 / lam_lo, n_lam)
k = int(np.argmin(np.abs(freqs - 1.0 / LAM)))
print(f"[sweep64] readout bin {k} of {n_lam}: {1.0/freqs[k]:.6f} um "
      f"(asked {LAM}, comb spacing {abs(1/freqs[1]-1/freqs[0])*1e3:.2f} nm)")

d = dict(np.load(os.path.join(DESIGN, "datasets", "harmonics.npz")))
comps = [str(c) for c in np.asarray(d["components"]).reshape(-1)]
Y = np.asarray(d["outputs"])                       # (64, n_comp * n_lam * n_pts)
N_t = Y.shape[0]
per_comp = Y.shape[1] // len(comps)
ci = comps.index("Ey")
Yc = Y[:, ci * per_comp:(ci + 1) * per_comp].reshape(N_t, n_lam, -1)
d["outputs"] = Yc[:, k, :]                         # (64, n_points) at one lambda

res = n4.harmonic_specter(d)
tones = [int(t) for t in np.asarray(d["tones"]).reshape(-1)]
by_kind, by_order = res["power_by_kind"], res["power_by_order"]
fund = by_kind["fundamental"] or 1e-300

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

nu = np.asarray(res["spec_nu"]); P = np.asarray(res["spec_power"])
kind = list(res["spec_kind"]); lab = list(res["spec_label"])
col = {"dc": "0.6", "fundamental": "tab:green", "harmonic": "tab:red",
       "intermod": "tab:orange", "other": "tab:blue"}
floor = max(P.max() * 1e-14, 1e-300)
ax1.bar(nu, np.maximum(P, floor), width=0.8,
        color=[col.get(kk, "tab:blue") for kk in kind])
ax1.set_yscale("log")
ax1.set_xlim(-0.5, min(nu.max(), 4 * max(tones)) + 0.5)
ax1.set_xlabel("sweep-DFT bin")
ax1.set_ylabel("power at 1.064 um")
ax1.set_title(f"sweep spectrum — tones {tones}")
ax1.grid(alpha=0.25, axis="y")
for x, p, kk, ll in zip(nu, P, kind, lab):
    if p > P.max() * 1e-9 and kk in ("fundamental", "harmonic", "intermod"):
        ax1.text(x, p, ll or str(x), rotation=90, fontsize=7, ha="center",
                 va="bottom", color=col[kk])

kinds = ["fundamental", "harmonic", "intermod", "dc", "other"]
ax2.bar(range(len(kinds)), [max(by_kind[kk] / fund, 1e-16) for kk in kinds],
        color=[col[kk] for kk in kinds])
ax2.set_yscale("log")
ax2.set_xticks(range(len(kinds)))
ax2.set_xticklabels(kinds, rotation=20)
ax2.set_ylabel("power / fundamental")
ax2.set_title("power by kind")
ax2.grid(alpha=0.25, axis="y")
ax2.text(0.02, 0.05,
         f"THD {res['thd']:.3e}\nIMD {res['imd']:.3e}\n"
         f"distortion_frac {res['distortion_frac']:.3e}\nmax order {res['max_order']}",
         transform=ax2.transAxes, fontsize=9, va="bottom")

fig.suptitle(f"{os.path.basename(DESIGN)} — 64-pair amplitude sweep, "
             f"readout {1.0/freqs[k]:.4f} um", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)

print(f"linear={res['linear']}  thd={res['thd']:.4e}  imd={res['imd']:.4e}  "
      f"distortion_frac={res['distortion_frac']:.4e}  max_order={res['max_order']}")
for kk in kinds:
    print(f"  {kk:>12s}  {by_kind[kk]:.4e}  ({by_kind[kk]/fund:.3e} x fundamental)")
for o in sorted(by_order):
    print(f"  order {o}: {by_order[o]:.4e}  ({by_order[o]/fund:.3e})")
