#!/usr/bin/env python
"""64-pair amplitude sweep — nonlinearity by the established harmonics method.

Reads <design>/datasets/harmonics.npz (64 static CW runs; channel 0 = Ey driven
cos(3t_j), channel 1 = Ez driven cos(5t_j)), selects ONE optical bin from
monitor_2's comb, and DFTs the 64 outputs over the sweep index. Classification
comes from characterization/n4_harmonics_distortion — the same code the
reservoir designs were scored with.

    python plotting/plot_sweep64_harmonics.py <design_dir> <out.png> [lam]

Analysed PER OUTPUT POLARISATION, because that is where the sharp result lives.
Ey and Ez do not mix linearly in an isotropic 2D medium, so the only thing that
can put channel 0's tone into the Ez output is the shared inversion: driving one
polarisation burns gain the other one sees. Concretely, in the Ez output:

    bin 5              its own drive — present even if the medium is linear
    bins 1, 7, 11 ...  intermods whose decomposition uses tone 3
                       -> CROSS-CHANNEL, essentially zero linear background

Two built-in controls, both reported:
  * ODD ONLY. Saturation depends on intensity, so the gain is modulated at EVEN
    multiples of the drives (6, 10, ...) and multiplying back by the drive gives
    odd output bins. Significant EVEN-order power means either a genuine
    even-order process or -- far more likely -- an intensity readout sneaking in,
    which manufactures order-2 products from a perfectly linear medium.
  * DC. Same reasoning: a field readout of a symmetric map should leave bin 0 empty.
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
print(f"[sweep64] readout bin {k} of {n_lam}: {1.0 / freqs[k]:.6f} um (asked {LAM})")

d0 = dict(np.load(os.path.join(DESIGN, "datasets", "harmonics.npz")))
comps = [str(c) for c in np.asarray(d0["components"]).reshape(-1)]
tones = [int(t) for t in np.asarray(d0["tones"]).reshape(-1)]
Y = np.asarray(d0["outputs"])
N_t = Y.shape[0]
per_comp = Y.shape[1] // len(comps)

# which source drives which polarisation, so "cross-channel" is read off the
# design rather than assumed
drive_of = {}
for ch, key in enumerate(cfg.get("sweep_sources", [])):
    drive_of[str(cfg[key].get("component", "Ey"))] = ch

col = {"dc": "0.6", "fundamental": "tab:green", "harmonic": "tab:red",
       "intermod": "tab:orange", "other": "tab:blue"}
fig, axes = plt.subplots(len(comps), 1, figsize=(11, 3.6 * len(comps)),
                         squeeze=False)
rows = []
for r, c in enumerate(comps):
    d = dict(d0)
    d["outputs"] = Y[:, r * per_comp:(r + 1) * per_comp].reshape(N_t, n_lam, -1)[:, k, :]
    res = n4.harmonic_specter(d)
    nu = np.asarray(res["spec_nu"])
    P = np.asarray(res["spec_power"])
    kind = list(res["spec_kind"])
    order = np.asarray(res["spec_order"])
    label = list(res["spec_label"])
    total = float(P.sum()) + 1e-300

    own = drive_of.get(c)                       # channel this polarisation is driven on
    other = None if own is None else 1 - own
    cross = 0.0
    if other is not None:
        for x, p, ll in zip(nu, P, label):
            # a bin counts as cross-channel if its lowest-order decomposition
            # uses the OTHER channel's tone at all
            if ll and f"f{other + 1}" in ll and x != tones[other]:
                cross += float(p)
    even = float(P[(order >= 2) & (order % 2 == 0)].sum())
    dc = float(P[nu == 0].sum())
    fund = res["power_by_kind"]["fundamental"] or 1e-300
    rows.append((c, own, fund, res["power_by_kind"]["harmonic"],
                 res["power_by_kind"]["intermod"], cross, even, dc, total))

    ax = axes[r][0]
    floor = max(P.max() * 1e-14, 1e-300)
    ax.bar(nu, np.maximum(P, floor), width=0.8,
           color=[col.get(kk, "tab:blue") for kk in kind])
    ax.set_yscale("log")
    ax.set_xlim(-0.5, min(nu.max(), 4 * max(tones)) + 0.5)
    ax.set_ylabel(f"{c} out\npower at {1.0 / freqs[k]:.4f} um")
    ax.grid(alpha=0.25, axis="y")
    drv = "" if own is None else f" (driven on tone {tones[own]})"
    ax.set_title(f"{c} output{drv} — cross-channel {cross / fund:.2e} x fundamental")
    for x, p, kk, ll in zip(nu, P, kind, label):
        if p > P.max() * 1e-9 and kk in ("fundamental", "harmonic", "intermod"):
            ax.text(x, p, ll or str(x), rotation=90, fontsize=7, ha="center",
                    va="bottom", color=col[kk])
axes[-1][0].set_xlabel("sweep-DFT bin")
fig.suptitle(f"{os.path.basename(DESIGN)} — 64-pair sweep, tones {tones}, "
             f"readout {1.0 / freqs[k]:.4f} um", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)

print(f"{'out':>4} {'tone':>5} {'fundamental':>12} {'harmonic':>11} {'intermod':>11} "
      f"{'CROSS':>11} {'even(>=2)':>11} {'dc':>11}")
for c, own, fu, ha, im, cr, ev, dc, tot in rows:
    t = "-" if own is None else str(tones[own])
    print(f"{c:>4} {t:>5} {fu:>12.4e} {ha:>11.4e} {im:>11.4e} {cr:>11.4e} "
          f"{ev:>11.4e} {dc:>11.4e}")
    print(f"{'':>10} relative to own fundamental: harmonic {ha/fu:.3e}  "
          f"intermod {im/fu:.3e}  CROSS {cr/fu:.3e}  even {ev/fu:.3e}  dc {dc/fu:.3e}")
