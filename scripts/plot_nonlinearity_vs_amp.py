"""Nonlinearity vs drive amplitude, 04 (LC) vs 03b (isotropic control).

Left:  order-3 power relative to the fundamental — the honest measure of how
       nonlinear the FIELD map is (an intensity readout would fake order 2).
Right: even-order power relative to fundamental. Should sit at the floor: any
       even-order content would mean a broken symmetry (or an intensity leak).

Uses whatever harmonics*.npz have assembled, so it can be run mid-campaign.
"""
import os, sys, glob, re
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, "characterization")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import n4_harmonics_distortion as n4

DES = [("04_adding_LC", "04 — LC (anisotropic)", "C0", "o"),
       ("03b_isotropic_ds", "03b — isotropic control", "C3", "s")]
FLOOR = 1e-12


def series(des):
    out = []
    for f in sorted(glob.glob(os.path.join(R, "data/lasing_testing", des,
                                           "datasets", "harmonics*.npz"))):
        if "analysis" in os.path.basename(f):
            continue
        d = dict(np.load(f))
        if "outputs" not in d:
            continue
        comps = [str(c) for c in np.asarray(d["components"]).reshape(-1)]
        Y = np.asarray(d["outputs"])
        n = Y.shape[1] // len(comps)
        k = comps.index("Ey")
        d["outputs"] = Y[:, k * n:(k + 1) * n]
        res = n4.harmonic_specter(d)
        bo = res["power_by_order"]
        fund = bo.get(1, 0.0) or 1e-300
        odd = sum(p for o, p in bo.items() if o % 2 == 1 and o > 1)
        even = sum(p for o, p in bo.items() if o % 2 == 0 and o > 0)
        out.append((float(np.asarray(d["amps"]).reshape(-1)[0]),
                    odd / fund, max(even / fund, FLOOR), res["thd"]))
    return sorted(out)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
for des, label, col, mk in DES:
    s = series(des)
    if not s:
        continue
    a = np.array([r[0] for r in s])
    ax1.plot(a, [r[1] for r in s], mk + "-", color=col, label=label)
    ax2.plot(a, [r[2] for r in s], mk + "-", color=col, label=label)
    for x, y, _, _ in s:
        ax1.annotate(f"{y*100:.2g}%", (x, y), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8, color=col)

ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("drive amplitude"); ax1.set_ylabel("odd-order power / fundamental")
ax1.set_title("nonlinearity grows with drive → gain saturation")
ax1.legend(fontsize=8); ax1.grid(alpha=0.25, which="both")

ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xlabel("drive amplitude"); ax2.set_ylabel("even-order power / fundamental")
ax2.set_title("even orders stay at zero → centrosymmetric medium")
ax2.set_ylim(FLOOR / 10, 1.0)
ax2.legend(fontsize=8); ax2.grid(alpha=0.25, which="both")

fig.suptitle("Harmonic nonlinearity vs drive — LC vs isotropic control (Ey, "
             "complex field readout)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(R, "data/lasing_testing/04_adding_LC/figures",
                   "nonlinearity_vs_amp_04_vs_03b.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
for des, label, _, _ in DES:
    for amp, odd, even, thd in series(des):
        print(f"{des:20s} amp {amp:5g}  odd/fund {odd:.4g}  "
              f"even/fund {even:.1e}  THD {thd:.4g}")
