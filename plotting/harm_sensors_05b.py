"""Harmonic spectra, LC 05b at drive 50: n2f far field vs monitor_2 output line.

Two tones (f1=3, f2=5) swept over 64 phase steps; DFT over the sweep gives power per
integer bin. The claim under test is that the sensor cannot change the spectrum —
n2f is a linear map of the same fields, so every bin should land at the same relative
power. Both readouts (field, |E|^2) on both sensors, all 3 components.
"""
import os, sys
import numpy as np

REPO = "/home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode"
LIPS = "/home/ziga/Lips_project/reservoir_runs/reservoir_types"
sys.path[:0] = [os.path.join(REPO, "characterization"), REPO]
import n4_harmonics_distortion as n4                        # noqa: E402

NPZ = f"{LIPS}/res_lc_gain/05b/datasets/harmonics_a50.npz"
COMPS = ["Ex", "Ey", "Ez"]
LAM = 0.55

d = np.load(NPZ, allow_pickle=True)
lam = 1.0 / np.asarray(d["m2_freqs"])
il = int(np.argmin(np.abs(lam - LAM)))
print(f"monitor_2 comb: {lam.min():.3f}-{lam.max():.3f} um, taking lam={lam[il]:.4f}")

S = {
    "n2f": np.asarray(d["outputs"]),                                   # (64, 600)
    "out": np.concatenate([np.asarray(d[f"m2_{c}"])[:, il, :] for c in COMPS], 1),
}
tones = np.asarray(d["tones"])
res = {}
for sensor, X in S.items():
    for readout in ("field", "intensity"):
        Y = np.abs(X) ** 2 if readout == "intensity" else X
        r = n4.harmonic_specter({"outputs": Y, "tones": tones}, max_order=6)
        res[(sensor, readout)] = r
        k = r["power_by_kind"]
        tot = sum(k.values()) + 1e-30
        print(f"{sensor:4s} {readout:9s} n_ch={Y.shape[1]:5d}  "
              f"dist_frac={r['distortion_frac']:.4f}  max_ord={r['max_order']}  "
              f"dc={k['dc']/tot:.3f} fund={k['fundamental']/tot:.3f} "
              f"harm={k['harmonic']/tot:.3f} imd={k['intermod']/tot:.3f}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"dc": "0.55", "fundamental": "C0", "harmonic": "C3", "intermod": "C2",
       "other": "C4"}
NU_MAX = 22
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), sharex=True)
for r_i, readout in enumerate(("field", "intensity")):
    for c_i, sensor in enumerate(("n2f", "out")):
        ax = axes[r_i, c_i]
        R = res[(sensor, readout)]
        nu, P = R["spec_nu"], R["spec_power"]
        Pn = P / (P.sum() + 1e-30)                     # relative -> sensor-independent
        m = nu <= NU_MAX
        for kind in COL:
            sel = m & np.array([k == kind for k in R["spec_kind"]])
            if sel.any():
                ax.bar(nu[sel], np.maximum(Pn[sel], 1e-12), 0.8, color=COL[kind],
                       label=kind if (r_i == 0 and c_i == 0) else None)
        for x, y, lab in zip(nu[m], Pn[m], np.array(R["spec_label"])[m]):
            if y > 3e-4 and lab:
                ax.annotate(lab, (x, y), fontsize=6.5, ha="center", rotation=90,
                            textcoords="offset points", xytext=(0, 3))
        ax.set_yscale("log"); ax.set_ylim(1e-7, 3)
        ax.grid(alpha=.3, axis="y")
        ax.set_title(f"{sensor} — {readout}   "
                     f"distortion={R['distortion_frac']:.3f}, "
                     f"max order {R['max_order']}", fontsize=10)
        if r_i == 1:
            ax.set_xlabel("sweep-DFT bin $\\nu$")
        if c_i == 0:
            ax.set_ylabel("relative power")
fig.legend(loc="upper center", fontsize=9, ncol=5, bbox_to_anchor=(0.5, 0.905),
           frameon=False)
fig.suptitle("Harmonic / intermod spectrum — LC 05b, drive 50, tones "
             f"f1={tones[0]}, f2={tones[1]}, 64 phase steps, all 3 components\n"
             f"n2f far field vs monitor_2 line at $\\lambda$={lam[il]:.3f} um "
             "(power normalised per panel)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.87])
p = f"{REPO}/data/reservoir_types/harm_sensors_05b.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("\nwrote", p)

# how closely do the two sensors agree, bin by bin?
for readout in ("field", "intensity"):
    a = res[("n2f", readout)]["spec_power"]; a = a / a.sum()
    b = res[("out", readout)]["spec_power"]; b = b / b.sum()
    big = (a > 1e-4) | (b > 1e-4)
    print(f"{readout:9s}: {big.sum()} significant bins, max relative diff "
          f"{np.max(np.abs(a[big] - b[big]) / np.maximum(a[big], b[big])):.3f}")
