"""Saturation curves for 04 (LC) vs 03b (isotropic) from their amp_sweeps.

Left: normalized response ‖out‖/level vs drive (the compression curve).
Right: best-linear-map drift vs drive (Method C) — 0 = linear at that drive.
Both on Ey only. Shows that the two designs saturate almost identically, i.e.
the nonlinearity comes from the dye gain rather than the liquid crystal.
"""
import os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(R, "characterization")]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import n3_amplitude_dependant as n3

DES = [("04_adding_LC", "04 — LC (anisotropic)", "C0", "o"),
       ("03b_isotropic_ds", "03b — isotropic control", "C3", "s")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))
summary = []
for des, label, col, mk in DES:
    p = os.path.join(R, "data/lasing_testing", des, "datasets", "amp_sweep.npz")
    if not os.path.exists(p):
        print(f"skip {des}: no amp_sweep.npz")
        continue
    d = dict(np.load(p))
    comps = [str(c) for c in np.asarray(d["components"]).reshape(-1)]
    Y = np.asarray(d["outputs"])
    n_pts = Y.shape[1] // len(comps)
    k = comps.index("Ey")
    d["outputs"] = Y[:, k * n_pts:(k + 1) * n_pts]
    lv = np.asarray(d["levels"], float).reshape(-1)
    lid = np.asarray(d["level_id"]).reshape(-1)
    res = n3.amplitude_dependance(d)

    g = []
    for i, L in enumerate(lv):
        m = lid == i
        g.append(float(np.linalg.norm(np.abs(d["outputs"][m]))
                       / np.sqrt(m.sum()) / L) if m.any() else np.nan)
    g = np.array(g)
    ax1.plot(lv, g / g[0], mk + "-", color=col, label=f"{label}  (‖o‖/L={g[0]:.2f} at L=1)")
    ax2.plot(lv, res["drift"], mk + "-", color=col, label=label)
    summary.append((des, g[0], g / g[0]))

ax1.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax1.axvline(10, color="0.4", lw=0.8, ls="--")
ax1.annotate("design operating\npoint (amp 10)", (10, 0.55), fontsize=8,
             ha="center", color="0.35")
ax1.set_xscale("log")
ax1.set_xlabel("drive amplitude")
ax1.set_ylabel("normalized response  ‖out‖/level  (rel. to level 1)")
ax1.set_title("gain compression — saturable, not super-linear")
ax1.set_ylim(0, 1.1)
ax1.legend(fontsize=8)

ax2.axhline(0.0, color="0.6", lw=0.8, ls=":")
ax2.set_xscale("log")
ax2.set_xlabel("drive amplitude")
ax2.set_ylabel("‖ΔG‖/‖G‖ vs level 1")
ax2.set_title("best-linear-map drift (Method C)")
ax2.legend(fontsize=8)

fig.suptitle("Saturation: LC vs isotropic control (identical designs apart from "
             "reservoir.isotropic) — Ey readout", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = os.path.join(R, "data/lasing_testing/04_adding_LC/figures",
                   "saturation_04_vs_03b.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
for des, g0, rel in summary:
    print(f"{des:20s} L=1 gain {g0:7.3f} | rel " +
          " ".join(f"{v:.3f}" for v in rel))
