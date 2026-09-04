#!/usr/bin/env python
"""design01d_strips: how a single 1 um strip spreads through the guide+crystal.

Top: full-cell steady-state |Ey| at 1.064 (2Ddft field_map). The colour scale is
clipped at the 99th percentile — the monitor includes the source plane, whose
singular column otherwise owns vmax and blacks out the physics (known SimpleSim
gotcha).
Bottom: |Ey| cross-sections at five x stations, normalised per-station, showing
the strip profile widening and multimode beating as it propagates.

    python plotting/plot_strip_spread.py <design_dir> <out.png> [suffix]
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
SUFFIX = sys.argv[3] if len(sys.argv) > 3 else ""

tag = f"_{SUFFIX}" if SUFFIX else ""
simdir = os.path.join(DESIGN, "simulation_gpumeep")
d = np.load(os.path.join(simdir, f"field_map{tag}.npz"))
Ey = np.abs(np.asarray(d["Ey"]))
if Ey.ndim == 3:                      # (n_freq, Nx, Ny) — single-freq monitor
    Ey = Ey[0]
cfg = json.load(open(os.path.join(DESIGN, "simulation_data.json")))
mon = cfg["field_map"]["position"]["size"]
sx, sy = float(mon[0]), float(mon[1])
x = np.linspace(-sx / 2, sx / 2, Ey.shape[0])
y = np.linspace(-sy / 2, sy / 2, Ey.shape[1])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 7.6),
                               gridspec_kw={"height_ratios": [1.4, 1]})

vmax = np.percentile(Ey, 99.0)
im = ax1.pcolormesh(x, y, Ey.T, cmap="inferno", vmin=0, vmax=vmax,
                    shading="auto")
fig.colorbar(im, ax=ax1, label="|Ey| at 1.064 um (clipped @ p99)")
# geometry: crystal block and the two guides, from the design JSON
cw, ch = cfg["crystal"]["sizes"]
for x0, x1 in ((-cw / 2 - 1, -cw / 2), (cw / 2, cw / 2 + 1)):     # guides
    ax1.add_patch(plt.Rectangle((x0, -ch / 2), x1 - x0, ch,
                                fill=False, ec="0.7", lw=0.8))
ax1.add_patch(plt.Rectangle((-cw / 2, -ch / 2), cw, ch,
                            fill=False, ec="w", lw=0.8, ls="--"))
s1 = cfg["source_1"]["position"]
# The OTHER strip: read it from source_2 rather than hardcoding, so the same
# script serves design01d (1 um at +-1.25) and design03b (5.48 um at +-6.85).
# Fall back to mirroring source_1 if the design has no second strip yet.
_s2 = cfg.get("source_2", {}).get("position")
s2y = float(_s2["y"]) if _s2 and "y" in _s2 else -float(s1["y"])
sw = float(s1["size"][1])
ax1.plot([-cw / 2 - 1] * 2,
         [s1["y"] - sw / 2, s1["y"] + sw / 2],
         color="lime", lw=3, label="strip source")
ax1.legend(loc="lower right", fontsize=8)
ax1.set_ylabel("y [um]")
ax1.set_title(f"single {sw:g} um strip at y = {s1['y']:+g} — steady-state |Ey|")

stations = [-cw / 2 + 0.3, -cw / 4, 0.0, cw / 4, cw / 2 - 0.3]
for xs in stations:
    i = int(np.argmin(np.abs(x - xs)))
    prof = Ey[i]
    ax2.plot(y, prof / (prof.max() + 1e-30), lw=1.4,
             label=f"x = {x[i]:+.1f} um")
ax2.axvspan(s1["y"] - sw / 2, s1["y"] + sw / 2,
            color="lime", alpha=0.15, label="source strip")
ax2.axvline(s2y, color="0.5", ls=":", lw=1, label="other strip's centre")
ax2.set_xlim(-sy / 2, sy / 2)
ax2.set_xlabel("y [um]")
ax2.set_ylabel("|Ey| / max (per station)")
ax2.grid(alpha=0.25)
ax2.legend(fontsize=8, ncol=2)
ax2.set_title("cross-sections along the crystal — spread and multimode beating")

fig.suptitle(f"{os.path.basename(DESIGN)} — strip spread", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", OUT)

# quantify: intensity fraction reaching the OTHER strip's footprint vs x
oy, w = s2y, sw
mask_other = (y > oy - w / 2) & (y < oy + w / 2)
mask_own = (y > s1["y"] - w / 2) & (y < s1["y"] + w / 2)
for xs in stations:
    i = int(np.argmin(np.abs(x - xs)))
    I = Ey[i] ** 2
    print(f"x = {x[i]:+6.1f}: I(other strip)/I(own strip) = "
          f"{I[mask_other].sum() / (I[mask_own].sum() + 1e-30):.3f}")
