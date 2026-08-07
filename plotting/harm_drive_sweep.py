"""Harmonic distortion vs drive level — every 2-tone harmonics set on disk.

block 01 gives the only 3-point drive series (a10/a50/a100, Jul-31 old geometry,
single-frequency n2f -> valid 600-feature outputs). iso 05 has a1/a10; LC 05b has
a50 only (single point, no trend). All: tones f1=3/f2=5, 64 phase steps, n2f.
"""
import os, sys
import numpy as np

REPO = "/home/ziga/Nextcloud/Doktorski/Projects/Reservoir/gitcode"
LIPS = "/home/ziga/Lips_project/reservoir_runs/reservoir_types"
sys.path[:0] = [os.path.join(REPO, "characterization"), REPO]
import n4_harmonics_distortion as n4                            # noqa: E402

SETS = [
    ("block 01", 10,  f"{LIPS}/block_iso_gain/01/datasets/harmonics_a10.npz"),
    ("block 01", 50,  f"{LIPS}/block_iso_gain/01/datasets/harmonics_a50.npz"),
    ("block 01", 100, f"{LIPS}/block_iso_gain/01/datasets/harmonics_a100.npz"),
    ("iso 05",   1,   f"{REPO}/data/reservoir_types/res_iso_gain/05/datasets/harmonics.npz"),
    ("iso 05",   10,  f"{REPO}/data/reservoir_types/res_iso_gain/05/datasets/harmonics_a10.npz"),
    ("LC 05b",   50,  f"{LIPS}/res_lc_gain/05b/datasets/harmonics_a50.npz"),
]

rows = {}
for design, amp, path in SETS:
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["outputs"])
    tones = np.asarray(d["tones"])
    for readout in ("field", "intensity"):
        Y = np.abs(X) ** 2 if readout == "intensity" else X
        r = n4.harmonic_specter({"outputs": Y, "tones": tones}, max_order=6)
        rows[(design, amp, readout)] = r
        k = r["power_by_kind"]; tot = sum(k.values()) + 1e-30
        print(f"{design:9s} a{amp:<4d} {readout:9s} dist={r['distortion_frac']:.4f} "
              f"maxord={r['max_order']}  dc={k['dc']/tot:.3f} fund={k['fundamental']/tot:.3f} "
              f"harm={k['harmonic']/tot:.3f} imd={k['intermod']/tot:.3f}")

np.savez(f"{REPO}/data/reservoir_types/stats_harm_drive_sweep.npz",
         rows=np.array(rows, dtype=object))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DESIGNS = {"block 01": ("C0", "o"), "iso 05": ("C1", "s"), "LC 05b": ("C3", "^")}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

ax = axes[0]
for design, (col, mk) in DESIGNS.items():
    amps = sorted(a for d, a, r in rows if d == design and r == "field")
    v = [rows[(design, a, "field")]["distortion_frac"] for a in amps]
    ax.plot(amps, v, "-" if len(amps) > 1 else "", marker=mk, color=col, label=design)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("drive amplitude"); ax.set_ylabel("distortion fraction (AC)")
ax.set_title("FIELD readout — nonlinear fraction of AC power", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=9)

ax = axes[1]
for design, (col, mk) in DESIGNS.items():
    amps = sorted(a for d, a, r in rows if d == design and r == "field")
    for kind, ls in (("harmonic", "-"), ("intermod", "--")):
        v = [rows[(design, a, "field")]["power_by_kind"][kind] /
             (sum(rows[(design, a, "field")]["power_by_kind"].values()) + 1e-30)
             for a in amps]
        ax.plot(amps, v, ls, marker=mk, color=col,
                label=f"{design} {kind}" if kind == "harmonic" else None)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("drive amplitude"); ax.set_ylabel("relative power")
ax.set_title("FIELD — harmonic (solid) vs intermod (dashed)", fontsize=10)
ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)

# spectra of the block-01 series: the actual bins at the three drives
ax = axes[2]
W = 0.26
NU = 16
for j, amp in enumerate((10, 50, 100)):
    r = rows[("block 01", amp, "field")]
    nu, P = np.asarray(r["spec_nu"]), np.asarray(r["spec_power"])
    Pn = P / (P.sum() + 1e-30)
    m = (nu >= 1) & (nu <= NU)
    ax.bar(nu[m] + (j - 1) * W, np.maximum(Pn[m], 1e-12), W,
           color=["C0", "C1", "C3"][j], label=f"a{amp}")
ax.set_yscale("log"); ax.set_ylim(1e-7, 1.5)
ax.set_xlabel("sweep-DFT bin $\\nu$"); ax.set_ylabel("relative power")
ax.set_title("block 01 FIELD spectrum vs drive (f1=3, f2=5)", fontsize=10)
ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=9)

fig.suptitle("Harmonic distortion vs drive level — n2f sensor, tones 3 & 5, 64 phase steps",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
p = f"{REPO}/data/reservoir_types/harm_drive_sweep.png"
fig.savefig(p, dpi=140, bbox_inches="tight")
print("\nwrote", p)
