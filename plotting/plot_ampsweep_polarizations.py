"""All three polarizations at BOTH spectral lines, across a coupled amplitude sweep.

  python plotting/plot_ampsweep_polarizations.py --path <design> --stem amp_sweep_coupled

Design 02 carries two channels that share no polarization and no wavelength:

  signal   Ey at 0.55 um   driven directly   -> ODD orders in the drive
  pump     Ez at 0.45 um   fixed source      -> EVEN orders, via population depletion

A single-wavelength readout cannot see both, so this reads monitor_2, which keeps all
61 wavelengths, and plots each line separately. Plotting only 0.55 (what the near2far
readout returns) makes Ez look like numerical noise at 1e-4 and hides the entire even
channel; plotting only 0.45 hides the signal.

The third panel is the local log-log slope, which is where the parity shows up:
the signal starts near 1 (linear amplification), compresses as it burns the inversion,
then returns toward 1 once the gain is exhausted and the medium merely transmits.
The pump's depletion starts near 2 -- intensity, not field -- and falls as the higher
even orders enter with opposite sign.
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
DESIGN = Path("/home/ziga/Orion/resevoir/data/reservoir_types/block_iso_gain/02")
COMPS = ["Ex", "Ey", "Ez"]
COLORS = {"Ex": "#2a72c4", "Ey": "#e08a1e", "Ez": "#9b4dca"}
MARKERS = {"Ex": "o", "Ey": "s", "Ez": "^"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(DESIGN))
    ap.add_argument("--stem", default="amp_sweep_coupled")
    ap.add_argument("--lams", type=float, nargs=2, default=[0.55, 0.45],
                    help="signal line then pump line")
    ap.add_argument("--baseline", default=None, help="pump-only monitor_2 npz (A=0)")
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "ampsweep_polarizations_02.png"))
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(a.path) / "datasets" / f"{a.stem}.npz.parts")
                              + "/part_*.npz")):
        with np.load(f, allow_pickle=True) as d:
            lam = 1.0 / np.asarray(d["m2_freqs"])
            idx = [int(np.argmin(abs(lam - L))) for L in a.lams]
            rows.append((float(np.asarray(d["inp"]).ravel()[0]),
                         np.array([[np.linalg.norm(np.asarray(d["m2_" + c])[j])
                                    for c in COMPS] for j in idx])))
    rows.sort(key=lambda r: r[0])
    A = np.array([r[0] for r in rows])
    V = np.stack([r[1] for r in rows])              # (n_lvl, 2 lines, 3 comps)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for k, (ax, L) in enumerate(zip(axes[:2], a.lams)):
        for i, c in enumerate(COMPS):
            ax.plot(A, V[:, k, i], MARKERS[c] + "-", ms=5, lw=1.7, color=COLORS[c],
                    label=f"|{c}|")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("signal drive amplitude")
        ax.set_ylabel("$\\|E\\|_2$ at monitor 2")
        ax.set_title(f"{L:g} $\\mu$m — {'signal line' if k == 0 else 'pump line'}",
                     fontsize=10)
        ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, frameon=False, loc="best")

    # Parity panel: local slope of each channel's own response.
    ax3 = axes[2]
    ey = V[:, 0, 1]
    slope_sig = np.gradient(np.log(ey), np.log(A))
    ax3.plot(A, slope_sig, "s-", ms=5, lw=1.8, color=COLORS["Ey"],
             label="signal $E_y$(0.55): $d\\log E/d\\log A$")
    ez = V[:, 1, 2]
    ez0 = ez[0]
    if a.baseline:
        with np.load(a.baseline) as d:
            lam0 = 1.0 / np.asarray(d["freqs"])
            ez0 = np.linalg.norm(np.asarray(d["Ez"])[int(np.argmin(abs(lam0 - a.lams[1])))])
    dep = 1.0 - ez / ez0
    ok = dep > 0
    ax3.plot(A[ok], np.gradient(np.log(dep[ok]), np.log(A[ok])), "^-", ms=5, lw=1.8,
             color=COLORS["Ez"], label="pump depletion: $d\\log(1-E_z/E_z^0)/d\\log A$")
    for y, lab in ((1.0, "slope 1 — linear / odd"), (2.0, "slope 2 — intensity / even")):
        ax3.axhline(y, color="0.45", ls=":", lw=1.2)
        ax3.annotate(lab, xy=(A[0] * 1.1, y + 0.05), fontsize=7.5, color="0.4")
    ax3.set_xscale("log")
    ax3.set_xlabel("signal drive amplitude")
    ax3.set_ylabel("local log-log slope")
    ax3.set_title("parity of the two channels", fontsize=10)
    ax3.grid(alpha=0.25, lw=0.6); ax3.set_axisbelow(True)
    ax3.legend(fontsize=8, frameon=False, loc="center right")

    fig.suptitle("block_iso_gain/02 coupled — all polarizations, both lines", fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    hdr = "".join(f"{c}@{L:g}".rjust(13) for L in a.lams for c in COMPS)
    print(f"{'drive':>6}{hdr}")
    for j, lv in enumerate(A):
        print(f"{lv:6.0f}" + "".join(f"{V[j,k,i]:13.5g}"
                                     for k in range(2) for i in range(3)))
    print(f"\nwrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
