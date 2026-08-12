"""Output intensity spectrum of an IPC probe set, per polarization, drive by drive.

  python plotting/plot_ipc_spectra.py --sets a100=<npz> a20=<npz>

Each npz is {inputs, outputs} with `outputs` the near2far readout packed
component-major as [Ex | Ey | Ez], each block being n_lam wavelengths x n_points
far-field samples. Intensity per wavelength is the sum of |E|^2 over the far-field
points, averaged across probes; the band across probes is the 10-90 percentile, which
is wide by construction because the drive itself varies from probe to probe.

MIND THE WAVELENGTH ORDER. The readout is stored in ascending FREQUENCY, so the
wavelength axis inside each component block runs 0.6 down to 0.4. Plotting it as
stored silently mirrors the spectrum and swaps the two lines; this sorts first.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
COMPS = ["Ex", "Ey", "Ez"]
COLORS = ["#2a72c4", "#e08a1e", "#9b4dca", "#0f7d6b"]


def load(path, n_lam, n_pts):
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["outputs"])
    out = {}
    for i, c in enumerate(COMPS):
        blk = X[:, i * n_lam * n_pts:(i + 1) * n_lam * n_pts]
        out[c] = (np.abs(blk.reshape(len(X), n_lam, n_pts)) ** 2).sum(axis=2)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--n-lam", type=int, default=61)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--lam-range", type=float, nargs=2, default=[0.4, 0.6])
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "ipc_spectra_a100_vs_a20.png"))
    a = ap.parse_args()

    lam = 1.0 / np.linspace(1 / a.lam_range[1], 1 / a.lam_range[0], a.n_lam)
    order = np.argsort(lam)
    lam_s = lam[order]

    sets = []
    for spec in a.sets:
        label, path = spec.split("=", 1)
        sets.append((label, load(path, a.n_lam, a.n_points)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    for ax, c in zip(axes, COMPS):
        for j, (label, data) in enumerate(sets):
            I = data[c][:, order]
            med = np.median(I, axis=0)
            lo, hi = np.percentile(I, [10, 90], axis=0)
            ax.fill_between(lam_s, lo, hi, color=COLORS[j % len(COLORS)], alpha=0.18, lw=0)
            ax.plot(lam_s, med, lw=1.9, color=COLORS[j % len(COLORS)], label=label)
        for line, name in ((0.45, "pump"), (0.55, "signal")):
            ax.axvline(line, color="0.5", ls=":", lw=1.1)
            ax.annotate(name, xy=(line, 0.965), xycoords=("data", "axes fraction"),
                        fontsize=7.5, color="0.4", ha="center")
        ax.set_yscale("log")
        ax.set_xlabel("wavelength ($\\mu$m)")
        ax.set_title(f"$|{c[0]}_{{{c[1]}}}|^2$", fontsize=11)
        ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, frameon=False)
    axes[0].set_ylabel("intensity at the far screen (median over 1000 probes)")

    fig.suptitle("block_iso_gain/02 — output spectrum per polarization, "
                 "shaded band = 10-90% across probes", fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for label, data in sets:
        for c in COMPS:
            I = np.median(data[c], axis=0)
            print(f"{label:16s} {c}  peak at {lam[np.argmax(I)]:.4f} um   "
                  f"I(0.45) {I[np.argmin(abs(lam-0.45))]:.4g}   "
                  f"I(0.55) {I[np.argmin(abs(lam-0.55))]:.4g}")
    print(f"\nwrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
