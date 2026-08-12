"""IPC degree spectrum for a WAVELENGTH-SUMMED intensity readout.

  python plotting/plot_ipc_summed_intensity.py --sets a100=<npz> a20=<npz>

The readout here is what a spectrally blind detector sees: |E|^2 summed over the whole
comb, one number per far-field point per component (3 x 200 = 600 features instead of
3 x 61 x 200 = 36,600). By Parseval that sum is the time-integrated in-band energy, and
since the comb is uniform in FREQUENCY no per-bin weighting is needed.

Summing does not change the parity of the readout: every |E(lam)|^2 is individually even
in the drive amplitude, so the sum is even too and the odd degrees stay exactly zero
under amplitude encoding. What summing costs is spectral structure -- and how much it
costs is the point of comparing it against the per-wavelength readout.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(REPO))
from characterization import n6_dambre as n6          # noqa: E402
from plot_ipc_by_drive import _reduce_state           # noqa: E402

COLORS = ["#2a72c4", "#e08a1e", "#9b4dca", "#0f7d6b"]


def summed_intensity(X, n_lam, n_pts):
    return np.concatenate(
        [(np.abs(X[:, c * n_lam * n_pts:(c + 1) * n_lam * n_pts]
                 .reshape(len(X), n_lam, n_pts)) ** 2).sum(axis=1) for c in range(3)],
        axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", required=True, help="label=path pairs")
    ap.add_argument("--n-lam", type=int, default=61)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--max-degree", type=int, default=5)
    ap.add_argument("--title", default="wavelength-summed intensity readout")
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "ipc_summed_intensity.png"))
    a = ap.parse_args()

    res = []
    for spec in a.sets:
        label, path = spec.split("=", 1)
        d = np.load(path, allow_pickle=True)
        F = summed_intensity(np.asarray(d["outputs"]), a.n_lam, a.n_points)
        Xr = _reduce_state(F)
        r = n6.dambre_ipc({"inputs": np.asarray(d["inputs"]), "outputs": Xr},
                          max_degree=a.max_degree, max_features=Xr.shape[1])
        res.append((label, r, F.shape[1], Xr.shape[1]))

    degrees = list(range(1, a.max_degree + 1))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    width = 0.8 / len(res)
    for i, (label, r, nf, nc) in enumerate(res):
        vals = [r["ipc_by_degree"].get(d, 0.0) for d in degrees]
        xs = np.arange(len(degrees)) + (i - (len(res) - 1) / 2) * width
        ax.bar(xs, vals, width, color=COLORS[i % len(COLORS)],
               label=f"{label}   total {r['ipc_total']:.1f}")
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.2f}" if v < 1 else f"{v:.1f}", (x, v), ha="center",
                        va="bottom", fontsize=7.5, color="0.25")
    ax.set_xticks(np.arange(len(degrees)), [str(d) for d in degrees])
    ax.set_xlabel("target polynomial degree")
    ax.set_ylabel("IPC ($\\Sigma$ capacity)")
    ax.set_title(a.title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.6, axis="y"); ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False)

    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for label, r, nf, nc in res:
        print(f"{label:26s} {nf:5d} feats -> {nc:3d} ch   total {r['ipc_total']:6.2f}   "
              + "  ".join(f"d{k} {v:6.2f}" for k, v in sorted(r["ipc_by_degree"].items())))
    print(f"\nwrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
