"""Dambre IPC figure per drive amplitude, one per readout — no cross-drive comparison.

  python plotting/plot_ipc_by_drive.py
  python plotting/plot_ipc_by_drive.py --path data/reservoir_types/res_lc_gain/05b
  python plotting/plot_ipc_by_drive.py --drives 50 100 --max-degree 5

Reads <path>/datasets/ipc_4src_a<drive>.npz and writes the standard single-panel
IPC-by-degree figure (plot_function_n6_dambre_ipc) for the field and intensity
readouts of each drive, into <path>/figures.

This is plot_main.py's n6 step in isolation: plot_main has no n6-only flag and would
run n1–n7 for the whole suffix. State reduction mirrors Validator._reduce_state (PCA
to min(rank, M/10) leading channels), and max_degree defaults to 5 — the setting the
existing ipc_50_vs_100.png was made with, so the totals line up with it (05 field
a50 = 25.60). The Validator's own dambre() hardcodes max_degree=3; pass --max-degree 3
to reproduce the n6 figures plot_main writes.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from characterization import n6_dambre as n6
from plot_function_n6_dambre_ipc import plot_n6_dambre_ipc

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PATH = REPO / "data" / "reservoir_types" / "res_iso_gain" / "05"


def _reduce_state(X):
    """Mirror of characterization.class_reservoir_validator.Validator._reduce_state."""
    M = X.shape[0]
    Xf = X.reshape(M, -1)
    F = Xf.shape[1]
    k = int(min(F, max(4, M // 10)))
    if F <= k:
        return Xf
    Xc = Xf - Xf.mean(0, keepdims=True)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rank = int((S / (S[0] + 1e-300) > 1e-8).sum())
    k = min(k, max(4, rank))
    print(f"[ipc]   PCA-reduced state {F}→{k} channels (M={M} probes, numerical rank {rank})",
          flush=True)
    return Xc @ Vt[:k].conj().T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(DEFAULT_PATH),
                    help="reservoir dir holding datasets/ and figures/ (default: res_iso_gain/05)")
    ap.add_argument("--fig-dir", default=None, help="output dir (default: <path>/figures)")
    ap.add_argument("--stem", default="ipc_4src_a",
                    help="dataset stem; the drive is appended (default: ipc_4src_a)")
    ap.add_argument("--drives", type=int, nargs="+", default=[50, 100],
                    help="drive amplitudes to plot (default: 50 100)")
    ap.add_argument("--max-degree", type=int, default=5,
                    help="highest total polynomial degree in the IPC target family")
    a = ap.parse_args()

    path = Path(a.path)
    figdir = Path(a.fig_dir) if a.fig_dir else path / "figures"
    os.makedirs(figdir, exist_ok=True)

    for drive in a.drives:
        ds = path / "datasets" / f"{a.stem}{drive}.npz"
        if not ds.exists():
            print(f"[ipc] MISSING {ds} — skipping", flush=True)
            continue
        d = np.load(ds, allow_pickle=True)
        U, X = d["inputs"], np.asarray(d["outputs"])
        # field and intensity see different halves of the spectrum: |E|² is even in the
        # drive and annihilates every odd-degree target, the field is odd and annihilates
        # the even ones — so both are plotted, same split the validator uses.
        for mode, Xv in (("field", X), ("intensity", np.abs(X) ** 2)):
            print(f"[ipc] {ds.name} · {mode}", flush=True)
            Xr = _reduce_state(np.asarray(Xv))
            res = n6.dambre_ipc({"inputs": U, "outputs": Xr},
                                max_degree=a.max_degree, max_features=Xr.shape[1])
            # plot_n6_dambre_ipc already prefixes "n6_dambre_ipc", and the repo's
            # variant suffix is the part of the stem AFTER "ipc" (ipc_4src_a50.npz ->
            # n6_dambre_ipc_4src_a50_...), so strip it or the name reads "ipc_ipc".
            tag = a.stem[3:] if a.stem.startswith("ipc") else a.stem
            out = plot_n6_dambre_ipc(res, figdir, suffix=f"{tag}{drive}_n6_{mode}")
            print(f"[ipc]   total={res['ipc_total']:.2f} "
                  f"nonlinear_frac={res['nonlinear_fraction']:.3f} → {out}", flush=True)


if __name__ == "__main__":
    main()
