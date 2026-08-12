"""IPC degree spectrum under AMPLITUDE vs INTENSITY input encoding.

  python plotting/plot_encoding_parity.py --amp <ipc npz> --int <readouts npz>

An intensity readout is even in the drive amplitude, so under amplitude encoding it
can only reach EVEN polynomial degrees — the odd capacities come out at exactly 0.00,
not merely small. That is a selection rule from the u -> -u symmetry, not a property
of the medium.

Intensity encoding removes the symmetry: the drive is non-negative, and even-in-
amplitude becomes "any polynomial in the encoded variable" since A^(2k) = v^k. The
prediction is therefore specific and falsifiable — degree 1 on the intensity readout
must go from 0.00 to 4.00, the ceiling for 4 inputs (4 targets at capacity 1.0).

The intensity/intensity pair is also the only cleanly interpretable one: the readout's
|.|^2 exactly undoes the encoder's sqrt, so degree 1 IS the linear response and every
higher degree is the medium. A field readout under intensity encoding sees sqrt(v) and
mixes encoder nonlinearity into its spectrum.
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

COL = {"field": "#2a72c4", "intensity": "#e08a1e"}


def spectra(U, X, max_degree):
    out = {}
    for mode, Xv in (("field", X), ("intensity", np.abs(X) ** 2)):
        Xr = _reduce_state(np.asarray(Xv))
        out[mode] = n6.dambre_ipc({"inputs": U, "outputs": Xr},
                                  max_degree=max_degree, max_features=Xr.shape[1])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amp", required=True, help="amplitude-encoded ipc npz")
    ap.add_argument("--int", dest="inten", required=True, help="intensity-encoded npz")
    ap.add_argument("--amp-label", default="amplitude encoding")
    ap.add_argument("--int-label", default="intensity encoding")
    ap.add_argument("--max-degree", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "encoding_parity.png"))
    a = ap.parse_args()

    panels = []
    for path, label in ((a.amp, a.amp_label), (a.inten, a.int_label)):
        with np.load(path, allow_pickle=True) as d:
            panels.append((label, spectra(np.asarray(d["inputs"]),
                                          np.asarray(d["outputs"]), a.max_degree)))

    degrees = list(range(1, a.max_degree + 1))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    width = 0.38
    for ax, (label, res) in zip(axes, panels):
        for i, mode in enumerate(("field", "intensity")):
            vals = [res[mode]["ipc_by_degree"].get(dg, 0.0) for dg in degrees]
            xs = np.arange(len(degrees)) + (i - 0.5) * width
            ax.bar(xs, vals, width, color=COL[mode],
                   label=f"{mode} readout   total {res[mode]['ipc_total']:.1f}")
            for x, v in zip(xs, vals):
                ax.annotate(f"{v:.2f}" if v < 1 else f"{v:.1f}", (x, v), ha="center",
                            va="bottom", fontsize=7.5, color="0.25")
        ax.set_xticks(np.arange(len(degrees)), [str(d) for d in degrees])
        ax.set_xlabel("target polynomial degree")
        ax.set_title(label, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6, axis="y"); ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, frameon=False)
    axes[0].set_ylabel("IPC ($\\Sigma$ capacity)")
    axes[0].annotate("odd degrees exactly 0:\n$|E|^2$ is even in amplitude",
                     xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8, color="0.35")
    axes[1].annotate("degree 1 = 4.00, the ceiling:\n$|E|^2$ is LINEAR in the encoded variable",
                     xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8, color="0.35")

    fig.suptitle("Input encoding sets which half of the polynomial basis is reachable",
                 fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for label, res in panels:
        for mode in ("field", "intensity"):
            by = res[mode]["ipc_by_degree"]
            print(f"{label:22s} {mode:10s} total {res[mode]['ipc_total']:6.2f}  "
                  + "  ".join(f"d{k} {v:6.2f}" for k, v in sorted(by.items())))
    print(f"\nwrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
