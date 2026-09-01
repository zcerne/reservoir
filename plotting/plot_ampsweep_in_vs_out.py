"""Input vs output intensity across an amplitude sweep, per polarization and total.

  python plotting/plot_ampsweep_in_vs_out.py --sweep <amp_sweep_alllam.npz>

Output intensity is |E|^2 summed over the whole 61-wavelength comb and over the
far-field points, one number per polarization per drive. Input intensity is the
injected A^2 (the sweep drives one uniform strip, so A is the whole input).

A linear medium would put every curve on a slope-1 line here, since intensity out
would be proportional to intensity in. Departure from the dotted reference is the
saturation, and reading it off log-log without that anchor is guesswork.

Ez IS THE PUMP and its source is fixed, so it does not follow the input at all -- it
falls, because the signal depletes the inversion the pump is being absorbed by. It is
plotted on the same axes precisely to make that contrast visible: two polarizations
rising sublinearly and one falling is the signature of the coupled design.
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
COLORS = {"Ex": "#2a72c4", "Ey": "#e08a1e", "Ez": "#9b4dca",
          "signal": "#c2185b", "total": "#0f7d6b"}
MARKS = {"Ex": "o", "Ey": "s", "Ez": "^", "signal": "v", "total": "D"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--n-lam", type=int, default=61)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--sensor", choices=["n2f", "monitor_2"], default="n2f")
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "ampsweep_in_vs_out_02.png"))
    a = ap.parse_args()

    d = np.load(a.sweep, allow_pickle=True)
    U = np.asarray(d["inputs"]); A = (U[:, 0] if U.ndim > 1 else U)
    order = np.argsort(A); A = A[order]
    nw, npt = a.n_lam, a.n_points

    I = {}
    if a.sensor == "n2f":
        X = np.asarray(d["outputs"])[order]
        for c, comp in enumerate(COMPS):
            blk = X[:, c * nw * npt:(c + 1) * nw * npt].reshape(len(X), nw, npt)
            I[comp] = (np.abs(blk) ** 2).sum(axis=(1, 2))
    else:
        for comp in COMPS:
            I[comp] = (np.abs(np.asarray(d["m2_" + comp])[order]) ** 2).sum(axis=(1, 2))
    I["signal"] = I["Ex"] + I["Ey"]        # the TE mode the drive actually launches
    I["total"] = sum(I[c] for c in COMPS)
    I_in = A ** 2                      # injected intensity; one uniform strip

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for k in COMPS + ["signal", "total"]:
        lab = {"total": "total", "signal": "signal $E_x^2+E_y^2$"}.get(
            k, f"$|{k[0]}_{{{k[1]}}}|^2$")
        ax.plot(I_in, I[k], MARKS[k] + "-", ms=5.5, lw=1.8, color=COLORS[k], label=lab)
    # Anchor the linear reference on the SIGNAL, not the total: at low drive the total
    # is mostly the constant pump, and a constant term drags the log-log slope toward
    # zero however linear the signal is (slope = kI/(C+kI)). Anchoring on the total
    # therefore makes a linear medium look saturated.
    ref = I["signal"][0] * I_in / I_in[0]
    ax.plot(I_in, ref, ":", lw=1.5, color="0.45", label="linear medium (slope 1)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("input intensity  $A^2$  (arb.)")
    ax.set_ylabel(f"output intensity, summed over 61 wavelengths ({a.sensor})")
    ax.set_title("block_iso_gain/02 — in vs out, coupled pump-probe", fontsize=11)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.annotate("$E_z$ = pump: falls as the signal\ndepletes the inversion",
                xy=(I_in[-2], I["Ez"][-2]), xytext=(I_in[1] * 1.5, I["Ez"][0] * 0.06),
                fontsize=8, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1))

    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"{'A':>6}{'I_in':>10}" + "".join(f"{k:>13}" for k in COMPS + ["signal", "total"])
          + f"{'slope sig':>11}{'pump %':>9}")
    for i, lvl in enumerate(A):
        s = (f"{np.log(I['signal'][i]/I['signal'][i-1])/np.log(I_in[i]/I_in[i-1]):.3f}"
             if i else "")
        print(f"{lvl:6.0f}{I_in[i]:10.4g}"
              + "".join(f"{I[k][i]:13.5g}" for k in COMPS + ["signal", "total"])
              + f"{s:>11}{I['Ez'][i]/I['total'][i]*100:8.1f}%")
    print(f"\nwrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
