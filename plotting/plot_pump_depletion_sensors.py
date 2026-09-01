"""Pump-line depletion vs drive, measured in the FAR field and in the cell.

  python plotting/plot_pump_depletion_sensors.py --sweep <amp_sweep_alllam.npz>

Needs a sweep whose n2f readout kept the whole comb (`--n2f_lam all`); with the
historical single-wavelength readout the pump line is simply absent from `output` and
only the monitor_2 extras can show this.

WHY BOTH SENSORS. monitor_2 sits inside guide_2, so it is a diagnostic, not something
an experiment can place. The far field at 200 um is what a detector actually sees. If
the depletion survives the near-to-far transform then the even-order channel is
measurable rather than merely present, which is the difference between a result and a
simulation artefact.

The A=0 point comes from the pump-only run at the same pump amplitude -- without it the
low-drive exponent cannot be fitted, only guessed from ratios between driven points.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
COL = {"n2f far field": "#2a72c4", "monitor_2 in-cell": "#e08a1e"}
MARK = {"n2f far field": "o", "monitor_2 in-cell": "s"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--n2f-baseline", required=True, help="pump-only n2f_map npz (A=0)")
    ap.add_argument("--m2-baseline", required=True, help="pump-only monitor_2 npz (A=0)")
    ap.add_argument("--lam-pump", type=float, default=0.45)
    ap.add_argument("--n-lam", type=int, default=61)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--fit-max", type=float, default=20.0)
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "pump_depletion_sensors_02.png"))
    a = ap.parse_args()

    d = np.load(a.sweep, allow_pickle=True)
    U = np.asarray(d["inputs"]); lv = U[:, 0] if U.ndim > 1 else U
    order = np.argsort(lv); A = lv[order]
    nw, npt = a.n_lam, a.n_points
    lam = 1.0 / np.linspace(1 / 0.6, 1 / 0.4, nw)
    ip = int(np.argmin(abs(lam - a.lam_pump)))

    X = np.asarray(d["outputs"])
    ez_n2f = np.linalg.norm(X[:, 2 * nw * npt:3 * nw * npt]
                            .reshape(len(X), nw, npt)[:, ip, :], axis=1)[order]
    ez_m2 = np.linalg.norm(np.asarray(d["m2_Ez"])[:, ip, :], axis=1)[order]

    b = np.load(a.n2f_baseline)
    lb = np.asarray(b["lams"])
    n2f0 = np.linalg.norm(np.asarray(b["EH"])[int(np.argmin(abs(lb - a.lam_pump))), -1, :, 2])
    m = np.load(a.m2_baseline)
    lm = 1.0 / np.asarray(m["freqs"])
    m20 = np.linalg.norm(np.asarray(m["Ez"])[int(np.argmin(abs(lm - a.lam_pump)))])

    series = {"n2f far field": (ez_n2f, n2f0), "monitor_2 in-cell": (ez_m2, m20)}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, (ez, ez0) in series.items():
        ax.plot(A, ez / ez0, MARK[name] + "-", ms=5.5, lw=1.8, color=COL[name], label=name)
        dep = 1.0 - ez / ez0
        ok = dep > 0
        sl = np.polyfit(np.log(A[ok & (A <= a.fit_max)]),
                        np.log(dep[ok & (A <= a.fit_max)]), 1)[0]
        ax2.plot(A[ok], dep[ok], MARK[name] + "-", ms=5.5, lw=1.8, color=COL[name],
                 label=f"{name}   $A^{{{sl:.2f}}}$ below {a.fit_max:g}")
    ax.axhline(1.0, color="0.45", ls=":", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("signal drive amplitude")
    ax.set_ylabel(f"transmitted pump at {a.lam_pump:g} $\\mu$m, relative to A=0")
    ax.set_title("the signal attenuates the pump, in both sensors", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False)

    ref = 1.0 - series["n2f far field"][0][0] / series["n2f far field"][1]
    ax2.plot(A, ref * (A / A[0]) ** 2, ":", lw=1.4, color="0.45",
             label="$\\propto A^2$ (pure even)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("signal drive amplitude")
    ax2.set_ylabel("depletion $1 - E_z/E_z(A{=}0)$")
    ax2.set_title("depletion follows intensity", fontsize=10)
    ax2.grid(alpha=0.25, lw=0.6); ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, frameon=False)

    fig.suptitle("block_iso_gain/02 — the even-order pump channel is visible in the far field",
                 fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"{'drive':>7}" + "".join(f"{n:>26}" for n in series))
    for i, lvl in enumerate(A):
        row = "".join(f"{ez[i]:13.5g}{(1-ez[i]/ez0)*100:12.2f}%" for ez, ez0 in series.values())
        print(f"{lvl:7.0f}" + row)
    print(f"\nwrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
