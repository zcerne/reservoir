"""The pump line as an EVEN-order readout channel, from a coupled amplitude sweep.

  python plotting/plot_pump_even_channel.py --path <design> --stem amp_sweep_coupled \\
      --baseline <simulation_gpumeep/monitor_2_<pumponly suffix>.npz>

Design 02 overlaps a 450 nm Ez pump pulse with a 550 nm Ey signal. They share no field
component and no wavelength, so they couple ONLY through the dye populations: the
signal depletes N3 by stimulated emission, the extra N1 left behind absorbs more pump,
and the transmitted pump falls. Depletion follows signal INTENSITY, so the pump channel
carries A^2, A^4 ... — even orders by construction, which is the half of the polynomial
basis a field readout of an (almost) odd-symmetric medium cannot reach.

READ THE WAVELENGTH CAREFULLY. With dlam=0.005 the pump occupies 0.445-0.455 only, so
at the signal line 0.55 there is no pump left to modulate (|Ez| ~ 1e-4, five orders
down). The near2far readout takes a single wavelength — 0.55 by default — and would
therefore throw this entire channel away. It survives only in the monitor_2 extras,
which keep all 61 wavelengths. Any dataset meant to capture the even orders on this
design must read the pump line.

The baseline is the pump-only run (signal silenced) at the same pump amplitude: the
true A=0 point, without which the low-drive exponent cannot be fitted at all.
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
COL_PUMP, COL_SIG = "#9b4dca", "#e08a1e"


def line_norm(arr, lam_axis, lam):
    return np.linalg.norm(np.asarray(arr)[int(np.argmin(abs(lam_axis - lam)))])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(DESIGN))
    ap.add_argument("--stem", default="amp_sweep_coupled")
    ap.add_argument("--baseline", default=None, help="pump-only monitor_2 npz (the A=0 point)")
    ap.add_argument("--lam-pump", type=float, default=0.45)
    ap.add_argument("--lam-signal", type=float, default=0.55)
    ap.add_argument("--fit-max", type=float, default=20.0,
                    help="fit the exponent below this drive, before saturation bends it")
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "pump_even_channel_02.png"))
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(str(Path(a.path) / "datasets" / f"{a.stem}.npz.parts")
                              + "/part_*.npz")):
        with np.load(f, allow_pickle=True) as d:
            lam = 1.0 / np.asarray(d["m2_freqs"])
            rows.append((float(np.asarray(d["inp"]).ravel()[0]),
                         line_norm(d["m2_Ez"], lam, a.lam_pump),
                         line_norm(d["m2_Ey"], lam, a.lam_signal)))
    rows.sort()
    A = np.array([r[0] for r in rows])
    ez = np.array([r[1] for r in rows])
    ey = np.array([r[2] for r in rows])

    if a.baseline:
        with np.load(a.baseline) as d:
            lam0 = 1.0 / np.asarray(d["freqs"])
            ez0 = line_norm(d["Ez"], lam0, a.lam_pump)
    else:
        ez0 = ez[0]
    dep = 1.0 - ez / ez0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax.axhline(ez0, color="0.45", ls=":", lw=1.3)
    ax.annotate(f"pump alone (A=0): {ez0:.0f}", xy=(A[0], ez0), xytext=(A[0] * 1.1, ez0 * 1.006),
                fontsize=8, color="0.4")
    ax.plot(A, ez, "^-", ms=5.5, lw=1.8, color=COL_PUMP)
    ax.set_xscale("log")
    ax.set_xlabel("signal drive amplitude")
    ax.set_ylabel(f"transmitted pump $\\|E_z\\|$ at {a.lam_pump:g} $\\mu$m")
    ax.set_title("the signal attenuates the pump", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.annotate(f"{(1-ez[-1]/ez0)*100:.0f}% attenuation at A={A[-1]:g}",
                xy=(A[-1], ez[-1]), xytext=(3, ez[-1] * 1.02), fontsize=8, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1))

    m = A <= a.fit_max
    slope = np.polyfit(np.log(A[m]), np.log(dep[m]), 1)[0]
    ax2.plot(A, dep, "^-", ms=5.5, lw=1.8, color=COL_PUMP, label="pump depletion")
    ax2.plot(A, dep[0] * (A / A[0]) ** 2, ":", lw=1.4, color="0.45",
             label="$\\propto A^2$ (pure even, 2nd order)")
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("signal drive amplitude")
    ax2.set_ylabel("$1 - E_z / E_z(A{=}0)$")
    ax2.set_title(f"depletion follows intensity: fitted $A^{{{slope:.2f}}}$ below A={a.fit_max:g}",
                  fontsize=10)
    ax2.grid(alpha=0.25, lw=0.6); ax2.set_axisbelow(True)
    ax2.legend(fontsize=8.5, frameon=False, loc="upper left")

    fig.suptitle("block_iso_gain/02 coupled — the pump line is an even-order channel",
                 fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"A=0 baseline |Ez@{a.lam_pump}| = {ez0:.6g}")
    print(f"{'drive':>7}{'|Ez| pump':>12}{'depletion':>12}{'local slope':>13}{'|Ey| signal':>14}")
    for i, lv in enumerate(A):
        s = (f"{np.log(dep[i]/dep[i-1])/np.log(A[i]/A[i-1]):.2f}"
             if i and dep[i-1] > 0 else "")
        print(f"{lv:7.0f}{ez[i]:12.6g}{dep[i]:12.5f}{s:>13}{ey[i]:14.6g}")
    print(f"\nlow-drive exponent (A<={a.fit_max:g}): {slope:.2f}")
    print(f"wrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
