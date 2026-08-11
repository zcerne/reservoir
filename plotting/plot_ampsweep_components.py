"""Output amplitude per polarization and total intensity vs drive, from an amp sweep.

  python plotting/plot_ampsweep_components.py --path <design>
  python plotting/plot_ampsweep_components.py --path <design> --stem amp_sweep

Reads <path>/datasets/<stem>.npz, or its .parts directory if the sweep has not been
assembled yet, and plots per-component |E| and the total intensity against drive.

The readout is the near-field-to-far-field map at ONE wavelength (the signal line,
0.55), packed by _gen_common as [Ex | Ey | Ez] along the far-field column. So:

* amplitude_c = ||E_c||_2 over the far-field points, and intensity = sum_c amplitude_c^2,
  which keeps the two panels exactly consistent rather than mixing an L1 amplitude
  with an L2 intensity.
* Ez is the PUMP's polarization and its source is fixed, so its curve should be flat.
  It is plotted precisely because a flat line there is the control: any slope on Ez
  would mean the signal drive is leaking into the pump channel.

Both panels carry a reference line for a linear medium (amplitude proportional to
drive, intensity to drive squared), anchored at the lowest drive. Departure from it
IS the saturation, and reading it off a log-log plot without that anchor is guesswork.
"""
from __future__ import annotations
import argparse, glob, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
RUNS = Path("/home/ziga/Orion/resevoir/data/reservoir_types/block_iso_gain/02")
COMPS = ["Ex", "Ey", "Ez"]
COLORS = {"Ex": "#2a72c4", "Ey": "#e08a1e", "Ez": "#9b4dca"}
MARKERS = {"Ex": "o", "Ey": "s", "Ez": "^"}      # secondary encoding, not color alone


def load_sweep_m2(path, stem):
    """(levels, amps[n_lvl, 3]) from monitor_2, summed over ALL 61 wavelengths.

    The near2far `output` is a single wavelength (0.55 by default), which on the
    coupled design contains none of the 0.45 pump channel. Integrating monitor_2 over
    the whole comb keeps both lines, so Ez here is dominated by the pump and carries
    its depletion rather than reading as numerical residue.
    """
    fs = sorted(glob.glob(str(Path(path) / "datasets" / f"{stem}.npz.parts")
                          + "/part_*.npz"))
    if not fs:
        raise SystemExit(f"no parts for {stem} under {path}")
    lv, amps = [], []
    for f in fs:
        with np.load(f, allow_pickle=True) as d:
            lv.append(float(np.asarray(d["inp"]).ravel()[0]))
            amps.append([np.linalg.norm(np.asarray(d["m2_" + c])) for c in COMPS])
    o = np.argsort(lv)
    return np.array(lv)[o], np.array(amps)[o]


def load_sweep(path, stem):
    """(levels, amps[n_lvl, 3]) from the assembled npz, else straight from parts."""
    ds = Path(path) / "datasets" / f"{stem}.npz"
    if ds.exists():
        with np.load(ds, allow_pickle=True) as d:
            out, inp = np.asarray(d["outputs"]), np.asarray(d["inputs"])
        levels = inp[:, 0] if inp.ndim > 1 else inp
    else:
        fs = sorted(glob.glob(str(ds) + ".parts/part_*.npz"))
        if not fs:
            raise SystemExit(f"no {ds} and no parts beside it")
        print(f"[ampsweep] {ds.name} not assembled — reading {len(fs)} parts")
        out, levels = [], []
        for f in fs:
            with np.load(f, allow_pickle=True) as d:
                out.append(np.asarray(d["output"]))
                levels.append(float(np.asarray(d["inp"]).ravel()[0]))
        out, levels = np.array(out), np.array(levels)
    n = out.shape[1] // len(COMPS)
    amps = np.stack([np.linalg.norm(out[:, i * n:(i + 1) * n], axis=1)
                     for i in range(len(COMPS))], axis=1)
    o = np.argsort(levels)
    return levels[o], amps[o]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", default=str(RUNS))
    ap.add_argument("--stem", default="amp_sweep")
    ap.add_argument("--label", default="block_iso_gain/02 (pulsed pump)")
    ap.add_argument("--from-m2", action="store_true",
                    help="use monitor_2 summed over all 61 wavelengths instead of the "
                         "single-wavelength near2far readout")
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "ampsweep_components_02.png"))
    a = ap.parse_args()

    levels, amps = (load_sweep_m2(a.path, a.stem) if a.from_m2
                    else load_sweep(a.path, a.stem))
    inten = (amps ** 2).sum(axis=1)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    for i, c in enumerate(COMPS):
        ax.plot(levels, amps[:, i], MARKERS[c] + "-", ms=5, lw=1.7,
                color=COLORS[c], label=f"|{c}|")
    ref = amps[0, 1] * levels / levels[0]
    ax.plot(levels, ref, ":", lw=1.4, color="0.45", label="linear response")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("drive amplitude"); ax.set_ylabel("$\\|E\\|_2$, summed over all wavelengths" if a.from_m2
                  else "output $\\|E\\|_2$")
    ax.set_title("per-polarization amplitude", fontsize=10)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax.annotate("Ez = pump line" + (" — now falls: signal depletes it" if a.from_m2
                                    else ", fixed source"), xy=(30, amps[0, 2]),
                xytext=(1.15, amps[0, 2] * 0.13), fontsize=8, color="0.35",
                arrowprops=dict(arrowstyle="->", color="0.55", lw=1))

    # Total intensity is NOT the right thing to anchor a linear reference to: at low
    # drive it is almost entirely the fixed pump's own transmitted light (Ez), so an
    # A^2 line through it would claim the reservoir is saturating when really the
    # signal is just still below the pump floor. Split them.
    sig = (amps[:, 0] ** 2 + amps[:, 1] ** 2)      # Ex,Ey = the TE mode the signal drives
    pump = float(np.median(amps[:, 2] ** 2))
    ax2.plot(levels, inten, "o-", ms=5, lw=1.7, color="#0f7d6b", label="total")
    ax2.plot(levels, sig, "s--", ms=4.5, lw=1.6, color="#e08a1e",
             label="signal only ($E_x^2+E_y^2$)")
    ax2.axhline(pump, color="#9b4dca", lw=1.4, ls="-.", label="pump floor ($E_z^2$)")
    ax2.plot(levels, sig[0] * (levels / levels[0]) ** 2, ":", lw=1.4, color="0.45",
             label="linear medium ($\\propto A^2$)")
    cross = levels[np.argmax(sig > pump)]
    ax2.annotate(f"signal overtakes\nthe pump at A$\\approx${cross:g}",
                 xy=(cross, pump), xytext=(1.3, pump * 12), fontsize=8, color="0.35",
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=1))
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("drive amplitude"); ax2.set_ylabel("$\\sum |E|^2$")
    ax2.set_title("total intensity", fontsize=10)
    ax2.grid(alpha=0.25, lw=0.6); ax2.set_axisbelow(True)
    ax2.legend(fontsize=8.5, frameon=False, loc="upper left")

    fig.suptitle(f"Amplitude sweep — {a.label}", fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'drive':>7}{'|Ex|':>11}{'|Ey|':>11}{'|Ez|':>11}{'intensity':>13}"
          f"{'Ey/linear':>11}")
    for j, lv in enumerate(levels):
        print(f"{lv:7.0f}{amps[j,0]:11.4g}{amps[j,1]:11.4g}{amps[j,2]:11.4g}"
              f"{inten[j]:13.5g}{amps[j,1]/ref[j]:11.3f}")
    print(f"\nwrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
