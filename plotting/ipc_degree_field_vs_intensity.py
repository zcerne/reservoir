"""Dambre IPC degree spectrum, field vs intensity readout, two designs side by side.

  python plotting/ipc_degree_field_vs_intensity.py
  python plotting/ipc_degree_field_vs_intensity.py --drive 100 --max-degree 5

Uses the repo's own `n6_dambre.dambre_ipc` and the same PCA reduction as
plot_ipc_by_drive, so the bars are directly comparable with the existing
`n6_dambre_ipc_*` figures rather than being a parallel re-implementation.

THE PARITY SPLIT IS THE POINT. |E|^2 is an even function of the drive, so the
intensity readout annihilates every ODD-degree target exactly -- those bars are 0.00,
not small. The field is very nearly odd, so it carries the odd degrees and reaches the
even ones only through whatever weak even component the medium has. The two readouts
therefore span complementary halves of the target family, and neither total is
meaningful on its own.

Read the even-degree field bars with care: they can be large while the state holds
almost no even-degree VARIANCE, because capacity is set by a component's SNR and not
by its amplitude. In noiseless FDTD an even component at 1e-4 of the amplitude is
still perfectly readable, and it dies under realistic detector noise. See
plot_kr_gr_spectra.py for the same warning applied to singular spectra.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from characterization import n6_dambre as n6          # noqa: E402
from plot_ipc_by_drive import _reduce_state           # noqa: E402

RUNS = Path("/home/ziga/Lips_project/reservoir_runs")
DEFAULT_PATHS = [RUNS / "reservoir_types" / "block_iso_gain" / "01", RUNS / "05_cav_4src"]
DEFAULT_LABELS = ["01 block + gain", "05 cavity"]
COLORS = ["#0a6ebd", "#d1701a"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", nargs="+", default=[str(p) for p in DEFAULT_PATHS])
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--drive", type=int, default=100)
    ap.add_argument("--max-degree", type=int, default=5)
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "ipc_degree_field_vs_intensity.png"))
    a = ap.parse_args()

    degrees = list(range(1, a.max_degree + 1))
    res, records = {}, []
    for path, label in zip(a.paths, a.labels):
        d = np.load(Path(path) / "datasets" / f"ipc_4src_a{a.drive}.npz", allow_pickle=True)
        U, X = d["inputs"], np.asarray(d["outputs"])
        for mode, Xv in (("field", X), ("intensity", np.abs(X) ** 2)):
            Xr = _reduce_state(np.asarray(Xv))
            r = n6.dambre_ipc({"inputs": U, "outputs": Xr},
                              max_degree=a.max_degree, max_features=Xr.shape[1])
            res[(label, mode)] = r
            records.append({"design": label, "mode": mode, "total": r["ipc_total"],
                            "nonlinear_fraction": r["nonlinear_fraction"],
                            "ceiling": int(r["bound"]),
                            "by_degree": {int(k): float(v) for k, v in r["ipc_by_degree"].items()}})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    width = 0.38
    for ax, mode in zip(axes, ("field", "intensity")):
        for i, (label, colour) in enumerate(zip(a.labels, COLORS)):
            r = res[(label, mode)]
            vals = [r["ipc_by_degree"].get(dg, 0.0) for dg in degrees]
            xs = np.arange(len(degrees)) + (i - 0.5) * width
            ax.bar(xs, vals, width, color=colour,
                   label=f"{label}   total {r['ipc_total']:.1f}")
            for x, v in zip(xs, vals):
                ax.annotate(f"{v:.1f}" if v else "0", (x, v), ha="center",
                            va="bottom", fontsize=7.5, color="0.25")
        ax.set_xticks(np.arange(len(degrees)), [str(dg) for dg in degrees])
        ax.set_xlabel("target polynomial degree")
        ax.set_title(f"{mode} readout", fontsize=10)
        ax.grid(alpha=0.25, lw=0.6, axis="y"); ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, frameon=False)
    axes[0].set_ylabel("IPC ($\\Sigma$ capacity)")
    axes[1].annotate("odd degrees are exactly zero:\n$|E|^2$ is even in the drive",
                     xy=(0.03, 0.72), xycoords="axes fraction", fontsize=8, color="0.35")
    fig.suptitle(f"Dambre IPC by degree, drive a={a.drive}", fontsize=11)
    fig.tight_layout()

    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    with open(out.with_suffix(".json"), "w") as f:
        json.dump({"drive": a.drive, "max_degree": a.max_degree, "results": records}, f, indent=1)

    print(f"\n{'design':<18}{'mode':>10}{'total':>8}{'nonlin':>8}{'ceil':>6}   by degree")
    for rec in records:
        by = "  ".join(f"d{dg} {rec['by_degree'].get(dg, 0.0):5.2f}" for dg in degrees)
        print(f"{rec['design']:<18}{rec['mode']:>10}{rec['total']:8.2f}"
              f"{rec['nonlinear_fraction']:8.3f}{rec['ceiling']:6d}   {by}")
    print(f"\nwrote {out}, .pdf and .json")


if __name__ == "__main__":
    raise SystemExit(main())
