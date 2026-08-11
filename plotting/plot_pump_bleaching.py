"""Inversion built by a short vs a block-spanning pump pulse, from pump-only runs.

  python plotting/plot_pump_bleaching.py --long <n3_map_long.npz> --short <n3_map.npz>

Each input is the reduced map written from a `pop_monitor_<suffix>.npz` (N3 averaged
over the reservoir's y extent -> N3(x, t)).

THE POINT. A short pump crosses the block as a thin sliver and leaves a decaying ramp,
because it is gone before the leading edge can saturate. Lengthening it until it spans
the block lets the front reach full inversion, at which point it STOPS ABSORBING --
the absorption is saturable -- and the rest of the pulse propagates into fresh
medium. The result is a bleaching front that sweeps the block rather than an
exponential decay, and the difference is the third panel: 29% mean inversion becomes
98%.

A caveat the figure deliberately makes visible: at 98% the medium is essentially
transparent to the pump, so the transmitted pump becomes INSENSITIVE to further
population change. If the pump channel is meant to report on the signal (an even-order
readout, since depletion goes as intensity), full saturation is the wrong operating
point -- partial inversion is where the pump transmission actually responds.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FULL = 30.0
COL_SHORT, COL_LONG, COL_MID = "#2a72c4", "#e08a1e", "#9b4dca"


def load(npz):
    d = np.load(npz, allow_pickle=True)
    return d["x"], d["t"], d["n3"] / FULL * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", required=True)
    ap.add_argument("--short", default=None)
    ap.add_argument("--mid", default=None,
                    help="third map (e.g. the same long pulse at lower pump amplitude)")
    ap.add_argument("--mid-label", default="long pump, amplitude 50")
    ap.add_argument("--times", type=float, nargs="+", default=[50, 100, 125, 150, 200, 300])
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "pump_bleaching_02.png"))
    a = ap.parse_args()

    x, t, n3 = load(a.long)
    ncol = 3 if a.short else 2
    fig, axes = plt.subplots(1, ncol, figsize=(5.5 * ncol, 4.4))
    ax, ax2 = axes[0], axes[1]

    im = ax.pcolormesh(x, t, n3, cmap="magma", shading="auto", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, pad=0.02).set_label("inversion N$_3$ (% of full)")
    ax.set_xlabel("x through the block ($\\mu$m)")
    ax.set_ylabel("time (MEEP units)")
    ax.set_title("long pump: a bleaching front sweeps the block", fontsize=10)

    # Final profiles rather than one run's time slices: the comparison that decides
    # the operating point is amplitude against amplitude, not instant against instant.
    if a.mid:
        xm, tm, nm = load(a.mid)
        ax2.plot(xm, nm[-1], lw=2.0, color=COL_MID, label=a.mid_label)
    ax2.plot(x, n3[-1], lw=2.0, color=COL_LONG, label="long pump, amplitude 200")
    if a.short:
        xs0, ts0, ns0 = load(a.short)
        ax2.plot(xs0, ns0[-1], lw=2.0, color=COL_SHORT, label="short pump, amplitude 200")
    ax2.axhspan(40, 70, color="0.85", zorder=0)
    ax2.annotate("useful band for an Ez readout:\ngain to deplete, absorber left to notice",
                 xy=(-14, 55), fontsize=7.5, color="0.4")
    ax2.set_xlabel("x through the block ($\\mu$m)")
    ax2.set_ylabel("inversion N$_3$ (% of full)")
    ax2.set_title("final profile, three pump settings", fontsize=10)
    ax2.set_ylim(-3, 105)
    ax2.grid(alpha=0.25, lw=0.6); ax2.set_axisbelow(True)
    ax2.legend(fontsize=8, frameon=False, loc="lower left")

    if a.short:
        xs, ts, ns = load(a.short)
        ax3 = axes[2]
        # Mean over the block interior: the final 0.25 um reads exactly zero because
        # the sted block stops short of the last cells, and including that sliver
        # drags the mean down for no physical reason.
        inner = abs(x) < 14.8
        inner_s = abs(xs) < 14.8
        ax3.plot(ts, ns[:, inner_s].mean(1), "o-", ms=4, lw=1.7, color=COL_SHORT,
                 label="short pump, amp 200")
        ax3.plot(t, n3[:, inner].mean(1), "s-", ms=4, lw=1.7, color=COL_LONG,
                 label="long pump, amp 200")
        if a.mid:
            xm2, tm2, nm2 = load(a.mid)
            ax3.plot(tm2, nm2[:, abs(xm2) < 14.8].mean(1), "^-", ms=4, lw=1.7,
                     color=COL_MID, label=a.mid_label)
        ax3.axhline(100, color="0.45", ls=":", lw=1.2)
        ax3.annotate("full inversion — medium now transparent\nto the pump, so its "
                     "transmission stops\nresponding to population",
                     xy=(t.max() * 0.52, 100), xytext=(t.max() * 0.20, 62),
                     fontsize=7.5, color="0.35",
                     arrowprops=dict(arrowstyle="->", color="0.55", lw=1))
        ax3.set_xlabel("time (MEEP units)")
        ax3.set_ylabel("block-mean inversion (%)")
        ax3.set_title("what lengthening the pump bought", fontsize=10)
        ax3.grid(alpha=0.25, lw=0.6); ax3.set_axisbelow(True)
        ax3.legend(fontsize=8.5, frameon=False, loc="center right")
        print(f"final block-mean: short {ns[-1, inner_s].mean():.1f}%   "
              f"long {n3[-1, inner].mean():.1f}%")

    fig.suptitle("block_iso_gain/02 — inversion from the pump alone (signal silenced)",
                 fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
