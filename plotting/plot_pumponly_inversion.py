"""Inversion laid down by the pump PULSE in block_iso_gain/02, from a pump-only run.

  python plotting/plot_pumponly_inversion.py --npz <n3_map_pumponly.npz>

Input is the reduced map written from `pop_monitor_<suffix>.npz` (N3 averaged over
the reservoir's y extent -> N3(x, t)), plus the whole-block N(t) trace.

WHAT THE FIGURE IS FOR. 01 starts uniformly inverted and a CW area pump holds it
there. 02 replaces that with one guided pulse, which deposits inversion only where
it has not yet been absorbed -- so the medium becomes a gradient rather than a slab,
and the question is what the signal actually flies through. Both worldlines are
therefore drawn on the map: light travels at c/n, so pump and signal move at the same
speed and the signal never catches the front it is chasing; it only ever sees the
wake, 20 time units behind.
"""
from __future__ import annotations
import argparse, os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
FULL = 30.0          # population density per cell = fully inverted
N_INDEX = 1.5        # block index, sets both worldline slopes
SIGNAL_START = 20.0  # source_1.start_time in 02
PROFILE_TIMES = [10, 20, 30, 60, 100]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default=str(REPO / "data" / "reservoir_types"
                                         / "pumponly_inversion_02.png"))
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    x, t, n3 = d["x"], d["t"], d["n3"] / FULL * 100.0     # percent of full inversion

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4),
                                  gridspec_kw={"width_ratios": [1.35, 1]})

    im = ax.pcolormesh(x, t, n3, cmap="magma", shading="auto", vmin=0, vmax=100)
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("inversion N$_3$ (% of full)")

    # Worldlines: both pulses move at c/n through the same medium.
    v = 1.0 / N_INDEX
    tt = np.linspace(0, t.max(), 200)
    for t0, colour, label in ((0.0, "#4de1ff", "pump pulse"),
                              (SIGNAL_START, "#ffffff", "signal (start 20)")):
        xs = x.min() + v * (tt - t0)
        keep = (tt >= t0) & (xs <= x.max())
        ax.plot(xs[keep], tt[keep], color=colour, lw=1.6, ls="--", label=label)
    ax.set_xlabel("x through the block ($\\mu$m)")
    ax.set_ylabel("time (MEEP units)")
    ax.set_title("inversion laid down by the pump pulse", fontsize=10)
    ax.legend(fontsize=8, frameon=True, facecolor="black", edgecolor="none",
              labelcolor="white", loc="upper left")

    # Right: the same data as profiles, which is where the ramp is legible.
    cmap = plt.get_cmap("viridis")
    for i, tv in enumerate(PROFILE_TIMES):
        j = int(np.argmin(abs(t - tv)))
        ax2.plot(x, n3[j], lw=1.8, color=cmap(i / max(len(PROFILE_TIMES) - 1, 1)),
                 label=f"t = {t[j]:g}")
    ax2.set_xlabel("x through the block ($\\mu$m)")
    ax2.set_ylabel("inversion N$_3$ (% of full)")
    ax2.set_title("profile along propagation", fontsize=10)
    ax2.grid(alpha=0.25, lw=0.6); ax2.set_axisbelow(True)
    ax2.legend(fontsize=8.5, frameon=False)
    # Quote the exit value from just inside the block: the final 0.25 um reads
    # exactly zero because the sted block stops short of the last cells, and
    # sampling that sliver makes the gain look like it dies before the exit.
    j100 = int(np.argmin(abs(t - 100)))
    exit_pct = n3[j100][np.argmin(abs(x - (x.max() - 0.5)))]
    ax2.annotate(f"exit: {exit_pct:.0f}% left\n({n3[j100][0] / exit_pct:.0f}$\\times$ gradient"
                 f"\nacross the block)",
                 xy=(x.max() - 0.5, exit_pct), xytext=(1.0, 38), fontsize=8, color="0.35",
                 arrowprops=dict(arrowstyle="->", color="0.55", lw=1))

    fig.suptitle("block_iso_gain/02, pump pulse only (source_1 silenced)", fontsize=11)
    fig.tight_layout()
    out = Path(a.out)
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    j100 = int(np.argmin(abs(t - 100)))
    print(f"block mean at t=100: {n3[j100].mean():.1f}%   entry {n3[j100][0]:.1f}%   "
          f"exit {n3[j100][-1]:.1f}%")
    print(f"wrote {out} and .pdf")


if __name__ == "__main__":
    raise SystemExit(main())
