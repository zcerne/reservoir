"""Plot for characterization/n5_voltera_series.py's volterra_series() result."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_n5_volterra_series(res: dict, fig_dir: str | Path, suffix: str = "") -> Path:
    """Two panels: held-out R² vs max polynomial degree included (cumulative),
    and the incremental variance gain attributed to each order — the Volterra
    kernel-energy breakdown."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    degs = sorted(res["r2_by_maxdeg"])
    ax1.plot(degs, [res["r2_by_maxdeg"][d] for d in degs], "o-", color="C0")
    ax1.set_xlabel("max degree included")
    ax1.set_ylabel("held-out R²")
    ax1.set_title("cumulative fit quality")
    ax1.set_ylim(min(0.0, min(res["r2_by_maxdeg"].values())) - 0.05, 1.05)

    orders = sorted(res["gain_by_order"])
    colors = ["C0" if d == 1 else "C1" for d in orders]
    ax2.bar([str(d) for d in orders], [res["gain_by_order"][d] for d in orders], color=colors)
    ax2.set_xlabel("Volterra order")
    ax2.set_ylabel("incremental R² gain")
    ax2.set_title(f"variance gain by order (max order = {res['max_order']})")

    verdict = "LINEAR" if res["linear"] else f"NONLINEAR (order {res['max_order']})"
    fig.suptitle(f"Volterra series (Method E) | degree≤{res['degree']}, N={res['n']} | "
                f"linear frac={res['linear_fraction']:.3f}  "
                f"nonlinear frac={res['nonlinear_fraction']:.3f} -> {verdict}", fontsize=10)
    fig.tight_layout()
    out = Path(fig_dir) / f"n5_volterra_series{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
