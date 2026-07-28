"""Plot for characterization/n4_harmonics_distortion.py's harmonic_specter() result."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_n4_harmonics_distortion(res: dict, fig_dir: str | Path, suffix: str = "") -> Path:
    """Two panels: power by kind (dc/fundamental/harmonic/intermod/other) and
    power by order — where the nonlinear energy actually sits."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    kinds = ["dc", "fundamental", "harmonic", "intermod", "other"]
    vals = [res["power_by_kind"][k] for k in kinds]
    ax1.bar(kinds, vals, color=["gray", "C0", "C1", "C2", "C3"])
    ax1.set_yscale("log")
    ax1.set_ylabel("power")
    ax1.set_title("power by kind")
    ax1.tick_params(axis="x", rotation=30)

    orders = sorted(res["power_by_order"])
    ax2.bar([str(o) for o in orders], [res["power_by_order"][o] for o in orders], color="C1")
    ax2.set_yscale("log")
    ax2.set_xlabel("order")
    ax2.set_ylabel("power")
    ax2.set_title(f"power by order (max nonlinear order = {res['max_order']})")

    verdict = "LINEAR" if res["linear"] else f"NONLINEAR (order {res['max_order']})"
    fig.suptitle(f"Harmonic/intermod distortion (Method D) | tones={res['tones']} | "
                f"THD={res['thd']:.2e}  IMD={res['imd']:.2e}  "
                f"distortion_frac={res['distortion_frac']:.2e} -> {verdict}", fontsize=10)
    fig.tight_layout()
    out = Path(fig_dir) / f"n4_harmonics_distortion{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
