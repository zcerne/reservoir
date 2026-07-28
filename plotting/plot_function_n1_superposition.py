"""Plot for characterization/n1_superposition.py's super_position_test() result."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_n1_superposition(res: dict, fig_dir: str | Path, suffix: str = "") -> Path:
    """Histogram of per-trial superposition violations ‖out_combo−(α·out1+β·out2)‖/‖out_combo‖,
    with the mean and the LINEAR/NONLINEAR threshold (1e-6) marked."""
    viol = np.asarray(res["per_trial"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    logv = np.log10(viol + 1e-30)
    ax.hist(logv, bins=min(30, max(5, viol.size // 2)), color="C0", alpha=0.8)
    ax.axvline(np.log10(res["violation"] + 1e-30), color="crimson", ls="--",
              label=f"mean = {res['violation']:.2e}")
    ax.axvline(np.log10(1e-6), color="k", ls=":", label="linear threshold (1e-6)")
    ax.set_xlabel("log10(relative violation)")
    ax.set_ylabel("trial count")
    verdict = "LINEAR" if res["linear"] else "NONLINEAR"
    ax.set_title(f"Superposition test (Method A) | {res['n_trials']} trials | "
                f"R²={res['r2']:.4f} -> {verdict}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(fig_dir) / f"n1_superposition{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
