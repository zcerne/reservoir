"""Plot for characterization/n2_linear_residual.py's linear_residual() result."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_n2_linear_residual(res: dict, fig_dir: str | Path, suffix: str = "") -> Path:
    """Per-split held-out 1−R² (dots) around the mean (bar), vs the linear
    threshold — the direct visual of 'how much variance no linear map explains'."""
    per = np.asarray(res["per_repeat"])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = np.arange(per.size)
    ax.bar(["mean"], [res["residual_fraction"]], color="C0", alpha=0.5,
          yerr=[res["residual_std"]], capsize=4, width=0.5)
    ax.scatter(np.zeros(per.size) + 0.15 * (np.arange(per.size) - per.size / 2) / max(per.size, 1),
              per, color="C1", zorder=3, label="per split")
    ax.axhline(1e-6, color="k", ls=":", label="linear threshold (1e-6)")
    ax.set_yscale("log")
    ax.set_ylabel("held-out 1 − R²")
    verdict = "LINEAR" if res["linear"] else "NONLINEAR"
    warn = "  (underdetermined)" if res["underdetermined"] else ""
    ax.set_title(f"Linear residual (Method B) | N={res['n']}, f_in={res['f_in']}, "
                f"f_out={res['f_out']} -> {verdict}{warn}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(fig_dir) / f"n2_linear_residual{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
