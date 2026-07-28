"""Plot for characterization/n7_dimention_expansion.py's dimension_expansion() result."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_n7_dimension_expansion(res: dict, fig_dir: str | Path, suffix: str = "") -> Path:
    """Two panels: cumulative PCA-explained-variance of the output (a linear
    map keeps this ~K-dimensional, |E|² inflates it), and linear-fit R²(k) —
    how much of the output k linear input channels can explain."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    cum = np.asarray(res["cum_explained"])
    ax1.plot(np.arange(1, cum.size + 1), cum, "o-", color="C0")
    ax1.axhline(0.99, color="k", ls=":", label=f"99% at d99={res['d99']}")
    ax1.set_xlabel("# principal components")
    ax1.set_ylabel("cumulative explained variance")
    ax1.set_title(f"output PCA (pr={res['pr']:.2f})")
    ax1.legend(fontsize=8)

    ks = sorted(res["r2_vs_k"])
    ax2.plot(ks, [res["r2_vs_k"][k] for k in ks], "o-", color="C1")
    ax2.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax2.set_xlabel("# input channels used (k)")
    ax2.set_ylabel("held-out R² of best linear fit")
    ax2.set_title(f"linear-fit R²(k)  (plateau={res['plateau_r2']:.4f} at k={res['max_k']})")

    verdict = "LINEAR" if res["linear"] else "NONLINEAR"
    fig.suptitle(f"Dimension expansion (Method G) | {res['n_inputs']}→{res['n_outputs']}"
                f" -> {verdict}", fontsize=10)
    fig.tight_layout()
    out = Path(fig_dir) / f"n7_dimension_expansion{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
