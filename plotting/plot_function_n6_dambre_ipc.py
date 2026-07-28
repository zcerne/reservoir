"""Plot for characterization/n6_dambre.py's dambre_ipc() result."""
from __future__ import annotations
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_n6_dambre_ipc(res: dict, fig_dir: str | Path, suffix: str = "") -> Path:
    """Information Processing Capacity by polynomial degree, against the
    rank(X) ceiling — the gold-standard nonlinearity/capacity breakdown."""
    degs = sorted(res["ipc_by_degree"])
    vals = [res["ipc_by_degree"][d] for d in degs]
    colors = ["C0" if d == 1 else "C1" for d in degs]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar([str(d) for d in degs], vals, color=colors)
    ax.axhline(res["bound"], color="k", ls="--", label=f"ceiling = rank(X) = {res['bound']}")
    ax.set_xlabel("target polynomial degree")
    ax.set_ylabel("IPC (Σ capacity)")
    verdict = "LINEAR" if res["linear"] else "NONLINEAR"
    ax.set_title(f"Dambre IPC (Method F) | total={res['ipc_total']:.3f}, "
                f"{res['n_targets']} targets | nonlinear frac={res['nonlinear_fraction']:.3f}"
                f" -> {verdict}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(fig_dir) / f"n6_dambre_ipc{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
