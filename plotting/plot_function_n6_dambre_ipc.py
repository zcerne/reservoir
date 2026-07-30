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
    tbd = res.get("targets_by_degree", {})
    # The achievable maximum is min(rank(X), #targets): with K inputs and
    # max_degree 3 there are only n_targets functions to reconstruct, so a
    # rank-56 readout cannot show 56 capacity — drawing rank alone squashes
    # every real bar against the axis and implies unreachable headroom.
    n_tgt = int(res.get("n_targets", 0)) or None
    ceiling = min(res["bound"], n_tgt) if n_tgt else res["bound"]
    for x, (d, v) in enumerate(zip(degs, vals)):
        n_t = tbd.get(d)
        label = (f"{v:.2f}\n({100 * v / n_t:.0f}%)" if n_t else f"{v:.2f}")
        ax.annotate(label, (x, v), textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=8)
    lbl = f"max reachable = min(rank {res['bound']}, {n_tgt} targets) = {ceiling}" \
        if n_tgt and n_tgt < res["bound"] else f"ceiling = rank(X) = {res['bound']}"
    ax.axhline(ceiling, color="k", ls="--", label=lbl)
    ax.set_ylim(0, ceiling * 1.12)
    ax.set_xlabel("target polynomial degree")
    ax.set_ylabel("IPC (Σ capacity)")
    verdict = "LINEAR" if res["linear"] else "NONLINEAR"
    used = (f", {res['f_used']}/{res['f_out']} ch" if res.get("f_used") and
            res.get("f_out") and res["f_used"] != res["f_out"] else "")
    ax.set_title(f"Dambre IPC (Method F) | total={res['ipc_total']:.3f}, "
                f"{res['n_targets']} targets{used} | "
                f"nonlinear frac={res['nonlinear_fraction']:.3f}"
                f" -> {verdict}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(fig_dir) / f"n6_dambre_ipc{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
