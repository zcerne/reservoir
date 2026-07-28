"""Plot for characterization/n3_amplitude_dependant.py's amplitude_dependance() result."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_n3_amplitude_dependance(res: dict, fig_dir: str | Path, suffix: str = "") -> Path:
    """G-drift vs drive level (Pintelon-Schoukens style): a linear system's best
    linear map is amplitude-independent, so any drift with level is nonlinearity,
    resolved by WHERE it turns on."""
    levels = np.asarray(res["levels"])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(levels, res["drift"], "o-", color="C0", label="‖Gₗ−Gref‖/‖Gref‖")
    ax.plot(levels, res["sv_drift"], "s--", color="C1", label="singular-value drift")
    ax.axhline(1e-6, color="k", ls=":", label="linear threshold (1e-6)")
    ax.axvline(res["ref_level"], color="gray", ls=":", alpha=0.6)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("drive level")
    ax.set_ylabel("relative drift of the best-linear map G")
    verdict = "LINEAR" if res["linear"] else "NONLINEAR"
    ax.set_title(f"Amplitude-dependent BLA (Method C) | ref={res['ref_level']:g} -> {verdict}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(fig_dir) / f"n3_amplitude_dependance{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
