"""Plot for single_source_sweep.py's dataset — the input/output amplitude sweep.

Every source strip is driven at the same amplitude `level`; the sweep records
‖output‖ and gain = ‖out‖/level. A LINEAR device gives ‖out‖ ∝ level (slope 1 on
log-log) and a flat gain curve; any bend is the device's own amplitude
dependence — saturation (gain falling) or threshold-like buildup (gain rising).

Left panel: ‖out‖ vs drive on log-log against the linear reference anchored at
the weakest level — deviation from that dashed line IS the nonlinearity.
Right panel: gain normalised to the small-signal value, so a flat line at 100%
means linear and the scale reads directly as compression/expansion.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _sweep_arrays(res: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(drive, ‖out‖, gain) sorted by drive, from the sweep npz keys
    {levels, out_norm, gain} or the E_in/E_out aliases."""
    if "levels" in res:
        x = np.asarray(res["levels"], dtype=float).reshape(-1)
        y = np.asarray(res["out_norm"], dtype=float).reshape(-1)
    elif "E_in" in res:
        x = np.asarray(res["E_in"], dtype=float).reshape(-1)
        y = np.asarray(res["E_out"], dtype=float).reshape(-1)
    else:
        raise KeyError("sweep result needs levels/out_norm (or E_in/E_out)")
    g = (np.asarray(res["gain"], dtype=float).reshape(-1) if "gain" in res
         else y / np.where(x != 0, x, np.nan))
    o = np.argsort(x)
    return x[o], y[o], g[o]


def plot_single_source_amplitude_sweep(res: dict, fig_dir: str | Path,
                                       suffix: str = "") -> Path:
    x, y, gain = _sweep_arrays(res)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))

    # ---------- ‖out‖ vs drive ----------
    ax1.loglog(x, y, "o-", color="C0", label="‖out‖")
    ax1.loglog(x, y[0] * (x / x[0]), "k--", lw=1, label="linear (slope 1)")
    ax1.set_xlabel("drive amplitude per strip")
    ax1.set_ylabel("‖output‖")
    ax1.set_title("input–output response")
    ax1.legend(fontsize=8)

    # ---------- gain vs drive ----------
    g0 = gain[0] if np.isfinite(gain[0]) and gain[0] != 0 else 1.0
    rel = 100.0 * gain / g0
    ax2.semilogx(x, rel, "s-", color="C1")
    ax2.axhline(100.0, color="gray", lw=0.8, ls=":")
    for xv, rv in zip(x, rel):
        ax2.annotate(f"{rv:.0f}%", (xv, rv), textcoords="offset points",
                     xytext=(0, 4), ha="center", fontsize=7.5)
    ax2.set_xlabel("drive amplitude per strip")
    ax2.set_ylabel("gain / small-signal gain [%]")
    ax2.set_title("gain compression")

    dev = float(np.nanmax(np.abs(rel - 100.0)))
    verdict = "LINEAR" if dev < 1.0 else f"NONLINEAR (max {dev:.0f}% gain change)"
    fig.suptitle(f"Single-source amplitude sweep | {x[0]:g} … {x[-1]:g} "
                 f"({len(x)} levels) -> {verdict}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(fig_dir) / f"single_source_sweep{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out
