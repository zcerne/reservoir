"""Plot dye-level populations from pop_monitor.npz output.

  pop_monitor.npz keys: N(t, level, x, y), times(t), levels(4), x, y, snap_interval
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def plot_population_snapshots(npz_path: str | Path, fig_dir: str | Path,
                                 n: int = 3, level: int = 2,
                                 t_range: tuple[float, float] | None = None,
                                 suffix: str = "") -> Path:
    """Plot spatial population snapshots on an n×n grid.

    Selects n² evenly-spaced time snapshots across t_range (or full range) and
    arranges them in an n×n grid of colormesh panels.

    Parameters
    ----------
    npz_path : path to pop_monitor.npz
    fig_dir  : output directory for the saved PNG
    n        : grid size (n×n panels, default 3 → 9 snapshots)
    level    : which dye level to plot (0=N1, 1=N2, 2=N3, 3=N4; default 2=N3)
    t_range  : optional (t_min, t_max) to restrict the snapshot window
    suffix   : appended to the output filename
    """
    d = dict(np.load(npz_path, allow_pickle=True))
    N_all = np.asarray(d["N"])           # (T, 4, nx, ny)
    times = np.asarray(d["times"])       # (T,)
    x = np.asarray(d["x"])
    y = np.asarray(d["y"])
    levels = list(d["levels"])
    level_name = levels[level]

    # --- select n² evenly-spaced snapshots ---
    t_lo = t_range[0] if t_range else times[0]
    t_hi = t_range[1] if t_range else times[-1]
    mask = (times >= t_lo) & (times <= t_hi)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        raise ValueError(f"no snapshots in t_range {t_range} (times: {times[0]:.1f}–{times[-1]:.1f})")
    step = max(1, len(idxs) // (n * n))
    pick = idxs[::step][:n * n]
    # fill shortfall from the end
    if len(pick) < n * n:
        pick = np.sort(np.unique(np.concatenate([pick, idxs[-(n * n - len(pick)):]])))

    # --- plot ---
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    vmin, vmax = float(N_all[pick, level].min()), float(N_all[pick, level].max())
    for ax, ti in zip(axes.flat, pick):
        ax.pcolormesh(x, y, N_all[ti, level].T, shading="auto",
                      vmin=vmin, vmax=vmax, cmap="inferno", rasterized=True)
        ax.set_title(f"t = {times[ti]:.1f}")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    # hide unused panels
    for ax in axes.flat[len(pick):]:
        ax.set_visible(False)

    fig.suptitle(f"Concentration snapshots — {level_name}  "
                 f"(t = {times[pick[0]]:.1f} – {times[pick[-1]]:.1f})", fontsize=11)
    cbar = fig.colorbar(axes.flat[0].collections[0], ax=axes, shrink=0.92,
                        label=f"{level_name} population")
    fig.tight_layout()

    out = Path(fig_dir) / f"pop_snapshots_{level_name}{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_population_timeseries(npz_path: str | Path, fig_dir: str | Path,
                                  sum_over: str = "xy", suffix: str = "") -> Path:
    """Plot total population of each dye level vs time (1D).

    Parameters
    ----------
    npz_path : path to pop_monitor.npz
    fig_dir  : output directory for the saved PNG
    sum_over : dimensions to sum over — "xy" (default, full spatial sum),
               "x" (integrate over x, keep y), "y" (integrate over y, keep x)
    suffix   : appended to the output filename
    """
    d = dict(np.load(npz_path, allow_pickle=True))
    N_all = np.asarray(d["N"])           # (T, 4, nx, ny)
    times = np.asarray(d["times"])       # (T,)
    levels = list(d["levels"])

    # --- sum over spatial dims ---
    if sum_over == "xy":
        summed = N_all.sum(axis=(2, 3))   # (T, 4)
        ylabel = "total population  Σ_{x,y} N(x,y,t)"
    elif sum_over == "x":
        summed = N_all.sum(axis=2)         # (T, 4, ny)
        ylabel = "population  Σ_x N(x,y,t)"
    elif sum_over == "y":
        summed = N_all.sum(axis=3)         # (T, 4, nx)
        ylabel = "population  Σ_y N(x,y,t)"
    else:
        raise ValueError(f"sum_over must be 'xy', 'x', or 'y', got '{sum_over}'")

    colors = ["C0", "C1", "C2", "C3"]

    if sum_over == "xy":
        # --- simple 1D lines ---
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for i, (name, c) in enumerate(zip(levels, colors)):
            ax.plot(times, summed[:, i], color=c, lw=1.5, label=name)
        ax.set_xlabel("time"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.set_title("Dye-level populations over time")
        ax.grid(alpha=.3)
    else:
        # --- timeseries heatmaps (T, level*spatial) — 4-panel ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for ax, i, name in zip(axes.flat, range(4), levels):
            im = ax.pcolormesh(times, np.arange(summed.shape[2]), summed[:, i].T,
                               shading="auto", cmap="inferno", rasterized=True)
            ax.set_title(name)
            ax.set_xlabel("time"); ax.set_ylabel("spatial index")
            fig.colorbar(im, ax=ax)
        fig.suptitle("Concentration timeseries (spatial profile unfolded)")

    fig.tight_layout()
    out = Path(fig_dir) / f"pop_timeseries{suffix}.png"
    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ====================================================================== __main__
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Plot population snapshots + timeseries")
    ap.add_argument("npz", help="path to pop_monitor.npz")
    ap.add_argument("--fig-dir", default=".", help="output directory for figures")
    ap.add_argument("--n", type=int, default=3, help="grid size for snapshots (n×n)")
    ap.add_argument("--level", type=int, default=2, help="dye level for snapshots (0-3)")
    ap.add_argument("--snap-t1", type=float, default=None, help="snapshot window start")
    ap.add_argument("--snap-t2", type=float, default=None, help="snapshot window end")
    ap.add_argument("--sum-over", default="xy", choices=["xy", "x", "y"],
                    help="dimensions to sum over for timeseries")
    args = ap.parse_args()

    t_range = (args.snap_t1, args.snap_t2) if args.snap_t1 is not None or args.snap_t2 is not None else None
    s1 = plot_population_snapshots(args.npz, args.fig_dir, n=args.n, level=args.level,
                                       t_range=t_range)
    s2 = plot_population_timeseries(args.npz, args.fig_dir, sum_over=args.sum_over)
    print(f"[pop] snapshots → {s1}", flush=True)
    print(f"[pop] timeseries → {s2}", flush=True)
