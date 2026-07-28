"""Plot dye-level populations from pop_monitor.npz output.

Supports both formats:
  New (SimpleSim): t, N(n_steps,n_levels), levels, [+ snap_t, snap_N, x, y]
  Old (legacy):    times, N(n_snaps,4,nx,ny), levels, x, y, snap_interval
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def _resolve_snapshots(d: dict):
    """Return (N_spatial, t_snap, x, y, levels) for snapshot-grid plotting,
    or (None, ...) if no spatial data is present."""
    levels = list(d.get("levels", ["N1", "N2", "N3", "N4"]))
    # new format: separate snap_N + snap_t
    if "snap_N" in d and "snap_t" in d:
        return (np.asarray(d["snap_N"]), np.asarray(d["snap_t"]),
                np.asarray(d.get("x", [])), np.asarray(d.get("y", [])), levels)
    # old format: N is already (T, n_levels, nx, ny) at snap intervals
    N = np.asarray(d.get("N", []))
    if N.ndim == 4:
        times = np.asarray(d.get("times", d.get("t", [])))
        return (N, times,
                np.asarray(d.get("x", [])), np.asarray(d.get("y", [])), levels)
    return (None, np.array([]), np.array([]), np.array([]), levels)


def _resolve_timeseries(d: dict):
    """Return (N_summed, t, levels) as (n_pts, n_levels) totals."""
    levels = list(d.get("levels", ["N1", "N2", "N3", "N4"]))
    N = np.asarray(d["N"])
    if N.ndim == 2:
        # new format: already summed totals at every step
        t = np.asarray(d.get("t", d.get("times", [])))
        return N, t, levels
    # old format: full spatial N, sum over spatial axes
    t = np.asarray(d.get("times", d.get("t", [])))
    return N.sum(axis=(2, 3)), t, levels


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_population_snapshots(npz_path: str | Path, fig_dir: str | Path,
                               n: int = 3, level: int = 2,
                               t_range: tuple[float, float] | None = None,
                               suffix: str = "") -> Path | None:
    """Plot spatial population snapshots on an n×n grid.

    Requires full spatial data (``snap_N`` in new format, or 4-D ``N`` in
    old format).  Returns None if the file only contains per-level totals.

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
    N_spatial, times, x, y, levels = _resolve_snapshots(d)
    if N_spatial is None or len(times) == 0:
        print(f"[pop] no spatial snapshot data in {npz_path} — skipping grid plot")
        return None
    level_name = levels[level]
    n_lev = N_spatial.shape[1]

    # --- select n² evenly-spaced snapshots ---
    t_lo = t_range[0] if t_range else times[0]
    t_hi = t_range[1] if t_range else times[-1]
    mask = (times >= t_lo) & (times <= t_hi)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        raise ValueError(f"no snapshots in t_range {t_range} (times: {times[0]:.1f}–{times[-1]:.1f})")
    step = max(1, len(idxs) // (n * n))
    pick = idxs[::step][:n * n]
    if len(pick) < n * n:
        pick = np.sort(np.unique(np.concatenate([pick, idxs[-(n * n - len(pick)):]])))

    # --- plot ---
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    vmin = float(N_spatial[pick, level].min())
    vmax = float(N_spatial[pick, level].max())
    for ax, ti in zip(axes.flat, pick):
        ax.pcolormesh(x, y, N_spatial[ti, level].T, shading="auto",
                      vmin=vmin, vmax=vmax, cmap="inferno", rasterized=True)
        ax.set_title(f"t = {times[ti]:.1f}")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes.flat[len(pick):]:
        ax.set_visible(False)

    fig.suptitle(f"Population snapshots — {level_name}  "
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
                                suffix: str = "") -> Path:
    """Plot total population of each dye level vs time (1D lines).

    Works with both old (full-spatial) and new (pre-summed) formats.
    """
    d = dict(np.load(npz_path, allow_pickle=True))
    summed, times, levels = _resolve_timeseries(d)

    colors = ["C0", "C1", "C2", "C3"]
    n_lev = summed.shape[1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i in range(min(n_lev, len(levels))):
        ax.plot(times, summed[:, i], color=colors[i % len(colors)],
                lw=1.5, label=levels[i])
    ax.set_xlabel("time")
    ax.set_ylabel("total population  Σ N(t)")
    ax.legend(fontsize=8)
    ax.set_title("Dye-level populations over time")
    ax.grid(alpha=.3)
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
    args = ap.parse_args()

    t_range = (args.snap_t1, args.snap_t2) if args.snap_t1 is not None or args.snap_t2 is not None else None
    s1 = plot_population_snapshots(args.npz, args.fig_dir, n=args.n, level=args.level,
                                    t_range=t_range)
    s2 = plot_population_timeseries(args.npz, args.fig_dir)
    print(f"[pop] snapshots → {s1}", flush=True)
    print(f"[pop] timeseries → {s2}", flush=True)
