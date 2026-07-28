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


def sim_dir(design_path: str | Path) -> str:
    """Backend output dir — gpumeep if present, else the MEEP one."""
    for sub in ("simulation_gpumeep", "simulation"):
        p = os.path.join(design_path, sub)
        if os.path.isdir(p):
            return p
    return os.path.join(design_path, "simulation_gpumeep")


def sensor_npz(design_path: str | Path, key: str, suffix: str = "") -> str:
    """Path to sensor <key>'s npz. Mirrors SimpleSim's sim_data.tagged(): no
    suffix → "<key>.npz", suffix "2" → "<key>_2.npz". Falls back to the legacy
    trailing-underscore spelling ("<key>_.npz") for data written before that
    cleanup, so older runs still plot."""
    d = sim_dir(design_path)
    s = str(suffix).strip("_")
    clean = os.path.join(d, f"{key}_{s}.npz" if s else f"{key}.npz")
    if os.path.exists(clean):
        return clean
    legacy = os.path.join(d, f"{key}_{suffix}.npz")
    return legacy if os.path.exists(legacy) else clean


def pick_snapshots(times: np.ndarray, n: int,
                    t_range: tuple[float, float] | None) -> np.ndarray:
    """Indices of n² evenly-spaced snapshots inside t_range."""
    t_lo = t_range[0] if t_range and t_range[0] is not None else times[0]
    t_hi = t_range[1] if t_range and t_range[1] is not None else times[-1]
    idxs = np.where((times >= t_lo) & (times <= t_hi))[0]
    if len(idxs) == 0:
        raise ValueError(f"no snapshots in t_range {t_range} "
                         f"(times: {times[0]:.1f}–{times[-1]:.1f})")
    step = max(1, len(idxs) // (n * n))
    pick = idxs[::step][:n * n]
    if len(pick) < n * n:
        pick = np.sort(np.unique(np.concatenate([pick, idxs[-(n * n - len(pick)):]])))
    return pick


def grid_figure(maps: np.ndarray, times: np.ndarray, pick: np.ndarray,
                 x: np.ndarray, y: np.ndarray, n: int, title: str,
                 cbar_label: str, out: Path, cmap: str = "inferno",
                 norm=None) -> Path:
    """Draw picked 2-D maps on an n×n grid with one shared colorbar.

    ``maps`` is indexed by the ORIGINAL snapshot index (maps[pick[k]]), stored
    (nx, ny) so it is transposed for pcolormesh's (row, col) = (y, x).
    """
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n), layout="constrained")
    axes = np.atleast_1d(axes).reshape(-1)
    kw = dict(shading="auto", cmap=cmap, rasterized=True)
    if norm is not None:
        kw["norm"] = norm
    else:
        kw["vmin"] = float(np.min(maps[pick]))
        kw["vmax"] = float(np.max(maps[pick]))
    for ax, ti in zip(axes, pick):
        ax.pcolormesh(x, y, maps[ti].T, **kw)
        ax.set_title(f"t = {times[ti]:.1f}")
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(pick):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11)
    fig.colorbar(axes[0].collections[0], ax=axes.tolist(), shrink=0.92,
                 label=cbar_label)

    os.makedirs(out.parent, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_population_snapshots(design_path: str | Path,
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
    npz_path = sensor_npz(design_path, "pop_monitor", suffix)
    fig_dir = os.path.join(design_path, "figures")
    d = dict(np.load(npz_path, allow_pickle=True))
    N_spatial, times, x, y, levels = _resolve_snapshots(d)
    if N_spatial is None or len(times) == 0:
        print(f"[pop] no spatial snapshot data in {npz_path} — skipping grid plot")
        return None
    level_name = levels[level]

    pick = pick_snapshots(times, n, t_range)
    return grid_figure(
        N_spatial[:, level], times, pick, x, y, n,
        title=(f"Population snapshots — {level_name}  "
               f"(t = {times[pick[0]]:.1f} – {times[pick[-1]]:.1f})"),
        cbar_label=f"{level_name} population",
        out=Path(fig_dir) / f"pop_snapshots_{level_name}{suffix}.png")


def plot_population_timeseries(design_dir: str | Path,
                                suffix: str = "") -> Path:
    """Plot total population of each dye level vs time (1D lines).

    Works with both old (full-spatial) and new (pre-summed) formats.
    """
    npz_path = sensor_npz(design_dir, "pop_monitor", suffix)
    fig_dir = os.path.join(design_dir, "figures")
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
    ap.add_argument("design_path", help="design folder (holds simulation_gpumeep/)")
    ap.add_argument("--n", type=int, default=3, help="grid size for snapshots (n×n)")
    ap.add_argument("--level", type=int, default=2, help="dye level for snapshots (0-3)")
    ap.add_argument("--snap-t1", type=float, default=None, help="snapshot window start")
    ap.add_argument("--snap-t2", type=float, default=None, help="snapshot window end")
    ap.add_argument("--suffix", default="", help="run suffix (pop_monitor_<suffix>.npz)")
    args = ap.parse_args()

    t_range = (args.snap_t1, args.snap_t2) if args.snap_t1 is not None or args.snap_t2 is not None else None
    s1 = plot_population_snapshots(args.design_path, n=args.n, level=args.level,
                                    t_range=t_range, suffix=args.suffix)
    s2 = plot_population_timeseries(args.design_path, suffix=args.suffix)
    print(f"[pop] snapshots  → {s1}", flush=True)
    print(f"[pop] timeseries → {s2}", flush=True)
