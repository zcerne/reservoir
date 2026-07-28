"""Plot instantaneous field intensity |E|² snapshots on an n×n grid.

Reads whichever snapshot format the named monitor saved:
  2Dsnap sensor     → {t, Ex, Ey, Ez}, each (n_snaps, nx, ny)
  population sensor → {snap_t, snap_E (n_snaps, 3, nx, ny), x, y}  (the
                      whole-cell E saved alongside the level populations
                      when the JSON sets "snap_interval")

  python plotting/plot_field_snapshots.py data/lasing_testing/01_basic_test
  python plotting/plot_field_snapshots.py <design> --monitor pop_monitor --log
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

# snapshot-grid machinery is shared with the population snapshots so both
# figure types stay laid out identically
from plot_populations import sensor_npz, pick_snapshots, grid_figure


def _load_components(npz: dict) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """(components, times) from either snapshot format; ({}, []) if neither."""
    if "Ex" in npz and "t" in npz:
        return ({c: np.asarray(npz[c]) for c in ("Ex", "Ey", "Ez") if c in npz},
                np.asarray(npz["t"]))
    if "snap_E" in npz and "snap_t" in npz:
        E = np.asarray(npz["snap_E"])
        names = [str(s) for s in npz.get("snap_E_comps", ["Ex", "Ey", "Ez"])]
        return ({nm: E[:, i] for i, nm in enumerate(names)},
                np.asarray(npz["snap_t"]))
    return {}, np.array([])


def plot_field_snapshots(design_path: str | Path,
                         monitor: str = "snapshot_1",
                         n: int = 3,
                         component: str | None = None,
                         t_range: tuple[float, float] | None = None,
                         suffix: str = "",
                         log: bool = False,
                         dyn_range: float = 1e6,
                         fig_dir: str | Path | None = None,
                         data_dir: str | Path | None = None) -> Path | None:
    """Intensity |E|² snapshots on an n×n grid. None if the file has no fields.

    Parameters
    ----------
    monitor   : sensor name, i.e. <monitor>_<suffix>.npz in the backend dir
    n         : grid size (n×n panels, default 3 → 9 snapshots)
    component : None → total |E|² = |Ex|²+|Ey|²+|Ez|²; else "Ex"/"Ey"/"Ez"
                for that component alone
    t_range   : optional (t_min, t_max) to restrict the snapshot window
    log       : log-colour the intensity (useful once a pulse has decayed
                orders of magnitude below its peak)
    dyn_range : log-scale span below the peak. Clamping matters: away from the
                pulse the field is float underflow noise (~1e-300), so an
                unclamped LogNorm spends ~300 decades on nothing and washes
                every panel out.
    """
    npz_path = sensor_npz(design_path, monitor, suffix, data_dir)
    fig_dir = fig_dir or os.path.join(design_path, "figures")
    if not os.path.exists(npz_path):
        print(f"[field] no snapshot file {npz_path} — skipping intensity grid")
        return None

    d = dict(np.load(npz_path, allow_pickle=True))
    comps, times = _load_components(d)
    if not comps or len(times) == 0:
        print(f"[field] no field snapshots in {npz_path} — skipping intensity grid")
        return None

    if component is not None:
        if component not in comps:
            raise ValueError(f"component {component!r} not in {sorted(comps)}")
        # abs() first: snap fields are real, but a dft-sourced array is complex
        inten = np.abs(comps[component]) ** 2
        label = f"|{component}|²"
    else:
        inten = sum(np.abs(v) ** 2 for v in comps.values())
        label = "|E|²  (" + "+".join(f"|{c}|²" for c in comps) + ")"

    # grid coords: saved with the population format, otherwise pixel index
    nx, ny = inten.shape[1], inten.shape[2]
    x = np.asarray(d["x"]) if "x" in d and len(d["x"]) == nx else np.arange(nx)
    y = np.asarray(d["y"]) if "y" in d and len(d["y"]) == ny else np.arange(ny)

    pick = pick_snapshots(times, n, t_range)
    norm, cmap = None, "magma"
    if log:
        from matplotlib.colors import LogNorm
        vmax = float(np.max(inten[pick]))
        if vmax > 0:
            norm = LogNorm(vmin=vmax / dyn_range, vmax=vmax)
            # Regions the pulse hasn't reached are exactly 0. LogNorm MASKS
            # non-positive values rather than clipping them, so they render as
            # "bad" (transparent → white page) — the loudest thing on the page
            # for the emptiest part of the cell. set_bad covers those; set_under
            # covers the merely-below-vmin ones.
            cmap = matplotlib.colormaps["magma"].copy()
            cmap.set_under(cmap(0.0))
            cmap.set_bad(cmap(0.0))

    tag = f"_{component}" if component else ""
    return grid_figure(
        inten, times, pick, x, y, n,
        title=(f"Intensity snapshots — {monitor}  "
               f"(t = {times[pick[0]]:.1f} – {times[pick[-1]]:.1f})"),
        cbar_label=label,
        out=Path(fig_dir) / f"field_snapshots_{monitor}{tag}{suffix}.png",
        cmap=cmap, norm=norm)


# ====================================================================== __main__
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Intensity |E|² snapshots on an n×n grid")
    ap.add_argument("design_path", help="design folder (holds simulation_gpumeep/)")
    ap.add_argument("--monitor", default="snapshot_1",
                    help="sensor name (default snapshot_1; use pop_monitor for "
                         "the whole-cell snap_E field)")
    ap.add_argument("--n", type=int, default=3, help="grid size (n×n)")
    ap.add_argument("--component", default=None, choices=["Ex", "Ey", "Ez"],
                    help="single component (default: total |E|²)")
    ap.add_argument("--snap-t1", type=float, default=None, help="window start")
    ap.add_argument("--snap-t2", type=float, default=None, help="window end")
    ap.add_argument("--suffix", default="", help="run suffix (<monitor>_<suffix>.npz)")
    ap.add_argument("--log", action="store_true", help="log-scale the intensity")
    ap.add_argument("--dyn-range", type=float, default=1e6,
                    help="log-scale span below the peak (default 1e6 = 6 decades)")
    args = ap.parse_args()

    t_range = ((args.snap_t1, args.snap_t2)
               if args.snap_t1 is not None or args.snap_t2 is not None else None)
    out = plot_field_snapshots(args.design_path, monitor=args.monitor, n=args.n,
                               component=args.component, t_range=t_range,
                               suffix=args.suffix, log=args.log,
                               dyn_range=args.dyn_range)
    print(f"[field] intensity → {out}", flush=True)
