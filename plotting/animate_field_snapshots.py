"""Animate the 2Dsnap field movie: one panel per polarization, side by side.

Reads the same two snapshot formats as plot_field_snapshots.py (2Dsnap sensor
or the whole-cell snap_E written next to the level populations), and writes an
mp4 (ffmpeg) or gif (pillow).

  python plotting/animate_field_snapshots.py data/lasing_testing/05b_snapshot
  python plotting/animate_field_snapshots.py <design> --monitor snapshot_1 --intensity
  python plotting/animate_field_snapshots.py <design> --component Ez --fps 15 --gif

Signed field by default: the propagating wavefronts are what a movie shows that
a static grid cannot, and squaring throws away their sign. --intensity gives
|E|² instead.
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from plot_populations import sensor_npz
from plot_field_snapshots import _load_components


def _scale(frames: np.ndarray, intensity: bool, pct: float) -> tuple:
    """(vmin, vmax, cmap) held FIXED across the movie.

    Per-frame autoscaling is the classic field-animation bug: it renormalises
    every frame to its own peak, so a decaying pulse looks like it never decays
    and the noise floor between pulses blooms to full scale. One scale for the
    whole run is the only honest choice.

    The peak is a high percentile, not the max, so that a single hot pixel at
    the source does not compress the entire propagating field into one colour.
    """
    if intensity:
        hi = float(np.percentile(frames, pct))
        return (0.0, hi if hi > 0 else 1.0, "magma")
    hi = float(np.percentile(np.abs(frames), pct))
    hi = hi if hi > 0 else 1.0
    return (-hi, hi, "RdBu_r")


def animate_field_snapshots(design_path: str | Path,
                            monitor: str = "snapshot_1",
                            component: str | None = None,
                            intensity: bool = False,
                            per_panel: bool = False,
                            fps: int = 10,
                            pct: float = 99.5,
                            t_range: tuple[float, float] | None = None,
                            suffix: str = "",
                            gif: bool = False,
                            dpi: int = 110,
                            fig_dir: str | Path | None = None,
                            data_dir: str | Path | None = None) -> Path | None:
    """Movie of the snapshot fields. None if the file holds no field data.

    component : None → one panel per available polarization; else just that one
    intensity : plot |E|² instead of the signed field
    per_panel : give each polarization its own colour scale (default: one
                shared scale, so panel brightness is comparable across them)
    pct       : percentile setting the colour limits (99.5 clips the source cell)
    """
    npz_path = sensor_npz(design_path, monitor, suffix, data_dir)
    fig_dir = fig_dir or os.path.join(design_path, "figures")
    if not os.path.exists(npz_path):
        print(f"[anim] no snapshot file {npz_path} — skipping movie")
        return None

    d = dict(np.load(npz_path, allow_pickle=True))
    comps, times = _load_components(d)
    if not comps or len(times) == 0:
        print(f"[anim] no field snapshots in {npz_path} — skipping movie")
        return None

    if component is not None:
        if component not in comps:
            raise ValueError(f"component {component!r} not in {sorted(comps)}")
        comps = {component: comps[component]}

    keep = np.arange(len(times))
    if t_range:
        lo = t_range[0] if t_range[0] is not None else times[0]
        hi = t_range[1] if t_range[1] is not None else times[-1]
        keep = np.where((times >= lo) & (times <= hi))[0]
        if len(keep) == 0:
            raise ValueError(f"no snapshots in t_range {t_range} "
                             f"(times: {times[0]:.1f}–{times[-1]:.1f})")

    # abs() first: snap fields are real, but a dft-sourced array is complex
    data = {c: (np.abs(v[keep]) ** 2 if intensity else np.asarray(v[keep]).real)
            for c, v in comps.items()}
    names = list(data)
    nx, ny = data[names[0]].shape[1], data[names[0]].shape[2]
    has_xy = ("x" in d and len(d["x"]) == nx and "y" in d and len(d["y"]) == ny)
    x = np.asarray(d["x"]) if has_xy else np.arange(nx)
    y = np.asarray(d["y"]) if has_xy else np.arange(ny)
    ext = [float(y[0]), float(y[-1]), float(x[0]), float(x[-1])]
    unit = "µm" if has_xy else "px"   # 2Dsnap saves no coords; don't fake units

    # Default is ONE scale across panels, so the polarizations stay comparable:
    # in these runs Ey carries the signal and Ex is far weaker, and a per-panel
    # scale would blow Ex's noise up to look as strong as Ey's field.
    # --per-panel-scale trades that away when the weak component is the point.
    shared = None if per_panel else _scale(
        np.stack(list(data.values())), intensity, pct)

    fig, axes = plt.subplots(1, len(names), figsize=(4.6 * len(names) + 1.2, 4.8),
                             squeeze=False)
    axes = axes[0]
    ims = []
    for ax, c in zip(axes, names):
        vmin, vmax, cmap = shared or _scale(data[c], intensity, pct)
        im = ax.imshow(data[c][0].T, origin="lower", extent=ext, aspect="equal",
                       cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"|{c}|²" if intensity else c)
        ax.set_xlabel(f"y ({unit})")
        ims.append(im)
        if per_panel:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    axes[0].set_ylabel(f"x ({unit})")
    if not per_panel:
        fig.colorbar(ims[-1], ax=axes, fraction=0.035, pad=0.02,
                     label="|E|²" if intensity else "E (signed)")
    sup = fig.suptitle("")

    def frame(i):
        for im, c in zip(ims, names):
            im.set_data(data[c][i].T)
        sup.set_text(f"{monitor}   t = {times[keep[i]]:.2f}   "
                     f"({i + 1}/{len(keep)})")
        return (*ims, sup)

    tag = f"_{component}" if component else ""
    tag += "_I" if intensity else ""
    tag += "_pp" if per_panel else ""
    ext_ = "gif" if gif else "mp4"
    out = Path(fig_dir) / f"field_movie_{monitor}{tag}{suffix}.{ext_}"
    out.parent.mkdir(parents=True, exist_ok=True)

    anim = animation.FuncAnimation(fig, frame, frames=len(keep), blit=False)
    if gif:
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    else:
        anim.save(out, writer=animation.FFMpegWriter(
            fps=fps, bitrate=-1,
            # yuv420p + even dimensions, else the file will not play in
            # browsers/Discord despite being a valid mp4
            extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]),
            dpi=dpi)
    plt.close(fig)
    return out


# ====================================================================== __main__
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Animate 2Dsnap field snapshots")
    ap.add_argument("design_path", help="design folder (holds simulation_gpumeep/)")
    ap.add_argument("--monitor", default="snapshot_1",
                    help="sensor name (default snapshot_1; use pop_monitor for "
                         "the whole-cell snap_E field)")
    ap.add_argument("--component", default=None, choices=["Ex", "Ey", "Ez"],
                    help="single component (default: one panel per polarization)")
    ap.add_argument("--intensity", action="store_true",
                    help="plot |E|² instead of the signed field")
    ap.add_argument("--per-panel-scale", action="store_true",
                    help="one colour scale per polarization instead of a shared "
                         "one (use when the weak component is what matters)")
    ap.add_argument("--fps", type=int, default=10, help="frames per second")
    ap.add_argument("--pct", type=float, default=99.5,
                    help="percentile for the colour limits (default 99.5)")
    ap.add_argument("--snap-t1", type=float, default=None, help="window start")
    ap.add_argument("--snap-t2", type=float, default=None, help="window end")
    ap.add_argument("--suffix", default="", help="run suffix (<monitor>_<suffix>.npz)")
    ap.add_argument("--gif", action="store_true", help="write a gif instead of mp4")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--data-dir", default=None,
                    help="backend output dir, e.g. <design>/simulation_meep. "
                         "Only needed when a design has output from more than "
                         "one backend and the auto-pick chooses the wrong one.")
    args = ap.parse_args()

    t_range = ((args.snap_t1, args.snap_t2)
               if args.snap_t1 is not None or args.snap_t2 is not None else None)
    out = animate_field_snapshots(
        args.design_path, monitor=args.monitor, component=args.component,
        intensity=args.intensity, per_panel=args.per_panel_scale,
        fps=args.fps, pct=args.pct, t_range=t_range, suffix=args.suffix,
        gif=args.gif, dpi=args.dpi, data_dir=args.data_dir)
    print(f"[anim] movie → {out}", flush=True)
