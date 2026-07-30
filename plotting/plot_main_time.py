"""Single-entry driver for the TIME-domain plots — the plot_main.py of memory.

plot_main.py characterizes a reservoir with the n1–n7 methods, every one of
which collapses a run to scalars (THD, IPC, rank …). Nothing there shows the
time axis, and for the memory designs the time axis IS the measurement: whether
an input symbol is still legible in the output some symbols later.

So this driver plots what happened during the run instead of what it scored:

  * input vs output over time      — plot_output_over_time (always, if a
                                     point-sensor snapshot exists)
  * dye level populations vs time  — plot_populations (runs with a pump)
  * |E|² field snapshots           — plot_field_snapshots (runs that saved a
                                     2D snapshot monitor)

  python plotting/plot_main_time.py --path data/memory_testing/mem_R0.5_p150_s0
  python plotting/plot_main_time.py --path data/memory_testing --all
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from characterization.class_reservoir_validator import Validator
from plot_output_over_time import plot_output_over_time, find_snapshot
from plot_populations import plot_population_timeseries, plot_population_snapshots
from plot_field_snapshots import plot_field_snapshots


class PlotMainTime:
    """Discover a design's time-domain data (local → mirrors) and plot it all."""

    def __init__(self, design_path: str, fig_dir: str | None = None,
                 component: str = "Ey", t_max: float | None = None):
        self.path = Path(design_path)
        self.fig_dir = Path(fig_dir) if fig_dir else self.path / "figures"
        self.component = component
        self.t_max = t_max
        self.saved: list[Path] = []

    # ------------------------------------------------------------------ helpers
    def _data_dir(self) -> Path | None:
        """The mirror that actually holds this design's simulation_* output.

        Same problem the Validator solves for datasets/: the Nextcloud checkout
        versions only simulation_data.json, so the .npz payloads usually sit on
        Orion. Resolved off the point-sensor snapshot, which every run writes.
        """
        snap = find_snapshot(self.path)
        return snap.parent.parent if snap else None

    def _cfg(self) -> dict:
        return json.load(open(self.path / "simulation_data.json"))

    def _try(self, label, fn, *args, **kw):
        """Run one plot; a design missing that sensor is normal, not an error."""
        try:
            out = fn(*args, **kw)
        except FileNotFoundError:
            print(f"[plot_main_time] {label}: no data — skipped", flush=True)
            return
        except Exception as e:
            print(f"[plot_main_time] {label} FAILED: {e}", flush=True)
            return
        if out is None:
            print(f"[plot_main_time] {label}: nothing to plot — skipped", flush=True)
            return
        self.saved.append(Path(out))
        print(f"[plot_main_time] {out}", flush=True)

    # ----------------------------------------------------------------- dispatch
    def run(self) -> list[Path]:
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.saved = []

        data_dir = self._data_dir()
        if data_dir is None:
            print(f"[plot_main_time] {self.path.name}: no point_snap.npz in any "
                  f"mirror — nothing to plot", flush=True)
            return []
        if Path(data_dir).resolve() != (self.path / data_dir.name).resolve():
            print(f"[plot_main_time] data ← {data_dir}", flush=True)

        # --- input vs output over time (the reason this driver exists) ---
        self._try("output_over_time", plot_output_over_time, self.path,
                  fig_dir=self.fig_dir, component=self.component, t_max=self.t_max)

        # --- dye populations: only runs with a pump have a pop_monitor ---
        cfg = self._cfg()
        has_pump = any(isinstance(v, dict) and v.get("source_type") == "continuous"
                       for v in cfg.values())
        if has_pump:
            self._try("populations", plot_population_timeseries, self.path,
                      fig_dir=self.fig_dir, data_dir=data_dir)
            self._try("population_snapshots", plot_population_snapshots, self.path,
                      fig_dir=self.fig_dir, data_dir=data_dir)

        # --- field snapshots: whichever 2D snapshot monitor the JSON defined ---
        for name, v in cfg.items():
            if isinstance(v, dict) and "snap" in str(v.get("sensor_type", "")).lower():
                self._try(f"field_snapshots[{name}]", plot_field_snapshots,
                          self.path, monitor=name, component=self.component,
                          fig_dir=self.fig_dir, data_dir=data_dir)

        return self.saved


def design_dirs(root: str | Path) -> list[Path]:
    """`root` itself if it is a design, else every design directly under it."""
    root = Path(root)
    if (root / "simulation_data.json").exists():
        return [root]
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "simulation_data.json").exists())


# ====================================================================== __main__
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Plot every time-domain result")
    ap.add_argument("--path", required=True,
                    help="design dir, or a group dir with --all")
    ap.add_argument("--all", action="store_true",
                    help="treat --path as a group and plot every design under it")
    ap.add_argument("--fig-dir", default=None, help="default: <design>/figures")
    ap.add_argument("--component", default="Ey",
                    help="stored polarization to plot (default Ey)")
    ap.add_argument("--t-max", type=float, default=None,
                    help="truncate the time axis")
    a = ap.parse_args()

    targets = design_dirs(a.path) if a.all else [Path(a.path)]
    total = 0
    for d in targets:
        if len(targets) > 1:
            print(f"\n=== {d.name}", flush=True)
        pm = PlotMainTime(str(d), fig_dir=a.fig_dir, component=a.component,
                          t_max=a.t_max)
        total += len(pm.run())
    print(f"\n[done] {total} figures over {len(targets)} design(s)", flush=True)
