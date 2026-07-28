"""PlotSimulation — find a design's simulation output (workbox or Orion) and
plot every sensor in it: DFT spectra, flux, field snapshots, populations,
far-field maps.

Complements plot_main.py, which plots the CHARACTERIZATION datasets
(datasets/*.npz produced by data_gen/). This one plots the RAW simulation
output (simulation_<backend>/*.npz written by run.py's sensors).

  python plotting/plot_simulation.py data/lasing_testing/01_basic_test
  python plotting/plot_simulation.py <design> --backend meep --suffix v2 --log

Data discovery mirrors characterization/class_reservoir_validator.py: prefer
the local copy, fall back to the Orion mount (~/Orion/resevoir/<path>), so the
same command works whether the run happened on workbox or on smaug/orion.
Figures always go to the LOCAL design's figures/ dir, so plotting remote data
doesn't write across the mount.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_populations import (plot_population_snapshots, plot_population_timeseries,
                              pick_snapshots, grid_figure)
from plot_field_snapshots import plot_field_snapshots

#: where a design's data may live, in preference order. Orion's checkout is
#: the same CephFS smaug writes to, so a run launched there appears here.
ORION_ROOT = os.path.expanduser("~/Orion/resevoir")


class PlotSimulation:
    """Plot every sensor npz of one design's simulation output.

    Parameters
    ----------
    design_path : design folder, relative to the repo root (e.g.
        "data/lasing_testing/01_basic_test"). Resolved against the local
        tree first, then the Orion mount.
    backend : "gpumeep" | "meep" | None. None (default) plots every backend
        dir that exists, so a cross-backend run gets both sets of figures.
    suffix : run suffix, matching run.py's --suffix (no suffix -> plain
        <sensor>.npz).
    """

    def __init__(self, design_path: str | Path, backend: str | None = None,
                 suffix: str = "", fig_dir: str | Path | None = None,
                 log: bool = False, n: int = 3):
        self.rel = str(design_path).rstrip("/")
        self.data_dir = self._resolve(self.rel)
        self.backend = backend
        self.suffix = suffix
        self.log = log
        self.n = n
        # figures land locally even when the data came from Orion
        self.fig_dir = Path(fig_dir) if fig_dir else Path(self.rel) / "figures"
        self.saved: list[Path] = []

    # ------------------------------------------------------------- discovery
    @staticmethod
    def _resolve(rel: str) -> Path:
        """Local design dir if it holds simulation output, else the Orion one.
        (Only used for reporting/relative naming — backend_dirs() searches
        BOTH roots, since a design is often half-run in each place.)"""
        for c in (Path(rel), Path(ORION_ROOT) / rel):
            if any(c.glob("simulation*/*.npz")):
                return c
        return Path(rel)

    def backend_dirs(self) -> list[Path]:
        """Backend output dirs holding npz, merged over workbox + Orion.

        A backend present in both roots resolves to the LOCAL copy (same
        preference as characterization/class_reservoir_validator.py); one
        that exists only on Orion is still plotted, so a design run partly
        on smaug and partly here gets a complete set of figures.
        """
        roots = [Path(self.rel), Path(ORION_ROOT) / self.rel]
        found: dict[str, Path] = {}
        for root in roots:
            if not root.is_dir():
                continue
            for d in sorted(root.glob("simulation*")):
                if not d.is_dir() or not any(d.glob("*.npz")):
                    continue
                if self.backend and d.name != f"simulation_{self.backend}":
                    continue
                if d.name not in found:          # first root wins = local
                    found[d.name] = d
                    if root != roots[0]:
                        print(f"[plot_sim] {d.name}: using Orion data ({d})",
                              flush=True)
        return [found[k] for k in sorted(found)]

    #: written alongside a sensor's own npz but not itself plottable —
    #: near2far equivalence currents exist to be replayed (sensor.replay_n2f),
    #: and carry no field/spectrum arrays of their own
    NON_SENSOR_SUFFIXES = ("_n2f_eq",)

    def sensors(self, bdir: Path) -> list[Path]:
        """Sensor npz files in `bdir` matching this run's suffix."""
        s = str(self.suffix).strip("_")
        out = []
        for p in sorted(bdir.glob("*.npz")):
            if p.stem.endswith(self.NON_SENSOR_SUFFIXES):
                continue
            if s and not p.stem.endswith(f"_{s}"):
                continue
            out.append(p)
        return out

    # -------------------------------------------------------------- dispatch
    def run(self) -> list[Path]:
        """Plot every sensor of every selected backend dir.

        With one backend, figures go flat into figures/. With more than one
        they are split into figures/<backend>/, since the shared snapshot/
        population plotters name their output by sensor only and would
        otherwise overwrite each other.
        """
        self.saved = []
        bdirs = self.backend_dirs()
        if not bdirs:
            print(f"[plot_sim] no simulation output for {self.rel} "
                  f"(looked in . and {ORION_ROOT})", flush=True)
            return []
        for bdir in bdirs:
            tag = bdir.name.replace("simulation_", "") or "meep"
            tag = "meep" if tag == "simulation" else tag
            fdir = self.fig_dir / tag if len(bdirs) > 1 else self.fig_dir
            fdir.mkdir(parents=True, exist_ok=True)
            print(f"[plot_sim] {bdir}", flush=True)
            for npz in self.sensors(bdir):
                try:
                    self._plot_one(npz, tag, bdir, fdir)
                except Exception as e:                     # one bad sensor must
                    print(f"[plot_sim]   {npz.name}: FAILED ({type(e).__name__}: {e})",
                          flush=True)                      # not kill the rest
        print(f"[plot_sim] {len(self.saved)} figures -> {self.fig_dir}", flush=True)
        return self.saved

    def _plot_one(self, npz: Path, tag: str, bdir: Path, fdir: Path) -> None:
        """Dispatch one sensor npz by the keys it actually carries."""
        d = np.load(npz, allow_pickle=True)
        keys = set(d.files)
        name = npz.stem
        efields = [c for c in ("Ex", "Ey", "Ez") if c in keys]

        if "E2" in keys and "EH" in keys:                       # near2far
            self._add(self._plot_farfield(d, name, tag, fdir))
        elif "fluxes" in keys or ("freqs" in keys and "flux" in keys):
            self._add(self._plot_flux(d, name, tag, fdir))
        elif "freqs" in keys and efields:                       # *Ddft monitor
            self._add(self._plot_spectrum(d, name, tag, efields, fdir))
        elif "N" in keys:                                       # population
            self._plot_population(name, keys, bdir, fdir)
        elif "t" in keys and efields:                           # *Dsnap
            self._add(plot_field_snapshots(
                self.rel, monitor=self._monitor_key(name), n=self.n,
                suffix=self.suffix, log=self.log, fig_dir=fdir, data_dir=bdir))
        else:
            print(f"[plot_sim]   {npz.name}: unrecognized keys {sorted(keys)[:6]}",
                  flush=True)

    def _monitor_key(self, stem: str) -> str:
        """Sensor key from a filename stem (strip this run's suffix)."""
        s = str(self.suffix).strip("_")
        if s and stem.endswith(f"_{s}"):
            return stem[: -(len(s) + 1)]
        return stem.rstrip("_")

    def _add(self, p: Path | None) -> None:
        if p is not None:
            self.saved.append(Path(p))
            print(f"[plot_sim]   -> {Path(p).name}", flush=True)

    # ----------------------------------------------------------- plot kinds
    def _plot_spectrum(self, d, name: str, tag: str, efields: list[str],
                       fdir: Path) -> Path:
        """DFT monitor: |E(λ)| per component, summed over the monitor line."""
        freqs = np.asarray(d["freqs"])
        lam = 1.0 / np.where(freqs == 0, np.nan, freqs)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for c in efields:
            v = np.asarray(d[c])
            # (n_freq, n_points) -> per-frequency L2 over the line
            amp = np.linalg.norm(np.abs(v), axis=tuple(range(1, v.ndim))) \
                if v.ndim > 1 else np.abs(v)
            if amp.max() > 0:
                ax.plot(lam, amp, lw=1.5, label=c)
        ax.set_xlabel("wavelength (µm)")
        ax.set_ylabel("|E| (L2 over monitor)")
        ax.set_title(f"{name} spectrum — {tag}")
        ax.grid(alpha=.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
        return self._save(fig, f"{name}_spectrum_{tag}", fdir)

    def _plot_flux(self, d, name: str, tag: str, fdir: Path) -> Path:
        freqs = np.asarray(d["freqs"])
        flux = np.asarray(d["fluxes"] if "fluxes" in d.files else d["flux"])
        lam = 1.0 / np.where(freqs == 0, np.nan, freqs)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(lam, flux, lw=1.5, color="C2")
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("wavelength (µm)")
        ax.set_ylabel("flux (Poynting)")
        ax.set_title(f"{name} flux — {tag}")
        ax.grid(alpha=.3)
        return self._save(fig, f"{name}_flux_{tag}", fdir)

    def _plot_farfield(self, d, name: str, tag: str, fdir: Path) -> Path:
        """near2far: far-field |E|² area map."""
        E2 = np.asarray(d["E2"])
        x, y = np.asarray(d["x"]), np.asarray(d["y"])
        lam = float(d["lam"]) if "lam" in d.files else float("nan")
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        # x/y are saved as 2D meshgrids by the sensor; pcolormesh takes either
        m = ax.pcolormesh(x, y, E2, shading="auto", cmap="inferno", rasterized=True)
        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        ax.set_aspect("equal")
        ax.set_title(f"{name} far field |E|²  (λ = {lam:g} µm) — {tag}")
        fig.colorbar(m, ax=ax, label="|E|²")
        return self._save(fig, f"{name}_farfield_{tag}", fdir)

    def _plot_population(self, name: str, keys: set, bdir: Path,
                         fdir: Path) -> None:
        """Population sensor: level timeseries, spatial N snapshots, and the
        whole-cell E snapshots saved alongside them (snap_E)."""
        self._add(plot_population_timeseries(self.rel, suffix=self.suffix,
                                             fig_dir=fdir, data_dir=bdir))
        if "snap_N" in keys:
            self._add(plot_population_snapshots(self.rel, n=self.n,
                                                suffix=self.suffix,
                                                fig_dir=fdir, data_dir=bdir))
        if "snap_E" in keys:
            self._add(plot_field_snapshots(
                self.rel, monitor=self._monitor_key(name), n=self.n,
                suffix=self.suffix, log=self.log, fig_dir=fdir, data_dir=bdir))

    def _save(self, fig, stem: str, fdir: Path) -> Path:
        out = fdir / f"{stem}{'_' + self.suffix.strip('_') if self.suffix else ''}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out


# ====================================================================== __main__
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Plot every sensor of a design's simulation output "
                    "(found locally or on the Orion mount)")
    ap.add_argument("design_path", help="design folder, e.g. data/lasing_testing/01_basic_test")
    ap.add_argument("--backend", default=None, choices=["gpumeep", "meep"],
                    help="only this backend's output (default: every one present)")
    ap.add_argument("--suffix", default="", help="run suffix (as passed to run.py)")
    ap.add_argument("--fig-dir", default=None, help="output dir (default <design>/figures)")
    ap.add_argument("--n", type=int, default=3, help="snapshot grid size (n×n)")
    ap.add_argument("--log", action="store_true", help="log-scale the intensity grids")
    args = ap.parse_args()

    PlotSimulation(args.design_path, backend=args.backend, suffix=args.suffix,
                   fig_dir=args.fig_dir, log=args.log, n=args.n).run()
