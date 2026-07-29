#!/usr/bin/env python
"""Reservoir runner — thin wrapper over the SimpleSim library, same shape as
focusProject's run.py: plain simplesim.Simulation, no reservoir-specific
extensions wired in.

    python run.py data/test2D                     # relax + FDTD + plots
    python run.py data/test2D --backend meep      # MEEP instead of GPUmeep
    python run.py data/test2D --relax-only        # LC relaxation only
    python run.py data/test2D --plot              # figures from saved npz
    python run.py data/test2D --suffix v30         # tag every saved/found
                                                   # OUTPUT file (sensor
                                                   # npz + figures) so a
                                                   # design folder can hold
                                                   # multiple runs side by
                                                   # side
    python run.py data/test2D --design-suffix v30   # save/load a distinct
                                                   # LC-relax DESIGN
                                                   # VARIANT (e.g. a
                                                   # different applied
                                                   # voltage) — separate
                                                   # from --suffix above;
                                                   # combine both to relax
                                                   # + simulate + save
                                                   # each variant fully
                                                   # independently:
                                                   #   --design-suffix v0  --suffix v0
                                                   #   --design-suffix v30 --suffix v30
"""
from __future__ import annotations

import argparse
import os
import sys


def _ensure_simplesim():
    if "simplesim" in sys.modules:
        return
    cands = [os.environ.get("SIMPLESIM_PATH")]
    d = os.path.dirname(os.path.abspath(__file__))
    while d not in ("/", ""):
        cands += [os.path.join(d, "SimpleSim", "gitcode"),
                  os.path.join(d, "SimpleSim")]
        d = os.path.dirname(d)
    cands.append(os.path.expanduser("~/Nextcloud/Doktorski/Projects/SimpleSim/gitcode"))
    for p in cands:
        if p and os.path.isdir(os.path.join(p, "simplesim")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return
    raise ModuleNotFoundError(
        "SimpleSim not found — set SIMPLESIM_PATH to its gitcode dir")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design", help="design folder with simulation_data.json")
    ap.add_argument("--backend", choices=["meep", "gpumeep"], default=None)
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp64")
    ap.add_argument("--relax-only", action="store_true")
    ap.add_argument("--force-relax", action="store_true")
    ap.add_argument("--empty", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--suffix", default="",
                    help="tag appended to every saved/searched output filename "
                         "(sensor npz + figures) as _<suffix>, so a design "
                         "folder can hold multiple parameter variants side by "
                         "side. No suffix -> plain <sensor>.npz")
    ap.add_argument("--design-suffix", default="",
                    help="selects/saves a distinct LC-relax design variant "
                         "(lc_fields_<key>_<design-suffix>.npz) — separate "
                         "from --suffix, which only tags output filenames")
    a = ap.parse_args()

    _ensure_simplesim()
    from simplesim import Simulation
    from symbols_source import register as _register_symbols
    _register_symbols()

    sim = Simulation(a.design, backend=a.backend, precision=a.precision,
                     suffix=a.suffix, design_suffix=a.design_suffix)
    if a.plot:
        sim.plot()
        return
    sim.relax(force=a.force_relax)
    if a.relax_only:
        return
    sim.run(empty=a.empty)
    sim.plot()


if __name__ == "__main__":
    main()
