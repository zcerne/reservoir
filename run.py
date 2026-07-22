#!/usr/bin/env python
"""Reservoir runner — thin wrapper over the SimpleSim library, same
functionality as focusProject's run.py, using ReservoirSimulation (the
SimpleSim Simulation subclass wired with reservoir/mirror/slm/concentration-
sensor/legacy-pulse-source support — see class_simulation.py / _simplesim_ext.py).

    python run.py data/test2D                     # relax + FDTD + plots
    python run.py data/test2D --backend meep      # MEEP instead of GPUmeep
    python run.py data/test2D --relax-only        # LC relaxation only
    python run.py data/test2D --plot              # figures from saved npz
    python run.py data/test2D --prefix v30_        # tag every saved/found
                                                   # file so a design folder
                                                   # can hold multiple runs
                                                   # (e.g. --prefix v0_ / v30_)
                                                   # side by side
"""
from __future__ import annotations

import argparse


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
    ap.add_argument("--prefix", default="",
                    help="prepended to every saved/searched output filename "
                         "(sensor npz + figures), so a design folder can hold "
                         "multiple parameter variants side by side")
    a = ap.parse_args()

    import _lcrelax_locate  # noqa: F401
    import _simplesim_locate  # noqa: F401
    from class_simulation import ReservoirSimulation

    sim = ReservoirSimulation(a.design, backend=a.backend, precision=a.precision,
                              prefix=a.prefix)
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
