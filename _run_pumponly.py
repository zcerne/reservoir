"""Run a design with the signal source silenced, tagging the output with a suffix.

    python _run_pumponly.py data/reservoir_types/block_iso_gain/02 --suffix pumponly

No design-folder copy: SimulationData.load takes an `overrides` dict that is
shallow-merged into the JSON in memory, which is the documented way to swap a
source amplitude for a sweep "without touching the file on disk". run.py has no
flag for it, hence this shim. The design folder therefore holds the pump-only
outputs alongside the normal ones, separated only by --suffix.

Population snapshots follow whatever the design's pop_monitor says (2 time units
for 02), so this writes whole-cell E+N snapshots -- the dataset generators force
snap_interval=0 and keep only the N(t) trace, which is not what we want here.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as _run

ap = argparse.ArgumentParser()
ap.add_argument("design")
ap.add_argument("--suffix", default="pumponly")
ap.add_argument("--source", default="source_1",
                help="source key to silence (default source_1, the signal)")
ap.add_argument("--backend", default="gpumeep")
ap.add_argument("--pump-amp", type=float, default=None,
                help="override source_pump.amplitude — the knob that sets how far "
                     "the bleaching front gets, and hence the inversion level")
ap.add_argument("--pump", default="source_pump", help="key of the pump source")
ap.add_argument("--snap-interval", type=float, default=None,
                help="override pop_monitor.snap_interval; raise it on long runs or "
                     "the whole-cell E+N dump grows linearly with run_until "
                     "(470 tu at interval 2 would be ~6 GB)")
a = ap.parse_args()

_run._ensure_simplesim()
from simplesim import Simulation
from symbols_source import register as _register_symbols
_register_symbols()

ov = {a.source: {"amplitude": [0.0, 0.0]}}
if a.pump_amp is not None:
    ov[a.pump] = {"amplitude": a.pump_amp}      # scalar: stays one uniform line source
if a.snap_interval is not None:
    ov["pop_monitor"] = {"snap_interval": a.snap_interval}
sim = Simulation(a.design, backend=a.backend, precision="fp64", suffix=a.suffix,
                 overrides=ov)
print(f"[pumponly] {a.design}: {a.source}.amplitude zeroed, suffix={a.suffix!r}", flush=True)
sim.relax()
sim.run()
sim.plot()
