"""class_simulation.py — Reservoir's engine, built on the shared SimpleSim
library (guide/source/sensor/backend orchestration) instead of a duplicated
copy of it. Reservoir-specific physics (reservoir/mirror/slm objects, the
STED concentration sensor, legacy fs-parameterized pulse sources) plug into
SimpleSim's Simulation via the registries in _simplesim_ext.py.

`Simulation` below is a backward-compatible facade preserving the public
entry point every existing script uses (`Simulation(path).run_simulation()`
/ `.run_empty()`); the real engine is `ReservoirSimulation`.

    python class_simulation.py --path data/test2D [--empty-only|--lc-only]
"""
from __future__ import annotations

import _lcrelax_locate  # noqa: F401
import _simplesim_locate  # noqa: F401

from simplesim.simulation import Simulation as _SSSimulation

from _simplesim_ext import (DEFAULT_ARGS, OBJECT_TYPES, SENSOR_TYPES,
                            SIZE_RESOLVERS, SOURCE_TYPES)


def _null_object(args: dict, folder: str, mp):
    return None


#: "reservoir"/"voltage_reservoir" dropped entirely — for the air-reference
#: run (see Simulation.run_empty below), which unlike SimpleSim's generic
#: `run(empty=True)` (drops ALL geometry) only removes the LC region and
#: keeps guides/mirrors/slm in place.
_AIR_REFERENCE_OBJECT_TYPES = {"reservoir": _null_object, "voltage_reservoir": _null_object}


class ReservoirSimulation(_SSSimulation):
    OBJECT_TYPES = OBJECT_TYPES
    SENSOR_TYPES = SENSOR_TYPES
    SOURCE_TYPES = SOURCE_TYPES
    SIZE_RESOLVERS = SIZE_RESOLVERS
    DEFAULT_ARGS = DEFAULT_ARGS


class Simulation:
    """Backward-compatible facade over ReservoirSimulation(backend="meep") —
    every existing script keeps using Simulation(path).run_simulation() /
    .run_empty() unchanged; the MEEP engine now runs through SimpleSim."""

    def __init__(self, args_path: str) -> None:
        self.folder_path = args_path
        self.sim: ReservoirSimulation | None = None

    def run_simulation(self) -> None:
        self.sim = ReservoirSimulation(self.folder_path, backend="meep")
        self.sim.relax()
        self.sim.run(empty=False, out_name="simulation")

    def run_empty(self) -> None:
        """Air-reference run: the LC/reservoir region becomes background,
        everything else (guides, mirrors, SLM) stays — NOT the same as
        SimpleSim's `run(empty=True)`, which drops all geometry. Saved to
        "simulation_empty" (Reservoir's historical bare naming, not
        SimpleSim's backend-suffixed default)."""
        sim = ReservoirSimulation(self.folder_path, backend="meep",
                                  object_types=_AIR_REFERENCE_OBJECT_TYPES)
        sim.run(empty=False, out_name="simulation_empty")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/test2D")
    parser.add_argument("--empty-only", action="store_true")
    parser.add_argument("--lc-only", action="store_true")
    args = parser.parse_args()
    simulation = Simulation(args.path)
    if args.empty_only:
        simulation.run_empty()
    elif args.lc_only:
        simulation.run_simulation()
    else:
        simulation.run_simulation()
        simulation.run_empty()
