"""class_simulation.py — Reservoir's ONE engine entry point, built on the
shared SimpleSim library (guide/source/sensor/backend orchestration) instead
of a duplicated per-backend copy of it (the old class_simulation_gpu.py +
class_guide_gpu/class_sensor_gpu/class_source_gpu/class_mirror_gpu, retired
2026-07-27 — they were an unsynced reimplementation that gave measurably
wrong STED gain, see data_gen/_gen_common.py:open_reservoir). Reservoir-
specific physics (reservoir/mirror/slm objects, the STED concentration
sensor, legacy fs-parameterized pulse sources) plug into SimpleSim's
Simulation via the registries in _simplesim_ext.py, for BOTH backends —
gpumeep is a MEEP-API drop-in so the same registries serve either engine.

`Simulation` below is a backward-compatible facade preserving the public
entry point every existing script uses (`Simulation(path).run_simulation()`
/ `.run_empty()`); the real engine is `ReservoirSimulation`.

    python class_simulation.py --path data/test2D --backend meep|gpumeep [--empty-only|--lc-only]
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
    """Backward-compatible facade over ReservoirSimulation(backend=...) —
    every existing script keeps using Simulation(path).run_simulation() /
    .run_empty() unchanged, now for EITHER engine (was MEEP-only; the old
    gpumeep orchestrator class_simulation_gpu.SimulationGPU is retired —
    this facade replaces it too, same bare "simulation"/"simulation_empty"
    output naming it used, backend-agnostic)."""

    def __init__(self, args_path: str, backend: str = "meep",
                precision: str = "fp64") -> None:
        self.folder_path = args_path
        self.backend = backend
        self.precision = precision
        self.sim: ReservoirSimulation | None = None

    def run_simulation(self) -> None:
        self.sim = ReservoirSimulation(self.folder_path, backend=self.backend,
                                       precision=self.precision)
        self.sim.relax()
        self.sim.run(empty=False, out_name="simulation")

    def run_empty(self) -> None:
        """Air-reference run: the LC/reservoir region becomes background,
        everything else (guides, mirrors, SLM) stays — NOT the same as
        SimpleSim's `run(empty=True)`, which drops all geometry. Saved to
        "simulation_empty" (Reservoir's historical bare naming, not
        SimpleSim's backend-suffixed default)."""
        sim = ReservoirSimulation(self.folder_path, backend=self.backend,
                                  precision=self.precision,
                                  object_types=_AIR_REFERENCE_OBJECT_TYPES)
        sim.run(empty=False, out_name="simulation_empty")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/test2D")
    parser.add_argument("--backend", choices=["meep", "gpumeep"], default="meep")
    parser.add_argument("--precision", choices=["fp32", "fp64"], default="fp64")
    parser.add_argument("--empty-only", action="store_true")
    parser.add_argument("--lc-only", action="store_true")
    args = parser.parse_args()
    simulation = Simulation(args.path, backend=args.backend, precision=args.precision)
    if args.empty_only:
        simulation.run_empty()
    elif args.lc_only:
        simulation.run_simulation()
    else:
        simulation.run_simulation()
        simulation.run_empty()
