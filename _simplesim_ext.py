"""Reservoir-specific extensions plugged into SimpleSim's Simulation via its
OBJECT_TYPES/SENSOR_TYPES/SOURCE_TYPES/SIZE_RESOLVERS registries (see
simplesim.simulation.Simulation). Kept local to this repo because none of
this is generic LC-block behavior — it's reservoir research physics
(isotropic/STED reservoir, Bragg mirror, SLM) or legacy JSON conventions
(femtosecond-parameterized pulses) that shouldn't leak into the shared
library other projects (BlockOptimization, focusProject) build on.
"""
from __future__ import annotations

import os

import numpy as np

import _lcrelax_locate  # noqa: F401
import _simplesim_locate  # noqa: F401
from simplesim.sensor import Sensor as _SSSensor
from simplesim.source import Source as _SSSource

from class_mirror import Mirror
from class_slm import SLM

#: 1 MEEP time unit = a/c0 = 1 µm / c0 = 3.33564 fs — used ONLY to keep
#: historical STED configs (pulse_fwhm_fs/pulse_delay_fs, snap_interval_fs)
#: working as-is. New configs should specify widths/intervals directly in
#: MEEP time units instead; this conversion stays local to Reservoir, not
#: the shared SimpleSim engine.
FS_PER_MEEP = 3.335640952


# ---------------------------------------------------------------------------
# SOURCE_TYPES: legacy fs-parameterized pulse ("source_type": "pulsed")
# ---------------------------------------------------------------------------

class PulsedFsSource(_SSSource):
    """Temporal Gaussian pulse parameterized in femtoseconds — the
    historical STED pump/depletion convention (pulse_fwhm_fs,
    pulse_delay_fs). Falls back to the shared meep-time gaussian/dlam
    construction when neither key is present."""

    def _set_source(self):
        if "pulse_fwhm_fs" not in self.args and "pulse_delay_fs" not in self.args:
            super()._set_source()
            return
        mp = self.mp
        fwhm_fs = float(self.args.get("pulse_fwhm_fs", 1309.0))
        delay_fs = float(self.args.get("pulse_delay_fs", 0.0))
        width = (fwhm_fs / FS_PER_MEEP) / 2.35482
        kw = {"width": width}
        if delay_fs > 0:
            kw["start_time"] = delay_fs / FS_PER_MEEP
        self.source = mp.GaussianSource(1 / self.lam, **kw)


# ---------------------------------------------------------------------------
# SENSOR_TYPES: STED 4-level population monitor ("type": "concentration")
# ---------------------------------------------------------------------------

class ConcentrationSensor(_SSSensor):
    """STED gain-medium population monitor. Stepped like a *snap sensor
    (driven by an at_every step func from backend_fdtd.py) rather than
    registered as engine DFT/flux state — reads sim.gain_populations().

    gain_populations() returns the population array over gpumeep's ENTIRE
    simulation grid (its state is allocated domain-wide — an intentional
    MEEP-compatibility choice, see multilevel.py::init_state_full's "N = N0
    at EVERY center" docstring — not just where the gain medium actually
    couples to the field). This sensor crops that down to its own area:
    the on-object's bounding box (x-width from `on_object_size_x`, y-height
    from `on_object_size_y` — both set by SimulationData._layout()) by
    default, or an explicit JSON `position.size` if given, same convention
    every other Sensor subclass uses."""

    def __init__(self, args: dict, mp_mod) -> None:
        super().__init__(args, mp_mod)
        self._snaps: list = []
        self._times: list = []
        self._box: tuple[int, int, int, int] | None = None   # (i_lo,i_hi,j_lo,j_hi)

    def add_to_simulation(self, sim) -> None:
        pass  # stepped — nothing to register on the engine

    def _step_interval(self) -> float:
        if "snap_interval" in self.args:
            return float(self.args["snap_interval"])
        return float(self.args.get("snap_interval_fs", 10.0)) / FS_PER_MEEP

    def get_step_func(self):
        # gain_populations() is a gpumeep-only introspection hook — real
        # MEEP's C++ engine never exposed multilevel-atom populations to
        # Python at all. The old MEEP-side Sensor never implemented a
        # "concentration" type either (silent no-op, same as every other
        # unrecognized sensor type); match that instead of crashing.
        if self.mp.__name__ == "meep":
            return None
        return self._step_interval(), self._record

    def _crop_box_um(self) -> tuple[float, float, float]:
        """(center_x, size_x, size_y) in µm — the on-object's own footprint
        by default (objects are always y-centered at 0 in this convention),
        or an explicit JSON position.size override."""
        req_sx, req_sy, _ = self._parse_size()
        obj_sx = float(self.args.get("on_object_size_x", 0.0))
        obj_sy = float(self.args.get("on_object_size_y", 0.0))
        sx = req_sx if req_sx > 0 else (obj_sx if obj_sx > 0 else float(self.args.get("cell_x", 0.0)))
        sy = req_sy if req_sy > 0 else (obj_sy if obj_sy > 0 else float(self.args.get("cell_y", 0.0)))
        return self.center_x, sx, sy

    def _grid_box(self, sim) -> tuple[int, int, int, int]:
        cx_um, sx, sy = self._crop_box_um()
        dx = float(sim.dx)
        i_lo = max(0, int(round((cx_um - sx / 2 + sim.cx) / dx)))
        i_hi = min(sim.Nx, int(round((cx_um + sx / 2 + sim.cx) / dx)))
        j_lo = max(0, int(round((-sy / 2 + sim.cy) / dx)))
        j_hi = min(sim.Ny, int(round((sy / 2 + sim.cy) / dx)))
        return i_lo, i_hi, j_lo, j_hi

    def _record(self, sim) -> None:
        N = sim.gain_populations()
        if N is None:
            raise ValueError("concentration monitor requires reservoir.sted")
        if self._box is None:
            self._box = self._grid_box(sim)
        i_lo, i_hi, j_lo, j_hi = self._box
        self._snaps.append(np.asarray(N[:, i_lo:i_hi, j_lo:j_hi], dtype=np.float32))
        self._times.append(sim.meep_time())

    def save(self, sim, path: str, suffix: str = "") -> None:
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, f"{self.key}_{suffix}.npz")
        N = np.array(self._snaps, dtype=np.float32)
        extra = {}
        if self._box is not None:
            i_lo, i_hi, j_lo, j_hi = self._box
            dx = float(sim.dx)
            extra["x"] = np.arange(i_lo, i_hi) * dx - sim.cx
            extra["y"] = np.arange(j_lo, j_hi) * dx - sim.cy
        np.savez(out, N=N, times=np.array(self._times),
                 levels=["N1", "N2", "N3", "N4"],
                 snap_interval=self._step_interval(), **extra)


# ---------------------------------------------------------------------------
# OBJECT_TYPES: reservoir / voltage_reservoir / mirror / slm
# ---------------------------------------------------------------------------

def build_reservoir(args: dict, folder: str, mp):
    """Both "reservoir" and "voltage_reservoir" JSON classes land here —
    voltage_reservoir is only a marker for an external preprocessing step
    (scripts/run_voltage_reservoir.py writes lc_fields.npz via
    class_voltage_reservoir.VoltageReservoir); the FDTD-side object is the
    same Reservoir/ReservoirGPU that loads that cache, engine-agnostic."""
    if mp.__name__ == "meep":
        from class_reservoir import Reservoir
        obj = Reservoir(folder)
        obj._meep_center_x = float(args["center_x_meep"])
        fields_file = os.path.join(folder, "simulation", "lc_fields.npz")
        if os.path.exists(fields_file):
            obj.load_fields()
        else:
            obj.run_minimization()
        return obj
    from class_reservoir_gpu import ReservoirGPU
    return ReservoirGPU(folder, args, args["cell_y"], args["cell_z"])


def build_mirror(args: dict, folder: str, mp):
    if mp.__name__ == "meep":
        margs = dict(args)
        margs["x_start"] = args["edge_x_meep"]
        margs["n_layers"] = args["n_layers_resolved"]
        margs.setdefault("size_y", args["cell_y"])
        return Mirror(margs)
    from class_mirror_gpu import MirrorGPU
    margs = dict(args)
    margs["x_start_meep"] = args["edge_x_meep"]
    margs["n_layers_resolved"] = args["n_layers_resolved"]
    margs.setdefault("size_y", args["cell_y"])
    return MirrorGPU(margs)


def build_slm(args: dict, folder: str, mp):
    if mp.__name__ != "meep":
        raise NotImplementedError("SLM objects are MEEP-only")
    sargs = dict(args)
    sargs["center"] = mp.Vector3(float(args["center_x_meep"]), 0, 0)
    return SLM(sargs)


def mirror_size_x(obj: dict) -> float:
    """Same analytic DBR sizing class_simulation.py used to precompute
    cell_x before layout — Mirror has no explicit `sizes`, just lam +
    (n_layers | transmission) + indexes."""
    lam = float(obj["lam"])
    indices = obj.get("n_indexes", obj.get("indexes", [1.0, 1.0]))
    if "n_layers" in obj:
        n_lays = int(obj["n_layers"])
    else:
        n_lays = Mirror._n_layers_for_transmission(float(obj["transmission"]), indices)
    obj["n_layers_resolved"] = n_lays
    return sum(lam / 4.0 / float(indices[i % 2]) for i in range(n_lays))


def slm_size_x(obj: dict) -> float:
    """d = λ / (2·Δn) — half-wave retarder thickness, same formula as
    SLM._set_width()."""
    n_o, n_e = float(obj["no_ne"][0]), float(obj["no_ne"][1])
    return float(obj["lam"]) / (2 * (n_e - n_o))


OBJECT_TYPES = {
    "reservoir": build_reservoir,
    "voltage_reservoir": build_reservoir,
    "mirror": build_mirror,
    "slm": build_slm,
}
SENSOR_TYPES = {"concentration": ConcentrationSensor}
SOURCE_TYPES = {"pulsed": PulsedFsSource}
SIZE_RESOLVERS = {"mirror": mirror_size_x, "slm": slm_size_x}
#: Reservoir's engines always ran a fixed post-source-off decay window
#: (MEEP: hardcoded 50; GPU: JSON `source_off_decay` defaulting to 50) — the
#: shared FdtdBackend defaults this to 0 (off) for other projects, so
#: Reservoir restores its own historical default here.
DEFAULT_ARGS = {"source_off_decay": 50.0}
