"""SimulationGPU — the gpumeep twin of class_simulation.Simulation.

Same JSON, same method skeleton, same npz outputs as the MEEP class; the
engine is the MEEP-exact gpumeep API (GPUmeep/src/gpumeep.py: D-form
stepping, MEEP UPML, Kottke subpixel, quadrature current sources, decimated
collocated DFT monitors, MultilevelAtom gain — ladder-validated against
MEEP v1.33). Object adapters live in their own files, mirroring the MEEP
layout:

    class_guide_gpu.GuideGPU        class_mirror_gpu.MirrorGPU
    class_source_gpu.SourceGPU      class_sensor_gpu.SensorGPU
    class_reservoir_gpu.ReservoirGPU
    gpumeep_setup                   (locates GPUmeep, imports gm)

    python class_simulation_gpu.py --path data/test2D [--empty]

Engine deltas vs the retired in-file engine (intentional upgrades):
  * DFT amplitudes use MEEP's dt/sqrt(2*pi) scale everywhere (the old plain-2D
    path used a 2/N CW convention — relative shapes unchanged, absolute scale
    differs).
  * A source-off decay run follows run_until (mirrors class_simulation)
    so DFT monitors integrate the ring-down exactly like MEEP.
  * PML is MEEP UPML on all boundaries (old path: CPML).

Ladder validation of this rewrite (2026-07-15, res40 fp64 CPU): cfg1 air
corr 0.99997, cfg2 LC corr 0.99967, cfg3 LC+dye corr 0.99966; amplitude
ratios 0.998–1.000.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from gpumeep_setup import gm, JDTYPE as _JDTYPE, FS_PER_MEEP as _FS_PER_MEEP
from class_guide_gpu import GuideGPU
from class_mirror_gpu import MirrorGPU
from class_source_gpu import SourceGPU
from class_sensor_gpu import SensorGPU
from class_reservoir_gpu import ReservoirGPU

# Engine-support symbols gpumeep.py imports from "class_simulation_gpu":
# re-export GPUmeep's own copies so external `from class_simulation_gpu
# import _STEDSource` keeps working against this module.
import sys as _sys

_engine = _sys.modules["_gpumeep_engine_csg"]
_STEDSource = _engine._STEDSource
_src_overlap_weights = _engine._src_overlap_weights
_src_delta_weights = _engine._src_delta_weights

import jax.numpy as jnp  # noqa: E402


class _SimGPU(gm.Simulation):
    """gm.Simulation + two additive extensions:
    (1) vectorized callable materials (material fn with .tensor6_vec) — the
        per-point python loop is unusable for LC reservoirs at res 40;
    (2) full-box 2D DFT monitors (the reservoir '2Ddft' sensor type)."""

    def _eps_at(self, X, Y, Z=None):
        t0 = self.default_material.tensor6()
        comps = [np.full(X.shape, t0[k], dtype=np.float64) for k in range(6)]
        for blk in self.geometry:
            m = self._block_mask(blk, X, Y, Z)
            mat = blk.material
            if callable(mat) and hasattr(mat, "tensor6_vec"):
                t = mat.tensor6_vec(X, Y, Z)          # (6, *X.shape)
                for k in range(6):
                    comps[k][m] = t[k][m]
            elif callable(mat):
                for idx in np.argwhere(m):
                    p = (gm.Vector3(X[tuple(idx)], Y[tuple(idx)])
                         if Z is None else
                         gm.Vector3(X[tuple(idx)], Y[tuple(idx)], Z[tuple(idx)]))
                    t = mat(p).tensor6()
                    for k in range(6):
                        comps[k][tuple(idx)] = t[k]
            else:
                t = mat.tensor6()
                for k in range(6):
                    comps[k][m] = t[k]
        return comps

    def add_dft_fields_box(self, fcen, i_lo, i_hi, j_lo, j_hi):
        """Full-grid single-frequency DFT of (Ex, Ey, Ez, Hz); cropped to the
        [i_lo:i_hi, j_lo:j_hi] box at save time. 2D only."""
        self._require_init()
        if self.dim != 2:
            raise NotImplementedError("2Ddft box monitor: 2D only")
        omega = 2.0 * np.pi * float(fcen)
        dt = self.dt
        z = jnp.zeros((self.Nx, self.Ny), dtype=_JDTYPE)

        def updater(m, f, t, step):
            rEx, iEx, rEy, iEy, rEz, iEz, rHz, iHz = m
            c = jnp.cos(omega * t); s = jnp.sin(omega * t)
            cH = jnp.cos(omega * (t - 0.5 * dt)); sH = jnp.sin(omega * (t - 0.5 * dt))
            return (rEx + c * f.Ex, iEx + s * f.Ex,
                    rEy + c * f.Ey, iEy + s * f.Ey,
                    rEz + c * f.Ez, iEz + s * f.Ez,
                    rHz + cH * f.Hz, iHz + sH * f.Hz)

        mon = {"kind": "dft2dbox", "freqs": np.array([float(fcen)]), "decim": 1,
               "i_lo": i_lo, "i_hi": i_hi, "j_lo": j_lo, "j_hi": j_hi,
               "updater": updater, "state": (z,) * 8}
        self._monitors.append(mon)
        return mon

    def get_dft_box(self, mon, component):
        """Complex full-grid array for a dft2dbox monitor, MEEP dt/√2π scale,
        cropped to the monitor box."""
        scl = self.dt / np.sqrt(2.0 * np.pi)
        idx = {"Ex": 0, "Ey": 2, "Ez": 4, "Hz": 6}[component]
        re, im = mon["state"][idx], mon["state"][idx + 1]
        arr = (np.asarray(re) + 1j * np.asarray(im)) * scl
        return arr[mon["i_lo"]:mon["i_hi"], mon["j_lo"]:mon["j_hi"]]


# ---------------------------------------------------------------------------
# The simulation orchestrator — method-for-method twin of
# class_simulation.Simulation.
# ---------------------------------------------------------------------------
@dataclass
class SimulationGPU:
    folder_path: str
    empty: bool = False
    # Per-run amplitude override for the SIGNAL source (basis/forward runs),
    # {source_key: [amps]} — mirrors SimulationT._run_basis.
    amp_override: dict | None = None
    run_until_override: object = None
    # Legacy flag: the gpumeep engine is ALWAYS full-vector MEEP-exact now.
    force_fullvector: bool = False

    args: dict = field(default_factory=dict)
    objects_args: list = field(default_factory=list)
    paths: dict = field(default_factory=dict)
    objects: list = field(default_factory=list)
    sources: list = field(default_factory=list)
    sensors: list = field(default_factory=list)
    resolution: int = 40
    simulation: object = None

    # ---------------- setup (mirrors class_simulation) ----------------

    def _set_data(self):
        with open(os.path.join(self.folder_path, "simulation_data.json")) as f:
            self.args.update(json.load(f))
        sim_dir = "simulation_empty" if self.empty else "simulation"
        self.paths = {
            "simulation": os.path.join(self.folder_path, sim_dir),
            "snapshots":  os.path.join(self.folder_path, sim_dir, "snapshots"),
            "figures":    os.path.join(self.folder_path, "figures"),
        }
        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)

    def _set_simulation_parameters(self):
        self.resolution = int(self.args["resolution"])

    @staticmethod
    def _pos_to_center_size(pos, on_edge_x, on_size_x, cell_x, cell_y, cell_z=0.0):
        if isinstance(pos, dict):
            label = pos.get("position", "center")
            raw = pos.get("size", [0.0, 0.0])
        else:
            label = str(pos) if pos else "center"
            raw = [0.0, 0.0]
        if isinstance(raw, (int, float)):
            raw = [float(raw), 0.0]
        if len(raw) >= 3:                     # [x, y, z] → area/volume source
            sx = float(raw[0])
            sy = float(raw[1]) if raw[1] else cell_y
            sz = float(raw[2])
        else:                                 # [y, z] → plane ⊥ x
            sx = 0.0
            sy = float(raw[0]) if raw[0] else cell_y
            sz = float(raw[1]) if len(raw) > 1 else 0.0
        x = {"left": on_edge_x, "right": on_edge_x + on_size_x}.get(
            label, on_edge_x + on_size_x / 2)
        return gm.Vector3(x, 0, 0), gm.Vector3(sx, sy, sz)

    def _update_all_args(self):
        self.objects_args = []
        pml = float(self.args.get("pml_size", 2.0))
        self.dim = self.args.get("dimention", 1)
        cell_y = (float(self.args.get("cell_size_y", 0.0)) if self.dim > 1
                  else 4.0 / self.resolution)
        cell_z = float(self.args.get("cell_size_z", 0.0)) if self.dim > 2 else 0.0

        current_x = 0.0
        for key in self.args["object_order"]:
            obj = dict(self.args[key])
            obj["_key"] = key
            obj["edge_x_local"] = current_x
            if isinstance(obj.get("sizes"), list):
                obj["size_x"] = float(obj["sizes"][0])
            if obj.get("class") == "mirror" and "size_x" not in obj:
                lam = float(obj["lam"])
                indices = obj.get("n_indexes", obj.get("indexes", [1.0, 1.0]))
                if "n_layers" in obj:
                    n_lays = int(obj["n_layers"])
                else:
                    from class_mirror import Mirror
                    n_lays = Mirror._n_layers_for_transmission(
                        float(obj["transmission"]), indices)
                obj["n_layers_resolved"] = n_lays
                obj["size_x"] = sum(lam / 4.0 / float(indices[i % 2])
                                    for i in range(n_lays))
            elif obj.get("class") == "mirror":
                obj.setdefault("n_layers_resolved", int(obj["n_layers"]))
            self.objects_args.append(obj)
            current_x += float(obj.get("size_x", 0.0))

        self.cell_x = current_x + 2 * pml
        self.cell_y = cell_y
        self.cell_z = cell_z
        x0 = -self.cell_x / 2 + pml

        for obj in self.objects_args:
            edge_x = obj["edge_x_local"] + x0
            size_x = float(obj.get("size_x", 0.0))
            cls = obj.get("class", "")
            if cls == "guide":
                obj["center"] = gm.Vector3(edge_x + size_x / 2, 0, 0)
                sizes_raw = obj.get("sizes")
                gy = (float(sizes_raw[1]) if isinstance(sizes_raw, list)
                      and len(sizes_raw) > 1 else float(obj.get("size_y", 0.0)))
                obj["sizes"] = gm.Vector3(size_x, gy if gy > 0 else gm.inf, gm.inf)
                obj["edge_x_meep"] = edge_x
            elif cls in ("reservoir", "voltage_reservoir"):
                obj["center_x_meep"] = edge_x + size_x / 2
                obj["edge_x_meep"] = edge_x
            elif cls == "mirror":
                obj["x_start_meep"] = edge_x
                obj["edge_x_meep"] = edge_x
            elif cls in ("monitor", "source"):
                on_object = obj.get("on_object", -1)
                if on_object == -1 and isinstance(obj.get("position"), dict):
                    on_object = obj["position"].get("on_object", -1)
                if isinstance(on_object, str):
                    ref = next((r for r in self.objects_args
                                if r.get("_key") == on_object), None)
                elif isinstance(on_object, int) and on_object >= 0:
                    ref = self.objects_args[on_object]
                else:
                    ref = None
                on_edge = ref["edge_x_local"] + x0 if ref else -self.cell_x / 2
                on_size = float(ref.get("size_x", 0.0)) if ref else self.cell_x
                obj["on_object_edge_x"] = on_edge
                obj["on_object_size_x"] = on_size
                obj["cell_x"] = self.cell_x
                obj["cell_y"] = cell_y
                obj["cell_z"] = cell_z
                if cls == "source":
                    center, size = self._pos_to_center_size(
                        obj.get("position", {}), on_edge, on_size,
                        self.cell_x, cell_y, cell_z)
                    obj["center"] = center
                    obj["size"] = size

    def get_object(self, args):
        cls = args["class"]
        if cls == "guide":
            return GuideGPU(args)
        if cls == "source":
            a = dict(args)
            if self.amp_override and args.get("_key") in self.amp_override:
                a["amplitude"] = list(self.amp_override[args["_key"]])
            return SourceGPU(a)
        if cls == "monitor":
            return SensorGPU(args)
        if cls in ("reservoir", "voltage_reservoir"):
            if self.empty:
                return None
            return ReservoirGPU(self.folder_path, args, self.cell_y, self.cell_z)
        if cls == "mirror":
            return MirrorGPU(args)
        if cls == "slm":
            raise NotImplementedError("SLM objects are MEEP-only")
        return None

    def _set_object_list(self):
        self.objects, self.sources, self.sensors = [], [], []
        self._update_all_args()
        for obj_args in self.objects_args:
            obj = self.get_object(obj_args)
            if obj is None:
                continue
            cls = obj_args["class"]
            if cls == "source":
                self.sources.extend(obj.return_source_object())
            elif cls == "monitor":
                self.sensors.append(obj)
            else:
                self.objects.append(obj)

    def _set_cell(self):
        y = self.cell_y if self.dim > 1 else 4.0 / self.resolution
        z = self.cell_z if self.dim > 2 else 0.0
        self.cell = gm.Vector3(self.cell_x, y, z)

    def _set_geometry(self):
        self.geometry = [b for obj in self.objects
                         for b in obj.get_geometry_blocks()]

    def _set_pmls(self):
        if float(self.args.get("pml_size", 2.0)) == 0.0:
            self.pmls = []
        else:
            if self.args.get("periodic"):
                print("SimulationGPU: periodic + PML applies PML on ALL "
                      "boundaries (x-only PML is MEEP-only)")
            self.pmls = [gm.PML(float(self.args["pml_size"]))]

    def _set_simulation(self):
        bg = float(self.args.get("background_index", 1.0))
        self.simulation = _SimGPU(
            cell_size=self.cell,
            resolution=self.resolution,
            geometry=self.geometry,
            sources=self.sources,
            boundary_layers=self.pmls,
            default_material=gm.Medium(index=bg) if bg != 1.0 else gm.air,
            k_point=gm.Vector3(0, 0, 0) if self.args.get("periodic") else False,
            Courant=float(self.args.get("courant", 0.5)),
            eps_averaging=(not bool(os.environ.get("MEEP_NO_SUBPIXEL"))),
        )

    def _setup_sensors(self):
        for sensor in self.sensors:
            sensor.add_to_simulation(self.simulation)

    def _set_everything(self):
        self._set_data()
        self._set_simulation_parameters()
        self._set_object_list()
        self._set_cell()
        self._set_geometry()
        self._set_pmls()
        self._set_simulation()
        self._setup_sensors()

    # ---------------- run ----------------

    def _run_once(self):
        run_until = float(self.run_until_override
                          if self.run_until_override is not None
                          else self.args.get("run_until", 200))
        decay = float(self.args.get("source_off_decay", 50.0))
        sim = self.simulation
        stepped = [s for s in self.sensors if s.stepped]
        t0 = time.time()
        if stepped:
            sim._require_init()
            # gm.run(until=T) advances T additional units → chunk the run at
            # the smallest sensor interval and record between chunks.
            chunk_t = min(s.step_interval() for s in stepped)
            n_chunks = max(1, int(round(run_until / chunk_t)))
            nexts = {id(s): 0.0 for s in stepped}
            for s in stepped:
                s.record(sim, 0.0)
                nexts[id(s)] += s.step_interval()
            for _ in range(n_chunks):
                sim.run(until=chunk_t)
                t = sim._t_steps * sim.dt
                for s in stepped:
                    if t + 1e-9 >= nexts[id(s)]:
                        s.record(sim, t)
                        nexts[id(s)] += s.step_interval()
        else:
            sim.run(until=run_until)
        # source-off ring-down, mirroring class_simulation (DFT keeps accumulating)
        if decay > 0:
            sim.change_sources([])
            sim.run(until=decay)
        print(f"Run finished in {time.time() - t0:.1f} s "
              f"({sim._t_steps} steps)")

    def _save_all(self):
        for sensor in self.sensors:
            sensor.save(self.simulation, self.paths["simulation"])
        for obj in self.objects:
            if isinstance(obj, ReservoirGPU):
                obj.save_fields()

    def run_simulation(self):
        self._set_everything()
        self._run_once()
        self._save_all()

    # old public API
    def run(self):
        return self.run_simulation()

    def run_empty(self):
        self.empty = True
        self.args = {}
        try:
            self.run_simulation()
        finally:
            self.empty = False

    def run_basis(self, amplitude_list, source_key=None):
        """One forward run with a given SIGNAL amplitude → complex (Ey, Ex, Ez)
        at monitor_2 (open_reservoir dispatch, same as before)."""
        if source_key is None:
            self._set_data()
            self._update_all_args()
            source_key = next(o["_key"] for o in self.objects_args
                              if o.get("class") == "source"
                              and o.get("_key") != "source_2")
        self.amp_override = {source_key: list(amplitude_list)}
        self.args = {}
        self.run_simulation()
        m2 = np.load(os.path.join(self.paths["simulation"], "monitor_2.npz"))
        return (np.asarray(m2["Ey"]).ravel(), np.asarray(m2["Ex"]).ravel(),
                np.asarray(m2["Ez"]).ravel())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/test2D")
    parser.add_argument("--empty", action="store_true")
    cli = parser.parse_args()
    sim = SimulationGPU(folder_path=cli.path, empty=cli.empty)
    sim.run_simulation()
