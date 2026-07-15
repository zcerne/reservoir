"""gpumeep twin of class_sensor.Sensor.

Types: flux / 1Ddft / 2Ddft / concentration / 2Dsnap (any *snap).
DFT and flux sensors register gpumeep monitors; concentration and snap
sensors are STEPPED — SimulationGPU._run_once chunks the run at the sensor
interval and calls record() between chunks. Saves the same npz schemas the
MEEP Sensor and the old GPU class wrote."""
import os

import numpy as np

from gpumeep_setup import gm, FS_PER_MEEP


class SensorGPU:
    def __init__(self, args):
        self.args = args
        self.sensor_type = args.get("type", "1Ddft")
        self._name = args.get("name", args.get("_key", self.sensor_type))
        self._handle = None
        self._freqs = np.array([])
        self.snaps: list = []          # stepped-sensor records (filled by driver)
        self.snap_times: list = []
        self._box = None               # (i_lo, i_hi, j_lo, j_hi) for snap types
        self.center, self.size = self._geometry()

    # ---------------- geometry ----------------

    def _parse_position(self):
        pos = self.args.get("position", {})
        if isinstance(pos, dict):
            label = pos.get("position", "center")
            raw = pos.get("size", 0)
            if isinstance(raw, (int, float)):
                sx, sy = 0.0, float(raw)
            elif isinstance(raw, list) and len(raw) >= 2 and float(raw[1]) > 0:
                sx, sy = float(raw[0]), float(raw[1])     # [sx, sy] 2D box
            elif isinstance(raw, list) and len(raw) >= 1:
                sx, sy = 0.0, float(raw[0])               # [sy] / [sy, 0]
            else:
                sx, sy = 0.0, 0.0
        else:
            label, sx, sy = (str(pos) if pos else "center"), 0.0, 0.0
        return label, sx, sy

    def _geometry(self):
        label, sx, sy = self._parse_position()
        edge = float(self.args.get("on_object_edge_x", 0.0))
        osx = float(self.args.get("on_object_size_x", 0.0))
        cell_x = float(self.args.get("cell_x", 0.0))
        cell_y = float(self.args.get("cell_y", 0.0))
        cell_z = float(self.args.get("cell_z", 0.0))
        x = {"left": edge, "right": edge + osx}.get(label, edge + osx / 2)
        sy = sy if sy > 0 else cell_y
        t = self.sensor_type
        if t in ("2Ddft", "2Dsnap") and cell_z == 0:      # 2D sim: xy box
            sx = sx if sx > 0 else (osx if osx > 0 else cell_x)
            return gm.Vector3(x, 0, 0), gm.Vector3(sx, sy, 0)
        if t == "2Ddft":                                   # 3D sim: yz plane
            return gm.Vector3(x, 0, 0), gm.Vector3(0, sy, cell_z)
        if t == "concentration":
            return gm.Vector3(x, 0, 0), gm.Vector3(sx, sy, cell_z)
        return gm.Vector3(x, 0, 0), gm.Vector3(0, sy, cell_z)

    def _freq_band(self):
        lam_range = self.args.get("lam_range", [0.5, 0.5])
        n_lam = int(self.args.get("n_lam", 1))
        fmin, fmax = 1.0 / lam_range[1], 1.0 / lam_range[0]
        return (fmin + fmax) / 2, fmax - fmin, n_lam

    # ---------------- stepped sensors (concentration / snap) ----------------

    @property
    def stepped(self):
        return (self.sensor_type == "concentration"
                or self.sensor_type.endswith("snap"))

    def step_interval(self):
        """Interval in MEEP time units between records."""
        if self.sensor_type == "concentration":
            return float(self.args.get("snap_interval_fs", 10.0)) / FS_PER_MEEP
        return float(self.args.get("dt", 10.0))

    def record(self, sim, t_units: float):
        if self.sensor_type == "concentration":
            N = sim.gain_populations()
            if N is None:
                raise ValueError("concentration monitor requires reservoir.sted")
            self.snaps.append(np.asarray(N, dtype=np.float32))
            self.snap_times.append(t_units * FS_PER_MEEP)
        else:                                   # field snapshot over the box
            if self._box is None:
                self._box = self._grid_box(sim)
            i_lo, i_hi, j_lo, j_hi = self._box
            self.snaps.append(tuple(
                np.asarray(sim.get_array(c))[i_lo:i_hi, j_lo:j_hi]
                .astype(np.float32) for c in (gm.Ex, gm.Ey, gm.Ez)))
            self.snap_times.append(t_units)

    def _grid_box(self, sim):
        sx, sy = float(self.size.x), float(self.size.y)
        x0, y0 = float(self.center.x), float(self.center.y)
        sx = sx if sx > 0 else sim.cell_x
        i_lo = max(0, int(round((x0 - sx / 2 + sim.cx) / sim.dx)))
        i_hi = min(sim.Nx, int(round((x0 + sx / 2 + sim.cx) / sim.dx)))
        j_lo = max(0, int(round((y0 - sy / 2 + sim.cy) / sim.dx)))
        j_hi = min(sim.Ny, int(round((y0 + sy / 2 + sim.cy) / sim.dx)))
        return i_lo, i_hi, j_lo, j_hi

    # ---------------- registration + save ----------------

    def add_to_simulation(self, sim):
        fcen, df, n = self._freq_band()
        if self.sensor_type == "flux":
            self._freqs = gm.Simulation._linspace_freqs(fcen, df, n)
            self._handle = sim.add_flux(fcen, df, n,
                                        gm.FluxRegion(center=self.center,
                                                      size=self.size))
        elif self.sensor_type == "2Ddft" and sim.dim == 2:
            self._freqs = np.array([fcen])
            self._handle = sim.add_dft_fields_box(fcen, *self._grid_box(sim))
        elif self.sensor_type.endswith("dft"):
            self._freqs = gm.Simulation._linspace_freqs(fcen, df, n)
            self._handle = sim.add_dft_fields(
                [gm.Ex, gm.Ey], fcen, df, n, center=self.center, size=self.size)
        elif self.stepped:
            pass                                # driven by SimulationGPU
        else:
            raise NotImplementedError(f"SensorGPU: type {self.sensor_type}")

    def save(self, sim, path: str):
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, f"{self._name}.npz")
        t = self.sensor_type
        if t == "flux":
            np.savez(out, freqs=np.array(sim.get_flux_freqs(self._handle)),
                     fluxes=np.array(sim.get_fluxes(self._handle)))
            print(f"Saved {out}: {len(self._freqs)} flux freqs")
        elif t == "concentration":
            N = np.array(self.snaps, dtype=np.float32)
            np.savez(out, N=N, times=np.array(self.snap_times),
                     levels=["N1", "N2", "N3", "N4"],
                     snap_interval_fs=float(self.args.get("snap_interval_fs", 10.0)))
            print(f"Saved {out}: N shape {N.shape}")
        elif t.endswith("snap"):
            np.savez(out,
                     t=np.array(self.snap_times),
                     Ex=np.array([s[0] for s in self.snaps]),
                     Ey=np.array([s[1] for s in self.snaps]),
                     Ez=np.array([s[2] for s in self.snaps]))
            print(f"Saved {out}: {len(self.snaps)} snapshots "
                  f"{self.snaps[0][0].shape if self.snaps else ()}")
        elif t == "2Ddft" and sim.dim == 2:
            Ex = sim.get_dft_box(self._handle, "Ex")[None]
            Ey = sim.get_dft_box(self._handle, "Ey")[None]
            Ez = sim.get_dft_box(self._handle, "Ez")[None]
            m = self._handle
            np.savez(out, Ex=Ex, Ey=Ey, Ez=Ez, freqs=self._freqs,
                     i_lo=m["i_lo"], i_hi=m["i_hi"],
                     j_lo=m["j_lo"], j_hi=m["j_hi"])
            print(f"Saved {out}: 2D shape {Ex.shape}")
        else:
            nf = len(self._freqs)
            Ex = np.array([sim.get_dft_array(self._handle, gm.Ex, i) for i in range(nf)])
            Ey = np.array([sim.get_dft_array(self._handle, gm.Ey, i) for i in range(nf)])
            if sim.dim == 3:
                Ez = np.array([sim.get_dft_array(self._handle, gm.Ez, i)
                               for i in range(nf)])
            else:
                Ez = np.zeros_like(Ex)                    # 2D TE: Ez ≡ 0
            np.savez(out, Ex=Ex, Ey=Ey, Ez=Ez, freqs=self._freqs)
            print(f"Saved {out}: shape {Ex.shape}")
