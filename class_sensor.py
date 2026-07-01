import meep as mp
import numpy as np
import os


class Sensor:
    """MEEP monitor/sensor.

    Types
    -----
    flux                         power (Poynting) through the sensor plane
    0Ddft / 0Dsnap               point
    1Ddft / 1Dsnap               line  in y
    2Ddft / 2Dsnap               plane in yz
    3Ddft / 3Dsnap               volume xyz

    Position
    --------
    'left' | 'center' | 'right'  x-placement on on_object (or full cell if omitted)
    position.size                y-extent in µm; defaults to cell_y

    z-extent always spans the full cell_z (set by class_simulation).
    """

    def __init__(self, args: dict) -> None:
        self.args = args
        self.sensor_type: str = args.get("type", "2Ddft")
        self._name: str = args.get("name", args.get("_key", self.sensor_type))
        self._monitor_handle = None
        self._dft_freqs: list[float] = []
        self._snap_data: list[tuple[float, np.ndarray]] = []

        self.center: mp.Vector3 = self._compute_center()
        self.size: mp.Vector3   = self._compute_size()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _parse_position(self) -> tuple[str, float]:
        """Return (label, requested y-size). For list-form size, applies the
        BlockOpt-style heuristic: size[1] > 0 → [sx, sy] (2D box, sy = size[1]);
        else → [sy] / [sy, 0] (1D plane, sy = size[0]).
        """
        pos = self.args.get("position", {})
        if isinstance(pos, dict):
            label = pos.get("position", "center")
            raw = pos.get("size", 0)
            if isinstance(raw, (int, float)):
                req_size = float(raw)
            elif isinstance(raw, list) and len(raw) == 1:
                req_size = float(raw[0])
            elif isinstance(raw, list) and len(raw) >= 2 and float(raw[1]) > 0:
                req_size = float(raw[1])      # [sx, sy] 2D
            elif isinstance(raw, list) and len(raw) >= 1:
                req_size = float(raw[0])      # [sy, 0] plane
            else:
                req_size = 0.0
        else:
            label    = str(pos) if pos else "center"
            req_size = 0.0
        return label, req_size

    def _compute_center(self) -> mp.Vector3:
        label, _ = self._parse_position()
        edge_x = float(self.args.get("on_object_edge_x", 0.0))
        size_x = float(self.args.get("on_object_size_x", 0.0))
        x = {"left": edge_x, "right": edge_x + size_x}.get(label, edge_x + size_x / 2)
        return mp.Vector3(x, 0, 0)

    def _compute_size(self) -> mp.Vector3:
        _, req_size = self._parse_position()
        cell_x = float(self.args.get("cell_x", 0.0))
        cell_y = float(self.args.get("cell_y", 0.0))
        cell_z = float(self.args.get("cell_z", 0.0))
        obj_sx = float(self.args.get("on_object_size_x", 0.0))
        sy = req_size if req_size > 0 else cell_y
        t  = self.sensor_type

        if t == "flux":
            return mp.Vector3(0, sy, cell_z)
        dim = t[0] if t and t[0].isdigit() else "2"
        if dim == "0":
            return mp.Vector3(0, 0, 0)
        if dim == "1":
            return mp.Vector3(0, sy, 0)
        if dim == "2":
            sx_span = cell_x if cell_z == 0 else 0.0  # full XY for 2D sim; YZ slice for 3D
            return mp.Vector3(sx_span, sy, cell_z)
        # "3" — volume
        sx = obj_sx if obj_sx > 0 else cell_x
        return mp.Vector3(sx, sy, cell_z)

    # ------------------------------------------------------------------
    # MEEP registration
    # ------------------------------------------------------------------

    def add_to_simulation(self, sim: mp.Simulation) -> None:
        if self.sensor_type == "flux":
            lam_range = self.args.get("lam_range", [0.4, 0.7])
            n_lam     = self.args.get("n_lam", 100)
            f_min  = 1.0 / lam_range[1]
            f_max  = 1.0 / lam_range[0]
            fcen   = (f_min + f_max) / 2
            fwidth = f_max - f_min
            self._monitor_handle = sim.add_flux(
                fcen, fwidth, n_lam,
                mp.FluxRegion(center=self.center, size=self.size)
            )
        elif self.sensor_type.endswith("dft"):
            lam_range = self.args.get("lam_range", [0.4, 0.7])
            n_lam     = self.args.get("n_lam", 1)
            self._dft_freqs = list(
                np.linspace(1.0 / lam_range[1], 1.0 / lam_range[0], n_lam)
            )
            self._monitor_handle = sim.add_dft_fields(
                [mp.Ex, mp.Ey, mp.Ez], self._dft_freqs, center=self.center, size=self.size
            )

    def get_step_func(self) -> tuple[float, object] | None:
        if self.sensor_type.endswith("snap"):
            dt = float(self.args.get("dt", 10.0))
            return dt, self._record_snap
        return None

    # ------------------------------------------------------------------
    # Step functions
    # ------------------------------------------------------------------

    def _record_snap(self, sim: mp.Simulation) -> None:
        if self.sensor_type == "0Dsnap":
            val = np.array([float(np.real(sim.get_field_point(mp.Ez, self.center)))])
            self._snap_data.append((sim.meep_time(), val, val, val))
        else:
            kw = dict(center=self.center, size=self.size)
            ex = sim.get_array(component=mp.Ex, **kw).copy()
            ey = sim.get_array(component=mp.Ey, **kw).copy()
            ez = sim.get_array(component=mp.Ez, **kw).copy()
            self._snap_data.append((sim.meep_time(), ex, ey, ez))

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, sim: mp.Simulation, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        out = os.path.join(path, f"{self._name}.npz")

        if self.sensor_type == "flux" and self._monitor_handle is not None:
            np.savez(out,
                     freqs=np.array(mp.get_flux_freqs(self._monitor_handle)),
                     fluxes=np.array(mp.get_fluxes(self._monitor_handle)))

        elif self.sensor_type.endswith("dft") and self._monitor_handle is not None:
            Ex = np.array([sim.get_dft_array(self._monitor_handle, mp.Ex, i) for i in range(len(self._dft_freqs))])
            Ey = np.array([sim.get_dft_array(self._monitor_handle, mp.Ey, i) for i in range(len(self._dft_freqs))])
            Ez = np.array([sim.get_dft_array(self._monitor_handle, mp.Ez, i) for i in range(len(self._dft_freqs))])
            np.savez(out, Ex=Ex, Ey=Ey, Ez=Ez, freqs=np.array(self._dft_freqs))

        elif self.sensor_type.endswith("snap") and self._snap_data:
            np.savez(out,
                     t=np.array([e[0] for e in self._snap_data]),
                     Ex=np.array([e[1] for e in self._snap_data]),
                     Ey=np.array([e[2] for e in self._snap_data]),
                     Ez=np.array([e[3] for e in self._snap_data]))
