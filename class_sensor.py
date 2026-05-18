import meep as mp
import numpy as np
import os


class Sensor:
    """MEEP monitor/sensor — computes own geometry from JSON args.

    Required args (injected by _update_all_args before construction):
        "on_object_edge_x"  — left edge x of the referenced object (MEEP coords)
        "on_object_size_x"  — x-width of the referenced object
        "cell_x", "cell_y"  — full cell dimensions

    Optional args:
        "type"      — "flux" | "2Ddft" | "1Ddft" | "2Dsnap" | "point"
        "position"  — "left" | "center" | "right" (vertical) or
                      dict with {"orientation": "horizontal"|"vertical",
                                 "position": "up"|"down"|"center"|"left"|"right",
                                 "size": 0}
        "lam_range" — [lam_min, lam_max] for flux/dft monitors
        "n_lam"     — number of frequency samples
        "dt"        — snapshot/point recording interval
        "name"      — filename stem for saved output
    """

    def __init__(self, args: dict) -> None:
        self.args = args
        self.sensor_type: str = args.get("type", "flux")
        self._name: str = args.get("name", args.get("_key", self.sensor_type))
        self._monitor_handle = None
        self._dft_freqs: list = []
        self._snapshot_data: list = []
        self._point_data: list = []
        self._point_times: list = []

        self.center: mp.Vector3 = self._compute_center()
        self.size: mp.Vector3 = self._compute_size()

    # ------------------------------------------------------------------
    # Geometry — all position/size logic lives here
    # ------------------------------------------------------------------

    def _parse_position(self) -> tuple[str, str, float]:
        """Return (orientation, position_label, requested_size)."""
        pos = self.args.get("position", {})
        if isinstance(pos, dict):
            orientation = pos.get("orientation", "vertical")
            label = pos.get("position", "center")
            req_size = float(pos.get("size", 0))
        else:
            orientation = "vertical"
            label = str(pos) if pos else "center"
            req_size = 0.0
        return orientation, label, req_size

    def _compute_center(self) -> mp.Vector3:
        orientation, label, _ = self._parse_position()
        edge_x   = float(self.args.get("on_object_edge_x", 0.0))
        size_x   = float(self.args.get("on_object_size_x", 0.0))
        cell_y   = float(self.args.get("cell_y", 0.0))

        if orientation == "vertical":
            x = {"left": edge_x, "right": edge_x + size_x}.get(label, edge_x + size_x / 2)
            return mp.Vector3(x, 0, 0)
        else:  # horizontal
            x = edge_x + size_x / 2
            y = {"up": cell_y / 2, "down": -cell_y / 2}.get(label, 0.0)
            return mp.Vector3(x, y, 0)

    def _compute_size(self) -> mp.Vector3:
        orientation, _, req_size = self._parse_position()
        cell_x   = float(self.args.get("cell_x", 0.0))
        cell_y   = float(self.args.get("cell_y", 0.0))
        size_x   = float(self.args.get("on_object_size_x", 0.0))
        on_object = self.args.get("on_object", -1)

        if self.sensor_type in ("2Ddft", "2Dsnap") and on_object == -1:
            return mp.Vector3(cell_x, cell_y, 0)

        if orientation == "vertical":
            return mp.Vector3(0, req_size if req_size > 0 else cell_y, 0)
        else:
            return mp.Vector3(req_size if req_size > 0 else size_x, 0, 0)

    # ------------------------------------------------------------------
    # MEEP registration
    # ------------------------------------------------------------------

    def add_to_simulation(self, sim: mp.Simulation) -> None:
        if self.sensor_type == "flux":
            lam_range = self.args.get("lam_range", [0.4, 0.7])
            n_lam     = self.args.get("n_lam", 100)
            f_min = 1.0 / lam_range[1]
            f_max = 1.0 / lam_range[0]
            fcen  = (f_min + f_max) / 2
            fwidth = f_max - f_min
            self._monitor_handle = sim.add_flux(
                fcen, fwidth, n_lam,
                mp.FluxRegion(center=self.center, size=self.size)
            )

        elif self.sensor_type in ("2Ddft", "1Ddft"):
            lam_range = self.args.get("lam_range", [0.4, 0.7])
            n_lam     = self.args.get("n_lam", 20)
            self._dft_freqs = list(np.linspace(1.0 / lam_range[1], 1.0 / lam_range[0], n_lam))
            self._monitor_handle = sim.add_dft_fields(
                [mp.Ez], self._dft_freqs, center=self.center, size=self.size
            )

        # "2Dsnap" and "point" collect data via step functions — nothing to register here.

    def get_step_func(self) -> tuple[float, object] | None:
        """Return (interval, step_func) for mp.at_every, or None."""
        if self.sensor_type == "2Dsnap":
            dt = float(self.args.get("dt", 10.0))
            return dt, self._record_snapshot
        if self.sensor_type == "point":
            dt = float(self.args.get("dt", 0.1))
            return dt, self._record_point
        return None

    # ------------------------------------------------------------------
    # Step functions (called by MEEP runner)
    # ------------------------------------------------------------------

    def _record_snapshot(self, sim: mp.Simulation) -> None:
        data = sim.get_array(center=self.center, size=self.size, component=mp.Ez)
        self._snapshot_data.append((sim.meep_time(), data.copy()))

    def _record_point(self, sim: mp.Simulation) -> None:
        val = sim.get_field_point(mp.Ez, self.center)
        self._point_data.append(float(np.real(val)))
        self._point_times.append(sim.meep_time())

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

        elif self.sensor_type in ("2Ddft", "1Ddft") and self._monitor_handle is not None:
            Ez = np.array([
                sim.get_dft_array(self._monitor_handle, mp.Ez, i)
                for i in range(len(self._dft_freqs))
            ])
            np.savez(out, Ez=Ez, freqs=np.array(self._dft_freqs))

        elif self.sensor_type == "2Dsnap" and self._snapshot_data:
            np.savez(out,
                     t=np.array([t for t, _ in self._snapshot_data]),
                     Ez=np.array([d for _, d in self._snapshot_data]))

        elif self.sensor_type == "point" and self._point_data:
            np.savez(out,
                     t=np.array(self._point_times),
                     Ez=np.array(self._point_data))
