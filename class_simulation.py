import numpy as np
import json
import os
import meep as mp
from class_guide import Guide
from class_source import Source
from class_sensor import Sensor
from class_reservoir import Reservoir as LCReservoir
from class_slm import SLM
from class_mirror import Mirror


class Simulation:
    def __init__(self, args_path) -> None:
        self.folder_path = args_path
        self.paths: dict[str, str] = {}
        self.args: dict = {}
        self.objects_args: list[dict] = []
        self.objects: list = []
        self.sources: list = []
        self.sensors: list[Sensor] = []
        self.cell: mp.Vector3 | None = None
        self.size_yz = [0.0, 0.0]
        self._cell_x = 0.0
        self._cell_y = 0.0
        self._cell_z = 0.0
        self.resolution: int = 40
        self.snapshots: dict[str, list] = {"x": [], "y": [], "z": [], "t": []}
        self._empty: bool = False

    def _set_data(self):
        simulation_data_path = os.path.join(self.folder_path, "simulation_data.json")
        with open(simulation_data_path, "r") as f:
            self.args.update(json.load(f))

        sim_dir = "simulation_empty" if self._empty else "simulation"
        self.paths = {
            "simulation": os.path.join(self.folder_path, sim_dir),
            "snapshots":  os.path.join(self.folder_path, sim_dir, "snapshots"),
            "figures":    os.path.join(self.folder_path, "figures"),
        }
        for p in self.paths.values():
            os.makedirs(p, exist_ok=True)

    def _set_simulation_parameters(self):
        self.resolution = self.args["resolution"]

    @staticmethod
    def _pos_to_center_size(pos, on_edge_x, on_size_x, cell_x, cell_y, cell_z=0.0):
        if isinstance(pos, dict):
            label = pos.get("position", "center")
            raw   = pos.get("size", [0.0, 0.0])
        else:
            label = str(pos) if pos else "center"
            raw   = [0.0, 0.0]
        if isinstance(raw, (int, float)):
            raw = [float(raw), 0.0]
        # Source extent convention:
        #   [y, z]      (len<=2)  → plane ⊥ propagation, x-extent 0 (legacy: source_1 etc.)
        #   [x, y, z]   (len==3)  → full box; x-extent honored → AREA/VOLUME source
        #                           (e.g. a pump filling the reservoir in the xy plane).
        if len(raw) >= 3:
            sx = float(raw[0])
            sy = float(raw[1]) if raw[1] else cell_y
            sz = float(raw[2])
        else:
            sx = 0.0
            sy = float(raw[0]) if raw[0] else cell_y
            sz = float(raw[1]) if len(raw) > 1 else 0.0
        x = {"left": on_edge_x, "right": on_edge_x + on_size_x}.get(
            label, on_edge_x + on_size_x / 2)
        return mp.Vector3(x, 0, 0), mp.Vector3(sx, sy, sz)

    def _update_all_args(self):
        object_keys = self.args["object_order"]
        pml    = float(self.args.get("pml_size", 2.0))
        dim    = self.args.get("dimention", 1)
        cell_y = float(self.args.get("cell_size_y", 0.0)) if dim > 1 else 4.0 / float(self.args["resolution"])
        cell_z = float(self.args.get("cell_size_z", 0.0)) if dim > 2 else 0.0

        current_x = 0.0
        for obj_key in object_keys:
            obj_args = self.args[obj_key]
            obj_args["_key"] = obj_key
            obj_args["edge_x_local"] = current_x
            if isinstance(obj_args.get("sizes"), list):
                obj_args["size_x"] = float(obj_args["sizes"][0])
            cls = obj_args.get("class", "")
            if cls == "slm" and "size_x" not in obj_args:
                n_o, n_e = float(obj_args["no_ne"][0]), float(obj_args["no_ne"][1])
                obj_args["size_x"] = float(obj_args["lam"]) / (2 * (n_e - n_o))
            elif cls == "mirror" and "size_x" not in obj_args:
                lam     = float(obj_args["lam"])
                indices = obj_args.get("n_indexes", obj_args.get("indexes", [1.0, 1.0]))
                if "n_layers" in obj_args:
                    n_lays = int(obj_args["n_layers"])
                else:
                    n_lays = Mirror._n_layers_for_transmission(float(obj_args["transmission"]), indices)
                obj_args["size_x"] = sum(lam / 4.0 / float(indices[i % 2]) for i in range(n_lays))
            self.objects_args.append(obj_args)
            current_x += float(obj_args.get("size_x", 0.0))

        cell_x = current_x + 2 * pml
        x0     = -cell_x / 2 + pml
        self._cell_x = cell_x
        self._cell_y = cell_y
        self._cell_z = cell_z

        for obj_args in self.objects_args:
            edge_x = obj_args["edge_x_local"] + x0
            size_x = float(obj_args.get("size_x", 0.0))
            cls    = obj_args["class"]

            if cls == "guide":
                obj_args["center"] = mp.Vector3(edge_x + size_x / 2, 0, 0)
                sizes_raw = obj_args.get("sizes")
                guide_y = float(sizes_raw[1]) if isinstance(sizes_raw, list) and len(sizes_raw) > 1 else float(obj_args.get("size_y", 0.0))
                obj_args["sizes"] = mp.Vector3(size_x, guide_y if guide_y > 0 else mp.inf, mp.inf)

            elif cls in ("reservoir", "voltage_reservoir"):
                obj_args["center"] = mp.Vector3(edge_x + size_x / 2, 0, 0)

            elif cls == "slm":
                obj_args["center"] = mp.Vector3(edge_x + size_x / 2, 0, 0)
                if "size_y" not in obj_args:
                    obj_args["size_y"] = cell_y
                if "size_z" not in obj_args:
                    obj_args["size_z"] = cell_z

            elif cls == "mirror":
                obj_args["x_start"]    = edge_x
                obj_args["resolution"] = self.resolution
                obj_args["cell_y"]     = cell_y
                if "size_y" not in obj_args:
                    obj_args["size_y"] = cell_y

            elif cls in ("monitor", "source"):
                on_object = obj_args.get("on_object", -1)
                if on_object == -1 and isinstance(obj_args.get("position"), dict):
                    on_object = obj_args["position"].get("on_object", -1)
                if isinstance(on_object, int) and on_object >= 0:
                    ref = self.objects_args[on_object]
                elif isinstance(on_object, str):
                    ref = next((r for r in self.objects_args if r.get("_key") == on_object), None)
                else:
                    ref = None

                on_edge = ref["edge_x_local"] + x0 if ref else -cell_x / 2
                on_size = float(ref.get("size_x", 0.0)) if ref else cell_x

                obj_args["on_object_edge_x"] = on_edge
                obj_args["on_object_size_x"] = on_size
                obj_args["cell_x"]           = cell_x
                obj_args["cell_y"]           = cell_y
                obj_args["cell_z"]           = cell_z

                if cls == "source":
                    pos = obj_args.get("position", {})
                    center, size = self._pos_to_center_size(pos, on_edge, on_size, cell_x, cell_y, cell_z)
                    obj_args["center"] = center
                    obj_args["size"]   = size

    def get_object(self, args):
        cls = args["class"]
        if cls == "guide":
            return Guide(args)
        if cls == "source":
            return Source(args)
        if cls == "monitor":
            return Sensor(args)
        if cls in ("reservoir", "voltage_reservoir"):
            if self._empty:
                return None  # reservoir region stays as air background
            return LCReservoir(self.folder_path)
        if cls == "slm":
            return SLM(args)
        if cls == "mirror":
            return Mirror(args)
        return None

    def _set_object_list(self):
        self._update_all_args()
        for obj_args in self.objects_args:
            obj = self.get_object(obj_args)
            if obj is None:
                continue
            cls = obj_args["class"]
            if cls == "source":
                self.sources.extend(obj.return_source_object())  # type: ignore[union-attr]
            elif cls == "monitor":
                self.sensors.append(obj)  # type: ignore[arg-type]
            elif cls in ("reservoir", "voltage_reservoir"):
                fields_file = os.path.join(self.folder_path, "simulation", "lc_fields.npz")
                if os.path.exists(fields_file):
                    obj.load_fields()  # type: ignore[union-attr]
                else:
                    obj.run_minimization()  # type: ignore[union-attr]
                obj._meep_center_x = float(obj_args["center"].x)  # type: ignore[union-attr]
                self.objects.append(obj)
            else:
                self.objects.append(obj)

    def _set_cell(self):
        dim = self.args.get("dimention", 1)
        pml = float(self.args.get("pml_size", 2.0))
        if dim == 1:
            y = 4.0 / self.args["resolution"]
            z = 0.0
            self.size_yz[0] = y
        elif dim == 2:
            y = self._cell_y
            z = 0.0
            self.size_yz[0] = y - 2 * pml
        else:
            y = self._cell_y
            z = self._cell_z if self._cell_z > 0 else self._cell_y
            self.size_yz[0] = y - 2 * pml
            self.size_yz[1] = z - 2 * pml if z > 2 * pml else 0.0
        self.cell = mp.Vector3(self._cell_x, y, z)

    def _set_geometry(self):
        self.geometry = [
            block
            for obj in self.objects
            for block in obj.get_geometry_blocks()
        ]
        # Media that appear only inside a material function (e.g. the reservoir's
        # STED MultilevelAtom) must be declared to mp.Simulation(extra_materials=...)
        # or MEEP won't allocate their polarization fields → susceptibility ignored.
        self.extra_materials = [
            m
            for obj in self.objects
            if hasattr(obj, "get_extra_materials")
            for m in obj.get_extra_materials()
        ]

    def _set_pmls(self):
        self.pmls = [mp.PML(self.args["pml_size"], mp.X)] if self.args["periodic"] else [mp.PML(self.args["pml_size"])]

    def _add_snapshot(self, sim):
        assert self.cell is not None
        region = mp.Vector3(self.cell.x, self.cell.y, self.cell.z)
        self.snapshots["x"].append(sim.get_array(center=mp.Vector3(), size=region, component=mp.Ex))
        self.snapshots["y"].append(sim.get_array(center=mp.Vector3(), size=region, component=mp.Ey))
        self.snapshots["z"].append(sim.get_array(center=mp.Vector3(), size=region, component=mp.Ez))
        self.snapshots["t"].append(sim.meep_time())

    def _make_snap_func(self):
        if "snapshot_t1" in self.args:
            t1 = float(self.args["snapshot_t1"])
            t2 = float(self.args["snapshot_t2"])
            dt = float(self.args["snapshot_dt"])
            next_t = [t1]
            def windowed(sim):
                t = sim.meep_time()
                while next_t[0] <= t2 + 1e-9 and t >= next_t[0] - 1e-9:
                    self._add_snapshot(sim)
                    next_t[0] += dt
            return windowed
        return mp.at_every(self.args["snapshot_time"], self._add_snapshot)

    def _set_simulation(self):
        bg_index = self.args.get("background_index", 1.0)
        default_material = mp.Medium(index=bg_index) if bg_index != 1.0 else mp.air
        use_cw = bool(self.args.get("use_cw", False))
        self.simulation = mp.Simulation(
            cell_size=self.cell,  # pyright: ignore
            boundary_layers=self.pmls,
            geometry=self.geometry,  # pyright: ignore
            sources=self.sources,
            resolution=self.resolution,
            default_material=default_material,
            extra_materials=getattr(self, "extra_materials", []),
            k_point=mp.Vector3(0, 0, 0) if self.args["periodic"] else False,
            force_complex_fields=use_cw,
        )

    def _setup_sensors(self):
        for sensor in self.sensors:
            sensor.add_to_simulation(self.simulation)

    def _set_everything(self):
        self._set_data()
        self._set_simulation_parameters()
        self._set_object_list()
        self._set_pmls()
        self._set_cell()
        self._set_geometry()
        self._set_simulation()
        self._setup_sensors()

    def plot_setup(self) -> None:
        if not mp.am_master():
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        assert self.cell is not None
        is_3d = self._cell_z > 0
        fig, ax = plt.subplots(figsize=(12, 5))
        kw = dict(ax=ax, plot_sources_flag=True, plot_monitors_flag=True, plot_boundaries_flag=True)
        if is_3d:
            kw["output_plane"] = mp.Volume(center=mp.Vector3(), size=mp.Vector3(self._cell_x, self._cell_y, 0))
        self.simulation.plot2D(**kw)
        ax.set_title("Simulation setup" + (" (XY slice)" if is_3d else ""))
        fig.tight_layout()
        suffix = "_empty" if self._empty else ""
        fig.savefig(os.path.join(self.paths["figures"], f"setup{suffix}.png"), dpi=150)
        plt.close(fig)

    def _run_meep_once(self) -> None:
        self.snapshots = {"x": [], "y": [], "z": [], "t": []}
        use_cw = bool(self.args.get("use_cw", False))

        if use_cw:
            # Frequency-domain stationary-state solver.
            # Requires a short initialization run so fields are non-zero, then iterates to convergence.
            # DFT/flux monitors are reset and filled with the converged field automatically.
            cw_init  = float(self.args.get("cw_init_time", 200))
            tol      = float(self.args.get("cw_tol",       1e-6))
            maxiters = int(self.args.get("cw_maxiters",    10000))
            L        = int(self.args.get("cw_L",           10))
            self.simulation.run(until=cw_init)
            self.simulation.solve_cw(tol, maxiters, L)
            return

        run_until = self.args.get("run_until", 200)
        if self._empty:
            # empty (air) run: DFT/flux monitors accumulate in C++ automatically
            self.simulation.run(until=run_until)
            self.simulation.change_sources([])
            self.simulation.run(until=50)
            return
        snap_func = self._make_snap_func()
        sensor_funcs = []
        for sensor in self.sensors:
            sf = sensor.get_step_func()
            if sf is not None:
                interval, func = sf
                sensor_funcs.append(mp.at_every(interval, func))
        self.simulation.run(snap_func, *sensor_funcs, until=run_until)
        self.simulation.change_sources([])
        # source-off decay: sensors only, no snapshots (avoid RAM spike)
        self.simulation.run(*sensor_funcs, until=50)

    def _save_all(self) -> None:
        sim_path = self.paths["simulation"]
        if self.snapshots["x"]:
            np.savez(
                os.path.join(sim_path, "snapshots.npz"),
                Ex=np.array(self.snapshots["x"]),
                Ey=np.array(self.snapshots["y"]),
                Ez=np.array(self.snapshots["z"]),
                t=np.array(self.snapshots["t"]),
            )
        for sensor in self.sensors:
            sensor.save(self.simulation, sim_path)
        for obj in self.objects:
            if isinstance(obj, LCReservoir):
                obj.save_fields()

    def run_simulation(self) -> None:
        self._set_everything()
        assert self.cell is not None
        self.plot_setup()
        self._run_meep_once()
        self._save_all()

    def run_empty(self) -> None:
        self._empty = True
        self.objects_args = []
        self.objects      = []
        self.sources      = []
        self.sensors      = []
        self.args         = {}
        self.snapshots    = {"x": [], "y": [], "z": [], "t": []}
        self.simulation   = None  # free MEEP sim before allocating new one
        import gc; gc.collect()
        self._set_everything()
        assert self.cell is not None
        self.plot_setup()
        self._run_meep_once()
        self._save_all()
        self._empty = False


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
