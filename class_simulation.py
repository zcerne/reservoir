import numpy as np
import json
import os
import meep as mp
from class_guide import Guide
from class_source import Source
from class_sensor import Sensor
from class_reservoir import Reservoir as LCReservoir


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
        self.resolution: int = 40
        self.snapshots: dict[str, list] = {"x": [], "y": [], "z": []}

    def _set_data(self):
        simulation_data_path = os.path.join(self.folder_path, "simulation_data.json")
        with open(simulation_data_path, "r") as f:
            self.args.update(json.load(f))

        amp = self._source_amplitude()
        amp_label = f"amp_{int(amp)}" if amp == int(amp) else f"amp_{amp}"
        amp_dir = self.args.pop("_amp_dir_override", None) or os.path.join(self.folder_path, amp_label)
        self.paths = {
            "simulation": os.path.join(amp_dir, "simulation"),
            "snapshots":  os.path.join(amp_dir, "simulation", "snapshots"),
            "figures":    os.path.join(amp_dir, "figures"),
        }
        os.makedirs(self.paths["simulation"], exist_ok=True)

    def _source_amplitude(self) -> float:
        for key in self.args.get("object_order", []):
            obj = self.args.get(key, {})
            if obj.get("class") == "source":
                return float(obj.get("amplitude", 1.0))
        return 1.0

    def _set_simulation_parameters(self):
        self.resolution = self.args["resolution"]

    @staticmethod
    def _pos_to_center_size(pos, on_edge_x, on_size_x, cell_x, cell_y):
        if isinstance(pos, dict):
            orientation = pos.get("orientation", "vertical")
            label       = pos.get("position", "center")
            req_size    = float(pos.get("size", 0.0))
        else:
            orientation = "vertical"
            label       = str(pos) if pos else "center"
            req_size    = 0.0

        if orientation == "vertical":
            x = {"left": on_edge_x, "right": on_edge_x + on_size_x}.get(
                label, on_edge_x + on_size_x / 2)
            return mp.Vector3(x, 0, 0), mp.Vector3(0, req_size or cell_y, 0)
        else:
            x = on_edge_x + on_size_x / 2
            y = {"up": cell_y / 2, "down": -cell_y / 2}.get(label, 0.0)
            return mp.Vector3(x, y, 0), mp.Vector3(req_size or on_size_x, 0, 0)

    def _update_all_args(self):
        object_keys = self.args["object_order"]
        pml    = float(self.args.get("pml_size", 2.0))
        dim    = self.args.get("dimention", 1)
        cell_y = float(self.args.get("cell_size_y", 0.0)) if dim > 1 else 4.0 / float(self.args["resolution"])

        current_x = 0.0
        for obj_key in object_keys:
            obj_args = self.args[obj_key]
            obj_args["_key"] = obj_key
            obj_args["edge_x_local"] = current_x
            self.objects_args.append(obj_args)
            current_x += float(obj_args.get("size_x", 0.0))

        cell_x = current_x + 2 * pml
        x0     = -cell_x / 2 + pml
        self._cell_x = cell_x
        self._cell_y = cell_y

        for obj_args in self.objects_args:
            edge_x = obj_args["edge_x_local"] + x0
            size_x = float(obj_args.get("size_x", 0.0))
            cls    = obj_args["class"]

            if cls == "guide":
                obj_args["center"] = mp.Vector3(edge_x + size_x / 2, 0, 0)
                guide_y = float(obj_args.get("size_y", 0.0))
                obj_args["sizes"] = mp.Vector3(size_x, guide_y if guide_y > 0 else mp.inf, mp.inf)

            elif cls == "reservoir":
                obj_args["center"] = mp.Vector3(edge_x + size_x / 2, 0, 0)

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

                if cls == "source":
                    pos = obj_args.get("position", {})
                    center, size = self._pos_to_center_size(pos, on_edge, on_size, cell_x, cell_y)
                    obj_args["center"] = center
                    obj_args["size"]   = size

    def get_object(self, args):
        cls = args["class"]
        if cls == "guide":
            return Guide(args)
        if cls == "monitor":
            return Sensor(args)
        if cls == "reservoir":
            return LCReservoir(self.folder_path)
        return None

    def _set_object_list(self):
        self._update_all_args()
        for obj_args in self.objects_args:
            cls = obj_args["class"]
            if cls == "source":
                self.sources.append(Source(obj_args).return_source_object())
                continue
            obj = self.get_object(obj_args)
            if obj is None:
                continue
            if cls == "monitor":
                self.sensors.append(obj)  # type: ignore[arg-type]
            elif cls == "reservoir":
                obj.run_minimization()  # type: ignore[union-attr]
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
            z = self._cell_y
            self.size_yz[0] = y - 2 * pml
            self.size_yz[1] = z - 2 * pml if z > 2 * pml else 0.0
        self.cell = mp.Vector3(self._cell_x, y, z)

    def _set_geometry(self):
        self.geometry = [
            block
            for obj in self.objects
            if isinstance(obj, Guide)
            for block in obj.get_geometry_blocks()
        ]

    def _set_pmls(self):
        self.pmls = [mp.PML(self.args["pml_size"], mp.X)] if self.args["periodic"] else [mp.PML(self.args["pml_size"])]

    def _add_snapshot(self, sim):
        assert self.cell is not None
        self.snapshots["x"].append(sim.get_array(center=mp.Vector3(0, 0), size=mp.Vector3(self.cell.x, self.cell.y), component=mp.Ex))
        self.snapshots["y"].append(sim.get_array(center=mp.Vector3(0, 0), size=mp.Vector3(self.cell.x, self.cell.y), component=mp.Ey))
        self.snapshots["z"].append(sim.get_array(center=mp.Vector3(0, 0), size=mp.Vector3(self.cell.x, self.cell.y), component=mp.Ez))

    def _set_simulation(self):
        bg_index = self.args.get("background_index", 1.0)
        default_material = mp.Medium(index=bg_index) if bg_index != 1.0 else mp.air
        self.simulation = mp.Simulation(
            cell_size=self.cell,  # pyright: ignore
            boundary_layers=self.pmls,
            geometry=self.geometry,  # pyright: ignore
            sources=self.sources,
            resolution=self.resolution,
            default_material=default_material,
            k_point=mp.Vector3(0, 0, 0) if self.args["periodic"] else False
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
        os.makedirs(self.paths["figures"], exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        self.simulation.plot2D(ax=ax, plot_sources_flag=True, plot_monitors_flag=True, plot_boundaries_flag=True)
        ax.set_title("Simulation setup")
        fig.tight_layout()
        fig.savefig(os.path.join(self.paths["figures"], "setup.png"), dpi=150)
        plt.close(fig)

    def _run_meep_once(self) -> None:
        self.snapshots = {"x": [], "y": [], "z": []}
        run_until = self.args.get("run_until", 200)
        step_funcs = [mp.at_every(self.args["snapshot_time"], self._add_snapshot)]
        for sensor in self.sensors:
            sf = sensor.get_step_func()
            if sf is not None:
                interval, func = sf
                step_funcs.append(mp.at_every(interval, func))
        self.simulation.run(*step_funcs, until=run_until)
        self.simulation.change_sources([])
        self.simulation.run(*step_funcs, until=50)

    def _save_all(self) -> None:
        sim_path = self.paths["simulation"]
        os.makedirs(sim_path, exist_ok=True)
        if self.snapshots["x"]:
            np.savez(
                os.path.join(sim_path, "snapshots.npz"),
                Ex=np.array(self.snapshots["x"]),
                Ey=np.array(self.snapshots["y"]),
                Ez=np.array(self.snapshots["z"]),
            )
        for sensor in self.sensors:
            sensor.save(self.simulation, sim_path)

    def run_simulation(self) -> None:
        self._set_everything()
        assert self.cell is not None
        self.plot_setup()
        self._run_meep_once()
        self._save_all()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/test")
    args = parser.parse_args()
    simulation = Simulation(args.path)
    simulation.run_simulation()
