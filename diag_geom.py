"""Dump mirror layer positions + sensor position in BOTH engines for config 4."""
import os, sys, numpy as np
sys.path.insert(0, ".")
from ladder.ladder import build_json
p = build_json(4)

# --- MEEP geometry ---
from class_simulation import Simulation
s = Simulation(p)
s._set_data()
s._set_object_list()
print("=== MEEP ===")
for obj in s.objects:
    cls = type(obj).__name__
    if cls == "Mirror":
        print("MEEP mirror front_edge=%.4f direction=%s n_layers=%d" %
              (obj.front_edge, getattr(obj, "direction", "?"), obj.n_of_layers))
        for L in obj.layers[:4]:
            c = L.size.x
            print("   layer center? size.x=%.4f" % c)
# sensor
for sen in s.sensors:
    if sen._name == "monitor_2":
        print("MEEP monitor_2 center=%s size=%s" % (sen.center, sen.size))

# --- gpumeep geometry ---
import importlib
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
sg = csg.SimulationGPU(folder_path=p)
sg._set_data(); sg._update_all_args()
print("=== gpumeep ===")
for obj in sg.objects_args:
    if obj.get("class") == "mirror":
        print("gpu mirror %s x_start_meep=%.4f size_x=%.4f" %
              (obj["_key"], obj["x_start_meep"], obj["size_x"]))
        for (x0, x1, n) in sg._mirror_layers(obj)[:4]:
            print("   layer x=[%.4f, %.4f] n=%.2f" % (x0, x1, n))
    if obj.get("class") == "monitor" and obj["_key"] == "monitor_2":
        print("gpu monitor_2 center_x_meep=%.4f size_y=%.4f" %
              (obj.get("center_x_meep", -999), obj.get("size_y_meep", -999)))
print("gpu cell_x=%.4f  MEEP cell_x=%.4f" % (sg.cell_x, s._cell_x))
