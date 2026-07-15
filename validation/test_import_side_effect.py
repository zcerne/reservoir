import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("/home/cernez/resevoir"))
os.environ["LADDER_RUN_UNTIL"] = "200"

from ladder.ladder import build_json, ensure_lc
p = build_json(2)

# fresh relax
import shutil
shutil.rmtree(p, ignore_errors=True)
p = build_json(2)
ensure_lc(p)

# MEEP first
from class_simulation import Simulation
s = Simulation(p)
s.run_simulation()
m1 = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
print("MEEP before gpumeep import: %.2f" % m1.max())

# NOW import gpumeep
import jax
jax.config.update("jax_enable_x64", True)
import importlib
sys.path.insert(0, os.path.expanduser("/home/cernez/resevoir"))
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
print("gpumeep imported")

# Run MEEP again
s2 = Simulation(p)
s2.run_simulation()
m2 = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
print("MEEP AFTER gpumeep import: %.2f" % m2.max())

# does gpumeep run affect it too?
os.environ["GPUMEEP_PATH"] = "/home/cernez/GPUmeep/src"
sim = csg.SimulationGPU(folder_path=p)
sim.run()
m3 = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
print("gpumeep |Ey| max: %.2f" % m3.max())

# MEEP after gpumeep run
s3 = Simulation(p)
s3.run_simulation()
m4 = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
print("MEEP AFTER gpumeep RUN: %.2f" % m4.max())
