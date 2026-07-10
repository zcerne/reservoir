"""Test: gpumeep Kottke ON vs OFF on config 4 (mirrors — sharp boundaries)."""
import os, sys, numpy as np
sys.path.insert(0, ".")
from ladder.ladder import build_json, ensure_lc
p = build_json(4)
# No LC in config 4 — skip ensure_lc

# MEEP reference
from class_simulation import Simulation
sm = Simulation(p); sm.run_simulation()
mm = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()

# gpumeep area-fraction (no Kottke)
os.environ.pop("GPUMEEP_KOTTKE", None)
sys.modules.pop("class_simulation_gpu", None)
import importlib, jax
jax.config.update("jax_enable_x64", True)
csg = importlib.import_module("class_simulation_gpu")
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg.run()
ga = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()

# gpumeep Kottke ON
os.environ["GPUMEEP_KOTTKE"] = "1"
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
sg2 = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg2.run()
gk = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()

print("MEEP: %.3f   gpu(area-frac): %.3f   gpu(Kottke): %.3f" % (mm.max(), ga.max(), gk.max()))
print("area-frac/MEEP: %.3f   Kottke/MEEP: %.3f" % (ga.max()/mm.max(), gk.max()/mm.max()))
