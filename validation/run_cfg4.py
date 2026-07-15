import os, sys, numpy as np, shutil
sys.path.insert(0, "/home/cernez/resevoir")
os.environ["LADDER_RUN_UNTIL"] = "200"
os.environ["GPUMEEP_KOTTKE"] = "1"   # match MEEP eps_averaging on the sharp DBR
from ladder.ladder import build_json
p = build_json(4)
from class_simulation import Simulation
print("MEEP config 4 (mirrors)...")
sm = Simulation(p); sm.run_simulation()
mm = np.asarray(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
shutil.copy(p + "/simulation/monitor_2.npz", p + "/simulation/monitor_2_meep.npz")
print("MEEP |Ey| max=%.4f" % np.abs(mm).max())
import jax; jax.config.update("jax_enable_x64", True)
import importlib
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True); sg.run()
gg = np.asarray(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
shutil.copy(p + "/simulation/monitor_2.npz", p + "/simulation/monitor_2_gpumeep.npz")
print("gpumeep |Ey| max=%.4f" % np.abs(gg).max())
m = mm.ravel(); g = gg.ravel()
ym = np.linspace(-3,3,len(m)); yg = np.linspace(-3,3,len(g))
gi = np.interp(ym,yg,g.real)+1j*np.interp(ym,yg,g.imag)
tr = int(0.05*len(m)); a=m[tr:-tr]; b=gi[tr:-tr]
print("complex-corr=%.4f max-ratio=%.4f" % (np.abs(np.vdot(b,a))/(np.linalg.norm(b)*np.linalg.norm(a)), np.abs(b).max()/np.abs(a).max()))
print("DONE")
