import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
os.environ["LADDER_RUN_UNTIL"] = "200"

from ladder.ladder import build_json, ensure_lc
p = build_json(3)
ensure_lc(p)

# MEEP
from class_simulation import Simulation
print("running MEEP config 3 (dye+pump)...")
sm = Simulation(p); sm.run_simulation()
mm = np.asarray(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
print("MEEP |Ey| max=%.4f mean=%.4f" % (np.abs(mm).max(), np.abs(mm).mean()))
import shutil
shutil.copy(p + "/simulation/monitor_2.npz", p + "/simulation/monitor_2_meep.npz")

# gpumeep
print("running gpumeep config 3...")
import jax
jax.config.update("jax_enable_x64", True)
import importlib
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
sg = csg.SimulationGPU(folder_path=p, force_fullvector=True)
sg.run()
gg = np.asarray(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
print("gpumeep |Ey| max=%.4f mean=%.4f" % (np.abs(gg).max(), np.abs(gg).mean()))
shutil.copy(p + "/simulation/monitor_2.npz", p + "/simulation/monitor_2_gpumeep.npz")

# compare
m = np.asarray(mm).ravel(); g = np.asarray(gg).ravel()
n = min(len(m), len(g))
ym = np.linspace(-3, 3, len(m)); yg = np.linspace(-3, 3, len(g))
gi = np.interp(ym, yg, g.real) + 1j * np.interp(ym, yg, g.imag)
tr = int(0.05 * n); a = m[tr:-tr]; b = gi[tr:-tr]
cc = np.abs(np.vdot(b, a)) / (np.linalg.norm(b) * np.linalg.norm(a))
print("complex-corr=%.4f max-ratio=%.4f" % (cc, np.abs(b).max()/np.abs(a).max()))
print("DONE")
