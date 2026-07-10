import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
from ladder.ladder import build_json, ensure_lc
p = build_json(2)
import shutil; shutil.rmtree(p, ignore_errors=True)
p = build_json(2); ensure_lc(p)
from class_simulation import Simulation
print("MEEP_NO_SUBPIXEL env:", repr(os.environ.get("MEEP_NO_SUBPIXEL")))
s = Simulation(p)
s._set_everything()
print("eps_averaging:", s.simulation.eps_averaging)
s.run_simulation()
m = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
print("MEEP max:", m.max())
