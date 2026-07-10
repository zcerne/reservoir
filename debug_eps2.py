import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
from ladder.ladder import build_json, ensure_lc
p = build_json(2)
# DON'T delete — reuse cached field if present
ensure_lc(p)
from class_simulation import Simulation
s = Simulation(p)
s._set_everything()
print("eps_averaging:", s.simulation.eps_averaging)
s.run_simulation()
m = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
print("MEEP max:", m.max())
