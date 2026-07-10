import os, sys, numpy as np
sys.path.insert(0, "/home/cernez/resevoir")
from ladder.ladder import build_json
p = build_json(2)
# use the EXISTING cached field (no relax)
from class_simulation import Simulation
for label, val in [("ON (default)", None), ("OFF", "1")]:
    if val: os.environ["MEEP_NO_SUBPIXEL"] = val
    elif "MEEP_NO_SUBPIXEL" in os.environ: del os.environ["MEEP_NO_SUBPIXEL"]
    s = Simulation(p)
    s.run_simulation()
    m = np.abs(np.load(p + "/simulation/monitor_2.npz")["Ey"]).flatten()
    print("eps_averaging %s: MEEP |Ey| max=%.2f" % (label, m.max()))
