"""Config 4 shows a global phase gpu-vs-MEEP (-4.3deg at run_until=120, res40).
Is it (a) ring-down truncation (both under-converged) → shrinks with run_until,
or (b) resonance-condition discretization → shrinks with resolution, or
(c) a fixed code phase → constant. Measure global phase vs both knobs."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_gpumeep, run_meep  # noqa: E402


def gphase(path_m, path_g):
    def load(p):
        E = np.asarray(np.load(p)["Ey"]); return E.reshape(-1) if E.ndim == 1 else E[0]
    m = load(path_m); g = load(path_g)
    n = min(len(m), len(g)); m = m[(len(m)-n)//2:(len(m)-n)//2+n]; g = g[(len(g)-n)//2:(len(g)-n)//2+n]
    ip = np.vdot(m, g)
    return np.degrees(np.angle(ip)), abs(ip)/(np.linalg.norm(m)*np.linalg.norm(g)), np.abs(g).max()/np.abs(m).max()


import shutil
sim = "/home/cernez/resevoir/data/ladder/config_4_mirrors_air/simulation"
for res in [40, 80]:
    os.environ["LADDER_RES"] = str(res)
    for ru in [120, 300, 600]:
        os.environ["LADDER_RUN_UNTIL"] = str(ru)
        path = build_json(4)
        run_meep(path); shutil.copy(sim+"/monitor_2.npz", sim+"/monitor_2_meep.npz")
        run_gpumeep(path); shutil.copy(sim+"/monitor_2.npz", sim+"/monitor_2_gpumeep.npz")
        ph, corr, ratio = gphase(sim+"/monitor_2_meep.npz", sim+"/monitor_2_gpumeep.npz")
        print(f"res={res} run_until={ru}: global_phase={ph:+.3f}deg |corr|={corr:.5f} ampratio={ratio:.4f}",
              flush=True)
print("DONE_CFG4PH")
