"""Config 4 global phase (gpu vs MEEP) vs RESOLUTION, at fixed run_until=120
(window already ruled out). Discriminator:
  phase ∝ 1/res (halves 40→80→...)  → a half-timestep / dt-scaled reference bug
  phase → const                      → fixed cavity round-trip (structural)
  phase → 0 as O(1/res²)             → resonance discretisation, converges
Also prints MEEP vs gpu total timestep counts to confirm identical run time.
"""
import os, re, sys, shutil, subprocess  # noqa: F401
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_gpumeep, run_meep  # noqa: E402

sim = "/home/cernez/resevoir/data/ladder/config_4_mirrors_air/simulation"
os.environ["LADDER_RUN_UNTIL"] = "120"


def gphase(a, b):
    def load(p):
        E = np.asarray(np.load(p)["Ey"]); return E.reshape(-1) if E.ndim == 1 else E[0]
    m = load(a); g = load(b)
    n = min(len(m), len(g)); m = m[(len(m)-n)//2:(len(m)-n)//2+n]; g = g[(len(g)-n)//2:(len(g)-n)//2+n]
    ip = np.vdot(m, g)
    return np.degrees(np.angle(ip)), abs(ip)/(np.linalg.norm(m)*np.linalg.norm(g))


for res in [40, 80, 120]:
    os.environ["LADDER_RES"] = str(res)
    path = build_json(4)
    run_meep(path); shutil.copy(sim+"/monitor_2.npz", sim+"/monitor_2_meep.npz")
    run_gpumeep(path); shutil.copy(sim+"/monitor_2.npz", sim+"/monitor_2_gpumeep.npz")
    ph, corr = gphase(sim+"/monitor_2_meep.npz", sim+"/monitor_2_gpumeep.npz")
    print(f"res={res}: global_phase={ph:+.4f}deg  |corr|={corr:.6f}", flush=True)
print("DONE_CFG4PHRES")
