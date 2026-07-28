import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)
import numpy as np
from class_simulation import ReservoirSimulation

path = "data/lasing_testing/02_adding_pump"
level = 10.0

def run_one(pump_amp, suffix):
    sim = ReservoirSimulation(
        path, backend="gpumeep", suffix=suffix,
        overrides={"source_1": {"amplitude": [level, level]},
                   "source_pump": {"amplitude": pump_amp}})
    sim.relax()
    sim.run(empty=False, out_name=f"simplesim_test_{suffix}")
    out_dir = sim.data.output_dir("gpumeep")
    npz_path = os.path.join(f"{path}", f"simplesim_test_{suffix}", f"monitor_2_{suffix}.npz")
    if not os.path.exists(npz_path):
        # try default naming convention
        for cand in os.listdir(os.path.join(path, f"simplesim_test_{suffix}")):
            print("  found file:", cand)
    m2 = np.load(npz_path)
    Ey = np.asarray(m2["Ey"]); Ex = np.asarray(m2["Ex"]) if "Ex" in m2.files else np.zeros_like(Ey)
    return float(np.linalg.norm(np.concatenate([Ey.ravel(), Ex.ravel()])))

out_on = run_one(200.0, "on")
out_off = run_one(0.0, "off")
print(f"[simplesim-gain] level={level:g}  out_on={out_on:.6g}  out_off={out_off:.6g}  gain={out_on/out_off:.6g}")
