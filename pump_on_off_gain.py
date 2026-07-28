import os, sys
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.environ["GPUMEEP_PATH"])
import argparse
import numpy as np
from class_simulation_gpu import SimulationGPU

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True)
ap.add_argument("--levels", required=True)
ap.add_argument("--pump", type=float, default=200.0)
args = ap.parse_args()

levels = [float(x) for x in args.levels.split(",")]

def run_one(level, pump_amp):
    sim = SimulationGPU(folder_path=args.path)
    sim._set_data()
    sim._update_all_args()
    sim.amp_override = {"source_1": [level, level], "source_pump": [pump_amp]}
    sim.args = {}
    sim.run()
    m2 = np.load(os.path.join(sim.paths["simulation"], "monitor_2.npz"))
    return float(np.linalg.norm(np.concatenate([m2["Ey"].ravel(), m2["Ex"].ravel()])))

for lv in levels:
    out_on = run_one(lv, args.pump)
    out_off = run_one(lv, 0.0)
    print(f"[pump-gain] level={lv:g}  out_pump_on={out_on:.6g}  out_pump_off={out_off:.6g}  gain={out_on/out_off:.6g}", flush=True)
