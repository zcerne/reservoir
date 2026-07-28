import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True)
ap.add_argument("--levels", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

import data_gen._gen_common as gc
forward, n_strips, is_master = gc.open_reservoir(args.path, ["Ey"])
assert n_strips == 2, n_strips

levels = [float(x) for x in args.levels.split(",")]
results = []
for lv in levels:
    E = np.array([lv, lv])
    out = forward(E)
    out_norm = float(np.linalg.norm(out))
    print(f"[single-source] level={lv:g}  |out|={out_norm:.6g}  gain={out_norm/lv:.6g}", flush=True)
    results.append((lv, out_norm))

arr = np.array(results)
np.savez(args.out, levels=arr[:, 0], out_norm=arr[:, 1])
print("saved", args.out)
