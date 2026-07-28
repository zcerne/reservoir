import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as _run
_run._ensure_simplesim()
import numpy as np
import data_gen._gen_common as gc

PATH = "data/reservoir_clasifications/13_2D_sted_gaintest"
OLD_NPZ = "data/reservoir_clasifications/13_2D_sted/datasets/amp_sweep.npz"
ROWS = [0, 30]

old = np.load(OLD_NPZ)
comps = list(old["components"])
forward, n_strips, is_master = gc.open_reservoir(PATH, comps)

for row in ROWS:
    E = old["inputs"][row]
    print(f"=== row {row}  E={E} ===", flush=True)
    new_out_meep = forward(E)
    if not is_master:
        continue
    gpu = np.load(f"_gaintest_row{row}.npz")
    new_out_gpu = gpu["new_out"]
    n_lam = new_out_meep.size // new_out_gpu.size
    meep_center = new_out_meep.reshape(n_lam, new_out_gpu.size)[n_lam // 2]
    diff = np.abs(meep_center - new_out_gpu)
    rel = diff / (np.abs(meep_center) + 1e-30)
    print(f"  gpumeep max|Ey|={np.abs(new_out_gpu).max():.4g}  meep max|Ey|={np.abs(meep_center).max():.4g}")
    print(f"  max rel diff={rel.max():.4g}  "
         f"RMS ratio={np.linalg.norm(meep_center-new_out_gpu)/(np.linalg.norm(meep_center)+1e-30):.4g}",
         flush=True)
    np.savez(f"_meepcheck_row{row}.npz", meep_out=meep_center, gpu_out=new_out_gpu, E=E)
if is_master:
    print("DONE", flush=True)
