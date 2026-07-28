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
    old_out = old["outputs"][row]
    print(f"=== row {row}  level_id={int(old['level_id'][row])}  "
         f"level={float(old['levels'][int(old['level_id'][row])]):.3g}  E={E} ===",
         flush=True)
    new_out_full = forward(E)
    if new_out_full.shape != old_out.shape and new_out_full.size % old_out.size == 0:
        # Old dataset stored a single-frequency slice (802,); current
        # forward() concatenates ALL n_lam monitor_2 frequencies raveled
        # together (n_lam*802,) -- a monitor-config convention change since
        # this dataset was generated (2026-07-07), unrelated to the gain fix.
        # Reshape and take the CENTER frequency bin (source_1.lam sits at
        # the middle of monitor_2's lam_range) for a like-for-like compare.
        n_lam = new_out_full.size // old_out.size
        new_out = new_out_full.reshape(n_lam, old_out.size)[n_lam // 2]
        print(f"  [shape mismatch: old=(802,) new=({new_out_full.shape[0]},) "
             f"-> reshaped new to ({n_lam},802), using center freq bin {n_lam//2}]")
    else:
        new_out = new_out_full
    diff = np.abs(new_out - old_out)
    rel = diff / (np.abs(old_out) + 1e-30)
    print(f"  old max|Ey|={np.abs(old_out).max():.4g}  new max|Ey|={np.abs(new_out).max():.4g}")
    print(f"  max rel diff={rel.max():.4g}  RMS ratio={np.linalg.norm(new_out-old_out)/(np.linalg.norm(old_out)+1e-30):.4g}",
         flush=True)
    np.savez(f"_gaintest_row{row}.npz", old_out=old_out, new_out=new_out, E=E)
print("DONE", flush=True)
