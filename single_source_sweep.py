"""Input-output sweep: drive every source strip at the same amplitude and
record the output norm, one full FDTD run per level.

    python single_source_sweep.py --path data/lasing_testing/03_n2f_testing
    python single_source_sweep.py --path <design> --levels 0.1,1,10,100

Saves to <path>/datasets/single_source_sweep.npz (same convention as
data_gen/generate_*.py) unless --out overrides it: {levels, out_norm, gain}.
"""
import os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True)
ap.add_argument("--levels", default="0.1,0.3,1,3,10,30,100")
ap.add_argument("--out", default=None,
                help="output npz (default <path>/datasets/single_source_sweep.npz)")
ap.add_argument("--assemble", action="store_true",
                help="don't run FDTD: merge per-level npz (default glob "
                     "<path>/datasets/sat4src_*.npz, each {levels,out_norm}) "
                     "into one sorted {levels,out_norm,gain} at --out.")
ap.add_argument("--glob", default=None,
                help="assemble glob (default <path>/datasets/sat4src_*.npz)")
args = ap.parse_args()

out_path = args.out or os.path.join(args.path, "datasets",
                                    "single_source_sweep.npz")

if args.assemble:
    import glob
    pat = args.glob or os.path.join(args.path, "datasets", "sat4src_*.npz")
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f"no per-level npz matched {pat}")
    lv, on = [], []
    for f in files:
        d = np.load(f)
        lv.extend(np.atleast_1d(d["levels"]).tolist())
        on.extend(np.atleast_1d(d["out_norm"]).tolist())
    lv = np.array(lv, float); on = np.array(on, float)
    o = np.argsort(lv); lv, on = lv[o], on[o]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, levels=lv, out_norm=on, gain=on / lv)
    print(f"assembled {len(files)} files -> {out_path}")
    for a, b in zip(lv, on):
        print(f"  level={a:g}  |out|={b:.6g}  gain={b/a:.6g}")
    sys.exit(0)

import data_gen._gen_common as gc
forward, n_strips, is_master = gc.open_reservoir(args.path, ["Ey"])

levels = [float(x) for x in args.levels.split(",")]
results = []
for lv in levels:
    # every strip at the same level — a single-channel (uniform) input
    E = np.full(n_strips, lv)
    out = forward(E)                      # MPI collective: every rank runs the FDTD
    if not is_master:                     # but forward() returns None off master —
        continue                          # only master has the monitor to reduce
    out_norm = float(np.linalg.norm(out))
    print(f"[single-source] level={lv:g}  |out|={out_norm:.6g}  gain={out_norm/lv:.6g}", flush=True)
    results.append((lv, out_norm))

if is_master:
    arr = np.array(results)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, levels=arr[:, 0], out_norm=arr[:, 1],
             gain=arr[:, 1] / arr[:, 0])
    print("saved", out_path)
