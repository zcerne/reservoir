"""Verify the MEEP-exact source weighting removes the resolution drift.

Runs config 1 (air) MEEP vs gpu (default src_scale = C·res, now with fractional
cell weighting) across resolutions. If the fix is right, |Ey|_gpu/|Ey|_meep is
now RESOLUTION-FLAT (a single constant), so no per-res tuning is possible or
needed. Also reports the central-region ratio + phase for a per-pixel check.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_meep, run_gpumeep  # noqa: E402

YOFF = os.environ.get("GPUMEEP_SRC_YOFF", "(default 0.5 for Ey)")
print(f"y-offset = {YOFF}")
rows = []
for res in [20, 40, 80]:
    os.environ["LADDER_RES"] = str(res)
    path = build_json(1)
    a = np.asarray(run_meep(path)).reshape(-1).astype(complex)
    b = np.asarray(run_gpumeep(path)).reshape(-1).astype(complex)
    n = min(len(a), len(b))
    a = a[(len(a) - n) // 2:(len(a) - n) // 2 + n]
    b = b[(len(b) - n) // 2:(len(b) - n) // 2 + n]
    rmax = np.abs(b).max() / np.abs(a).max()
    c0, c1 = n // 4, 3 * n // 4
    rc = (np.abs(b)[c0:c1] / (np.abs(a)[c0:c1] + 1e-30)).mean()
    ph = np.degrees(np.angle((b[c0:c1] / a[c0:c1]).mean()))
    rows.append((res, np.abs(a).max(), np.abs(b).max(), rmax, rc, ph))
    print(f"RES={res}: |Ey|meep={np.abs(a).max():.5f} |Ey|gpu={np.abs(b).max():.5f} "
          f"ratio_max={rmax:.5f} central_ratio={rc:.5f} phase={ph:+.3f}deg", flush=True)

r = np.array([x[0] for x in rows], float)
rm = np.array([x[3] for x in rows], float)
print("\n--- resolution flatness of gpu/meep amplitude ratio ---")
print(f"  ratio_max across res: {rm}")
print(f"  spread max-min = {rm.max() - rm.min():.5f} ({100*(rm.max()-rm.min())/rm.mean():.2f}% of mean)")
b_slope, _ = np.polyfit(np.log(r), np.log(rm), 1)
print(f"  log-log slope of ratio vs res = {b_slope:+.4f}  (0 => perfectly res-flat)")
print(f"  => single constant to set ratio=1: divide src C by {rm.mean():.5f}")
print("DONE_SRC_FIX")
