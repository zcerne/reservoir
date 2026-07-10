"""Separate source-amplitude vs propagation as the source of the residual drift.
Put the monitor NEAR the source (guide_1 center, ~0 propagation) vs FAR
(guide_2 center, ~7.75um) and measure gpu/meep amplitude ratio across res.
  - ratio ~flat & ~1 near, drifts far  => numerical propagation amplitude
  - ratio already drifts near          => residual source-amplitude scaling
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_meep, run_gpumeep  # noqa: E402

os.environ["LADDER_RUN_UNTIL"] = "120"
for where in ["guide_1", "guide_2"]:
    os.environ["LADDER_MON_OBJ"] = where
    for res in [40, 80]:
        os.environ["LADDER_RES"] = str(res)
        path = build_json(1)
        a = np.abs(np.asarray(run_meep(path)).reshape(-1))
        b = np.abs(np.asarray(run_gpumeep(path)).reshape(-1))
        n = min(len(a), len(b))
        a = a[(len(a) - n) // 2:(len(a) - n) // 2 + n]
        b = b[(len(b) - n) // 2:(len(b) - n) // 2 + n]
        print(f"mon={where} RES={res}: |Ey|meep={a.max():.5f} |Ey|gpu={b.max():.5f} "
              f"ratio={b.max()/a.max():.5f}", flush=True)
print("DONE_DIST")
