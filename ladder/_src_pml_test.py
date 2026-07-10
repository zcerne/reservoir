"""Decisive test: is the residual ~2.5% (res40) the source-extends-into-PML
effect (planewave-into-PML, is_integrated-sensitive per MEEP docs)?
Compare source spanning to the PML edge (sy=6, interior [-3,3]) vs pulled 1um
off it (sy=4, [-2,2]) at res 40 and 80. If the ratio collapses toward 1 with
the shorter source, that's the cause; if unchanged, it's fundamental O(dx²)
source-coupling.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_meep, run_gpumeep  # noqa: E402

os.environ["LADDER_RUN_UNTIL"] = "120"
for sy in [6.0, 4.0]:
    os.environ["LADDER_SRC_SY"] = str(sy)
    for res in [40, 80]:
        os.environ["LADDER_RES"] = str(res)
        path = build_json(1)
        a = np.abs(np.asarray(run_meep(path)).reshape(-1))
        b = np.abs(np.asarray(run_gpumeep(path)).reshape(-1))
        n = min(len(a), len(b))
        a = a[(len(a) - n) // 2:(len(a) - n) // 2 + n]
        b = b[(len(b) - n) // 2:(len(b) - n) // 2 + n]
        # central-half ratio (avoid the y-edges where |Ey|->0)
        c0, c1 = n // 4, 3 * n // 4
        rc = (b[c0:c1] / (a[c0:c1] + 1e-30)).mean()
        print(f"src_sy={sy} (into_PML={sy>=6}) RES={res}: "
              f"|Ey|meep={a.max():.5f} |Ey|gpu={b.max():.5f} "
              f"ratio_max={b.max()/a.max():.5f} central_ratio={rc:.5f}", flush=True)
print("DONE_SRCPML")
