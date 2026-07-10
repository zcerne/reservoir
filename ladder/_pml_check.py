"""Is the ~2% continuum gap (gpu->0.756 vs MEEP->0.772) the PML absorber diff?

Config 1 (vacuum), sweep PML thickness x resolution. If the gpu/meep amplitude
ratio moves toward 1 and the residual shrinks as PML thickens, the leftover
discrepancy is the absorber (gpu CPML vs MEEP PML), NOT the source/propagation.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_meep, run_gpumeep  # noqa: E402

for pml in [1.5, 3.0]:
    os.environ["LADDER_PML"] = str(pml)
    for res in [40, 80]:
        os.environ["LADDER_RES"] = str(res)
        path = build_json(1)
        a = np.asarray(run_meep(path)).reshape(-1).astype(complex)
        b = np.asarray(run_gpumeep(path)).reshape(-1).astype(complex)
        n = min(len(a), len(b))
        a = a[(len(a) - n) // 2:(len(a) - n) // 2 + n]
        b = b[(len(b) - n) // 2:(len(b) - n) // 2 + n]
        rmax = np.abs(b).max() / np.abs(a).max()
        print(f"PML={pml} RES={res}: |Ey|meep={np.abs(a).max():.5f} "
              f"|Ey|gpu={np.abs(b).max():.5f} ratio={rmax:.5f}", flush=True)
print("DONE_PML")
