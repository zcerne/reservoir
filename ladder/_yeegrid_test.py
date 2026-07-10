"""Is the residual 2.5% the MEEP DFT interpolation (yee_grid=False averages
neighbouring Yee points -> cos(k*dx/2) O(dx^2) peak reduction)? Compare gpu's
raw face values against MEEP interpolated (default) AND MEEP raw (yee_grid=True)
at res 40 and 80. If MEEP-raw matches gpu (ratio->1) while MEEP-interp is ~2.5%
low, the interpolation is the cause.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_meep, run_gpumeep  # noqa: E402

os.environ["LADDER_RUN_UNTIL"] = "120"
for res in [40, 80]:
    os.environ["LADDER_RES"] = str(res)
    path = build_json(1)
    g = np.abs(np.asarray(run_gpumeep(path)).reshape(-1))
    os.environ["GPUMEEP_YEE_GRID"] = "0"           # MEEP interpolated (default)
    mi = np.abs(np.asarray(run_meep(path)).reshape(-1))
    os.environ["GPUMEEP_YEE_GRID"] = "1"           # MEEP raw Yee grid
    mr = np.abs(np.asarray(run_meep(path)).reshape(-1))
    print(f"RES={res}: gpu_max={g.max():.5f}  MEEPinterp_max={mi.max():.5f} "
          f"(gpu/interp={g.max()/mi.max():.5f})  MEEPraw_max={mr.max():.5f} "
          f"(gpu/raw={g.max()/mr.max():.5f})", flush=True)
print("DONE_YEE")
