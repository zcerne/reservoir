"""Config 4 (high-Q DBR cavity): is gpu's DFT truncated because it stops at
run_until=120 while the cavity is still ringing? Sweep run_until; if |Ey| keeps
changing, the DFT window is cutting off the ring-down (unlike config 1, which
was window-invariant). Also print MEEP's value for reference."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_gpumeep, run_meep  # noqa: E402

os.environ["LADDER_RES"] = "40"
# MEEP reference (its own run length)
path = build_json(4)
m = np.abs(np.asarray(run_meep(path)).reshape(-1)).max()
print(f"MEEP |Ey|max = {m:.5f}", flush=True)
for ru in [120, 240, 480, 800]:
    os.environ["LADDER_RUN_UNTIL"] = str(ru)
    path = build_json(4)
    g = np.abs(np.asarray(run_gpumeep(path)).reshape(-1)).max()
    print(f"gpu run_until={ru}: |Ey|max={g:.5f}  gpu/meep={g/m:.4f}", flush=True)
print("DONE_CFG4WIN")
