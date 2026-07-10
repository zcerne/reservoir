"""Do MEEP and gpu converge to the SAME config-4 transmission when both are run
long enough? MEEP integrates [0, run_until+50], gpu [0, run_until]. Sweep
run_until for both; if they meet at large run_until, the mismatch is just an
under-run / window mismatch, and the fix is a matched (long) window."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_gpumeep, run_meep  # noqa: E402

os.environ["LADDER_RES"] = "40"
for ru in [120, 300, 600]:
    os.environ["LADDER_RUN_UNTIL"] = str(ru)
    path = build_json(4)
    m = np.abs(np.asarray(run_meep(path)).reshape(-1)).max()
    g = np.abs(np.asarray(run_gpumeep(path)).reshape(-1)).max()
    print(f"run_until={ru}: MEEP(→{ru}+50)={m:.5f}  gpu(→{ru})={g:.5f}  gpu/meep={g/m:.4f}",
          flush=True)
print("DONE_CFG4CONV")
