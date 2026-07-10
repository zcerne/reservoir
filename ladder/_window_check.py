"""Is gpu's monitor DFT normalization window-dependent? extract_complex_2d uses
(2/n_steps)*Sigma, correct only for a CW tone over the whole window. For a
decaying pulse, lengthening run_until adds ~zero-field steps that inflate
n_steps and SHRINK the estimate -> |Ey| should drop with run_until if the
normalization is wrong. MEEP's (dt/sqrt(2pi))*Sigma is a true integral, window
independent. gpu-only, res 40."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json, run_gpumeep  # noqa: E402

os.environ["LADDER_RES"] = "40"
for ru in [120, 180, 260]:
    os.environ["LADDER_RUN_UNTIL"] = str(ru)
    path = build_json(1)
    b = np.abs(np.asarray(run_gpumeep(path)).reshape(-1))
    print(f"run_until={ru}: |Ey|max={b.max():.6f}", flush=True)
print("DONE_WINDOW")
