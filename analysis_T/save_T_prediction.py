"""Pre-compute and save T-matrix prediction for current JSON amplitude."""
import json, os, sys
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "data/test2D"
T_dir  = os.path.join(folder, "simulation_T")

with open(os.path.join(folder, "simulation_data.json")) as f:
    cfg = json.load(f)
src_key = next(k for k, v in cfg.items() if isinstance(v, dict) and v.get("class") == "source")
amplitude = cfg[src_key].get("amplitude", [1.0])
if not isinstance(amplitude, list):
    amplitude = [amplitude]

T   = np.load(os.path.join(T_dir, "T_matrix.npz"))
a   = np.array(amplitude, dtype=complex)
E_Ey = T["T_Ey"] @ a
E_Ex = T["T_Ex"] @ a
E_Ez = T["T_Ez"] @ a
I_out = np.abs(E_Ey)**2 + np.abs(E_Ex)**2 + np.abs(E_Ez)**2

out = os.path.join(T_dir, "T_prediction.npz")
np.savez(out, amplitude=np.array(amplitude), E_Ey=E_Ey, E_Ex=E_Ex, E_Ez=E_Ez, I_out=I_out)
print(f"Saved {out}")
print(f"amplitude={amplitude}")
print(f"I_total max={I_out.max():.4f}  sum={I_out.sum():.4f}")
