"""Compare T-matrix prediction for [1,1,1,1] against direct MEEP output."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys

folder = sys.argv[1] if len(sys.argv) > 1 else "data/test2D"
sim_dir = os.path.join(folder, "simulation")
T_dir   = os.path.join(folder, "simulation_T")
fig_dir = os.path.join(folder, "figures")
os.makedirs(fig_dir, exist_ok=True)

import json
with open(os.path.join(folder, "simulation_data.json")) as f:
    _cfg = json.load(f)
_src_key = next(k for k, v in _cfg.items() if isinstance(v, dict) and v.get("class") == "source")
amplitude = _cfg[_src_key].get("amplitude", [1.0])
if not isinstance(amplitude, list):
    amplitude = [amplitude]

# ── T-matrix prediction ────────────────────────────────────────────────────
T = np.load(os.path.join(T_dir, "T_matrix.npz"))
T_Ey = T["T_Ey"]   # (N_y, 4)
T_Ex = T["T_Ex"]
T_Ez = T["T_Ez"]
a    = np.array(amplitude, dtype=complex)

E_Ey_T = T_Ey @ a
E_Ex_T = T_Ex @ a
E_Ez_T = T_Ez @ a
I_T    = np.abs(E_Ey_T)**2 + np.abs(E_Ex_T)**2 + np.abs(E_Ez_T)**2

# ── Direct MEEP output ─────────────────────────────────────────────────────
m2 = np.load(os.path.join(sim_dir, "monitor_2.npz"))
print("monitor_2 keys:", list(m2.keys()))
print("monitor_2 shapes:", {k: m2[k].shape for k in m2.keys()})

# DFT arrays can be (1, N_y) or (N_y,) — flatten
def load_1d(arr):
    return arr.flatten()

E_Ey_M = load_1d(m2["Ey"]) if "Ey" in m2 else np.zeros_like(E_Ey_T)
E_Ex_M = load_1d(m2["Ex"]) if "Ex" in m2 else np.zeros_like(E_Ex_T)
E_Ez_M = load_1d(m2["Ez"]) if "Ez" in m2 else np.zeros_like(E_Ez_T)
I_M    = np.abs(E_Ey_M)**2 + np.abs(E_Ex_M)**2 + np.abs(E_Ez_M)**2

N = min(len(I_T), len(I_M))
y = np.linspace(-10, 10, N)

print(f"\nT-matrix:   I_total max={I_T[:N].max():.4f}  sum={I_T[:N].sum():.4f}")
print(f"MEEP direct: I_total max={I_M[:N].max():.4f}  sum={I_M[:N].sum():.4f}")

# Normalise for shape comparison
I_T_n = I_T[:N] / I_T[:N].max()
I_M_n = I_M[:N] / I_M[:N].max()
corr   = np.corrcoef(I_T_n, I_M_n)[0, 1]
rmse   = np.sqrt(np.mean((I_T_n - I_M_n)**2))
print(f"\nNormalized comparison: correlation={corr:.6f}  RMSE={rmse:.6f}")

# ── Figure ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
comps = [
    ("Ey", E_Ey_T[:N], E_Ey_M[:N]),
    ("Ex", E_Ex_T[:N], E_Ex_M[:N]),
    ("Ez", E_Ez_T[:N], E_Ez_M[:N]),
]

for row, (name, ET, EM) in enumerate(comps):
    # Intensity
    ax = axes[row, 0]
    ax.plot(y, np.abs(ET)**2, label="T-matrix", lw=1.5)
    ax.plot(y, np.abs(EM)**2, "--", label="MEEP", lw=1.5)
    ax.set_xlabel("y (µm)")
    ax.set_ylabel(f"|{name}|²")
    ax.legend(fontsize=8)
    ax.set_title(f"|{name}|² at output")

    # Phase
    ax = axes[row, 1]
    ax.plot(y, np.angle(ET), label="T-matrix", lw=1.5)
    ax.plot(y, np.angle(EM), "--", label="MEEP", lw=1.5)
    ax.set_xlabel("y (µm)")
    ax.set_ylabel(f"phase({name}) (rad)")
    ax.legend(fontsize=8)
    ax.set_title(f"phase({name}) at output")

fig.suptitle(
    f"T-matrix vs MEEP  |  amplitude={amplitude}\n"
    f"I_total corr={corr:.4f}  RMSE={rmse:.4f}  "
    f"(T max={I_T[:N].max():.3f}, MEEP max={I_M[:N].max():.3f})",
    fontsize=10
)
fig.tight_layout()
out = os.path.join(fig_dir, "T_vs_meep_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved {out}")
