"""Compare the SOLVER-RELEVANT epsilon: gpu's ε_yy at the Ey face and ε_xx at the
Ex face vs MEEP's per-component chi1inv (fields.get_chi1inv) at the SAME Yee points.
This is what each engine actually steps with — the true test of ε agreement."""
import os, sys, importlib
import numpy as np
RESV = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RESV); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ladder import build_json  # noqa: E402

path = build_json(4)

import meep as mp
from meep.simulation import py_v3_to_vec
from class_simulation import Simulation
sm = Simulation(path); sm._set_everything(); sm.simulation.init_sim()
f = sm.simulation.fields
DIM = sm.simulation.dimensions
def _vec(x, y):
    return py_v3_to_vec(DIM, mp.Vector3(x, y), False)

# gpu face epsilons
import jax; jax.config.update("jax_enable_x64", True)
sys.modules.pop("class_simulation_gpu", None)
# Import from the CANONICAL gpumeep driver (GPUMEEP_PATH), not the stale resevoir copy.
gpu_src = os.environ.get("GPUMEEP_PATH") or RESV
sys.path.insert(0, gpu_src)
csg = importlib.import_module("class_simulation_gpu")
g = csg.SimulationGPU(folder_path=path); g.force_fullvector = True
g._set_data(); g._update_all_args(); g._build_material()
eyy_gpu = 1.0 / np.asarray(g.material.iyy_Ey)   # ε_yy at Ey-face (i, j+1/2)
exx_gpu = 1.0 / np.asarray(g.material.ixx_Ex)   # ε_xx at Ex-face (i+1/2, j)
jmid = g.Ny // 2

# MEEP chi1inv at the SAME physical Yee coords. Ey-face at (i, j+1/2), Ex-face (i+1/2, j).
yc = (jmid + 0.5) * g.dx - g.cy           # y of Ey-face row
yn = jmid * g.dx - g.cy                   # y of Ex-face row (node y)
d_yy = []; d_xx = []
for i in range(g.Nx):
    xEy = i * g.dx - g.cx
    xEx = (i + 0.5) * g.dx - g.cx
    inv_yy = f.get_chi1inv(mp.Ey, mp.Y, _vec(xEy, yc), 0.0, True).real
    inv_xx = f.get_chi1inv(mp.Ex, mp.X, _vec(xEx, yn), 0.0, True).real
    d_yy.append(1.0 / inv_yy if inv_yy != 0 else 1.0)
    d_xx.append(1.0 / inv_xx if inv_xx != 0 else 1.0)
eyy_meep = np.array(d_yy); exx_meep = np.array(d_xx)

gy = eyy_gpu[:, jmid]; gx = exx_gpu[:, jmid]
dyy = gy - eyy_meep; dxx = gx - exx_meep
print(f"ε_yy(Ey-face): max|Δ|={np.abs(dyy).max():.4f} RMS={np.sqrt(np.mean(dyy**2)):.4f} "
      f"n|Δ|>0.01={int(np.sum(np.abs(dyy)>0.01))}/{g.Nx}")
print(f"ε_xx(Ex-face): max|Δ|={np.abs(dxx).max():.4f} RMS={np.sqrt(np.mean(dxx**2)):.4f} "
      f"n|Δ|>0.01={int(np.sum(np.abs(dxx)>0.01))}/{g.Nx}")
for k in np.argsort(-np.abs(dyy))[:6]:
    print(f"  ε_yy x={k*g.dx-g.gx/2:+.4f}  gpu={gy[k]:.4f}  MEEP={eyy_meep[k]:.4f}  Δ={dyy[k]:+.4f}")
print("DONE_FACE")
