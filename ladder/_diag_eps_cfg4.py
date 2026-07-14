"""Compare MEEP's runtime chi1inv arrays vs GPUmeep material maps for config 4.

Samples fields.get_chi1inv at exact Yee points along an x-row through the
DBR mirrors and reports where (and by how much) the two engines' epsilon
discretizations differ.
"""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
import ladder  # noqa: E402

path = ladder.build_json(4)

import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

sim = Simulation(path)
sim._set_everything()
s = sim.simulation
s.init_sim()

# ---------------- GPU material ----------------
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
sys.path.insert(0, gpu_src)
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")

g = csg.SimulationGPU(folder_path=path)
g.force_fullvector = True
g._set_data(); g._update_all_args(); g._build_material()
Nx, Ny = g.Nx, g.Ny
print(f"grid {Nx}x{Ny}, dx={g.dx}")

iyy_gpu = np.asarray(g.material.iyy_Ey)   # at (i, j+1/2)
ixx_gpu = np.asarray(g.material.ixx_Ex)   # at (i+1/2, j)
izz_gpu = np.asarray(g.eps_inv_zz)        # at (i, j)

j_row = Ny // 2                            # center row (inside mirror span)
y_Ey = (j_row + 0.5) * g.dx - g.cy
y_nd = j_row * g.dx - g.cy

iyy_m = np.array([s.fields.get_chi1inv(mp.Ey, mp.Y, mp.vec(i * g.dx - g.cx, y_Ey))
                  for i in range(Nx)])
ixx_m = np.array([s.fields.get_chi1inv(mp.Ex, mp.X, mp.vec((i + 0.5) * g.dx - g.cx, y_nd))
                  for i in range(Nx)])

for name, gpu, meep_ in [("iyy(Ey)", iyy_gpu[:, j_row], iyy_m),
                         ("ixx(Ex)", ixx_gpu[:, j_row], ixx_m)]:
    d = np.abs(gpu - meep_)
    bad = np.nonzero(d > 1e-12)[0]
    print(f"{name}: max|diff|={d.max():.3e}  n(diff>1e-12)={len(bad)}/{Nx}")
    for i in bad[:12]:
        x = i * g.dx - g.cx
        print(f"   i={i:4d} x={x:+.4f}: meep={meep_[i]:.12f} gpu={gpu[i]:.12f} "
              f"diff={gpu[i]-meep_[i]:+.3e}")
    if len(bad) > 12:
        print(f"   ... {len(bad)-12} more")

# also a row through the mirror EDGE region (y just inside +3)
j_edge = int(round((3.0 + g.cy) / g.dx - 0.5)) - 1   # Ey row just below y=+3
y_e = (j_edge + 0.5) * g.dx - g.cy
iyy_me = np.array([s.fields.get_chi1inv(mp.Ey, mp.Y, mp.vec(i * g.dx - g.cx, y_e))
                   for i in range(Nx)])
d = np.abs(iyy_gpu[:, j_edge] - iyy_me)
print(f"iyy(Ey) @ y={y_e:.4f} (mirror edge row): max|diff|={d.max():.3e} "
      f"n>1e-12: {int((d > 1e-12).sum())}")
