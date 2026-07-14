"""Full-grid chi1inv comparison MEEP vs GPUmeep for config 4 — list every
disagreeing pixel with values and position."""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
import ladder  # noqa: E402

path = ladder.build_json(float(os.environ.get("DIAG_CFG", "4")))
if float(os.environ.get("DIAG_CFG", "4")).is_integer():
    path = ladder.build_json(int(float(os.environ.get("DIAG_CFG", "4"))))

import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

sim = Simulation(path)
sim._set_everything()
s = sim.simulation
s.init_sim()

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
Nx, Ny, dx, cx, cy = g.Nx, g.Ny, g.dx, g.cx, g.cy

gi = s.fields.get_chi1inv
gpu_arrays = {"ixx": (np.asarray(g.material.ixx_Ex), 0.5, 0.0, mp.Ex, mp.X),
              "iyy": (np.asarray(g.material.iyy_Ey), 0.0, 0.5, mp.Ey, mp.Y),
              "izz": (np.asarray(g.eps_inv_zz), 0.0, 0.0, mp.Ez, mp.Z)}

for nm, (gpu, xo, yo, comp, dcomp) in gpu_arrays.items():
    diffs = []
    for i in range(Nx):
        x = (i + xo) * dx - cx
        for j in range(Ny):
            y = (j + yo) * dx - cy
            m = np.real(gi(comp, dcomp, mp.vec(x, y)))
            if abs(m - gpu[i, j]) > 1e-12:
                diffs.append((i, j, x, y, m, gpu[i, j]))
    print(f"{nm}: {len(diffs)} diffs")
    for (i, j, x, y, m, gg) in diffs[:20]:
        print(f"   i={i:4d} j={j:4d} x={x:+9.5f} y={y:+8.5f} meep={m:.10f} gpu={gg:.10f}")
    if len(diffs) > 20:
        print(f"   ... {len(diffs)-20} more")
