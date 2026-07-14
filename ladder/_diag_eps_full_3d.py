"""Full-grid chi1inv comparison MEEP vs GPUmeep for a 3D ladder config."""
import os, sys
import numpy as np

RESEVOIR = os.environ.get("RESEVOIR", "/home/cernez/resevoir")
sys.path.insert(0, RESEVOIR)
sys.path.insert(0, os.path.join(RESEVOIR, "ladder"))
import ladder  # noqa: E402

path = ladder.build_json(int(os.environ.get("DIAG_CFG", "2")))

import meep as mp  # noqa: E402
from class_simulation import Simulation  # noqa: E402

sim = Simulation(path)
sim._set_everything()
s = sim.simulation
s.init_sim()

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src"))
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")

g = csg.SimulationGPU(folder_path=path)
g._set_data(); g._update_all_args(); g._build_material_3d()
Nx, Ny, Nz, dx = g.Nx, g.Ny, g.Nz, g.dx
cx, cy, cz = g.cx, g.cy, g.cz
gi = s.fields.get_chi1inv
M = g.material

# sample a coarse but complete lattice (every 2nd point) for speed
step = int(os.environ.get("DIAG_STRIDE", "2"))
specs = [
    ("ixx@Ex", np.asarray(M.ixx_Ex), (0.5, 0.0, 0.0), mp.Ex, mp.X),
    ("ixy@Ex", np.asarray(M.ixy_Ex), (0.5, 0.0, 0.0), mp.Ex, mp.Y),
    ("ixz@Ex", np.asarray(M.ixz_Ex), (0.5, 0.0, 0.0), mp.Ex, mp.Z),
    ("iyy@Ey", np.asarray(M.iyy_Ey), (0.0, 0.5, 0.0), mp.Ey, mp.Y),
    ("iyz@Ey", np.asarray(M.iyz_Ey), (0.0, 0.5, 0.0), mp.Ey, mp.Z),
    ("izz@Ez", np.asarray(M.izz_Ez), (0.0, 0.0, 0.5), mp.Ez, mp.Z),
]
for nm, gpu, (xo, yo, zo), comp, dcomp in specs:
    diffs = []
    worst = (0.0, None)
    for i in range(0, Nx, step):
        x = (i + xo) * dx - cx
        for j in range(0, Ny, step):
            y = (j + yo) * dx - cy
            for k in range(0, Nz, step):
                z = (k + zo) * dx - cz
                m = np.real(gi(comp, dcomp, mp.vec(x, y, z)))
                d = abs(m - gpu[i, j, k])
                if d > 1e-9:
                    diffs.append((i, j, k, x, y, z, m, gpu[i, j, k]))
                    if d > worst[0]:
                        worst = (d, (i, j, k, x, y, z, m, gpu[i, j, k]))
    print(f"{nm}: {len(diffs)} diffs (stride {step})")
    for (i, j, k, x, y, z, m, gg) in diffs[:6]:
        print(f"   ({i},{j},{k}) x={x:+.4f} y={y:+.4f} z={z:+.4f} "
              f"meep={m:.8f} gpu={gg:.8f}")
    if len(diffs) > 6:
        print(f"   ... {len(diffs)-6} more; worst: {worst[1]}")
