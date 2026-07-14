"""Confirm cfg4 residual is entirely the edge-pixel epsilon: run GPUmeep with
MEEP's OWN chi1inv arrays (sampled at every Yee point) and compare the sensor
against the stored MEEP result.
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

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
gpu_src = os.environ.get("GPUMEEP_PATH", "/home/cernez/GPUmeep/src")
sys.path.insert(0, gpu_src)
import importlib  # noqa: E402
sys.modules.pop("class_simulation_gpu", None)
csg = importlib.import_module("class_simulation_gpu")
import fdtd_2d as f2  # noqa: E402
import jax.numpy as jnp  # noqa: E402

g = csg.SimulationGPU(folder_path=path)
g.force_fullvector = True
g._set_data(); g._update_all_args(); g._build_material()
Nx, Ny, dx, cx, cy = g.Nx, g.Ny, g.dx, g.cx, g.cy

print("sampling MEEP chi1inv arrays ...")
gi = s.fields.get_chi1inv
ixx = np.empty((Nx, Ny)); iyy = np.empty((Nx, Ny)); izz = np.empty((Nx, Ny))
for i in range(Nx):
    x_f = (i + 0.5) * dx - cx; x_n = i * dx - cx
    for j in range(Ny):
        y_f = (j + 0.5) * dx - cy; y_n = j * dx - cy
        ixx[i, j] = np.real(gi(mp.Ex, mp.X, mp.vec(x_f, y_n)))
        iyy[i, j] = np.real(gi(mp.Ey, mp.Y, mp.vec(x_n, y_f)))
        izz[i, j] = np.real(gi(mp.Ez, mp.Z, mp.vec(x_n, y_n)))

d_ixx = np.abs(ixx - np.asarray(g.material.ixx_Ex))
d_iyy = np.abs(iyy - np.asarray(g.material.iyy_Ey))
d_izz = np.abs(izz - np.asarray(g.eps_inv_zz))
for nm, d in [("ixx", d_ixx), ("iyy", d_iyy), ("izz", d_izz)]:
    jj, ii = np.nonzero(d.T > 1e-12)
    print(f"{nm}: n(diff)={len(ii)} max={d.max():.3e} rows(j) with diffs: "
          f"{sorted(set(jj.tolist()))[:12]}")

# patch the GPU material with MEEP's arrays and run the driver
J = lambda a: jnp.asarray(a, jnp.float64)
z = jnp.zeros((Nx, Ny), dtype=jnp.float64)
patched = f2.AnisoFull2D(
    ixx_Ex=J(ixx), ixy_Ex=z, ixz_Ex=z,
    ixy_Ey=z, iyy_Ey=J(iyy), iyz_Ey=z,
    ixz_nd=z, iyz_nd=z, izz_nd=J(izz))

_orig = csg.SimulationGPU._build_material
def _patched_build(self):
    _orig(self)
    self.material = patched
    self.eps_inv_zz = J(izz)
csg.SimulationGPU._build_material = _patched_build

g2 = csg.SimulationGPU(folder_path=path)
g2.force_fullvector = True
g2.run()

sim_dir = os.path.join(path, "simulation")
a = np.load(os.path.join(sim_dir, "monitor_2_meep.npz"))["Ey"][0]
b = np.load(os.path.join(sim_dir, "monitor_2.npz"))["Ey"][0]
n = min(len(a), len(b))
a = a[(len(a) - n) // 2:][:n]; b = b[(len(b) - n) // 2:][:n]
inner = np.vdot(a, b); c = abs(inner) / (np.linalg.norm(a) * np.linalg.norm(b))
dr = b * np.exp(-1j * np.angle(inner))
print(f"cfg4 WITH MEEP eps: |corr|={c:.8f} phase={np.degrees(np.angle(inner)):+.4f}deg "
      f"max-ratio={np.abs(b).max()/np.abs(a).max():.6f} "
      f"relL2(derot)={np.linalg.norm(dr-a)/np.linalg.norm(a):.3e}")
